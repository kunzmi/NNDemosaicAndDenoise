using System;
using System.Collections.Generic;
using System.IO;
using System.Globalization;
using System.Runtime.InteropServices;
using System.Drawing;
using System.Drawing.Imaging;
using ManagedCuda;
using ManagedCuda.BasicTypes;
using ManagedCuda.VectorTypes;
using ManagedCuda.CudaBlas;
using ManagedCuda.CudaDNN;
using ManagedCuda.NPP;
using KernelClasses;
using NeuralNetworkTraining;

namespace TrainNetwork
{
    class Program
    {
        static double learning_rate = 0.005;
        static int deviceID = 0;
        static string ISO = "100";
        static bool crosscheck = false;
        static bool saveImages = false;
        static int warmStart = 0;

        static float3[][] baOriginal;
        static float[][] baRAW;

        static List<string> fileRawList = new List<string>();
        static List<string> fileTrouthList = new List<string>();

        static DeBayerGreenKernel deBayerGreenKernel;
        static DeBayerRedBlueKernel deBayerRedBlueKernel;
        static PrepareDataKernel prepareDataKernel;
        static RestoreImageKernel restoreImageKernel;
        static CudaContext ctx;

        private static float[] ReadRAWFloat(string filename)
        {
            FileStream fs = new FileStream(filename, FileMode.Open, FileAccess.Read);
            BinaryReader bw = new BinaryReader(fs);
            int dimX = bw.ReadInt32();
            int dimY = bw.ReadInt32();
            int c = bw.ReadInt32();
            if (c != 1)
                throw new FileLoadException("This is not a one channel image!");

            float[] data = new float[dimX * dimY];

            unsafe
            {
                fixed (float* ptr = data)
                {
                    byte[] buffer = bw.ReadBytes(dimX * dimY * sizeof(float));
                    Marshal.Copy(buffer, 0, (IntPtr)ptr, dimX * dimY * sizeof(float));
                }
            }
            fs.Close();
            bw.Dispose();
            fs.Dispose();
            return data;
        }

        private static float3[] ReadRAWFloat3(string filename)
        {
            FileStream fs = new FileStream(filename, FileMode.Open, FileAccess.Read);
            BinaryReader bw = new BinaryReader(fs);
            int dimX = bw.ReadInt32();
            int dimY = bw.ReadInt32();
            int c = bw.ReadInt32();
            if (c != 3)
                throw new FileLoadException("This is not a three channel image!");

            float3[] data = new float3[dimX * dimY];

            unsafe
            {
                fixed (float3* ptr = data)
                {
                    byte[] buffer = bw.ReadBytes(dimX * dimY * (int)float3.SizeOf);
                    Marshal.Copy(buffer, 0, (IntPtr)ptr, dimX * dimY * (int)float3.SizeOf);
                }
            }
            fs.Close();
            bw.Dispose();
            fs.Dispose();
            return data;
        }

        static float GetLearningRate(int iteration)
        {
            double lr_gamma = 0.00001;
            double lr_power = 0.75;
            return (float)(learning_rate * Math.Pow((1.0 + lr_gamma * iteration), (-lr_power)));
        }
        
        static void Main(string[] args)
        {
            //Read CL arguments
            for (int i = 0; i < args.Length; i++)
            {
                if (args[i] == "-d")
                {
                    deviceID = int.Parse(args[++i]);
                }
                if (args[i] == "-lr")
                {
                    learning_rate = double.Parse(args[++i], System.Globalization.NumberStyles.AllowDecimalPoint, CultureInfo.InvariantCulture);
                }
                if (args[i] == "-iso")
                {
                    ISO = args[++i];
                }
                if (args[i] == "-t")
                {
                    crosscheck = true;
                }
                if (args[i] == "-w")
                {
                    warmStart = int.Parse(args[++i]);
                    Console.WriteLine("Start with epoch " + warmStart);
                }
                if (args[i] == "-s")
                {
                    saveImages = true;
                }
            }

            Console.WriteLine("Using device ID: " + deviceID);
            Console.WriteLine("Learning rate: " + learning_rate);

            //Init Cuda stuff
            ctx = new PrimaryContext(deviceID);
            ctx.SetCurrent();
            Console.WriteLine("Context created");
            CUmodule modPatch = ctx.LoadModulePTX("PatchProcessing.ptx");
            Console.WriteLine("modPatch loaded");
            CUmodule modBorder = ctx.LoadModulePTX("BorderTreatment.ptx");
            Console.WriteLine("modBorder loaded");
            CUmodule modError = ctx.LoadModulePTX("ErrorComputation.ptx");
            Console.WriteLine("modError loaded");
            CUmodule modPRelu = ctx.LoadModulePTX("PRelu.ptx");
            Console.WriteLine("modPRelu loaded");
            CUmodule modDeBayer = ctx.LoadModulePTX("DeBayer.ptx");
            Console.WriteLine("all modules loaded");
            deBayerGreenKernel = new DeBayerGreenKernel(modDeBayer, ctx);
            deBayerRedBlueKernel = new DeBayerRedBlueKernel(modDeBayer, ctx);
            //Both deBayer kernels are load from the same module: setting the constant variable for bayer pattern one is enough...
            deBayerGreenKernel.BayerPattern = new BayerColor[] { BayerColor.Red, BayerColor.Green, BayerColor.Green, BayerColor.Blue };
           
            prepareDataKernel = new PrepareDataKernel(modPatch, ctx);
            restoreImageKernel = new RestoreImageKernel(modPatch, ctx);
            Console.WriteLine("kernels loaded");


            int countOwn = 468083;
            int count5k = 33408;

            
            string fileBase = @"/ssd/data/TrainingsDataNN/";
           


            List<float3> WhiteBalanceFactors = new List<float3>();
            FileStream fs1 = new FileStream(fileBase + "FromOwnDataset/WhiteBalancesOwn.txt", FileMode.Open, FileAccess.Read);
            FileStream fs2 = new FileStream(fileBase + "From5kDataset/WhiteBalances5k.txt", FileMode.Open, FileAccess.Read);
            StreamReader sr1 = new StreamReader(fs1);
            StreamReader sr2 = new StreamReader(fs2);

            for (int i = 0; i < countOwn; i++)
            {
                fileRawList.Add(fileBase + "FromOwnDataset/ISO" + ISO + "/img_" + i.ToString("0000000") + ".bin");
                fileTrouthList.Add(fileBase + "FromOwnDataset/GroundTruth/img_" + i.ToString("0000000") + ".bin");

                string line = sr1.ReadLine();
                string[] values = line.Split('\t');
                float3 wb = new float3(float.Parse(values[1], System.Globalization.NumberStyles.AllowDecimalPoint, CultureInfo.InvariantCulture),
                                       float.Parse(values[2], System.Globalization.NumberStyles.AllowDecimalPoint, CultureInfo.InvariantCulture),
                                       float.Parse(values[3], System.Globalization.NumberStyles.AllowDecimalPoint, CultureInfo.InvariantCulture));

                WhiteBalanceFactors.Add(wb);
            }
            for (int i = 0; i < count5k; i++)
            {
                fileRawList.Add(fileBase + "From5kDataset/ISO" + ISO + "/img_" + i.ToString("0000000") + ".bin");
                fileTrouthList.Add(fileBase + "From5kDataset/GroundTruth/img_" + i.ToString("0000000") + ".bin");

                string line = sr2.ReadLine();
                string[] values = line.Split('\t');
                float3 wb = new float3(float.Parse(values[1], System.Globalization.NumberStyles.AllowDecimalPoint, CultureInfo.InvariantCulture),
                                       float.Parse(values[2], System.Globalization.NumberStyles.AllowDecimalPoint, CultureInfo.InvariantCulture),
                                       float.Parse(values[3], System.Globalization.NumberStyles.AllowDecimalPoint, CultureInfo.InvariantCulture));

                WhiteBalanceFactors.Add(wb);
            }
            sr2.Close();
            sr1.Close();

            baOriginal = new float3[countOwn + count5k][];
            baRAW = new float[countOwn + count5k][];

            Random rand = new Random(0);

            //random order for the image patches
            for (int i = 0; i < countOwn + count5k - 1; i++)
            {
                int r = i + (rand.Next() % (countOwn + count5k - i));
                string temp = fileRawList[i];
                fileRawList[i] = fileRawList[r];
                fileRawList[r] = temp;

                temp = fileTrouthList[i];
                fileTrouthList[i] = fileTrouthList[r];
                fileTrouthList[r] = temp;

                float3 tempf = WhiteBalanceFactors[i];
                WhiteBalanceFactors[i] = WhiteBalanceFactors[r];
                WhiteBalanceFactors[r] = tempf;
            }


            Console.WriteLine("Initialization done!");

            int trainingSize = (int)((countOwn + count5k) * 0.9f); //4 patches per file
            int testSize = fileRawList.Count - trainingSize;

            CudaBlas blas = new CudaBlas(PointerMode.Host);
            CudaDNNContext cudnn = new CudaDNNContext();

            int patchSize = 31;
            int patchSize4 = 66; //Size of an 2x2 patch read from file
            int batch = 64;
            float normalization = 0.5f;

            //define neural network:
            StartLayer start = new StartLayer(patchSize, patchSize, 3, batch);
            FinalLayer final = new FinalLayer(patchSize, patchSize, 3, batch, FinalLayer.Norm.Mix, ctx, modError);
            ConvolutionalLayer conv1 = new ConvolutionalLayer(patchSize, patchSize, 3, patchSize, patchSize, 64, batch, 9, 9, ConvolutionalLayer.Activation.PRelu, blas, cudnn, ctx, modBorder, modPRelu);
            ConvolutionalLayer conv2 = new ConvolutionalLayer(patchSize, patchSize, 64, patchSize, patchSize, 64, batch, 5, 5, ConvolutionalLayer.Activation.PRelu, blas, cudnn, ctx, modBorder, modPRelu);
            ConvolutionalLayer conv3 = new ConvolutionalLayer(patchSize, patchSize, 64, patchSize, patchSize, 3, batch, 5, 5, ConvolutionalLayer.Activation.None, blas, cudnn, ctx, modBorder, modPRelu);
            start.ConnectFollowingLayer(conv1);
            conv1.ConnectFollowingLayer(conv2);
            conv2.ConnectFollowingLayer(conv3);
            conv3.ConnectFollowingLayer(final);

            CudaDeviceVariable<float3> imgA = new CudaDeviceVariable<float3>(patchSize4 * patchSize4);
            CudaDeviceVariable<float3> imgB = new CudaDeviceVariable<float3>(patchSize4 * patchSize4);
            CudaDeviceVariable<float> rawd = new CudaDeviceVariable<float>(patchSize4 * patchSize4);

            CudaDeviceVariable<float> inputImgs = new CudaDeviceVariable<float>(patchSize * patchSize * 3 * batch);
            CudaDeviceVariable<float> groundTrouth = new CudaDeviceVariable<float>(patchSize * patchSize * 3 * batch);
            NPPImage_8uC3 imgU3a = new NPPImage_8uC3(patchSize, patchSize);
            NPPImage_8uC3 imgU3b = new NPPImage_8uC3(patchSize, patchSize);
            NPPImage_8uC3 imgU3c = new NPPImage_8uC3(patchSize, patchSize);

            Bitmap a = new Bitmap(patchSize, patchSize, PixelFormat.Format24bppRgb);
            Bitmap b = new Bitmap(patchSize, patchSize, PixelFormat.Format24bppRgb);
            Bitmap c = new Bitmap(patchSize, patchSize, PixelFormat.Format24bppRgb);

            Random randImageOutput = new Random(0);
            Random randForInit = new Random(0);
            start.InitRandomWeight(randForInit);
            conv1.SetActivation(0.1f);
            conv2.SetActivation(0.1f);

            int startEpoch = warmStart;

            FileStream fs;
            //restore network in case of warm start:
            if (warmStart > 0)
            {
                fs = new FileStream("epoch_" + learning_rate.ToString(CultureInfo.InvariantCulture) + "_" + ISO + "_" + (warmStart - 1) + ".cnn", FileMode.Open, FileAccess.Read);
                start.RestoreValues(fs);
                fs.Close();
                fs.Dispose();
            }

            //validate results on validation data set
            if (crosscheck)
            {

                FileStream csvResult = new FileStream("results_" + learning_rate.ToString(CultureInfo.InvariantCulture) + "_" + ISO + ".csv", FileMode.Append, FileAccess.Write);
                StreamWriter sw = new StreamWriter(csvResult);

                sw.WriteLine("L1;L2;Mix;Filename");
                for (int i = 0; i < 2000; i += 1)
                {
                    string filename = "epoch_" + learning_rate.ToString(CultureInfo.InvariantCulture) + "_" + ISO + "_" + i + ".cnn";
                    try
                    {
                        FileStream cnn = new FileStream(filename, FileMode.Open, FileAccess.Read);
                        start.RestoreValues(cnn);
                        cnn.Close();
                        cnn.Dispose();

                    }
                    catch (Exception)
                    {
                        Console.WriteLine("Skipping: " + i);
                        continue;
                    }

                    double errorL1 = 0;
                    double errorL2 = 0;
                    double errorMix = 0;
                    for (int iter = 0; iter < testSize / batch * 4; iter++)
                    {
                        //Prepare batch for training:
                        for (int ba = 0; ba < batch / 4; ba++)
                        {
                            int idx = iter * (batch / 4) + ba + trainingSize;

                            float3[] original;
                            float[] raw;
                            if (baRAW[idx - trainingSize] == null)
                            {
                                original = ReadRAWFloat3(fileTrouthList[idx]);
                                raw = ReadRAWFloat(fileRawList[idx]);
                                baOriginal[idx - trainingSize] = original;
                                baRAW[idx - trainingSize] = raw;
                            }
                            else
                            {
                                original = baOriginal[idx - trainingSize];
                                raw = baRAW[idx - trainingSize];
                            }

                            rawd.CopyToDevice(raw);
                            imgA.CopyToDevice(original);

                            deBayerGreenKernel.RunSafe(rawd, imgB, patchSize4, new float3(0, 0, 0), WhiteBalanceFactors[idx]);
                            deBayerRedBlueKernel.RunSafe(rawd, imgB, patchSize4, new float3(0, 0, 0), WhiteBalanceFactors[idx]);
                            prepareDataKernel.RunSafe(imgA, imgB, groundTrouth, inputImgs, ba, normalization, WhiteBalanceFactors[idx]);
                        }

                        start.SetData(inputImgs);
                        final.SetGroundTrouth(groundTrouth);

                        float err = start.InferenceTraining(inputImgs);

                        errorMix += err;
                        errorL1 += final.GetError(FinalLayer.Norm.L1);
                        errorL2 += final.GetError(FinalLayer.Norm.L2);


                    }
                    Console.WriteLine("Results for: " + filename);
                    Console.WriteLine("Mean Error L1: " + errorL1 / testSize * batch / 4);
                    Console.WriteLine("Mean Error L2: " + errorL2 / testSize * batch / 4);
                    Console.WriteLine("Mean Error Mix: " + errorMix / testSize * batch / 4);
                    sw.Write((errorL1 / testSize * batch / 4).ToString().Replace(".", ","));
                    sw.Write(";");
                    sw.Write((errorL2 / testSize * batch / 4).ToString().Replace(".", ","));
                    sw.Write(";");
                    sw.Write((errorMix / testSize * batch / 4).ToString().Replace(".", ","));
                    sw.Write(";");
                    sw.WriteLine(filename);
                    sw.Flush();
                }
                sw.Close();
                csvResult.Close();
                csvResult.Dispose();
            }
            //or train existing network:
            else
            {
                double error = 0;
                double errorEpoch = 0;
                for (int epoch = startEpoch; epoch < 2000; epoch++)
                {
                    errorEpoch = 0;
                    error = 0;

                    for (int iter = 0; iter < trainingSize / batch * 4; iter++)
                    {
                        //Prepare batch for training:
                        for (int ba = 0; ba < batch / 4; ba++)
                        {
                            int idx = iter * (batch / 4) + ba;

                            float3[] original;
                            float[] raw;
                            if (baRAW[idx] == null)
                            {
                                original = ReadRAWFloat3(fileTrouthList[idx]);
                                raw = ReadRAWFloat(fileRawList[idx]);
                                baOriginal[idx] = original;
                                baRAW[idx] = raw;
                            }
                            else
                            {
                                original = baOriginal[idx];
                                raw = baRAW[idx];
                            }

                            rawd.CopyToDevice(raw);
                            imgA.CopyToDevice(original);

                            deBayerGreenKernel.RunSafe(rawd, imgB, patchSize4, new float3(0, 0, 0), WhiteBalanceFactors[idx]);
                            deBayerRedBlueKernel.RunSafe(rawd, imgB, patchSize4, new float3(0, 0, 0), WhiteBalanceFactors[idx]);
                            prepareDataKernel.RunSafe(imgA, imgB, groundTrouth, inputImgs, ba, normalization, WhiteBalanceFactors[idx]);
                        }

                        start.SetData(inputImgs);
                        final.SetGroundTrouth(groundTrouth);

                        float err = start.InferenceTraining(inputImgs);

                        final.BackPropagation(groundTrouth);

                        start.UpdateWeights(GetLearningRate(epoch * (trainingSize) / batch * 4 + iter));//*0+951342

                        error += err;
                        errorEpoch += err;
                        if ((epoch * trainingSize / batch * 4 + iter) % 1000 == 0 && iter != 0)
                        {
                            FileStream status = new FileStream("status_" + learning_rate.ToString(CultureInfo.InvariantCulture) + "_" + ISO + ".csv", FileMode.Append, FileAccess.Write);
                            StreamWriter sw = new StreamWriter(status);

                            sw.WriteLine((error / 1000.0).ToString().Replace(".", ",") + ";" + GetLearningRate(epoch * trainingSize / batch * 4 + iter).ToString().Replace(".", ","));

                            sw.Close();
                            status.Close();
                            status.Dispose();
                            error = 0;
                        }

                        //if ((epoch * trainingSize / batch * 4 + iter) % 10000 == 0)
                        //{
                        //    fs = new FileStream("iter_" + learning_rate.ToString(CultureInfo.InvariantCulture) + "_" + ISO + "_" + (epoch * trainingSize / batch * 4 + iter) + ".cnn", FileMode.Create, FileAccess.Write);
                        //    start.SaveValues(fs);
                        //    fs.Close();
                        //    fs.Dispose();
                        //    Console.WriteLine("Network saved for iteration " + (epoch * trainingSize / batch * 4 + iter) + "!");
                        //}

                        Console.WriteLine("Epoch: " + epoch + " Iteration: " + (epoch * trainingSize / batch * 4 + iter) + ", Error: " + err);

                        if (saveImages && iter == 0)//(epoch * trainingSize / batch * 4 + iter) % 10000 == 0 && 
                        {
                            for (int i = 0; i < 1; i++)
                            {
                                int imgidx = randImageOutput.Next(batch);
                                float3 wb = WhiteBalanceFactors[iter * (batch / 4) + imgidx / 4];
                                restoreImageKernel.RunSafe(groundTrouth, imgU3a, imgidx, wb.x, wb.y, wb.z, normalization);
                                restoreImageKernel.RunSafe(inputImgs, imgU3b, imgidx, wb.x, wb.y, wb.z, normalization);
                                CudaDeviceVariable<float> res = final.GetResult();
                                restoreImageKernel.RunSafe(res, imgU3c, imgidx, wb.x, wb.y, wb.z, normalization);

                                imgU3a.CopyToHost(a);
                                imgU3b.CopyToHost(b);
                                imgU3c.CopyToHost(c);

                                a.Save("GroundTrouth_" + learning_rate.ToString(CultureInfo.InvariantCulture) + "_" + ISO + "_" + epoch + "_" + imgidx + ".png");// * trainingSize / batch * 4 + iter
                                b.Save("Input_" + learning_rate.ToString(CultureInfo.InvariantCulture) + "_" + ISO + "_" + epoch + "_" + imgidx + ".png");
                                c.Save("Result_" + learning_rate.ToString(CultureInfo.InvariantCulture) + "_" + ISO + "_" + epoch + "_" + imgidx + ".png");
                            }
                        }
                    }
                    errorEpoch /= trainingSize / batch * 4;
                    fs = new FileStream("errorEpoch_" + learning_rate.ToString(CultureInfo.InvariantCulture) + "_" + ISO + ".csv", FileMode.Append, FileAccess.Write);
                    StreamWriter sw2 = new StreamWriter(fs);
                    sw2.WriteLine(errorEpoch.ToString().Replace(".", ","));
                    sw2.Close();
                    fs.Close();
                    fs.Dispose();

                    fs = new FileStream("epoch_" + learning_rate.ToString(CultureInfo.InvariantCulture) + "_" + ISO + "_" + epoch + ".cnn", FileMode.Create, FileAccess.Write);
                    start.SaveValues(fs);
                    fs.Close();
                    fs.Dispose();
                }
            }
        }
    }
}
