using System;
using System.Collections.Generic;
using System.IO;
using System.Runtime.InteropServices;
using System.Globalization;
using System.Text;
using System.Threading.Tasks;
using KernelClasses;
using PentaxPefFile;
using ManagedCuda;
using ManagedCuda.BasicTypes;
using ManagedCuda.VectorTypes;
using ManagedCuda.NPP;

namespace CreateTrainingData
{
    class Program
    {
        private static List<NppiRect> GetROIs(NppiRect maxRoi, int patchSize)
        {
            List<NppiRect> liste = new List<NppiRect>();

            for (int y = maxRoi.Top; y < maxRoi.Bottom - patchSize; y += patchSize)
            {
                for (int x = maxRoi.Left; x < maxRoi.Right - patchSize; x += patchSize)
                {
                    liste.Add(new NppiRect(x, y, patchSize, patchSize));
                }
            }
            return liste;
        }

        private static void WriteRAWFile(string filename, float3[] data, int dimX, int dimY)
        {
            FileStream fs = new FileStream(filename, FileMode.Create, FileAccess.Write);
            BinaryWriter bw = new BinaryWriter(fs);
            bw.Write(dimX);
            bw.Write(dimY);
            bw.Write((int)3);
            bw.Flush();
            unsafe
            {
                fixed (float3* ptr = data)
                {
                    UnmanagedMemoryStream ums = new UnmanagedMemoryStream((byte*)ptr, data.LongLength * float3.SizeOf);
                    ums.CopyTo(fs);
                    ums.Dispose();
                }
            }
            fs.Flush();
            fs.Close();
            bw.Dispose();
            fs.Dispose();
        }

        private static void WriteRAWFile(string filename, float[] data, int dimX, int dimY)
        {
            FileStream fs = new FileStream(filename, FileMode.Create, FileAccess.Write);
            BinaryWriter bw = new BinaryWriter(fs);
            bw.Write(dimX);
            bw.Write(dimY);
            bw.Write((int)1);
            bw.Flush();
            unsafe
            {
                fixed (float* ptr = data)
                {
                    UnmanagedMemoryStream ums = new UnmanagedMemoryStream((byte*)ptr, data.LongLength * sizeof(float));
                    ums.CopyTo(fs);
                    ums.Dispose();
                }
            }
            fs.Flush();
            fs.Close();
            bw.Dispose();
            fs.Dispose();
        }

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

        static void Main(string[] args)
        {
            CudaContext ctx = null;
            DeBayersSubSampleKernel kernelDeBayersSubSample = null;
            DeBayersSubSampleDNGKernel kernelDeBayersSubSampleDNG = null;
            SetupCurandKernel kernelSetupCurand = null;
            CreateBayerWithNoiseKernel kernelCreateBayerWithNoise = null;
            CudaDeviceVariable<ushort> rawImg;

            string outputPathOwn = @"C:\Users\kunz_\Desktop\TrainingDataNN\FromOwnDataset\";
            string outputPath5k = @"C:\Users\kunz_\Desktop\TrainingDataNN\From5kDataset\";


            const int patchSize = 66;
            //These are the noise levels I measured for each ISO of my camera:
            double[] noiseLevels = new double[] { 6.66667E-05, 0.0001, 0.000192308, 0.000357143, 0.000714286, 0.001388889, 0.0025 };
            string[] noiseLevelsFolders = new string[] { "ISO100", "ISO200", "ISO400", "ISO800", "ISO1600", "ISO3200", "ISO6400" };


            //Process files from my own dataset:
            string[] files = File.ReadAllLines("FileListOwnImages.txt");


            if (ctx == null)
            {
                ctx = new PrimaryContext();
                ctx.SetCurrent();
                CUmodule mod = ctx.LoadModulePTX("DeBayer.ptx");
                kernelDeBayersSubSample = new DeBayersSubSampleKernel(ctx, mod);
                kernelDeBayersSubSampleDNG = new DeBayersSubSampleDNGKernel(ctx, mod);
                kernelSetupCurand = new SetupCurandKernel(ctx, mod);
                kernelCreateBayerWithNoise = new CreateBayerWithNoiseKernel(ctx, mod);
            }

            FileStream fs = new FileStream("ImagesCompleted.txt", FileMode.Append, FileAccess.Write);
            StreamWriter sw = new StreamWriter(fs);

            PEFFile pef = new PEFFile(files[0]);
            BayerColor[] bayerPattern = new BayerColor[pef.BayerPattern.Length];
            for (int i = 0; i < pef.BayerPattern.Length; i++)
            {
                bayerPattern[i] = (BayerColor)pef.BayerPattern[i];
            }
            kernelDeBayersSubSample.BayerPattern = bayerPattern;



            rawImg = new CudaDeviceVariable<ushort>(pef.RawWidth * pef.RawHeight);
            NPPImage_32fC3 img = new NPPImage_32fC3(pef.RawWidth / 2, pef.RawHeight / 2);
            NPPImage_32fC3 imgsmall = new NPPImage_32fC3(pef.RawWidth / 8, pef.RawHeight / 8);
            NPPImage_32fC3 patch = new NPPImage_32fC3(patchSize, patchSize);
            NPPImage_32fC1 patchBayerWithNoise = new NPPImage_32fC1(patchSize, patchSize);
            //NPPImage_8uC3 img8u = new NPPImage_8uC3(patchSize, patchSize);
            //Bitmap bmp = new Bitmap(patchSize, patchSize, System.Drawing.Imaging.PixelFormat.Format24bppRgb);
            CudaDeviceVariable<byte> states = new CudaDeviceVariable<byte>(patchSize * patchSize * 48); //one state has the size of 48 bytes

            kernelSetupCurand.RunSafe(states, patchSize * patchSize);

            NppiRect maxRoi = new NppiRect(10, 10, pef.RawWidth / 8 - 10, pef.RawHeight / 8 - 10);

            List<NppiRect> ROIs = GetROIs(maxRoi, patchSize);

            float3[] imgGroundTruth = new float3[patchSize * patchSize];
            float[] noisyPatchBayer = new float[patchSize * patchSize];


            int counter = 0;

            int fileCounter = 0;

            FileStream fsWB1 = new FileStream("WhiteBalancesOwn.txt", FileMode.Create, FileAccess.Write);
            StreamWriter swWB1 = new StreamWriter(fsWB1);
            
            foreach (var file in files)
            {
                pef = new PEFFile(file);

                float whiteLevelAll = pef.WhiteLevel.Value;
                float3 whitePoint = new float3(whiteLevelAll, whiteLevelAll, whiteLevelAll);
                float3 blackPoint = new float3(pef.BlackPoint.Value[0], pef.BlackPoint.Value[1], pef.BlackPoint.Value[3]);
                whitePoint -= blackPoint;
                float scale = pef.Scaling.Value;
                float3 scaling = new float3(pef.WhitePoint.Value[0] / scale, pef.WhitePoint.Value[1] / scale, pef.WhitePoint.Value[3] / scale);

                int RoiCounter = 0;
                foreach (var roi in ROIs)
                {
                    swWB1.WriteLine((counter + RoiCounter).ToString("0000000") + "\t" + scaling.x.ToString(CultureInfo.InvariantCulture) + "\t" + scaling.y.ToString(CultureInfo.InvariantCulture) + "\t" + scaling.z.ToString(CultureInfo.InvariantCulture));

                    RoiCounter++;
                }
                fileCounter++;
                Console.WriteLine("Done " + fileCounter + " of " + files.Length);


                rawImg.CopyToDevice(pef.RawImage);
                kernelDeBayersSubSample.RunSafe(rawImg, img, (float)Math.Pow(2.0, pef.BitDepth));

                imgsmall.ResetRoi();
                img.ResizeSqrPixel(imgsmall, 0.25, 0.25, 0, 0, InterpolationMode.SuperSampling);

                RoiCounter = 0;
                foreach (var roi in ROIs)
                {
                    imgsmall.SetRoi(roi);
                    imgsmall.Copy(patch);
                    patch.CopyToHost(imgGroundTruth);
                    WriteRAWFile(outputPathOwn + @"GroundTruth\img_" + (counter + RoiCounter).ToString("0000000") + ".bin", imgGroundTruth, patchSize, patchSize);

                    RoiCounter++;
                }

                RoiCounter = 0;
                foreach (var roi in ROIs)
                {
                    imgsmall.SetRoi(roi);
                    for (int i = 0; i < 7; i++)
                    {
                        imgsmall.Copy(patch);
                        kernelCreateBayerWithNoise.RunSafe(states, patch, patchBayerWithNoise, (float)noiseLevels[i], 0);

                        patchBayerWithNoise.CopyToHost(noisyPatchBayer);
                        WriteRAWFile(outputPathOwn + noiseLevelsFolders[i] + @"\img_" + (counter + RoiCounter).ToString("0000000") + ".bin", noisyPatchBayer, patchSize, patchSize);

                    }
                    RoiCounter++;
                }
                fileCounter++;
                counter += ROIs.Count;
                Console.WriteLine("Done " + fileCounter + " of " + files.Length);
                sw.WriteLine(file);
                sw.Flush();
            }
            sw.Close();
            sw.Dispose();

            swWB1.Flush();
            swWB1.Close();


            rawImg.Dispose();
            img.Dispose();
            imgsmall.Dispose();
            patch.Dispose();
            patchBayerWithNoise.Dispose();




            //Move on to DNG images from 5k dataset:
            files = File.ReadAllLines("FileListe5KKomplett.txt");
            fs = new FileStream("ImagesCompleted5k.txt", FileMode.Append, FileAccess.Write);
            sw = new StreamWriter(fs);

            DNGFile dng = new DNGFile(files[0]);

            int maxWidth = 7000;
            int maxHeight = 5000;


            rawImg = new CudaDeviceVariable<ushort>(maxWidth * maxHeight);
            img = new NPPImage_32fC3(maxWidth, maxHeight); // /2
            imgsmall = new NPPImage_32fC3(maxWidth, maxHeight); // /8
            patch = new NPPImage_32fC3(patchSize, patchSize);
            patchBayerWithNoise = new NPPImage_32fC1(patchSize, patchSize);
            


            imgGroundTruth = new float3[patchSize * patchSize];
            noisyPatchBayer = new float[patchSize * patchSize];


            counter = 0;

            fileCounter = 0;
            int roiCount = 0;

            FileStream fsWB2 = new FileStream("WhiteBalances5k.txt", FileMode.Create, FileAccess.Write);
            StreamWriter swWB2 = new StreamWriter(fsWB2);


            foreach (var file in files)
            {
                dng = new DNGFile(file);

                bayerPattern = new BayerColor[dng.BayerPattern.Length];
                for (int i = 0; i < dng.BayerPattern.Length; i++)
                {
                    bayerPattern[i] = (BayerColor)dng.BayerPattern[i];
                }
                kernelDeBayersSubSampleDNG.BayerPattern = bayerPattern;

                maxRoi = new NppiRect(10, 10, dng.RawWidth / 8 - 10, dng.RawHeight / 8 - 10);

                ROIs = GetROIs(maxRoi, patchSize);
                roiCount += ROIs.Count;

                float[] wb = dng.AsShotNeutral;
                int RoiCounter = 0;
                foreach (var roi in ROIs)
                {
                    swWB2.WriteLine((counter + RoiCounter).ToString("0000000") + "\t" + (1.0f / wb[0]).ToString(CultureInfo.InvariantCulture) + "\t" + (1.0f / wb[1]).ToString(CultureInfo.InvariantCulture) + "\t" + (1.0f / wb[2]).ToString(CultureInfo.InvariantCulture));

                    RoiCounter++;
                }
                fileCounter++;
                Console.WriteLine("Done " + fileCounter + " of " + files.Length);


                Console.WriteLine("RoiCoint: " + ROIs.Count);
                unsafe
                {
                    fixed (ushort* ptr = dng.RawImage)
                    {
                        rawImg.CopyToDevice((IntPtr)ptr, 0, 0, dng.RawWidth * dng.RawHeight * 2);

                    }
                }
                NppiRect rect = new NppiRect(0, 0, dng.RawWidth / 2, dng.RawHeight / 2);
                img.SetRoi(rect);
                kernelDeBayersSubSampleDNG.RunSafe(rawImg, img, dng.MaxVal, dng.MinVal);
                rect = new NppiRect(0, 0, dng.RawWidth / 8, dng.RawHeight / 8);
                imgsmall.SetRoi(rect);
                img.ResizeSqrPixel(imgsmall, 0.25, 0.25, 0, 0, InterpolationMode.SuperSampling);

                RoiCounter = 0;
                foreach (var roi in ROIs)
                {
                    imgsmall.SetRoi(roi);
                    imgsmall.Copy(patch);
                    patch.CopyToHost(imgGroundTruth);
                    WriteRAWFile(outputPath5k + @"GroundTruth\img_" + (counter + RoiCounter).ToString("0000000") + ".bin", imgGroundTruth, patchSize, patchSize);

                    RoiCounter++;
                }

                RoiCounter = 0;
                foreach (var roi in ROIs)
                {
                    imgsmall.SetRoi(roi);
                    for (int i = 0; i < 7; i++)
                    {
                        imgsmall.Copy(patch);
                        kernelCreateBayerWithNoise.RunSafe(states, patch, patchBayerWithNoise, (float)noiseLevels[i], 0);

                        patchBayerWithNoise.CopyToHost(noisyPatchBayer);
                        WriteRAWFile(outputPath5k + noiseLevelsFolders[i] + @"\img_" + (counter + RoiCounter).ToString("0000000") + ".bin", noisyPatchBayer, patchSize, patchSize);

                    }
                    RoiCounter++;
                }
                fileCounter++;
                counter += ROIs.Count;
                Console.WriteLine("Done " + fileCounter + " of " + files.Length);
                sw.WriteLine(file);
                sw.Flush();
            }
            sw.Close();
            sw.Dispose();

            swWB2.Flush();
            swWB2.Close();
            Console.WriteLine("Total cropped ROIs: " + roiCount);


            rawImg.Dispose();
            img.Dispose();
            imgsmall.Dispose();
            patch.Dispose();
            patchBayerWithNoise.Dispose();            
            states.Dispose();

            ctx.Dispose();
        }
    }
}
