using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Data;
using System.Drawing;
using System.Drawing.Imaging;
using System.Globalization;
using System.IO;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Windows.Forms;
using ManagedCuda;
using ManagedCuda.BasicTypes;
using ManagedCuda.VectorTypes;
using NeuralNetworkInference;
using ManagedCuda.NPP;
using KernelClasses;
using PentaxPefFile;

namespace Demo
{
    public partial class Form1 : Form
    {
        CudaContext ctx;
        CUmodule modPRelu;
        CUmodule modDeBayer;
        CUmodule modColor;
        CreateBayerWithNoiseKernel createBayerKernel;
        DeBayerGreenKernel deBayerGreenKernel;
        DeBayerRedBlueKernel deBayerRedBlueKernel;
        SetupCurandKernel setupCurandKernel;
        HighlightRecoveryKernel highlightRecoveryKernel;
        ConvertCamToXYZKernel camToXYZKernel;
        ConvertRGBTosRGBKernel convertRGBTosRGBKernel;

        //These are the noise levels I measured for each ISO of my camera:
        float[] noiseLevels = new float[] { 6.66667E-05f, 0.0001f, 0.000192308f, 0.000357143f, 0.000714286f, 0.001388889f, 0.0025f };
        string[] noiseLevelsFolders = new string[] { "100", "200", "400", "800", "1600", "3200", "6400" };
        double learningRate = 0.011; //Used in filename of the network

        float[,] twist = new float[,] { { 0, 0, 1, 0}, { 0, 1, 0, 0}, { 1, 0, 0, 0} }; //Convert BGR to RGB and vice versa;
        const int TileSize = 512;

        DenoiseAndDemoisaic denoiseAndDemoisaic;

        NPPImage_32fC3 tile;
        NPPImage_8uC1 inputImage8uC1;
        NPPImage_8uC3 inputImage8uC3;
        NPPImage_8uC4 inputImage8uC4;
        NPPImage_32fC1 imageBayer;
        NPPImage_32fC3 inputImage32f;
        NPPImage_8uC3 noisyImage8u;
        NPPImage_32fC3 noiseImage32f;
        NPPImage_8uC3 resultImage8u;
        NPPImage_32fC3 resultImage32f;
        CudaDeviceVariable<byte> CuRandStates;
        Bitmap bmpInput;
        Bitmap bmpNoisy;
        Bitmap bmpResult;
        PEFFile pef;
        
        public Form1()
        {
            InitializeComponent();
        }

        private void Form1_Load(object sender, EventArgs e)
        {
            ctx = new PrimaryContext();
            ctx.SetCurrent();

            modPRelu = ctx.LoadModulePTX("PRelu.ptx");
            modDeBayer = ctx.LoadModulePTX("DeBayer.ptx");
            modColor = ctx.LoadModulePTX("ImageColorProcessing.ptx");

            createBayerKernel = new CreateBayerWithNoiseKernel(ctx, modDeBayer);
            deBayerGreenKernel = new DeBayerGreenKernel(modDeBayer, ctx);
            deBayerRedBlueKernel = new DeBayerRedBlueKernel(modDeBayer, ctx);
            setupCurandKernel = new SetupCurandKernel(ctx, modDeBayer);
            highlightRecoveryKernel = new HighlightRecoveryKernel(modColor, ctx);
            camToXYZKernel = new ConvertCamToXYZKernel(modColor, ctx);
            convertRGBTosRGBKernel = new ConvertRGBTosRGBKernel(modColor, ctx);

            //constant variable is set for the entire module!
            createBayerKernel.BayerPattern = new BayerColor[] {BayerColor.Red, BayerColor.Green, BayerColor.Green, BayerColor.Blue };

            //If you do not have CUDNN, set the last parameter to false (use NPP instead)
            denoiseAndDemoisaic = new DenoiseAndDemoisaic(TileSize, ctx, modPRelu, true);
            CuRandStates = new CudaDeviceVariable<byte>(TileSize * TileSize * 48); //one state has the size of 48 bytes
            setupCurandKernel.RunSafe(CuRandStates, TileSize * TileSize);
            tile = new NPPImage_32fC3(TileSize, TileSize);
            cmb_IsoValue.SelectedIndex = 0;

        }

        private void AllocateImagesNPP(Bitmap size)
        {
            int w = size.Width;
            int h = size.Height;

            if (inputImage8uC3 == null)
            {
                inputImage8uC1 = new NPPImage_8uC1(w, h);
                inputImage8uC3 = new NPPImage_8uC3(w, h);
                inputImage8uC4 = new NPPImage_8uC4(w, h);
                imageBayer = new NPPImage_32fC1(w, h);
                inputImage32f = new NPPImage_32fC3(w, h);
                noisyImage8u = new NPPImage_8uC3(w, h);
                noiseImage32f = new NPPImage_32fC3(w, h);
                resultImage8u = new NPPImage_8uC3(w, h);
                resultImage32f = new NPPImage_32fC3(w, h);
                return;
            }

            if (inputImage8uC3.Width >= w && inputImage8uC3.Height >= h)
            {
                inputImage8uC1.SetRoi(0, 0, w, h);
                inputImage8uC3.SetRoi(0, 0, w, h);
                inputImage8uC4.SetRoi(0, 0, w, h);
                imageBayer.SetRoi(0, 0, w, h);
                inputImage32f.SetRoi(0, 0, w, h);
                noisyImage8u.SetRoi(0, 0, w, h);
                noiseImage32f.SetRoi(0, 0, w, h);
                resultImage8u.SetRoi(0, 0, w, h);
                resultImage32f.SetRoi(0, 0, w, h);
            }
            else
            {
                inputImage8uC1.Dispose();
                inputImage8uC3.Dispose();
                inputImage8uC4.Dispose();
                imageBayer.Dispose();
                inputImage32f.Dispose();
                noisyImage8u.Dispose();
                noiseImage32f.Dispose();
                resultImage8u.Dispose();
                resultImage32f.Dispose();

                inputImage8uC1 = new NPPImage_8uC1(w, h);
                inputImage8uC3 = new NPPImage_8uC3(w, h);
                inputImage8uC4 = new NPPImage_8uC4(w, h);
                imageBayer = new NPPImage_32fC1(w, h);
                inputImage32f = new NPPImage_32fC3(w, h);
                noisyImage8u = new NPPImage_8uC3(w, h);
                noiseImage32f = new NPPImage_32fC3(w, h);
                resultImage8u = new NPPImage_8uC3(w, h);
                resultImage32f = new NPPImage_32fC3(w, h);
            }
        }

        private void btn_OpenImage_Click(object sender, EventArgs e)
        {
            OpenFileDialog ofd = new OpenFileDialog();
            ofd.Filter = "Images|*.bmp;*.jpg;*.png;*.tif";

            if (ofd.ShowDialog() != DialogResult.OK) return;

            bmpInput = new Bitmap(ofd.FileName);
            pef = null;

            if (bmpInput.PixelFormat != PixelFormat.Format24bppRgb && bmpInput.PixelFormat != PixelFormat.Format32bppArgb)
            {
                MessageBox.Show("Only three and four channel color image are supported!");
                bmpInput = null;
                return;
            }

            pictureBox1.Image = bmpInput;
            bmpNoisy = new Bitmap(bmpInput.Width, bmpInput.Height, PixelFormat.Format24bppRgb);
            bmpResult = new Bitmap(bmpInput.Width, bmpInput.Height, PixelFormat.Format24bppRgb);

            AllocateImagesNPP(bmpInput);
        }

        private void btn_Process_Click(object sender, EventArgs e)
        {
            if (bmpInput == null)
                return;

            denoiseAndDemoisaic.LoadNetwork("epoch_" + learningRate.ToString(CultureInfo.InvariantCulture) + "_" + noiseLevelsFolders[cmb_IsoValue.SelectedIndex] + "_1999.cnn");
            
            if (bmpInput.PixelFormat == PixelFormat.Format32bppArgb)
            {
                inputImage8uC4.CopyToDeviceRoi(bmpInput, new NppiRect(0, 0, bmpInput.Width, bmpInput.Height));
                //Convert C4 to C3 and BGR to RGB
                inputImage8uC4.Copy(inputImage8uC1, 0);
                inputImage8uC1.Copy(inputImage8uC3, 2);
                inputImage8uC4.Copy(inputImage8uC1, 1);
                inputImage8uC1.Copy(inputImage8uC3, 1);
                inputImage8uC4.Copy(inputImage8uC1, 2);
                inputImage8uC1.Copy(inputImage8uC3, 0);
            }
            else
            {
                inputImage8uC3.CopyToDeviceRoi(bmpInput, new NppiRect(0, 0, bmpInput.Width, bmpInput.Height));
                inputImage8uC3.ColorTwist(twist);
            }

            inputImage8uC3.Convert(inputImage32f);
            inputImage32f.Div(new float[] { 255, 255, 255 });

            NppiRect oldRoi = new NppiRect(0, 0, inputImage32f.WidthRoi, inputImage32f.HeightRoi);
            IEnumerable<Tiler.RoiInputOutput> rois = Tiler.GetROIs(oldRoi, TileSize, 0);

            foreach (var roi in rois)
            {
                inputImage32f.SetRoi(roi.inputROI);
                tile.ResetRoi();
                inputImage32f.Copy(tile);
                tile.SetRoi(roi.outputROI);
                imageBayer.SetRoi(roi.positionInFinalImage);
                createBayerKernel.RunSafe(CuRandStates, tile, imageBayer, noiseLevels[cmb_IsoValue.SelectedIndex], 0);
            }
            imageBayer.SetRoi(oldRoi);
            inputImage32f.SetRoi(oldRoi);

            deBayerGreenKernel.RunSafe(imageBayer, inputImage32f, new float3(), new float3(1, 1, 1));
            deBayerRedBlueKernel.RunSafe(imageBayer, inputImage32f, new float3(), new float3(1, 1, 1));

            inputImage32f.Mul(new float[] { 255, 255, 255 }, noiseImage32f);
            noiseImage32f.Convert(noisyImage8u, NppRoundMode.Near);
            noisyImage8u.ColorTwist(twist);
            noisyImage8u.CopyToHostRoi(bmpNoisy, new NppiRect(0, 0, bmpNoisy.Width, bmpNoisy.Height));

            inputImage32f.Sub(new float[] { 0.5f, 0.5f, 0.5f });

            CudaStopWatch csw = new CudaStopWatch();
            csw.Start();

            denoiseAndDemoisaic.RunImage(inputImage32f, resultImage32f);

            csw.Stop();

            Console.WriteLine("Needed time: " + csw.GetElapsedTime() + " [msec]");
            csw.Dispose();

            resultImage32f.Add(new float[] { 0.5f, 0.5f, 0.5f });
            resultImage32f.Mul(new float[] { 255, 255, 255 });
            resultImage32f.Convert(resultImage8u, NppRoundMode.Near);
            resultImage8u.ColorTwist(twist);
            resultImage8u.SetRoi(0, 0, bmpResult.Width - 16, bmpResult.Height - 16);
            resultImage8u.CopyToHostRoi(bmpResult, new NppiRect(8, 8, bmpResult.Width-16, bmpResult.Height-16));

            pictureBox2.Image = bmpNoisy;
            pictureBox3.Image = bmpResult;
        }

        private void chk_Zoom_CheckedChanged(object sender, EventArgs e)
        {
            if (chk_Zoom.Checked)
            {
                pictureBox1.SizeMode = PictureBoxSizeMode.CenterImage;
                pictureBox2.SizeMode = PictureBoxSizeMode.CenterImage;
                pictureBox3.SizeMode = PictureBoxSizeMode.CenterImage;
            }
            else
            {
                pictureBox1.SizeMode = PictureBoxSizeMode.Zoom;
                pictureBox2.SizeMode = PictureBoxSizeMode.Zoom;
                pictureBox3.SizeMode = PictureBoxSizeMode.Zoom;
            }
        }

        private void btn_OpenPEF_Click(object sender, EventArgs e)
        {
            OpenFileDialog ofd = new OpenFileDialog();
            ofd.Filter = "Pentax PEF files|*.pef";

            if (ofd.ShowDialog() != DialogResult.OK) return;

            bmpInput = null;
            pef = new PEFFile(ofd.FileName);

            pictureBox1.Image = bmpInput;
            bmpNoisy = new Bitmap(pef.RawWidth, pef.RawHeight, PixelFormat.Format24bppRgb);
            bmpResult = new Bitmap(pef.RawWidth, pef.RawHeight, PixelFormat.Format24bppRgb);

            AllocateImagesNPP(bmpNoisy);

            switch (pef.ISO)
            {
                case 100:
                    cmb_IsoValue.SelectedIndex = 0;
                    break;
                case 200:
                    cmb_IsoValue.SelectedIndex = 1;
                    break;
                case 400:
                    cmb_IsoValue.SelectedIndex = 2;
                    break;
                case 800:
                    cmb_IsoValue.SelectedIndex = 3;
                    break;
                case 1600:
                    cmb_IsoValue.SelectedIndex = 4;
                    break;
                case 3200:
                    cmb_IsoValue.SelectedIndex = 5;
                    break;
                case 6400:
                    cmb_IsoValue.SelectedIndex = 6;
                    break;    
                default:
                    cmb_IsoValue.SelectedIndex = 0;
                    break;
            }
            lbl_ISO.Text = pef.ISO.ToString();
        }

        private void Btn_ProcessPEF_Click(object sender, EventArgs e)
        {
            if (pef == null)
                return;

            denoiseAndDemoisaic.LoadNetwork("epoch_" + learningRate.ToString(CultureInfo.InvariantCulture) + "_" + noiseLevelsFolders[cmb_IsoValue.SelectedIndex] + "_1999.cnn");
            NPPImage_16uC1 rawTemp = new NPPImage_16uC1(pef.RawWidth, pef.RawHeight);
            rawTemp.CopyToDevice(pef.RawImage);
            rawTemp.Convert(imageBayer);

            float whiteLevelAll = pef.WhiteLevel.Value;
            float3 whitePoint = new float3(whiteLevelAll, whiteLevelAll, whiteLevelAll);
            float3 blackPoint = new float3(pef.BlackPoint.Value[0], pef.BlackPoint.Value[1], pef.BlackPoint.Value[3]);
            whitePoint -= blackPoint;
            float scale = pef.Scaling.Value;
            float3 scaling = new float3(pef.WhitePoint.Value[0] / scale, pef.WhitePoint.Value[1] / scale, pef.WhitePoint.Value[3] / scale);


            inputImage32f.Set(new float[] { 0, 0, 0 });
            deBayerGreenKernel.RunSafe(imageBayer, inputImage32f, blackPoint, scaling);
            deBayerRedBlueKernel.RunSafe(imageBayer, inputImage32f, blackPoint, scaling);
            inputImage32f.Div(new float[] { whitePoint.x * scaling.x, whitePoint.y * scaling.y, whitePoint.z * scaling.z });



            highlightRecoveryKernel.RunSafe(inputImage32f, new float3(scaling.x, scaling.y, scaling.z), 1);

            inputImage32f.Sub(new float[] { 0.5f, 0.5f, 0.5f }, noiseImage32f);

            Console.WriteLine("Start denoising...");
            CudaStopWatch csw = new CudaStopWatch();
            csw.Start();
            denoiseAndDemoisaic.RunImage(noiseImage32f, resultImage32f);
            csw.Stop();
            Console.WriteLine("Needed time: " + csw.GetElapsedTime() + " [msec]");
            csw.Dispose();

            resultImage32f.Add(new float[] { 0.5f, 0.5f, 0.5f });

            ColorManagment cm = new ColorManagment();

            float3 wp = 1.0f / scaling;
            double3 wb = new double3(wp.x, wp.y, wp.z);
            double2 neutralXY = cm.NeutralToXY(wb);

            cm.SetWhiteXY(neutralXY);
            ColorMatrix camToXYZ2 = cm.CameraToPCS;


            ColorMatrix d50Tod65 = new ColorMatrix(new double[] { 0.9555766, -0.0230393, 0.0631636, -0.0282895, 1.0099416, 0.0210077, 0.0122982, -0.0204830, 1.3299098 });
            ColorMatrix d65TosRGB = new ColorMatrix(new double[] { 3.2406, -1.5372, -0.4986, -0.9689, 1.8758, 0.0415, 0.0557, -0.2040, 1.0570 });
            ColorMatrix final = d65TosRGB * d50Tod65 * camToXYZ2;
            float[] matData = new float[9];
            for (int i = 0; i < 3; i++)
            {
                for (int j = 0; j < 3; j++)
                {
                    matData[j + i * 3] = (float)final[i, j];
                }
            }

            camToXYZKernel.RunSafe(inputImage32f, matData);
            camToXYZKernel.RunSafe(resultImage32f, matData);

            //This is a LUT that maps well to most of the JPEGs out of camera, but not always... found somewhere on internet, if I remember well from darktable?
            float[] x = new float[] {0,
                                0.004754f,
                                0.009529f,
                                0.023713f,
                                0.031866f,
                                0.046734f,
                                0.059989f,
                                0.088415f,
                                0.13661f,
                                0.17448f,
                                0.205192f,
                                0.228896f,
                                0.286411f,
                                0.355314f,
                                0.440014f,
                                0.567096f,
                                0.620597f,
                                0.760355f,
                                0.875139f,
                                1};
            float[] y = new float[] {0,
                                0.002208f,
                                0.004214f,
                                0.013508f,
                                0.020352f,
                                0.034063f,
                                0.052413f,
                                0.09603f,
                                0.190629f,
                                0.256484f,
                                0.30743f,
                                0.348447f,
                                0.42868f,
                                0.513527f,
                                0.607651f,
                                0.732791f,
                                0.775968f,
                                0.881828f,
                                0.960682f,
                                1};

            CudaDeviceVariable<float> d_x = x;
            CudaDeviceVariable<float> d_y = y;

            inputImage32f.LUTCubic(new CudaDeviceVariable<float>[] { d_y, d_y, d_y }, new CudaDeviceVariable<float>[] { d_x, d_x, d_x });
            resultImage32f.LUTCubic(new CudaDeviceVariable<float>[] { d_y, d_y, d_y }, new CudaDeviceVariable<float>[] { d_x, d_x, d_x });

            convertRGBTosRGBKernel.RunSafe(inputImage32f);
            convertRGBTosRGBKernel.RunSafe(resultImage32f);

            inputImage32f.Convert(noisyImage8u, NppRoundMode.Near);
            resultImage32f.Convert(resultImage8u, NppRoundMode.Near);

            noisyImage8u.SetRoi(0, 0, bmpNoisy.Width - 4, bmpNoisy.Height - 4);
            noisyImage8u.CopyToHostRoi(bmpNoisy, new NppiRect(2, 2, bmpNoisy.Width - 4, bmpNoisy.Height - 4));

            resultImage8u.SetRoi(0, 0, bmpResult.Width - 16, bmpResult.Height - 16);
            resultImage8u.CopyToHostRoi(bmpResult, new NppiRect(8, 8, bmpResult.Width - 16, bmpResult.Height - 16));

            pictureBox2.Image = bmpNoisy;
            pictureBox3.Image = bmpResult;

            rawTemp.Dispose();
            d_y.Dispose();
            d_x.Dispose();
        }

        private void btn_SaveNoisy_Click(object sender, EventArgs e)
        {
            if (bmpNoisy == null)
                return;

            SaveFileDialog sfd = new SaveFileDialog();
            sfd.Filter = "PNG image|*.png|Bitmap image|*.bmp|TIFF image|*.tif";

            if (sfd.ShowDialog() != DialogResult.OK)
                return;

            ImageFormat format = ImageFormat.Bmp;
            if (sfd.FileName.EndsWith(".png"))
                format = ImageFormat.Png;
            if (sfd.FileName.EndsWith(".tif"))
                format = ImageFormat.Tiff;

            bmpNoisy.Save(sfd.FileName, format);
        }

        private void btn_Save_Click(object sender, EventArgs e)
        {
            if (bmpResult == null)
                return;

            SaveFileDialog sfd = new SaveFileDialog();
            sfd.Filter = "PNG image|*.png|Bitmap image|*.bmp|TIFF image|*.tif";

            if (sfd.ShowDialog() != DialogResult.OK)
                return;

            ImageFormat format = ImageFormat.Bmp;
            if (sfd.FileName.EndsWith(".png"))
                format = ImageFormat.Png;
            if (sfd.FileName.EndsWith(".tif"))
                format = ImageFormat.Tiff;

            bmpResult.Save(sfd.FileName, format);
        }
    }
}
