using System;
using System.Collections.Generic;
using System.IO;
using System.Text;
using System.Threading.Tasks;
using ManagedCuda;
using ManagedCuda.BasicTypes;
using ManagedCuda.NPP;
using ManagedCuda.CudaDNN;

namespace NeuralNetworkInference
{
    public class DenoiseAndDemoisaic
    {
        StartLayer start;
        FinalLayer final;
        ConvolutionalLayer conv1;
        ConvolutionalLayer conv2;
        ConvolutionalLayer conv3;
        ConvolutionalLayerNPP conv1NPP;
        ConvolutionalLayerNPP conv2NPP;
        ConvolutionalLayerNPP conv3NPP;

        CudaDeviceVariable<float> tileAsPlanes;
        NPPImage_32fC3 tile;

        int _tileSize;

        public DenoiseAndDemoisaic(int tileSize, CudaContext ctx, CUmodule mod, bool UseCUDNN)
        {
            _tileSize = tileSize;
            start = new StartLayer(tileSize, tileSize, 3, 1);
            final = new FinalLayer(tileSize - 16, tileSize - 16, 3, 1, ctx, mod);

            if (UseCUDNN)
            {
                CudaDNNContext cuddn = new CudaDNNContext();
                conv1 = new ConvolutionalLayer(tileSize, tileSize, 3, tileSize - 8, tileSize - 8, 64, 1, 9, 9, ConvolutionalLayer.Activation.PRelu, cuddn, ctx, mod);
                conv2 = new ConvolutionalLayer(tileSize - 8, tileSize - 8, 64, tileSize - 12, tileSize - 12, 64, 1, 5, 5, ConvolutionalLayer.Activation.PRelu, cuddn, ctx, mod);
                conv3 = new ConvolutionalLayer(tileSize - 12, tileSize - 12, 64, tileSize - 16, tileSize - 16, 3, 1, 5, 5, ConvolutionalLayer.Activation.None, cuddn, ctx, mod);
                start.ConnectFollowingLayer(conv1);
                conv1.ConnectFollowingLayer(conv2);
                conv2.ConnectFollowingLayer(conv3);
                conv3.ConnectFollowingLayer(final);
            }
            else
            {
                conv1NPP = new ConvolutionalLayerNPP(tileSize, tileSize, 3, tileSize - 8, tileSize - 8, 64, 1, 9, 9, ConvolutionalLayerNPP.Activation.PRelu, ctx, mod);
                conv2NPP = new ConvolutionalLayerNPP(tileSize - 8, tileSize - 8, 64, tileSize - 12, tileSize - 12, 64, 1, 5, 5, ConvolutionalLayerNPP.Activation.PRelu, ctx, mod);
                conv3NPP = new ConvolutionalLayerNPP(tileSize - 12, tileSize - 12, 64, tileSize - 16, tileSize - 16, 3, 1, 5, 5, ConvolutionalLayerNPP.Activation.None, ctx, mod);
                start.ConnectFollowingLayer(conv1NPP);
                conv1NPP.ConnectFollowingLayer(conv2NPP);
                conv2NPP.ConnectFollowingLayer(conv3NPP);
                conv3NPP.ConnectFollowingLayer(final);
            }

            tileAsPlanes = new CudaDeviceVariable<float>(tileSize * tileSize * 3);
            tile = new NPPImage_32fC3(tileSize, tileSize);
        }

        public void LoadNetwork(string filename)
        {
            FileStream fs = new FileStream(filename, FileMode.Open, FileAccess.Read);

            start.RestoreValues(fs);

            long test = fs.Position - fs.Length;
            fs.Close();
            fs.Dispose();

            if (test != 0)
            {
                throw new IOException(filename + " doesn't fit to this network!");
            }
        }

        public void RunImage(NPPImage_32fC3 input, NPPImage_32fC3 output)
        {
            NppiRect roiOrig = new NppiRect(input.PointRoi, input.SizeRoi);

            output.Set(new float[] { 0, 0, 0 });
            IEnumerable<Tiler.RoiInputOutput> rois = Tiler.GetROIs(new NppiRect(new NppiPoint(8, 8), new NppiSize(input.WidthRoi - 16, input.HeightRoi - 16)), _tileSize, 8);

            foreach (var item in rois)
            {
                input.SetRoi(item.inputROI);
                output.SetRoi(item.positionInFinalImage);
                tile.ResetRoi();

                input.Copy(tile);

                NPPImage_32fC1 npp32fCR = new NPPImage_32fC1(tileAsPlanes.DevicePointer, _tileSize, _tileSize, _tileSize * sizeof(float));
                NPPImage_32fC1 npp32fCG = new NPPImage_32fC1(tileAsPlanes.DevicePointer + (_tileSize * _tileSize * sizeof(float)), _tileSize, _tileSize, _tileSize * sizeof(float));
                NPPImage_32fC1 npp32fCB = new NPPImage_32fC1(tileAsPlanes.DevicePointer + 2 * (_tileSize * _tileSize * sizeof(float)), _tileSize, _tileSize, _tileSize * sizeof(float));

                tile.Copy(npp32fCR, 0);
                tile.Copy(npp32fCG, 1);
                tile.Copy(npp32fCB, 2);

                start.Inference(tileAsPlanes);

                CudaDeviceVariable<float> res = final.GetResult();
                int size = _tileSize - 16;
                npp32fCR = new NPPImage_32fC1(res.DevicePointer, size, size, size * sizeof(float));
                npp32fCG = new NPPImage_32fC1(res.DevicePointer + (size * size * sizeof(float)), size, size, size * sizeof(float));
                npp32fCB = new NPPImage_32fC1(res.DevicePointer + 2 * (size * size * sizeof(float)), size, size, size * sizeof(float));

                tile.SetRoi(0, 0, _tileSize - 16, _tileSize - 16);

                npp32fCR.Copy(tile, 0);
                npp32fCG.Copy(tile, 1);
                npp32fCB.Copy(tile, 2);

                tile.SetRoi(item.outputROI);
                tile.Copy(output);
            }
            input.SetRoi(roiOrig);
            output.SetRoi(roiOrig);
        }
    }
}
