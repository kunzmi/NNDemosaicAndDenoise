using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using ManagedCuda;
using ManagedCuda.BasicTypes;
using ManagedCuda.VectorTypes;
using ManagedCuda.NPP;

namespace KernelClasses
{
    public class AddBorderKernel : CudaKernel
    {
        public AddBorderKernel(CUmodule module, CudaContext ctx)
            : base("addBorder", module, ctx)
        {
            //block size must be set in user code!
        }

        public float RunSafe(CudaDeviceVariable<float> imgIn, CudaDeviceVariable<float> imgOut, int channelCount, int batchCount, int patchSize, int borderSize)
        {
            SetComputeSize((uint)patchSize + 2 * (uint)borderSize, (uint)patchSize + 2 * (uint)borderSize, (uint)channelCount * (uint)batchCount);
            return base.Run(imgIn.DevicePointer, imgOut.DevicePointer, channelCount, batchCount, patchSize, borderSize);
        }
    }

    public class CropBorderKernel : CudaKernel
    {
        public CropBorderKernel(CUmodule module, CudaContext ctx)
            : base("cropBorder", module, ctx)
        {
            //block size must be set in user code!
        }

        public float RunSafe(CudaDeviceVariable<float> imgIn, CudaDeviceVariable<float> imgOut, int channelCount, int batchCount, int patchSize, int borderSize)
        {
            SetComputeSize((uint)patchSize, (uint)patchSize, (uint)channelCount * (uint)batchCount);
            return base.Run(imgIn.DevicePointer, imgOut.DevicePointer, channelCount, batchCount, patchSize, borderSize);
        }
    }

    public enum BayerColor : int
    {
        Red = 0,
        Green = 1,
        Blue = 2,
        Cyan = 3,
        Magenta = 4,
        Yellow = 5,
        White = 6
    }

    public class CreateBayerKernel : CudaKernel
    {
        public CreateBayerKernel(CUmodule module, CudaContext ctx)
            : base("CreateBayer", module, ctx)
        {
            BlockDimensions = new dim3(32, 16, 1);
            GridDimensions = new dim3(1, 1, 1);
        }

        public float RunSafe(NPPImage_32fC3 imgIn, NPPImage_32fC1 imgOut)
        {
            SetComputeSize((uint)imgIn.Width, (uint)imgIn.Height, 1);
            return base.Run(imgIn.DevicePointer, imgOut.DevicePointer, imgIn.Width, imgIn.Height, imgIn.Pitch, imgOut.Pitch);
        }

        public BayerColor[] BayerPattern
        {
            set
            {
                int[] temp = new int[value.Length];
                for (int i = 0; i < value.Length; i++)
                {
                    temp[i] = (int)value[i];
                }
                base.SetConstantVariable<int>("c_cfaPattern", temp);
            }
        }
    }

    public class DeBayerGreenKernel : CudaKernel
    {
        public DeBayerGreenKernel(CUmodule module, CudaContext ctx)
            : base("deBayerGreenKernel", module, ctx)
        {
            BlockDimensions = new dim3(32, 16, 1);
            GridDimensions = new dim3(1, 1, 1);
        }

        public float RunSafe(NPPImage_32fC1 imgIn, NPPImage_32fC3 imgOut, float3 blackPoint, float3 scale)
        {
            //const int width, const int height, const float* __restrict__ imgIn, int strideIn, float3 *outImage, int strideOut, float3 blackPoint, float3 scale
            SetComputeSize((uint)imgIn.Width, (uint)imgIn.Height, 1);
            return base.Run(imgIn.Width, imgIn.Height, imgIn.DevicePointer, imgIn.Pitch, imgOut.DevicePointer, imgOut.Pitch, blackPoint, scale);
        }

        public float RunSafe(CudaDeviceVariable<float> imgIn, CudaDeviceVariable<float3> imgOut, int patchSize, float3 blackPoint, float3 scale)
        {
            SetComputeSize((uint)patchSize, (uint)patchSize, 1);
            return base.Run(patchSize, patchSize, imgIn.DevicePointer, patchSize * 4, imgOut.DevicePointer, patchSize * 12, blackPoint, scale);
        }

        public BayerColor[] BayerPattern
        {
            set
            {
                int[] temp = new int[value.Length];
                for (int i = 0; i < value.Length; i++)
                {
                    temp[i] = (int)value[i];
                }
                base.SetConstantVariable<int>("c_cfaPattern", temp);
            }
        }
    }

    public class DeBayerRedBlueKernel : CudaKernel
    {
        public DeBayerRedBlueKernel(CUmodule module, CudaContext ctx)
            : base("deBayerRedBlueKernel", module, ctx)
        {
            BlockDimensions = new dim3(32, 16, 1);
            GridDimensions = new dim3(1, 1, 1);
        }

        public float RunSafe(NPPImage_32fC1 imgIn, NPPImage_32fC3 imgOut, float3 blackPoint, float3 scale)
        {
            //const int width, const int height, const float* __restrict__ imgIn, int strideIn, float3 *outImage, int strideOut, float3 blackPoint, float3 scale
            SetComputeSize((uint)imgIn.Width, (uint)imgIn.Height, 1);
            return base.Run(imgIn.Width, imgIn.Height, imgIn.DevicePointer, imgIn.Pitch, imgOut.DevicePointer, imgOut.Pitch, blackPoint, scale);
        }

        public float RunSafe(CudaDeviceVariable<float> imgIn, CudaDeviceVariable<float3> imgOut, int patchSize, float3 blackPoint, float3 scale)
        {
            SetComputeSize((uint)patchSize, (uint)patchSize, 1);
            return base.Run(patchSize, patchSize, imgIn.DevicePointer, patchSize * 4, imgOut.DevicePointer, patchSize * 12, blackPoint, scale);
        }

        public BayerColor[] BayerPattern
        {
            set
            {
                int[] temp = new int[value.Length];
                for (int i = 0; i < value.Length; i++)
                {
                    temp[i] = (int)value[i];
                }
                base.SetConstantVariable<int>("c_cfaPattern", temp);
            }
        }
    }
    
    public class DeBayersSubSampleKernel : CudaKernel
    {
        private const uint BlockSizeX = 16;
        private const uint BlockSizeY = 16;

        public DeBayersSubSampleKernel(CudaContext ctx, CUmodule module)
            : base("deBayersSubSample3", module, ctx, BlockSizeX, BlockSizeY)
        {
            //deBayersSubSample(unsigned short* dataIn, float3* imgOut, int bitDepth, int dimX, int dimY, int strideOut)
        }

        public float RunSafe(CudaDeviceVariable<ushort> imgIn, NPPImage_32fC3 imgOut, float maxVal)
        {
            SetComputeSize((uint)imgOut.WidthRoi, (uint)imgOut.HeightRoi);
            return base.Run(imgIn.DevicePointer, imgOut.DevicePointer, maxVal, imgOut.WidthRoi, imgOut.HeightRoi, imgOut.Pitch);
        }

        public BayerColor[] BayerPattern
        {
            set
            {
                int[] temp = new int[value.Length];
                for (int i = 0; i < value.Length; i++)
                {
                    temp[i] = (int)value[i];
                }
                base.SetConstantVariable<int>("c_cfaPattern", temp);
            }
        }
    }

    public class DeBayersSubSampleDNGKernel : CudaKernel
    {
        private const uint BlockSizeX = 16;
        private const uint BlockSizeY = 16;

        public DeBayersSubSampleDNGKernel(CudaContext ctx, CUmodule module)
            : base("deBayersSubSample3DNG", module, ctx, BlockSizeX, BlockSizeY)
        {
            //deBayersSubSample(unsigned short* dataIn, float3* imgOut, int bitDepth, int dimX, int dimY, int strideOut)
        }

        public float RunSafe(CudaDeviceVariable<ushort> imgIn, NPPImage_32fC3 imgOut, float[] maxVal, float[] minVal)
        {
            SetComputeSize((uint)imgOut.WidthRoi, (uint)imgOut.HeightRoi);
            return base.Run(imgIn.DevicePointer, imgOut.DevicePointer, maxVal[0], maxVal[1], maxVal[2], minVal[0], minVal[1], minVal[2], imgOut.WidthRoi, imgOut.HeightRoi, imgOut.Pitch);
        }

        public BayerColor[] BayerPattern
        {
            set
            {
                int[] temp = new int[value.Length];
                for (int i = 0; i < value.Length; i++)
                {
                    temp[i] = (int)value[i];
                }
                base.SetConstantVariable<int>("c_cfaPattern", temp);
            }
        }
    }

    public class SetupCurandKernel : CudaKernel
    {
        private const uint BlockSizeX = 512;
        private const uint BlockSizeY = 1;

        public SetupCurandKernel(CudaContext ctx, CUmodule module)
            : base("setupCurand", module, ctx, BlockSizeX, BlockSizeY)
        {
            //setupCurand(curandState* states, int count)
        }

        public float RunSafe(CudaDeviceVariable<byte> states, int count)
        {
            SetComputeSize((uint)count);
            return base.Run(states.DevicePointer, count);
        }
    }

    public class CreateBayerWithNoiseKernel : CudaKernel
    {
        private const uint BlockSizeX = 16;
        private const uint BlockSizeY = 16;

        public CreateBayerWithNoiseKernel(CudaContext ctx, CUmodule module)
            : base("CreateBayerWithNoise", module, ctx, BlockSizeX, BlockSizeY)
        {
            //CreateBayerWithNoise(curandState* states, float3* imgIn, float* imgOut, int dimX, int dimY, int strideIn, int strideOut, float alpha, float beta)
        }

        public float RunSafe(CudaDeviceVariable<byte> states, NPPImage_32fC3 imgIn, NPPImage_32fC1 imgOut, float alpha, float beta)
        {
            SetComputeSize((uint)imgOut.WidthRoi, (uint)imgOut.HeightRoi);
            return base.Run(states.DevicePointer, imgIn.DevicePointerRoi, imgOut.DevicePointerRoi, imgOut.WidthRoi, imgOut.HeightRoi, imgIn.Pitch, imgOut.Pitch, alpha, beta);
        }

        public BayerColor[] BayerPattern
        {
            set
            {
                int[] temp = new int[value.Length];
                for (int i = 0; i < value.Length; i++)
                {
                    temp[i] = (int)value[i];
                }
                base.SetConstantVariable<int>("c_cfaPattern", temp);
            }
        }
    }

    public class MSSSIML1Kernel : CudaKernel
    {
        public MSSSIML1Kernel(CUmodule module, CudaContext ctx)
            : base("MSSSIML1", module, ctx)
        {
            GridDimensions = new dim3(1, 1, 1);
            BlockDimensions = new dim3(64, 3, 1);
        }

        public float RunSafe(CudaDeviceVariable<float> computed, CudaDeviceVariable<float> groundTrouth, CudaDeviceVariable<float> msssiml1, CudaDeviceVariable<float> d_msssiml1, int channelCount, int batchCount, float alpha)
        {
            SetComputeSize((uint)batchCount, (uint)channelCount, 1);
            //MSSSIML1(float* imgPatch1, float* imgPatch2, float* msssiml1, float* d_mssiml1, int channelCount, int batchCount, float alpha)
            return base.Run(computed.DevicePointer, groundTrouth.DevicePointer, msssiml1.DevicePointer, d_msssiml1.DevicePointer, channelCount, batchCount, alpha);
        }
    }

    public class ConvertCamToXYZKernel : CudaKernel
    {
        public ConvertCamToXYZKernel(CUmodule module, CudaContext ctx)
            : base("convertCamToXYZKernel", module, ctx)
        {
            BlockDimensions = new dim3(32, 16, 1);
        }

        public float RunSafe(NPPImage_32fC3 img, float[] ColorMatrix)
        {
            float[] RgbToXyz = ColorMatrix;
            float3 xyzX = new float3(RgbToXyz[0], RgbToXyz[1], RgbToXyz[2]);
            float3 xyzY = new float3(RgbToXyz[3], RgbToXyz[4], RgbToXyz[5]);
            float3 xyzZ = new float3(RgbToXyz[6], RgbToXyz[7], RgbToXyz[8]);

            //const int width, const int height, float3 *inOutImage, int strideOut, float3 xyzX, float3 xyzY, float3 xyzZ
            SetComputeSize((uint)img.WidthRoi, (uint)img.HeightRoi, 1);
            return base.Run(img.WidthRoi, img.HeightRoi, img.DevicePointerRoi, img.Pitch, xyzX, xyzY, xyzZ);
        }
    }

    public class HighlightRecoveryKernel : CudaKernel
    {
        public HighlightRecoveryKernel(CUmodule module, CudaContext ctx)
            : base("HighlightRecovery", module, ctx)
        {
            BlockDimensions = new dim3(32, 16, 1);
        }

        public float RunSafe(NPPImage_32fC3 img, float3 whiteBalanceFactors, float maxMultiplier)
        {
            //const int width, const int height, float3 *inOutImage, int strideOut, float3 WhiteBalanceFactors
            SetComputeSize((uint)img.WidthRoi, (uint)img.HeightRoi, 1);
            return base.Run(img.WidthRoi, img.HeightRoi, img.DevicePointerRoi, img.Pitch, whiteBalanceFactors, maxMultiplier);
        }
    }

    public class ConvertRGBTosRGBKernel : CudaKernel
    {
        public ConvertRGBTosRGBKernel(CUmodule module, CudaContext ctx)
               : base("convertRGBTosRGBKernel", module, ctx)
        {
            BlockDimensions = new dim3(32, 16, 1);
        }

        public float RunSafe(NPPImage_32fC3 img)
        {
            //convertXYZTosRGBKernel(const int width, const int height, float3 *inOutImage, int strideOut)
            SetComputeSize((uint)img.WidthRoi, (uint)img.HeightRoi, 1);
            return base.Run(img.WidthRoi, img.HeightRoi, img.DevicePointerRoi, img.Pitch);
        }
    }

    public class PrepareDataKernel : CudaKernel
    {
        public PrepareDataKernel(CUmodule module, CudaContext ctx)
            : base("prepareData", module, ctx)
        {
            GridDimensions = new dim3(1, 1, 4);
            BlockDimensions = new dim3(31, 31, 1);
        }

        public float RunSafe(CudaDeviceVariable<float3> imgInA, CudaDeviceVariable<float3> imgInB, CudaDeviceVariable<float> imgOutA, CudaDeviceVariable<float> imgOutB, int batch, float sub, float3 whiteBalance)
        {
            //const float3* __restrict__ imgInA, const float3* __restrict__ imgInB, float* imgOutA, float* imgOutB, int batch
            return base.Run(imgInA.DevicePointer, imgInB.DevicePointer, imgOutA.DevicePointer, imgOutB.DevicePointer, batch, sub, whiteBalance);
        }
    }

    public class RestoreImageKernel : CudaKernel
    {
        public RestoreImageKernel(CUmodule module, CudaContext ctx)
            : base("restoreImage", module, ctx)
        {
            GridDimensions = new dim3(1, 1, 1);
            BlockDimensions = new dim3(31, 31, 1);
        }

        public float RunSafe(CudaDeviceVariable<float> imgIn, NPPImage_8uC3 imgOut, int imgOffset, float facR, float facG, float facB, float add)
        {
            //restoreImage(const float* __restrict__ imgIn, uchar3* imgOut, float facR, float facG, float facB)
            return base.Run(imgIn.DevicePointer, imgOut.DevicePointer, imgOut.Pitch, imgOffset, facR, facG, facB, add);
        }
    }

    public class PReluForwardKernel : CudaKernel
    {
        public PReluForwardKernel(CUmodule module, CudaContext ctx)
            : base("PRelu_Forward", module, ctx)
        {
            GridDimensions = new dim3(1, 1, 1);
            BlockDimensions = new dim3(512, 1, 1);
        }

        public float RunSafe(CudaDeviceVariable<float> dataIn, CudaDeviceVariable<float> a, CudaDeviceVariable<float> dataOut, int dataCount, int channelCount, int batchCount)
        {
            //__global__ void PRelu_Forward(const float* __restrict__ dataIn, const float* __restrict__ a, float* dataOut, int dataCount, int channelCount)
            return base.Run(dataIn.DevicePointer, a.DevicePointer, dataOut.DevicePointer, dataCount, channelCount, batchCount);
        }
    }

    public class PReluBackwardKernel : CudaKernel
    {
        public PReluBackwardKernel(CUmodule module, CudaContext ctx)
            : base("PRelu_Backward", module, ctx)
        {
            GridDimensions = new dim3(1, 1, 1);
            BlockDimensions = new dim3(512, 1, 1);
        }

        public float RunSafe(CudaDeviceVariable<float> dataIn, CudaDeviceVariable<float> dY, CudaDeviceVariable<float> a, CudaDeviceVariable<float> dData, CudaDeviceVariable<float> dA, int dataCount, int channelCount, int batchCount)
        {
            //__global__ void PRelu_Backward(const float* __restrict__ dataIn, const float* __restrict__ dY, const float* __restrict__ a, float* dData, float* dA, int dataCount, int channelCount)
            return base.Run(dataIn.DevicePointer, dY.DevicePointer, a.DevicePointer, dData.DevicePointer, dA.DevicePointer, dataCount, channelCount, batchCount);
        }
    }

    public class PReluBackward1Kernel : CudaKernel
    {
        public PReluBackward1Kernel(CUmodule module, CudaContext ctx)
            : base("PRelu_Backward1", module, ctx)
        {
            GridDimensions = new dim3(1, 1, 1);
            BlockDimensions = new dim3(32, 32, 1);
        }

        public float RunSafe(CudaDeviceVariable<float> dataIn, CudaDeviceVariable<float> dY, CudaDeviceVariable<float> a, CudaDeviceVariable<float> dData, CudaDeviceVariable<float> dA, CudaDeviceVariable<float> temp, int dataCount, int channelCount, int batchCount)
        {
            SetComputeSize((uint)channelCount, (uint)batchCount);
            //__global__ void PRelu_Backward(const float* __restrict__ dataIn, const float* __restrict__ dY, const float* __restrict__ a, float* dData, float* dA, int dataCount, int channelCount)
            return base.Run(dataIn.DevicePointer, dY.DevicePointer, a.DevicePointer, dData.DevicePointer, dA.DevicePointer, temp.DevicePointer, dataCount, channelCount, batchCount);
        }
    }

    public class PReluBackward2Kernel : CudaKernel
    {
        public PReluBackward2Kernel(CUmodule module, CudaContext ctx)
            : base("PRelu_Backward2", module, ctx)
        {
            GridDimensions = new dim3(1, 1, 1);
            BlockDimensions = new dim3(512, 1, 1);
        }

        public float RunSafe(CudaDeviceVariable<float> dataIn, CudaDeviceVariable<float> dY, CudaDeviceVariable<float> a, CudaDeviceVariable<float> dData, CudaDeviceVariable<float> dA, CudaDeviceVariable<float> temp, int dataCount, int channelCount, int batchCount)
        {
            SetComputeSize((uint)channelCount);
            //__global__ void PRelu_Backward(const float* __restrict__ dataIn, const float* __restrict__ dY, const float* __restrict__ a, float* dData, float* dA, int dataCount, int channelCount)
            return base.Run(dataIn.DevicePointer, dY.DevicePointer, a.DevicePointer, dData.DevicePointer, dA.DevicePointer, temp.DevicePointer, dataCount, channelCount, batchCount);
        }
    }
}
