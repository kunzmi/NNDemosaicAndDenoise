using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using ManagedCuda;
using ManagedCuda.BasicTypes;
using ManagedCuda.NPP;
using System.IO;
using KernelClasses;

namespace NeuralNetworkInference
{
    class ConvolutionalLayerNPP : Layer
    {
        public enum Activation
        {
            None,
            Relu,
            PRelu,
            LeakyRelu
        }
        int _filterX, _filterY;
        
        CudaDeviceVariable<float> _weights;
        CudaDeviceVariable<float> _bias;
        CudaDeviceVariable<float> _y;
        CudaDeviceVariable<float> _z;
        CudaDeviceVariable<float> _input;
        CudaDeviceVariable<float> _aRelu;
        CudaDeviceVariable<float> _tempConvolution;
        float[] bHost;

        PReluForwardKernel _KernelPReluForward;
        Activation _activation;

        public ConvolutionalLayerNPP(int widthIn, int heightIn, int channelsIn, int widthOut, int heightOut, int channelsOut, int batch, int filterWidth, int filterHeight, Activation activation, CudaContext ctx, CUmodule module)
            : base(widthIn, heightIn, channelsIn, widthOut, heightOut, channelsOut, batch)
        {
            _activation = activation;
            _filterX = filterWidth;
            _filterY = filterHeight;
            _weights = new CudaDeviceVariable<float>(filterWidth * filterHeight * channelsIn * channelsOut);
            _bias = new CudaDeviceVariable<float>(channelsOut);
            _y = new CudaDeviceVariable<float>(widthOut * heightOut * channelsOut * batch);
            _z = new CudaDeviceVariable<float>(widthOut * heightOut * channelsOut * batch);
            _tempConvolution = new CudaDeviceVariable<float>(widthOut * heightOut);


            if (_activation == Activation.PRelu || _activation == Activation.LeakyRelu)
            {
                _aRelu = new CudaDeviceVariable<float>(channelsOut);
                _KernelPReluForward = new PReluForwardKernel(module, ctx);
                _KernelPReluForward.SetComputeSize((uint)widthOut * (uint)heightOut, (uint)channelsOut, (uint)batch);
            }
            else
                if (_activation == Activation.Relu)
                {
                    _aRelu = new CudaDeviceVariable<float>(channelsOut);
                    _aRelu.Memset(0); //use a fixed alpha of 0 for Relu...
                    _KernelPReluForward = new PReluForwardKernel(module, ctx);
                    _KernelPReluForward.SetComputeSize((uint)widthOut * (uint)heightOut, (uint)channelsOut, (uint)batch);
                }
        }

        public override float Inference(CudaDeviceVariable<float> input)
        {
            _input = input;

            NPPImage_32fC1 tempConv = new NPPImage_32fC1(_tempConvolution.DevicePointer, InWidth, InHeight, InWidth * sizeof(float));
            for (int outLayer = 0; outLayer < OutChannels; outLayer++)
            {
                SizeT offsetOut = outLayer * OutWidth * OutHeight * sizeof(float);
                CUdeviceptr ptrWithOffsetOut = _z.DevicePointer + offsetOut;
                NPPImage_32fC1 imgOut = new NPPImage_32fC1(ptrWithOffsetOut, OutWidth, OutHeight, OutWidth * sizeof(float));
                imgOut.Set(0);

                for (int inLayer = 0; inLayer < InChannels; inLayer++)
                {
                    SizeT offsetIn = inLayer * InWidth * InHeight * sizeof(float);
                    CUdeviceptr ptrWithOffsetIn = _input.DevicePointer + offsetIn;
                    NPPImage_32fC1 imgIn = new NPPImage_32fC1(ptrWithOffsetIn, InWidth, InHeight, InWidth * sizeof(float));

                    imgIn.SetRoi(_filterX / 2, _filterY / 2, InWidth - _filterX + 1, InHeight - _filterY + 1);

                    SizeT offsetFilter = (outLayer * InChannels * _filterX * _filterY + inLayer * _filterX * _filterY) * sizeof(float);
                    CudaDeviceVariable<float> filter = new CudaDeviceVariable<float>(_weights.DevicePointer + offsetFilter, false, _filterX * _filterY * sizeof(float));
                    
                    imgIn.Filter(tempConv, filter, new NppiSize(_filterX, _filterY), new NppiPoint(_filterX / 2, _filterY / 2));
                    imgOut.Add(tempConv);
                }
                imgOut.Add(bHost[outLayer]);
            }

            switch (_activation)
            {
                case Activation.None:
                    _y.CopyToDevice(_z);
                    break;
                case Activation.Relu:
                    //_aRelu is set to 0!
                    _KernelPReluForward.RunSafe(_z, _aRelu, _y, _outWidth * _outHeight, _outChannels, _batch);
                    break;
                case Activation.PRelu:
                    _KernelPReluForward.RunSafe(_z, _aRelu, _y, _outWidth * _outHeight, _outChannels, _batch);
                    break;
                case Activation.LeakyRelu:
                    _KernelPReluForward.RunSafe(_z, _aRelu, _y, _outWidth * _outHeight, _outChannels, _batch);
                    break;
                default:
                    break;
            }

            return _nextLayer.Inference(_y);
        }

        public override void RestoreValues(Stream stream)
        {
            BinaryReader br = new BinaryReader(stream);
            float[] w = new float[_weights.Size];
            bHost = new float[_bias.Size];

            for (int i = 0; i < _weights.Size; i++)
            {
                w[i] = br.ReadSingle();
            }
            for (int i = 0; i < _bias.Size; i++)
            {
                bHost[i] = br.ReadSingle();
            }
            if (_activation == Activation.PRelu || _activation == Activation.LeakyRelu)
            {
                float[] a = new float[_aRelu.Size];
                for (int i = 0; i < _aRelu.Size; i++)
                {
                    a[i] = br.ReadSingle();
                }
                _aRelu.CopyToDevice(a);
            }
            _weights.CopyToDevice(w);
            _bias.CopyToDevice(bHost);

            base.RestoreValues(stream);
        }
    }
}
