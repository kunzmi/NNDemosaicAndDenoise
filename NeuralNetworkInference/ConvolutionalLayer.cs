using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.IO;
using ManagedCuda;
using ManagedCuda.BasicTypes;
using ManagedCuda.CudaDNN;
using ManagedCuda.NPP.NPPsExtensions;
using KernelClasses;

namespace NeuralNetworkInference
{
    public class ConvolutionalLayer : Layer
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
        CudaDeviceVariable<byte> _workspace;
        CudaDeviceVariable<float> _aRelu;
        CudaDeviceVariable<float> _dARelu;
        ActivationDescriptor _descActivation;
        TensorDescriptor _descBias;
        TensorDescriptor _descDataIn;
        TensorDescriptor _descDataInBorder;
        TensorDescriptor _descDataOut;
        FilterDescriptor _descFilter;
        //FilterDescriptor _descFilterBorder;
        ConvolutionDescriptor _descConv;
        ConvolutionDescriptor _descConvBorder;
        CudaDNNContext _cudnn;
        PReluForwardKernel _KernelPReluForward;
        PReluBackwardKernel _KernelPReluBackward;
        Activation _activation;
        cudnnConvolutionFwdAlgo _algoFwd;
        cudnnConvolutionBwdDataAlgo _algoBwdData;
        cudnnConvolutionBwdFilterAlgo _algoBwdFilter;
        float alpha = 1.0f;
        float beta = 0.0f;

        public ConvolutionalLayer(int widthIn, int heightIn, int channelsIn, int widthOut, int heightOut, int channelsOut, int batch, int filterWidth, int filterHeight, Activation activation, CudaDNNContext cudnnCtx, CudaContext ctx, CUmodule module)
            : base(widthIn, heightIn, channelsIn, widthOut, heightOut, channelsOut, batch)
        {
            _activation = activation;
            _filterX = filterWidth;
            _filterY = filterHeight;
            _weights = new CudaDeviceVariable<float>(filterWidth * filterHeight * channelsIn * channelsOut);
            _bias = new CudaDeviceVariable<float>(channelsOut);
            _y = new CudaDeviceVariable<float>(widthOut * heightOut * channelsOut * batch);
            _z = new CudaDeviceVariable<float>(widthOut * heightOut * channelsOut * batch);

            _cudnn = cudnnCtx;
            
            _descActivation = new ActivationDescriptor();
            _descActivation.SetActivationDescriptor(cudnnActivationMode.Relu, cudnnNanPropagation.NotPropagateNan, 0);
            _descBias = new TensorDescriptor();
            _descBias.SetTensor4dDescriptor(cudnnTensorFormat.NCHW, cudnnDataType.Float, 1, channelsOut, 1, 1);
            _descDataInBorder = new TensorDescriptor();
            _descDataIn = new TensorDescriptor();
            _descDataIn.SetTensor4dDescriptor(cudnnTensorFormat.NCHW, cudnnDataType.Float, batch, channelsIn, heightIn , widthIn );
            _descDataOut = new TensorDescriptor();
            _descDataOut.SetTensor4dDescriptor(cudnnTensorFormat.NCHW, cudnnDataType.Float, batch, channelsOut, heightOut, widthOut);
            _descFilter = new FilterDescriptor();
            _descFilter.SetFilter4dDescriptor(cudnnDataType.Float, cudnnTensorFormat.NCHW, channelsOut, channelsIn, filterWidth, filterHeight);
            _descConv = new ConvolutionDescriptor();
            _descConvBorder = new ConvolutionDescriptor();
            _descConv.SetConvolution2dDescriptor(0, 0, 1, 1, 1, 1, cudnnConvolutionMode.Convolution, cudnnDataType.Float);

            int n = 0;
            int c = 0;
            int h = 0;
            int w = 0;
            _descConv.GetConvolution2dForwardOutputDim(_descDataIn, _descFilter, ref n, ref c, ref h, ref w);
            
            if (_activation == Activation.PRelu || _activation == Activation.LeakyRelu)
            {
                _aRelu = new CudaDeviceVariable<float>(channelsOut);
                _dARelu = new CudaDeviceVariable<float>(channelsOut);
                _KernelPReluForward = new PReluForwardKernel(module, ctx);
                _KernelPReluBackward = new PReluBackwardKernel(module, ctx);
                _KernelPReluForward.SetComputeSize((uint)widthOut * (uint)heightOut, (uint)channelsOut, (uint)batch);
                _KernelPReluBackward.SetComputeSize((uint)channelsOut, 1, 1);
            }

            cudnnConvolutionFwdAlgoPerf[] algos = 
            _cudnn.FindConvolutionForwardAlgorithm(_descDataIn, _descFilter, _descConv, _descDataOut, 5);

            cudnnConvolutionBwdDataAlgoPerf[] algos2 = _cudnn.FindConvolutionBackwardDataAlgorithm(_descFilter, _descDataOut, _descConv, _descDataIn, 5);

            _algoFwd = _cudnn.GetConvolutionForwardAlgorithm(_descDataIn, _descFilter, _descConv,
                _descDataOut, cudnnConvolutionFwdPreference.PreferFastest, 0);


            SizeT sizeInBytes = 0, tmpsize = 0;
            sizeInBytes = _cudnn.GetConvolutionForwardWorkspaceSize(_descDataIn, _descFilter,
                _descConv, _descDataOut, _algoFwd);

            _algoBwdFilter = _cudnn.GetConvolutionBackwardFilterAlgorithm(_descDataIn, _descDataOut, _descConv, _descFilter,
                cudnnConvolutionBwdFilterPreference.PreferFastest, 0);
            
            tmpsize = _cudnn.GetConvolutionBackwardFilterWorkspaceSize(_descDataIn, _descDataOut, _descConv, _descFilter, _algoBwdFilter);
            sizeInBytes = Math.Max(sizeInBytes, tmpsize);

            _algoBwdData = _cudnn.GetConvolutionBackwardDataAlgorithm(_descFilter, _descDataOut, _descConv, _descDataIn, cudnnConvolutionBwdDataPreference.PreferFastest, 0);
            
            tmpsize = _cudnn.GetConvolutionBackwardDataWorkspaceSize(_descFilter, _descDataOut, _descConv, _descDataIn, _algoBwdData);
            sizeInBytes = Math.Max(sizeInBytes, tmpsize);

            if (sizeInBytes > 0)
                _workspace = new CudaDeviceVariable<byte>(sizeInBytes);
            else
                _workspace = CudaDeviceVariable<byte>.Null;
        }

        public override float Inference(CudaDeviceVariable<float> input)
        {
            _input = input;
         
            _cudnn.ConvolutionForward(alpha, _descDataIn, _input,
                                               _descFilter, _weights, _descConv,
                                               _algoFwd, _workspace, beta,
                                               _descDataOut, _z);


            _cudnn.AddTensor(alpha, _descBias, _bias, alpha, _descDataOut, _z);

            switch (_activation)
            {
                case Activation.None:
                    _y.CopyToDevice(_z);
                    break;
                case Activation.Relu:
                    _cudnn.ActivationForward(_descActivation, alpha, _descDataOut, _z, beta, _descDataOut, _y);
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
            float[] b = new float[_bias.Size];

            for (int i = 0; i < _weights.Size; i++)
            {
                w[i] = br.ReadSingle();
            }
            for (int i = 0; i < _bias.Size; i++)
            {
                b[i] = br.ReadSingle();
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
            _bias.CopyToDevice(b);
            base.RestoreValues(stream);
        }
    }
}
