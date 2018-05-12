using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using ManagedCuda;
using ManagedCuda.BasicTypes;
using ManagedCuda.CudaDNN;
using ManagedCuda.NPP.NPPsExtensions;
using ManagedCuda.CudaBlas;
using System.IO;
using KernelClasses;

namespace NeuralNetworkTraining
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
        AddBorderKernel _kernelAddBorder;
        CropBorderKernel _kernelCropBorder;

        CudaDeviceVariable<float> _withBorderInput;
        CudaDeviceVariable<float> _withBorderDx;
        CudaDeviceVariable<float> _weights;
        CudaDeviceVariable<float> _d_weights;
        CudaDeviceVariable<float> _bias;
        CudaDeviceVariable<float> _d_bias;
        CudaDeviceVariable<float> _dx;
        CudaDeviceVariable<float> _dy;
        CudaDeviceVariable<float> _y;
        CudaDeviceVariable<float> _z;
        CudaDeviceVariable<float> _ones;
        CudaDeviceVariable<float> _input;
        CudaDeviceVariable<byte> _workspace;
        CudaDeviceVariable<float> _aRelu;
        CudaDeviceVariable<float> _dARelu;
        CudaDeviceVariable<float> _temp;
        ActivationDescriptor _descActivation;
        TensorDescriptor _descBias;
        TensorDescriptor _descDataIn;
        TensorDescriptor _descDataInBorder;
        TensorDescriptor _descDataOut;
        FilterDescriptor _descFilter;
        //FilterDescriptor _descFilterBorder;
        ConvolutionDescriptor _descConv;
        ConvolutionDescriptor _descConvBorder;
        CudaBlas _blas;
        CudaDNNContext _cudnn;
        PReluForwardKernel _KernelPReluForward;
        PReluBackwardKernel _KernelPReluBackward;
        PReluBackward1Kernel _KernelPReluBackward1;
        PReluBackward2Kernel _KernelPReluBackward2;
        Activation _activation;
        cudnnConvolutionFwdAlgo _algoFwd;
        cudnnConvolutionBwdDataAlgo _algoBwdData;
        cudnnConvolutionBwdFilterAlgo _algoBwdFilter;
        float alpha = 1.0f;
        float beta = 0.0f;

        public ConvolutionalLayer(int widthIn, int heightIn, int channelsIn, int widthOut, int heightOut, int channelsOut, int batch, int filterWidth, int filterHeight, Activation activation, CudaBlas blasCtx, CudaDNNContext cudnnCtx, CudaContext ctx, CUmodule moduleBorder, CUmodule modulePrelu)
            : base(widthIn, heightIn, channelsIn, widthOut, heightOut, channelsOut, batch)
        {
            _activation = activation;
            _filterX = filterWidth;
            _filterY = filterHeight;
            _weights = new CudaDeviceVariable<float>(filterWidth * filterHeight * channelsIn * channelsOut);
            _d_weights = new CudaDeviceVariable<float>(filterWidth * filterHeight * channelsIn * channelsOut);
            _bias = new CudaDeviceVariable<float>(channelsOut);
            _d_bias = new CudaDeviceVariable<float>(channelsOut);
            _dx = new CudaDeviceVariable<float>(widthIn * heightIn * channelsIn * batch);
            _y = new CudaDeviceVariable<float>(widthOut * heightOut * channelsOut * batch);
            _dy = new CudaDeviceVariable<float>(widthOut * heightOut * channelsOut * batch);
            _z = new CudaDeviceVariable<float>(widthOut * heightOut * channelsOut * batch);
            _ones = new CudaDeviceVariable<float>(batch);
            _withBorderInput = new CudaDeviceVariable<float>((widthIn + filterWidth - 1) * (heightIn + filterHeight - 1)* channelsIn * batch);
            _withBorderDx = new CudaDeviceVariable<float>((widthIn + filterWidth - 1) * (heightIn + filterHeight - 1)* channelsIn * batch);
            _cudnn = cudnnCtx;
            _blas = blasCtx;
            _descActivation = new ActivationDescriptor();
            _descActivation.SetActivationDescriptor(cudnnActivationMode.Relu, cudnnNanPropagation.NotPropagateNan, 0);
            _descBias = new TensorDescriptor();
            _descBias.SetTensor4dDescriptor(cudnnTensorFormat.NCHW, cudnnDataType.Float, 1, channelsOut, 1, 1);
            _descDataInBorder = new TensorDescriptor();
            _descDataIn = new TensorDescriptor();
            _descDataIn.SetTensor4dDescriptor(cudnnTensorFormat.NCHW, cudnnDataType.Float, batch, channelsIn, heightIn + filterHeight - 1, widthIn + filterWidth - 1);
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

            _kernelAddBorder = new AddBorderKernel(moduleBorder, ctx);
            _kernelAddBorder.BlockDimensions = new ManagedCuda.VectorTypes.dim3(widthIn + filterWidth - 1, (heightIn + filterHeight - 1) / 2+1, 1);
            _kernelCropBorder = new CropBorderKernel(moduleBorder, ctx);
            _kernelCropBorder.BlockDimensions = new ManagedCuda.VectorTypes.dim3(widthIn, heightIn / 2+1, 1);

            if (_activation == Activation.PRelu || _activation == Activation.LeakyRelu)
            {
                _temp = new CudaDeviceVariable<float>(channelsOut * batch);
                _aRelu = new CudaDeviceVariable<float>(channelsOut);
                _dARelu = new CudaDeviceVariable<float>(channelsOut);
                _KernelPReluForward = new PReluForwardKernel(modulePrelu, ctx);
                _KernelPReluBackward = new PReluBackwardKernel(modulePrelu, ctx);
                _KernelPReluBackward1 = new PReluBackward1Kernel(modulePrelu, ctx);
                _KernelPReluBackward2 = new PReluBackward2Kernel(modulePrelu, ctx);
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

        public override void InitRandomWeight(Random rand)
        {
            // Xavier weight filling * _outChannels
            float wconv1 = (float)Math.Sqrt(3.0f / (_filterX * _filterY * _inChannels));
            float[] w = new float[_weights.Size];
            float[] b = new float[_bias.Size];

            // Randomize network
            for (int i = 0; i < _weights.Size; i++)
            {
                w[i] = (float)((rand.NextDouble() * 2.0 - 1.0) * wconv1);
            }
            for (int i = 0; i < _bias.Size; i++)
            {
                b[i] = (float)((rand.NextDouble() * 2.0 - 1.0) * wconv1);
            }
            _weights.CopyToDevice(w);
            _bias.CopyToDevice(b);

            switch (_activation)
            {
                case Activation.PRelu:
                    _aRelu.Set(0.25f);
                    break;
                case Activation.LeakyRelu:
                    _aRelu.Set(0.25f);
                    break;
                default:
                    break;
            }
            base.InitRandomWeight(rand);
        }

        public void SetWeights(float[] w)
        {
            _weights.CopyToDevice(w);
        }

        public void SetBias(float[] b)
        {
            _bias.CopyToDevice(b);
        }

        public void SetActivation(float[] a)
        {
            _aRelu.CopyToDevice(a);
        }

        public void SetActivation(float a)
        {
            _aRelu.Set(a);
        }

        public override float InferenceTraining(CudaDeviceVariable<float> input)
        {
            _kernelAddBorder.RunSafe(input, _withBorderInput, _inChannels, _batch, _inWidth, _filterX / 2);

            _input = _withBorderInput;

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

            return _nextLayer.InferenceTraining(_y);
        }

        public override CudaDeviceVariable<float> BackPropagation(CudaDeviceVariable<float> dy)
        {
            switch (_activation)
            {
                case Activation.None:
                    _dy.CopyToDevice(dy);
                    break;
                case Activation.Relu:
                    _cudnn.ActivationBackward(_descActivation, alpha, _descDataOut, _y, _descDataOut, dy, _descDataOut, _z, beta, _descDataOut, _dy);
                    break;
                case Activation.PRelu:
                    _KernelPReluBackward1.RunSafe(_z, dy, _aRelu, _dy, _dARelu, _temp, _outWidth * _outHeight, _outChannels, _batch);
                    _KernelPReluBackward2.RunSafe(_z, dy, _aRelu, _dy, _dARelu, _temp, _outWidth * _outHeight, _outChannels, _batch);
                    break;
                case Activation.LeakyRelu:
                    _KernelPReluBackward1.RunSafe(_z, dy, _aRelu, _dy, _dARelu, _temp, _outWidth * _outHeight, _outChannels, _batch);
                    _KernelPReluBackward2.RunSafe(_z, dy, _aRelu, _dy, _dARelu, _temp, _outWidth * _outHeight, _outChannels, _batch);
                    break;
                default:
                    break;
            }
            _cudnn.ConvolutionBackwardBias(alpha, _descDataOut,
                                                    _dy, beta, _descBias, _d_bias);


            _cudnn.ConvolutionBackwardFilter(alpha, _descDataIn,
                                                      _input, _descDataOut, _dy, _descConv,
                                                      _algoBwdFilter, _workspace,
                                                      beta, _descFilter, _d_weights);

            _cudnn.ConvolutionBackwardData(alpha, _descFilter,
                                                    _weights, _descDataOut, _dy, _descConv, _algoBwdData, _workspace,
                                                    beta, _descDataIn, _withBorderDx);

            _kernelCropBorder.RunSafe(_withBorderDx, _dx, _inChannels, _batch, _inWidth, _filterX/2);

            return _previousLayer.BackPropagation(_dx);
        }

        public override void UpdateWeights(float learningRate)
        {
            _d_weights.MulC(-learningRate);
            _weights.Add(_d_weights);

            _d_bias.MulC(-learningRate);
            _bias.Add(_d_bias);

            if (_activation == Activation.PRelu)
            {
                _dARelu.MulC(-learningRate);
                _aRelu.Add(_dARelu);
            }

            base.UpdateWeights(learningRate);
        }

        public override void SaveValues(Stream stream)
        {
            BinaryWriter bw = new BinaryWriter(stream);
            float[] w = _weights;
            float[] b = _bias;


            for (int i = 0; i < _weights.Size; i++)
            {
                bw.Write(w[i]);
            }
            for (int i = 0; i < _bias.Size; i++)
            {
                bw.Write(b[i]);
            }
            if (_activation == Activation.PRelu || _activation == Activation.LeakyRelu)
            {
                float[] a = _aRelu;
                for (int i = 0; i < _aRelu.Size; i++)
                {
                    bw.Write(a[i]);
                }
            }
            bw.Flush();
            base.SaveValues(stream);
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
