using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using ManagedCuda;
using ManagedCuda.BasicTypes;
using ManagedCuda.NPP.NPPsExtensions;
using KernelClasses;

namespace NeuralNetworkTraining
{
    public class FinalLayer : Layer
    {
        CudaDeviceVariable<float> _groundTrouthData;
        CudaDeviceVariable<float> _dx;
        CudaDeviceVariable<float> _temp;
        CudaDeviceVariable<float> _msssiml1;
        CudaDeviceVariable<float> _res;
        CudaDeviceVariable<byte> _buffer;
        CudaDeviceVariable<float> _summedError;
        CudaDeviceVariable<double> _mean;
        Norm _norm;
        MSSSIML1Kernel _kernelMSSSIML1;

        public enum Norm
        {
            L1,
            L2,
            MSSSIM,
            Mix
        }

        public FinalLayer(int width, int height, int channels, int batch, Norm norm, CudaContext ctx, CUmodule mod)
            : base(width, height, channels, 0, 0, 0, batch)
        {
            _groundTrouthData = new CudaDeviceVariable<float>(width * height * channels * batch);
            _dx = new CudaDeviceVariable<float>(width * height * channels * batch);
            _temp = new CudaDeviceVariable<float>(width * height * channels * batch);
            _res = new CudaDeviceVariable<float>(width * height * channels * batch);
            _buffer = new CudaDeviceVariable<byte>(_temp.SumGetBufferSize()*100);
            _summedError = new CudaDeviceVariable<float>(1);
            _mean = new CudaDeviceVariable<double>(1);
            _norm = norm;
            _kernelMSSSIML1 = new MSSSIML1Kernel(mod, ctx);

            if (_norm == Norm.Mix || _norm == Norm.MSSSIM)
            {
                _msssiml1 = new CudaDeviceVariable<float>(channels * batch);
            }
        }

        public void SetGroundTrouth(float[] data)
        {
            _groundTrouthData.CopyToDevice(data);
        }

        public CudaDeviceVariable<float> GetResult()
        {
            return _res;
        }

        public void SetGroundTrouth(CudaDeviceVariable<float> data)
        {
            _groundTrouthData.CopyToDevice(data);
        }

        public float GetError(Norm norm)
        {
            float error = 0;
            switch (norm)
            {
                case Norm.L1:
                    _groundTrouthData.Sub(_res, _temp);
                    _temp.Abs();
                    _temp.Sum(_summedError, _buffer);

                    error = _summedError;
                    error = error / _batch / _inChannels / _inWidth / _inHeight;
                    return error;
                case Norm.L2:
                    _groundTrouthData.Sub(_res, _temp);
                    _temp.Sqr();
                    _temp.Sum(_summedError, _buffer);

                    error = _summedError;
                    error = error / _batch / _inChannels / _inWidth / _inHeight;
                    return error;
                case Norm.MSSSIM:
                    return error;
                case Norm.Mix:
                    if (_msssiml1 == null)
                        _msssiml1 = new CudaDeviceVariable<float>(_inChannels * _batch);

                    if (_kernelMSSSIML1 == null)
                    {

                    }

                    _kernelMSSSIML1.RunSafe(_res, _groundTrouthData, _msssiml1, _dx, _inChannels, _batch, 0.84f);
                    _msssiml1.Sum(_summedError, _buffer);
                    error = _summedError;
                    return error;
                default:
                    return 0;
            }
        }

        public override float InferenceTraining(CudaDeviceVariable<float> input)
        {
            float error = 0;
            
            switch (_norm)
            {
                case Norm.L1:
                    //derivative of cost-function. Hier L1-Norm:

                    _res.CopyToDevice(input);
                    _groundTrouthData.Sub(input, _dx);
                    _dx.Threshold_GTVal(0, 1);
                    _dx.Threshold_LTVal(0, -1);
                    _dx.DivC(_batch * _inChannels * _inWidth * _inHeight);
                    
                    _groundTrouthData.Sub(input, _temp);
                    _temp.Abs();
                    _temp.Sum(_summedError, _buffer);

                    error = _summedError;
                    error = error / _batch / _inChannels / _inWidth / _inHeight;
                    break;
                case Norm.L2:
                    //derivative of cost-function. Hier L2-Norm:
                    _res.CopyToDevice(input);
                    _groundTrouthData.Sub(input, _dx);
                    _dx.DivC(_batch * _inChannels * _inWidth * _inHeight);

                    _groundTrouthData.Sub(input, _temp);
                    _temp.Sqr();
                    _temp.Sum(_summedError, _buffer);

                    error = _summedError;
                    error = error / _batch / _inChannels / _inWidth / _inHeight;
                    break;
                case Norm.MSSSIM:
                    _res.CopyToDevice(input);
                    _kernelMSSSIML1.RunSafe(input, _groundTrouthData, _msssiml1, _dx, _inChannels, _batch, 1.0f);
                    
                    _msssiml1.Sum(_summedError, _buffer);
                    error = _summedError;
                    error = error / _batch / _inChannels;
                    break;
                case Norm.Mix:
                    _res.CopyToDevice(input);
                    _kernelMSSSIML1.RunSafe(input, _groundTrouthData, _msssiml1, _dx, _inChannels, _batch, 0.84f);
                    _msssiml1.Sum(_summedError, _buffer);
                    error = _summedError;
                    break;
                default:
                    break;
            }
            
            
            return error;
        }

        public override CudaDeviceVariable<float> BackPropagation(CudaDeviceVariable<float> groundTrouth)
        {
            return _previousLayer.BackPropagation(_dx);
        }

        public override void ConnectFollowingLayer(Layer layer)
        {
            throw new NotImplementedException("You can't connect a layer to the final layer!");
        }
    }
}
