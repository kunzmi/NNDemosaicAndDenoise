using System;
using System.Collections.Generic;
using System.IO;
using System.Text;
using System.Threading.Tasks;
using ManagedCuda;

namespace NeuralNetworkTraining
{
    public abstract class Layer
    {
        protected int _inWidth;
        protected int _inHeight;
        protected int _inChannels;
        protected int _outWidth;
        protected int _outHeight;
        protected int _outChannels;
        protected int _batch;
        protected Layer _previousLayer;
        protected Layer _nextLayer;

        public Layer(int inWidth, int inHeight, int inChannels, int outWidth, int outHeight, int outChannels, int batch)
        {
            _inWidth = inWidth;
            _inHeight = inHeight;
            _inChannels = inChannels;

            _outWidth = outWidth;
            _outHeight = outHeight;
            _outChannels = outChannels;

            _batch = batch;
        }

        public int InChannels
        {
            get { return _inChannels; }
        }

        public int OutChannels
        {
            get { return _outChannels; }
        }

        public int InWidth
        {
            get { return _inWidth; }
        }

        public int InHeight
        {
            get { return _inHeight; }
        }

        public int OutWidth
        {
            get { return _outWidth; }
        }

        public int OutHeight
        {
            get { return _outHeight; }
        }

        public int Batch
        {
            get { return _batch; }
        }

        protected bool CanConnectToOutput(Layer next)
        {
            bool ret = true;
            ret &= this.OutChannels == next.InChannels;
            ret &= this.OutWidth == next.InWidth;
            ret &= this.OutHeight == next.InHeight;
            ret &= this.Batch == next.Batch;
            return ret;
        }

        public virtual void ConnectFollowingLayer(Layer layer)
        {
            if (!CanConnectToOutput(layer))
            {
                throw new ArgumentException("Output dimensions of this layer don't fit to input dimensions of the next layer.");
            }
            this._nextLayer = layer;
            layer._previousLayer = this;
        }

        public virtual float InferenceTraining(CudaDeviceVariable<float> input)
        {
            return 0;
        }

        public virtual CudaDeviceVariable<float> BackPropagation(CudaDeviceVariable<float> groundTrouth)
        {
            return null;
        }

        public virtual void UpdateWeights(float learningRate)
        {
            if (_nextLayer != null)
                _nextLayer.UpdateWeights(learningRate);
        }

        public virtual void InitRandomWeight(Random rand)
        {
            if (_nextLayer != null)
                _nextLayer.InitRandomWeight(rand);
        }

        public virtual void SaveValues(Stream stream)
        {
            if (_nextLayer != null)
                _nextLayer.SaveValues(stream);
        }

        public virtual void RestoreValues(Stream stream)
        {
            if (_nextLayer != null)
                _nextLayer.RestoreValues(stream);
        }
    }
}
