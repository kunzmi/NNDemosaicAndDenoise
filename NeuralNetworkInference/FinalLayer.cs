using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using ManagedCuda;
using ManagedCuda.BasicTypes;
using ManagedCuda.NPP.NPPsExtensions;

namespace NeuralNetworkInference
{
    public class FinalLayer : Layer
    {
        CudaDeviceVariable<float> _res;

        public FinalLayer(int width, int height, int channels, int batch, CudaContext ctx, CUmodule mod)
            : base(width, height, channels, 0, 0, 0, batch)
        {

        }

        public CudaDeviceVariable<float> GetResult()
        {
            return _res;
        }

        public override float Inference(CudaDeviceVariable<float> input)
        {
            _res = input;
            return 0;
        }

        public override void ConnectFollowingLayer(Layer layer)
        {
            throw new NotImplementedException("You can't connect a layer to the final layer!");
        }
    }
}
