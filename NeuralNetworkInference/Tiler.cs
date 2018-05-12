using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using ManagedCuda.NPP;

namespace NeuralNetworkInference
{
    public static class Tiler
    {
        public struct RoiInputOutput
        {
            public NppiRect inputROI;
            public NppiRect outputROI;
            public NppiRect positionInFinalImage;
        }

        public static IEnumerable<RoiInputOutput> GetROIs(NppiRect inputROI, int tileSize, int borderSize)
        {
            List<RoiInputOutput> rois = new List<RoiInputOutput>();

            int tileStep = tileSize - 2 * borderSize;

            for (int y = inputROI.y - borderSize; y < inputROI.y + inputROI.height + borderSize; y += tileStep)
            {
                for (int x = inputROI.x - borderSize; x < inputROI.x + inputROI.width + borderSize; x += tileStep)
                {
                    RoiInputOutput roi = new RoiInputOutput();
                    if (x + tileSize < inputROI.width + borderSize)
                    {
                        roi.inputROI.x = x;
                        roi.inputROI.width = tileSize;

                        roi.outputROI.x = 0;
                        roi.outputROI.width = tileStep;

                        roi.positionInFinalImage.x = x;
                        roi.positionInFinalImage.width = tileStep;
                    }
                    else
                    {
                        roi.inputROI.x = inputROI.x + inputROI.width + borderSize - tileSize;
                        roi.inputROI.width = tileSize;

                        roi.outputROI.x = x - (inputROI.x + inputROI.width + borderSize - tileSize);
                        roi.outputROI.width = inputROI.width - x;

                        roi.positionInFinalImage.x = x;
                        roi.positionInFinalImage.width = inputROI.width - x;
                    }

                    if (y + tileSize < inputROI.height + borderSize)
                    {
                        roi.inputROI.y = y;
                        roi.inputROI.height = tileSize;

                        roi.outputROI.y = 0;
                        roi.outputROI.height = tileStep;

                        roi.positionInFinalImage.y = y;
                        roi.positionInFinalImage.height = tileStep;
                    }
                    else
                    {
                        roi.inputROI.y = inputROI.y + inputROI.height + borderSize - tileSize;
                        roi.inputROI.height = tileSize;

                        roi.outputROI.y = y - (inputROI.y + inputROI.height + borderSize - tileSize);
                        roi.outputROI.height = inputROI.height - y;

                        roi.positionInFinalImage.y = y;
                        roi.positionInFinalImage.height = inputROI.height - y;
                    }

                    rois.Add(roi);
                }
            }


            return rois;
        }

        static NppiRect GetOutputROI(NppiRect inputROI, int borderSize)
        {
            NppiRect output = new NppiRect();

            output.x = inputROI.x + borderSize;
            output.y = inputROI.y + borderSize;
            output.width = inputROI.width - 2 * borderSize;
            output.height = inputROI.height - 2 * borderSize;

            return output;
        }
    }
}
