#include "cuda_runtime.h"
#include "device_launch_parameters.h"



extern "C"
__global__ void addBorder(float* imgIn, float* imgOut, int channelCount, int batchCount, int patchSize, int borderSize)
{
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;
	int channelbatch = blockIdx.z * blockDim.z + threadIdx.z;

	if (x >= patchSize + 2 * borderSize || y >= patchSize + 2 * borderSize || channelbatch >= batchCount * channelCount)
		return;

	int idxNew = y * (patchSize + 2 * borderSize) + x + channelbatch * (patchSize + 2 * borderSize) * (patchSize + 2 * borderSize);
	int xOrig = x - borderSize;
	if (xOrig < 0)
		xOrig *= -1;
	if (xOrig >= patchSize)
		xOrig = 2 * patchSize - 2 - xOrig;
	int yOrig = y - borderSize;
	if (yOrig < 0)
		yOrig *= -1;
	if (yOrig >= patchSize)
		yOrig = 2 * patchSize - 2 - yOrig;

	int idxOrig = yOrig * (patchSize)+xOrig + channelbatch * (patchSize) * (patchSize);

	imgOut[idxNew] = imgIn[idxOrig];
}

extern "C"
__global__ void cropBorder(float* imgIn, float* imgOut, int channelCount, int batchCount, int patchSize, int borderSize)
{
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;
	int channelbatch = blockIdx.z * blockDim.z + threadIdx.z;

	if (x >= patchSize || y >= patchSize || channelbatch >= batchCount * channelCount)
		return;

	int idxNew = y * (patchSize)+x + channelbatch * (patchSize) * (patchSize);
	int xOrig = x + borderSize;
	int yOrig = y + borderSize;

	int idxOrig = yOrig * (patchSize + 2 * borderSize) + xOrig + channelbatch * (patchSize + 2 * borderSize) * (patchSize + 2 * borderSize);

	imgOut[idxNew] = imgIn[idxOrig];
}
