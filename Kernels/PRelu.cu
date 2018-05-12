#include "cuda_runtime.h"
#include "device_launch_parameters.h"


extern "C"
__global__ void PRelu_Forward(const float* __restrict__ dataIn, const float* __restrict__ a, float* dataOut, int dataCount, int channelCount, int batchCount)
{
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int channel = blockIdx.y * blockDim.y + threadIdx.y;
	int batch = blockIdx.z * blockDim.z + threadIdx.z;

	if (x >= dataCount || channel >= channelCount || batch >= batchCount)
		return;

	int i = batch * channelCount * dataCount + channel * dataCount + x;
	float val = dataIn[i];
	float aVal = a[channel];

	val = fmaxf(0.0f, val) + aVal * fminf(0.0f, val);

	dataOut[i] = val;
}

extern "C"
__global__ void PRelu_Backward(const float* __restrict__ dataIn, const float* __restrict__ dY, const float* __restrict__ a, float* dData, float* dA, int dataCount, int channelCount, int batchCount)
{
	int channel = blockIdx.x * blockDim.x + threadIdx.x;

	if (channel >= channelCount)
		return;

	float aVal = a[channel];
	float d_a = 0;

	for (int batch = 0; batch < batchCount; batch++)
	{
		for (int x = 0; x < dataCount; x++)
		{

			int i = batch * channelCount * dataCount + channel * dataCount + x;
			float val = dataIn[i];

			dData[i] = dY[i] * ((val > 0) + aVal * (val <= 0));
			d_a += (val <= 0) * val * dY[i];
		}
	}

	dA[channel] = d_a / batchCount;
}

extern "C"
__global__ void PRelu_Backward1(const float* __restrict__ dataIn, const float* __restrict__ dY, const float* __restrict__ a, float* dData, float* dA, float* temp, int dataCount, int channelCount, int batchCount)
{
	int channel = blockIdx.x * blockDim.x + threadIdx.x;
	int batch = blockIdx.y * blockDim.y + threadIdx.y;

	if (channel >= channelCount)
		return;
	if (batch >= batchCount)
		return;

	float aVal = a[channel];
	float d_a = 0;

	//for (int batch = 0; batch < batchCount; batch++)
	{
		for (int x = 0; x < dataCount; x++)
		{

			int i = batch * channelCount * dataCount + channel * dataCount + x;
			float val = dataIn[i];

			dData[i] = dY[i] * ((val > 0) + aVal * (val <= 0));
			d_a += (val <= 0) * val * dY[i];
		}
	}
	temp[channel * batchCount + batch] = d_a;
	//dA[channel] = d_a / batchCount;
}

extern "C"
__global__ void PRelu_Backward2(const float* __restrict__ dataIn, const float* __restrict__ dY, const float* __restrict__ a, float* dData, float* dA, const float* __restrict__ temp, int dataCount, int channelCount, int batchCount)
{
	int channel = blockIdx.x * blockDim.x + threadIdx.x;

	if (channel >= channelCount)
		return;

	//float aVal = a[channel];
	float d_a = 0;

	for (int batch = 0; batch < batchCount; batch++)
	{
		/*for (int x = 0; x < dataCount; x++)
		{

			int i = batch * channelCount * dataCount + channel * dataCount + x;
			float val = dataIn[i];

			dData[i] = dY[i] * ((val > 0) + aVal * (val <= 0));
			d_a += (val <= 0) * val * dY[i];
		}*/

		d_a += temp[channel * batchCount + batch];
	}

	dA[channel] = d_a / batchCount;
}
