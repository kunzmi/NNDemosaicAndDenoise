#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "texture_fetch_functions.h"

extern "C"
__global__ void convertCamToXYZKernel(const int width, const int height, float3 *inOutImage, int strideOut, float3 xyzX, float3 xyzY, float3 xyzZ)
{
	// integer pixel coordinates	
	const int x = blockIdx.x * blockDim.x + threadIdx.x;
	const int y = blockIdx.y * blockDim.y + threadIdx.y;

	if (x >= width || x < 0) return;
	if (y >= height || y < 0) return;

	float3* line = (float3*)(((char*)inOutImage) + y * strideOut);
	float3 pixelIn = line[x];
	float3 pixelOut;

	pixelOut.x = pixelIn.x * xyzX.x + pixelIn.y * xyzX.y + pixelIn.z * xyzX.z;
	pixelOut.y = pixelIn.x * xyzY.x + pixelIn.y * xyzY.y + pixelIn.z * xyzY.z;
	pixelOut.z = pixelIn.x * xyzZ.x + pixelIn.y * xyzZ.y + pixelIn.z * xyzZ.z;

	line[x] = pixelOut;
}

extern "C"
__global__ void HighlightRecovery(const int width, const int height, float3 *inOutImage, int strideOut, float3 WhiteBalanceFactors, float maxMultiplier)
{
	// integer pixel coordinates	
	const int x = blockIdx.x * blockDim.x + threadIdx.x;
	const int y = blockIdx.y * blockDim.y + threadIdx.y;

	if (x >= width || x < 0) return;
	if (y >= height || y < 0) return;

	float3* line = (float3*)(((char*)inOutImage) + y * strideOut);
	float3 pixelIn = line[x];

	float3 weightedPixel;
	weightedPixel.x = pixelIn.x * WhiteBalanceFactors.x * maxMultiplier;
	weightedPixel.y = pixelIn.y * WhiteBalanceFactors.y * maxMultiplier;
	weightedPixel.z = pixelIn.z * WhiteBalanceFactors.z * maxMultiplier;

	float3 pixelOut;

	pixelOut.x = pixelIn.x;
	pixelOut.y = pixelIn.y;
	pixelOut.z = pixelIn.z;

	if ((weightedPixel.x > 1.0f || weightedPixel.z > 1.0f) && weightedPixel.y >= 0.999f)
	{
		pixelOut.y = (weightedPixel.x + weightedPixel.z) / 2.0f / maxMultiplier;
	}
	line[x] = pixelOut;
}


extern "C"
__global__ void toneCurveSimple(const int width, const int height, float3 *inOutImage, int strideOut, cudaTextureObject_t tex_toneCurve)
{
	// integer pixel coordinates	
	const int x = blockIdx.x * blockDim.x + threadIdx.x;
	const int y = blockIdx.y * blockDim.y + threadIdx.y;

	if (x >= width || x < 0) return;
	if (y >= height || y < 0) return;

	float3* line = (float3*)(((char*)inOutImage) + y * strideOut);
	float3 pixelIn = line[x];
	float3 pixelOut;

	pixelIn.x = fminf(fmaxf(pixelIn.x, 0.0f), 1.0f);
	pixelIn.y = fminf(fmaxf(pixelIn.y, 0.0f), 1.0f);
	pixelIn.z = fminf(fmaxf(pixelIn.z, 0.0f), 1.0f);

	pixelOut.x = tex1D<float>(tex_toneCurve, pixelIn.x);
	pixelOut.y = tex1D<float>(tex_toneCurve, pixelIn.y);
	pixelOut.z = tex1D<float>(tex_toneCurve, pixelIn.z);
	/*pixelOut.x = toneCurve[(int)(8191 * pixelIn.x)];
	pixelOut.y = toneCurve[(int)(8191 * pixelIn.y)];
	pixelOut.z = toneCurve[(int)(8191 * pixelIn.z)];*/

	line[x] = pixelOut;
}


extern "C"
__global__ void convertRGBTosRGBKernel(const int width, const int height, float3 *inOutImage, int strideOut)
{
	// integer pixel coordinates	
	const int x = blockIdx.x * blockDim.x + threadIdx.x;
	const int y = blockIdx.y * blockDim.y + threadIdx.y;

	if (x >= width || x < 0) return;
	if (y >= height || y < 0) return;

	float3* line = (float3*)(((char*)inOutImage) + y * strideOut);
	float3 pixelIn = line[x];
	float3 pixelOut;
	//Swap Red and blue to obtain BGR images!
	pixelOut.z = pixelIn.x;
	pixelOut.y = pixelIn.y;
	pixelOut.x = pixelIn.z;

	//apply gamma:
	if (pixelOut.x <= 0.0031308f)
	{
		pixelOut.x = 12.92f * pixelOut.x;
	}
	else
	{
		pixelOut.x = (1.0f + 0.055f) * powf(pixelOut.x, 1.0f / 2.4f) - 0.055f;
	}

	if (pixelOut.y <= 0.0031308f)
	{
		pixelOut.y = 12.92f * pixelOut.y;
	}
	else
	{
		pixelOut.y = (1.0f + 0.055f) * powf(pixelOut.y, 1.0f / 2.4f) - 0.055f;
	}

	if (pixelOut.z <= 0.0031308f)
	{
		pixelOut.z = 12.92f * pixelOut.z;
	}
	else
	{
		pixelOut.z = (1.0f + 0.055f) * powf(pixelOut.z, 1.0f / 2.4f) - 0.055f;
	}

	pixelOut.x = fminf(fmaxf(pixelOut.x, 0.0f), 1.0f);
	pixelOut.y = fminf(fmaxf(pixelOut.y, 0.0f), 1.0f);
	pixelOut.z = fminf(fmaxf(pixelOut.z, 0.0f), 1.0f);

	pixelOut.x *= 255;
	pixelOut.y *= 255;
	pixelOut.z *= 255;
	//Save as BGR image for Bitmap-compatibilty
	line[x] = pixelOut;
}
