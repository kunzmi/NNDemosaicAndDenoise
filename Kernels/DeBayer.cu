#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <curand_kernel.h>



enum BayerColor : int
{
	Red = 0,
	Green = 1,
	Blue = 2,
	Cyan = 3,
	Magenta = 4,
	Yellow = 5,
	White = 6
};


extern "C"
__device__ __constant__ BayerColor c_cfaPattern[2][2];

extern "C"
__global__ void CreateBayer(float3* imgIn, float* imgOut, int dimX, int dimY, int strideIn, int strideOut)
{
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;

	if (x >= dimX || y >= dimY)
		return;

	float3* lineIn = (float3*)((char*)imgIn + y * strideIn);
	float* lineOut = (float*)((char*)imgOut + y * strideOut);
	float value = 0;

	BayerColor thisPixel = c_cfaPattern[y % 2][x % 2];

	if (thisPixel == Green)
	{
		value = lineIn[x].y;
	}
	else
		if (thisPixel == Red)
		{
			value = lineIn[x].x;
		}
		else
			if (thisPixel == Blue)
			{
				value = lineIn[x].z;
			}


	lineOut[x] = value;
}

#define RAW(xx, yy) (*(((float*)((char*)imgIn + (yy) * strideIn)) + (xx)))
#define RAWR(xx, yy) ((RAW(xx, yy) - blackPoint.x) * scale.x)
#define RAWG(xx, yy) ((RAW(xx, yy) - blackPoint.y) * scale.y)
#define RAWB(xx, yy) ((RAW(xx, yy) - blackPoint.z) * scale.z)

#define RED(xx, yy) ((*(((float3*)((char*)outImage + (yy) * strideOut)) + (xx))).x)
#define GREEN(xx, yy) ((*(((float3*)((char*)outImage + (yy) * strideOut)) + (xx))).y)
#define BLUE(xx, yy) ((*(((float3*)((char*)outImage + (yy) * strideOut)) + (xx))).z)


//Simple gradient and laplacian supported weighted interpolation of green channel. See e.g. Wu and Zhang
extern "C"
__global__ void deBayerGreenKernel(const int width, const int height, const float* __restrict__ imgIn, int strideIn, float3 *outImage, int strideOut, float3 blackPoint, float3 scale)
{
	// integer pixel coordinates	
	const int x = blockIdx.x * blockDim.x + threadIdx.x;
	const int y = blockIdx.y * blockDim.y + threadIdx.y;

	if (x >= width - 2 || x < 2) return;
	if (y >= height - 2 || y < 2) return;

	BayerColor thisPixel = c_cfaPattern[y % 2][x % 2];

	float g = 0;
	float p;
	float xMinus2;
	float xMinus1;
	float xPlus1;
	float xPlus2;

	float yMinus2;
	float yMinus1;
	float yPlus1;
	float yPlus2;
	float gradientX;
	float gradientY;

	float laplaceX;
	float laplaceY;

	float interpolX;
	float interpolY;

	float weight;

	switch (thisPixel)
	{
	case Green:
		g = RAWG(x, y);
		break;

	case Red:
		p = RAWR(x, y);
		xMinus2 = RAWR(x - 2, y);
		xMinus1 = RAWG(x - 1, y);
		xPlus1 = RAWG(x + 1, y);
		xPlus2 = RAWR(x + 2, y);

		yMinus2 = RAWR(x, y - 2);
		yMinus1 = RAWG(x, y - 1);
		yPlus1 = RAWG(x, y + 1);
		yPlus2 = RAWR(x, y + 2);

		gradientX = 0.5f * fabs(xPlus1 - xMinus1);
		gradientY = 0.5f * fabs(yPlus1 - yMinus1);

		laplaceX = 0.25f * fabs(2.0f * p - xMinus2 - xPlus2);
		laplaceY = 0.25f * fabs(2.0f * p - yMinus2 - yPlus2);

		interpolX = 0.125f * (-xMinus2 + 4.0f * xMinus1 + 2.0f * p + 4.0f * xPlus1 - xPlus2);
		interpolY = 0.125f * (-yMinus2 + 4.0f * yMinus1 + 2.0f * p + 4.0f * yPlus1 - yPlus2);

		weight = (gradientY + laplaceY) / (gradientX + gradientY + laplaceX + laplaceY + 0.000000001f);

		g = weight * interpolX + (1.0f - weight) * interpolY;

		break;
	case Blue:
		p = RAWB(x, y);
		xMinus2 = RAWB(x - 2, y);
		xMinus1 = RAWG(x - 1, y);
		xPlus1 = RAWG(x + 1, y);
		xPlus2 = RAWB(x + 2, y);

		yMinus2 = RAWB(x, y - 2);
		yMinus1 = RAWG(x, y - 1);
		yPlus1 = RAWG(x, y + 1);
		yPlus2 = RAWB(x, y + 2);

		gradientX = 0.5f * fabs(xPlus1 - xMinus1);
		gradientY = 0.5f * fabs(yPlus1 - yMinus1);

		laplaceX = 0.25f * fabs(2.0f * p - xMinus2 - xPlus2);
		laplaceY = 0.25f * fabs(2.0f * p - yMinus2 - yPlus2);

		interpolX = 0.125f * (-xMinus2 + 4.0f * xMinus1 + 2.0f * p + 4.0f * xPlus1 - xPlus2);
		interpolY = 0.125f * (-yMinus2 + 4.0f * yMinus1 + 2.0f * p + 4.0f * yPlus1 - yPlus2);

		weight = (gradientY + laplaceY) / (gradientX + gradientY + laplaceX + laplaceY + 0.000000001f);

		g = weight * interpolX + (1.0f - weight) * interpolY;


		break;
	}
	GREEN(x, y) = g;
}

//interpolate color difference to green channel
extern "C"
__global__ void deBayerRedBlueKernel(const int width, const int height, const float* __restrict__ imgIn, int strideIn, float3 *outImage, int strideOut, float3 blackPoint, float3 scale)
{
	// integer pixel coordinates	
	const int x = blockIdx.x * blockDim.x + threadIdx.x;
	const int y = blockIdx.y * blockDim.y + threadIdx.y;

	if (x >= width - 2 || x < 2) return;
	if (y >= height - 2 || y < 2) return;

	BayerColor thisPixel = c_cfaPattern[y % 2][x % 2];
	BayerColor thisRow = c_cfaPattern[y % 2][(x + 1) % 2];
	float r, b;
	float g = GREEN(x, y);

	switch (thisPixel)
	{
	case Green:
		if (thisRow == Red)
		{
			float xMinus1r = RAWR(x - 1, y);
			float xPlus1r = RAWR(x + 1, y);
			float xMinus1g = GREEN(x - 1, y);
			float xPlus1g = GREEN(x + 1, y);
			r = g + 0.5f * ((xMinus1r - xMinus1g) + (xPlus1r - xPlus1g));

			float yMinus1b = RAWB(x, y - 1);
			float yPlus1b = RAWB(x, y + 1);
			float yMinus1g = GREEN(x, y - 1);
			float yPlus1g = GREEN(x, y + 1);
			b = g + 0.5f * ((yMinus1b - yMinus1g) + (yPlus1b - yPlus1g));
		}
		else
		{
			float xMinus1b = RAWB(x - 1, y);
			float xPlus1b = RAWB(x + 1, y);
			float xMinus1g = GREEN(x - 1, y);
			float xPlus1g = GREEN(x + 1, y);
			b = g + 0.5f * ((xMinus1b - xMinus1g) + (xPlus1b - xPlus1g));

			float yMinus1r = RAWR(x, y - 1);
			float yPlus1r = RAWR(x, y + 1);
			float yMinus1g = GREEN(x, y - 1);
			float yPlus1g = GREEN(x, y + 1);
			r = g + 0.5f * ((yMinus1r - yMinus1g) + (yPlus1r - yPlus1g));
		}
		break;
	case Red:
		r = RAWR(x, y);
		{
			float mmB = RAWB(x - 1, y - 1);
			float pmB = RAWB(x + 1, y - 1);
			float ppB = RAWB(x + 1, y + 1);
			float mpB = RAWB(x - 1, y + 1);
			float mmG = GREEN(x - 1, y - 1);
			float pmG = GREEN(x + 1, y - 1);
			float ppG = GREEN(x + 1, y + 1);
			float mpG = GREEN(x - 1, y + 1);
			b = g + 0.25f * ((mmB - mmG) + (pmB - pmG) + (ppB - ppG) + (mpB - mpG));
		}
		break;
	case Blue:
		b = RAWB(x, y);
		{
			float mmR = RAWR(x - 1, y - 1);
			float pmR = RAWR(x + 1, y - 1);
			float ppR = RAWR(x + 1, y + 1);
			float mpR = RAWR(x - 1, y + 1);
			float mmG = GREEN(x - 1, y - 1);
			float pmG = GREEN(x + 1, y - 1);
			float ppG = GREEN(x + 1, y + 1);
			float mpG = GREEN(x - 1, y + 1);
			r = g + 0.25f * ((mmR - mmG) + (pmR - pmG) + (ppR - ppG) + (mpR - mpG));
		}
		break;

	}
	RED(x, y) = r;
	BLUE(x, y) = b;
}

#undef RAW
#undef RAWR
#undef RAWG
#undef RAWB

#undef RED
#undef GREEN
#undef BLUE

#define RAW2(x, y) ((float)dataIn[(y) * dimX * 2 + (x)])
extern "C"
__global__ void deBayersSubSample3(unsigned short* dataIn, float3* imgOut, float maxVal, int dimX, int dimY, int strideOut)
{
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;

	if (x >= dimX || y >= dimY)
		return;



	float3 pixel = make_float3(0, 0, 0);

	float3* lineOut = (float3*)((char*)imgOut + y * strideOut);
	float factor = 1.0f / maxVal;

	for (int ix = 0; ix < 2; ix++)
	{
		for (int iy = 0; iy < 2; iy++)
		{
			BayerColor thisPixel = c_cfaPattern[iy][ix];

			if (thisPixel == Green)
			{
				pixel.y += RAW2(2 * x + ix, 2 * y + iy) * factor * 0.5f; //we have two green pixels per bayer
			}
			else
				if (thisPixel == Red)
				{
					pixel.x = RAW2(2 * x + ix, 2 * y + iy) * factor;
				}
				else
					if (thisPixel == Blue)
					{
						pixel.z = RAW2(2 * x + ix, 2 * y + iy) * factor;
					}
		}
	}

	lineOut[x] = pixel;
}

extern "C"
__global__ void deBayersSubSample3DNG(unsigned short* dataIn, float3* imgOut, float maxValR, float maxValG, float maxValB, float minValR, float minValG, float minValB, int dimX, int dimY, int strideOut)
{
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;

	if (x >= dimX || y >= dimY)
		return;



	float3 pixel = make_float3(0, 0, 0);

	float3* lineOut = (float3*)((char*)imgOut + y * strideOut);
	//float factor = 1.0f / maxVal;

	for (int ix = 0; ix < 2; ix++)
	{
		for (int iy = 0; iy < 2; iy++)
		{
			BayerColor thisPixel = c_cfaPattern[iy][ix];

			if (thisPixel == Green)
			{
				pixel.y += (RAW2(2 * x + ix, 2 * y + iy) - minValG) / maxValG * 0.5f; //we have two green pixel per bayer
			}
			else
				if (thisPixel == Red)
				{
					pixel.x = (RAW2(2 * x + ix, 2 * y + iy) - minValR) / maxValR;
				}
				else
					if (thisPixel == Blue)
					{
						pixel.z = (RAW2(2 * x + ix, 2 * y + iy) - minValB) / maxValB;
					}
		}
	}

	lineOut[x] = pixel;
}

#undef RAW2

extern "C"
__global__
void setupCurand(curandState* states, int count)
{
	int x = blockIdx.x * blockDim.x + threadIdx.x;

	if (x >= count)
		return;

	curand_init(1234, x, 0, &states[x]);

}

extern "C"
__global__ void CreateBayerWithNoise(curandState* states, float3* imgIn, float* imgOut, int dimX, int dimY, int strideIn, int strideOut, float alpha, float beta)
{
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;

	if (x >= dimX || y >= dimY)
		return;

	float3* lineIn = (float3*)((char*)imgIn + y * strideIn);
	float* lineOut = (float*)((char*)imgOut + y * strideOut);
	float value = 0;

	BayerColor thisPixel = c_cfaPattern[y % 2][x % 2];

	if (thisPixel == Green)
	{
		value = lineIn[x].y;
	}
	else
		if (thisPixel == Red)
		{
			value = lineIn[x].x;
		}
		else
			if (thisPixel == Blue)
			{
				value = lineIn[x].z;
			}


	int idxState = y * dimX + x;

	curandState state = states[idxState];

	if (alpha == 0)
	{
		value += sqrtf(beta) * curand_normal(&state);
	}
	else
	{
		double chi = 1.0 / alpha;
		unsigned int poisson = curand_poisson(&state, chi * value);
		value = (float)(poisson / chi) + sqrtf(beta) * curand_normal(&state);
	}

	states[idxState] = state;
	value = fmaxf(0.0, fminf(1.0f, value));

	lineOut[x] = value;
}