#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#define PATCH4_DIM 66
#define PATCH_DIM 31
#define PATCH_DIM_HALF 15

//Copies data from an 2x2 RGB image to 1x1 planar image batch (subtracts potential normalization)
extern "C"
__global__ void prepareData(const float3* __restrict__ imgInA, const float3* __restrict__ imgInB, float* imgOutA, float* imgOutB, int batch, float sub, float3 whiteBalance)
{
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;
	int z = blockIdx.z * blockDim.z + threadIdx.z;

	if (x >= PATCH_DIM || y >= PATCH_DIM || z >= 4)
		return;

	float3 pixelA, pixelB;

	int inX = x + 1;
	if (z == 1 || z == 3)
		inX += PATCH_DIM;

	int inY = y + 1;
	if (z > 1)
		inY += PATCH_DIM;

	int idxIn = inY * PATCH4_DIM + inX;

	pixelA = imgInA[idxIn];
	pixelB = imgInB[idxIn];

	/*pixelA.x /= whiteBalance.x;
	pixelA.y /= whiteBalance.y;
	pixelA.z /= whiteBalance.z;*/

	pixelB.x /= whiteBalance.x;
	pixelB.y /= whiteBalance.y;
	pixelB.z /= whiteBalance.z;

	int idxOut = z * PATCH_DIM * PATCH_DIM * 3 + y * PATCH_DIM + x;
	idxOut += batch * 4 * PATCH_DIM * PATCH_DIM * 3;
	imgOutA[idxOut + 0] = pixelA.x - sub;
	imgOutB[idxOut + 0] = pixelB.x - sub;
	imgOutA[idxOut + PATCH_DIM * PATCH_DIM] = pixelA.y - sub;
	imgOutB[idxOut + PATCH_DIM * PATCH_DIM] = pixelB.y - sub;
	imgOutA[idxOut + PATCH_DIM * PATCH_DIM * 2] = pixelA.z - sub;
	imgOutB[idxOut + PATCH_DIM * PATCH_DIM * 2] = pixelB.z - sub;
}

//Copies one patch of a batch block to an image for visual inspection
extern "C"
__global__ void restoreImage(const float* __restrict__ imgIn, uchar3* imgOut, int pitchOut, int imgOffset, float facR, float facG, float facB, float add)
{
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;

	if (x >= PATCH_DIM || y >= PATCH_DIM)
		return;


	float r = (imgIn[imgOffset * PATCH_DIM * PATCH_DIM * 3 + y * PATCH_DIM + x] + add) * facR * 255.0f;
	float g = (imgIn[imgOffset * PATCH_DIM * PATCH_DIM * 3 + y * PATCH_DIM + x + PATCH_DIM * PATCH_DIM] + add) * facG  * 255.0f;
	float b = (imgIn[imgOffset * PATCH_DIM * PATCH_DIM * 3 + y * PATCH_DIM + x + PATCH_DIM * PATCH_DIM * 2] + add) * facB  * 255.0f;

	r = fmaxf(fminf(r, 255.0f), 0.0f);
	g = fmaxf(fminf(g, 255.0f), 0.0f);
	b = fmaxf(fminf(b, 255.0f), 0.0f);

	uchar3 pixel;
	pixel.x = (unsigned char)b;
	pixel.y = (unsigned char)g;
	pixel.z = (unsigned char)r;

	uchar3* line = (uchar3*)(((char*)imgOut) + y * pitchOut);
	line[x] = pixel;
}
