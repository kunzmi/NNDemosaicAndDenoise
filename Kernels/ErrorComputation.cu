#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include <device_functions.h>
#include <math_functions.h>
#include <math_constants.h>



#define PATCH_DIM 31
#define PATCH_DIM_HALF 15
#define SIGN(a) (a >= 0 ? 1 : -1)

__device__ float getGauss(int x, int y, float sigma)
{
	x -= PATCH_DIM_HALF;
	y -= PATCH_DIM_HALF;
	x *= x;
	y *= y;
	float sigma2 = 2.0f * sigma * sigma;
	float norm = 1.0f / (CUDART_PI_F * sigma2);
	float gauss = norm * expf(-(x + y) / (sigma2));
	return gauss;
}

extern "C"
__global__ void MSSSIML1(float* imgPatch1, float* imgPatch2, float* msssiml1, float* d_mssiml1, int channelCount, int batchCount, float alpha)
{
	int batch = blockIdx.x * blockDim.x + threadIdx.x;
	int channel = blockIdx.y * blockDim.y + threadIdx.y;

	if (batch >= batchCount || channel >= channelCount)
		return;

	int idxSSIM = batch * 3 + channel;

	const float C1 = 0.01f * 0.01f;
	const float C2 = 0.03f * 0.03f;

	const float sigma[] = { 0.5f, 1.0f, 2.0f, 4.0f, 8.0f };

	float l; // only for last sigma
	float cs[5];
	float Pcs = 1;

	float l1 = 0;
	float mux[] = { 0, 0, 0, 0, 0 };
	float muy[] = { 0, 0, 0, 0, 0 };
	float sigmax2[] = { 0, 0, 0, 0, 0 };
	float sigmay2[] = { 0, 0, 0, 0, 0 };
	float sigmaxy[] = { 0, 0, 0, 0, 0 };
	float normGauss[] = { 0, 0, 0, 0, 0 };

	for (int i = 0; i < 5; i++)
	{
		for (int y = 0; y < PATCH_DIM; y++)
		{
			for (int x = 0; x < PATCH_DIM; x++)
			{
				float gauss = getGauss(x, y, sigma[i]);
				int idxPixel = idxSSIM * PATCH_DIM * PATCH_DIM + y * PATCH_DIM + x;
				float pix1 = imgPatch1[idxPixel] + 0.5f;
				float pix2 = imgPatch2[idxPixel] + 0.5f;
				mux[i] += gauss * pix1;
				muy[i] += gauss * pix2;
				normGauss[i] += gauss;
				if (i == 4)
				{
					l1 += fabs(gauss * (pix1 - pix2));
				}
			}
		}
		mux[i] /= normGauss[i];
		muy[i] /= normGauss[i];

		for (int y = 0; y < PATCH_DIM; y++)
		{
			for (int x = 0; x < PATCH_DIM; x++)
			{
				float gauss = getGauss(x, y, sigma[i]);
				int idxPixel = idxSSIM * PATCH_DIM * PATCH_DIM + y * PATCH_DIM + x;
				float pix1 = imgPatch1[idxPixel] + 0.5f;
				float pix2 = imgPatch2[idxPixel] + 0.5f;
				sigmax2[i] += gauss * pix1 * pix1;
				sigmay2[i] += gauss * pix2 * pix2;
				sigmaxy[i] += gauss * pix1 * pix2;
			}
		}
		sigmax2[i] /= normGauss[i];
		sigmay2[i] /= normGauss[i];
		sigmaxy[i] /= normGauss[i];

		sigmax2[i] -= mux[i] * mux[i];
		sigmay2[i] -= muy[i] * muy[i];
		sigmaxy[i] -= mux[i] * muy[i];

		l = (2 * mux[i] * muy[i] + C1) / (mux[i] * mux[i] + muy[i] * muy[i] + C1); //only keep last sigma value!
		cs[i] = (2 * sigmaxy[i] + C2) / (sigmax2[i] + sigmay2[i] + C2);

		Pcs *= cs[i];
	}
	l1 /= normGauss[4];
	float SSIM = 1 - (l * Pcs);
	msssiml1[idxSSIM] = (alpha * SSIM + (1.0f - alpha) * l1) / (float)(batchCount * channelCount);



	//Derivatives:
	float dlOhneW = 2 * (muy[4] - mux[4] * l) / (mux[4] * mux[4] + muy[4] * muy[4] + C1);


	for (int y = 0; y < PATCH_DIM; y++)
	{
		for (int x = 0; x < PATCH_DIM; x++)
		{
			float gauss4 = getGauss(x, y, sigma[4]) / normGauss[4];
			float dMSSSIM = dlOhneW * gauss4;
			int idxPixel = idxSSIM * PATCH_DIM * PATCH_DIM + y * PATCH_DIM + x;
			float pix1 = imgPatch1[idxPixel] + 0.5f;
			float pix2 = imgPatch2[idxPixel] + 0.5f;

			for (int i = 0; i < 5; i++)
			{
				float gauss = getGauss(x, y, sigma[i]) / normGauss[i];

				float dcs = 2 / (sigmax2[i] + sigmay2[i] + C2) * gauss * ((pix2 - muy[i]) - cs[i] * (pix1 - mux[i]));
				dMSSSIM += dcs / cs[i] * l;
			}
			dMSSSIM *= Pcs;

			float diff_L1 = SIGN(pix1 - pix2) * gauss4 / (float)(batchCount * channelCount);
			d_mssiml1[idxPixel] = (alpha * (-dMSSSIM / (float)(batchCount * channelCount)) + (1.0f - alpha) * diff_L1);
		}
	}

}
