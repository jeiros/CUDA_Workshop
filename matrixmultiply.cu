#include "cuda_runtime.h"
#include "device_launch_paramters.h"
#include <stdio.h>
#include <cuda.h>



void matrixMultiplyCPU(float *a, float *b, float *c, int width)
{
	float result;
	for(int col = 0; row < width; row++)
	{
		result = 0;
		for(int k = 0; k < width; k++)
		{
			result += a[row * width + k] * b[k * width + col];
		}
		c[row * width + col] = result;
	}
}