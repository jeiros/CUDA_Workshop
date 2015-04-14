#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include <omp.h>
#include <cuda.h>
#define TILE_WIDTH 16
void matrixMultiplyCPU(float *a_h, float *b_h, float *c_h, int width)
{
	float result;

	for(int row=0; row<width; row++)
	{
		for(int col=0; col<width; col++)
		{
			result = 0;

			for(int k=0; k<width; k++)
			{
				result += a_h[row * width + k] * b_h[k * width + col];
			}
			c_h[row * width + col] = result;
		}
	}
}
__global__ void matrixMyself(float *a, float *b, float *c, int width)
{
	int tx = threadIdx.x;
        int ty = threadIdx.y;
	int row = blockIdx.y * blockDim.y + threadIdx.y;
        int col = blockIdx.x * blockDim.x + threadIdx.x;

	extern __shared__ float s_a[];
	extern __shared__ float s_b[];

	for(int i=0; i<width/TILE_WIDTH; i++)
	{
		s_a[(row%TILE_WIDTH) * width + i * tx] = a[row*width + i * tx];
		s_b[(col%TILE_WIDTH) * width + i * ty] = b[col + i * ty * width];
		__syncthreads();
	}
	float result = 0;
	for(int i=0; i< width; i++)
	{
		result += s_a[row*width + i] * s_b[col*width+i];
	}
	c[row*width+col] = result;
}

__global__ void matrixMultiplyOptimised(float *a, float *b, float *c, int width)
{
	int tx = threadIdx.x;
        int ty = threadIdx.y;

	__shared__ float s_a[TILE_WIDTH][TILE_WIDTH];
	__shared__ float s_b[TILE_WIDTH][TILE_WIDTH];

	int row = blockIdx.y * blockDim.y + threadIdx.y;
	int col = blockIdx.x * blockDim.x + threadIdx.x;

	float result = 0;

	for (int p=0; p< width/TILE_WIDTH; p++)
	{
		s_a[ty][tx] = a[row*width + (p*TILE_WIDTH + tx)];
		s_b[ty][tx] = b[(p*TILE_WIDTH +ty)* width + col];
		__syncthreads();

		for(int i=0; i<TILE_WIDTH; i++)
		{
			result += s_a[ty][i] * s_b[i][tx];
		}
		__syncthreads();
	}
	c[row*width+col] = result;
}
__global__ void matrixMultiplySimple(float *a_d, float *b_d, float *c_d, int width)
{
	float result;

	int row = blockIdx.y * blockDim.y + threadIdx.y;
	int col = blockIdx.x * blockDim.x + threadIdx.x;
	if (row < width && col < width)
	{
		for(int i=0; i<width; i++)
		{
			result += a_d[row*width+i] * b_d[i*width+col];
		}

		c_d[row*width+col] = result;
	}
}

int main()
{
	cudaSetDevice(3);
	//define size of matrix, allocate memory and fill matrices

	int N = 2048;
	float *a_h = NULL;
	float *b_h = NULL;
	float *c_h = NULL;
	float *d_h = NULL;
        float *a_d = NULL;
        float *b_d = NULL;
        float *c_d = NULL;

	dim3 block(TILE_WIDTH, TILE_WIDTH);
	dim3 grid(N/block.x, N/block.y);
	size_t size = (N * N) * sizeof(float);
	a_h = (float *) malloc(size);
        b_h = (float *) malloc(size);
        c_h = (float *) malloc(size);
        d_h = (float *) malloc(size);

	for(int i=0; i<N; i++)
	{
		for(int j=0; j<N; j++)
        	{
                	a_h[i * N +j] = static_cast<float>(i);
                        b_h[i * N +j] = static_cast<float>(i);
        	}
	}
       	cudaMalloc((void**)&a_d, (N * N) * sizeof(float));
        cudaMalloc((void**)&b_d, (N * N) * sizeof(float));
	cudaMalloc((void**)&c_d, (N * N) * sizeof(float));

	//copy data to GPU

	cudaMemcpy(a_d, a_h,  (N * N) * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(b_d, b_h,  (N * N) * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(c_d, c_h,  (N * N) * sizeof(float), cudaMemcpyHostToDevice);

	//timing and kernel or CPU launch
	cudaEvent_t start;
	cudaEvent_t stop;

	float elapsedTime;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	cudaEventRecord(start, 0);

	matrixMultiplySimple<<<grid,block>>>(a_d, b_d, c_d, N);

	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&elapsedTime, start, stop);

	printf("Time to calculate results on GPU %f ms. \n", elapsedTime);

        cudaEventRecord(start, 0);

        matrixMultiplyCPU(a_h, b_h, c_h, N);

        cudaEventRecord(stop, 0);
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&elapsedTime, start, stop);

        printf("Time to calculate results on CPU %f ms. \n", elapsedTime);

        cudaEventRecord(start, 0);

        matrixMultiplyOptimised<<<grid,block>>>(a_d, b_d, c_d, N);

        cudaEventRecord(stop, 0);
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&elapsedTime, start, stop);

        printf("Time to calculate results on improved GPU %f ms. \n", elapsedTime);

        cudaEventRecord(start, 0);

        matrixMyself<<<grid,block,N*TILE_WIDTH>>>(a_d, b_d, c_d, N);

        cudaEventRecord(stop, 0);
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&elapsedTime, start, stop);

        printf("Time to calculate results on my GPU %f ms. \n", elapsedTime);
	//copy results back and print

	cudaMemcpy(d_h, c_d, (N * N) * sizeof(float), cudaMemcpyDeviceToHost);

//	printf("CPU-result:");
//	for(int i=0; i<N; i++)
  //      {
    //            for(int j=0; j<N; j++)
      //          {
        //                printf("%f \t",c_h[i * N +j]);
	//		if (j== (N-1)) {printf("\n");};
          //      }
       // }

//        printf("GPU-result:");
  //      for(int i=0; i<N; i++)
    //    {
//                for(int j=0; j<N; j++)
  //              {
    //                    printf("%f \t",d_h[i * N +j]); 
      //                  if (j==(N-1)) {printf("\n");};
        //        }
       // }

	//clean up
	free(a_h);
	free(b_h);
	free(c_h);
	free(d_h);
	cudaFree(a_d);
        cudaFree(b_d);
        cudaFree(c_d);
	cudaEventDestroy(start);
        cudaEventDestroy(stop);

	return 0;
}
