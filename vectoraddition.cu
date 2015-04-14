#include <stdio.h>
#include <cuda.h>

// size of array
#define N 4096

//vector addition kernel
__global__ void vectorAddKernel(int *a, int *b, int *c)
{
	int tdx = blockIdx.x * blockDim.x + threadIdx.x;
	if(tdx < N)
	{
		c[tdx] = a[tdx] + b[tdx];
	}

}

int main(void)
{
	cudaSetDevice(3);
	// grid and block sizes
	dim3 grid(8,1,1);
	dim3 block(512,1,1);
	// host arrays
	int a_h[N];
	int b_h[N];
	int c_h[N];
	// device memory pointers
	int *a_d;
	int *b_d;
	int *c_d;
	// load arrays with some numbers
	for(int i=0; i<N; i++)
	{
		a_h[i] = i;
		b_h[i] = i*1;
	}
	//allocate device memory
	cudaMalloc((void**)&a_d, N*sizeof(int));
	cudaMalloc((void**)&b_d, N*sizeof(int));
	cudaMalloc((void**)&c_d, N*sizeof(int));

	//copy the host arrays to device
	cudaMemcpy(a_d, a_h, N*sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(b_d, b_h, N*sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(c_d, c_h, N*sizeof(int), cudaMemcpyHostToDevice);

	//CUDA events to measure time
	cudaEvent_t start;
	cudaEvent_t stop;
	float elapsedTime;

	//start timer
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	cudaEventRecord(start,0);

	//launch kernel
	vectorAddKernel<<<grid,block>>>(a_d, b_d, c_d);

	//stop timer
	cudaEventRecord(stop,0);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&elapsedTime, start, stop);

	//copy the result to host
	cudaMemcpy(c_h, c_d, N * sizeof(int), cudaMemcpyDeviceToHost);

	//print the results
	for(int i =0; i < N; i++)
	{
		printf("%i+%i = %i\n", a_h[i], b_h[i], c_h[i]);
	}

	//print out execution time
	printf("Time to calculate results: %f ms.\n", elapsedTime);

	//clean up
	cudaFree(a_h);
	cudaFree(b_h);
	cudaFree(c_h);

	cudaEventDestroy(start);
	cudaEventDestroy(stop);

	return 0;
}
