#include <stdio.h>
#include <cuda.h>

// CPU code to do matrix ADdition
void matrixAdd(int *a, int *b, int *c, int N)
{
	int index;
	for(int col=0; col<N; col++)
	{
		for(int row=0; row<N; row++)
		{
			index = row * N + col;
			c[index] = a[index] + b[index];
		}
	}
}

// GPU code to do matrix addition
__global__ void matrixAddKernel(int *a, int *b, int *c, int N)
{
	int col = threadIdx.x + blockDim.x * blockIdx.x;
	int row = threadIdx.y + blockDim.y * blockIdx.y;

	int index = row * N + col;

	if(col < N && row < N)
	{
		c[index] = a[index] + b[index];
	}
}

int main(void)
{
	cudaSetDevice(3);
	//size of the matrix (the matrix will have NxN elements)
	int N = 2000;

	dim3 grid(16,1,1);
	dim3 block(1024,1,1);

	// pointers to host memory
	int *a_h;
	int *b_h;
	int *c_h;
	int *d_h;
	// pointers to device memory
	int *a_d;
	int *b_d;
	int *c_d;

	// this variable holds the number of bytes required by arrays
	int size;

	// use CUDA events to measure time
	cudaEvent_t start;
	cudaEvent_t stop;
	float elapsedTime;

	//print out the information about number of blocks and threads
	printf("Number of threads: %i (%ix%i)\n", block.x*block.y, block.x, block.y);
	printf("Number of blocks: %i (%ix%i)\n", grid.x*grid.y, grid.x, grid.y);

	//dynamically alocate host memory and load the arrays with some data
	size = N * N * sizeof(int);
	a_h = (int*) malloc(size);
	b_h = (int*) malloc(size);
	c_h = (int*) malloc(size);
	d_h = (int*) malloc(size);
	for(int i=0; i<N; i++)
	{
		for(int j=0; j<N; j++)
		{
			a_h[i * N + j] = i;
			b_h[i * N + j] = i;
		}
	}


	//allocate memory on the device
	cudaMalloc((void**)&a_d, size);
	cudaMalloc((void**)&b_d, size);
	cudaMalloc((void**)&c_d, size);

	//copy the host memory to the device
	cudaMemcpy(a_d, a_h, size, cudaMemcpyHostToDevice);
	cudaMemcpy(b_d, b_h, size, cudaMemcpyHostToDevice);
	cudaMemcpy(c_d, c_h, size, cudaMemcpyHostToDevice);

	//start the timers
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	cudaEventRecord(start, 0);

	//launch the kernel
	matrixAddKernel<<<grid,block>>>(a_d, b_d, c_d, N);

	//stop the timer and print out the execution time
	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&elapsedTime, start, stop);
	printf("Time to calculate results on GPU: %f ms.\n",
	elapsedTime);

	//copy the results to host
	cudaMemcpy(c_h, c_d, size ,cudaMemcpyDeviceToHost);

	//time to measure CPU performance
	cudaEventRecord(start,0);

	//launch the matrixAdd function that executes on the CPU
	matrixAdd(a_h, b_h, d_h, N);

	//strop the timer and print out the results

	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&elapsedTime, start, stop );
	printf("Time to calculate results on CPU: %f ms.\n",
	elapsedTime);

	//check that GPU and CPU results match
	for(int i=0; i<N*N; i++)
	{
		if (c_h[i] != d_h[i]) printf("Error: CPU and GPU results do not match\n");
			break;
	}

	// clean up
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
