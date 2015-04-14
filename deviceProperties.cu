#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
int main()
{
	int deviceCount;
	cudaGetDeviceCount(&deviceCount);
	printf("\nNumber of GPU devices: %i\n", deviceCount);

	int driverVersion;
	int runtimeVersion;
	cudaDriverGetVersion(&driverVersion);
	cudaRuntimeGetVersion(&runtimeVersion);
	cudaDeviceProp deviceProperties;
	printf("CUDA Driver Version / Runtime Version: %d.%d / %d.%d\n\n\n",
	driverVersion/1000, (driverVersion%100)/10, runtimeVersion/1000,
	(runtimeVersion%100)/10);

	for(int i=0; i<deviceCount; i++)
	{
		cudaGetDeviceProperties(&deviceProperties, i);
		printf("Name: %s\n", deviceProperties.name);
	}
}
