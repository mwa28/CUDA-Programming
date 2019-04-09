#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "device_functions.h"
#include<math.h>
#include <stdio.h>
#include<time.h>
#include <iostream>
#include <stdlib.h>
#include "GpuTimer.h"
using namespace std;

#define BLOCK_SIZE 256
#define SegLength 1024*10



//CUDA Kernel Device code
//Computes the element-wise vector addition of A and B into C: C[i] = A[i] + B[i].
//The 3 vectors have the same number of elements numElements.
__global__ void vectorAdd( float *A,  float *B, float *C, int numElements)
	
{
	//@@ Insert  your code here to implement vector addition where each thread performs one addition.
	int index = threadIdx.x + blockDim.x*blockIdx.x;
	if (index < numElements)
		C[index] = A[index] + B[index];
	
}

/**
* Host main routine
*/
int main(void)
{
	// Error code to check return values for CUDA calls
	cudaError_t err = cudaSuccess;

	// Print the vector length to be used, and compute its size
	float EPS = 0.0001;
	int numElements = 4096000;
	size_t size = numElements * sizeof(float);
	printf("[Vector addition of %d elements]\n", numElements);

	//Implement Vector Addition without using CUDA Streams
	// Allocate the host input vector A
	float *h_A = (float *)malloc(size);

	// Allocate the host input vector B
	float *h_B = (float *)malloc(size);

	// Allocate the host output vector C
	float *h_C = (float *)malloc(size);

	// Verify that allocations succeeded
	if (h_A == NULL || h_B == NULL || h_C == NULL)
	{
		fprintf(stderr, "Failed to allocate host vectors!\n");
		exit(EXIT_FAILURE);
	}

	// Initialize the host input vectors
	for (int i = 0; i < numElements; ++i)
	{
		h_A[i] = float(i);
		h_B[i] = 1/(i+EPS);
	}


	GpuTimer timer;
	timer.Start();
	// Allocate the device input vector A
	float *d_A = NULL;
	err = cudaMalloc((void **)&d_A, size);

	if (err != cudaSuccess)
	{
		fprintf(stderr, "Failed to allocate device vector A (error code %s)!\n", cudaGetErrorString(err));
		exit(EXIT_FAILURE);
	}

	// Allocate the device input vector B
	float *d_B = NULL;
	err = cudaMalloc((void **)&d_B, size);

	if (err != cudaSuccess)
	{
		fprintf(stderr, "Failed to allocate device vector B (error code %s)!\n", cudaGetErrorString(err));
		exit(EXIT_FAILURE);
	}

	// Allocate the device output vector C
	float *d_C = NULL;
	err = cudaMalloc((void **)&d_C, size);

	if (err != cudaSuccess)
	{
		fprintf(stderr, "Failed to allocate device vector C (error code %s)!\n", cudaGetErrorString(err));
		exit(EXIT_FAILURE);
	}

	// Copy the host input vectors A and B in host memory to the device input vectors in
	// device memory
	printf("Copy input data from the host memory to the CUDA device\n");
	err = cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);

	if (err != cudaSuccess)
	{
		fprintf(stderr, "Failed to copy vector A from host to device (error code %s)!\n", cudaGetErrorString(err));
		exit(EXIT_FAILURE);
	}

	err = cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice);

	if (err != cudaSuccess)
	{
		fprintf(stderr, "Failed to copy vector B from host to device (error code %s)!\n", cudaGetErrorString(err));
		exit(EXIT_FAILURE);
	}

	// Launch the Vector Add CUDA Kernel
	int threadsPerBlock = 256;
	int blocksPerGrid = (numElements + threadsPerBlock - 1) / threadsPerBlock;
	printf("CUDA kernel launch with %d blocks of %d threads\n", blocksPerGrid, threadsPerBlock);
	vectorAdd << <blocksPerGrid, threadsPerBlock >> >(d_A, d_B, d_C, numElements);
	err = cudaGetLastError();

	if (err != cudaSuccess)
	{
		fprintf(stderr, "Failed to launch vectorAdd kernel (error code %s)!\n", cudaGetErrorString(err));
		exit(EXIT_FAILURE);
	}

	// Copy the device result vector in device memory to the host result vector
	// in host memory.
	printf("Copy output data from the CUDA device to the host memory\n");
	err = cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost);

	if (err != cudaSuccess)
	{
		fprintf(stderr, "Failed to copy vector C from device to host (error code %s)!\n", cudaGetErrorString(err));
		exit(EXIT_FAILURE);
	}

	
	timer.Stop();
	printf("Time for vector addition implemenation without using CUDA streams: %f msecs.\n", timer.Elapsed());


	// Verify that the result vector is correct
	for (int i = 0; i < numElements; ++i)
	{
		if (fabs((h_A[i] + h_B[i]) - h_C[i]) > 1e-5)
		{
			fprintf(stderr, "Result verification failed at element %d!\n", i);
			exit(EXIT_FAILURE);
		}
	}

	printf("Test PASSED\n");

	// Free device global memory
	cudaFree(d_A);
	cudaFree(d_B);
	cudaFree(d_C);

	cudaDeviceProp prop;
	int dev_count;

	cudaGetDeviceCount(&dev_count);
	for (int i = 0; i < dev_count; i++)
	{
		cudaGetDeviceProperties(&prop, i);
		if (!prop.deviceOverlap)
		{
			printf("Unable to handle overlap. Exiting...\n");
			return 0;
		}
	}
	cudaStream_t stream1, stream2, stream3, stream4;

	cudaStreamCreate(&stream1);
	cudaStreamCreate(&stream2);
	cudaStreamCreate(&stream3);
	cudaStreamCreate(&stream4);

	//Implement Vector Addition Using CUDA Streams
	
	GpuTimer timer1;
	timer1.Start();
	
	//@@ Insert your code here to implement Vector Addition using streams and Time your implementation.Use the already allocated and initialized host arrays
	//h_A, h_B, and h-C;
	cudaHostAlloc((void **)&h_A, size, cudaHostAllocDefault);
	cudaHostAlloc((void **)&h_B, size, cudaHostAllocDefault);
	cudaHostAlloc((void **)&h_C, size, cudaHostAllocDefault);

	float *d_A0 = NULL; float *d_A1 = NULL; float *d_A2 = NULL; float *d_A3 = NULL;
	cudaMalloc((void **)&d_A0, size); cudaMalloc((void **)&d_A1, size); cudaMalloc((void **)&d_A2, size); cudaMalloc((void **)&d_A3, size);

	float *d_B0 = NULL; float *d_B1 = NULL; float *d_B2 = NULL; float *d_B3 = NULL; 
	cudaMalloc((void **)&d_B0, size); cudaMalloc((void **)&d_B1, size); cudaMalloc((void **)&d_B2, size); cudaMalloc((void **)&d_B3, size); 

	float *d_C0 = NULL; float *d_C1 = NULL; float *d_C2 = NULL; float *d_C3 = NULL;
	cudaMalloc((void **)&d_C0, size); cudaMalloc((void **)&d_C1, size); cudaMalloc((void **)&d_C2, size); cudaMalloc((void **)&d_C3, size); 

	for (int i = 0; i < numElements; i += SegLength * 4) {
		cudaMemcpyAsync(d_A0, h_A + i, SegLength * sizeof(float), cudaMemcpyHostToDevice,stream1);
		cudaMemcpyAsync(d_B0, h_B + i, SegLength * sizeof(float), cudaMemcpyHostToDevice, stream1);
		
		cudaMemcpyAsync(d_A1, h_A + i + SegLength, SegLength * sizeof(float), cudaMemcpyHostToDevice, stream2);
		cudaMemcpyAsync(d_B1, h_B + i + SegLength, SegLength * sizeof(float), cudaMemcpyHostToDevice, stream2);

		cudaMemcpyAsync(d_A2, h_A + i + 2 * SegLength, SegLength * sizeof(float), cudaMemcpyHostToDevice, stream3);
		cudaMemcpyAsync(d_B2, h_B + i + 2 * SegLength, SegLength * sizeof(float), cudaMemcpyHostToDevice, stream3);

		cudaMemcpyAsync(d_A3, h_A + i + 3 * SegLength, SegLength * sizeof(float), cudaMemcpyHostToDevice, stream4);
		cudaMemcpyAsync(d_B3, h_B + i + 3 * SegLength, SegLength * sizeof(float), cudaMemcpyHostToDevice, stream4);

		vectorAdd <<< blocksPerGrid, threadsPerBlock, 0, stream1 >>> (d_A0, d_B0, d_C0, numElements);
		vectorAdd <<< blocksPerGrid, threadsPerBlock, 0, stream2 >>> (d_A1, d_B1, d_C1, numElements);
		vectorAdd <<< blocksPerGrid, threadsPerBlock, 0, stream3 >>> (d_A2, d_B2, d_C2, numElements);
		vectorAdd <<< blocksPerGrid, threadsPerBlock, 0, stream4 >>> (d_A3, d_B3, d_C3, numElements);

		cudaMemcpyAsync(h_C + i, d_C0, SegLength * sizeof(float), cudaMemcpyDeviceToHost, stream1);
		cudaMemcpyAsync(h_C + i + SegLength, d_C1, SegLength * sizeof(float), cudaMemcpyDeviceToHost, stream2);
		cudaMemcpyAsync(h_C + i + 2 * SegLength, d_C2, SegLength * sizeof(float), cudaMemcpyDeviceToHost, stream3);
		cudaMemcpyAsync(h_C + i + 3 * SegLength, d_C3, SegLength * sizeof(float), cudaMemcpyDeviceToHost, stream4);

	}
	cudaStreamSynchronize(stream1);
	cudaStreamSynchronize(stream2);
	cudaStreamSynchronize(stream3);
	cudaStreamSynchronize(stream4);
	
	timer1.Stop();
	printf("Time for vector addition implementation using CUDA streams: %f msecs.\n", timer1.Elapsed());

	//@@Insert your code to free device memory and streams
	cudaFreeHost(h_A);
	cudaFreeHost(h_B);
	cudaFreeHost(h_C);
	cudaFree(d_A0); cudaFree(d_A1); cudaFree(d_A2); cudaFree(d_A3); 
	cudaFree(d_B0); cudaFree(d_B1); cudaFree(d_B2); cudaFree(d_B3);
	cudaFree(d_C0); cudaFree(d_C1); cudaFree(d_C2); cudaFree(d_C3); 
	cudaStreamDestroy(stream1);
	cudaStreamDestroy(stream2);
	cudaStreamDestroy(stream3);
	cudaStreamDestroy(stream4);

	printf("Done\n");

	return 0;
}

