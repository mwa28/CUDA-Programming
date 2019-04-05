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

#define BLOCK_SIZE 512

// Given a list (lst) of length n
// Output its sum = lst[0] + lst[1] + ... + lst[n-1];


//GPU Kernel that implements reduction sum on input vector of length len, 
//the results of reduction done be the thread blocks is stored in a vector output
__global__ void total(float *input, float *output, int len) {

	__shared__ float partialSum[2 * BLOCK_SIZE];
	unsigned int tx = threadIdx.x;
	unsigned int bx = blockIdx.x;
	unsigned int start = 2 * bx*blockDim.x;
	//@@ Load a segment of the input vector into shared memory
	if (tx < len) {
		partialSum[tx] = input[start + tx];
		partialSum[blockDim.x + tx] = input[start + blockDim.x + tx];
		//@@ Traverse the reduction tree
		for (unsigned int stride = blockDim.x; stride > 0; stride /= 2) {
			__syncthreads();
			if (tx < stride)
				partialSum[tx] += partialSum[tx + stride];
		}
		//@@ Write the computed sum of the block to the output vector at the correct index
		output[tx] = partialSum[tx];
	}
}


// Serial implementation for running on CPU using a single thread.
float totalCpu(float *input, int len)
{
	//@@ Insert Your Code Here for the CPU Function that implements reduction (sum) on an input vector of lenght len
	for (unsigned int i = 0; i < len; i++)
	{
		input[i] = i;
	}
	int stride = 1;
	for (int j = 1; j < len; j = j * 2)
	{
		for (int i = 0; 2 * i < len; i += j) {
			if (2 * i + j < len)
				input[2 * i] = input[2 * i] + input[(2 * i) + j];
		}
		stride = j;
	}
	stride *= 2;
	input[0] += input[stride];
	return input[0];
}


int main(void)
{
	
	int inputLength;  // number of elements in the input list
	int outputLength; // number of elements in the output
 	float *hostInput; // The input 1D list
	float *hostOutput; // The output list
	float *deviceInput;
	float *deviceOutput;
	float Sum=0;
	float Sum_CPU=0;

	//ask the user to enter the length of the input vector
	printf("Please enter the length of the input array\n");
	scanf("%d", &inputLength);

	//determine the length of the output list 
	outputLength = ceil((float)inputLength / (float)(BLOCK_SIZE << 1));
	
	

	//Allocate the host memory for the input list and output list
	hostInput= (float *)malloc(inputLength * sizeof(float));
	hostOutput = (float *)malloc(outputLength * sizeof(float));
	


	//Random Initialize input array. 
	//There are several ways to do this, such as making functions for manual input or using random numbers. 
	
	// Set the Seed for the random number generator rand() 
	srand(clock());
	for (int i = 0; i < inputLength; i++)
	{
		hostInput[i] = (float)rand()/float(RAND_MAX) ; 
	}



	//Allocate memory on the device for input list and output list and record the needed time
	cudaError_t	err = cudaSuccess;
	GpuTimer timer;
	timer.Start();

	//@@Insert Your Code Here to allocate memory for deviceInput and deviceOutput
	err = cudaMalloc((void	**)&deviceInput, inputLength * sizeof(float));
	if (err != cudaSuccess)
	{
		fprintf(stderr, "Failed	to allocate device input (error code %s)!\n", cudaGetErrorString(err));
		exit(EXIT_FAILURE);
	}

	err = cudaMalloc((void	**)&deviceOutput, outputLength * sizeof(float));
	if (err != cudaSuccess)
	{
		fprintf(stderr, "Failed	to allocate device output (error code %s)!\n", cudaGetErrorString(err));
		exit(EXIT_FAILURE);
	}
	timer.Stop();
	printf("Time to allocate memory on the device is: %f msecs.\n", timer.Elapsed());



	//Copy the input list from the host to the device and record the needed time
	GpuTimer timer1;
	timer1.Start();
	
	//@@ Insert Your Code Here to copy input array from Host to Device
	cudaMemcpy(deviceInput, hostInput, inputLength * sizeof(float), cudaMemcpyHostToDevice);
	timer1.Stop();
	printf("Time to copy the input array from the host to the device is: %f msecs.\n", timer1.Elapsed());


	//Do the Processing on the GPU for reduction
	//@@ Insert Kernel Execution Configuration Parameters for the total kernel
	dim3 dimBlock(BLOCK_SIZE, 1,1);
	dim3 dimGrid((inputLength - 1) / BLOCK_SIZE + 1, 1, 1);

	//Invoke the total kernel and record the needed time for its execution
	GpuTimer timer2;
	timer2.Start();
	//@@ Insert Your Code Here for Kernel Invocation
	total<<<dimGrid,dimBlock>>>(deviceInput, deviceOutput, inputLength);
	timer2.Stop();
	printf("Implemented CUDA code for reduction sum ran in: %f msecs.\n", timer2.Elapsed());



	//Copy resulting output list from device to host and record the needed time
	GpuTimer timer3;
	timer3.Start();
	//@@ Insert Your Code Here to Copy the resulting output from deviceOutput to hostOutput 
	cudaMemcpy(hostOutput, deviceOutput, outputLength * sizeof(float), cudaMemcpyDeviceToHost);
	timer3.Stop();
	printf("Time to copy the resulting output list from the device to the host is: %f msecs.\n", timer3.Elapsed());

	//Write th code for the CPU loop that finds the sum of the output list 
	/********************************************************************
	/ * Reduce output vector on the host
	/ * NOTE: One could also perform the reduction of the output vector
	/ * recursively and support any size input. For simplicity, we do not
	/ * require that for this lab.
	********************************************************************/
	clock_t begin = clock();
	//@@ Insert Your Code Here to Find the Sum of hostOutput
	Sum = hostOutput[0];
	clock_t end = clock();
	double time_spent = (double)(end - begin) / CLOCKS_PER_SEC * 1000;
	printf("Time to perform final reduction of output list on CPU: %f msecs.\n", time_spent);


	//Do the Processing on the CPU
	clock_t begin2 = clock();
	//@@ Insert Your Code Here to call the CPU function totalCpu where the resulting sum is Sum_CPU
	Sum_CPU = totalCpu(hostInput, inputLength);
	clock_t end2 = clock();
	double time_spent2 = (double)(end2 - begin2) / CLOCKS_PER_SEC;
	printf("Implemented CPU serial code ran in: %f msecs.\n", time_spent2);
	//Verify Results Computed by GPU and CPU
	
		
	if (fabs(Sum-Sum_CPU) > 1e-1)
	{
		fprintf(stderr, "Result verification failed!\n");
		exit(EXIT_FAILURE);
	}
	printf("Test PASSED\n");


	//Free host memory
	
	free(hostInput);
	free(hostOutput);
	

	//Free device memory
	//@@ Insert Your Code Here to Free Device Memory
	cudaFree(deviceInput); cudaFree(deviceOutput);

	return 0;

}