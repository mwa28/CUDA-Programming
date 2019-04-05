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
#define NUM_BINS 4096
#define PRIVATE 4096

//Compute Histogram
// Serial implementation for running on CPU using a single thread.
void HistogramCpu(unsigned int *input, unsigned int *bins, unsigned int num_elements, unsigned int num_bins)
{
	//@@ Insert Your Code Here for the CPU Function to Compute the Histogram with the output bins saturated at 127.
	for (int i = 0; i < num_elements; i++) {
		if (bins[input[i]] < 127) {
			bins[input[i]]++;
		}
	}
}


//GPU Kernel for Basic Histogram Computation without sauration
__global__ void histogram_kernel(unsigned int *input, unsigned int *bins, unsigned int num_elements, unsigned int num_bins)
{
	//@@ Insert Your Code Here for the CUDA Kernel for Basic Histogram Computation without saturation
	int tx = threadIdx.x; int bx = blockIdx.x;

    // compute global thread coordinates
    int i = (bx * blockDim.x) + tx;

    // create a private histogram copy for each thread block
    __shared__ unsigned int hist[PRIVATE];

    // each thread must initialize more than 1 location
    if (PRIVATE > BLOCK_SIZE) {
        for (int j=tx; j<PRIVATE; j+=BLOCK_SIZE) {
            if (j < PRIVATE) {
                hist[j] = 0;
            }
        }
    }
    // use the first `PRIVATE` threads of each block to init
    else {
        if (tx < PRIVATE) {
            hist[tx] = 0;
        }
    }
    // wait for all threads in the block to finish
    __syncthreads();

    // update private histogram
    if (i < num_elements) {
        atomicAdd(&(hist[input[i]]), 1);
    }
    // wait for all threads in the block to finish
    __syncthreads();

    // each thread must update more than 1 location
    if (PRIVATE > BLOCK_SIZE) {
        for (int j=tx; j<PRIVATE; j+=BLOCK_SIZE) {
            if (j < PRIVATE) {
                atomicAdd(&(bins[j]), hist[j]);
            }
        }
    }
    // use the first `PRIVATE` threads to update final histogram
    else {
        if (tx < PRIVATE) {
            atomicAdd(&(bins[tx]), hist[tx]);
        }
	}
}


//GPU kernel for converting the output bins into saturated bins at a maximum value of 127
__global__ void convert_kernel(unsigned int *bins, unsigned int num_bins)
{
	//@@ Insert Your Code Here for the CUDA Kernel for ensuring that the output bins are saturated at 127.
	int index = blockIdx.x * blockDim.x + threadIdx.x;
	if (index < num_bins) {
		if (bins[index] > 127) {
			bins[index] = 127;
		}
	}
}

int main(void)
{

	int inputLength;
	unsigned int *hostInput;
	unsigned int *hostBins;
	unsigned int *hostBins_CPU;
	unsigned int *deviceInput;
	unsigned int *deviceBins;

	//ask the user to enter the length of the input vector
	printf("Please enter the length of the input array\n");//the length of the vector should have a maximum value of (2^32-1)
	scanf("%d", &inputLength);


	//Allocate the host memory for the input array and output histogram array
	hostInput = (unsigned int *)malloc(inputLength * sizeof(unsigned int));
	hostBins = (unsigned int *)malloc(NUM_BINS * sizeof(unsigned int));
	hostBins_CPU = (unsigned int *)malloc(NUM_BINS * sizeof(unsigned int));


	//Random Initialize input array.
	//There are several ways to do this, such as making functions for manual input or using random numbers.

	// Set the Seed for the random number generator rand()
	srand(clock());
	for (int i = 0; i < inputLength; i++)

		hostInput[i] = int((float)rand()  * (NUM_BINS - 1) / float(RAND_MAX)); //the values will range from 0 to (NUM_BINS - 1)


	for (int i = 0; i < NUM_BINS; i++)
		hostBins_CPU[i] = 0;//initialize CPU histogram array to 0






	//Allocate memory on the device for input array and output histogram array and record the needed time

	GpuTimer timer;
	timer.Start();

	//--@@Insert Your Code Here to allocate memory for deviceInput and deviceBins
	cudaMalloc((void**)&deviceInput, inputLength * sizeof(unsigned int));
	cudaMalloc((void**)&deviceBins, NUM_BINS * sizeof(unsigned int));
	//initialize deviceBins to zero
	cudaMemset(deviceBins, 0, NUM_BINS * sizeof(unsigned int));

	timer.Stop();
	printf("Time to allocate memory on the device is: %f msecs.\n", timer.Elapsed());



	//Copy the input array from the host to the device and record the needed time
	GpuTimer timer1;
	timer1.Start();

	//--@@ Insert Your Code Here to copy input array from Host to Device
	cudaMemcpy(deviceInput, hostInput, inputLength * sizeof(unsigned int), cudaMemcpyHostToDevice);
	timer1.Stop();
	printf("Time to copy the input array from the host to the device is: %f msecs.\n", timer1.Elapsed());


	//Do the Processing on the GPU for Basic Histogram computation without saturation
	//--@@ Insert Kernel Execution Configuration Parameters for the histogram_kernel
	dim3 dimBlock(BLOCK_SIZE, 1, 1);
	dim3 dimGrid(ceil(inputLength / (float)BLOCK_SIZE), 1, 1);

	//Invoke the histogram_kernel kernel and record the needed time for its execution
	GpuTimer timer2;
	timer2.Start();
	//@@ Insert Your Code Here for Kernel Invocation
	histogram_kernel << <dimGrid, dimBlock >> > (deviceInput, deviceBins, inputLength, NUM_BINS);
	timer2.Stop();
	printf("Implemented CUDA code for basic histogram calculation ran in: %f msecs.\n", timer2.Elapsed());


	//Do the Processing on the GPU for convert_kernel
	//--@@ Insert Kernel Execution Configuration Parameters for the convert_kernel
	dim3 dimBlock2(BLOCK_SIZE, 1, 1);
	dim3 dimGrid2(ceil(NUM_BINS / (float)BLOCK_SIZE), 1, 1);

	//Invoke the convert_kernel kernel and record the needed time for its execution
	GpuTimer timer3;
	timer3.Start();
	//--@@ Insert Your Code Here for Kernel Invocation
	convert_kernel << <dimGrid2, dimBlock2 >> > (deviceBins, NUM_BINS);
	timer3.Stop();
	printf("Implemented CUDA code for output saturation ran in: %f msecs.\n", timer3.Elapsed());


	//Copy resulting histogram array from device to host and record the needed time
	GpuTimer timer4;
	timer4.Start();
	//--@@ Insert Your Code Here to Copy the resulting histogram deviceBins from device to the Host hostBins
	cudaMemcpy(hostBins, deviceBins, NUM_BINS * sizeof(unsigned int), cudaMemcpyDeviceToHost);
	timer4.Stop();
	printf("Time to copy the resulting Histogram from the device to the host is: %f msecs.\n", timer4.Elapsed());


	//Do the Processing on the CPU
	clock_t begin = clock();
	//--@@ Insert Your Code Here to call the CPU function HistogramCpu where the resulting vector is hostBins_CPU
	HistogramCpu(hostInput, hostBins_CPU, inputLength, NUM_BINS);
	clock_t end = clock();
	double time_spent = (double)(end - begin) / CLOCKS_PER_SEC * 1000;
	printf("Implemented CPU serial code ran in: %f msecs.\n", time_spent);

	//Verify Results Computed by GPU and CPU
	for (int i = 0; i < NUM_BINS; i++)

		if (abs(int(hostBins_CPU[i] - hostBins[i])) > 0)
		{
			fprintf(stderr, "Result verification failed at element (%d)!\n", i);
			exit(EXIT_FAILURE);
		}
	printf("Test PASSED\n");


	//Free host memory
	free(hostBins);
	free(hostBins_CPU);
	free(hostInput);


	//Free device memory
	//@@ Insert Your Code Here to Free Device Memory
	cudaFree(deviceInput);
	cudaFree(deviceBins);
	return 0;

}