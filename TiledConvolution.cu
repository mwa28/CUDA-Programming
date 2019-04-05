#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/opencv.hpp>
#include <iostream>
#include <cuda_runtime.h>
#include "device_launch_parameters.h"
#include "GpuTimer.h"
#include <time.h>

using namespace cv;
using namespace std;

#define FILTER_WIDTH 5
#define FILTER_RADIUS FILTER_WIDTH / 2
#define O_TILE_WIDTH 12
#define BLOCK_WIDTH (O_TILE_WIDTH + FILTER_WIDTH - 1)

// Conv2D_CPU applies a 2D convolution on an RGB image and runs on the CPU
void Conv2D_CPU(unsigned char* outImg, unsigned char* inImg, float* filter, int numRows, int numCols, int numChans) {
	float cumSum;
	int cornerRow, cornerCol;
	int filterRow, filterCol;

	// loop through the pixels in the output image
	for (int row = 0; row < numRows; row++) {
		for (int col = 0; col < numCols; col++) {
			// compute coordinates of top-left corner
			cornerRow = row - FILTER_RADIUS;
			cornerCol = col - FILTER_RADIUS;

			// loop through the channels
			for (int c = 0; c < numChans; c++) {
				// reset accumulator
				cumSum = 0;

				// accumulate values inside filter
				for (int i = 0; i < FILTER_WIDTH; i++) {
					for (int j = 0; j < FILTER_WIDTH; j++) {
						// compute pixel coordinates inside filter
						filterRow = cornerRow + i;
						filterCol = cornerCol + j;

						// make sure we are within image boundaries
						if ((filterRow >= 0) && (filterRow <= numRows) && (filterCol >= 0) && (filterCol <= numCols)) {
							cumSum += inImg[(filterRow*numCols + filterCol)*numChans + c] * filter[i*FILTER_WIDTH + j];
						}
					}
				}
				outImg[(row*numCols + col)*numChans + c] = (unsigned char)cumSum;
			}
		}
	}
}

// Conv2D_GPU applies a 2D convolution on an RGB image and runs on the GPU
__global__ void Conv2D_GPU(unsigned char* outImg, unsigned char* inImg, const float* __restrict__ filter, int numRows, int numCols, int numChans) {
	int filterRow, filterCol;
	int cornerRow, cornerCol;
	int tx = threadIdx.x; int ty = threadIdx.y;
	int bx = blockIdx.x; int by = blockIdx.y;
	int bdx = blockDim.x; int bdy = blockDim.y;

	// compute global thread coordinates in output image
	int row = by * (bdy - 2 * FILTER_RADIUS) + ty;
	int col = bx * (bdx - 2 * FILTER_RADIUS) + tx;

	// make sure thread is within augmented boundaries
	if ((row < numRows + FILTER_RADIUS) && (col < numCols + FILTER_RADIUS)) {
		// allocate a 2D chunk of shared memory
		__shared__ unsigned char chunk[BLOCK_WIDTH][BLOCK_WIDTH];

		// loop through the channels
		for (int c = 0; c < numChans; c++) {
			// load into shared memory
			int relativeRow = row - FILTER_RADIUS;
			int relativeCol = col - FILTER_RADIUS;
			if ((relativeRow < numRows) && (relativeCol < numCols) && (relativeRow >= 0) && (relativeCol >= 0)) {
				chunk[ty][tx] = inImg[(relativeRow*numCols + relativeCol)*numChans + c];
			}
			else {
				chunk[ty][tx] = 0;
			}

			// ensure all threads have loaded to SM
			__syncthreads();

			// instantiate accumulator
			float cumSum = 0;

			// only a subset of threads in block need to do computation
			if ((tx >= FILTER_RADIUS) && (ty >= FILTER_RADIUS) && (ty < bdy - FILTER_RADIUS) && (tx < bdx - FILTER_RADIUS)) {
				// top-left corner coordinates
				cornerRow = ty - FILTER_RADIUS;
				cornerCol = tx - FILTER_RADIUS;

				for (int i = 0; i < FILTER_WIDTH; i++) {
					for (int j = 0; j < FILTER_WIDTH; j++) {
						// filter coordinates
						filterRow = cornerRow + i;
						filterCol = cornerCol + j;

						// accumulate sum
						if ((filterRow >= 0) && (filterRow <= numRows) && (filterCol >= 0) && (filterCol <= numCols)) {
							cumSum += chunk[filterRow][filterCol] * filter[i*FILTER_WIDTH + j];
						}
					}
				}
				// write to global memory
				outImg[(relativeRow*numCols + relativeCol)*numChans + c] = (unsigned char)cumSum;
			}
		}
	}
}

int main(void) {
	
	// read image
	Mat img;
	img = imread("Island.jpg", IMREAD_COLOR);
	if (img.empty()) {
		printf("Cannot read image file %s", "lena_color.bmp");
		exit(1);
	}
	unsigned char* h_inImg = img.data;

	// grab image dimensions
	int imgChans = img.channels();
	int imgWidth = img.cols;
	int imgHeight = img.rows;

	// useful params
	size_t imgSize = sizeof(unsigned char)*imgWidth*imgHeight*imgChans;
	size_t filterSize = sizeof(float)*FILTER_WIDTH*FILTER_WIDTH;
	GpuTimer timer0, timer1, timer2, timer3;

	// allocate host memory
	float* h_filter = (float*)malloc(filterSize);
	unsigned char* h_outImg = (unsigned char*)malloc(imgSize);
	unsigned char* h_outImg_CPU = (unsigned char*)malloc(imgSize);

	// hardcoded filter values
	float filter[FILTER_WIDTH*FILTER_WIDTH] = {
		1 / 273.0, 4 / 273.0, 7 / 273.0, 4 / 273.0, 1 / 273.0,
		4 / 273.0, 16 / 273.0, 26 / 273.0, 16 / 273.0, 4 / 273.0,
		7 / 273.0, 26 / 273.0, 41 / 273.0, 26 / 273.0, 7 / 273.0,
		4 / 273.0, 16 / 273.0, 26 / 273.0, 16 / 273.0, 4 / 273.0,
		1 / 273.0, 4 / 273.0, 7 / 273.0, 4 / 273.0, 1 / 273.0
	};
	h_filter = filter;

	// allocate device memory
	float* d_filter;
	unsigned char* d_inImg;
	unsigned char* d_outImg;
	timer0.Start();
	cudaMalloc((void**)&d_filter, filterSize);
	cudaMalloc((void**)&d_inImg, imgSize);
	cudaMalloc((void**)&d_outImg, imgSize);
	timer0.Stop();
	float t0 = timer0.Elapsed();
	printf("Time to allocate memory on the device is: %f msecs.\n", t0);

	// host2device transfer
	timer1.Start();
	cudaMemcpy(d_filter, h_filter, filterSize, cudaMemcpyHostToDevice);
	cudaMemcpy(d_inImg, h_inImg, imgSize, cudaMemcpyHostToDevice);
	timer1.Stop();
	float t1 = timer1.Elapsed();
	printf("Host to Device transfer: %f msecs.\n", t1);

	// kernel launch
	dim3 dimBlock(BLOCK_WIDTH, BLOCK_WIDTH, 1);
	dim3 dimGrid(ceil(imgWidth / (float)O_TILE_WIDTH), ceil(imgHeight / (float)O_TILE_WIDTH), 1);
	timer2.Start();
	Conv2D_GPU << <dimGrid, dimBlock >> > (d_outImg, d_inImg, d_filter, imgWidth, imgHeight, imgChans);
	timer2.Stop();
	float t2 = timer2.Elapsed();
	printf("Implemented CUDA code ran in: %f msecs.\n", t2);

	// device2host transfer
	timer3.Start();
	cudaMemcpy(h_outImg, d_outImg, imgSize, cudaMemcpyDeviceToHost);
	timer3.Stop();
	float t3 = timer3.Elapsed();
	printf("Time to copy the output image from the device to the host is: %f msecs.\n", t3);

	// do the processing on the CPU
	clock_t begin = clock();
	Conv2D_CPU(h_outImg_CPU, h_inImg, h_filter, imgWidth, imgHeight, imgChans);
	clock_t end = clock();

	// calculate total time for CPU and GPU
	double time_spent = (double)(end - begin) / CLOCKS_PER_SEC * 1000;
	printf("Total CPU code ran in: %f msecs.\n", time_spent);
	float total_device_time = t0 + t1 + t2 + t3;
	printf("Total GPU code ran in: %f msecs.\n", total_device_time);
	float speedup = (float)time_spent / total_device_time;
	printf("GPU Speedup: %f\n", speedup);

	// display images
	Mat img1(imgHeight, imgWidth, CV_8UC3, h_outImg); //colored output image mat object
	Mat img2(imgHeight, imgWidth, CV_8UC3, h_outImg_CPU); //colored output image mat object



	namedWindow("CPUImage", WINDOW_NORMAL); //Create window to display the image
	namedWindow("GPUImage", WINDOW_NORMAL);
	imshow("GPUImage", img1);
	imshow("CPUImage", img2); //Display the image in the window
	waitKey(0); //Wait till you press a key 

	// free host and device memory
	img.release(); img1.release(); img2.release();
	free(h_outImg_CPU); free(h_outImg); free(h_filter);
	cudaFree(d_outImg); cudaFree(d_inImg); cudaFree(d_filter);

	return 0;
}