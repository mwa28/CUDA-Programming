#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/opencv.hpp>
#include <iostream>
#include <cuda_runtime.h>
#include "device_launch_parameters.h"
#include "GpuTimer.h"
#include<time.h>

using namespace cv;
using namespace std;


#define Mask_width 5

#define Mask_radius Mask_width / 2

#define O_TILE_WIDTH 12

#define BLOCK_WIDTH (O_TILE_WIDTH + Mask_width - 1)


//In OpenCV the image is read in BGR format, that is for each pixel, the Blue, Green, then Red components are read from the image file.

// Serial implementation for running on CPU using a single thread.
void Convolution_2D_Cpu(unsigned char* InputImage, unsigned char* OutputImage, float* M, int numRows, int numCols, int Channels)
{
	//@@ Insert your code here
	float sum;
	int cornerRow, cornerCol;
	int filterRow, filterCol;

	for (int row = 0; row < numRows; row++) {
		for (int col = 0; col < numCols; col++) {
		
			cornerRow = row - Mask_radius;
			cornerCol = col - Mask_radius;

			for (int c = 0; c < Channels; c++) {
				
				sum = 0;

				for (int i = 0; i < Mask_width; i++) {
					for (int j = 0; j < Mask_width; j++) {
						
						filterRow = cornerRow + i;
						filterCol = cornerCol + j;

						if ((filterRow >= 0) && (filterRow <= numRows) && (filterCol >= 0) && (filterCol <= numCols)) {
							sum += InputImage[(filterRow*numCols + filterCol)*Channels + c] * M[i*Mask_width + j];
						}
					}
				}
				OutputImage[(row*numCols + col)*Channels + c] = (unsigned char)sum;
			}
		}
	}
	
}


// we have 3 channels corresponding to B, G, and R components of each pixel
// The input image is encoded as unsigned characters [0, 255]

__global__ void TiledConvolution_2D(unsigned char * InputImage, unsigned char * OutputImage, const float *__restrict__ M,int numRows, int numCols, int Channels)
{
	//@@ Insert Your Kernel code Here
	int filterRow, filterCol;
	int cornerRow, cornerCol;
	int tx = threadIdx.x; int ty = threadIdx.y;
	int bx = blockIdx.x; int by = blockIdx.y;
	int bdx = blockDim.x; int bdy = blockDim.y;

	int row = by * (bdy - 2 * Mask_radius) + ty;
	int col = bx * (bdx - 2 * Mask_radius) + tx;

	if ((row < numRows + Mask_radius) && (col < numCols + Mask_radius)) {
		
		__shared__ unsigned char chunk[BLOCK_WIDTH][BLOCK_WIDTH];

		
		for (int c = 0; c < Channels; c++) {
		
			int relativeRow = row - Mask_radius;
			int relativeCol = col - Mask_radius;
			if ((relativeRow < numRows) && (relativeCol < numCols) && (relativeRow >= 0) && (relativeCol >= 0)) {
				chunk[ty][tx] = InputImage[(relativeRow*numCols + relativeCol)*Channels + c];
			}
			else {
				chunk[ty][tx] = 0;
			}

		
			__syncthreads();

			float sum = 0;

			if ((tx >= Mask_radius) && (ty >= Mask_radius) && (ty < bdy - Mask_radius) && (tx < bdx - Mask_radius)) {
				
				cornerRow = ty - Mask_radius;
				cornerCol = tx - Mask_radius;

				for (int i = 0; i < Mask_width; i++) {
					for (int j = 0; j < Mask_width; j++) {
						
						filterRow = cornerRow + i;
						filterCol = cornerCol + j;

						if ((filterRow >= 0) && (filterRow <= numRows) && (filterCol >= 0) && (filterCol <= numCols)) {
							sum += chunk[filterRow][filterCol] * M[i*Mask_width + j];
						}
					}
				}
				OutputImage[(relativeRow*numCols + relativeCol)*Channels + c] = (unsigned char)sum;
			}
		}
	}
	
}



int main(void)
{
	//Read the image using OpenCV
	Mat image; //Create matrix to read image
	image = imread("lena_color.bmp", IMREAD_COLOR);
	if (image.empty()) {
		printf("Cannot read image file %s", "lena_color.bmp");
		exit(1);
	}


	int imageChannels = 3;
	int imageWidth = image.cols;
	int imageHeight = image.rows;

	//Allocate the host image vectors
	unsigned char *h_InputImage = (unsigned char *)malloc(sizeof(unsigned char)*imageWidth*imageHeight*imageChannels);
	unsigned char *h_OutputImage = (unsigned char *)malloc(sizeof(unsigned char)*imageWidth*imageHeight*imageChannels);
	unsigned char *h_OutputImage_CPU = (unsigned char *)malloc(sizeof(unsigned char)*imageWidth*imageHeight*imageChannels);
	float *hostMaskData=(float *)malloc(sizeof(float)*Mask_width*Mask_width);

	h_InputImage = image.data; //The data member of a Mat object returns the pointer to the first row, first column of the image.
							
	float Mask[Mask_width*Mask_width] = {1/273.0, 4/273.0, 7/273.0, 4/273.0, 1/273.0, 4/273.0, 16/273.0, 26/273.0, 16 / 273.0, 4 / 273.0, 7 / 273.0, 26 / 273.0, 41 / 273.0, 26 / 273.0, 7 / 273.0, 4 / 273.0, 16 / 273.0, 26 / 273.0, 16 / 273.0, 4 / 273.0, 1 / 273.0, 4 / 273.0, 7 / 273.0, 4 / 273.0, 1/273.0 };
	hostMaskData = Mask;
	
	//Allocate memory on the device for the input image and the output image and record the needed time
	unsigned char *d_InputImage, *d_OutputImage;
	float *deviceMaskData;
	GpuTimer timer;
	cudaError_t err1 = cudaSuccess;
	cudaError_t err2 = cudaSuccess;
	timer.Start();

	//@@ Insert Your code Here to allocate memory on the device for input and output images
	err1 = cudaMalloc((void	**)&d_InputImage, sizeof(unsigned char)*imageWidth*imageHeight*imageChannels);
	if (err1 != cudaSuccess)
	{
		fprintf(stderr, "Failed	to allocate device RGB Image Vector for Input(error code %s)!\n", cudaGetErrorString(err1));
		exit(EXIT_FAILURE);
	}
	err2 = cudaMalloc((void	**)&d_OutputImage, sizeof(unsigned char)*imageWidth*imageHeight*imageChannels);
	if (err2 != cudaSuccess)
	{
		fprintf(stderr, "Failed	to allocate device RGB Image Vector for Output(error code %s)!\n", cudaGetErrorString(err2));
		exit(EXIT_FAILURE);
	}
	//@@Insert your code Here to allocate memory on the device for the Mask data
	cudaMalloc((void**)&deviceMaskData, sizeof(float)*Mask_width*Mask_width);
	timer.Stop();
	printf("Time to allocate memory on the device is: %f msecs.\n", timer.Elapsed());


	//Copy the input image and mask data from the host to the device and record the needed time
	GpuTimer timer1;
	timer1.Start();

	//@@ Insert your code here to Copy the input image from the host to the device
	cudaMemcpy(d_InputImage, h_InputImage, sizeof(unsigned char)*imageWidth*imageHeight*imageChannels, cudaMemcpyHostToDevice);
	//@@Insert your code here to copy the mask data from host to device
	cudaMemcpy(deviceMaskData, hostMaskData, sizeof(float)*Mask_width*Mask_width, cudaMemcpyHostToDevice);
	
	timer1.Stop();
	printf("Time to copy the input image from the host to the device is: %f msecs.\n", timer1.Elapsed());

	

	//Do the Processing on the GPU
	//Kernel Execution Configuration Parameters
	dim3 dimBlock(BLOCK_WIDTH, BLOCK_WIDTH, 1);

	//@@ Insert Your code Here for grid dimensions
	dim3 dimGrid(ceil(imageWidth / (float)O_TILE_WIDTH), ceil(imageHeight / (float)O_TILE_WIDTH), 1);

	//Invoke the 2DTiledConvolution kernel and record the needed time for its execution

	//GpuTimer timer;
	GpuTimer timer2;
	timer2.Start();

	//@@ Insert your code here for kernel invocation
	TiledConvolution_2D <<<dimGrid, dimBlock >>> (d_OutputImage, d_InputImage, deviceMaskData, imageWidth, imageHeight, imageChannels);

	timer2.Stop();
	printf("Implemented CUDA code ran in: %f msecs.\n", timer2.Elapsed());

	//Copy resulting output image from device to host and record the needed time
	GpuTimer timer3;
	timer3.Start();

	//@@ Insert your code here to Copy resulting output image from device to host 
	cudaMemcpy(h_OutputImage, d_OutputImage, sizeof(unsigned char)*imageWidth*imageHeight*imageChannels, cudaMemcpyDeviceToHost);

	timer3.Stop();
	printf("Time to copy the output image from the device to the host is: %f msecs.\n", timer3.Elapsed());

	//Do the Processing on the CPU
	clock_t begin = clock();

	//@@ Insert your code her to call the cpu function for 2DConvolution on the CPU	
	Convolution_2D_Cpu(h_OutputImage_CPU, h_InputImage, hostMaskData, imageWidth, imageHeight, imageChannels);

	clock_t end = clock();
	double time_spent = (double)(end - begin) / CLOCKS_PER_SEC * 1000;
	printf("Implemented CPU code ran in: %f msecs.\n", time_spent);

	//Postprocess and Display the resulting images using OpenCV
	Mat Image1(imageHeight, imageWidth, CV_8UC3, h_OutputImage); //colored output image mat object
	Mat Image2(imageHeight, imageWidth, CV_8UC3, h_OutputImage_CPU); //colored output image mat object



	namedWindow("CPUImage", WINDOW_NORMAL); //Create window to display the image
	namedWindow("GPUImage", WINDOW_NORMAL);
	imshow("GPUImage", Image1);
	imshow("CPUImage", Image2); //Display the image in the window
	waitKey(0); //Wait till you press a key 



	//Free host memory
	//free(h_OutputImage);
	image.release();
	Image1.release();
	Image2.release();
	free(h_OutputImage);
	free(h_OutputImage_CPU);

	//Free device memory

	//@@ Insert your code here to free device memory
	cudaFree(d_OutputImage); cudaFree(d_InputImage); cudaFree(deviceMaskData);

	return 0;

}