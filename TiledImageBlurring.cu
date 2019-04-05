#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include<opencv2/opencv.hpp>
#include <iostream>
#include <cuda_runtime.h>
#include "device_launch_parameters.h"
#include "GpuTimer.h"
#include<time.h>

using namespace cv;
using namespace std;

#define BLUR_SIZE 3
#define BLOCK_SIZE 16
#define TILE BLOCK_SIZE + (2*BLUR_SIZE)

/* 
   *******************************************************************************************************************************************************
   | The below code was implemented using CUDA 10.0 with OpenCV 4.0.1. The minor change is the flag used in IMREAD() that was changed to IMREAD_GRAYSCALE|																																				|
   *******************************************************************************************************************************************************
*/
// Serial implementation for running on CPU using a single thread.
void ImageBlurCpu(unsigned char* blurImg, unsigned char* InputImg,int width, int height)
{
	int sum, pixelnum;
	int lastRow, lastCol;
	int rowfilter, colfilter;
	int blurSize = 2 * BLUR_SIZE + 1;

	for (int row = 0; row < height; row++) {
		for (int col = 0; col < width; col++) {
			lastRow = row - BLUR_SIZE;
			lastCol = col - BLUR_SIZE;

			pixelnum = 0;
			sum = 0;

			for (int i = 0; i < blurSize; i++) {
				for (int j = 0; j < blurSize; j++) {
					rowfilter = lastRow + i;
					colfilter = lastCol + j;
					if ((rowfilter >= 0) && (rowfilter <= height) && (colfilter >= 0) && (colfilter <= width)) {
						sum += InputImg[rowfilter*width + colfilter];
						pixelnum++;
					}
				}
			}
			blurImg[row*width + col] = (unsigned char)(sum / pixelnum);
		}
	}
}


// The input image is grayscale and is encoded as unsigned characters [0, 255]
__global__ void ImageBlur(unsigned char *out, unsigned char *in, int width, int height) 
{
	int rowfilter, colfilter;
	int lastrow, lastcol;
	int blurSize = 2 * BLUR_SIZE + 1;
	int tx = threadIdx.x; 
	int ty = threadIdx.y;
	int bx = blockIdx.x; 
	int by = blockIdx.y;
	int bdx = blockDim.x; 
	int bdy = blockDim.y;

	int row = by * (bdy - 2 * BLUR_SIZE) + ty;
	int col = bx * (bdx - 2 * BLUR_SIZE) + tx;

	if ((row < height + BLUR_SIZE) && (col < width + BLUR_SIZE)) {
		__shared__ unsigned char tiled[TILE][TILE];

		int relativeRow = row - BLUR_SIZE;
		int relativeCol = col - BLUR_SIZE;
		if ((relativeRow < height) && (relativeCol < width) && (relativeRow >= 0) && (relativeCol >= 0)) {
			tiled[ty][tx] = in[relativeRow*width + relativeCol];
		}
		else {
			tiled[ty][tx] = 0;
		}

		__syncthreads();

		int pixelnum = 0;
		int sum = 0;

		if ((tx >= BLUR_SIZE) && (ty >= BLUR_SIZE) && (ty < bdy - BLUR_SIZE) && (tx < bdx - BLUR_SIZE)) {
			lastrow = ty - BLUR_SIZE;
			lastcol = tx - BLUR_SIZE;

			for (int i = 0; i < blurSize; i++) {
				for (int j = 0; j < blurSize; j++) {
					rowfilter = lastrow + i;
					colfilter = lastcol + j;

					if ((rowfilter >= 0) && (rowfilter <= height) && (colfilter >= 0) && (colfilter <= width)) {
						sum += tiled[rowfilter][colfilter];
						pixelnum++;
					}
				}
			}
			out[relativeRow*width + relativeCol] = (unsigned char)(sum/pixelnum);
		}
	}
}



int main(void)
{
	//Read the image using OpenCV
	Mat image; //Create matrix to read iamge
	image= imread("Tiger.jpg", IMREAD_GRAYSCALE);
	if (image.empty()) {
		printf("Cannot read image file %s", "Tiger.jpg");
		exit(1);
	}

	
	
	int imageWidth=image.cols;
	int imageHeight=image.rows;

	//Allocate the host image vectors
	unsigned char *h_OrigImage;
	unsigned char *h_BlurImage= (unsigned char *)malloc(sizeof(unsigned char)*imageWidth*imageHeight);
	unsigned char *h_BlurImage_CPU= (unsigned char *)malloc(sizeof(unsigned char)*imageWidth*imageHeight);

	h_OrigImage = image.data; //The data member of a Mat object returns the pointer to the first row, first column of the image.
							 //try image.ptr()

	cudaError_t	err = cudaSuccess;
	//Allocate memory on the device for the original image and the blurred image and record the needed time
	unsigned char *d_OrigImage, *d_BlurImage;
	GpuTimer timer;
	timer.Start();
	
	//@@ Insert Your code Here to allocate memory on the device for original and blurred images
	err = cudaMalloc((void	**)&d_OrigImage, sizeof(unsigned char)*imageWidth*imageHeight);
	if (err != cudaSuccess)
	{
		fprintf(stderr, "Failed	to allocate device Original Image Vector (error code %s)!\n", cudaGetErrorString(err));
		exit(EXIT_FAILURE);
	}
	err = cudaMalloc((void	**)&d_BlurImage, sizeof(unsigned char)*imageWidth*imageHeight);
	if (err != cudaSuccess)
	{
		fprintf(stderr, "Failed	to allocate device Blurred Image Vector (error code %s)!\n", cudaGetErrorString(err));
		exit(EXIT_FAILURE);
	}
	timer.Stop();
	printf("Time to allocate memory on the device is: %f msecs.\n", timer.Elapsed());

	

	//Copy the original image from the host to the device and record the needed time
	GpuTimer timer1;
	timer1.Start();
	
	//@@ Insert your code here to Copy the original image from the host to the device
	cudaMemcpy(d_OrigImage, h_OrigImage, sizeof(unsigned char)*imageWidth*imageHeight, cudaMemcpyHostToDevice);
	timer1.Stop();
	printf("Time to copy the Original image from the host to the device is: %f msecs.\n", timer1.Elapsed());

	
	//Do the Processing on the GPU
	//Kernel Execution Configuration Parameters
	dim3 dimBlock(16, 16, 1);
	
	//@@ Insert Your code Here for grid dimensions
	dim3 dimGrid(ceil(imageWidth / 16.0), ceil(imageHeight / 16.0), 1);
	
	//Invoke the ImageBlur kernel and record the needed time for its execution
	//GpuTimer timer;
	GpuTimer timer2;
	timer2.Start();

	//@@ Insert your code here for kernel invocation
	ImageBlur<<< dimGrid, dimBlock >>>(d_BlurImage, d_OrigImage, imageWidth, imageHeight);
	timer2.Stop();
	printf("Implemented ImageBlur Kernel ran in: %f msecs.\n", timer2.Elapsed());

	//Copy resulting blurred image from device to host and record the needed time
	GpuTimer timer3;
	timer3.Start();
	
	//@@ Insert your code here to Copy resulting blurred image from device to host 
	cudaMemcpy(h_BlurImage, d_BlurImage, sizeof(unsigned char)*imageWidth*imageHeight, cudaMemcpyDeviceToHost);
	timer3.Stop();
	printf("Time to copy the blurred image from the device to the host is: %f msecs.\n", timer3.Elapsed());

	

	//Do the Processing on the CPU
	clock_t begin = clock();
	
	//@@ Insert your code her to call the cpu function for ImageBlur on the CPU	
	ImageBlurCpu(h_BlurImage_CPU,h_OrigImage, imageWidth, imageHeight);

	clock_t end = clock();
	double time_spent = (double)(end - begin) / CLOCKS_PER_SEC*1000;
	printf("Implemented CPU code ran in: %f msecs.\n", time_spent);

	//Postprocess and Display the resulting images using OpenCV
	Mat Image1(imageHeight, imageWidth,CV_8UC1,h_BlurImage); //grayscale image mat object
	Mat Image2(imageHeight,imageWidth,CV_8UC1,h_BlurImage_CPU ); //grayscale image mat object

	

	namedWindow("CPUImage", WINDOW_NORMAL); //Create window to display the image
	namedWindow("GPUImage", WINDOW_NORMAL);
	namedWindow("OriginalImage", WINDOW_NORMAL);
	imshow("GPUImage",Image1);
	imshow("CPUImage",Image2); //Display the image in the window
	imshow("OriginalImage", image); //Display the original image in the window
	waitKey(0); //Wait till you press a key 

	
	
	//Free host memory
	image.release();
	Image1.release();
	Image2.release();
	free(h_BlurImage);
	free(h_BlurImage_CPU);

	//Free device memory
	
	//@@ Insert your code here to free device memory
	cudaFree(d_OrigImage); cudaFree(d_BlurImage);

	return 0;

}