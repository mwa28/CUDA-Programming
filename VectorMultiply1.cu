#include <stdio.h>
#include <stdlib.h>
#include <math.h>
/**
*	Element-wise	Vector	Multiplication:	C[i]	=	A[i]	*	B[i].
*	This	sample	is	a	very	basic	sample	that	implements	element	by	element	vector	multiplication.
*/
//	For	the	CUDA	runtime	routines	(prefixed	with	"cuda_")
#include	<cuda_runtime.h>
#include	"device_launch_parameters.h"
/**
*	CUDA	Kernel	Device	code
*	Computes	the	element-wise	vector	multiplication	of	A	and	B	into	C.	The	3	vectors	have	the	same	number	of
elements	numElements.
*/
__global__	void	vectorMultiply(float	*A, float	*B, float	*C, int	numElements)
{
	int size = numElements * sizeof(float); 
	float *d_A, *d_B, *d_C;
	int i = threadIdx.x + blockDim.x*blockIdx.x;
	if (i < numElements) C[i] = A[i] * B[i];
}
//Host	main	routine
int	main(void)
{
	//	Error	code	to	check	return	values	for	CUDA	calls
	cudaError_t	err = cudaSuccess;
	//	Print	the	vector	length	to	be	used,	and	compute	its	size
	float	EPS = 0.00001;
	int	numElements = 50000;
	size_t	size = numElements * sizeof(float);
	printf("[Vector multiplication of %d elements]\n", numElements);
	//	Allocate	the	host	input	vector	A
	float	*h_A = (float	*)malloc(size);
	//	Allocate	the	host	input	vector	B
	float	*h_B = (float	*)malloc(size);
	//	Allocate	the	host	output	vector	C
	float	*h_C = (float	*)malloc(size);
	//	Verify	that	allocations	succeeded
	if (h_A == NULL || h_B == NULL || h_C == NULL)
	{
		fprintf(stderr, "Failed	to allocate host vectors!\n");
		exit(EXIT_FAILURE);
	}
	//	Initialize	the	host	input	vectors
	for (int i = 0; i < numElements; i++)
	{
		*(h_A + i) = (float)i;
		//printf("h_A = %f\n", h_A[i]);
	}
	for (int i = 0; i < numElements; i++)
		*(h_B + i) = (1 / (EPS + i));
	//	Allocate	the	device	input	vector	A
	float	*d_A = NULL;
	err = cudaMalloc((void	**)&d_A, size);
	if (err != cudaSuccess)
	{
		fprintf(stderr, "Failed	to allocate device vector A (error code %s)!\n", cudaGetErrorString(err));
		exit(EXIT_FAILURE);
	}
	//	Allocate	the	device	input	vector	B
	float *d_B = NULL;
	err = cudaMalloc((void	**)&d_B, size);
	if (err != cudaSuccess)
	{
		fprintf(stderr, "Failed to allocate device vector B (error code	%s)!\n", cudaGetErrorString(err));
		exit(EXIT_FAILURE);
	}
	//	Allocate	the	device	output	vector	C
	float *d_C = NULL;
	err = cudaMalloc((void	**)&d_C, size);
	if (err != cudaSuccess)
	{
		fprintf(stderr, "Failed	to allocate	device vector C (error code	%s)!\n", cudaGetErrorString(err));
		exit(EXIT_FAILURE);
	}
	//	Copy	the	host	input	vectors	A	and	B	in	host	memory	to	the	device	input	vectors	in	device	memory
	printf("Copy input data	from the host memory to the CUDA device\n");
	cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);
	cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice);
	//	Launch	the	VectorMultiply CUDA	Kernel
	int	threadsPerBlock = 256;

	int blocksPerGrid = ceil(numElements / (float) threadsPerBlock);
	vectorMultiply <<< blocksPerGrid, threadsPerBlock >>>(d_A, d_B, d_C, numElements);

	printf("CUDA kernel launch with	%d blocks of %d threads\n", blocksPerGrid, threadsPerBlock);	
	err = cudaGetLastError();
	if (err != cudaSuccess)
	{
		fprintf(stderr, "Failed	to launch vectorAdd	kernel (error code %s)!\n", cudaGetErrorString(err));
		exit(EXIT_FAILURE);
	}
	//	Copy	the	device	result	vector	in	device	memory	to	the	host	result	vector
	//	in	host	memory.
	printf("Copy output data from the CUDA device to the host memory\n");
	cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost);
	//	Verify	that	the	result	vector	is	correct
	for (int i = 0; i < numElements; ++i)
	{
		if (fabs((h_A[i] * h_B[i]) - h_C[i])	>	1e-5)
		{
			fprintf(stderr, "Result	verification failed	at element %d!\n", i);
			exit(EXIT_FAILURE);
		}
	}
	printf("Test PASSED\n");
	//	Free	device	global	memory
	cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);
	//	Free	host	memory
	free(h_A);
	free(h_B);
	free(h_C);
	printf("Done\n");
	return	0;
}