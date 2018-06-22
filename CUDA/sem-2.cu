/*
 ============================================================================
 Name        : sem-2.cu
 Author      : Boyarskikh_Nikita
 Version     :
 Copyright   : 
 Description : CUDA compute reciprocals
 ============================================================================
 */

#include <iostream>
#include <stdlib.h>

static void CheckCudaErrorAux (const char *, unsigned, const char *, cudaError_t);
#define CUDA_CHECK_RETURN(value) CheckCudaErrorAux(__FILE__,__LINE__, #value, value)

#define TIME 0.5
#define LENGTH 100
#define STEPX 1
#define STEPT 0.5

#define A1 (STEPT/STEPX/STEPX)
#define A2 (1-2*A1)

/**
 * Initialization data and matrix
 */
void initialize(double *data, unsigned size)
{
	for (unsigned i = 0; i < size-1; ++i)
		data[i] = 0;
	data[size-1] = 5;
}

void init_matrix(double *matrix) {
	for (int i=0; i<int(LENGTH/STEPX); i++) {
		for (int j=0; j<int(TIME/STEPT); j++)
			if(abs(i-j)>1||i==0||i==int(LENGTH/STEPX)-1)
				matrix[i*LENGTH/STEPX+j]=0;
			else
				if(i-j==1)
					matrix[i*LENGTH/STEPX+j]=A2;
				else if(i-j==-1)
					matrix[i*LENGTH/STEPX+j]=A2;
				else
					matrix[i*LENGTH/STEPX+j]=A1;
	}
	matrix[0]=1;
	matrix[int(TIME/STEPT*LENGTH/STEPX)-1]=1;
	matrix[int(LENGTH/STEPX*TIME/STEPT)-2]=-1;
}

/**
 * CUDA kernel that computes reciprocal values for a given vector
 */
__global__ void reciprocalKernel(double *new_data, double *old_data, double *matrix, int height, int width) {
	unsigned idx = blockIdx.x*blockDim.x+threadIdx.x;
	if(idx < heigth)
	{
		double sum=0;
		for (int i=0; i<width; ++i)
		{
			sum+=matrix[idx][i];
		}
		new_data[idx]=old_data[idx]*sum;
	}
}

/**
 * Host function that copies the data and launches the work on GPU
 */
double *gpuReciprocal(double *data)
{
	double *gpuData, *new_data;

	CUDA_CHECK_RETURN(cudaMalloc((void **)&new_data, sizeof(double)*int(LENGTH/STEPX)));
	CUDA_CHECK_RETURN(cudaMemcpy(new_data, data, sizeof(double)*int(LENGTH/STEPX), cudaMemcpyHostToDevice));
	CUDA_CHECK_RETURN(cudaMalloc((void **)&gpuData, sizeof(double)*int(LENGTH/STEPX)));

	static const int BLOCK_SIZE = 32;
	static const double matrix = new[int(LENGTH/STEPX)*int(TIME/STEPT)];
	init_matrix(matrix);
	const int blockCount = (LENGTH/STEPX)/BLOCK_SIZE;
	for (unsigned i=0; i < int(TIME/STEPT); i++)
	{
		if(i&1)
		{
			reciprocalKernel<<<blockCount, BLOCK_SIZE>>> (gpuData, new_data, matrix, int(LENGTH/STEPX), int(TIME/STEPT));
		}
		else
		{
			reciprocalKernel<<<blockCount, BLOCK_SIZE>>> (new_data, gpuData, matrix, int(LENGTH/STEPX), int(TIME/STEPT));
		}
	}

	if(!int(TIME/STEPT)&1)
		CUDA_CHECK_RETURN(cudaMemcpy(data, gpuData, sizeof(double)*int(LENGTH/STEPX), cudaMemcpyDeviceToHost));
	else
		CUDA_CHECK_RETURN(cudaMemcpy(data, new_data, sizeof(double)*int(LENGTH/STEPX), cudaMemcpyDeviceToHost));
	
	CUDA_CHECK_RETURN(cudaFree(gpuData));
	CUDA_CHECK_RETURN(cudaFree(new_data));
	return data;
}

int main(void)
{
	double *data = new double[int(LENGTH/STEPX)];

	initialize (data, int(LENGTH/STEPX));

	data = gpuReciprocal(data);

	/* Verify the results */
	for (unsigned long i=0; i<int(LENGTH/STEPX); i++)
	{
		std::cout<<data[i]<<std::endl;
	}

	/* Free memory */
	delete[] data;

	return 0;
}

/**
 * Check the return value of the CUDA runtime API call and exit
 * the application if the call has failed.
 */
static void CheckCudaErrorAux (const char *file, unsigned line, const char *statement, cudaError_t err)
{
	if (err == cudaSuccess)
		return;
	std::cerr << statement<<" returned " << cudaGetErrorString(err) << "("<<err<< ") at "<<file<<":"<<line << std::endl;
	exit (1);
}
