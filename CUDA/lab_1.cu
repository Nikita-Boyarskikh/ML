/*
 ============================================================================
 Name        : lab_1.cu
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

#define TIME 100
#define LENGTH 100
#define STEPX 1
#define STEPT 0.5


/**
 * CUDA kernel that computes reciprocal values for a given vector
 */
__global__ void reciprocalKernel(float *data, float *new_data, const float time, float step_x, float step_t, const unsigned length) {
	unsigned idx = blockIdx.x*blockDim.x+threadIdx.x;
	if(idx%int(length/step_x))
	{
			data[int(length/step_x-1)]=new_data[int(length/step_x-1)]+5*step_t;
			if(idx%int(length/step_x)!=int(length/step_x-1))
			{
					new_data[idx%int(length/step_x)]=(data[idx%int(length/step_x)+1]-2*data[idx%int(length/step_x)]+data[idx%int(length/step_x)-1])/step_x/step_x*step_t+data[idx%int(length/step_x)];
			}
	}
}

/**
 * Host function that copies the data and launches the work on GPU
 */
float *gpuReciprocal(float *data)
{
	float *gpuData, *new_data;

	CUDA_CHECK_RETURN(cudaMalloc((void **)&new_data, sizeof(float)*LENGTH/STEPX));
	CUDA_CHECK_RETURN(cudaMemcpy(new_data, data, sizeof(float)*LENGTH/STEPX, cudaMemcpyHostToDevice));
	CUDA_CHECK_RETURN(cudaMalloc((void **)&gpuData, sizeof(float)*LENGTH/STEPX));
	CUDA_CHECK_RETURN(cudaMemcpy(gpuData, data, sizeof(float)*LENGTH/STEPX, cudaMemcpyHostToDevice));
	
	static const int BLOCK_SIZE = 10;
	const int blockCount = (LENGTH/STEPX)/BLOCK_SIZE;
	for (unsigned i=0; i < TIME/STEPT; i++)
	{
		if(i&1)
		{
			reciprocalKernel<<<blockCount, BLOCK_SIZE>>> (gpuData, new_data, TIME, STEPX, STEPT, LENGTH);
		}
		else
		{
			reciprocalKernel<<<blockCount, BLOCK_SIZE>>> (new_data, gpuData, TIME, STEPX, STEPT, LENGTH);
		}
	}

	if(!int(TIME/STEPT)&1)
		CUDA_CHECK_RETURN(cudaMemcpy(data, gpuData, sizeof(float)*LENGTH/STEPX, cudaMemcpyDeviceToHost));
	else
		CUDA_CHECK_RETURN(cudaMemcpy(data, new_data, sizeof(float)*LENGTH/STEPX, cudaMemcpyDeviceToHost));

	CUDA_CHECK_RETURN(cudaFree(gpuData));
	CUDA_CHECK_RETURN(cudaFree(new_data));
	return data;
}

void initialize(float *data)
{
	for (unsigned i = 0; i < LENGTH/STEPX; ++i)
	{
		data[i] = 0;
	}
}

int main(void)
{
	float *data = new float[int(LENGTH/STEPX)];
	initialize(data);

	/* Verify the results */
	data=gpuReciprocal(data);

	for (unsigned long i=0; i<LENGTH/STEPX; i++)
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

