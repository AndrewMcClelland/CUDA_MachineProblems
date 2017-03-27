
#include "cuda.h"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <iostream>
#include <algorithm>
#include <random>

using namespace std;

// See slide 34 on GPU Computing Slide Set for full heterogeneous code example
// See slide 45 on GPU Computing Slide Set for vector addition
// See slide 63 on GPU Computing Slide Set for matrix multiplication
// See slide 23/35 on NVIDIA textbook for partial MatrixMull / full VecAdd
// See slide 36 on NVIDIA textbook for some 2D array multiplication

// Device code
__global__ void MatrixMult_Device(const float* d_a, const float* d_b, float* d_c, const int n)
{
	int column = threadIdx.x + (blockIdx.x * blockDim.x);
	int row = threadIdx.y + (blockIdx.y * blockDim.y);
	
	if ( (column < n) && (row < n)) {
		d_c[column][row] = d_a[column][row] * d_b[column][row];
	}
}


// Host code
int main()
{
    const int N = 128;
	const int arraySize = N * N;
	const int arraySizeBytes = arraySize * sizeof(float);

	float *h_a, *h_b, *h_c;
	float *d_a, *d_b, *d_c;

	// Allocate space for host copies on CPU
	h_a = (float *)malloc(arraySizeBytes);
	h_b = (float *)malloc(arraySizeBytes);
	h_c = (float *)malloc(arraySizeBytes);

	// Allocate space for device copies on GPU
	cudaMalloc((void **)& d_a, arraySizeBytes);
	cudaMalloc((void **)& d_b, arraySizeBytes);
	cudaMalloc((void **)& d_c, arraySizeBytes);

	//------------------IS THIS SINGLE PRECISION---------------
	// Fill 2D host input matrices with random single-precision floating point numbers
	float random_num_a, random_num_b;
	float range = (10.0 - (-10.0)); // from -10 to + 10
	float div = RAND_MAX / range;
	srand(time(NULL));

	// Use C++ std::fill instead? (slide 34 of GPU Lecture Set)
	for(int i = 0; i < N; i++) {
		for(int j = 0; j < N; j++) {
			random_num_a = (-10) + (rand() / div); // float in range -10 to +10
			random_num_b = (-10) + (rand() / div); // float in range -10 to +10

			h_a[i][j] = random_num_a;
			h_b[i][j] = random_num_b;

			printf("%f\n", h_a[i][j]);
		}
	}

	// Copy input matrices from host memory to device memory
	cudaMemcpy(d_a, h_a, arraySizeBytes, cudaMemcpyHostToDevice);
	cudaMemcpy(d_b, h_b, arraySizeBytes, cudaMemcpyHostToDevice);

	//---------------- Invoke kernel (slide 35 of NVIDIA textbook)
	int threadsPerBlock = 256;
	int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;
	
	MatrixMult_Device<<<blocksPerGrid, threadsPerBlock>>>(d_a, d_b, d_c, N);

	// Copy result from device to host
	cudaMemcpy(h_c, d_c, arraySizeBytes, cudaMemcpyDeviceToHost);

	// Free device memory
	cudaFree(d_a);
	cudaFree(d_b);
	cudaFree(d_c);

    return 0;
}
