
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

#define BLOCK_WIDTH 16

// Device code
__global__ void MatrixMult_Device(const float* d_a, const float* d_b, float* d_c, const int n)
{
	int column = threadIdx.x + (blockIdx.x * blockDim.x);
	int row = threadIdx.y + (blockIdx.y * blockDim.y);
	float c_value;
	
	
	if ((column < n) && (row < n)) {
		c_value = 0;
		for (int k = 0; k < n; ++k) {
			c_value += d_a[(n * k) + row] * d_b[(n * column) + k];
		}

		d_c[(n * column) + row] = c_value;
	}
}


// Host code
int main()
{
    cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);

    const int N = 128;
	const int arraySize = N * N;
	const int arraySizeBytes = arraySize * sizeof(float);

	cudaError_t err;

	float *h_a, *h_b, *h_c, *verify_result;
	float *d_a, *d_b, *d_c;

	// Allocate space for host copies on CPU
	h_a = (float *)malloc(arraySizeBytes);
	h_b = (float *)malloc(arraySizeBytes);
	h_c = (float *)malloc(arraySizeBytes);
	verify_result = (float *)malloc(arraySizeBytes);

	// Allocate space for device copies on GPU
	cudaMalloc((void **)& d_a, arraySizeBytes);
	cudaMalloc((void **)& d_b, arraySizeBytes);
	cudaMalloc((void **)& d_c, arraySizeBytes);

	// Fill 2D host input matrices with random single-precision floating point numbers
	int random_num_a, random_num_b;
	float range = (10.0 - (-10.0)); // from -10 to + 10
	float div = RAND_MAX / range;
	srand(time(NULL));
	
	// Populate h_A and h_B input arrays with numbers
	for(int i = 0; i < N; i++) {
		for(int j = 0; j < N; j++) {
			random_num_a = (-10) + (rand() / div); // float in range -10 to +10
			random_num_b = (-10) + (rand() / div); // float in range -10 to +10

			h_a[(j * N) + i] = random_num_a;
			h_b[(i * N) + j] = random_num_b;			
		}
	}

	
	// Calculate result on CPU
	float result;
	clock_t begin = clock();
	for(int i = 0; i < N; ++i) {
		for(int j = 0; j < N; ++j) {
			result = 0;
			for(int k = 0; k < N; ++k) {
				result += h_a[(N * k) + i] * h_b[(N * j) + k];				
			}
			verify_result[(N * j) + i] = result;
		}
	}
	clock_t end = clock();
	cout << "CPU multiplication time: " << 1000.0 * (double)(end - begin) / (double)CLOCKS_PER_SEC << endl;
	

	// Copy input matrices from host memory to device memory
	cudaEventRecord(start, 0);
	cudaMemcpy(d_a, h_a, arraySizeBytes, cudaMemcpyHostToDevice);
	cudaEventRecord(stop, 0);

	float gpu_time = 0;
	unsigned long int counter = 0;
	while(cudaEventQuery(stop) == cudaErrorNotReady) {
		counter++;
	}

	cudaEventElapsedTime(&gpu_time, start, stop);
	 // print the GPU times
	 printf("Time spent copying 1 NxN matrix to GPU: %.2f\n", gpu_time);

	cudaMemcpy(d_b, h_b, arraySizeBytes, cudaMemcpyHostToDevice);
	
	// Invoke kernel
	int NumBlocks = N / BLOCK_WIDTH;
	if (N % BLOCK_WIDTH) NumBlocks++;

	dim3 dimGrid(NumBlocks, NumBlocks, 1);
	dim3 dimBlock(BLOCK_WIDTH, BLOCK_WIDTH, 1);
	
	cudaEventRecord(start, 0);
	MatrixMult_Device<<<dimGrid, dimBlock>>>(d_a, d_b, d_c, N);
	cudaEventRecord(stop, 0);

	gpu_time = 0;
	counter = 0;
	while(cudaEventQuery(stop) == cudaErrorNotReady) {
		counter++;
	}

	cudaEventElapsedTime(&gpu_time, start, stop);
	// print the GPU times
	printf("Time spent multiplying matrices on GPU: %.2f\n", gpu_time);


	err = cudaThreadSynchronize();
	printf("Run kernel: %s\n", cudaGetErrorString(err));

	// Copy result from device to host
	cudaMemcpy(h_c, d_c, arraySizeBytes, cudaMemcpyDeviceToHost);

	// Free device memory
	cudaFree(d_a);
	cudaFree(d_b);
	cudaFree(d_c);

	// Compare CPU results to GPU results
	bool correct = true;

	for(int i = 0; i < (N * N); i++) {
		if(verify_result[i] != h_c[i]){
			correct = false;
			break;
		}
	}

	if(correct) {
		printf("Verification passed :)");
	}
	else {
		printf("Verification failed (:");
	}

	// Keeps terminal open until user hits 'Return' on terminal
	cin.get();
    return 0;
}
