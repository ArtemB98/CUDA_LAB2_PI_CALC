
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <curand.h>
#include <curand_kernel.h>
#include <stdio.h>
#include <iostream>
#include <stdio.h>
#include <ctime>

using namespace std;

__global__ void init(unsigned long seed, curandState* states) {
	int tid = threadIdx.x + blockIdx.x * blockDim.x;
	curand_init(seed, tid, 0, &states[tid]);
}

__global__ void randoms(curandState* globalState, float* randomArray)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	curandState localState = globalState[idx];
	float RANDOM = curand_uniform(&localState);
	randomArray[idx] = RANDOM;
	globalState[idx] = localState;
}

__global__ void pi_calc(float* dev_randX, float* dev_randY, int* dev_blocks_num, int n, int bs) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    __shared__ int count_pi[512];
    int cont = 0;
    for (int i = tid * bs; i < bs * (tid + 1); i++) {
        if (dev_randX[i] * dev_randX[i] + dev_randY[i] * dev_randY[i] < 1.0f) {
            cont++;
        }
    }
    count_pi[threadIdx.x] = cont;
    __syncthreads();
    if (threadIdx.x == 0) {
        int total = 0;
        for (int j = 0; j < 512; j++) {
            total += count_pi[j];
        }
        dev_blocks_num[blockIdx.x] = total;
    }
}

int main() {
	srand((unsigned)time(NULL));
	clock_t c_start, c_end;
	int n = 0;
	cout << "Enter number of points: ";
	cin >> n;
	curandState* states;
	cudaDeviceProp devProp;
	cudaGetDeviceProperties(&devProp, 0);
	int dimm = devProp.maxThreadsDim[0];
	dim3 threads = dim3(dimm / 2, 1);
	int blocksCount = floor(n / threads.x) + 1;
	dim3 blocks = dim3(blocksCount, 1);
	cudaMalloc((void**)&states, n * sizeof(curandState));
	init << <blocks, threads >> > (time(NULL), states);
	float* randX = new float[n];
	float* dev_randX;
	float* randY = new float[n];
	float* dev_randY;
	cudaMalloc((void**)&dev_randX, n * sizeof(float));
	randoms << <blocks, threads >> > (states, dev_randX);
	cudaMalloc((void**)&dev_randY, n * sizeof(float));
	randoms << <blocks, threads >> > (states, dev_randY);
	cudaMemcpy(randX, dev_randX, n * sizeof(float), cudaMemcpyDeviceToHost);
	cudaMemcpy(randY, dev_randY, n * sizeof(float), cudaMemcpyDeviceToHost);
	c_start = clock();
	int c_count = 0;
	//CPU calculation
	for (int i = 0; i < n; i++) {
		if (randX[i] * randX[i] + randY[i] * randY[i] < 1.0f) {
			c_count++;
		}
	}
	c_end = clock();
	float cpu_time = (float)(c_end - c_start) / CLOCKS_PER_SEC;
	float cpu_res = float(c_count) * 4.0 / n;
	cout << "CPU result: " << cpu_res << endl;
	cout << "CPU time: " << cpu_time * 1000 << " ms" << endl;
	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	cudaEventRecord(start, 0);
	int threadsPerBlock = 512;
	int block_num = n / (128 * threadsPerBlock);
	int* dev_blocks_num;
	cudaMalloc((void**)&dev_blocks_num, block_num * sizeof(int));
	pi_calc << <block_num, threadsPerBlock >> > (dev_randX, dev_randY, dev_blocks_num,n, 128);
	int* blocks_num = new int[block_num];
	cudaMemcpy(blocks_num, dev_blocks_num, block_num * sizeof(int), cudaMemcpyDeviceToHost);
	int g_count = 0;
	for (int i = 0; i < block_num; i++) {
		g_count += blocks_num[i];
	};
	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	float gpu_time;
	cudaEventElapsedTime(&gpu_time, start, stop);
	cudaEventDestroy(start);
	cudaEventDestroy(stop);
	float gpu_res = float(g_count) * 4.0 / n;
	cout << "GPU result: " << gpu_res << endl;
	cout << "GPU time: " << gpu_time << " ms" << endl;
	cudaFree(states);
	cudaFree(dev_randY);
	cudaFree(dev_randX);
	delete[] randX;
	delete[] randY;
}