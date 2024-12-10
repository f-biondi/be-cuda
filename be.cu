#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#define N 11
#define EDGES 20

void checkCUDAError(const char*);

__global__ void compute_weights(int *d_index_a, int *d_index_b, int *d_weights) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
    if(i < EDGES) {
        atomicAdd(&d_index_b[i], 1);
    }
}

int main(void) {
    int edge_index[2][20] = {
        {0,1,2,3,0,0,1,4,1,5,1,6,2,7,2,8,3,9,3,10},
        {1,0,0,0,2,3,4,1,5,1,6,1,7,2,8,2,9,3,10,3}
    };
    int* weights;
	int *d_index_a, *d_index_b, *d_nodes, *d_weights;
	unsigned int nodeSize = N * sizeof(int);
	unsigned int edgeSize = EDGES * sizeof(int);

	weights = (int *)malloc(nodeSize);
	cudaMalloc((void **)&d_index_a, edgeSize);
	cudaMalloc((void **)&d_index_b, edgeSize);
	cudaMalloc((void **)&d_nodes, nodeSize);
	cudaMalloc((void **)&d_weights, nodeSize);
	checkCUDAError("CUDA malloc");

	cudaMemcpy(d_index_a, edge_index[0], edgeSize, cudaMemcpyHostToDevice);
	cudaMemcpy(d_index_b, edge_index[1], edgeSize, cudaMemcpyHostToDevice);
	checkCUDAError("CUDA memcpy");

	compute_weights<<<(EDGES+255)/256, 256>>>(d_index_a, d_index_b, d_weights);
	checkCUDAError("Kernel start");

	cudaMemcpy(weights, d_weights, nodeSize, cudaMemcpyDeviceToHost);
	checkCUDAError("CUDA memcpy");

    for (int i =0; i< N; ++i){
        printf("%d \n", weights[i]);
    }
	return 0;
}

void checkCUDAError(const char *msg)
{
	cudaError_t err = cudaGetLastError();
	if (cudaSuccess != err)
	{
		fprintf(stderr, "CUDA ERROR: %s: %s.\n", msg, cudaGetErrorString(err));
		exit(EXIT_FAILURE);
	}
}
