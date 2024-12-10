#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#define N 11
#define EDGES 20

void checkCUDAError(const char*);

__global__ void compute_weights(int *edge_start, int *edge_end, int *weights, int *node_blocks, int *splitters, int *current_splitter_index) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
    if(i < EDGES) {
        atomicAdd(
            &weights[edge_end[i]],
            node_blocks[edge_start[i]] == splitters[*current_splitter_index]
        );
    }
}

int main(void) {
    int edge_index[2][20] = {
        {0,1,2,3,0,0,1,4,1,5,1,6,2,7,2,8,3,9,3,10},
        {1,0,0,0,2,3,4,1,5,1,6,1,7,2,8,2,9,3,10,3}
    };
    int *weights, *current_splitter_index;
	int *d_edge_start, *d_edge_end, *d_node_blocks, *d_weights,
        *d_splitters, *d_current_splitter_index, *d_splitters_mask;
	unsigned int nodeSize = N * sizeof(int);
	unsigned int edgeSize = EDGES * sizeof(int);

	weights = (int *)malloc(nodeSize);
	current_splitter_index = (int *)malloc(sizeof(int));
    *current_splitter_index = 0;

	cudaMalloc((void **)&d_edge_start, edgeSize);
	cudaMalloc((void **)&d_edge_end, edgeSize);
	cudaMalloc((void **)&d_node_blocks, nodeSize);
	cudaMalloc((void **)&d_weights, nodeSize);
	cudaMalloc((void **)&d_splitters, nodeSize);
	cudaMalloc((void **)&d_splitters_mask, nodeSize);
	cudaMalloc((void **)&d_current_splitter_index, sizeof(int));
	checkCUDAError("CUDA malloc");

	cudaMemcpy(d_edge_start, edge_index[0], edgeSize, cudaMemcpyHostToDevice);
	cudaMemcpy(d_edge_end, edge_index[1], edgeSize, cudaMemcpyHostToDevice);
	cudaMemcpy(d_current_splitter_index, current_splitter_index, sizeof(int), cudaMemcpyHostToDevice);
	checkCUDAError("CUDA memcpy");


    while(*current_splitter_index >= 0) {
        cudaMemcpy(current_splitter_index, d_current_splitter_index, sizeof(int), cudaMemcpyDeviceToHost);
        *current_splitter_index = (*current_splitter_index) - 1;
        cudaMemcpy(d_current_splitter_index, current_splitter_index, sizeof(int), cudaMemcpyHostToDevice);
        checkCUDAError("CUDA memcpy");

        compute_weights<<<(EDGES+255)/256, 256>>>(d_edge_start, d_edge_end, d_weights, d_node_blocks, d_splitters, d_current_splitter_index);
        checkCUDAError("Compute Weights");

        cudaMemcpy(weights, d_weights, nodeSize, cudaMemcpyDeviceToHost);
        checkCUDAError("CUDA memcpy");

        for (int i =0; i< N; ++i){
            printf("%d \n", weights[i]);
        }
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
