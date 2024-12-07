#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#define N 11

void checkCUDAError(const char* msg);
void getMatrix(int* m, int n_nodes);

int main(void) {
	int *matrix;			                          // host copies
	int *d_matrix, *d_nodes, d_weights;			      // device copies
	unsigned int matrixSize = N * N * sizeof(int);
	unsigned int nodeSize = N * sizeof(int);

    matrix = (int*)malloc(matrixSize);
    getMatrix(matrix, N);

	cudaMalloc((void **)&d_matrix, matrixSize);
	cudaMalloc((void **)&d_nodes, nodeSize);
	cudaMalloc((void **)&d_weights, nodeSize);
	checkCUDAError("CUDA malloc");

	cudaMemcpy(d_matrix, matrix, matrixSize, cudaMemcpyHostToDevice);
	checkCUDAError("CUDA memcpy");



	return 0;
}

void checkCudaError(const char*)
{
	cudaError_t err = cudaGetLastError();
	if (cudaSuccess != err)
	{
		fprintf(stderr, "CUDA ERROR: %s: %s.\n", msg, cudaGetErrorString(err));
		exit(EXIT_FAILURE);
	}
}

void getMatrix(int* m, int n_nodes)
{
    int edge_index[2][20] = {
        {0,1,2,3,0,0,1,4,1,5,1,6,2,7,2,8,3,9,3,10},
        {1,0,0,0,2,3,4,1,5,1,6,1,7,2,8,2,9,3,10,3}
    };
    int l_index = 20;

    for(int i=0; i< l_index; ++i) {
        m[(edge_index[0][i] * n_nodes) + edge_index[1][i]] = 1;
        m[(edge_index[1][i] * n_nodes) + edge_index[0][i]] = 1;
    }
}
