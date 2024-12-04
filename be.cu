#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#define N 2050
#define THREADS_PER_BLOCK 128

void checkCUDAError(const char*);
void random_ints(int *a);
void vectorAddCPU(int* a, int* b, int* c, int max);
void validate(int* c, int* c_ref, int max);
void index_to_matrix(int *m, int n_nodes, int[][] edge_index, int l_index);

__global__ void vectorAdd(int *a, int *b, int *c, int max) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	c[i] = a[i] + b[i];
}

void vectorAddCPU (int *a, int *b, int *c, int max) {
	for(int i=0; i<max; ++i) {
			 c[i] = a[i] + b[i];
	}
}

void validate(int* c, int* c_ref, int max) {
		 int errors = 0;
		 for(int i=0; i<max; ++i) {
					if(c[i] != c_ref[i]) {
							 printf("Error at %d: %d != %d \n", i, c[i], c_ref[i]);
							 errors++;
					}
		 }
		 printf("Total errors: %d \n", errors);
}

int main(void) {
	int *a, *b, *c, *c_ref;			// host copies of a, b, c
	int *d_a, *d_b, *d_c;			// device copies of a, b, c
	unsigned int size = N * sizeof(int);

    int edge_index[2][20] = {
        {0,1,2,3,0,0,1,4,1,5,1,6,2,7,2,8,3,9,3,10},
        {1,0,0,0,2,3,4,1,5,1,6,1,7,2,8,2,9,3,10,3}
    };
    int n_nodes = 11;
    int *matrix = (int *)malloc(n_nodes * n_nodes * sizeof(int));
    index_to_matrix(matrix, n_nodes, edge_index, 20);
    for(int i=0; i<n_nodes; ++i) {
        for(int k=0; k<n_nodes; ++k) {
            printf(" %d", matrix[(i*n_nodes)+k]);
        }
        printf("\n");
    }

    /*
	// Alloc space for device copies of a, b, c
	cudaMalloc((void **)&d_a, size);
	cudaMalloc((void **)&d_b, size);
	cudaMalloc((void **)&d_c, size);
	checkCUDAError("CUDA malloc");

	// Alloc space for host copies of a, b, c and setup input values
	a = (int *)malloc(size); random_ints(a);
	b = (int *)malloc(size); random_ints(b);
	c = (int *)malloc(size);
	c_ref = (int *)malloc(size);

	// Copy inputs to device
	cudaMemcpy(d_a, a, size, cudaMemcpyHostToDevice);
	cudaMemcpy(d_b, b, size, cudaMemcpyHostToDevice);
	checkCUDAError("CUDA memcpy");

	// Launch add() kernel on GPU
    dim3 nBlocks(ceil((float)N / THREADS_PER_BLOCK), 1, 1);
    dim3 nThreads(THREADS_PER_BLOCK, 1, 1);
	vectorAdd <<< nBlocks, nThreads >>>(d_a, d_b, d_c, N);
	checkCUDAError("CUDA kernel");

	// Copy result back to host
	cudaMemcpy(c, d_c, size, cudaMemcpyDeviceToHost);
	checkCUDAError("CUDA memcpy");

	vectorAddCPU(a, b, c_ref, N);
	validate(c, c_ref, N);

	// Cleanup
	free(a); free(b); free(c);
	cudaFree(d_a); cudaFree(d_b); cudaFree(d_c);
	checkCUDAError("CUDA cleanup");
    */

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

void random_ints(int *a)
{
	for (unsigned int i = 0; i < N; i++){
		a[i] = rand();
	}
}

void index_to_matrix(int *m, int n_nodes, int[][] edge_index, int l_index)
{
    for(int i=0; i< l_index; ++i) {
        m[(edge_index[0][i] * n_nodes) + edge_index[1][i]] = 1;
        m[(edge_index[1][i] * n_nodes) + edge_index[0][i]] = 1;
    }
}
