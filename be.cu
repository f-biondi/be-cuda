#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <chrono>

#define THREAD_N 256

void checkCUDAError(const char*);
int* read_file_graph(int* edge_n, int* node_n);
int* compute_connections(int* edge_index, int edge_n, int node_n, int *max_node_w, int* connections_n, int* connections_sum);
int read_file_int(FILE *file);

__global__ void compute_weights(int *connections, int *connections_n, int *connections_sum, int *node_n, int *weights, int *node_blocks, int *splitters, int *splitters_mask, int *current_splitter_index) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int node_number = (*node_n);
    if(i < node_number) {
        int node_connection_n = connections_n[i];
        int node_connection_sum = connections_sum[i];
        int csi = *current_splitter_index;
        int splitter = splitters[csi];
        for(int k=0; k < node_connection_n; ++k) {
            weights[i] += (node_blocks[connections[node_connection_sum + k]] == splitter);
        }
        splitters_mask[splitter] = 0;
    }
}


__global__ void block_ballot(int *node_blocks, int *max_node_w, int *weights, int *weight_adv, int *node_n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if(i < (*node_n)) {
        int weight = weights[i];
        int leader = node_blocks[i];
        if(i == leader || weight != weights[leader]) {
            int max_w = (*max_node_w);
            long int adv_index = ((long int)max_w) * ((long int)leader) + ((long int)weight);
            weight_adv[adv_index] = i;
        }
    }
}

__global__ void split(int *new_node_blocks, int *node_blocks, int *max_node_w, int *weights, int *weight_adv, int *node_n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if(i < (*node_n)) {
        int max_w = (*max_node_w);
        int block = node_blocks[i];
        int weight = weights[i];
        long int adv_index = ((long int)max_w) * ((long int)block) + ((long int)weight);
        int new_block = weight_adv[adv_index];
        new_node_blocks[i] = new_block;
        weights[i] = 0;
    }
}

__global__ void add_splitters(int *new_node_blocks, int *node_blocks, int *splitters, int *current_splitter_index, int *splitters_mask, int *node_n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if(i < (*node_n)) {
        int new_node_block = new_node_blocks[i];
        int old_node_block = node_blocks[i];
        if(i == new_node_block && new_node_block != old_node_block) {
            int new_splitter_index = atomicAdd(current_splitter_index, 1);
            splitters[new_splitter_index] = i;
            splitters_mask[i] = 1;
            int* splitter_mask_add = &splitters_mask[old_node_block];
            int old_splitted = atomicExch(splitter_mask_add, 1);
            if(!old_splitted) {
                new_splitter_index = atomicAdd(current_splitter_index, 1);
                splitters[new_splitter_index] = old_node_block;
            }
        }
    }
}

int main(void) {
    int node_n = 0;
    int edge_n = 0;
    int max_node_w = 0;
    int* edge_index = read_file_graph(&edge_n, &node_n);
    size_t node_size = node_n * sizeof(int);
    size_t edge_size = edge_n * sizeof(int);
    int* connections_n = (int*)calloc(node_n, sizeof(int));
    int* connections_sum = (int*)calloc(node_n, sizeof(int));
    int* connections = compute_connections(edge_index, edge_n, node_n, &max_node_w, connections_n, connections_sum);
    int* current_splitter_index = (int *)calloc(1, sizeof(int));

    int *d_node_n, *d_new_node_blocks, *d_node_blocks, *d_current_splitter_index, *d_max_node_w,
        *d_weights, *d_splitters, *d_splitters_mask, *d_weight_adv, *d_connections, *d_connections_n,
        *d_connections_sum, *d_swap;

    cudaMalloc((void **)&d_weights, node_size);
    cudaMalloc((void **)&d_weight_adv, node_size * max_node_w);
    cudaMalloc((void **)&d_connections, edge_size);
    cudaMalloc((void **)&d_connections_n, node_size);
    cudaMalloc((void **)&d_connections_sum, node_size);
    cudaMalloc((void **)&d_node_n, sizeof(int));
    cudaMalloc((void **)&d_new_node_blocks, node_size);
    cudaMalloc((void **)&d_node_blocks, node_size);
    cudaMalloc((void **)&d_splitters, node_size);
    cudaMalloc((void **)&d_splitters_mask, node_size);
    cudaMalloc((void **)&d_current_splitter_index, sizeof(int));
    cudaMalloc((void **)&d_max_node_w, sizeof(int));
    checkCUDAError("CUDA malloc");

    cudaMemset(d_weights,0, node_size);
    cudaMemset(d_new_node_blocks,0, node_size);
    cudaMemset(d_node_blocks,0, node_size);
    cudaMemset(d_splitters,0, node_size);
    cudaMemset(d_splitters_mask,0, node_size);
    cudaMemset(d_current_splitter_index,0, sizeof(int));
    checkCUDAError("CUDA memset");

    cudaMemcpy(d_connections, connections, edge_size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_connections_n, connections_n, node_size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_connections_sum, connections_sum, node_size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_node_n, &node_n, sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_max_node_w, &max_node_w, sizeof(int), cudaMemcpyHostToDevice);
    checkCUDAError("CUDA memcpy 1");

    while((*current_splitter_index) >= 0) {
        compute_weights<<<(node_n+(THREAD_N-1)) / THREAD_N, THREAD_N>>>(
                d_connections,
                d_connections_n,
                d_connections_sum,
                d_node_n,
                d_weights,
                d_node_blocks,
                d_splitters,
                d_splitters_mask,
                d_current_splitter_index
        );
        checkCUDAError("Compute Weights");

        block_ballot<<<(node_n+(THREAD_N-1)) / THREAD_N, THREAD_N>>>(d_node_blocks, d_max_node_w, d_weights, d_weight_adv, d_node_n);
        checkCUDAError("Block ballot");

        split<<<(node_n+(THREAD_N-1)) / THREAD_N, THREAD_N>>>(d_new_node_blocks, d_node_blocks, d_max_node_w, d_weights, d_weight_adv, d_node_n);
        checkCUDAError("Split");

        add_splitters<<<(node_n+(THREAD_N-1)) / THREAD_N, THREAD_N>>>(d_new_node_blocks, d_node_blocks, d_splitters, d_current_splitter_index, d_splitters_mask, d_node_n);
        checkCUDAError("Add splitters");

    	cudaDeviceSynchronize();
        cudaMemcpy(current_splitter_index, d_current_splitter_index, sizeof(int), cudaMemcpyDeviceToHost);
        checkCUDAError("CUDA memcpy 3");
        *current_splitter_index = (*current_splitter_index) - 1;
        cudaMemcpy(d_current_splitter_index, current_splitter_index, sizeof(int), cudaMemcpyHostToDevice);
        checkCUDAError("CUDA memcpy 4");

        d_swap = d_node_blocks;
        d_node_blocks = d_new_node_blocks;
        d_new_node_blocks = d_swap;
    }
    cudaDeviceSynchronize();

    int *result = (int*)malloc(node_n * sizeof(int));
    cudaMemcpy(result, d_node_blocks, node_size, cudaMemcpyDeviceToHost);
    checkCUDAError("CUDA memcpy 5");

    printf("[%d", result[0]);
    for (int i=1; i<node_n; ++i) {
        printf(",%d", result[i]);
    }
    printf("]");

    return 0;
}

int* read_file_graph(int* edge_n, int* node_n) {
    FILE *file = fopen("graph.txt", "r");
    *node_n = read_file_int(file);
    *edge_n = read_file_int(file);
    int index_size = (*edge_n) * 2 * sizeof(int);
    int *edge_index = (int*)malloc(index_size);
    for(int i=0; i<(*edge_n); ++i) {
        edge_index[i] = read_file_int(file); 
        edge_index[(*edge_n) + i] = read_file_int(file); 
    }
    return edge_index;
}

int read_file_int(FILE *file) {
    char ch = fgetc(file);
    int n = 0;
    int c = 0;
    while(ch != ' ' && ch != '\n') {
        c = ch - '0';   
        n = (n*10) + c;
        ch = fgetc(file);
    }
    return n;
}

int* compute_connections(int* edge_index, int edge_n, int node_n, int *max_node_w, int* connections_n, int* connections_sum) {
    for(int i=0; i<edge_n; ++i) {
        //int node = edge_index[edge_n + i];
        int node = edge_index[i];
        connections_n[node]++;
        if(connections_n[node] > (*max_node_w)) {
            *max_node_w = connections_n[node];
        }
    }

    for(int i=1; i<node_n; ++i) {
        connections_sum[i] = connections_sum[i-1] + connections_n[i-1];
    }

    int* connections = (int*)malloc(edge_n * sizeof(int));
    int* connections_cur = (int*)malloc(node_n * sizeof(int));

    for(int i=0; i<edge_n; ++i) {
        //int node = edge_index[edge_n + i];
        //connections[connections_sum[node] + connections_cur[node]] = edge_index[i];
        int node = edge_index[i];
        connections[connections_sum[node] + connections_cur[node]] = edge_index[edge_n + i];
        connections_cur[node]++;
    }

    free(connections_cur);
    return connections;
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

