#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <chrono>

#define THREAD_N 256

void checkCUDAError(const char*);
int* read_file_graph(int* edge_n, int* node_n);
int read_file_int(FILE *file);
void skip_lines(FILE *file, int n);
void skip_chars(FILE *file, int n);

__global__ void compute_weights(int *edge_start, int *edge_end, int *edge_n, int *weights, int *node_blocks, int *splitters, int *splitters_mask, int *current_splitter_index) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if(i < (*edge_n)) {
        int es = edge_start[i];
        int block = node_blocks[es];
        int splitter = splitters[*current_splitter_index];
        atomicAdd(
            &weights[edge_end[i]],
            block == splitter
        );
        splitters_mask[splitters[*current_splitter_index]] = 0;
    }
}

__global__ void compute_max_node_w(int *weights, int *max_node_w, int *node_n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if(i < (*node_n)) {
        atomicMax(max_node_w, weights[i]);
    }
}

__global__ void init_ballot(int *node_blocks, int *max_node_w, int *weights, int *weight_adv, int *node_n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if(i < (*node_n)) {
        weight_adv[(*max_node_w) * node_blocks[i] + weights[i]] = (*node_n);
    }
}

__global__ void block_ballot(int *node_blocks, int *max_node_w, int *weights, int *weight_adv, int *node_n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if(i < (*node_n)) {
        atomicMin(
            &weight_adv[(*max_node_w) * node_blocks[i] + weights[i]],
            i
        );
    }
}

__global__ void split(int *new_node_blocks, int *node_blocks, int *max_node_w, int *weights, int *weight_adv, int *node_n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if(i < (*node_n)) {
        new_node_blocks[i] = weight_adv[(*max_node_w) * node_blocks[i] + weights[i]];
        weights[i] = 0;
    }
}

__global__ void add_splitters(int *new_node_blocks, int *node_blocks, int *splitters, int *current_splitter_index, int *splitters_mask, int *node_n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if(i < (*node_n) && i == new_node_blocks[i] && new_node_blocks[i] != node_blocks[i]) {
        int new_splitter_index = atomicAdd(current_splitter_index, 1);
        splitters[new_splitter_index] = i;
        splitters_mask[i] = 1;
        int old_block = node_blocks[i];
        int old_splitted = atomicExch(&splitters_mask[old_block], 1);
        if(!old_splitted) {
            new_splitter_index = atomicAdd(current_splitter_index, 1);
            splitters[new_splitter_index] = old_block;
        }
    }
}

int main(void) {
    int node_n = 0;
    int edge_n = 0;
    int* edge_index = read_file_graph(&edge_n, &node_n);

    int *d_edge_start, *d_edge_end, *d_edge_n, *d_node_n, *d_new_node_blocks,
        *d_node_blocks,*d_current_splitter_index, *d_max_node_w, *d_weights, *d_splitters,
        *d_splitters_mask,*d_weight_adv, *d_swap;

    unsigned int node_size = node_n * sizeof(int);
    unsigned int edge_size = edge_n * sizeof(int);

    int *current_splitter_index = (int *)calloc(sizeof(int),0);
    int *max_node_w = (int *)calloc(sizeof(int), 0);
    int init = 0;

    cudaMalloc((void **)&d_weights, node_size);
    cudaMalloc((void **)&d_edge_n, sizeof(int));
    cudaMalloc((void **)&d_node_n, sizeof(int));
    cudaMalloc((void **)&d_edge_start, edge_size);
    cudaMalloc((void **)&d_edge_end, edge_size);
    cudaMalloc((void **)&d_new_node_blocks, node_size);
    cudaMalloc((void **)&d_node_blocks, node_size);
    cudaMalloc((void **)&d_splitters, node_size);
    cudaMalloc((void **)&d_splitters_mask, node_size);
    cudaMalloc((void **)&d_current_splitter_index, sizeof(int));
    cudaMalloc((void **)&d_max_node_w, sizeof(int));
    checkCUDAError("CUDA malloc");

    cudaMemset(d_weights,0, node_size);
    cudaMemset(d_edge_n,0, sizeof(int));
    cudaMemset(d_node_n,0, sizeof(int));
    cudaMemset(d_edge_start,0, edge_size);
    cudaMemset(d_edge_end,0, edge_size);
    cudaMemset(d_new_node_blocks,0, node_size);
    cudaMemset(d_node_blocks,0, node_size);
    cudaMemset(d_splitters,0, node_size);
    cudaMemset(d_splitters_mask,0, node_size);
    cudaMemset(d_current_splitter_index,0, sizeof(int));
    cudaMemset(d_max_node_w,0, sizeof(int));
    checkCUDAError("CUDA memset");

    cudaMemcpy(d_edge_start, edge_index, edge_size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_edge_end, &edge_index[edge_n], edge_size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_edge_n, &edge_n, sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_node_n, &node_n, sizeof(int), cudaMemcpyHostToDevice);
    checkCUDAError("CUDA memcpy 1");


    std::chrono::steady_clock::time_point start = std::chrono::steady_clock::now();
    while((*current_splitter_index) >= 0) {
        compute_weights<<<(edge_n+(THREAD_N-1)) / THREAD_N, THREAD_N>>>(
                d_edge_start,
                d_edge_end,
                d_edge_n,
                d_weights,
                d_node_blocks,
                d_splitters,
                d_splitters_mask,
                d_current_splitter_index
        );
        checkCUDAError("Compute Weights");

       if(!init) {
            compute_max_node_w<<<(node_n+(THREAD_N-1)) / THREAD_N, THREAD_N>>>(d_weights, d_max_node_w, d_node_n);
            checkCUDAError("Computing Max Weight");
            cudaMemcpy(max_node_w, d_max_node_w, sizeof(int), cudaMemcpyDeviceToHost);
            checkCUDAError("CUDA memcpy 2");
            cudaMalloc((void **)&d_weight_adv, (node_n*((*max_node_w)+1))*sizeof(int));
            checkCUDAError("CUDA malloc");
            init = 1;
        }

        init_ballot<<<(node_n+(THREAD_N-1)) / THREAD_N, THREAD_N>>>(d_node_blocks, d_max_node_w, d_weights, d_weight_adv, d_node_n);
        checkCUDAError("Init advert");

        block_ballot<<<(node_n+(THREAD_N-1)) / THREAD_N, THREAD_N>>>(d_node_blocks, d_max_node_w, d_weights, d_weight_adv, d_node_n);
        checkCUDAError("Block ballot");

        split<<<(node_n+(THREAD_N-1)) / THREAD_N, THREAD_N>>>(d_new_node_blocks, d_node_blocks, d_max_node_w, d_weights, d_weight_adv, d_node_n);
        checkCUDAError("Split");

        add_splitters<<<(node_n+(THREAD_N-1)) / THREAD_N, THREAD_N>>>(d_new_node_blocks, d_node_blocks, d_splitters, d_current_splitter_index, d_splitters_mask, d_node_n);
        checkCUDAError("Add splitters");

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
    std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
    int time = std::chrono::duration_cast<std::chrono::seconds>(end - start).count();
    printf("%d\n",time);

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
    skip_lines(file, 2);
    skip_chars(file, 9);
    *node_n = read_file_int(file);
    skip_chars(file, 7);
    *edge_n = read_file_int(file);
    int *edge_index = (int*)malloc(sizeof(int) * (*edge_n) * 2);
    skip_lines(file,1);
    for(int i=0; i<(*edge_n); ++i) {
        edge_index[i] = read_file_int(file); 
        edge_index[(*edge_n) + i] = read_file_int(file); 
    }
    fclose(file);
    return edge_index;
}

 int read_file_int(FILE *file) {
    int ch = fgetc(file);
    int n = 0;
    int c = 0;
    while(ch != EOF && ch != '\t' && ch != ' '  && ch != '\n' && ch !='\r') {
        c = ch - '0';   
        n = (n*10) + c;
        ch = fgetc(file);
    }
    if (ch == '\r') {
        skip_lines(file, 1);
    }
    return n;
}

void skip_lines(FILE *file, int n) {
    int counter = 0;
    char ch = fgetc(file);
    while(ch != EOF) {
        if(ch == '\n') {
            counter++;
            if(counter == n) {
                break;
            }
        }
        ch = fgetc(file);
    }
}

void skip_chars(FILE *file, int n) {
    int counter = 0;
    char ch = fgetc(file);
    counter++;
    while(ch != EOF && counter < n) {
        ch = fgetc(file);
        counter++;
    }
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
