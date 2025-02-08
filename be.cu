#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <cuda_fp16.h>
#include <cuda.h>
#define THREAD_N 256

#define CHECK_CUDA(func)                                                       \
{                                                                              \
    cudaError_t status = (func);                                               \
    if (status != cudaSuccess) {                                               \
        printf("CUDA API failed at line %d with error: %s (%d)\n",             \
               __LINE__, cudaGetErrorString(status), status);                  \
        return EXIT_FAILURE;                                                   \
    }                                                                          \
}

typedef struct {
    int* node_blocks;
    int* splitters;
    int current_splitter_index;
} partition_t;

int* read_file_graph(unsigned int* edge_n, unsigned int* node_n, int* max_node_w);
partition_t read_file_partition(int node_n);
int read_file_int(FILE *file);


__global__ void compute_weights(unsigned int* edge_n, int* edge_start, int* edge_end, int* node_blocks, int* splitters, int* current_splitter_index, __half* weights) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if(i < (*edge_n)) {
        int csi = *current_splitter_index;
        int splitter = splitters[csi];
        int node = edge_end[i];
        int block = node_blocks[node];
        int s = edge_start[i];

        if(block == splitter) {
            atomicAdd(weights + s, 1);
        }
    }
}

__global__ void block_ballot(int* node_blocks, int* max_node_w, __half* weights, int* weight_adv, unsigned int* node_n) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if(i < (*node_n)) {
        int weight = weights[i];
        int leader = node_blocks[i];
        if(i != leader && weight != ((int)weights[leader])) {
            long int adv_index = ((long int)*max_node_w) * leader + weight;
            weight_adv[adv_index] = i;
        }
    }
}

__global__ void split(int* new_node_blocks, int* node_blocks, int* max_node_w, __half* weights, int* weight_adv, unsigned int* node_n) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if(i < (*node_n)) {
        int block = node_blocks[i];
        int weight = weights[i];
        int new_block = block;
        if(i != block && weight != ((int)weights[block])) {
            long int adv_index = ((long int)*max_node_w) * block + weight;
            new_block = weight_adv[adv_index];
        }
        new_node_blocks[i] = new_block;
    }
}

__global__ void add_splitters(int* new_node_blocks, __half* weights, int* node_blocks, int* splitters, int* current_splitter_index, unsigned int* node_n) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if(i < (*node_n)) {
        weights[i] = 0;
        int new_node_block = new_node_blocks[i];
        int old_node_block = node_blocks[i];
        if(i == new_node_block && new_node_block != old_node_block) {
            int new_splitter_index = atomicAdd(current_splitter_index, 1);
            splitters[new_splitter_index] = i;
        }
    }
}

int main(int argc, char **argv) {
    unsigned int node_n = 0;
    unsigned int edge_n = 0;
    int max_node_w = 0;
    int* edge_index = read_file_graph(&edge_n, &node_n, &max_node_w);
    partition_t partition = read_file_partition(node_n);
    const int NODE_BLOCK_N = (node_n+(THREAD_N-1)) / THREAD_N;
    const int EDGE_BLOCK_N = (edge_n+(THREAD_N-1)) / THREAD_N;

    int  *d_new_node_blocks, *d_node_blocks, *d_current_splitter_index,
        *d_max_node_w, *d_splitters, *d_weight_adv, *d_swap, *d_edge_start,
        *d_edge_end;
    unsigned int *d_node_n, *d_edge_n;

    __half *d_weights;

    CHECK_CUDA( cudaMalloc((void **)&d_weights, node_n * sizeof(__half)) );
    CHECK_CUDA( cudaMalloc((void **)&d_edge_start, ((size_t)edge_n) * sizeof(int)) );
    CHECK_CUDA( cudaMalloc((void **)&d_edge_end, ((size_t)edge_n) * sizeof(int)) );
    CHECK_CUDA( cudaMalloc((void **)&d_weight_adv, node_n * sizeof(int) * max_node_w) );
    CHECK_CUDA( cudaMalloc((void **)&d_node_n, sizeof(unsigned int)) );
    CHECK_CUDA( cudaMalloc((void **)&d_edge_n, sizeof(unsigned int)) );
    CHECK_CUDA( cudaMalloc((void **)&d_new_node_blocks, node_n * sizeof(int)) );
    CHECK_CUDA( cudaMalloc((void **)&d_node_blocks, node_n * sizeof(int)) );
    CHECK_CUDA( cudaMalloc((void **)&d_splitters, node_n * sizeof(int)) );
    CHECK_CUDA( cudaMallocHost((void **)&d_current_splitter_index, sizeof(int)) );
    CHECK_CUDA( cudaMalloc((void **)&d_max_node_w, sizeof(int)) );

    CHECK_CUDA( cudaMemset(d_weights, 0, node_n * sizeof(__half)) );
    CHECK_CUDA( cudaMemset(d_weight_adv, 0, max_node_w * node_n * sizeof(int)) );

    CHECK_CUDA( cudaMemcpy(d_node_n, &node_n, sizeof(unsigned int), cudaMemcpyHostToDevice) );
    CHECK_CUDA( cudaMemcpy(d_edge_n, &edge_n, sizeof(unsigned int), cudaMemcpyHostToDevice) );
    CHECK_CUDA( cudaMemcpy(d_edge_start, edge_index, ((size_t)edge_n) * sizeof(int), cudaMemcpyHostToDevice) );
    CHECK_CUDA( cudaMemcpy(d_edge_end, edge_index + edge_n, ((size_t)edge_n) * sizeof(int), cudaMemcpyHostToDevice) );
    CHECK_CUDA( cudaMemcpy(d_node_blocks, partition.node_blocks, node_n * sizeof(int), cudaMemcpyHostToDevice) );
    CHECK_CUDA( cudaMemcpy(d_new_node_blocks, partition.node_blocks, node_n * sizeof(int), cudaMemcpyHostToDevice) );
    CHECK_CUDA( cudaMemcpy(d_splitters, partition.splitters, node_n * sizeof(int), cudaMemcpyHostToDevice) );
    CHECK_CUDA( cudaMemcpy(d_max_node_w, &max_node_w, sizeof(int), cudaMemcpyHostToDevice) );
    CHECK_CUDA( cudaMemcpy(d_current_splitter_index, &partition.current_splitter_index, sizeof(int), cudaMemcpyHostToDevice) );

    while(partition.current_splitter_index >= 0) {
        compute_weights<<<EDGE_BLOCK_N, THREAD_N>>>(d_edge_n, d_edge_start, d_edge_end, d_node_blocks, d_splitters, d_current_splitter_index, d_weights);

        block_ballot<<<NODE_BLOCK_N, THREAD_N>>>(d_node_blocks, d_max_node_w, d_weights, d_weight_adv, d_node_n);

        split<<<NODE_BLOCK_N, THREAD_N>>>(d_new_node_blocks, d_node_blocks, d_max_node_w, d_weights, d_weight_adv, d_node_n);

        add_splitters<<<NODE_BLOCK_N, THREAD_N>>>(d_new_node_blocks, d_weights, d_node_blocks, d_splitters, d_current_splitter_index, d_node_n);
        CHECK_CUDA( cudaMemcpy(&partition.current_splitter_index, d_current_splitter_index, sizeof(int), cudaMemcpyDeviceToHost) );
        partition.current_splitter_index--;
        CHECK_CUDA( cudaMemcpy(d_current_splitter_index, &partition.current_splitter_index, sizeof(int), cudaMemcpyHostToDevice) );

        d_swap = d_node_blocks;
        d_node_blocks = d_new_node_blocks;
        d_new_node_blocks = d_swap;
    }
    //cudaDeviceSynchronize(); -- Device already synchronized on cudaMemcpy
    CHECK_CUDA( cudaMemcpy(partition.node_blocks, d_node_blocks, node_n * sizeof(int), cudaMemcpyDeviceToHost) );
    printf("[%d", partition.node_blocks[0]);
    for (int i=1; i<node_n; ++i) {
        printf(",%d", partition.node_blocks[i]);
    }
    printf("]");
    return 0;
}

int* read_file_graph(unsigned int* edge_n, unsigned int* node_n, int* max_node_w) {
    FILE *file = fopen("graph.txt", "r");
    *node_n = read_file_int(file);
    *edge_n = read_file_int(file);
    int* weights = (int*)calloc(*node_n, sizeof(int));
    size_t index_size = ((size_t)*edge_n) * 2 * sizeof(int);
    int *edge_index = (int*)malloc(index_size);
    for(unsigned int i=0; i<(*edge_n); ++i) {
        edge_index[i] = read_file_int(file);
        edge_index[(*edge_n) + i] = read_file_int(file);
        weights[edge_index[i]]++;
        if(weights[edge_index[i]] > *max_node_w) {
            *max_node_w = weights[edge_index[i]];
        }
    }
    fclose(file);
    free(weights);
    return edge_index;
}

partition_t read_file_partition(int node_n) {
    FILE *file = fopen("partition.txt", "r");
    partition_t partition;
    partition.node_blocks = (int*)malloc(sizeof(int) * node_n);
    partition.splitters = (int*)malloc(sizeof(int) * node_n);
    partition.current_splitter_index = -1;

    for(int i=0; i<node_n; ++i) {
        int nb = read_file_int(file);
        if(nb == i) {
            partition.splitters[++partition.current_splitter_index] = nb;
        }
        partition.node_blocks[i] = nb;
    }
    fclose(file);
    return partition;
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
