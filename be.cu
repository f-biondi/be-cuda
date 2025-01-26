#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <cuda_fp16.h>
#include <cuda.h>
#include <cusparse.h>

#define WEIGHT_THREAD_N 128
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

#define CHECK_CUSPARSE(func)                                                   \
{                                                                              \
    cusparseStatus_t status = (func);                                          \
    if (status != CUSPARSE_STATUS_SUCCESS) {                                   \
        printf("CUSPARSE API failed at line %d with error: %s (%d)\n",         \
               __LINE__, cusparseGetErrorString(status), status);              \
        return EXIT_FAILURE;                                                   \
    }                                                                          \
}

typedef struct {
    int* index;
    size_t wg_n;
    int* wg_offsets;
    int* wg_nodes;
} edge_index_t;

edge_index_t read_file_graph(int* edge_n, int* node_n, int* max_degree);
int read_file_int(FILE *file);

__global__ void compute_edge_activation(int* edge_n, unsigned char* edge_activation, int* edge_end, int* splitters, int* current_splitter_index, int* node_blocks) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if(i < (*edge_n)) {
        int splitter = splitters[*current_splitter_index];
        int node = edge_end[i];
        int block = node_blocks[node];
        edge_activation[i] = block == splitter;
    }
}

__global__ void compute_weights(int* wg_offsets, int* wg_nodes, unsigned char* edge_activation, __half* weights) {
    int start = wg_offsets[blockIdx.x];
    int end = wg_offsets[blockIdx.x + 1];
    int dest = wg_nodes[blockIdx.x];
    int g_size = end - start;
    extern __shared__ int sact[];
    int i = start + threadIdx.x;
    sact[threadIdx.x] = i<end ?  edge_activation[i] : 0;
    __syncthreads();

    for(int stride = blockDim.x/ 2; stride > 0; stride>>=1) {
        if(threadIdx.x < stride) {
            sact[threadIdx.x] += sact[threadIdx.x + stride];
        }
        __syncthreads();
    }

    if(!threadIdx.x) {
        atomicAdd(&weights[dest], sact[threadIdx.x]);
    }
}

__global__ void block_ballot(int* node_blocks, int* max_degree, __half* weights, int* weight_adv, int* node_n) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if(i < (*node_n)) {
        int weight = weights[i];
        int leader = node_blocks[i];
        if(i != leader && weight != ((int)weights[leader])) {
            long int adv_index = ((long int)*max_degree) * leader + weight;
            weight_adv[adv_index] = i;
        }
    }
}

__global__ void split(int* new_node_blocks, int* node_blocks, int* max_degree, __half* weights, int* weight_adv, int* node_n) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if(i < (*node_n)) {
        int block = node_blocks[i];
        int weight = weights[i];
        int new_block = block;
        if(i != block && weight != ((int)weights[block])) {
            long int adv_index = ((long int)*max_degree) * block + weight;
            new_block = weight_adv[adv_index];
        }
        new_node_blocks[i] = new_block;
    }
}

__global__ void add_splitters(int* new_node_blocks, __half* weights, int* node_blocks, int* splitters, int* current_splitter_index, int* node_n) {
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
    int node_n = 0;
    int edge_n = 0;
    int max_degree = 0;
    edge_index_t edge_index = read_file_graph(&edge_n, &node_n, &max_degree);
    const int BLOCK_N = (node_n+(THREAD_N-1)) / THREAD_N;
    int current_splitter_index = 0;

    int *d_node_n, *d_new_node_blocks, *d_node_blocks, *d_current_splitter_index,
        *d_max_degree, *d_splitters, *d_weight_adv, *d_swap, *d_edge_start,
        *d_edge_end, *d_edge_n, *d_wg_offsets, *d_wg_nodes;

    unsigned char* d_edge_activation;
    __half *d_weights;

    CHECK_CUDA( cudaMalloc((void **)&d_weights, node_n * sizeof(__half)) );
    CHECK_CUDA( cudaMalloc((void **)&d_edge_activation, edge_n * sizeof(unsigned char)) );
    CHECK_CUDA( cudaMalloc((void **)&d_edge_start, edge_n * sizeof(int)) );
    CHECK_CUDA( cudaMalloc((void **)&d_wg_offsets, (edge_index.wg_n + 1) * sizeof(int)) );
    CHECK_CUDA( cudaMalloc((void **)&d_wg_nodes, edge_index.wg_n * sizeof(int)) );
    CHECK_CUDA( cudaMalloc((void **)&d_edge_end, edge_n * sizeof(int)) );
    CHECK_CUDA( cudaMalloc((void **)&d_weight_adv, node_n * sizeof(int) * max_degree) );
    CHECK_CUDA( cudaMalloc((void **)&d_node_n, sizeof(int)) );
    CHECK_CUDA( cudaMalloc((void **)&d_edge_n, sizeof(int)) );
    CHECK_CUDA( cudaMalloc((void **)&d_new_node_blocks, node_n * sizeof(int)) );
    CHECK_CUDA( cudaMalloc((void **)&d_node_blocks, node_n * sizeof(int)) );
    CHECK_CUDA( cudaMalloc((void **)&d_splitters, node_n * sizeof(int)) );
    CHECK_CUDA( cudaMalloc((void **)&d_current_splitter_index, sizeof(int)) );
    CHECK_CUDA( cudaMalloc((void **)&d_max_degree, sizeof(int)) );

    CHECK_CUDA( cudaMemset(d_weights, 0, node_n * sizeof(__half)) );
    CHECK_CUDA( cudaMemset(d_weight_adv, 0, max_degree * node_n * sizeof(int)) );
    CHECK_CUDA( cudaMemset(d_new_node_blocks, 0, node_n * sizeof(int)) );
    CHECK_CUDA( cudaMemset(d_node_blocks, 0, node_n * sizeof(int)) );
    CHECK_CUDA( cudaMemset(d_splitters, 0, node_n * sizeof(int)) );
    CHECK_CUDA( cudaMemset(d_current_splitter_index, 0, sizeof(int)) );

    CHECK_CUDA( cudaMemcpy(d_node_n, &node_n, sizeof(int), cudaMemcpyHostToDevice) );
    CHECK_CUDA( cudaMemcpy(d_edge_n, &edge_n, sizeof(int), cudaMemcpyHostToDevice) );
    CHECK_CUDA( cudaMemcpy(d_wg_offsets, edge_index.wg_offsets, sizeof(int) * (edge_index.wg_n + 1), cudaMemcpyHostToDevice) );
    CHECK_CUDA( cudaMemcpy(d_wg_nodes, edge_index.wg_nodes, sizeof(int) * edge_index.wg_n, cudaMemcpyHostToDevice) );
    CHECK_CUDA( cudaMemcpy(d_edge_start, edge_index.index, sizeof(int) * edge_n, cudaMemcpyHostToDevice) );
    CHECK_CUDA( cudaMemcpy(d_edge_end, edge_index.index + edge_n, sizeof(int) * edge_n, cudaMemcpyHostToDevice) );

    CHECK_CUDA( cudaMemcpy(d_max_degree, &max_degree, sizeof(int), cudaMemcpyHostToDevice) );

    while(current_splitter_index >= 0) {
        compute_edge_activation<<<(edge_n+(THREAD_N-1)) / THREAD_N, THREAD_N>>>(d_edge_n, d_edge_activation, d_edge_end, d_splitters, d_current_splitter_index, d_node_blocks);

        compute_weights<<< edge_index.wg_n, WEIGHT_THREAD_N, WEIGHT_THREAD_N >>>(d_wg_offsets, d_wg_nodes, d_edge_activation, d_weights);

        block_ballot<<<BLOCK_N, THREAD_N>>>(d_node_blocks, d_max_degree, d_weights, d_weight_adv, d_node_n);

        split<<<BLOCK_N, THREAD_N>>>(d_new_node_blocks, d_node_blocks, d_max_degree, d_weights, d_weight_adv, d_node_n);

        add_splitters<<<BLOCK_N, THREAD_N>>>(d_new_node_blocks, d_weights, d_node_blocks, d_splitters, d_current_splitter_index, d_node_n);

        CHECK_CUDA( cudaMemcpy(&current_splitter_index, d_current_splitter_index, sizeof(int), cudaMemcpyDeviceToHost) );
        current_splitter_index--;
        CHECK_CUDA( cudaMemcpy(d_current_splitter_index, &current_splitter_index, sizeof(int), cudaMemcpyHostToDevice) );

        d_swap = d_node_blocks;
        d_node_blocks = d_new_node_blocks;
        d_new_node_blocks = d_swap;
    }
    int *result = (int*)malloc(node_n * sizeof(int));
    CHECK_CUDA( cudaMemcpy(result, d_node_blocks, node_n * sizeof(int), cudaMemcpyDeviceToHost) );
    printf("[%d", result[0]);
    for (int i=1; i<node_n; ++i) {
        printf(",%d", result[i]);
    }
    printf("]");
    return 0;
}

edge_index_t read_file_graph(int* edge_n, int* node_n, int* max_degree) {
    FILE *file = fopen("graph.txt", "r");
    *node_n = read_file_int(file);
    *edge_n = read_file_int(file);
    edge_index_t res;
    int* degrees = (int*)calloc(*node_n, sizeof(int));
    res.index = (int*)malloc(((size_t)*edge_n) * 2 * sizeof(int));

    size_t connected_nodes_n = 0;
    for(int i=0; i<(*edge_n); ++i) {
        res.index[i] = read_file_int(file);
        res.index[(*edge_n) + i] = read_file_int(file);
        if(!degrees[res.index[i]]){
            connected_nodes_n++;
        }
        degrees[res.index[i]]++;
        if(degrees[res.index[i]] > *max_degree) {
            *max_degree = degrees[res.index[i]];
        }
    }

    int* connected_nodes = (int*)malloc(connected_nodes_n * sizeof(int));
    size_t last = 0;
    res.wg_n = 0;
    for(int i=0; i < *node_n; ++i) {
        if(degrees[i] > 0) {
            connected_nodes[last++] = i;
            float size = max(WEIGHT_THREAD_N, degrees[i]);
            res.wg_n += ceil(size / WEIGHT_THREAD_N);
        }
    }

    res.wg_offsets = (int*)malloc((res.wg_n + 1) * sizeof(int));
    res.wg_nodes = (int*)malloc(res.wg_n * sizeof(int));
    last = 0;
    int last_real_group_size = 0;

    for(int i=0; i<connected_nodes_n; ++i) {
        float size = max(WEIGHT_THREAD_N, degrees[connected_nodes[i]]);
        int groups = ceil(size / WEIGHT_THREAD_N);
        for(int g=0; g<groups; ++g) {
            int group_size = min(WEIGHT_THREAD_N, degrees[connected_nodes[i]] - (g * WEIGHT_THREAD_N));
            size_t old_offset = !last ? 0 : res.wg_offsets[last-1];
            res.wg_offsets[last] = old_offset + last_real_group_size;
            res.wg_nodes[last] = connected_nodes[i];
            last++;
            last_real_group_size = group_size;
        }
    }
    res.wg_offsets[last] = res.wg_offsets[last-1] + last_real_group_size;

    free(degrees);
    free(connected_nodes);
    return res;
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
