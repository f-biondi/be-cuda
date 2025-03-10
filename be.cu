#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <cuda_fp16.h>
#include <cuda.h>
#include <cusparse.h>         

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
    int node;
    int n;
} node_connections_t;

typedef struct {
    int* slice_offsets;
    int* column_indices;
    __half* values;
    int slice_size;
    int sell_values_size;
    int* node_index_of;
} sell_data_t;

int nodecmp(const void *p1, const void *p2);
sell_data_t gen_sell(int* edge_index, int edge_n, int node_n, int n_slices);
int* read_file_graph(int* edge_n, int* node_n, int* max_node_w);
int read_file_int(FILE *file);
__half* gen_ones(int n);

__global__ void compute_weight_mask(int *node_n, __half *weight_mask, int *node_blocks, int *splitters, int *current_splitter_index) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int node_number = (*node_n);
    if(i < node_number) {
        int csi = *current_splitter_index;
        int splitter = splitters[csi];
        int block = node_blocks[i];
        weight_mask[i] = block == splitter ? 1.0 : 0.0;
    }
}

__global__ void block_ballot(int* node_blocks, int* max_node_w, __half* weights, int* weight_adv, int* node_n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if(i < (*node_n)) {
        int weight = weights[i];
        int leader = node_blocks[i];
        if(i != leader && weight != ((int)weights[leader])) {
            long int adv_index = ((long int)*max_node_w) * leader + weight;
            weight_adv[adv_index] = i;
        }
    }
}

__global__ void split(int* new_node_blocks, int* node_blocks, int* max_node_w, __half* weights, int* weight_adv, int* node_n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
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

__global__ void add_splitters(int* new_node_blocks, int* node_blocks, int* splitters, int* current_splitter_index, int* node_n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if(i < (*node_n)) {
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
    int max_node_w = 0;
    int* edge_index = read_file_graph(&edge_n, &node_n, &max_node_w);
    const int BLOCK_N = (node_n+(THREAD_N-1)) / THREAD_N;
    size_t node_size = node_n * sizeof(int);
    int current_splitter_index = 0;
    int n_slices = node_n/20;
    //int n_slices = atoi(argv[1]);
    //int n_slices = 3;
    sell_data_t sell_data = gen_sell(edge_index, edge_n, node_n, n_slices);

    int *d_node_n, *d_new_node_blocks, *d_node_blocks, *d_current_splitter_index, 
        *d_max_node_w, *d_splitters, *d_weight_adv, *d_swap, *d_slice_offsets,
        *d_columns;

    __half *d_weights, *d_weight_mask, *d_values;

    CHECK_CUDA( cudaMalloc((void **)&d_values, sell_data.sell_values_size * sizeof(__half)) );
    CHECK_CUDA( cudaMalloc((void **)&d_slice_offsets, (n_slices+1) * sizeof(int)) );
    CHECK_CUDA( cudaMalloc((void **)&d_columns, sell_data.sell_values_size * sizeof(int)) );
    CHECK_CUDA( cudaMalloc((void **)&d_weights, node_n * sizeof(__half)) );
    CHECK_CUDA( cudaMalloc((void **)&d_weight_mask, node_n * sizeof(__half)) );
    CHECK_CUDA( cudaMalloc((void **)&d_weight_adv, node_size * max_node_w) );
    CHECK_CUDA( cudaMalloc((void **)&d_node_n, sizeof(int)) );
    CHECK_CUDA( cudaMalloc((void **)&d_new_node_blocks, node_size) );
    CHECK_CUDA( cudaMalloc((void **)&d_node_blocks, node_size) );
    CHECK_CUDA( cudaMalloc((void **)&d_splitters, node_size) );
    CHECK_CUDA( cudaMalloc((void **)&d_current_splitter_index, sizeof(int)) );
    CHECK_CUDA( cudaMalloc((void **)&d_max_node_w, sizeof(int)) );

    CHECK_CUDA( cudaMemset(d_weights, 0, node_n * sizeof(__half)) );
    CHECK_CUDA( cudaMemset(d_new_node_blocks, 0, node_size) );
    CHECK_CUDA( cudaMemset(d_node_blocks, 0, node_size) );
    CHECK_CUDA( cudaMemset(d_splitters, 0, node_size) );
    CHECK_CUDA( cudaMemset(d_current_splitter_index, 0, sizeof(int)) );

    CHECK_CUDA( cudaMemcpy(d_node_n, &node_n, sizeof(int), cudaMemcpyHostToDevice) );
    CHECK_CUDA( cudaMemcpy(d_max_node_w, &max_node_w, sizeof(int), cudaMemcpyHostToDevice) );
    CHECK_CUDA( cudaMemcpy(d_values, sell_data.values, sell_data.sell_values_size * sizeof(__half), cudaMemcpyHostToDevice) );
    CHECK_CUDA( cudaMemcpy(d_slice_offsets, sell_data.slice_offsets, (n_slices + 1) * sizeof(int), cudaMemcpyHostToDevice) );
    CHECK_CUDA( cudaMemcpy(d_columns, sell_data.column_indices, sell_data.sell_values_size * sizeof(int), cudaMemcpyHostToDevice) );

    cusparseHandle_t handle = NULL;
    cusparseSpMatDescr_t adj_mat;
    cusparseDnVecDescr_t vecX, vecY;
    void* dBuffer = NULL;
    float alpha = 1.0;
    float beta = 0;
    size_t bufferSize = 0;

    CHECK_CUSPARSE( cusparseCreate(&handle) )

    CHECK_CUSPARSE( cusparseCreateSlicedEll(&adj_mat, node_n, node_n, edge_n,
                                        sell_data.sell_values_size, sell_data.slice_size,
                                        d_slice_offsets, d_columns, d_values,
                                        CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
                                        CUSPARSE_INDEX_BASE_ZERO, CUDA_R_16F) );

    // Create dense vector X
    CHECK_CUSPARSE( cusparseCreateDnVec(&vecX, node_n, d_weight_mask, CUDA_R_16F) )
    // Create dense vector y
    CHECK_CUSPARSE( cusparseCreateDnVec(&vecY, node_n, d_weights, CUDA_R_16F) )
    // allocate an external buffer if needed
    CHECK_CUSPARSE( cusparseSpMV_bufferSize(
                                 handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
                                 &alpha, adj_mat, vecX, &beta, vecY, CUDA_R_32F,
                                 CUSPARSE_SPMV_ALG_DEFAULT, &bufferSize) )
    CHECK_CUDA( cudaMalloc(&dBuffer, bufferSize) )
    while(current_splitter_index >= 0) {
        compute_weight_mask<<<BLOCK_N, THREAD_N>>>(d_node_n, d_weight_mask, d_node_blocks, d_splitters, d_current_splitter_index);

        CHECK_CUSPARSE( cusparseSpMV(handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
                                 &alpha, adj_mat, vecX, &beta, vecY, CUDA_R_32F,
                                 CUSPARSE_SPMV_ALG_DEFAULT, dBuffer) );

        block_ballot<<<BLOCK_N, THREAD_N>>>(d_node_blocks, d_max_node_w, d_weights, d_weight_adv, d_node_n);

        split<<<BLOCK_N, THREAD_N>>>(d_new_node_blocks, d_node_blocks, d_max_node_w, d_weights, d_weight_adv, d_node_n);

        add_splitters<<<BLOCK_N, THREAD_N>>>(d_new_node_blocks, d_node_blocks, d_splitters, d_current_splitter_index, d_node_n);
        CHECK_CUDA( cudaMemcpy(&current_splitter_index, d_current_splitter_index, sizeof(int), cudaMemcpyDeviceToHost) );
        current_splitter_index--;
        CHECK_CUDA( cudaMemcpy(d_current_splitter_index, &current_splitter_index, sizeof(int), cudaMemcpyHostToDevice) );

        d_swap = d_node_blocks;
        d_node_blocks = d_new_node_blocks;
        d_new_node_blocks = d_swap;
    }
    //cudaDeviceSynchronize(); -- Device already synchronized on cudaMemcpy
    int *result = (int*)malloc(node_size);
    CHECK_CUDA( cudaMemcpy(result, d_node_blocks, node_size, cudaMemcpyDeviceToHost) );
    printf("[%d", result[sell_data.node_index_of[0]]);
    for (int i=1; i<node_n; ++i) {
        printf(",%d", result[sell_data.node_index_of[i]]);
    }
    printf("]");
    return 0;
}

int nodecmp(const void *p1, const void *p2) {
    node_connections_t ip1 = *(node_connections_t*)p1;
    node_connections_t ip2 = *(node_connections_t*)p2;
    int r = ip1.n - ip2.n;
    return r ? r : (ip1.node - ip2.node);
}

sell_data_t gen_sell(int* edge_index, int edge_n, int node_n, int n_slices) {
    sell_data_t res;
    res.slice_size = ceil((float)node_n / n_slices);
    res.slice_offsets = (int*)calloc(n_slices + 1, sizeof(int));
    res.node_index_of = (int*)malloc(node_n * sizeof(int));
    int* connections_sum = (int*)malloc(node_n * sizeof(int));
    int* connections_strided = (int*)malloc(edge_n * sizeof(int));
    int* connections_cur = (int*)calloc(node_n, sizeof(int));
    int* padding_sizes = (int*)malloc(n_slices * sizeof(int));
    node_connections_t* sorted_nodes = (node_connections_t*)malloc(node_n * sizeof(node_connections_t));

    for(int i =0; i< node_n; ++i) {
        sorted_nodes[i].node = i;
        sorted_nodes[i].n = 0;
    }

    for(int i=0; i<edge_n; ++i) {
        int node = edge_index[i];
        sorted_nodes[node].n++;
    }

    for(int i=0; i<node_n; ++i) {
        connections_sum[i] = i ? connections_sum[i-1] + sorted_nodes[i-1].n : 0;
    }

    qsort(sorted_nodes, node_n, sizeof(node_connections_t), nodecmp);

    for(int i =0;i<node_n;++i) {
        node_connections_t c = sorted_nodes[i];
        res.node_index_of[c.node] = i;
    }

    for(int i=0; i<edge_n; ++i) {
        int node = edge_index[i];
        connections_strided[connections_sum[node] + connections_cur[node]] = edge_index[edge_n + i];
        connections_cur[node]++;
    }

    res.sell_values_size = 0;
    for(int i=0; i<n_slices; ++i) {
        int max_row_len = 0;
        for(int k=0; k<res.slice_size; ++k) {
            int node_index = (i * res.slice_size) + k;
            int row_len = node_index < node_n ? sorted_nodes[node_index].n : 0;
            max_row_len = max(max_row_len, row_len);
        }
        padding_sizes[i] = max_row_len;
        int slice_len = max_row_len * res.slice_size;
        res.sell_values_size += slice_len;
        res.slice_offsets[i+1] = res.slice_offsets[i] + slice_len;
    }

    res.column_indices = (int*)malloc(res.sell_values_size * sizeof(int));
    res.values = (__half*)malloc(res.sell_values_size * sizeof(__half));
    memset(connections_cur, 0, node_n * sizeof(int));

    int values_last = 0;
    for(int i=0; i<n_slices; ++i) {
        for(int k=0; k<padding_sizes[i]; ++k) {
            for(int c=0; c<res.slice_size; ++c) {
                int node_index = (i * res.slice_size) + c;
                if(node_index < node_n) { 
                    int node = sorted_nodes[node_index].node;
                    if (connections_cur[node] < sorted_nodes[node_index].n) {
                        res.column_indices[values_last] = res.node_index_of[connections_strided[connections_sum[node] + connections_cur[node]]];
                        res.values[values_last] = 1.0;
                        connections_cur[node]++;
                    } else {
                        res.column_indices[values_last] = -1;
                        res.values[values_last] = 0.0;
                    }
                } else {
                    res.column_indices[values_last] = -1;
                    res.values[values_last] = 0.0;
                }
                ++values_last;
            }
        }    
    }

    free(connections_sum);
    free(connections_strided);
    free(connections_cur);
    free(padding_sizes);
    
    return res;
}

int* read_file_graph(int* edge_n, int* node_n, int* max_node_w) {
    FILE *file = fopen("graph.txt", "r");
    *node_n = read_file_int(file);
    *edge_n = read_file_int(file);
    int* weights = (int*)calloc(*node_n, sizeof(int));
    size_t index_size = (*edge_n) * 2 * sizeof(int);
    int *edge_index = (int*)malloc(index_size);
    for(int i=0; i<(*edge_n); ++i) {
        edge_index[i] = read_file_int(file); 
        edge_index[(*edge_n) + i] = read_file_int(file); 
        weights[edge_index[i]]++; if(weights[edge_index[i]] > *max_node_w) {
            *max_node_w = weights[edge_index[i]];
        }
    }
    free(weights);
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

__half* gen_ones(int n) {
    __half* values = (__half*)malloc(n * sizeof(__half));
    for(int i=0; i<n; ++i) values[i] = 1.0;
    return values;
}
