/*
Transformer function helper function.

Written by tomztyang,
2021/08/23
*/

#include <math.h>
#include <stdio.h>

#define THREADS_PER_BLOCK 256
#define DIVUP(m,n) ((m) / (n) + ((m) % (n) > 0))
// #define DEBUG


__global__ void rpe_q_forward(
    int b, int total_query_num, int local_size, int nhead, int hdim, int l,
    const int *query_batch_cnt,
    const float *relpos, const float* lookup_table, const float* query_features,
    float *output) {
    // dim3 blocks(DIVUP(total_query_num * local_size, THREADS_PER_BLOCK), nhead, hdim);
    // params query_batch_cnt: [b]
    // params relpos: [total_query_num, local_size]
    // params lookup_table: [l, nhead, hdim]
    // params query_features: [total_query_num, nhead, hdim]
    // params output: [total_query_num, local_size, nhead]

    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int head_idx = blockIdx.y;
    int hdim_idx = blockIdx.z;
    if (index >= total_query_num * local_size ||
        head_idx >= nhead ||
        hdim_idx >= hdim) return;

    // 1. Obtain query features.
    int query_idx = index / local_size;
    query_features += query_idx * nhead * hdim + head_idx * hdim + hdim_idx;

    // 2. Obtain quantize relative position.
    relpos += index;
    int quantize_relpos = min(max(int(floor(relpos[0])), 0), l - 1);
    lookup_table += quantize_relpos * nhead * hdim + head_idx * hdim + hdim_idx;

    // 3. Obtain output position.
    output += index * nhead + head_idx;
    atomicAdd(
        output,
        query_features[0] * lookup_table[0]);
}


void rpe_q_launcher(
    int b, int total_query_num, int local_size, int nhead, int hdim, int l,
    const int *query_batch_cnt,
    const float *relpos, const float* lookup_table, const float* query_features,
    float *output){
    // params query_batch_cnt: [b]
    // params relpos: [total_query_num, local_size]
    // params lookup_table: [l, nhead, hdim]
    // params query_features: [total_query_num, nhead, hdim]
    // params output: [total_query_num, local_size, nhead]

    dim3 blocks(DIVUP(total_query_num * local_size, THREADS_PER_BLOCK), nhead, hdim);
    dim3 threads(THREADS_PER_BLOCK);
    rpe_q_forward<<<blocks, threads>>>(
        b, total_query_num, local_size, nhead, hdim, l,
        query_batch_cnt,
        relpos, lookup_table, query_features,
        output);
}


__global__ void rpe_q_backward(
    int b, int total_query_num, int local_size, int nhead, int hdim, int l,
    const int *query_batch_cnt,
    const float *relpos, const float* lookup_table, const float* query_features,
    float *grad_out, float * grad_lookup_table, float * grad_query_features) {
    // dim3 blocks(DIVUP(total_query_num * local_size, THREADS_PER_BLOCK), nhead, hdim);
    // params query_batch_cnt: [b]
    // params relpos: [total_query_num, local_size]
    // params lookup_table: [l, nhead, hdim]
    // params query_features: [total_query_num, nhead, hdim]
    // params grad_out: [total_query_num, local_size, nhead]
    // params grad_lookup_table: [l, nhead, hdim]
    // params grad_query_features: [total_query_num, nhead, hdim]

    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int head_idx = blockIdx.y;
    int hdim_idx = blockIdx.z;
    if (index >= total_query_num * local_size ||
        head_idx >= nhead ||
        hdim_idx >= hdim) return;

    // 1. Obtain query features.
    int query_idx = index / local_size;
    query_features += query_idx * nhead * hdim + head_idx * hdim + hdim_idx;
    grad_query_features += query_idx * nhead * hdim + head_idx * hdim + hdim_idx;

    // 2. Obtain quantize relative position.
    relpos += index;
    int quantize_relpos = min(max(int(floor(relpos[0])), 0), l - 1);
    lookup_table += quantize_relpos * nhead * hdim + head_idx * hdim + hdim_idx;
    grad_lookup_table += quantize_relpos * nhead * hdim + head_idx * hdim + hdim_idx;

    // 3. Obtain output position.
    grad_out += index * nhead + head_idx;
    atomicAdd(
        grad_query_features,
        grad_out[0] * lookup_table[0]);
    atomicAdd(
        grad_lookup_table,
        grad_out[0] * query_features[0]);
}


void rpe_q_grad_launcher(
    int b, int total_query_num, int local_size, int nhead, int hdim, int l,
    const int *query_batch_cnt,
    const float *relpos, const float* lookup_table, const float* query_features,
    float *grad_out, float* grad_lookup_table, float* grad_query_features){
    // params query_batch_cnt: [b]
    // params relpos: [total_query_num, local_size]
    // params lookup_table: [l, nhead, hdim]
    // params query_features: [total_query_num, nhead, hdim]
    // params grad_out: [total_query_num, local_size, nhead]
    // params grad_lookup_table: [l, nhead, hdim]
    // params grad_query_features: [total_query_num, nhead, hdim]

    dim3 blocks(DIVUP(total_query_num * local_size, THREADS_PER_BLOCK), nhead, hdim);
    dim3 threads(THREADS_PER_BLOCK);
    rpe_q_backward<<<blocks, threads>>>(
        b, total_query_num, local_size, nhead, hdim, l,
        query_batch_cnt, relpos, lookup_table, query_features,
        grad_out, grad_lookup_table, grad_query_features);
}

