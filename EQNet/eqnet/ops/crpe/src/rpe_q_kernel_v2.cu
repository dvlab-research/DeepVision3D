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


template <unsigned int d>
__global__ void rpe_q_forward_v2(
    int b, int total_query_num, int local_size, int nhead, int hdim, int l,
    const int *query_batch_cnt,
    const float *relpos, const float* lookup_table, const float* query_features,
    float *output) {
    // dim3 blocks(total_query_num, nhead); dim3 threads(local_size);
    // params query_batch_cnt: [b]
    // params relpos: [total_query_num, local_size]
    // params lookup_table: [l, nhead, hdim]
    // params query_features: [total_query_num, nhead, hdim]
    // params output: [total_query_num, local_size, nhead]

    int query_idx = blockIdx.x;
    int head_idx = blockIdx.y;
    int local_key_idx = threadIdx.x;

    if (query_idx >= total_query_num ||
        head_idx >= nhead ||
        local_key_idx >= local_size) return;

    // get query features for attention computation.
    __shared__ float shared_query_features[d];
    for(int i = local_key_idx; i < hdim; i += blockDim.x){
        shared_query_features[i] = query_features[
            query_idx * nhead * hdim + head_idx * hdim + i];
    }
    __syncthreads();
    // 1. obtain quantize relative position.
    relpos += query_idx * local_size + local_key_idx;
    int quantize_relpos = min(max(int(floor(relpos[0])), 0), l - 1);
    lookup_table += quantize_relpos * nhead * hdim + head_idx * hdim;
    output += query_idx * local_size * nhead + local_key_idx * nhead + head_idx;

    float attn_weight = 0;
    for (int i = 0; i < hdim; i++){
        attn_weight += shared_query_features[i] * lookup_table[i];
    }
    output[0] = attn_weight;
}


void rpe_q_launcher_v2(
    int b, int total_query_num, int local_size, int nhead, int hdim, int l,
    const int *query_batch_cnt,
    const float *relpos, const float* lookup_table, const float* query_features,
    float *output){
    // params query_batch_cnt: [b]
    // params relpos: [total_query_num, local_size]
    // params lookup_table: [l, nhead, hdim]
    // params query_features: [total_query_num, nhead, hdim]
    // params output: [total_query_num, local_size, nhead]
    if (hdim > 100){
        throw "hdim should be <= 100.";
    }

    dim3 blocks(total_query_num, nhead);
    dim3 threads(local_size);
    switch (hdim){
        case 4:
            rpe_q_forward_v2<4><<<blocks, threads>>>(
                b, total_query_num, local_size, nhead, hdim, l,
                query_batch_cnt,
                relpos, lookup_table, query_features,
                output);
            break;
        case 8:
            rpe_q_forward_v2<8><<<blocks, threads>>>(
                b, total_query_num, local_size, nhead, hdim, l,
                query_batch_cnt,
                relpos, lookup_table, query_features,
                output);
            break;
        case 16:
            rpe_q_forward_v2<16><<<blocks, threads>>>(
                b, total_query_num, local_size, nhead, hdim, l,
                query_batch_cnt,
                relpos, lookup_table, query_features,
                output);
            break;
        case 32:
            rpe_q_forward_v2<32><<<blocks, threads>>>(
                b, total_query_num, local_size, nhead, hdim, l,
                query_batch_cnt,
                relpos, lookup_table, query_features,
                output);
            break;
        default:
            rpe_q_forward_v2<100><<<blocks, threads>>>(
                b, total_query_num, local_size, nhead, hdim, l,
                query_batch_cnt,
                relpos, lookup_table, query_features,
                output);
            break;
    }

}


template <unsigned int d>
__global__ void rpe_q_backward_v2(
    int b, int total_query_num, int local_size, int nhead, int hdim, int l,
    const int *query_batch_cnt,
    const float *relpos, const float* lookup_table, const float* query_features,
    float *grad_out, float * grad_lookup_table, float * grad_query_features) {
    // dim3 blocks(total_query_num, nhead); dim3 blocks(local_size);
    // params query_batch_cnt: [b]
    // params relpos: [total_query_num, local_size]
    // params lookup_table: [l, nhead, hdim]
    // params query_features: [total_query_num, nhead, hdim]
    // params grad_out: [total_query_num, local_size, nhead]
    // params grad_lookup_table: [l, nhead, hdim]
    // params grad_query_features: [total_query_num, nhead, hdim]

    int query_idx = blockIdx.x;
    int head_idx = blockIdx.y;
    int local_key_idx = threadIdx.x;

    // out-range judgement.
    if (query_idx >= total_query_num ||
        head_idx >= nhead ||
        local_key_idx >= local_size) return;

    // get shared query features and shared grad query features.
    __shared__ float shared_query_features[d], shared_grad_query_features[d];
    for (int i = local_key_idx; i < hdim; i += blockDim.x){
        shared_query_features[i] = query_features[
            query_idx * nhead * hdim + head_idx * hdim + i];
        shared_grad_query_features[i] = 0;
    }
    __syncthreads();

    // 2. Obtain quantize relative position.
    relpos += query_idx * local_size + local_key_idx;
    int quantize_relpos = min(max(int(floor(relpos[0])), 0), l - 1);

    lookup_table += quantize_relpos * nhead * hdim + head_idx * hdim;
    grad_lookup_table += quantize_relpos * nhead * hdim + head_idx * hdim;
    grad_query_features += query_idx * nhead * hdim + head_idx * hdim;

    float gradient = grad_out[query_idx * local_size * nhead + local_key_idx * nhead + head_idx];
    for (int i = 0; i < hdim; i++){
        atomicAdd(
            grad_lookup_table + i,
            gradient * shared_query_features[i]);
        atomicAdd(
            shared_grad_query_features + i,
            gradient * lookup_table[i]);
    }
    __syncthreads();

    for (int i = local_key_idx; i < hdim; i += blockDim.x){
        grad_query_features[i] = shared_grad_query_features[i];
    }
}


void rpe_q_grad_launcher_v2(
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
    if (hdim > 100){
        throw "hdim should be <= 100.";
    }

    dim3 blocks(total_query_num, nhead);
    dim3 threads(local_size);
    switch (hdim){
        case 4:
            rpe_q_backward_v2<4><<<blocks, threads>>>(
                b, total_query_num, local_size, nhead, hdim, l,
                query_batch_cnt, relpos, lookup_table, query_features,
                grad_out, grad_lookup_table, grad_query_features);
            break;
        case 8:
            rpe_q_backward_v2<8><<<blocks, threads>>>(
                b, total_query_num, local_size, nhead, hdim, l,
                query_batch_cnt, relpos, lookup_table, query_features,
                grad_out, grad_lookup_table, grad_query_features);
            break;
        case 16:
            rpe_q_backward_v2<16><<<blocks, threads>>>(
                b, total_query_num, local_size, nhead, hdim, l,
                query_batch_cnt, relpos, lookup_table, query_features,
                grad_out, grad_lookup_table, grad_query_features);
            break;
        case 32:
            rpe_q_backward_v2<32><<<blocks, threads>>>(
                b, total_query_num, local_size, nhead, hdim, l,
                query_batch_cnt, relpos, lookup_table, query_features,
                grad_out, grad_lookup_table, grad_query_features);
            break;
        default:
            rpe_q_backward_v2<100><<<blocks, threads>>>(
                b, total_query_num, local_size, nhead, hdim, l,
                query_batch_cnt, relpos, lookup_table, query_features,
                grad_out, grad_lookup_table, grad_query_features);
            break;
    }
}

