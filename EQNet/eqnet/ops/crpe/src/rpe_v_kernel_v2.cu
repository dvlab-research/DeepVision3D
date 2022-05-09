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
__global__ void rpe_v_forward_v2(
    int b, int total_query_num, int local_size,
    int total_key_num, int nhead, int hdim, int l,
    const int *query_batch_cnt, const int *key_batch_cnt, const int* index_pair_batch,
    const int *index_pair,
    const float *attn_weight, const float* value_features,
    const float *relpos, const float *lookup_table,
    float *output) {
    // dim3 blocks(total_query_num, nhead, hdim); dim3 threads(local_size);
    // params query_batch_cnt: [b]
    // params key_batch_cnt: [b]
    // params index_pair_batch: [total_query_num]
    // params index_pair: [total_query_num, local_size]
    // params attn_weight: [total_query_num, local_size, nhead]
    // params value_features: [total_key_num, nhead, hdim]
    // params relpos: [total_query_num, local_size]
    // params lookup_table: [l, nhead, hdim]
    // params output: [total_query_num, nhead, hdim]
    int query_idx = blockIdx.x;
    int head_idx = blockIdx.y;
    int hdim_idx = threadIdx.x;
    if (query_idx >= total_query_num ||
        head_idx >= nhead ||
        hdim_idx >= hdim) return;

    // get key_start_idx.
    int batch_idx = index_pair_batch[query_idx];
    int key_start_idx = 0;
    for (int i = 0; i < batch_idx; i++){
        key_start_idx += key_batch_cnt[i];
    }
    int cur_key_idx;
    float cur_relpos;

    // get shared attn_weight.
    __shared__ float shared_attn_weight[d];  // d == local_size
    __shared__ int shared_value_indices[d], shared_quan_relpos[d];  // d == local_size
    for (int i = hdim_idx; i < local_size; i += blockDim.x){
        shared_attn_weight[i] = attn_weight[
            query_idx * local_size * nhead + i * nhead + head_idx];

        cur_relpos = relpos[query_idx * local_size + i];
        shared_quan_relpos[i] = min(max(int(floor(cur_relpos)), 0), l - 1);

        cur_key_idx = index_pair[query_idx * local_size + i];
        if (cur_key_idx == -1){
            shared_value_indices[i] = -1;
            continue;
        }
        cur_key_idx += key_start_idx;
        shared_value_indices[i] = cur_key_idx;
    }
    __syncthreads();

    output += query_idx * nhead * hdim + head_idx * hdim + hdim_idx;

    float attn_result = 0;
    for (int i = 0; i < local_size; i++){
        if(shared_value_indices[i] == -1) continue;
        attn_result += shared_attn_weight[i] * (
            value_features[shared_value_indices[i] * nhead * hdim + head_idx * hdim + hdim_idx] +
            lookup_table[shared_quan_relpos[i] * nhead * hdim + head_idx * hdim + hdim_idx]);
    }
    output[0] = attn_result;
}


void rpe_v_launcher_v2(
    int b, int total_query_num, int local_size,
    int total_key_num, int nhead, int hdim, int l,
    const int *query_batch_cnt, const int *key_batch_cnt, const int* index_pair_batch,
    const int *index_pair,
    const float *attn_weight, const float* value_features,
    const float *relpos, const float *lookup_table,
    float *output){
    // params query_batch_cnt: [b]
    // params key_batch_cnt: [b]
    // params index_pair_batch: [total_query_num]
    // params index_pair: [total_query_num, local_size]
    // params attn_weight: [total_query_num, local_size, nhead]
    // params value_features: [total_key_num, nhead, hdim]
    // params relpos: [total_query_num, local_size]
    // params lookup_table: [l, nhead, hdim]
    // params output: [total_query_num, nhead, hdim]
    dim3 blocks(total_query_num, nhead);
    dim3 threads(hdim);
    if (local_size > 200){
        throw "local_size should be <= 200.";
    }

    switch (local_size){
        case 16:
            rpe_v_forward_v2<16><<<blocks, threads>>>(
                b, total_query_num, local_size, total_key_num, nhead, hdim, l,
                query_batch_cnt, key_batch_cnt, index_pair_batch,
                index_pair, attn_weight, value_features,
                relpos, lookup_table,
                output);
            break;
        case 32:
            rpe_v_forward_v2<32><<<blocks, threads>>>(
                b, total_query_num, local_size, total_key_num, nhead, hdim, l,
                query_batch_cnt, key_batch_cnt, index_pair_batch,
                index_pair, attn_weight, value_features,
                relpos, lookup_table,
                output);
            break;
        case 64:
            rpe_v_forward_v2<64><<<blocks, threads>>>(
                b, total_query_num, local_size, total_key_num, nhead, hdim, l,
                query_batch_cnt, key_batch_cnt, index_pair_batch,
                index_pair, attn_weight, value_features,
                relpos, lookup_table,
                output);
            break;
        case 128:
            rpe_v_forward_v2<128><<<blocks, threads>>>(
                b, total_query_num, local_size, total_key_num, nhead, hdim, l,
                query_batch_cnt, key_batch_cnt, index_pair_batch,
                index_pair, attn_weight, value_features,
                relpos, lookup_table,
                output);
            break;
        default:
            rpe_v_forward_v2<200><<<blocks, threads>>>(
                b, total_query_num, local_size, total_key_num, nhead, hdim, l,
                query_batch_cnt, key_batch_cnt, index_pair_batch,
                index_pair, attn_weight, value_features,
                relpos, lookup_table,
                output);
            break;
    }
}


template <unsigned int d>
__global__ void rpe_v_backward_v2(
    int b, int total_query_num, int local_size,
    int total_key_num, int nhead, int hdim, int l,
    const int *query_batch_cnt, const int *key_batch_cnt, const int* index_pair_batch,
    const int *index_pair,
    const float *attn_weight, const float* value_features,
    const float* relpos, const float* lookup_table,
    float *grad_out, float * grad_attn_weight, float * grad_value_features,
    float *grad_lookup_table) {
    // dim3 blocks(total_query_num, nhead); dim3 threads(hdim);
    // params query_batch_cnt: [b]
    // params key_batch_cnt: [b]
    // params index_pair_batch: [total_query_num]
    // params index_pair: [total_query_num, local_size]
    // params attn_weight: [total_query_num, local_size, nhead]
    // params value_features: [total_key_num, nhead, hdim]
    // params relpos: [total_query_num, local_size]
    // params lookup_table: [l, nhead, hdim]
    // params grad_out: [total_query_num, nhead, hdim]
    // params grad_attn_weight: [total_query_num, local_size, nhead]
    // params grad_value_features: [total_key_num, nhead, hdim]
    // params grad_lookup_table: [l, nhead, hdim]
    int query_idx = blockIdx.x;
    int head_idx = blockIdx.y;
    int hdim_idx = threadIdx.x;
    if (query_idx >= total_query_num ||
        head_idx >= nhead ||
        hdim_idx >= hdim) return;

    int batch_idx = index_pair_batch[query_idx];
    int key_start_idx = 0;
    for (int i = 0; i < batch_idx; i++){
        key_start_idx += key_batch_cnt[i];
    }
    int cur_key_idx;
    float cur_relpos;

    // get shared variables.
    __shared__ float shared_attn_weight[d], shared_grad_attn_weight[d];  // d == local_size
    __shared__ int shared_value_indices[d], shared_quan_relpos[d];  // d == local_size
    for (int i = hdim_idx; i < local_size; i += blockDim.x){
        shared_attn_weight[i] = attn_weight[
            query_idx * local_size * nhead + i * nhead + head_idx];
        shared_grad_attn_weight[i] = 0;

        cur_relpos = relpos[query_idx * local_size + i];
        shared_quan_relpos[i] = min(max(int(floor(cur_relpos)), 0), l - 1);

        cur_key_idx = index_pair[query_idx * local_size + i];
        if (cur_key_idx == -1){
            shared_value_indices[i] = -1;
            continue;
        }
        cur_key_idx += key_start_idx;
        shared_value_indices[i] = cur_key_idx;
    }
    __syncthreads();

    float gradient = grad_out[query_idx * nhead * hdim + head_idx * hdim + hdim_idx];
    for (int i = 0; i < local_size; i++){
        if (shared_value_indices[i] == -1) continue;

        atomicAdd(
            shared_grad_attn_weight + i,
            gradient * (
                value_features[shared_value_indices[i] * nhead * hdim + head_idx * hdim + hdim_idx] +
                lookup_table[shared_quan_relpos[i] * nhead * hdim + head_idx * hdim + hdim_idx]
            )
        );

        atomicAdd(
            grad_value_features + shared_value_indices[i] * nhead * hdim + head_idx * hdim + hdim_idx,
            gradient * shared_attn_weight[i]
        );

        atomicAdd(
            grad_lookup_table + shared_quan_relpos[i] * nhead * hdim + head_idx * hdim + hdim_idx,
            gradient * shared_attn_weight[i]
        );

    }
    __syncthreads();

    for (int i = hdim_idx; i < local_size; i += blockDim.x){
        grad_attn_weight[query_idx * local_size * nhead + i * nhead + head_idx] = shared_grad_attn_weight[i];
    }
}


void rpe_v_grad_launcher_v2(
    int b, int total_query_num, int local_size,
    int total_key_num, int nhead, int hdim, int l,
    const int *query_batch_cnt, const int *key_batch_cnt, const int* index_pair_batch,
    const int *index_pair,
    const float *attn_weight, const float* value_features,
    const float* relpos, const float* lookup_table,
    float *grad_out, float* grad_attn_weight, float* grad_value_features,
    float *grad_lookup_table){
    // params query_batch_cnt: [b]
    // params key_batch_cnt: [b]
    // params index_pair_batch: [total_query_num]
    // params index_pair: [total_query_num, local_size]
    // params attn_weight: [total_query_num, local_size, nhead]
    // params value_features: [total_key_num, nhead, hdim]
    // params relpos: [total_query_num, local_size]
    // params lookup_table: [l, nhead, hdim]
    // params grad_out: [total_query_num, nhead, hdim]
    // params grad_attn_weight: [total_query_num, local_size, nhead]
    // params grad_value_features: [total_key_num, nhead, hdim]
    // params grad_lookup_table: [l, nhead, hdim]
    dim3 blocks(total_query_num, nhead);
    dim3 threads(hdim);
    if (local_size > 200){
        throw "local_size should be <= 200.";
    }

    switch(local_size){
        case 16:
            rpe_v_backward_v2<16><<<blocks, threads>>>(
                b, total_query_num, local_size, total_key_num, nhead, hdim, l,
                query_batch_cnt, key_batch_cnt, index_pair_batch,
                index_pair, attn_weight, value_features,
                relpos, lookup_table,
                grad_out, grad_attn_weight, grad_value_features, grad_lookup_table);
            break;
        case 32:
            rpe_v_backward_v2<32><<<blocks, threads>>>(
                b, total_query_num, local_size, total_key_num, nhead, hdim, l,
                query_batch_cnt, key_batch_cnt, index_pair_batch,
                index_pair, attn_weight, value_features,
                relpos, lookup_table,
                grad_out, grad_attn_weight, grad_value_features, grad_lookup_table);
            break;
        case 64:
            rpe_v_backward_v2<64><<<blocks, threads>>>(
                b, total_query_num, local_size, total_key_num, nhead, hdim, l,
                query_batch_cnt, key_batch_cnt, index_pair_batch,
                index_pair, attn_weight, value_features,
                relpos, lookup_table,
                grad_out, grad_attn_weight, grad_value_features, grad_lookup_table);
            break;
        case 128:
            rpe_v_backward_v2<128><<<blocks, threads>>>(
                b, total_query_num, local_size, total_key_num, nhead, hdim, l,
                query_batch_cnt, key_batch_cnt, index_pair_batch,
                index_pair, attn_weight, value_features,
                relpos, lookup_table,
                grad_out, grad_attn_weight, grad_value_features, grad_lookup_table);
            break;
        default:
            rpe_v_backward_v2<200><<<blocks, threads>>>(
                b, total_query_num, local_size, total_key_num, nhead, hdim, l,
                query_batch_cnt, key_batch_cnt, index_pair_batch,
                index_pair, attn_weight, value_features,
                relpos, lookup_table,
                grad_out, grad_attn_weight, grad_value_features, grad_lookup_table);
            break;
    }
}

