#ifndef _CRPE_FUNC_H
#define _CRPE_FUNC_H

#include <torch/serialize/tensor.h>
#include<vector>
#include <cuda.h>
#include <cuda_runtime_api.h>


// crpe q.
void rpe_q_launcher(
    int b, int total_query_num, int local_size, int nhead, int hdim, int l,
    const int *query_batch_cnt,
    const float* relpos, const float* lookup_table, const float* query_features,
    float *output);


int rpe_q_wrapper(
    int b, int total_query_num, int local_size, int nhead, int hdim, int l,
    at::Tensor query_batch_cnt, at::Tensor relpos, at::Tensor lookup_table, at::Tensor query_features,
    at::Tensor output);


void rpe_q_grad_launcher(
    int b, int total_query_num, int local_size, int nhead, int hdim, int l,
    const int *query_batch_cnt,
    const float* relpos, const float* lookup_table, const float* query_features,
    float *grad_out, float* grad_lookup_table, float* grad_query_features);


int rpe_q_grad_wrapper(
    int b, int total_query_num, int local_size, int nhead, int hdim, int l,
    at::Tensor query_batch_cnt,
    at::Tensor relpos, at::Tensor lookup_table, at::Tensor query_features,
    at::Tensor grad_out, at::Tensor grad_lookup_table, at::Tensor grad_query_features);


// crpe k.
void rpe_k_launcher(
    int b, int total_query_num, int local_size,
    int total_key_num, int nhead, int hdim, int l,
    const int *query_batch_cnt, const int *key_batch_cnt, const int* index_pair_batch,
    const int *index_pair,
    const float* relpos, const float* lookup_table, const float* key_features,
    float *output);


int rpe_k_wrapper(
    int b, int total_query_num, int local_size, int total_key_num, int nhead, int hdim, int l,
    at::Tensor query_batch_cnt, at::Tensor key_batch_cnt, at::Tensor index_pair_batch,
    at::Tensor index_pair, at::Tensor relpos, at::Tensor lookup_table, at::Tensor key_features,
    at::Tensor output);


void rpe_k_grad_launcher(
    int b, int total_query_num, int local_size,
    int total_key_num, int nhead, int hdim, int l,
    const int *query_batch_cnt, const int *key_batch_cnt, const int* index_pair_batch,
    const int *index_pair,
    const float* relpos, const float* lookup_table, const float* key_features,
    float *grad_out, float* grad_lookup_table, float* grad_key_features);


int rpe_k_grad_wrapper(
    int b, int total_query_num, int local_size, int total_key_num, int nhead, int hdim, int l,
    at::Tensor query_batch_cnt, at::Tensor key_batch_cnt, at::Tensor index_pair_batch,
    at::Tensor index_pair,
    at::Tensor relpos, at::Tensor lookup_table, at::Tensor key_features,
    at::Tensor grad_out, at::Tensor grad_lookup_table, at::Tensor grad_key_features);


// crpe v.
void rpe_v_launcher(
    int b, int total_query_num, int local_size,
    int total_key_num, int nhead, int hdim, int l,
    const int *query_batch_cnt, const int *key_batch_cnt, const int* index_pair_batch,
    const int *index_pair,
    const float *attn_weight, const float* value_features,
    const float *relpos, const float* lookup_table,
    float *output);


int rpe_v_wrapper(
    int b, int total_query_num, int local_size, int total_key_num, int nhead, int hdim, int l,
    at::Tensor query_batch_cnt, at::Tensor key_batch_cnt, at::Tensor index_pair_batch,
    at::Tensor index_pair, at::Tensor attn_weight, at::Tensor value_features,
    at::Tensor relpos, at::Tensor lookup_table,
    at::Tensor output);


void rpe_v_grad_launcher(
    int b, int total_query_num, int local_size,
    int total_key_num, int nhead, int hdim, int l,
    const int *query_batch_cnt, const int *key_batch_cnt, const int* index_pair_batch,
    const int *index_pair,
    const float *attn_weight, const float* value_features,
    const float* relpos, const float* lookup_table,
    float *grad_out, float* grad_attn_weight, float* grad_value_features,
    float *grad_lookup_table);


int rpe_v_grad_wrapper(
    int b, int total_query_num, int local_size, int total_key_num, int nhead, int hdim, int l,
    at::Tensor query_batch_cnt, at::Tensor key_batch_cnt, at::Tensor index_pair_batch,
    at::Tensor index_pair, at::Tensor attn_weight, at::Tensor value_features,
    at::Tensor relpos, at::Tensor lookup_table,
    at::Tensor grad_out, at::Tensor grad_attn_weight, at::Tensor grad_value_features,
    at::Tensor grad_lookup_table);

#endif