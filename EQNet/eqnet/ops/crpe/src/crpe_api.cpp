#include <torch/serialize/tensor.h>
#include <torch/extension.h>

#include "crpe_func.h"
#include "crpe_func_v2.h"


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("rpe_q_wrapper", &rpe_q_wrapper,
        "crpe for query");
    m.def("rpe_q_grad_wrapper", &rpe_q_grad_wrapper,
        "crpe for query.");
    m.def("rpe_k_wrapper", &rpe_k_wrapper,
        "crpe for key.");
    m.def("rpe_k_grad_wrapper", &rpe_k_grad_wrapper,
        "crpe for key.");
    m.def("rpe_v_wrapper", &rpe_v_wrapper,
        "crpe for value.");
    m.def("rpe_v_grad_wrapper", &rpe_v_grad_wrapper,
        "crpe for value.");

    m.def("rpe_q_wrapper_v2", &rpe_q_wrapper_v2,
        "crpe for query");
    m.def("rpe_q_grad_wrapper_v2", &rpe_q_grad_wrapper_v2,
        "crpe for query.");
    m.def("rpe_k_wrapper_v2", &rpe_k_wrapper_v2,
        "crpe for key.");
    m.def("rpe_k_grad_wrapper_v2", &rpe_k_grad_wrapper_v2,
        "crpe for key.");
    m.def("rpe_v_wrapper_v2", &rpe_v_wrapper_v2,
        "crpe for value.");
    m.def("rpe_v_grad_wrapper_v2", &rpe_v_grad_wrapper_v2,
        "crpe for value.");
}
