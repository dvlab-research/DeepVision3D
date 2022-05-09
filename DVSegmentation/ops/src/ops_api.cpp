#include <torch/extension.h>
#include <torch/serialize/tensor.h>

#include "ops.h"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m){
    m.def("voxelize_idx", &voxelize_idx_3d, "voxelize_idx");
    m.def("voxelize_fp", &voxelize_fp_feat, "voxelize_fp");
    m.def("voxelize_bp", &voxelize_bp_feat, "voxelize_bp");
    m.def("point_recover_fp", &point_recover_fp_feat, "point_recover_fp");
    m.def("point_recover_bp", &point_recover_bp_feat, "point_recover_bp");
}