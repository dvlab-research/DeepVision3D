import os

from setuptools import find_packages, setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension


def make_cuda_ext(name, module, sources):
    cuda_ext = CUDAExtension(
        name='%s.%s' % (module, name),
        sources=[os.path.join(*module.split('.'), src) for src in sources]
    )
    return cuda_ext


if __name__ == '__main__':

    setup(
        name='eqnet',
        description='A Unified Query-based Paradigm for Point Cloud Understanding',
        install_requires=[
            'numpy',
            'timm',
        ],

        author='Zetong Yang',
        author_email='tomztyang@gmail.com',
        license='Apache License 2.0',
        packages=find_packages(),
        cmdclass={
            'build_ext': BuildExtension,
        },
        ext_modules=[
            make_cuda_ext(
                name='knn_cuda',
                module='eqnet.ops.knn',
                sources=[
                    'src/knn.cpp',
                    'src/knn_cuda.cu',
                ]
            ),
            make_cuda_ext(
                name='grouping_cuda',
                module='eqnet.ops.grouping',
                sources=[
                    'src/group_points.cpp',
                    'src/group_points_gpu.cu',
                ]
            ),
            make_cuda_ext(
                name='attention_cuda',
                module='eqnet.ops.attention',
                sources=[
                    'src/attention_api.cpp',
                    'src/attention_func.cpp',
                    'src/attention_func_v2.cpp',
                    'src/attention_value_computation_kernel.cu',
                    'src/attention_value_computation_kernel_v2.cu',
                    'src/attention_weight_computation_kernel.cu',
                    'src/attention_weight_computation_kernel_v2.cu',
                ]
            ),
            make_cuda_ext(
                name='crpe_cuda',
                module='eqnet.ops.crpe',
                sources=[
                    'src/crpe_api.cpp',
                    'src/crpe_func.cpp',
                    'src/crpe_func_v2.cpp',
                    'src/rpe_k_kernel.cu',
                    'src/rpe_k_kernel_v2.cu',
                    'src/rpe_q_kernel.cu',
                    'src/rpe_q_kernel_v2.cu',
                    'src/rpe_v_kernel.cu',
                    'src/rpe_v_kernel_v2.cu',
                ]
            ),
        ],
    )
