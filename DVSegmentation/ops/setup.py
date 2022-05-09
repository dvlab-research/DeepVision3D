from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='PG_OP',
    ext_modules=[
        CUDAExtension(
            'PG_OP',
            ['src/ops_api.cpp',
             'src/ops.cpp',
             'src/cuda.cu'],
            extra_compile_args={'cxx': ['-g'], 'nvcc': ['-O2']},
            include_dirs=['/mnt/backup2/home/ztyang/anaconda3/envs/deepvision3d/include/'],
        )
    ],
    cmdclass={'build_ext': BuildExtension}
)