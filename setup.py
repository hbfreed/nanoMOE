from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='nanomoe_ops',
    packages=[],  # Don't auto-discover packages, we're only building a C++ extension
    ext_modules=[
        CUDAExtension(
            name='nanomoe_ops',
            sources=['nanomoe_ops.cu'],
            extra_compile_args={
                'cxx': ['-O3'],
                'nvcc': ['-O3', '--use_fast_math']
            }
        )
    ],
    cmdclass={
        'build_ext': BuildExtension
    }
)
