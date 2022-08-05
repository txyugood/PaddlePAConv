from paddle.utils.cpp_extension import CUDAExtension, setup

setup(
    name='custom_ops',
    ext_modules=CUDAExtension(
        sources=['assign_score_withk_cuda.cpp', 'assign_score_withk_kernel.cu'],
        extra_compile_args={"nvcc":['-arch=sm_60']}
    )
)