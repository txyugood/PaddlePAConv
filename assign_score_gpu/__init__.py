from paddle.utils.cpp_extension import load
import paddle

__all__ = ['assign_score_withk']

if paddle.device.get_device() == 'cpu':
    custom_ops = load(
        name="assign_score_withk",
        sources=[
            "assign_score_gpu/assign_score_withk_cuda.cc"
        ]
    )
else:
    custom_ops = load(
        name="assign_score_withk",
        sources=[
            "assign_score_gpu/assign_score_withk_cuda.cc", "assign_score_gpu/assign_score_withk_kernel.cu"
        ],
        extra_cuda_cflags=['-arch=sm_60', '-DPADDLE_WITH_CUDA'],
        extra_cxx_cflags=['-DPADDLE_WITH_CUDA']
    )
assign_score_withk = custom_ops.assign_score_withk
