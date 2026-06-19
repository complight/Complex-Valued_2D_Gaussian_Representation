import os
from torch.utils.cpp_extension import load

def build_cuda_module():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    
    compile_dir = os.path.join(current_dir, 'compile')
    if not os.path.exists(compile_dir):
        os.makedirs(compile_dir)
    
    cpp_dir = os.path.join(current_dir, 'cpp')
    cuda_dir = os.path.join(current_dir, 'cuda')
    
    cpp_sources = [
        os.path.join(cpp_dir, 'bandlimited_propagation.cpp'),
        os.path.join(cpp_dir, 'batch_matrix_mult.cpp'),
        os.path.join(cpp_dir, 'sum_last_dim.cpp'),
        os.path.join(cpp_dir, 'element_wise_mult.cpp'),
        os.path.join(cpp_dir, 'bandlimited_propagation_me.cpp'),
        os.path.join(cpp_dir, 'module_def.cpp')
    ]
    
    cuda_sources = [
        os.path.join(cuda_dir, 'bandlimited_propagation_cuda.cu'),
        os.path.join(cuda_dir, 'batch_matrix_mult_cuda.cu'),
        os.path.join(cuda_dir, 'sum_last_dim_cuda.cu'),
        os.path.join(cuda_dir, 'element_wise_mult_cuda.cu'),
        os.path.join(cuda_dir, 'bandlimited_propagation_cuda_me.cu'),
    ]
    
    sources = cpp_sources + cuda_sources
    extra_cuda_flags = [
        '-O3',
        '--generate-code=arch=compute_80,code=sm_80',  # A100
        '--generate-code=arch=compute_86,code=sm_86',  # 3090 / A6000
        # '--generate-code=arch=compute_89,code=sm_89',  # 4090
        # '--generate-code=arch=compute_90,code=sm_90'   # H100
    ]
    return load(
        name="cuda_modules",
        sources=sources,
        extra_cflags=['-O3'],
        extra_cuda_cflags=extra_cuda_flags,
        build_directory=compile_dir,
        verbose=True,
        is_python_module=True
    )

if __name__ == "__main__":
    build_cuda_module()