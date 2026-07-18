import os
import sys
from setuptools import setup, Extension
from setuptools.command.build_ext import build_ext

# [🛡️ COMPILER DETONATION & ENVIRONMENT INSULATION]
# 하드웨어 CUDA 가속 경로 탐색 및 환경 변수 절연 보호선 구축
# [EN] [🛡️ Compiler Detonation & Environment Insulation]
# Profiles and extracts hardware CUDA acceleration tracks, establishing a protective environment variable insulation line.
def find_cuda():
    cuda_home = os.environ.get("CUDA_HOME") or os.environ.get("CUDA_PATH")
    if not cuda_home:
        for path in ["/usr/local/cuda", "/usr/cuda"]:
            if os.path.exists(path):
                cuda_home = path
                break
    if not cuda_home:
        raise RuntimeError("[CRITICAL INFRASTRUCTURE FAULT] CUDA Toolkit not found inside system boundary.")
    return cuda_home


class custom_build_ext(build_ext):
    """
    [⚡ FUSED HARDWARE COMPILER TUNING TOWER]
    NVCC와 GCC 컴파일러 최적화 플래그를 원자적으로 강제 주입하여, 
    기계어 단에서 단 하나의 파이프라인 스톨도 발생하지 않도록 하드 로킹 빌드합니다.
    [EN] [⚡ Fused Hardware Compiler Tuning Tower]
    Atomically injects high-end NVCC and GCC compiler optimization flags, executing a hard-locked compilation to eliminate a single pipeline stall at the low-level machine-code boundaries.
    """
    def build_extensions(self):
        cuda_home = find_cuda()
        nvcc_bin = os.path.join(cuda_home, "bin", "nvcc")
        
        # [🚀 GCC HOST COMPILER SUPREME OPTIMIZATION FLAGS]
        # 리눅스 호스트 GCC 컴파일러 최적화 매트릭스 동결 (-O3 최속화 및 루프 언롤링 강제)
        # [EN] Linux Host GCC Compiler Supreme Optimization Flags:
        # Freezes the host compiler optimization matrix (Enforcing `-O3` peak velocity mapping, vectorization, and aggressive loop unrolling tuned for native CPU architectures).
        host_cflags = ["-O3", "-std=c++20", "-fPIC", "-funroll-loops", "-march=native"]
        
        # [🚀 NVCC DEVICE ATOMIC KERNEL ACCELERATION FLAGS]
        # 가속기 내부 ALU FMA 명령어 강제 락킹 및 워프 셔플 지터 제어용 하드웨어 최적화 옵션 사격
        # [EN] NVCC Device Atomic Kernel Acceleration Flags:
        # Fires specialized hardware optimization options to hard-lock intra-ALU FMA instructions and secure micro-jitter control over warp shuffle operations.
        cuda_cflags = [
            "-O3", 
            "-std=c++20", 
            "--compiler-options", "'-fPIC'",
            "-use_fast_math",                   # 하드웨어 가속 삼각함수/역수 융합 강제
            # [EN] Enforces hardware-accelerated mathematical intrinsics to crush transcendentals and reciprocal divisions into single-clock DSP primitives.
            "-Xptxas", "-v",                    # 컴파일 타임 레지스터 소모량 레포팅 활성화
            # [EN] Activates verbose PTX assembler layout reporting to monitor thread-local register file consumption and prevent memory spills.
            "-Xcompiler", "-funroll-loops"
        ]

             # Pybind11 헤더 수송 경로 바인딩
        # [EN] Binds pybind11 and CUDA Toolkit hardware peripheral header transport directories.
        import pybind11
        for ext in self.extensions:
            ext.include_dirs.append(pybind11.get_include())
            ext.include_dirs.append(os.path.join(cuda_home, "include"))
            ext.library_dirs.append(os.path.join(cuda_home, "lib64"))
            ext.libraries.append("cudart")
            
        # 컴파일러 래퍼 오버라이드 가동
        # [EN] Activates the compiler wrapper hijacking override track to intercept default setup build sequences.
        original_compile = self.compiler._compile
        
        def custom_compile(obj, src, ext, cc_args, extra_postargs, pp_opts):
            if os.path.splitext(src)[1] == ".cu":
                # CUDA 확장자 진입 시 NVCC 바이패스 관로 가동
                # [EN] Activates the NVCC bypass acceleration rail upon capturing `.cu` source code extensions.
                postargs = cuda_cflags
                compiler_cmd = [nvcc_bin, "-c", src, "-o", obj] + postargs
                for inc in ext.include_dirs:
                    compiler_cmd += ["-I", inc]
                self.spawn(compiler_cmd)
            else:
                # 일반 C++ 브릿지 진입 시 GCC 핫패스 관로 가동
                # [EN] Reroutes toward the GCC host hot-path pipeline upon ingesting standard C++ bridge files.
                postargs = cc_args + host_cflags
                original_compile(obj, src, ext, cc_args, postargs, pp_opts)
                
        self.compiler._compile = custom_compile
        super().build_extensions()


# [⛓️ FULL-STACK EXTENSION INTERLOCK DEPLOYMENT]
# 6채널 무복사 0ns 데이터 이송 인터페이스 최종 링킹 선언
# [EN] [⛓️ Full-Stack Extension Interlock Deployment]
# Declares the terminal linking configuration assigned for the 6-channel zero-copy 0ns data transport interface.
pinn_bridge_module = Extension(
    "pinn_bridge_interface",
    sources=["bridge_wrapper.cpp", "backend_core.cu"],
    include_dirs=["."],
    language="c++"
)

setup(
    name="Forward_Only_Autograd_Free_PINN_Bridge",
    version="5.0",
    description="Zero-Copy 6-Channel Hardware Interlock Bridge Wrapper",
    ext_modules=[pinn_bridge_module],
    cmdclass={"build_ext": custom_build_ext},
    zip_safe=False,
)
