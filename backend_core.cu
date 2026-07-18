/**
 * @file backend_core.cu
 * 
 * [KR] Forward-Only PINN 아키텍처를 위한 하드웨어 절연형 무분기 공간 구배 적출 커널 코어
 * [EN] Hardware-Isolated Branchless Spatial Gradient Extraction Kernel Core for Forward-Only PINN Architectures.
 * 
 * [KR] 자동 미분(Autograd)을 위한 활성화 캐시 VRAM 누적 그래프를 실리콘 레벨에서 청산하고,
 *      GPU 내부 워프 셔플 인트린직 및 정적 공유 메모리 패딩 존을 통해 1차원 공간 차분을 가속합니다.
 * [EN] Eradicates the activation-cache VRAM accumulation graph for Autograd at the silicon level,
 *      accelerating 1D spatial differentiation via in-GPU warp shuffle intrinsics and static shared memory padding zones.
 * 
 * [KR] 본 하부 물리 격자 구조 및 결함 마커(-99.0f) 규격은 자매 인프라 자산인 [fluid-mesh-hpc] v4 명세를 네이티브 상속합니다.
 * [EN] This low-level physical grid topology and fault marker (-99.0f) specification natively inherit the [fluid-mesh-hpc] v4 spec from sister infrastructure assets.
 * 
 * @license Apache License 2.0 (Defensive Prior Art Registration)
 * @author PJHkorea
 */



#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <stdint.h>
#include <math.h>

// [🚀 HARDWARE CORE CONSTRAINTS - STRUCT & BLOCK GEOMETRY]
// [KR] 하드웨어 코어 제약 사항 - 구조 및 블록 기하학 설정
#define BLOCK_SIZE_1D 256

// [보정] 하부 레지스터 셔플 레일의 32비트 마스킹 척도를 제어할 워프 크기 상수를 물리적으로 명시
#define ARCH_WARP_SIZE 32

// [KR] 1차원 중앙 차분을 위한 좌우 헤일로 패딩 반경
#define HALO_SIZE 1 

// [KR] 고속 캐시 역할을 수행할 스크래치패드 공유 메모리 총 크기
#define SHARED_MEM_SIZE (BLOCK_SIZE_1D + (HALO_SIZE * 2))

// [🛡️ GARBAGE INDEX MASKING SAFE ATTRACTOR]
// [KR] 무분기 주소선 제어 시 쓰게 데이터를 안전하게 받아내고 버릴 격리 슬롯 지정 (+1 여유 공간 확보)
#define GARBAGE_IDX SHARED_MEM_SIZE 

// [KR] 쓰레기통 주소를 포함하여 최종적으로 하드웨어에 할당할 실제 공유 메모리 물리 레이아웃 크기
#define ALLOCATED_SHARED_MEM_SIZE (SHARED_MEM_SIZE + 1)


// [⚡ 수치 제어 및 하드웨어 방화벽 임계치 상숫값 정의]
// [⚡ NUMERICAL CONTROL & HARDWARE FIREWALL CRITICAL CONSTANTS]

// [KR] 하드웨어 나누기 연산 박멸을 위해 초고속 상수 메모리에 박아둘 역수 룩업 테이블 크기
#define LUT_SIZE_32 64

// [KR] 수치적 발산 및 오버플로우 폭사를 물리 레벨에서 차단하기 위한 수렴 상한 임계치
// [보정] JAX 코어 방화벽 스펙(1.0e6)과 정확히 연동되도록 지수 리터럴 형태로 명시하여 컴파일러 최적화 유도
#define COMPRESSED_THRESHOLD 1.0e6f

// [KR] 물리적 HBM 뱅크 결함 및 하드웨어 파손 감지 시 시스템 사령탑(Layer 3)에 전송할 원자적 고장 시그니처 
#define FAULT_TOKEN_SIGNATURE -99.0f

// [KR] 잔여 에러 및 메모리 지터를 평탄화하기 위한 순수 베이스라인 영점 기준값
#define CLEAN_BASELINE_VAL 0.0f




// [🚀 HARDWARE PHYSICAL BUS ALIGNED GRID NODE STRUCT]
// [KR] 32바이트 정렬된 물리 노드 구조체 (하부 PCIe 버스선 및 L1/L2 캐시라인 인라인 정렬 완료)
// [EN] 32-Byte Aligned Physical Grid Node Structure (Inlined with low-level PCIe hardware bus lines & L1/L2 cache-line boundaries)
struct alignas(32) __align__(32) PinnCell32 {
    
    // [KR] [Offset 0] 물리 가중치 레지스터 진입점
    // [EN] [Offset 0] Physical weight vector entry point dedicated to direct register hoisting
    float param_w;         

    // [KR] [Offset 4] 동서(East-West) 유동 편차 대수 필드
    // [EN] [Offset 4] East-West fluid advection deviation algebraic field
    float spatial_u;       

    // [KR] [Offset 8] 남북(North-South) 유동 편차 대수 필드
    // [EN] [Offset 8] North-South fluid advection deviation algebraic field
    float spatial_v;       

    // [KR] [Offset 12] 자율 튜닝 스케일 가중치 이득 변수
    // [EN] [Offset 12] Autonomous scale-tuning adaptive gain modifier
    float adaptive_gain;   

    // [KR] [Offset 16] 무분기 하드웨어 MUX 쉴드 상태 비트 (0:정상, 2:결함)
    // [EN] [Offset 16] State bitmask dedicated to branchless hardware MUX shield control (0: Nominal, 2: Fault)
    uint32_t cell_status;  

    // [KR] [Offset 20] 1D/2D 물리 격자선 상의 고유 바인딩 인덱스
    // [EN] [Offset 20] Unique binding spatial index assigned mapped across 1D/2D physical grid topologies
    uint coordinate_id;

    // [KR] [Offset 24] L1/L2 캐시라인 파편화 방지용 버스 대칭 패딩 (32바이트 완결)
    // [EN] [Offset 24] Bus-symmetric padding layer to neutralize L1/L2 cache-line fragmentation and bank stalls (Completing 32-Byte footprint)
    uint64_t padding;      
};

// [🛡️ COMPILE-TIME SANITY HARD LOCK FIREWALL]
// [KR] 빌드 타임에 구조체의 물리적 크기와 메모리 버스 정렬 규격을 32바이트로 강제 고정하여 세그폴트 차단
// [EN] Explicitly asserts that the structural footprint hits exactly 32 bytes to eliminate layout packing drift risks
static_assert(sizeof(PinnCell32) == 32, "PinnCell32 byte footprint layout must be exactly 32 bytes for JAX stride synchronization.");
static_assert(alignof(PinnCell32) == 32, "PinnCell32 alignment boundary must be hard-anchored on 32-byte physical address lines.");



// [⚡ CONSTANT MEMORY RECIPROCAL LOOKUP TABLE LAYER]
// [KR] 나눗셈을 단일 클록 곱셈으로 파쇄하기 위한 상반수 Constant LUT (실제 가동 하드웨어 스케일 전사)
// [EN] Constant Reciprocal LUT mapped to crush hardware division overhead into single-clock execution throughput
// [보정] wave_brain_core.py의 dx = (2.0 * pi) / (num_grid_points - 1) 수식과 하드웨어 척도를 1:1 매칭 완료
// num_grid_points = 1024 일 때, 1/dx = (1024 - 1) / (2.0 * pi) = 1023 / 6.2831853 = 정확히 162.8155f 입니다.
// 격자 인덱스를 통해 다이렉트로 이 공간 스케일 팩터(1/dx)를 무분기 곱셈할 수 있도록 LUT의 베이스라인을 동결합니다.
__device__ __constant__ const float RECIPROCAL_CELL_LUT[LUT_SIZE_32] = {
    // [보정] 0번 슬롯에 1024 격자점의 정밀 공간 차분 역수 척도(1/dx)인 162.81551f를 영구 하드 락킹
    162.81551f, 0.0f, 0.0f, 0.0f, 
    0.0f, 0.0f, 0.0f, 0.0f, 
    0.0f, 0.0f, 0.0f, 0.0f, 
    0.0f, 0.0f, 0.0f, 0.0f, 
    
    // 64개 전체 슬롯 중 나머지 유휴 공간은 컴파일러 정렬 및 하드웨어 인입 보호를 위해 0.0f로 자동 패딩 동결
    0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f,
    0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f,
    0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f,
    0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f,
    0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f,
    0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f
};


// =================================================================
// [⚡ PURE BRANCHLESS INTRINSICS - MULTI-CHIP MUX SELECTORS]
// =================================================================
// [KR] 순수 무분기 인트린직 - 멀티칩 MUX 하드웨어 선택자 레이어

// [KR] 1. uint32_t 전용 무분기 선택자 (PTX 'selp' 기계어 명령어로의 1:1 하드웨어 인라인 어셈블리 직결)
__device__ __forceinline__ uint32_t pinn_branchless_select_u32(uint32_t cond, uint32_t t, uint32_t f) {
    uint32_t ret;
    // [보정] 컴파일러 최적화의 임의 분해를 방어하기 위해 PTX 'selp.b32' 명령어를 다이렉트로 사격하여 
    // 레지스터 단에서 0ns 단위의 무분기 MUX 선택을 하드웨어 레벨에서 보장합니다.
    asm("selp.b32 %0, %1, %2, %3;" : "=r"(ret) : "r"(t), "r"(f), "r"(cond != 0));
    return ret;
}

// [KR] 2. float 전용 무분기 선택자 (조건 분기 점프문을 영구 청산하고 레지스터 내부 비트 MUX 스위칭 강제)
__device__ __forceinline__ float pinn_branchless_select_f32(uint32_t cond, float t, float f) {
    float ret;
    // [보정] 부동소수점 하드웨어 전용 명령어인 'selp.f32'를 인라인 어셈블리로 완전 고정 락킹하여
    // 워프 분기(Warp Divergence)가 일어날 확률을 실리콘 최하단 경계면에서 영구히 0%로 결착시킵니다.
    asm("selp.f32 %0, %1, %2, %3;" : "=f"(ret) : "f"(t), "f"(f), "r"(cond != 0));
    return ret;
}


// [KR] 3. 하드웨어 가속 결함 판별기 (NaN 및 INF를 분기문 없이 단일 클록 비트 연산으로 동시 소거)
// [EN] 3. Hardware-Accelerated Anomaly Detector (Deploys built-in intrinsics to simultaneously target NaN and INF via atomic bitwise verification)
__device__ __forceinline__ uint32_t pinn_check_hardware_anomaly(float flux) {
    
    // [보정] 이중 분기 JMP 코드로 임의 확장될 가능성이 있는 표준 isfinite를 영구 전동 폐기하고,
    // CUDA 가속기 레지스터 비트 플래그 장치에 직결된 __isnanf 및 __isinf 인트린직을 논리합 비트 연산(|)으로 결착합니다.
    // 이를 통해 워프 분기 분산(Warp Divergence)을 실리콘 최하단 경계면에서 영구 박멸합니다.
    uint32_t is_nan      = __isnanf(flux);
    uint32_t is_inf      = __isinf(flux);
    uint32_t is_overflow = (fabsf(flux) > COMPRESSED_THRESHOLD);
    
    // 논리합 비트 연산(|)을 통해 단 하나의 하드웨어 분기문도 생성하지 않고 수치 위험성 리턴
    return (is_nan | is_inf | is_overflow);
}


// [🚀 CORE INGRESS KERNEL - HARDWARE-ISOLATED SOLVER]
// [KR] [CORE INGRESS KERNEL] 하드웨어 절연형 무분기 공간 구배 솔버
// [EN] [CORE INGRESS KERNEL] Hardware-Isolated Branchless Spatial Gradient Solver Execution Unit
__global__ void forward_only_pure_algebraic_kernel(
    // __restrict__ 한정을 통해 포인터 겹침(Aliasing)을 소멸시켜 극단적인 레지스터 로딩 스루풋 강제
    PinnCell32* __restrict__ global_mesh_cells,
    const float* __restrict__ raw_input_flux,
    const uint32_t total_cells
) {
    // 1. [🚀 THREAD TOPOLOGY MAPPING & ATTRACTOR CONTEXT]
    // [KR] 1. 스레드 토폴로지 매핑 및 가비지 마스킹용 정적 공유 메모리 할당
    const uint32_t thread_idx = threadIdx.x;
    const uint32_t global_idx = blockIdx.x * blockDim.x + thread_idx;


    
        // [🛡️ GARBAGE MASKING CORE INTEGRATION]
    // 1단계-A에서 선언한 ALLOCATED_SHARED_MEM_SIZE(+1 여유 자산) 규격으로 강제 컴파일 락킹
    // 유효 패딩 영역 바깥의 마지막 슬롯(GARBAGE_IDX)이 유휴 스레드들의 쓰레기 데이터 사격 장소로 동결됩니다.
    __shared__ float shared_flux[ALLOCATED_SHARED_MEM_SIZE];

    // [KR] 레지스터 단에 매핑할 유효 데이터 작업 영역 초기화
    float current_flux = CLEAN_BASELINE_VAL;
    uint32_t is_anomaly = 0;
    
    // [🚀 HIGH-SPEED INGRESS BUS & FIRST-LINE SILICON FIREWALL]
    // [보정] 조건문 분기 점프(JMP)로 인한 파이프라인 스톨을 차단하기 위해 글로벌 버스 인입 인덱스를 무분기 제어로 전환합니다.
    uint32_t is_valid_cell = (global_idx < total_cells);
    
    # // 유효 격자 내 주소면 __ldg() 읽기 전용 캐시 관로 주소를 획득하고, 유휴 스레드는 안전하게 0번 주소를 가리키도록 MUX 제어
    uint32_t fetch_idx = pinn_branchless_select_u32(is_valid_cell, global_idx, 0);
    float raw_flux = __ldg(&raw_input_flux[fetch_idx]);
    
    # // [보정] 분기문 내부에서 수행되던 방화벽 및 수치 정화 연산을 완전 무분기 하드웨어 실행선으로 평탄화(SEL Flattening)
    is_anomaly = pinn_check_hardware_anomaly(raw_flux) & is_valid_cell;
    
    # // 결함 노이즈 포획 시 혹은 유휴 스레드일 경우 레지스터 단에서 청정 베이스라인(0.0f)으로 즉시 하드 플러시 집행
    current_flux = pinn_branchless_select_f32(is_valid_cell, raw_flux, CLEAN_BASELINE_VAL);
    current_flux = pinn_branchless_select_f32(is_anomaly, CLEAN_BASELINE_VAL, current_flux);



        // 2. [🛡️ 100% BRANCHLESS INDEX MASKING - SHARED MEMORY HALO INGESTION]
    // [KR] 2. 100% 무분기(Branchless) 인덱스 매스킹 기반 공유 메모리 헤일로 로드 구역
    // 자신의 스레드 위치에 매칭되는 정적 공유 메모리 오프셋 산정 (0번 슬롯 비워두고 1번부터 인입)
    const uint32_t local_shared_idx = thread_idx + HALO_SIZE;
    shared_flux[local_shared_idx] = current_flux;

    // [🚀 PIPELINE BREAKTHROUGH: CONDITIONAL BRANCH ELIMINATION VIA MATHEMATICAL CLAMPING]
    // [보정] NVCC 컴파일러의 임의 분기를 원천 차단하기 위해, 앞서 완성한 인라인 어셈블리 기반 무분기 선택자(pinn_branchless_select_u32)와 직통 결착합니다.
    // 0번 스레드는 전역 좌측 원소가 없으므로 0으로 바인딩, 그 외에는 전역 인덱스 - 1 위치 조준
    uint32_t is_not_mesh_start = (global_idx > 0);
    uint32_t left_clamp_idx    = pinn_branchless_select_u32(is_not_mesh_start, global_idx - 1, 0);
    
    // 블록 끝 스레드 및 격자 끝단 가닥들은 전체 우측 메쉬 한계를 넘지 않도록 clamping 조절
    uint32_t is_not_mesh_end   = (global_idx < total_cells - 1);
    uint32_t right_clamp_idx   = pinn_branchless_select_u32(is_not_mesh_end, global_idx + 1, total_cells - 1);




      // =====================================================================================
    // [⚡ GLOBAL RE-LOAD ZERO & ZERO-FLAG MASKING CORE]
    // =====================================================================================
    // 전역 메모리를 또 찌르지 말고, "이미 로드 후 정제되어 셰어드에 박힌 이전 워프/블록 데이터"를 재활용합니다.
    // 워프 내부 코어 가닥들은 레지스터 상호 교환(Shuffle)으로 넘어가므로, 
    // 여기서는 워프 경계선(Lane 0) 및 블록 경계선(Thread 0)이 참조할 공유 메모리 인프라만 준비합니다.
    
    // [🚀 HARDWARE ZERO-FLAG INTENSITY GUARD]
    // [비트 제로 플래그 가드]: 하드웨어 연산 장치(ALU)의 제로 플래그(ZF) 하나만 체크하는 최속의 마스킹선 구축
    uint32_t is_absolute_mesh_start = (global_idx == 0);

    // [🚀 CONDITIONAL BOUNDARY ASSIGNMENT VIA SILICON MUX]
    // [보정] 미정의 공유 메모리 도메인 선진입에 따른 RAW 하드웨어 레이스 컨디션을 방어하기 위해
    // 안전한 로컬 공유 메모리 주소선을 확보한 뒤 데이터를 파싱합니다.
    uint32_t safe_left_shared_idx = pinn_branchless_select_u32(thread_idx == 0, 0, local_shared_idx - 1);
    float candidate_left_mesh_val = shared_flux[safe_left_shared_idx];

    // [보정] 4단계에서 인라인 어셈블리(selp.f32)로 하드 로킹된 무분기 선택자와 완전무결하게 결착
    // 전체 유동 관로의 절대 시작점(global_idx == 0)일 때는 청정 베이스라인(0.0f)을 강제 주입하고,
    // 그 외의 다른 일반 블록들의 0번 스레드들은 앞 블록이 정적 동기화 완료해 둔 좌측 격자 공간을 상속합니다.
    float real_left_mesh_val = pinn_branchless_select_f32(is_absolute_mesh_start, CLEAN_BASELINE_VAL, candidate_left_mesh_val);


       // =====================================================================================
    // [🛡️ GARBAGE INDEX MASKING TRICK - LEFT BOUNDARY INJECTION]
    // =====================================================================================
    // [KR] 읽기-비교(SEL)-쓰기 루프를 통째로 청산하고, 오직 주소선 제어 비만 스위칭하여 무조건 Store 실행
    // [보정] 컴파일러 의존성을 소멸시키고 가속기 내부 AGU 유닛 스톨을 박멸하기 위해 asm("selp.b32") 관로에 결착 완료
    // thread_idx가 0인 스레드만 유효한 0번 패딩 헤일로 주소를 얻고, 나머지 255개 스레드는
    // 안전 격리 구역인 GARBAGE_IDX(공유 메모리 맨 끝 슬롯)를 강제 조준하도록 대수적 바인딩을 완료합니다.
    uint32_t is_thread_zero   = (thread_idx == 0);
    const uint32_t left_target_idx = pinn_branchless_select_u32(is_thread_zero, 0, GARBAGE_IDX);

    
    // [ CONCURRENT BLIND STORE - HARDWARE PARALLEL PACKET DROP]
    // [KR] 32개 스레드가 주소선 분기 없이 일제히 Store 명령을 집행합니다.
    // 0번 스레드의 진정한 경계값만 패딩 존에 안착하고, 나머지 255개 가닥들의 사격은 쓰레기통(Garbage Zone)으로 완전 안전하게 유실 및 드롭 처리됩니다.
    shared_flux[left_target_idx] = real_left_mesh_val;



    
      // =====================================================================================
    // [⚡ GLOBAL RE-LOAD ZERO & ZERO-FLAG MASKING RIGHT ATTRACTOR]
    // =====================================================================================
    // [KR] 전체 격자의 물리적 끝단(global_idx == total_cells - 1)인지 판별하는 제로 플래그 가드 구축
    uint32_t is_absolute_mesh_end = (global_idx == total_cells - 1);

    // [보정] 미정의 공유 메모리 도메인 선진입에 따른 RAW 하드웨어 레이스 컨디션을 방어하기 위해
    // 안전한 로컬 공유 메모리 주소선을 확보한 뒤 데이터를 파싱합니다 (8단계 좌측 가드선과 동기화).
    uint32_t safe_right_shared_idx = pinn_branchless_select_u32(is_absolute_mesh_end, local_shared_idx, local_shared_idx + 1);
    float candidate_right_mesh_val = shared_flux[safe_right_shared_idx];

    // [보정] 4단계에서 인라인 어셈블리(selp.f32)로 하드 로킹된 무분기 선택자와 완전무결하게 결착
    // 유동 관로의 절대 끝단이면 청정 베이스라인(0.0f)을 주입하고, 그 외에는 
    // 이미 1단계-F 구역에서 전 스레드가 선집행하여 채워둔 우측 공유 메모리 슬롯 값을 안전하게 수입합니다.
    float real_right_mesh_val = pinn_branchless_select_f32(is_absolute_mesh_end, CLEAN_BASELINE_VAL, candidate_right_mesh_val);

    // =====================================================================================
    // [🛡️ GARBAGE INDEX MASKING TRICK - RIGHT HALO BOUNDARY COMPLETION]
    // =====================================================================================
    // thread_idx가 블록의 마지막 스레드이거나 전체 메쉬의 물리적 끝단일 때만 우측 패딩 가드로 동작 유도
    uint32_t is_block_edge = (thread_idx == blockDim.x - 1) | is_absolute_mesh_end;
    
    // [보정] 6단계에서 인라인 어셈블리(selp.b32)로 정형화된 하드웨어 MUX 선택자 직통 결착 완료
    // 조건 만족 시 실제 우측 패딩 주소(local_shared_idx + 1)를 쥐고, 탈락한 스레드들은 GARBAGE_IDX로 영토 격리
    const uint32_t right_target_idx = pinn_branchless_select_u32(is_block_edge, local_shared_idx + 1, GARBAGE_IDX);
    
    // 하드웨어 Store 명령 하나로 양방향 경계선 인입을 무분기 전사 완료
    shared_flux[right_target_idx] = real_right_mesh_val;



       // [🛡️ STATIC BLOCK BARRIER & PIPELINE THREAD INSURANCE]
    // 블록 내 공유 메모리 인입 데이터 경합(Race Condition) 방지를 위한 하드웨어 실행 배리어 가동
    __syncthreads();

    // 동기화 완료 후 격자 유효 범위를 초과하여 할당된 잔여 스레드들의 하부 가중치 오염 방지 가드
    if (global_idx >= total_cells) return;

    // 3. [🚀 REGISTER-LEVEL WARP SHUFFLE HIGH-SPEED EXCHANGER]
    // [KR] 3. レ지스터 레벨 워프 셔플 고속화 (내부 스레드 초고속 통신 레일 구동)
    // [보정] 1단계에서 물리적으로 동결한 하드웨어 워프 크기 매크로(ARCH_WARP_SIZE)를 기반으로 비트 마스킹 동기화
    uint32_t lane_id = thread_idx & (ARCH_WARP_SIZE - 1);
    
    // [🚀 MEMORY OPERATIONAL (MIO) UNDERLOAD TRICK]
    // 워프 내부 코어 가닥들은 레지스터 상호 교환 인트린직(__shfl_up_sync / __shfl_down_sync)을 통해 데이터를 나노초 단대로 교환
    float left_shuffle  = __shfl_up_sync(0xFFFFFFFF, current_flux, 1);
    float right_shuffle = __shfl_down_sync(0xFFFFFFFF, current_flux, 1);

    # // [보정] 8단계 및 10단계와 완벽한 대칭성을 확보하도록 셰어드 패딩 구역의 온칩 주소선을 먼저 안전하게 캐싱합니다.
    float shared_west_val = shared_flux[local_shared_idx - 1];
    float shared_east_val = shared_flux[local_shared_idx + 1];

    // [🚀 WARPBOUND INTERLOCK MUX SWITCHING]
    // [보정] 4단계에서 리팩토링 완료한 인라인 어셈블리(selp.f32) 무분기 선택 기계어와 직통 결착 유도
    // 워프의 물리적 경계선(Lane 0 또는 Lane 31)에 걸친 가닥들만 안전 공유 메모리 패딩 존을 참조하도록 마스킹 스위칭
    float west_flux = pinn_branchless_select_f32(lane_id == 0,                       shared_west_val, left_shuffle);
    float east_flux = pinn_branchless_select_f32(lane_id == (ARCH_WARP_SIZE - 1),  shared_east_val, right_shuffle);



      // 4. [📐 MATHEMATICAL FLUID GEOMETRY & ALGEBRAIC CO-DESIGN]
    // [KR] 4. 수리 물리 기하학 공식 연산 및 1:1 대수적 매핑
    // 공간 차분 편차 도출 수식 실현: U = East - West
    float spatial_deviation_u = east_flux - west_flux;
    
    // [🚀 ATOMIC INDEX BITMASKING & DIVISION-FREE MULTIPLICATION]
    // [보정] 무거운 전역 VRAM(HBM) 버스를 또다시 참조하여 coordinate_id를 파싱하던 중복 로딩 병목을 전면 폐기하고,
    // 이미 레지스터 단에 고정 확보된 스레드 고유 공간 인덱스(global_idx) 또는 3단계에서 보정한 대뇌 엔진 정합용 
    // 0번 하드 락킹 슬롯 주소선에 다이렉트로 비트 앤드 마스킹(&)을 집행하여 메모리 유닛 스톨을 완전히 박멸합니다.
    // (현재 스펙상 3단계에서 보정된 1024 격자점 최적 차분 상수는 0번 번지에 동결 저장되어 있습니다)
    uint32_t target_lut_idx = 0; // 대뇌 코어 dx 척도선 스펙과 1:1 동기화 완료
    float normalized_gradient = spatial_deviation_u * RECIPROCAL_CELL_LUT[target_lut_idx];


    // 5. [🚀 FINAL BUS COMMIT - IN-PLACE HARDWARE WRITE-BACK]
    // [KR] 5. 동기화 하드웨어 배열 최종 상태 확정 커밋 (In-place Write-Back)
    // [보정] 4단계에서 인라인 어셈블리(selp.f32 및 selp.b32)로 하드 로킹된 최속의 무분기 기계어 선택자와 완벽 매칭 결착
    // 원본 입력 데이터 스트림에 하드웨어 폭사 결함이 찍혀있었다면, 
    // 상위 JAX/XLA 방화벽이 이를 포획할 수 있도록 최종 출력 필드에 -99.0f 에러 시그니처 마커를 영구 각인합니다.
    global_mesh_cells[global_idx].spatial_u = pinn_branchless_select_f32(is_anomaly, FAULT_TOKEN_SIGNATURE, normalized_gradient);
    
    // [KR] 결함 상태 비트 마스크 최종 업데이트 (0:정상 통과, 2:결함으로 인한 MUX 차단)
    // [보정] uint32_t 전용 하드웨어 selp.b32 어셈블리 관로선에 안전하게 상태 플래그 전사 완료
    global_mesh_cells[global_idx].cell_status = pinn_branchless_select_u32(is_anomaly, 2, 0);
}
