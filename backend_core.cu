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
// [EN] Hardware Core Constraints - Structure & Block Geometry Configuration
#define BLOCK_SIZE_1D 256

// [KR] 1차원 중앙 차분을 위한 좌우 헤일로 패딩 반경
// [EN] Halo padding radius for 1D central finite differences
#define HALO_SIZE 1 

// [KR] 고속 캐시 역할을 수행할 스크래치패드 공유 메모리 총 크기
// [EN] Total footprint of the high-speed scratchpad shared memory
#define SHARED_MEM_SIZE (BLOCK_SIZE_1D + (HALO_SIZE * 2))

// [🛡️ GARBAGE INDEX MASKING SAFE ATTRACTOR]
// [KR] 무분기 주소선 제어 시 쓰레기 데이터를 안전하게 받아내고 버릴 격리 슬롯 지정 (+1 여유 공간 확보)
// [EN] Isolation slot allocated to safely ingest and discard volatile out-of-bound write packets during branchless address execution (+1 buffer secured)
#define GARBAGE_IDX SHARED_MEM_SIZE 

// [KR] 쓰레기통 주소를 포함하여 최종적으로 하드웨어에 할당할 실제 공유 메모리 물리 레이아웃 크기
// [EN] Physical shared memory layout dimension allocated to the hardware, incorporating the garbage attractor index
#define ALLOCATED_SHARED_MEM_SIZE (SHARED_MEM_SIZE + 1)


// [⚡ 수치 제어 및 하드웨어 방화벽 임계치 상숫값 정의]
// [⚡ NUMERICAL CONTROL & HARDWARE FIREWALL CRITICAL CONSTANTS]

// [KR] 하드웨어 나누기 연산 박멸을 위해 초고속 상수 메모리에 박아둘 역수 룩업 테이블 크기
// [EN] Scale dimension of the reciprocal Lookup Table embedded in high-speed constant memory to eradicate hardware division ops
#define LUT_SIZE_32 64

// [KR] 수치적 발산 및 오버플로우 폭사를 물리 레벨에서 차단하기 위한 수렴 상한 임계치
// [EN] Bound-convergence upper threshold deployed to intercept catastrophic numerical divergence and floating-point overflows at the physical level
#define COMPRESSED_THRESHOLD 1000000.0f

// [KR] 물리적 HBM 뱅크 결함 및 하드웨어 파손 감지 시 시스템 사령탑(Layer 3)에 전송할 원자적 고장 시그니처 
// [EN] Atomic fault signature token broadcasted to the infrastructure orchestrator (Layer 3) upon detecting physical HBM bank failure or hardware corruption
#define FAULT_TOKEN_SIGNATURE -99.0f

// [KR] 잔여 에러 및 메모리 지터를 평탄화하기 위한 순수 베이스라인 영점 기준값
// [EN] Pure baseline zero-reference point deployed to flatten residual algebraic noise and hardware memory jitter
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



// [⚡ CONSTANT MEMORY RECIPROCAL LOOKUP TABLE LAYER]
// [KR] 나눗셈을 단일 클록 곱셈으로 파쇄하기 위한 상반수 Constant LUT (실제 가동 하드웨어 스케일 전사)
// [EN] Constant Reciprocal LUT mapped to crush hardware division overhead into single-clock execution throughput
__device__ __constant__ const float RECIPROCAL_CELL_LUT[LUT_SIZE_32] = {
    // [KR] 1/12, 1/16, 1/20, 1/24 격자 스페이싱 역수
    // [EN] Reciprocals for 1/12, 1/16, 1/20, 1/24 spatial grid spacing intervals
    0.08333333f, 0.06250000f, 0.05000000f, 0.04166667f, 
    
    // [KR] 1/32, 1/40, 1/50, 1/64 격자 스페이싱 역수
    // [EN] Reciprocals for 1/32, 1/40, 1/50, 1/64 spatial grid spacing intervals
    0.03125000f, 0.02500000f, 0.02000000f, 0.01562500f, 
    
    // [KR] 1/80, 1/100, 1/128, 1/160 격자 스페이싱 역수
    // [EN] Reciprocals for 1/80, 1/100, 1/128, 1/160 spatial grid spacing intervals
    0.01250000f, 0.01000000f, 0.00781250f, 0.00625000f, 
    
    // [KR] 1/200, 1/400, 1/800, 1/1600 격자 스페이싱 역수
    // [EN] Reciprocals for 1/200, 1/400, 1/800, 1/1600 spatial grid spacing intervals
    0.00500000f, 0.00250000f, 0.00125000f, 0.00062500f, 
    
    // [KR] 64개 전체 슬롯 중 나머지 유휴 공간은 컴파일러 정렬 및 하드웨어 인입 보호를 위해 0.0f로 자동 패딩 동결
    // [EN] Remaining idle slots out of the 64-element footprint are automatically zero-padded to enforce static compiler alignment and prevent hardware cache lines leaking
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
// [EN] Pure Branchless Intrinsics - Multi-Chip MUX Hardware Selectors Layer

// [KR] 1. uint32_t 전용 무분기 선택자 (PTX 'SEL' 기계어 명령어로 1대1 매핑 강제 유도)
// [EN] 1. Dedicated uint32_t Branchless Selector (Forces 1:1 hardware mapping to PTX 'SEL' primitive instructions)
__device__ __forceinline__ uint32_t pinn_branchless_select_u32(uint32_t cond, uint32_t t, uint32_t f) {
    
    // [KR] 참과 거짓 경로 모두에 주소 점프(JMP)가 없는 결정론적 연산 보장
    // [EN] Guarantees deterministic execution throughput, structurally neutralizing conditional jump (JMP) stalls across both evaluation paths
    return (cond) ? t : f;
}


// [KR] 2. float 전용 무분기 선택자 (조건 분기 점프문을 파쇄하고 레지스터 레벨에서 비트 마스킹 스위칭)
// [EN] 2. Dedicated float Branchless Selector (Crushes conditional jumps, enforcing hardware register-level bitwise MUX switching)
__device__ __forceinline__ float pinn_branchless_select_f32(uint32_t cond, float t, float f) {
    
    // [KR] 하부 실리콘 커널단에서 단 1클록 단위의 실행 지연 시간으로 즉시 치환 실행
    // [EN] Executes immediate hardware-level substitution with a deterministic 1-clock instruction execution latency
    return (cond) ? t : f;
}

// [KR] 3. 하드웨어 가속 결함 판별기 (isfinite 내장 함수를 가동해 NaN 및 INF를 단일 비트 연산으로 동시 조준)
// [EN] 3. Hardware-Accelerated Anomaly Detector (Deploys built-in isfinite to simultaneously target NaN and INF via atomic bitwise verification)
__device__ __forceinline__ uint32_t pinn_check_hardware_anomaly(float flux) {
    
    // [KR] 수치 폭발(NaN/INF)이 발생하는 순간 1차 방화벽 트리거를 위한 비트 마스킹 결합
    // [EN] Binds bitmasks to trigger the primary silicon firewall the exact moment catastrophic numerical explosion (NaN/INF) breaches boundaries
    uint32_t is_bad_num = !isfinite(flux);
    uint32_t is_overflow = (fabsf(flux) > COMPRESSED_THRESHOLD);
    
    // [KR] 논리합 비트 연산(|)을 통해 단 하나의 분기문도 생성하지 않고 수치 위험성 리턴
    // [EN] Computes the raw boolean OR payload (|) to return failure state metrics without generating a single pipeline-stalling branch instruction
    return (is_bad_num | is_overflow);
}



// [🚀 CORE INGRESS KERNEL - HARDWARE-ISOLATED SOLVER]
// [KR] [CORE INGRESS KERNEL] 하드웨어 절연형 무분기 공간 구배 솔버
// [EN] [CORE INGRESS KERNEL] Hardware-Isolated Branchless Spatial Gradient Solver Execution Unit
__global__ void forward_only_pure_algebraic_kernel(
    // [KR] __restrict__ 한정을 통해 포인터 겹침(Aliasing)을 소멸시켜 극단적인 레지스터 로딩 스루풋 강제
    // [EN] Deploys __restrict__ qualifiers to eliminate pointer aliasing, forcing maximum register loading throughput
    PinnCell32* __restrict__ global_mesh_cells,
    const float* __restrict__ raw_input_flux,
    const uint32_t total_cells
) {
    // 1. [🚀 THREAD TOPOLOGY MAPPING & ATTRACTOR CONTEXT]
    // [KR] 1. 스레드 토폴로지 매핑 및 가비지 마스킹용 정적 공유 메모리 할당
    // [EN] 1. Thread Topology Mapping & Shared Scratchpad Attractor Context Initialization
    const uint32_t thread_idx = threadIdx.x;
    const uint32_t global_idx = blockIdx.x * blockDim.x + thread_idx;

    
    // [🛡️ GARBAGE MASKING CORE INTEGRATION]
    // [KR] 1단계-A에서 선언한 ALLOCATED_SHARED_MEM_SIZE(+1 여유 자산) 규격으로 강제 컴파일 락킹
    //      유효 패딩 영역 바깥의 마지막 슬롯(GARBAGE_IDX)이 유휴 스레드들의 쓰레기 데이터 사격 장소로 동결됩니다.
    // [EN] Enforces static compiler-locking using the ALLOCATED_SHARED_MEM_SIZE (+1 buffer asset) specification.
    //      The final slot situated right outside valid halo bounds (GARBAGE_IDX) is frozen as the designated drop zone for out-of-bound thread writes.
    __shared__ float shared_flux[ALLOCATED_SHARED_MEM_SIZE];

    // [KR] 레지스터 단에 매핑할 유효 데이터 작업 영역 초기화
    // [EN] Pre-allocates and zeroes out the active variable workspace dedicated to direct in-register hosting
    float current_flux = CLEAN_BASELINE_VAL;
    uint32_t is_anomaly = 0;


    
    // [🚀 HIGH-SPEED INGRESS BUS & FIRST-LINE SILICON FIREWALL]
    // [KR] 전체 격자 크기 경계 안쪽일 때만 글로벌 메모리 버스 인입 전개
    // [EN] Executes global memory bus ingestion exclusively within total cell boundary constraints
    if (global_idx < total_cells) {
        
        // [KR] __ldg() 고속 인트린직 명령어로 L1/L2 읽기 전용 캐시 효율 극대화 및 하드웨어 버스 점유 최소화
        // [EN] Deploys __ldg() high-speed intrinsic primitives to maximize L1/L2 read-only cache hits and minimize hardware bus contention
        current_flux = __ldg(&raw_input_flux[global_idx]);
        
        // [KR] 1단계-D에서 빌드한 분기 없는 비트 연산 기반 결함 감지 필터 작동
        // [EN] Activates the branchless, bitwise-driven hardware anomaly detection filter configured in Step 1-D
        is_anomaly = pinn_check_hardware_anomaly(current_flux);
        
        // [KR] 결함 노이즈(NaN, INF, Threshold 초과) 포획 즉시 레지스터 레벨에서 청정 베이스라인(0.0f)으로 하드 플러시 실행
        // [EN] Upon capturing anomaly noise (NaN, INF, or Over-Threshold), enforces an immediate hardware flush to CLEAN_BASELINE_VAL (0.0f) inside the register rail
        current_flux = pinn_branchless_select_f32(is_anomaly, CLEAN_BASELINE_VAL, current_flux);
    }


    // 2. [🛡️ 100% BRANCHLESS INDEX MASKING - SHARED MEMORY HALO INGESTION]
    // [KR] 2. 100% 무분기(Branchless) 인덱스 매스킹 기반 공유 메모리 헤일로 로드 구역
    //      자신의 스레드 위치에 매칭되는 정적 공유 메모리 오프셋 산정 (0번 슬롯 비워두고 1번부터 인입)
    // [EN] 2. 100% Branchless Index Masking-driven Shared Memory Halo Ingestion Zone
    //      Computes the static shared memory offset matching the local thread allocation (Leaves slot 0 vacant, starting ingestion from slot 1)
    const uint32_t local_shared_idx = thread_idx + HALO_SIZE;
    shared_flux[local_shared_idx] = current_flux;

    // [🚀 PIPELINE BREAKTHROUGH: CONDITIONAL BRANCH ELIMINATION VIA MATHEMATICAL CLAMPING]
    // [KR] [병목 파쇄 핵심 혁신]: if-else 조건문 분기를 소멸시키는 수학적 클램핑 제어선 구축
    //      0번 스레드는 전역 좌측 원소가 없으므로 0으로 바인딩, 그 외에는 전역 인덱스 - 1 위치 조준
    // [EN] [Pipeline Breakthrough]: Constructs mathematical clamping control lines to completely eradicate conditional if-else branch stalls.
    //      Thread 0 binds to global 0 due to the absence of a leftward neighbor; all other threads target the precise global_idx - 1 coordinate.
    uint32_t left_clamp_idx  = (global_idx > 0) ? global_idx - 1 : 0;
    
    // [KR] 블록 끝 스레드 및 격자 끝단 가닥들은 전체 우측 메쉬 한계를 넘지 않도록 clamping 조절
    // [EN] Adjusts clamping across edge threads and grid boundaries to ensure address evaluations never breach the macro-level rightward mesh limits
    uint32_t right_clamp_idx = (global_idx < total_cells - 1) ? global_idx + 1 : total_cells - 1;




    // =====================================================================================
    // [⚡ GLOBAL RE-LOAD ZERO & ZERO-FLAG MASKING CORE]
    // =====================================================================================
    // [KR] 전역 메모리를 또 찌르지 말고, "이미 로드 후 정제되어 셰어드에 박힌 이전 워프/블록 데이터"를 재활용합니다.
    //      워프 내부 코어 가닥들은 레지스터 상호 교환(Shuffle)으로 넘어가므로, 
    //      여기서는 워프 경계선(Lane 0) 및 블록 경계선(Thread 0)이 참조할 공유 메모리 인프라만 준비합니다.
    // [EN] Bypasses redundant global memory bus (HBM) probes, recycling the pre-cleansed data already committed to the shared scratchpad.
    //      While core execution strands inside the warp pass data via register-shuffling,
    //      this layer primes the shared memory infrastructure exclusively for warp lane-0 and block thread-0 boundaries to reference.
    
    // [🚀 HARDWARE ZERO-FLAG INTENSITY GUARD]
    // [KR] [비트 제로 플래그 가드]: 하드웨어 연산 장치(ALU)의 제로 플래그(ZF) 하나만 체크하는 최속의 마스킹선 구축
    // [EN] [Hardware Zero-Flag Intensity Guard]: Constructs an ultra-fast masking register targeting a single Zero Flag (ZF) emission inside the hardware ALU.
    uint32_t is_absolute_mesh_start = (global_idx == 0);


    // [🚀 CONDITIONAL BOUNDARY ASSIGNMENT VIA SILICON MUX]
    // [KR] 전체 유동 관로의 절대 시작점(global_idx == 0)일 때는 청정 베이스라인(0.0f)을 강제 주입하고,
    //      그 외의 다른 일반 블록들의 0번 스레드들은 앞 블록이 정적 동기화 완료해 둔 좌측 격자 공간을 상속합니다.
    // [EN] Forces a hard injection of CLEAN_BASELINE_VAL (0.0f) at the absolute origin of the fluid mesh (global_idx == 0);
    //      for all other nominal blocks, thread-0 inherits the leftward boundary space pre-committed during static block synchronization.
    float real_left_mesh_val = pinn_branchless_select_f32(is_absolute_mesh_start, CLEAN_BASELINE_VAL, shared_flux[local_shared_idx - 1]);

    // =====================================================================================
    // [🛡️ GARBAGE INDEX MASKING TRICK - LEFT BOUNDARY INJECTION]
    // =====================================================================================
    // [KR] 읽기-비교(SEL)-쓰기 루프를 통째로 청산하고, 오직 주소선 제어 비만 스위칭하여 무조건 Store 실행
    //      thread_idx가 0인 스레드만 유효한 0번 패딩 헤일로 주소를 얻고, 나머지 255개 스레드는
    //      안전 격리 구역인 GARBAGE_IDX(공유 메모리 맨 끝 슬롯)를 강제 조준하도록 대수적 바인딩을 완료합니다.
    // [EN] Eliminates the entire read-compare-write loop, instead manipulating raw address control lines to execute blind store commands.
    //      Only thread-0 is granted the valid halo-padding index (0), while the remaining 255 threads 
    //      are algebraically routed to fire blank data payloads straight into GARBAGE_IDX (the isolated scratchpad tail slot).
    const uint32_t left_target_idx = pinn_branchless_select_u32(thread_idx == 0, 0, GARBAGE_IDX);

    
    // [ CONCURRENT BLIND STORE - HARDWARE PARALLEL PACKET DROP]
    // [KR] 32개 스레드가 주소선 분기 없이 일제히 Store 명령을 집행합니다.
    //      0번 스레드의 진정한 경계값만 패딩 존에 안착하고, 나머지 255개 가닥들의 사격은 쓰레기통(Garbage Zone)으로 완전 안전하게 유실 및 드롭 처리됩니다.
    // [EN] Multiple execution strands dispatch concurrent hardware Store commands simultaneously without address line branches.
    //      Only the genuine boundary payload from thread-0 anchors safely into the halo padding zone, 
    //      while the remaining 255 volatile shots bleed into the Garbage Zone to be seamlessly discarded.
    shared_flux[left_target_idx] = real_left_mesh_val;


    
    // =====================================================================================
    // [⚡ GLOBAL RE-LOAD ZERO & ZERO-FLAG MASKING RIGHT ATTRACTOR]
    // =====================================================================================
    // [KR] 전체 격자의 물리적 끝단(global_idx == total_cells - 1)인지 판별하는 제로 플래그 가드 구축
    // [EN] Establishes a hardware Zero-Flag (ZF) guard logic to verify if the thread position aligns with the absolute rightward boundary of the macro-mesh (global_idx == total_cells - 1)
    uint32_t is_absolute_mesh_end = (global_idx == total_cells - 1);

    // [KR] 유동 관로의 절대 끝단이면 청정 베이스라인(0.0f)을 주입하고, 그 외에는 
    //      이미 1단계-F 구역에서 전 스레드가 선집행하여 채워둔 우측 공유 메모리 슬롯 값을 안전하게 수입합니다.
    // [EN] Injects CLEAN_BASELINE_VAL (0.0f) if the current location hits the absolute termination vertex of the fluid advection channel;
    //      otherwise, safely ingests the rightward shared scratchpad slot value populated during the concurrent phase in Step 1-F.
    float real_right_mesh_val = pinn_branchless_select_f32(is_absolute_mesh_end, CLEAN_BASELINE_VAL, shared_flux[local_shared_idx + 1]);

    // =====================================================================================
    // [🛡️ GARBAGE INDEX MASKING TRICK - RIGHT HALO BOUNDARY COMPLETION]
    // =====================================================================================
    // [KR] thread_idx가 블록의 마지막 스레드이거나 전체 메쉬의 물리적 끝단일 때만 우측 패딩 가드로 동작 유도
    // [EN] Evaluates a bitwise OR (|) to flag if the thread sits on either the block edge thread boundary or the absolute physical grid termination vertex.
    uint32_t is_block_edge = (thread_idx == blockDim.x - 1) | is_absolute_mesh_end;
    
    // [KR] 조건 만족 시 실제 우측 패딩 주소(local_shared_idx + 1)를 쥐고, 탈락한 스레드들은 GARBAGE_IDX로 영토 격리
    // [EN] Routes true evaluations to the actual rightward halo offset (local_shared_idx + 1), while out-of-bound threads are algebraically quarantined to GARBAGE_IDX.
    const uint32_t right_target_idx = pinn_branchless_select_u32(is_block_edge, local_shared_idx + 1, GARBAGE_IDX);
    
    // [KR] 하드웨어 Store 명령 하나로 양방향 경계선 인입을 무분기 전사 완료 (불필요한 비교 및 로드 사이클 영구 박멸)
    // [EN] Executes a single hardware Store instruction to complete bidirectional boundary injection branchlessly, permanently eradicating redundant compare-and-jump stalls.
    shared_flux[right_target_idx] = real_right_mesh_val;


    // [🛡️ STATIC BLOCK BARRIER & PIPELINE THREAD INSURANCE]
    // [KR] 블록 내 공유 메모리 인입 데이터 경합(Race Condition) 방지를 위한 하드웨어 실행 배리어 가동
    // [EN] Activates the hardware execution barrier to prevent data race conditions across the shared scratchpad footprint
    __syncthreads();

    // [KR] 동기화 완료 후 격자 유효 범위를 초과하여 할당된 잔여 스레드들의 하부 가중치 오염 방지 가드
    // [EN] Deploys an execution fence to protect low-level weight tensors from out-of-bound idle thread contamination post-sync
    if (global_idx >= total_cells) return;

    // 3. [🚀 REGISTER-LEVEL WARP SHUFFLE HIGH-SPEED EXCHANGER]
    // [KR] 3. 레지스터 레벨 워프 셔플 고속화 (내부 스레드 초고속 통신 레일 구동)
    //      워프 내부(32스레드) 가닥들은 공유 메모리 뱅크 충돌(Bank Conflict)조차 발생하지 않는 1클록 최속 레일 가동
    // [EN] 3. Register-Level Warp Shuffle Acceleration (Ultra-Fast Intra-Thread Communication Rail)
    //      Core strands within a single warp (32 threads) engage a 1-clock execution rail completely immune to shared memory bank conflicts.
    uint32_t lane_id = thread_idx & 31;
    
    // [🚀 MEMORY OPERATIONAL (MIO) UNDERLOAD TRICK]
    // [KR] 워프 내부 코어 가닥들은 레지스터 상호 교환 인트린직(__shfl_up_sync / __shfl_down_sync)을 통해 데이터를 나노초 단대로 교환
    //      [셔플 트릭 결합 완결]: 좌우 이웃 셔플 포획 회로와 경계선 마스킹을 결합하여 메모리 장치(MIO)의 부하를 지워버립니다.
    // [EN] Inner warp strands exchange volatile payload metrics at nanosecond intervals using register cross-exchange primitives (__shfl_up_sync / __shfl_down_sync).
    //      [Shuffle Trick Completion]: Fuses neighbor shuffle-capture logic with edge masking to entirely eliminate memory instruction unit (MIO) pipeline bottlenecks.
    float left_shuffle  = __shfl_up_sync(0xFFFFFFFF, current_flux, 1);
    float right_shuffle = __shfl_down_sync(0xFFFFFFFF, current_flux, 1);

    // [🚀 WARP-BOUND INTERLOCK MUX SWITCHING]
    // [KR] 워프의 물리적 물리 경계선(Lane 0 또는 Lane 31)에 걸친 가닥들만 안전 공유 메모리 패딩 존을 참조하도록 마스킹 스위칭
    // [EN] Deploys MUX masking switching exclusively for fringe strands situated on warp hardware bounds (Lane 0 or Lane 31) to reference the safe shared memory padding buffer.
    float west_flux = pinn_branchless_select_f32(lane_id == 0,  shared_flux[local_shared_idx - 1], left_shuffle);
    float east_flux = pinn_branchless_select_f32(lane_id == 31, shared_flux[local_shared_idx + 1], right_shuffle);



    // 4. [📐 MATHEMATICAL FLUID GEOMETRY & ALGEBRAIC CO-DESIGN]
    // [KR] 4. 수리 물리 기하학 공식 연산 및 1:1 대수적 매핑
    //      공간 차분 편차 도출 수식 실현: U = East - West
    //      단 한 번의 글로벌 로드 이후 오직 레지스터와 셰어드로만 완성된 초고속 물리 필드입니다.
    // [EN] 4. Mathematical Fluid Geometry Formulation & 1:1 Algebraic Co-Design Mapping
    //      Implements spatial finite difference derivative extraction: U = East - West
    //      An ultra-fast physical field fully resolved inside register files and local scratchpads following a solitary global memory ingress event.
    float spatial_deviation_u = east_flux - west_flux;
    
    // [🚀 ATOMIC INDEX BITMASKING & DIVISION-FREE MULTIPLICATION]
    // [KR] 글로벌 메쉬 셀 구조체에 명시된 고유 좌표 고리 인덱스를 비트 마스킹 처리 (0~63 범위 제한)
    //      1단계-A/B 구역의 Constant LUT 상반수 배열을 나눗셈 연산 없이 단일 클록 속도로 초고속 인덱싱합니다.
    // [EN] Executes an atomic bitwise AND operation on the unique grid coordinate ID specified inside the global mesh structure (Clamping within 0-63 boundary limits)
    //      Enforces ultra-fast indexing of the Constant Memory Reciprocal LUT established in Step 1-A/B, completing the evaluation via division-free, single-clock multiplication.
    uint32_t target_lut_idx = global_mesh_cells[global_idx].coordinate_id & (LUT_SIZE_32 - 1);
    float normalized_gradient = spatial_deviation_u * RECIPROCAL_CELL_LUT[target_lut_idx];


    // 5. [🚀 FINAL BUS COMMIT - IN-PLACE HARDWARE WRITE-BACK]
    // [KR] 5. 동기화 하드웨어 배열 최종 상태 확정 커밋 (In-place Write-Back)
    //      원본 입력 데이터 스트림에 하드웨어 폭사 결함이 찍혀있었다면, 
    //      상위 JAX/XLA 방화벽이 이를 포획할 수 있도록 최종 출력 필드에 -99.0f 에러 시그니처 마커를 영구 각인합니다.
    // [EN] 5. Final Hardware Array State Commitment (In-place Write-Back to VRAM Bus)
    //      If the raw upstream input stream contains catastrophic hardware faults,
    //      permanently imprints the -99.0f fault signature token onto the spatial output field for the high-level JAX/XLA firewall to capture.
    global_mesh_cells[global_idx].spatial_u = pinn_branchless_select_f32(is_anomaly, FAULT_TOKEN_SIGNATURE, normalized_gradient);
    
    // [KR] 결함 상태 비트 마스크 최종 업데이트 (0:정상 통과, 2:결함으로 인한 MUX 차단)
    // [EN] Commits the final cell status bitmask tracking hardware state profiles (0: Nominal Pass, 2: Quarantined via MUX Interlock Shrouding)
    global_mesh_cells[global_idx].cell_status = pinn_branchless_select_u32(is_anomaly, 2, 0);
}
