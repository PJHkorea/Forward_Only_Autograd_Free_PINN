#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <stdint.h>
#include <math.h>

// [🚀 HARDWARE CORE CONSTRAINTS - STRUCT & BLOCK GEOMETRY]
#define BLOCK_SIZE_1D 256
#define HALO_SIZE 1 // 1차원 중앙 차분을 위한 좌우 헤일로 패딩 반경
#define SHARED_MEM_SIZE (BLOCK_SIZE_1D + (HALO_SIZE * 2))

// [🛡️ GARBAGE INDEX MASKING SAFE ATTRACTOR]
// 무분기 주소선 제어 시 쓰레기 데이터를 받아낼 안전 격리 슬롯 지정 (+1 여유 자산 확보)
#define GARBAGE_IDX SHARED_MEM_SIZE 
#define ALLOCATED_SHARED_MEM_SIZE (SHARED_MEM_SIZE + 1)

#define LUT_SIZE_32 64
#define COMPRESSED_THRESHOLD 1000000.0f
#define FAULT_TOKEN_SIGNATURE -99.0f
#define CLEAN_BASELINE_VAL 0.0f


// 32바이트 정렬된 물리 노드 구조체 (하부 PCIe 버스선 및 L1/L2 캐시라인 인라인 정렬 완료)
struct alignas(32) __align__(32) PinnCell32 {
    float param_w;         // [Offset 0] 물리 가중치 레지스터 진입점
    float spatial_u;       // [Offset 4] 동서(East-West) 유동 편차 대수 필드
    float spatial_v;       // [Offset 8] 남북(North-South) 유동 편차 대수 필드
    float adaptive_gain;   // [Offset 12] 자율 튜닝 스케일 가중치 이득 변수
    uint32_t cell_status;  // [Offset 16] 무분기 하드웨어 MUX 쉴드 상태 비트 (0:정상, 2:결함)
    uint32_t coordinate_id;// [Offset 20] 1D/2D 물리 격자선 상의 고유 바인딩 인덱스
    uint64_t padding;      // [Offset 24] L1/L2 캐시라인 파편화 방지용 버스 대칭 패딩
};

// 나눗셈을 단일 클록 곱셈으로 파쇄하기 위한 상반수 Constant LUT (실제 가동 하드웨어 스케일 전사)
__device__ __constant__ const float RECIPROCAL_CELL_LUT[LUT_SIZE_32] = {
    0.08333333f, 0.06250000f, 0.05000000f, 0.04166667f, // 1/12, 1/16, 1/20, 1/24 격자 스페이싱 역수
    0.03125000f, 0.02500000f, 0.02000000f, 0.01562500f, // 1/32, 1/40, 1/50, 1/64 격자 스페이싱 역수
    0.01250000f, 0.01000000f, 0.00781250f, 0.00625000f, // 1/80, 1/100, 1/128, 1/160 격자 스페이싱 역수
    0.00500000f, 0.00250000f, 0.00125000f, 0.00062500f, // 1/200, 1/400, 1/800, 1/1600 격자 스페이싱 역수
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

// 1. uint32_t 전용 무분기 선택자 (PTX 'SEL' 기계어 명령어로 1대1 매핑 강제 유도)
__device__ __forceinline__ uint32_t pinn_branchless_select_u32(uint32_t cond, uint32_t t, uint32_t f) {
    // 참과 거짓 경로 모두에 주소 점프(JMP)가 없는 결정론적 연산 보장
    return (cond) ? t : f;
}

// 2. float 전용 무분기 선택자 (조건 분기 점프문을 파쇄하고 레지스터 레벨에서 비트 마스킹 스위칭)
__device__ __forceinline__ float pinn_branchless_select_f32(uint32_t cond, float t, float f) {
    // 하부 실리콘 커널단에서 단 1클록 단위의 실행 지연 시간으로 즉시 치환 실행
    return (cond) ? t : f;
}

// 3. 하드웨어 가속 결함 판별기 (isfinite 내장 함수를 가동해 NaN 및 INF를 단일 비트 연산으로 동시 조준)
__device__ __forceinline__ uint32_t pinn_check_hardware_anomaly(float flux) {
    // 수치 폭발(NaN/INF)이 발생하는 순간 1차 방화벽 트리거를 위한 비트 마스킹 결합
    uint32_t is_bad_num = !isfinite(flux);
    uint32_t is_overflow = (fabsf(flux) > COMPRESSED_THRESHOLD);
    
    // 논리합 비트 연산(|)을 통해 단 하나의 분기문도 생성하지 않고 수치 위험성 리턴
    return (is_bad_num | is_overflow);
}


// [CORE INGRESS KERNEL] 하드웨어 절연형 무분기 공간 구배 솔버
__global__ void forward_only_pure_algebraic_kernel(
    PinnCell32* __restrict__ global_mesh_cells,
    const float* __restrict__ raw_input_flux,
    const uint32_t total_cells
) {
    // 1. 스레드 토폴로지 매핑 및 가비지 마스킹용 정적 공유 메모리 할당
    const uint32_t thread_idx = threadIdx.x;
    const uint32_t global_idx = blockIdx.x * blockDim.x + thread_idx;
    
    // [🛡️ GARBAGE MASKING CORE INTEGRATION]
    // 1단계-A에서 선언한 ALLOCATED_SHARED_MEM_SIZE(+1 여유 자산) 규격으로 강제 컴파일 락킹
    // 유효 패딩 영역 바깥의 마지막 슬롯(GARBAGE_IDX)이 유휴 스레드들의 쓰레기 데이터 사격 장소로 동결됩니다.
    __shared__ float shared_flux[ALLOCATED_SHARED_MEM_SIZE];

    
       // 레지스터 단에 매핑할 유효 데이터 작업 영역 초기화
    float current_flux = CLEAN_BASELINE_VAL;
    uint32_t is_anomaly = 0;
    
    // 전체 격자 크기 경계 안쪽일 때만 글로벌 메모리 버스 인입 전개
    if (global_idx < total_cells) {
        // __ldg() 고속 인트린직 명령어로 L1/L2 읽기 전용 캐시 효율 극대화 및 하드웨어 버스 점유 최소화
        current_flux = __ldg(&raw_input_flux[global_idx]);
        
        // 1단계-C에서 빌드한 분기 없는 비트 연산 기반 결함 감지 필터 작동
        is_anomaly = pinn_check_hardware_anomaly(current_flux);
        
        // 결함 노이즈(NaN, INF, Threshold 초과) 포획 즉시 레지스터 레벨에서 청정 베이스라인(0.0f)으로 하드 플러시 실행
        current_flux = pinn_branchless_select_f32(is_anomaly, CLEAN_BASELINE_VAL, current_flux);
    }


       // 2. 100% 무분기(Branchless) 인덱스 매스킹 기반 공유 메모리 헤일로 로드 구역
    // 자신의 스레드 위치에 매칭되는 정적 공유 메모리 오프셋 산정 (0번 슬롯 비워두고 1번부터 인입)
    const uint32_t local_shared_idx = thread_idx + HALO_SIZE;
    shared_flux[local_shared_idx] = current_flux;

    // [병목 파쇄 핵심 혁신]: if-else 조건문 분기를 소멸시키는 수학적 클램핑 제어선 구축
    // 0번 스레드는 전역 좌측 원소가 없으므로 0으로 바인딩, 그 외에는 전역 인덱스 - 1 위치 조준
    uint32_t left_clamp_idx  = (global_idx > 0) ? global_idx - 1 : 0;
    
    // 블록 끝 스레드 및 격자 끝단 가닥들은 전체 우측 메쉬 한계를 넘지 않도록 clamping 조절
    uint32_t right_clamp_idx = (global_idx < total_cells - 1) ? global_idx + 1 : total_cells - 1;


       // 32개 스레드가 일제히 이웃 전역 메모리를 지터 없이 병렬 선독점 로드 (Read-Only 캐시 활용)
    float left_flux_raw  = __ldg(&raw_input_flux[left_clamp_idx]);
    float right_flux_raw = __ldg(&raw_input_flux[right_clamp_idx]);

    // 인입된 이웃 격자 원소에 대해서도 무분기 하드웨어 MUX 수치 정화 집행
    uint32_t left_anomaly  = pinn_check_hardware_anomaly(left_flux_raw);
    uint32_t right_anomaly = pinn_check_hardware_anomaly(right_flux_raw);
    
    float left_flux_clean  = pinn_branchless_select_f32(left_anomaly,  CLEAN_BASELINE_VAL, left_flux_raw);
    float right_flux_clean = pinn_branchless_select_f32(right_anomaly, CLEAN_BASELINE_VAL, right_flux_raw);

    // =====================================================================================
    // [🛡️ GARBAGE INDEX MASKING TRICK - THE CRITICAL ATTRACTOR INJECTION]
    // =====================================================================================
    // 읽기-조건부 비교(SEL)-쓰기 회로를 파괴하고, 쓰기 전용(Write-only) 주소 마스킹으로 고도화
    // thread_idx가 0인 스레드만 실제 0번 패딩 헤일로 주소를 얻고, 나머지 255개 스레드는 
    // 정적 안전 자산 구역인 GARBAGE_IDX(공유 메모리 맨 끝 슬롯)를 강제 조준하도록 비트 변환합니다.
    const uint32_t left_target_idx = pinn_branchless_select_u32(thread_idx == 0, 0, GARBAGE_IDX);
    
    // 32개 스레드가 일제히 주소선 분기 없이 하드웨어 Store 명령 실행 (나머지 스레드 값은 가비지 존에 무해하게 오버랩)
    shared_flux[left_target_idx] = left_flux_clean;

    
       // [🛡️ GARBAGE INDEX MASKING TRICK - RIGHT HALO BOUNDARY COMPLETION]
    // thread_idx가 블록의 마지막 스레드이거나 전체 메쉬의 물리적 끝단일 때만 우측 패딩 가드로 동작 유도
    uint32_t is_block_edge = (thread_idx == blockDim.x - 1) | (global_idx == total_cells - 1);
    
    // 조건 만족 시 실제 우측 패딩 주소(local_shared_idx + 1)를 쥐고, 탈락한 스레드들은 GARBAGE_IDX로 영토 격리
    const uint32_t right_target_idx = pinn_branchless_select_u32(is_block_edge, local_shared_idx + 1, GARBAGE_IDX);
    
    // 하드웨어 Store 명령 하나로 양방향 경계선 인입을 무분기 전사 완료 (불필요한 비교 및 로드 사이클 영구 박멸)
    shared_flux[right_target_idx] = right_flux_clean;

    // 블록 내 공유 메모리 인입 데이터 경합(Race Condition) 방지를 위한 하드웨어 실행 배리어 가동
    __syncthreads();

    // 동기화 완료 후 격자 유효 범위를 초과하여 할당된 잔여 스레드들의 하부 가중치 오염 방지 가드
    if (global_idx >= total_cells) return;

    // 3. 레지스터 레벨 워프 셔플 고속화 (내부 스레드 초고속 통신 레일 구동)
    // 워프 내부(32스레드) 가닥들은 공유 메모리 뱅크 충돌(Bank Conflict)조차 발생하지 않는 1클록 최속 레일 가동
    uint32_t lane_id = thread_idx & 31;
    
    // 워프의 물리적 물리 경계선(Lane 0 또는 Lane 31)에 걸친 가닥들만 안전 공유 메모리 패딩 존을 참조하도록 마스킹 스위칭
    // 워프 내부 코어 가닥들은 레지스터 상호 교환 인트린직(__shfl_up_sync / __shfl_down_sync)을 통해 데이터를 나노초 단대로 교환
    float west_flux = pinn_branchless_select_f32(lane_id == 0,  shared_flux[local_shared_idx - 1], __shfl_up_sync(0xFFFFFFFF, current_flux, 1));
    float east_flux = pinn_branchless_select_f32(lane_id == 31, shared_flux[local_shared_idx + 1], __shfl_down_sync(0xFFFFFFFF, current_flux, 1));


      // 4. 수리 물리 기하학 공식 연산 및 1:1 대수적 매핑
    // 공간 차분 편차 도출 수식 실현: U = East - West
    float spatial_deviation_u = east_flux - west_flux;
    
    // 글로벌 메쉬 셀 구조체에 명시된 고유 좌표 고리 인덱스를 비트 마스킹 처리 (0~63 범위 제한)
    // 1단계-A/B 구역의 Constant LUT 상반수 배열을 나눗셈 연산 없이 단일 클록 속도로 초고속 인덱싱합니다.
    uint32_t target_lut_idx = global_mesh_cells[global_idx].coordinate_id & (LUT_SIZE_32 - 1);
    float normalized_gradient = spatial_deviation_u * RECIPROCAL_CELL_LUT[target_lut_idx];

    // 5. 동기화 하드웨어 배열 최종 상태 확정 커밋 (In-place Write-Back)
    // 원본 입력 데이터 스트림에 하드웨어 폭사 결함이 찍혀있었다면, 
    // 상위 JAX/XLA 방화벽이 이를 포획할 수 있도록 최종 출력 필드에 -99.0f 에러 시그니처 마커를 영구 각인합니다.
    global_mesh_cells[global_idx].spatial_u = pinn_branchless_select_f32(is_anomaly, FAULT_TOKEN_SIGNATURE, normalized_gradient);
    
    // 결함 상태 비트 마스크 최종 업데이트 (0:정상 통과, 2:결함으로 인한 MUX 차단)
    global_mesh_cells[global_idx].cell_status = pinn_branchless_select_u32(is_anomaly, 2, 0);
}
