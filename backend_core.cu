#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <stdint.h>
#include <math.h>

// [🚀 HARDWARE CORE CONSTRAINTS - STRUCT & BLOCK GEOMETRY]
#define BLOCK_SIZE_1D 256
#define HALO_SIZE 1 // 1차원 중앙 차분을 위한 좌우 헤일로 패딩 반경
#define SHARED_MEM_SIZE (BLOCK_SIZE_1D + (HALO_SIZE * 2))

#define LUT_SIZE_32 64
#define COMPRESSED_THRESHOLD 1000000.0f
#define FAULT_TOKEN_SIGNATURE -99.0f
#define CLEAN_BASELINE_VAL 0.0f

// 32바이트 정렬된 물리 노드 구조체 (하부 PCIe 버스선 및 L1/L2 캐시라인 인라인 정렬 완료)
struct alignas(32) __align__(32) PinnCell32 {
    float param_w;         // [Offset 0] 물리 가중치 레지스터
    float spatial_u;       // [Offset 4] 동서(East-West) 유동 편차 성분
    float spatial_v;       // [Offset 8] 남북(North-South) 유동 편차 성분
    float adaptive_gain;   // [Offset 12] 자율 튜닝 스케일 가중치 이득
    uint32_t cell_status;  // [Offset 16] HW 오류 MUX 마스킹 상태 비트 (0: 정상, 2: 결함)
    uint32_t coordinate_id;// [Offset 20] 1D/2D 격자선 상의 고유 물리 좌표 고리
    uint64_t padding;      // [Offset 24] 32바이트 대칭형 버스 플러시용 더미 패딩
};

// 나눗셈을 단일 클록 곱셈으로 파쇄하기 위한 상반수 Constant LUT (실제 가동 하드웨어 스케일 전사)
__device__ __constant__ const float RECIPROCAL_CELL_LUT[LUT_SIZE_32] = {
    0.08333333f, 0.06250000f, 0.05000000f, 0.04166667f, // 1/12, 1/16, 1/20, 1/24
    0.03125000f, 0.02500000f, 0.02000000f, 0.01562500f, // 1/32, 1/40, 1/50, 1/64
    0.01250000f, 0.01000000f, 0.00781250f, 0.00625000f, // 1/80, 1/100, 1/128, 1/160
    0.00500000f, 0.00250000f, 0.00125000f, 0.00062500f  // 나머지 해상도 역수 확장 슬롯...
};

// =================================================================
// [⚡ PURE BRANCHLESS INTRINSICS - MULTI-CHIP MUX SELECTORS]
// =================================================================

// 1. uint32_t 전용 무분기 선택자 (PTX 'SEL' 명령어로 1대1 매핑 유도)
__device__ __forceinline__ uint32_t pinn_branchless_select_u32(uint32_t cond, uint32_t t, uint32_t f) {
    return (cond) ? t : f;
}

// 2. float 전용 무분기 선택자 (조건 분기 점프문을 소멸시키고 레지스터 레벨에서 비트 스위칭)
__device__ __forceinline__ float pinn_branchless_select_f32(uint32_t cond, float t, float f) {
    return (cond) ? t : f;
}

// 3. 하드웨어 가속 결함 판별기 (isfinite() 내장 함수를 활용해 NaN 및 INF를 단일 연산으로 동시 조준)
__device__ __forceinline__ uint32_t pinn_check_hardware_anomaly(float flux) {
    uint32_t is_bad_num = !isfinite(flux);
    uint32_t is_overflow = (fabsf(flux) > COMPRESSED_THRESHOLD);
    return (is_bad_num | is_overflow);
}


// [CORE INGRESS KERNEL] 하드웨어 절연형 무분기 공간 구배 솔버
__global__ void forward_only_pure_algebraic_kernel(
    PinnCell32* __restrict__ global_mesh_cells,
    const float* __restrict__ raw_input_flux,
    const uint32_t total_cells
) {
    // 1. 스레드 토폴로지 매핑 및 공유 메모리 할당
    const uint32_t thread_idx = threadIdx.x;
    const uint32_t global_idx = blockIdx.x * blockDim.x + thread_idx;
    
    // 블록 경계선 평탄화를 위한 좌우 헤일로 패딩 존 포함 공유 버퍼 할당
    __shared__ float shared_flux[SHARED_MEM_SIZE];
    
    // 레지스터 단에 매핑할 유효 데이터 작업 영역 초기화
    float current_flux = CLEAN_BASELINE_VAL;
    uint32_t is_anomaly = 0;
    
    // 전체 격자 크기 경계 안쪽일 때만 글로벌 메모리 인입 전개
    if (global_idx < total_cells) {
        current_flux = __ldg(&raw_input_flux[global_idx]);
        is_anomaly = pinn_check_hardware_anomaly(current_flux);
        
        // 결함 노이즈 포획 즉시 레지스터 레벨에서 청정 베이스라인(0.0f)으로 하드 플러시 실행
        current_flux = pinn_branchless_select_f32(is_anomaly, CLEAN_BASELINE_VAL, current_flux);
    }

    // 2. 100% 무분기(Branchless) 인덱스 매스킹 기반 공유 메모리 헤일로 로드 구역
    // 자신의 스레드 위치에 매칭되는 내부 공유 메모리 오프셋 산정
    const uint32_t local_shared_idx = thread_idx + HALO_SIZE;
    shared_flux[local_shared_idx] = current_flux;

    // [병목 파쇄 핵심 혁신]: if-else 조건문 분기를 소멸시키는 수학적 클램핑 제어선 구축
    // 0번 스레드는 좌측 원소가 없으므로 0으로 수렴 제안, 그 외에는 global_idx - 1 위치 조준
    uint32_t left_clamp_idx  = (global_idx > 0) ? global_idx - 1 : 0;
    // 블록 끝 스레드는 우측 격자 한계를 넘지 않도록 clamping 조절
    uint32_t right_clamp_idx = (global_idx < total_cells - 1) ? global_idx + 1 : total_cells - 1;

    // 32개 스레드가 일제히 이웃 전역 메모리를 지터 없이 병렬 선독점 로드 (Read-Only 캐시 활용)
    float left_flux_raw  = __ldg(&raw_input_flux[left_clamp_idx]);
    float right_flux_raw = __ldg(&raw_input_flux[right_clamp_idx]);

    // 인입된 이웃 격자 원소에 대해서도 무분기 하드웨어 MUX 수치 정화 집행
    uint32_t left_anomaly  = pinn_check_hardware_anomaly(left_flux_raw);
    uint32_t right_anomaly = pinn_check_hardware_anomaly(right_flux_raw);
    
    float left_flux_clean  = pinn_branchless_select_f32(left_anomaly,  CLEAN_BASELINE_VAL, left_flux_raw);
    float right_flux_clean = pinn_branchless_select_f32(right_anomaly, CLEAN_BASELINE_VAL, right_flux_raw);

    // 하드웨어 비트 선택 명령어를 이용해 블록 경계선 스레드의 헤일로 패딩 슬롯에만 동시 저격 사격
    // thread_idx가 0일 때만 shared_flux[0]에 좌측 원소가 박히고, 나머지 자리는 비파괴적 자동 방어 유지
    shared_flux[0] = pinn_branchless_select_f32(thread_idx == 0, left_flux_clean, shared_flux[0]);
    
    // thread_idx가 블록의 마지막 스레드이거나 전체 메쉬의 끝단일 때만 우측 패딩 영역(local_shared_idx + 1) 전사 완료
    uint32_t is_block_edge = (thread_idx == blockDim.x - 1) | (global_idx == total_cells - 1);
    shared_flux[local_shared_idx + 1] = pinn_branchless_select_f32(is_block_edge, right_flux_clean, shared_flux[local_shared_idx + 1]);

    // 블록 내 공유 메모리 인입 데이터 경합(Race Condition) 방지를 위한 비트 동기화 장치 가동
    __syncthreads();

    // 동기화 완료 후 격자 유효 범위 이탈 스레드들의 가중치 오염 방지 가드
    if (global_idx >= total_cells) return;

    // 3. 레지스터 레벨 워프 셔플 고속화 (내부 스레드 초고속 통신 레일 구동)
    // 워프 내부(32스레드) 가닥들은 뱅크 충돌이 아예 일어날 수 없는 내장 셔플을 통해 옆자리 데이터 교환
    uint32_t lane_id = thread_idx & 31;
    
    // 워프 경계선(Lane 0 또는 Lane 31)에 물리적으로 걸쳐 있는 가닥들만 공유 메모리 패딩 존을 참조하도록 마스킹 스위칭
    float west_flux = pinn_branchless_select_f32(lane_id == 0,  shared_flux[local_shared_idx - 1], __shfl_up_sync(0xFFFFFFFF, current_flux, 1));
    float east_flux = pinn_branchless_select_f32(lane_id == 31, shared_flux[local_shared_idx + 1], __shfl_down_sync(0xFFFFFFFF, current_flux, 1));

    // 4. 수리 물리 기하학 공식 연산 및 1:1 대수적 매핑
    // 공간 차분 편차 도출 수식 실현: U = East - West 
    float spatial_deviation_u = east_flux - west_flux;
    
    // 고장난 격자 좌표 고유 번호를 활용해 나눗셈 없이 단일 클록 속도로 격자 스페이싱 상반수 맵핑 인입
    uint32_t target_lut_idx = global_mesh_cells[global_idx].coordinate_id & (LUT_SIZE_32 - 1);
    float normalized_gradient = spatial_deviation_u * RECIPROCAL_CELL_LUT[target_lut_idx];

    // 5. 동기화 하드웨어 배열 최종 상태 확정 커밋 (In-place Write-Back)
    // 원본 입력 데이터에 결함이 찍혀있었다면 최종 출력 필드에 -99.0f 에러 시그니처 마커 각인
    global_mesh_cells[global_idx].spatial_u = pinn_branchless_select_f32(is_anomaly, FAULT_TOKEN_SIGNATURE, normalized_gradient);
    global_mesh_cells[global_idx].cell_status = pinn_branchless_select_u32(is_anomaly, 2, 0);
}
