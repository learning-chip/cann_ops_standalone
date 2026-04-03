#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

// Standalone CPU tiling generator for `chunk_gated_delta_rule`.
// It mirrors the relevant parts of `op_host/chunk_gated_delta_rule_tiling.cpp`,
// especially `DoMatmulTiling()` which fills `matmulTilingFp32` using CANN.
//
// Build: add `-I<path/to/chunk_gdn>/op_kernel` so `chunk_gated_delta_rule_tiling_data.h` resolves.

#include "adv_api/matmul/bmm_tiling.h"
#include "chunk_gated_delta_rule_tiling_data.h"

#include "tiling/platform/platform_ascendc.h"

using namespace std;

namespace {
constexpr int64_t SYS_WORKSPACE_SIZE = 16777216;  // 16MB

constexpr uint32_t MATMUL_BASE_M = 128;
constexpr uint32_t MATMUL_BASE_K = 128;
constexpr uint32_t MATMUL_BASE_N = 128;

constexpr uint32_t STAGE_ONE_TWO = 2;
constexpr uint32_t STAGE_ONE_THREE = 3;
constexpr uint32_t MASK_NUM = 4;
constexpr int64_t P_NUM = 2;

static void append_hex(const uint8_t* data, size_t n, ostream& os) {
  static const char* kHex = "0123456789abcdef";
  for (size_t i = 0; i < n; ++i) {
    uint8_t b = data[i];
    os << kHex[b >> 4] << kHex[b & 0x0F];
  }
}
}  // namespace

int main(int argc, char** argv) {
  // Args:
  // ai_core_num B T nk nv dk dv hasGamma chunkSize scale
  if (argc != 11) {
    cerr << "Usage: " << argv[0]
         << " ai_core_num B T nk nv dk dv hasGamma chunkSize scale\n";
    return 2;
  }

  int64_t ai_core_num = atoll(argv[1]);
  int64_t B = atoll(argv[2]);
  int64_t T = atoll(argv[3]);
  int64_t nk = atoll(argv[4]);
  int64_t nv = atoll(argv[5]);
  int64_t dk = atoll(argv[6]);
  int64_t dv = atoll(argv[7]);
  int64_t hasGamma = atoll(argv[8]);
  int64_t chunkSize = atoll(argv[9]);
  float scale = static_cast<float>(atof(argv[10]));

  if (ai_core_num <= 0 || B <= 0 || T <= 0 || nk <= 0 || nv <= 0 || dk <= 0 || dv <= 0 || chunkSize <= 0) {
    cerr << "Invalid args.\n";
    return 2;
  }

  // --- Build tiling struct (matches op_host DoOpTiling layout) ---
  ChunkGatedDeltaRule::ChunkGatedDeltaRuleTilingData tiling{};
  tiling.aiCoreNum = ai_core_num;
  tiling.t = T;
  tiling.nk = nk;
  tiling.dk = dk;
  tiling.nv = nv;
  tiling.dv = dv;
  tiling.b = B;
  tiling.hasGamma = hasGamma;
  tiling.chunkSize = chunkSize;

  // Matches op_host:
  //   maxGroupLength = p * aiCoreNum * c, where p=P_NUM (2) and c=chunkSize (fixed 64 in op_host).
  const int64_t c = chunkSize;
  const int64_t p = P_NUM;
  tiling.maxGroupLength = p * tiling.aiCoreNum * tiling.chunkSize;

  // stageOneParaNum = STAGE_ONE_TWO (2)
  tiling.stageOneParaNum = STAGE_ONE_TWO;
  tiling.scale = scale;

  // sizeHigh = sizeof(float)=4
  const int64_t sizeHigh = sizeof(float);
  const int64_t s = tiling.maxGroupLength;

  // interWorkspaceSz (exactly mirrors op_host DoOpTiling)
  tiling.interWorkspaceSz = 0;
  tiling.interWorkspaceSz += sizeHigh * tiling.nv * s;                              // gCumExp
  tiling.interWorkspaceSz += sizeHigh * tiling.nv * s * tiling.dk;                 // kCumDecay
  tiling.interWorkspaceSz += sizeHigh * tiling.nv * s * tiling.dv;                 // vInner
  tiling.interWorkspaceSz += sizeHigh * tiling.nv * s * tiling.dk;                 // qPrime
  tiling.interWorkspaceSz += sizeHigh * tiling.nv * s * tiling.dv;                 // attnInter
  tiling.interWorkspaceSz += sizeHigh * tiling.nv * s * tiling.dk;                 // kg
  tiling.interWorkspaceSz += sizeHigh * tiling.nv * s * c;                          // qkt
  tiling.interWorkspaceSz += sizeHigh * tiling.b * tiling.nv * tiling.dv * tiling.dk;  // highState
  tiling.interWorkspaceSz += sizeHigh * c * c * tiling.aiCoreNum * MASK_NUM;        // mask

  // stageWorkspaceSz (exactly mirrors op_host DoOpTiling)
  tiling.stageWorkspaceSz = sizeHigh * c *
                             (static_cast<int64_t>(STAGE_ONE_TWO) * c +
                              static_cast<int64_t>(STAGE_ONE_THREE) * tiling.dk + tiling.dv) *
                             static_cast<int64_t>(tiling.stageOneParaNum) * tiling.aiCoreNum;

  // --- Fill matmul tiling using MultiCoreMatmulTiling::GetTiling() ---
  // Matches op_host DoMatmulTiling().
  matmul_tiling::MultiCoreMatmulTiling mm_;
  uint64_t ubSize = 0, l1Size = 0, l0CSize = 0;

  auto* ascend = platform_ascendc::PlatformAscendCManager::GetInstance();
  if (ascend == nullptr) {
    cerr << "PlatformAscendCManager::GetInstance() returned null.\n";
    return 1;
  }

  ascend->GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSize);
  ascend->GetCoreMemSize(platform_ascendc::CoreMemType::L1, l1Size);
  ascend->GetCoreMemSize(platform_ascendc::CoreMemType::L0_C, l0CSize);

  mm_.SetBufferSpace(l1Size, l0CSize, ubSize);
  mm_.SetAType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, matmul_tiling::DataType::DT_FLOAT, true);
  mm_.SetBType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, matmul_tiling::DataType::DT_FLOAT, true);
  mm_.SetCType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, matmul_tiling::DataType::DT_FLOAT);
  mm_.SetBias(false);
  mm_.SetDim(1);
  mm_.SetShape(MATMUL_BASE_M, MATMUL_BASE_N, MATMUL_BASE_K);
  mm_.SetOrgShape(MATMUL_BASE_M, MATMUL_BASE_N, MATMUL_BASE_K);
  mm_.SetFixSplit(MATMUL_BASE_M, MATMUL_BASE_N, MATMUL_BASE_K);

  if (mm_.GetTiling(tiling.matmulTilingFp32) == -1) {
    cerr << "GetTiling failed.\n";
    return 1;
  }

  // op_host sets these after GetTiling()
  tiling.matmulTilingFp32.dbL0C = 1;
  tiling.matmulTilingFp32.stepKa = 1;
  tiling.matmulTilingFp32.stepKb = 1;
  tiling.matmulTilingFp32.depthA1 = 1;
  tiling.matmulTilingFp32.depthB1 = 1;
  tiling.matmulTilingFp32.stepM = 1;
  tiling.matmulTilingFp32.stepN = 1;

  // --- Output ---
  const uint64_t workspace_size =
      static_cast<uint64_t>(SYS_WORKSPACE_SIZE + tiling.interWorkspaceSz + tiling.stageWorkspaceSz);

  cout << workspace_size << "\n";

  const size_t tiling_size = sizeof(tiling);
  const uint8_t* raw = reinterpret_cast<const uint8_t*>(&tiling);
  append_hex(raw, tiling_size, cout);
  cout << "\n";
  return 0;
}

