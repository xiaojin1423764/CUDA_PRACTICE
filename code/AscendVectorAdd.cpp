#include "kernel_operator.h"

// 最小 AscendC 向量加法 kernel。
//
// 计算：z[i] = x[i] + y[i]，输入和输出均为 HBM（Global Memory）中的 FP32 数组。
// 本示例固定处理 8192 个元素；宿主侧必须用 8 个 AI Core 启动 vector_add，
// 且 x、y、z 都至少分配 8192 * sizeof(float) 字节的 device memory。
//
// 编译需要已安装并初始化 CANN/AscendC 工具链。不同 CANN 版本和芯片型号的
// 编译、链接、kernel launch 接口不同，因此本文件只提供 device kernel；可将其
// 集成到 CANN custom-op 工程，或按目标平台的 AscendC 样例编译并由 ACL 宿主程序调用。

using namespace AscendC;

namespace {

constexpr uint32_t kTotalLength = 8192;
constexpr uint32_t kBlockCount = 8;
constexpr uint32_t kBlockLength = kTotalLength / kBlockCount;
constexpr uint32_t kTileLength = 256;
constexpr uint32_t kTilesPerBlock = kBlockLength / kTileLength;
constexpr uint32_t kBufferCount = 2;

static_assert(kTotalLength % kBlockCount == 0,
              "total length must divide evenly across AI Cores");
static_assert(kBlockLength % kTileLength == 0,
              "each AI Core workload must divide evenly into tiles");

class VectorAddKernel {
 public:
  __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, GM_ADDR z) {
    // GetBlockIdx() 是当前 AI Core 的逻辑编号。每个 core 只处理连续的 1024 个
    // 元素，因此对 HBM 的输入/输出访问连续，便于高效搬运。
    const uint32_t offset = GetBlockIdx() * kBlockLength;
    x_gm_.SetGlobalBuffer((__gm__ float *)x + offset, kBlockLength);
    y_gm_.SetGlobalBuffer((__gm__ float *)y + offset, kBlockLength);
    z_gm_.SetGlobalBuffer((__gm__ float *)z + offset, kBlockLength);

    // TPipe 管理片上 Unified Buffer（UB）中的临时空间。每条输入队列和输出队列
    // 使用两个 buffer：当前 tile 在 Vector 单元计算时，下一 tile 可以进入另一块
    // buffer，形成数据搬运和计算之间的双缓冲流水。
    pipe_.InitBuffer(x_queue_, kBufferCount, kTileLength * sizeof(float));
    pipe_.InitBuffer(y_queue_, kBufferCount, kTileLength * sizeof(float));
    pipe_.InitBuffer(z_queue_, kBufferCount, kTileLength * sizeof(float));
  }

  __aicore__ inline void Process() {
    // 每个 core 负责 1024 个元素，即依次处理 4 个长度为 256 的 tile。
    for (uint32_t tile = 0; tile < kTilesPerBlock; ++tile) {
      CopyIn(tile);
      Compute();
      CopyOut(tile);
    }
  }

 private:
  __aicore__ inline void CopyIn(uint32_t tile) {
    // 从 HBM 取出一个 tile 到 UB；x/y 使用各自的队列，避免覆盖。
    LocalTensor<float> x_local = x_queue_.AllocTensor<float>();
    LocalTensor<float> y_local = y_queue_.AllocTensor<float>();
    const uint32_t offset = tile * kTileLength;
    DataCopy(x_local, x_gm_[offset], kTileLength);
    DataCopy(y_local, y_gm_[offset], kTileLength);
    x_queue_.EnQue(x_local);
    y_queue_.EnQue(y_local);
  }

  __aicore__ inline void Compute() {
    // Vector 单元在 UB 上执行逐元素 Add。计算结果先留在 UB，避免中间结果回写 HBM。
    LocalTensor<float> x_local = x_queue_.DeQue<float>();
    LocalTensor<float> y_local = y_queue_.DeQue<float>();
    LocalTensor<float> z_local = z_queue_.AllocTensor<float>();
    Add(z_local, x_local, y_local, kTileLength);
    x_queue_.FreeTensor(x_local);
    y_queue_.FreeTensor(y_local);
    z_queue_.EnQue(z_local);
  }

  __aicore__ inline void CopyOut(uint32_t tile) {
    // 将当前 tile 的结果连续写回 HBM，并归还 UB buffer 供后续 tile 复用。
    LocalTensor<float> z_local = z_queue_.DeQue<float>();
    DataCopy(z_gm_[tile * kTileLength], z_local, kTileLength);
    z_queue_.FreeTensor(z_local);
  }

  TPipe pipe_;
  GlobalTensor<float> x_gm_;
  GlobalTensor<float> y_gm_;
  GlobalTensor<float> z_gm_;
  TQue<QuePosition::VECIN, kBufferCount> x_queue_;
  TQue<QuePosition::VECIN, kBufferCount> y_queue_;
  TQue<QuePosition::VECOUT, kBufferCount> z_queue_;
};

}  // namespace

extern "C" __global__ __aicore__ void vector_add(GM_ADDR x, GM_ADDR y,
                                                   GM_ADDR z) {
  // 启动配置必须指定 blockDim = kBlockCount。每个 AI Core 运行一个实例，
  // 合计覆盖 [0, kTotalLength) 的全部元素。
  VectorAddKernel kernel;
  kernel.Init(x, y, z);
  kernel.Process();
}
