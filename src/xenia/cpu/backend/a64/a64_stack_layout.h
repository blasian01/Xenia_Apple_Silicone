/**
 * ARM64 Stack Layout for Xenia JIT Backend
 */
#ifndef XENIA_CPU_BACKEND_A64_A64_STACK_LAYOUT_H_
#define XENIA_CPU_BACKEND_A64_A64_STACK_LAYOUT_H_

#include "xenia/base/vec128.h"

namespace xe {
namespace cpu {
namespace backend {
namespace a64 {

class StackLayout {
 public:
  /**
   * Thunk stack (host→guest transition):
   * SP must be 16-byte aligned at all times on ARM64.
   *
   *  +------------------+
   *  | x19 (ctx)        | sp + 0x00
   *  | x20 (membase)    | sp + 0x08
   *  +------------------+
   *  | x21              | sp + 0x10
   *  | x22              | sp + 0x18
   *  +------------------+
   *  | x23              | sp + 0x20
   *  | x24              | sp + 0x28
   *  +------------------+
   *  | x25              | sp + 0x30
   *  | x26              | sp + 0x38
   *  +------------------+
   *  | x27              | sp + 0x40
   *  | x28              | sp + 0x48
   *  +------------------+
   *  | x29 (fp)         | sp + 0x50
   *  | x30 (lr)         | sp + 0x58
   *  +------------------+
   *  | v8               | sp + 0x60
   *  | v9               | sp + 0x70
   *  | v10              | sp + 0x80
   *  | v11              | sp + 0x90
   *  | v12              | sp + 0xA0
   *  | v13              | sp + 0xB0
   *  | v14              | sp + 0xC0
   *  | v15              | sp + 0xD0
   *  +------------------+
   *  Total: 0xE0 = 224 bytes (16-byte aligned)
   */
  static constexpr size_t THUNK_STACK_SIZE = 0xE0;

  /**
   * Guest function stack:
   *  +------------------+
   *  | scratch (48b)    | sp + 0x00  (6 x 8 bytes)
   *  +------------------+
   *  | guest ret addr   | sp + 0x30
   *  | call ret addr    | sp + 0x38
   *  +------------------+
   *  |  ... locals ...  | sp + 0x40+
   *  +------------------+
   *
   *  Total header: 0x40 = 64 bytes (minimum, 16-byte aligned)
   */
  static constexpr size_t GUEST_STACK_SIZE = 0x40;
  static constexpr size_t GUEST_SCRATCH = 0x00;
  static constexpr size_t GUEST_RET_ADDR = 0x30;
  static constexpr size_t GUEST_CALL_RET_ADDR = 0x38;
};

}  // namespace a64
}  // namespace backend
}  // namespace cpu
}  // namespace xe

#endif  // XENIA_CPU_BACKEND_A64_A64_STACK_LAYOUT_H_
