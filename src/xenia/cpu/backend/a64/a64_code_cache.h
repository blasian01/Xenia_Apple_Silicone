/**
 * ARM64 Code Cache for Xenia JIT
 */
#ifndef XENIA_CPU_BACKEND_A64_A64_CODE_CACHE_H_
#define XENIA_CPU_BACKEND_A64_A64_CODE_CACHE_H_

#include <memory>
#include "xenia/cpu/backend/code_cache_base.h"

namespace xe {
namespace cpu {
namespace backend {
namespace a64 {

class A64CodeCache : public CodeCacheBase<A64CodeCache> {
 public:
  ~A64CodeCache() override = default;

  static std::unique_ptr<A64CodeCache> Create();
  virtual bool Initialize();

  void* LookupUnwindInfo(uint64_t host_pc) override { return nullptr; }

  // CRTP hooks.
  uint32_t IndirectionSlotValue(void* code_execute_address) {
    // Slots hold 32-bit offsets into the JIT region (0 = unresolved) — the
    // region sits above 4GB on macOS so truncated addresses won't round-trip.
    return uint32_t(reinterpret_cast<uint8_t*>(code_execute_address) -
                    generated_code_execute_base_);
  }
  void FillCode(void* write_address, size_t size);
  void FlushCodeRange(void* address, size_t size);
  void OnCodePlaced(uint32_t guest_address, GuestFunction* function_info,
                    void* code_execute_address, size_t code_size);

  virtual UnwindReservation RequestUnwindReservation(uint8_t* entry_address) {
    return UnwindReservation();
  }
  virtual void PlaceCode(uint32_t guest_address, void* machine_code,
                         const EmitFunctionInfo& func_info,
                         void* code_execute_address,
                         UnwindReservation unwind_reservation) {}

 protected:
  A64CodeCache() = default;
};

}  // namespace a64
}  // namespace backend
}  // namespace cpu
}  // namespace xe

#endif  // XENIA_CPU_BACKEND_A64_A64_CODE_CACHE_H_
