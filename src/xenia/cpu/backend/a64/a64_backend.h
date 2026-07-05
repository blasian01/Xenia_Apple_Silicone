/**
 * ARM64 Backend for Xenia JIT
 */
#ifndef XENIA_CPU_BACKEND_A64_A64_BACKEND_H_
#define XENIA_CPU_BACKEND_A64_A64_BACKEND_H_

#include <memory>
#include <mutex>
#include <utility>
#include <vector>

#include "xenia/base/bit_map.h"
#include "xenia/base/cvar.h"
#include "xenia/cpu/backend/backend.h"

namespace xe {
class Exception;
}

namespace xe {
namespace cpu {
namespace backend {
namespace a64 {

class A64CodeCache;

typedef void* (*HostToGuestThunk)(void* target, void* arg0, void* arg1);
typedef void* (*GuestToHostThunk)(void* target, void* arg0, void* arg1);
typedef void (*ResolveFunctionThunk)();

static constexpr uint32_t GUEST_TRAMPOLINE_BASE = 0x80000000;
static constexpr uint32_t GUEST_TRAMPOLINE_END = 0x80040000;
static constexpr uint32_t GUEST_TRAMPOLINE_MIN_LEN = 8;
static constexpr uint32_t MAX_GUEST_TRAMPOLINES =
    (GUEST_TRAMPOLINE_END - GUEST_TRAMPOLINE_BASE) / GUEST_TRAMPOLINE_MIN_LEN;

#define RESERVE_BLOCK_SHIFT 16
#define RESERVE_NUM_ENTRIES \
  ((1024ULL * 1024ULL * 1024ULL * 4ULL) >> RESERVE_BLOCK_SHIFT)

struct ReserveHelper {
  uint64_t blocks[RESERVE_NUM_ENTRIES / 64];
  ReserveHelper() { memset(blocks, 0, sizeof(blocks)); }
};

struct A64BackendStackpoint {
  uint64_t host_stack_;
  unsigned guest_stack_;
  unsigned guest_return_address_;
};

struct A64BackendContext {
  uint64_t helper_scratch[8];
  ReserveHelper* reserve_helper_;
  uint64_t cached_reserve_value_;
  uint64_t* guest_tick_count;
  A64BackendStackpoint* stackpoints;
  uint64_t cached_reserve_offset;
  uint32_t cached_reserve_bit;
  unsigned int current_stackpoint_depth;
  unsigned int fpcr_fpu;
  unsigned int fpcr_vmx;
  unsigned int flags;
};

class A64Backend : public Backend {
 public:
  static constexpr uint32_t kForceReturnAddress = 0x9FFF0000u;

  explicit A64Backend();
  ~A64Backend() override;

  A64CodeCache* code_cache() const { return code_cache_.get(); }
  uintptr_t emitter_data() const { return emitter_data_; }

  HostToGuestThunk host_to_guest_thunk() const { return host_to_guest_thunk_; }
  GuestToHostThunk guest_to_host_thunk() const { return guest_to_host_thunk_; }
  ResolveFunctionThunk resolve_function_thunk() const {
    return resolve_function_thunk_;
  }

  bool Initialize(Processor* processor) override;
  void CommitExecutableRange(uint32_t guest_low,
                             uint32_t guest_high) override;
  std::unique_ptr<Assembler> CreateAssembler() override;
  std::unique_ptr<GuestFunction> CreateGuestFunction(
      Module* module, uint32_t address) override;
  uint64_t CalculateNextHostInstruction(ThreadDebugInfo* thread_info,
                                        uint64_t current_pc) override;

  void InstallBreakpoint(Breakpoint* breakpoint) override;
  void InstallBreakpoint(Breakpoint* breakpoint, Function* fn) override;
  void UninstallBreakpoint(Breakpoint* breakpoint) override;
  void InitializeBackendContext(void* ctx) override;
  void DeinitializeBackendContext(void* ctx) override;
  void PrepareForReentry(void* ctx) override;
  void SetGuestRoundingMode(void* ctx, unsigned int mode) override;

  uint32_t CreateGuestTrampoline(GuestTrampolineProc proc, void* userdata1,
                                 void* userdata2,
                                 bool long_term) override;
  void FreeGuestTrampoline(uint32_t trampoline_addr) override;

  A64BackendContext* BackendContextForGuestContext(void* ctx) {
    return reinterpret_cast<A64BackendContext*>(
        reinterpret_cast<intptr_t>(ctx) - sizeof(A64BackendContext));
  }

  // Writes the indirection-table slot for a resolved guest function if its
  // table page is committed and the host code lives in the JIT region.
  void TryPatchIndirection(uint32_t guest_address, uint64_t host_address);

 private:
  static bool ExceptionCallbackThunk(Exception* ex, void* data);
  bool ExceptionCallback(Exception* ex);

  std::unique_ptr<A64CodeCache> code_cache_;
  uintptr_t emitter_data_ = 0;

  HostToGuestThunk host_to_guest_thunk_ = nullptr;
  GuestToHostThunk guest_to_host_thunk_ = nullptr;
  ResolveFunctionThunk resolve_function_thunk_ = nullptr;

  alignas(64) ReserveHelper reserve_helper_;
  BitMap guest_trampoline_address_bitmap_;
  uint8_t* guest_trampoline_memory_ = nullptr;
  size_t trampoline_offset_ = 0;
};

}  // namespace a64
}  // namespace backend
}  // namespace cpu
}  // namespace xe

#endif  // XENIA_CPU_BACKEND_A64_A64_BACKEND_H_
