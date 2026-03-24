/**
 * ARM64 Backend Implementation
 */
#include "xenia/cpu/backend/a64/a64_backend.h"

#include <atomic>
#include <mutex>
#include <unordered_set>

#include "xenia/base/byte_order.h"
#include "xenia/base/exception_handler.h"
#include "xenia/base/logging.h"
#include "xenia/cpu/backend/a64/a64_assembler.h"
#include "xenia/cpu/backend/a64/a64_code_cache.h"
#include "xenia/cpu/backend/a64/a64_emitter.h"
#include "xenia/cpu/backend/a64/a64_function.h"
#include "xenia/cpu/backend/a64/a64_sequences.h"
#include "xenia/cpu/backend/a64/a64_stack_layout.h"
#include "xenia/cpu/breakpoint.h"
#include "xenia/cpu/function.h"
#include "xenia/cpu/ppc/ppc_opcode_info.h"
#include "xenia/cpu/processor.h"
#include "xenia/cpu/stack_walker.h"

#ifdef __APPLE__
#include <sys/mman.h>
#include <pthread.h>
#include <libkern/OSCacheControl.h>
#endif

namespace xe {
namespace cpu {
namespace backend {
namespace a64 {

namespace {

bool ShouldDumpHotLoopWindow(uint32_t guest_target) {
  switch (guest_target) {
    case 0x820CC1AC:
    case 0x820CC1B4:
    case 0x820CC1F4:
      return true;
    default:
      return false;
  }
}

bool ShouldLogHotLoopState(uint32_t guest_target) {
  switch (guest_target) {
    case 0x820CC1AC:
    case 0x820CC1B4:
    case 0x820CC1F4:
      return true;
    default:
      return false;
  }
}

void LogHotLoopState(ppc::PPCContext_s* guest_context, uint32_t guest_target) {
  static std::atomic<uint64_t> hot_loop_state_log_count{0};
  uint64_t count = ++hot_loop_state_log_count;
  if (count > 32 && (count % 128) != 0) {
    return;
  }

  XELOGI(
      "ARM64: loop state #{} target={:08X} r26={:08X} r27={:08X} r28={:08X} "
      "r29={:08X} r30={:08X} r31={:08X} lr={:08X} ctr={:08X}",
      count, guest_target, uint32_t(guest_context->r[26]),
      uint32_t(guest_context->r[27]), uint32_t(guest_context->r[28]),
      uint32_t(guest_context->r[29]), uint32_t(guest_context->r[30]),
      uint32_t(guest_context->r[31]), uint32_t(guest_context->lr),
      uint32_t(guest_context->ctr));
}

void LogGuestInstructionWindow(ThreadState* thread_state, Function* fn,
                               uint32_t guest_target) {
  uint32_t window_end = fn->end_address();
  if (!window_end || window_end <= guest_target) {
    window_end = guest_target + 0x40;
  }
  if (window_end > guest_target + 0x80) {
    window_end = guest_target + 0x80;
  }

  XELOGI("ARM64: hot loop window start={:08X} end={:08X} fn_end={:08X}",
         guest_target, window_end, fn->end_address());
  for (uint32_t address = guest_target; address < window_end; address += 4) {
    auto* guest_code_ptr =
        thread_state->processor()->memory()->TranslateVirtual<const uint32_t*>(
            address);
    if (!guest_code_ptr) {
      XELOGI("ARM64:   {:08X}: <unmapped>", address);
      break;
    }

    uint32_t guest_code = xe::byte_swap(*guest_code_ptr);
    StringBuffer disasm;
    if (!ppc::DisasmPPC(address, guest_code, &disasm)) {
      XELOGI("ARM64:   {:08X}: {:08X}", address, guest_code);
      continue;
    }

    XELOGI("ARM64:   {:08X}: {:08X} {}", address, guest_code,
           disasm.buffer());
  }
}

void LogGuestInstructionRange(ThreadState* thread_state, uint32_t center,
                              uint32_t instruction_count_before,
                              uint32_t instruction_count_after) {
  uint32_t start =
      center >= instruction_count_before * 4
          ? center - instruction_count_before * 4
          : 0;
  uint32_t end = center + instruction_count_after * 4;
  XELOGI("ARM64: guest instruction range start={:08X} center={:08X} end={:08X}",
         start, center, end);
  for (uint32_t address = start; address <= end; address += 4) {
    auto* guest_code_ptr =
        thread_state->processor()->memory()->TranslateVirtual<const uint32_t*>(
            address);
    if (!guest_code_ptr) {
      XELOGI("ARM64:   {:08X}: <unmapped>", address);
      continue;
    }

    uint32_t guest_code = xe::byte_swap(*guest_code_ptr);
    StringBuffer disasm;
    if (!ppc::DisasmPPC(address, guest_code, &disasm)) {
      XELOGI("ARM64:   {:08X}: {:08X}", address, guest_code);
      continue;
    }

    XELOGI("ARM64:   {:08X}: {:08X} {}", address, guest_code,
           disasm.buffer());
  }
}

}  // namespace

// C++ resolve helper — called from the resolve-function thunk
// ARM64 ABI: x0 = raw_context (PPCContext*), x1 = target PPC address
// Returns: host machine code address in x0
extern "C" uint64_t A64ResolveFunction(void* raw_context,
                                        uint64_t target_address) {
  static std::atomic<uint64_t> resolve_count{0};
  static std::mutex hot_target_log_mutex;
  static std::unordered_set<uint32_t> hot_target_log_once;
  auto count = ++resolve_count;
  if (count <= 10 || (count % 100) == 0) {
    XELOGI("ARM64: ResolveFunction #{} target={:08X}", count, target_address);
  }

  auto guest_context = reinterpret_cast<ppc::PPCContext_s*>(raw_context);
  auto thread_state = guest_context->thread_state;
  if (!target_address) {
    XELOGE(
        "ARM64: ResolveFunction requested null target lr={:08X} ctr={:08X} "
        "r1={:08X} r13={:08X}",
        uint32_t(guest_context->lr), uint32_t(guest_context->ctr),
        uint32_t(guest_context->r[1]), uint32_t(guest_context->r[13]));
    LogGuestInstructionRange(thread_state, uint32_t(guest_context->lr), 6, 6);
  }
  auto fn = thread_state->processor()->ResolveFunction(
      static_cast<uint32_t>(target_address));
  if (!fn) {
    XELOGE("ARM64: Failed to resolve function at {:08X}", target_address);
    return 0;
  }
  auto a64_fn = static_cast<A64Function*>(fn);

  uint32_t guest_target = static_cast<uint32_t>(target_address);
  if (ShouldLogHotLoopState(guest_target)) {
    LogHotLoopState(guest_context, guest_target);
  }
  if (guest_target >= 0x820C0000 && guest_target < 0x820D0000) {
    bool should_log = false;
    {
      std::lock_guard<std::mutex> lock(hot_target_log_mutex);
      should_log = hot_target_log_once.insert(guest_target).second;
    }
    if (should_log) {
      uint32_t guest_code = 0;
      StringBuffer disasm;
      if (auto* guest_code_ptr = thread_state->processor()->memory()
                                     ->TranslateVirtual<const uint32_t*>(
                                         guest_target)) {
        guest_code = xe::byte_swap(*guest_code_ptr);
        ppc::DisasmPPC(guest_target, guest_code, &disasm);
      }
      XELOGI(
          "ARM64: hot target {:08X} -> {:p} len={} behavior={} opcode={:08X} {}",
          guest_target, static_cast<void*>(a64_fn->machine_code()),
          a64_fn->machine_code_length(), static_cast<int>(fn->behavior()),
          guest_code, disasm.buffer());
      if (ShouldDumpHotLoopWindow(guest_target)) {
        LogGuestInstructionWindow(thread_state, fn, guest_target);
      }
    }
  }

  return reinterpret_cast<uint64_t>(a64_fn->machine_code());
}

A64Backend::A64Backend() {
  machine_info_.register_sets[0] = {
      0, "gpr", MachineInfo::RegisterSet::INT_TYPES, GPR_COUNT,
  };
  machine_info_.register_sets[1] = {
      1, "vec", MachineInfo::RegisterSet::FLOAT_TYPES |
                MachineInfo::RegisterSet::VEC_TYPES, VREG_COUNT,
  };
}

A64Backend::~A64Backend() = default;

bool A64Backend::Initialize(Processor* processor) {
  if (!Backend::Initialize(processor)) {
    return false;
  }

  // Register HIR opcode sequences
  RegisterSequences();

  // Create code cache
  code_cache_ = A64CodeCache::Create();
  Backend::code_cache_ = code_cache_.get();
  if (!code_cache_->Initialize()) {
    XELOGE("ARM64: Failed to initialize code cache");
    return false;
  }

  // Allocate guest trampoline memory using MAP_JIT for Apple Silicon
#ifdef __APPLE__
  guest_trampoline_memory_ = reinterpret_cast<uint8_t*>(
      mmap(nullptr, GUEST_TRAMPOLINE_END - GUEST_TRAMPOLINE_BASE,
           PROT_READ | PROT_WRITE | PROT_EXEC,
           MAP_PRIVATE | MAP_ANONYMOUS | MAP_JIT, -1, 0));
  if (guest_trampoline_memory_ == MAP_FAILED) {
    XELOGW("ARM64: Failed to allocate trampoline memory (non-fatal)");
    guest_trampoline_memory_ = nullptr;
  }
#else
  guest_trampoline_memory_ = reinterpret_cast<uint8_t*>(
      xe::memory::AllocFixed(
          nullptr,
          GUEST_TRAMPOLINE_END - GUEST_TRAMPOLINE_BASE,
          xe::memory::AllocationType::kCommit,
          xe::memory::PageAccess::kExecuteReadWrite));
#endif
  trampoline_offset_ = 0;
  XELOGI("ARM64: Trampoline memory: {:p}",
         (void*)guest_trampoline_memory_);

  XELOGI("ARM64: Generating thunks...");
  // Generate thunks using our ARM64 assembler
  A64Asm thunk_asm;

  // === Host-to-Guest Thunk ===
  // Callable as: void thunk(void* target, void* context, void* ret_addr)
  // ARM64 AAPCS64: x0=target, x1=context, x2=ret_addr
  {
    // Reserve the frame explicitly so the fixed offsets in StackLayout match
    // the actual save locations. Using a pre-index STP for the full frame
    // stores X29/X30 at SP+0 and would overlap the X19/X20 slots below.
    thunk_asm.SUB(SP, SP, static_cast<uint32_t>(StackLayout::THUNK_STACK_SIZE));
    thunk_asm.STP(X29, X30, SP, 0x50);
    thunk_asm.ADD(X29, SP, 0x50);

    // Save callee-saved GPRs
    thunk_asm.STP(X19, X20, SP, 0x00);
    thunk_asm.STP(X21, X22, SP, 0x10);
    thunk_asm.STP(X23, X24, SP, 0x20);
    thunk_asm.STP(X25, X26, SP, 0x30);
    thunk_asm.STP(X27, X28, SP, 0x40);

    // Save callee-saved NEON (lower 64 bits per AAPCS64, save full Q)
    thunk_asm.STP_Q(V8, V9, SP, 0x60);
    thunk_asm.STP_Q(V10, V11, SP, 0x80);
    thunk_asm.STP_Q(V12, V13, SP, 0xA0);
    thunk_asm.STP_Q(V14, V15, SP, 0xC0);

    // Set up context and membase registers
    thunk_asm.MOV(X19, X1);  // x19 = PPCContext*
    thunk_asm.LDR(X20, X19, offsetof(ppc::PPCContext, virtual_membase));

    // Call the target (guest function)
    thunk_asm.BLR(X0);

    // Restore callee-saved NEON
    thunk_asm.LDP_Q(V8, V9, SP, 0x60);
    thunk_asm.LDP_Q(V10, V11, SP, 0x80);
    thunk_asm.LDP_Q(V12, V13, SP, 0xA0);
    thunk_asm.LDP_Q(V14, V15, SP, 0xC0);

    // Restore callee-saved GPRs
    thunk_asm.LDP(X19, X20, SP, 0x00);
    thunk_asm.LDP(X21, X22, SP, 0x10);
    thunk_asm.LDP(X23, X24, SP, 0x20);
    thunk_asm.LDP(X25, X26, SP, 0x30);
    thunk_asm.LDP(X27, X28, SP, 0x40);

    // Restore frame and return.
    thunk_asm.LDP(X29, X30, SP, 0x50);
    thunk_asm.ADD(SP, SP, static_cast<uint32_t>(StackLayout::THUNK_STACK_SIZE));
    thunk_asm.RET();
  }

  // Place host-to-guest thunk
  EmitFunctionInfo thunk_info = {};
  thunk_info.code_size.total = thunk_asm.code_size();
  thunk_info.stack_size = StackLayout::THUNK_STACK_SIZE;
  void* h2g_execute = nullptr;
  void* h2g_write = nullptr;
  code_cache_->PlaceHostCode(0, const_cast<uint32_t*>(thunk_asm.code()),
                             thunk_info, h2g_execute, h2g_write);
  code_cache_->FlushCodeRange(h2g_execute, thunk_asm.code_size());
  host_to_guest_thunk_ = reinterpret_cast<HostToGuestThunk>(h2g_execute);

  // === Guest-to-Host Thunk ===
  // Called from guest to invoke a host C++ function.
  // ARM64: x0 = target C function, x1 = arg0, x2 = arg1
  // We need to: save volatile guest regs, set x0=context for the callee,
  // call target, restore, return.
  thunk_asm.Reset();
  {
    // Save x19 (context) to stack since target may clobber it
    thunk_asm.STP_pre(X29, X30, SP, -32);
    thunk_asm.ADD(X29, SP, 0);
    // Save the target fn pointer
    thunk_asm.MOV(X9, X0);
    // Set args: x0 = context (x19), x1 = arg0, x2 = arg1
    thunk_asm.MOV(X0, X19);  // context
    // x1, x2 already have arg0, arg1
    thunk_asm.BLR(X9);
    // Restore frame and return
    thunk_asm.LDP_post(X29, X30, SP, 32);
    thunk_asm.RET();
  }
  thunk_info.code_size.total = thunk_asm.code_size();
  void* g2h_execute = nullptr;
  void* g2h_write = nullptr;
  code_cache_->PlaceHostCode(0, const_cast<uint32_t*>(thunk_asm.code()),
                             thunk_info, g2h_execute, g2h_write);
  code_cache_->FlushCodeRange(g2h_execute, thunk_asm.code_size());
  guest_to_host_thunk_ = reinterpret_cast<GuestToHostThunk>(g2h_execute);

  // === Resolve Function Thunk ===
  // Called when a guest function isn't yet compiled.
  // Inputs: x19 = PPCContext*, x9 = target PPC address
  // Calls: A64ResolveFunction(context, ppc_addr) → host addr in x0
  // Then jumps to the resolved host address.
  thunk_asm.Reset();
  {
    thunk_asm.STP_pre(X29, X30, SP, -64);
    thunk_asm.ADD(X29, SP, 0);
    // Save callee-saved that we clobber
    thunk_asm.STP(X19, X20, SP, 16);
    // Preserve the synthetic guest return address across the resolve call.
    thunk_asm.STR(X2, SP, 32);

    // Call A64ResolveFunction(context=x19, target=x9)
    thunk_asm.MOV(X0, X19);  // arg0: context
    thunk_asm.MOV(X1, X9);   // arg1: PPC target address
    thunk_asm.MOV64(X8, reinterpret_cast<uint64_t>(&A64ResolveFunction));
    thunk_asm.BLR(X8);

    // x0 now has host machine code address
    thunk_asm.MOV(X8, X0);

    // Restore and jump to resolved function
    thunk_asm.LDR(X2, SP, 32);
    thunk_asm.LDP(X19, X20, SP, 16);
    thunk_asm.LDP_post(X29, X30, SP, 64);
    thunk_asm.BR(X8);  // Jump (not call) to resolved function
  }
  thunk_info.code_size.total = thunk_asm.code_size();
  void* resolve_execute = nullptr;
  void* resolve_write = nullptr;
  code_cache_->PlaceHostCode(0, const_cast<uint32_t*>(thunk_asm.code()),
                             thunk_info, resolve_execute, resolve_write);
  code_cache_->FlushCodeRange(resolve_execute, thunk_asm.code_size());
  resolve_function_thunk_ =
      reinterpret_cast<ResolveFunctionThunk>(resolve_execute);

  // Set indirection table default to resolve thunk address
  if (code_cache_->has_indirection_table()) {
    code_cache_->set_indirection_default(
        static_cast<uint32_t>(reinterpret_cast<uintptr_t>(
            resolve_function_thunk_)));
  }

  // Emitter data pointer (no constant pool needed yet)
  emitter_data_ = 0;

  // Register exception handler
  ExceptionHandler::Install(ExceptionCallbackThunk, this);

  XELOGI("ARM64 backend initialized successfully");
  XELOGI("  Host-to-Guest thunk: {:p}", (void*)host_to_guest_thunk_);
  XELOGI("  Guest-to-Host thunk: {:p}", (void*)guest_to_host_thunk_);
  XELOGI("  Resolve thunk:       {:p}", (void*)resolve_function_thunk_);

  return true;
}

void A64Backend::CommitExecutableRange(uint32_t guest_low,
                                       uint32_t guest_high) {
  code_cache_->CommitExecutableRange(guest_low, guest_high);
}

std::unique_ptr<Assembler> A64Backend::CreateAssembler() {
  return std::make_unique<A64Assembler>(this);
}

std::unique_ptr<GuestFunction> A64Backend::CreateGuestFunction(
    Module* module, uint32_t address) {
  return std::make_unique<A64Function>(module, address);
}

uint64_t A64Backend::CalculateNextHostInstruction(
    ThreadDebugInfo* thread_info, uint64_t current_pc) {
  // ARM64: instructions are always 4 bytes
  return current_pc + 4;
}

void A64Backend::InstallBreakpoint(Breakpoint* breakpoint) {}
void A64Backend::InstallBreakpoint(Breakpoint* breakpoint, Function* fn) {}
void A64Backend::UninstallBreakpoint(Breakpoint* breakpoint) {}

void A64Backend::InitializeBackendContext(void* ctx) {
  auto* a64ctx = BackendContextForGuestContext(ctx);
  memset(a64ctx, 0, sizeof(A64BackendContext));
  a64ctx->reserve_helper_ = &reserve_helper_;
}

void A64Backend::DeinitializeBackendContext(void* ctx) {
  auto* a64ctx = BackendContextForGuestContext(ctx);
  if (a64ctx->stackpoints) {
    delete[] a64ctx->stackpoints;
    a64ctx->stackpoints = nullptr;
  }
}

void A64Backend::PrepareForReentry(void* ctx) {
  auto* a64ctx = BackendContextForGuestContext(ctx);
  a64ctx->current_stackpoint_depth = 0;
}

void A64Backend::SetGuestRoundingMode(void* ctx, unsigned int mode) {
  // ARM64 FPCR rounding mode is in bits [23:22]
  // 00 = Round to Nearest, 01 = Round to +Inf,
  // 10 = Round to -Inf,    11 = Round to Zero
  // This is called from interpreter path; JIT uses SET_ROUNDING_MODE opcode
}

uint32_t A64Backend::CreateGuestTrampoline(GuestTrampolineProc proc,
                                           void* userdata1, void* userdata2,
                                           bool long_term) {
  if (!guest_trampoline_memory_) return 0;

  // Emit a small trampoline that calls proc(userdata1, userdata2)
  A64Asm tramp;
  tramp.MOV64(X0, reinterpret_cast<uint64_t>(userdata1));
  tramp.MOV64(X1, reinterpret_cast<uint64_t>(userdata2));
  tramp.MOV64(X8, reinterpret_cast<uint64_t>(proc));
  tramp.BR(X8);

  // Place into trampoline memory
  size_t size = tramp.code_size();
  if (trampoline_offset_ + size > (GUEST_TRAMPOLINE_END - GUEST_TRAMPOLINE_BASE)) {
    return 0;  // Out of space
  }

  uint8_t* dest = guest_trampoline_memory_ + trampoline_offset_;
#ifdef __APPLE__
  pthread_jit_write_protect_np(0);
#endif
  memcpy(dest, tramp.code(), size);
#ifdef __APPLE__
  pthread_jit_write_protect_np(1);
#endif

#ifdef __APPLE__
  sys_dcache_flush(dest, size);
  sys_icache_invalidate(dest, size);
#else
  __builtin___clear_cache(reinterpret_cast<char*>(dest),
                          reinterpret_cast<char*>(dest) + size);
#endif

  uint32_t addr = GUEST_TRAMPOLINE_BASE + (uint32_t)trampoline_offset_;
  trampoline_offset_ += (size + 15) & ~15;  // 16-byte align
  return addr;
}

void A64Backend::FreeGuestTrampoline(uint32_t trampoline_addr) {}

bool A64Backend::ExceptionCallbackThunk(Exception* ex, void* data) {
  return static_cast<A64Backend*>(data)->ExceptionCallback(ex);
}

bool A64Backend::ExceptionCallback(Exception* ex) {
  uint32_t inst = 0;
  if (ex->pc()) {
    // Only read inst if it's safe - but we could fault here. Ignore for now.
    // inst = *reinterpret_cast<uint32_t*>(ex->pc());
  }
  
  // Handle Access Violations
  if (ex->code() == Exception::Code::kAccessViolation) {
    XELOGE("ARM64: PAGE FAULT / ACCESS VIOLATION at pc={:016X} fault_addr={:016X} write={}",
           ex->pc(), ex->fault_address(), 
           ex->access_violation_operation() == Exception::AccessViolationOperation::kWrite);
    if (auto* thread_context = ex->thread_context()) {
      XELOGE(
          "ARM64: regs x19={:016X} x20={:016X} x21={:016X} x22={:016X} "
          "x23={:016X} x24={:016X} x25={:016X} x26={:016X} x27={:016X} "
          "x28={:016X} sp={:016X}",
          thread_context->x[19], thread_context->x[20], thread_context->x[21],
          thread_context->x[22], thread_context->x[23], thread_context->x[24],
          thread_context->x[25], thread_context->x[26], thread_context->x[27],
          thread_context->x[28], thread_context->sp);
      auto* guest_context =
          reinterpret_cast<ppc::PPCContext*>(thread_context->x[19]);
      if (guest_context) {
        XELOGE(
            "ARM64: guest ctx r1={:08X} r3={:08X} r13={:08X} lr={:08X} "
            "ctr={:08X} cr0={:02X}{:02X}{:02X}{:02X}",
            uint32_t(guest_context->r[1]), uint32_t(guest_context->r[3]),
            uint32_t(guest_context->r[13]), uint32_t(guest_context->lr),
            uint32_t(guest_context->ctr), uint8_t(guest_context->cr0.cr0_lt),
            uint8_t(guest_context->cr0.cr0_gt),
            uint8_t(guest_context->cr0.cr0_eq),
            uint8_t(guest_context->cr0.cr0_so));
      }
    }
    if (code_cache_) {
      if (auto* function =
              static_cast<A64Function*>(code_cache_->LookupFunction(ex->pc()))) {
        auto code_offset = uint32_t(ex->pc() -
                                    reinterpret_cast<uintptr_t>(
                                        function->machine_code()));
        XELOGE("ARM64: fault in guest function {:08X} host_offset={:X}",
               function->address(), code_offset);
        if (auto* source_entry = function->LookupMachineCodeOffset(code_offset)) {
          XELOGE("ARM64: nearest guest instruction {:08X} hir_offset={:08X}",
                 source_entry->guest_address, source_entry->hir_offset);
          if (auto* guest_code_ptr = processor()->memory()->TranslateVirtual<
                  const uint32_t*>(source_entry->guest_address)) {
            uint32_t guest_code = xe::byte_swap(*guest_code_ptr);
            StringBuffer disasm;
            if (ppc::DisasmPPC(source_entry->guest_address, guest_code,
                               &disasm)) {
              XELOGE("ARM64: guest opcode {:08X} {}", guest_code,
                     disasm.buffer());
            }
          }
        }
      }
    }
           
    // Let other handlers (like emulator.cc or mmio_handler) take care of it
    return false;
  }

  // Handle BRK instructions as breakpoints
  if (ex->code() == Exception::Code::kIllegalInstruction) {
    XELOGE("ARM64: BRK/illegal instruction at host PC {:016X} — likely unimplemented opcode",
           ex->pc());
    return processor()->OnThreadBreakpointHit(ex);
  }
  return false;
}

}  // namespace a64
}  // namespace backend
}  // namespace cpu
}  // namespace xe
