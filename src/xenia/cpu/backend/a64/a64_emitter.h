/**
 * ARM64 Emitter — translates HIR instructions to ARM64 machine code
 */
#ifndef XENIA_CPU_BACKEND_A64_A64_EMITTER_H_
#define XENIA_CPU_BACKEND_A64_A64_EMITTER_H_

#include <vector>

#include "xenia/base/arena.h"
#include "xenia/cpu/backend/a64/a64_asm.h"
#include "xenia/cpu/backend/code_cache_base.h"
#include "xenia/cpu/function.h"
#include "xenia/cpu/function_trace_data.h"
#include "xenia/cpu/hir/hir_builder.h"
#include "xenia/cpu/hir/instr.h"
#include "xenia/cpu/hir/value.h"
#include "xenia/memory.h"

namespace xe {
namespace cpu {
class Processor;
}
}  // namespace xe

namespace xe {
namespace cpu {
namespace backend {
namespace a64 {

class A64Backend;
class A64CodeCache;

// Register allocation: GPR mappings
// x19 = PPCContext*, x20 = MemBase
// x21-x28 = allocatable callee-saved GPRs (8 total)
// x0-x7, x9-x15 = scratch
static constexpr int GPR_COUNT = 8;    // x21-x28
static constexpr int VREG_COUNT = 16;  // v16-v31 (scratch NEON)

// Map HIR register index → ARM64 register
static constexpr GReg kContextReg = X19;
static constexpr GReg kMembaseReg = X20;
static constexpr GReg kScratch0 = X9;
static constexpr GReg kScratch1 = X10;
static constexpr GReg kScratch2 = X11;
static constexpr VReg kVScratch0 = V0;
static constexpr VReg kVScratch1 = V1;
static constexpr VReg kVScratch2 = V2;

class A64Emitter {
 public:
  A64Emitter(A64Backend* backend);
  ~A64Emitter();

  Processor* processor() const { return processor_; }
  A64Backend* backend() const { return backend_; }
  A64Asm& asm_() { return asm__; }

  bool Emit(GuestFunction* function, hir::HIRBuilder* builder,
            uint32_t debug_info_flags, FunctionDebugInfo* debug_info,
            void** out_code_address, size_t* out_code_size,
            std::vector<SourceMapEntry>* out_source_map);

  // Register mapping from HIR virtual reg to ARM64 physical reg
  static GReg GprForIndex(uint32_t index);
  static VReg VregForIndex(uint32_t index);

  // Utility: load a 64-bit immediate
  void MovImm64(GReg rd, uint64_t imm);
  // Load a constant into a NEON register
  void LoadConstantV128(VReg vd, const vec128_t& v);

  // Emit prologue/epilogue for a guest function
  void EmitPrologue(size_t stack_size);
  void EmitEpilogue(size_t stack_size);

  FunctionDebugInfo* debug_info() const { return debug_info_; }
  size_t stack_size() const { return stack_size_; }

  // Mark source offset for debug
  void MarkSourceOffset(const hir::Instr* i);

 private:
  void* Emplace(const EmitFunctionInfo& func_info,
                GuestFunction* function = nullptr);
  bool EmitBody(hir::HIRBuilder* builder, EmitFunctionInfo& func_info);

  Processor* processor_ = nullptr;
  A64Backend* backend_ = nullptr;
  A64CodeCache* code_cache_ = nullptr;
  A64Asm asm__;

  FunctionDebugInfo* debug_info_ = nullptr;
  uint32_t debug_info_flags_ = 0;
  Arena source_map_arena_;
  size_t stack_size_ = 0;
  uint32_t current_guest_function_ = 0;

  // GPR register map: index → physical register number
  static const uint32_t gpr_reg_map_[GPR_COUNT];
  // NEON register map
  static const uint32_t vreg_reg_map_[VREG_COUNT];
};

}  // namespace a64
}  // namespace backend
}  // namespace cpu
}  // namespace xe

#endif  // XENIA_CPU_BACKEND_A64_A64_EMITTER_H_
