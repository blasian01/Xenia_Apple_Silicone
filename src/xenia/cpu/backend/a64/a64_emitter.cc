/**
 * ARM64 Emitter Implementation
 */
#include "xenia/cpu/backend/a64/a64_emitter.h"

#include "xenia/base/logging.h"
#include "xenia/cpu/backend/a64/a64_backend.h"
#include "xenia/cpu/backend/a64/a64_code_cache.h"
#include "xenia/cpu/backend/a64/a64_function.h"
#include "xenia/cpu/backend/a64/a64_sequences.h"
#include "xenia/cpu/backend/a64/a64_stack_layout.h"
#include "xenia/cpu/processor.h"

namespace xe {
namespace cpu {
namespace backend {
namespace a64 {

// Map virtual register indices to physical ARM64 registers.
// x21-x28 are callee-saved and used for register allocation.
const uint32_t A64Emitter::gpr_reg_map_[GPR_COUNT] = {
    X21, X22, X23, X24, X25, X26, X27, X28,
};
// v16-v31 are caller-saved and used for NEON register allocation.
const uint32_t A64Emitter::vreg_reg_map_[VREG_COUNT] = {
    V16, V17, V18, V19, V20, V21, V22, V23,
    V24, V25, V26, V27, V28, V29, V30, V31,
};

A64Emitter::A64Emitter(A64Backend* backend)
    : processor_(backend->processor()),
      backend_(backend),
      code_cache_(backend->code_cache()) {}

A64Emitter::~A64Emitter() = default;

GReg A64Emitter::GprForIndex(uint32_t index) {
  assert(index < GPR_COUNT);
  return static_cast<GReg>(gpr_reg_map_[index]);
}

VReg A64Emitter::VregForIndex(uint32_t index) {
  assert(index < VREG_COUNT);
  return static_cast<VReg>(vreg_reg_map_[index]);
}

void A64Emitter::MovImm64(GReg rd, uint64_t imm) {
  asm__.MOV64(rd, imm);
}

void A64Emitter::LoadConstantV128(VReg vd, const vec128_t& v) {
  // Load 128-bit constant via scratch GPR + INS
  // Store constant to stack scratch area, then load via LDR Q
  asm__.MOV64(kScratch0, v.low);
  asm__.MOV64(kScratch1, v.high);
  // Use stack scratch area at [SP + StackLayout::GUEST_SCRATCH]
  asm__.STR(kScratch0, SP, StackLayout::GUEST_SCRATCH);
  asm__.STR(kScratch1, SP, StackLayout::GUEST_SCRATCH + 8);
  asm__.LDR_Q(vd, SP, StackLayout::GUEST_SCRATCH);
}

void A64Emitter::EmitPrologue(size_t stack_size) {
  // Save LR and allocate stack
  stack_size_ = stack_size;
  size_t total = StackLayout::GUEST_STACK_SIZE + stack_size;
  total = (total + 15) & ~15;  // 16-byte align

  // STP x29, x30, [sp, #-total]! (pre-index)
  asm__.STP_pre(X29, X30, SP, -(int32_t)total);
  // Set frame pointer
  asm__.MOV(X29, SP);
}

void A64Emitter::EmitEpilogue(size_t stack_size) {
  size_t total = StackLayout::GUEST_STACK_SIZE + stack_size;
  total = (total + 15) & ~15;

  // Restore frame and return
  asm__.LDP_post(X29, X30, SP, (int32_t)total);
  asm__.RET();
}

void A64Emitter::MarkSourceOffset(const hir::Instr* i) {
  // Track source mapping for debugging
}

bool A64Emitter::Emit(GuestFunction* function, hir::HIRBuilder* builder,
                      uint32_t debug_info_flags, FunctionDebugInfo* debug_info,
                      void** out_code_address, size_t* out_code_size,
                      std::vector<SourceMapEntry>* out_source_map) {
  asm__.Reset();
  debug_info_ = debug_info;
  debug_info_flags_ = debug_info_flags;
  current_guest_function_ = function->address();

  // Calculate stack size from HIR
  auto block = builder->first_block();
  stack_size_ = 0;
  while (block) {
    auto instr = block->instr_head;
    while (instr) {
      if (instr->dest && (instr->dest->flags & hir::VALUE_IS_ALLOCATED)) {
        // Account for spilled values
      }
      instr = instr->next;
    }
    block = block->next;
  }

  // Emit function info
  EmitFunctionInfo func_info = {};
  size_t prolog_start = asm__.offset();

  // Emit prologue
  EmitPrologue(stack_size_);
  func_info.prolog_stack_alloc_offset = asm__.offset() - prolog_start;
  func_info.code_size.prolog = asm__.offset() - prolog_start;
  func_info.stack_size = stack_size_;

  // Emit body
  size_t body_start = asm__.offset();
  if (!EmitBody(builder, func_info)) {
    return false;
  }
  func_info.code_size.body = asm__.offset() - body_start;

  // Emit epilogue
  size_t epilog_start = asm__.offset();
  EmitEpilogue(stack_size_);
  func_info.code_size.epilog = asm__.offset() - epilog_start;

  func_info.code_size.tail = 0;
  func_info.code_size.total = asm__.code_size();

  // Place code into cache
  void* code_execute_address = nullptr;
  void* code_write_address = nullptr;
  code_cache_->PlaceGuestCode(function->address(),
                              const_cast<uint32_t*>(asm__.code()),
                              func_info, function,
                              code_execute_address, code_write_address);

  *out_code_address = code_execute_address;
  *out_code_size = asm__.code_size();

  return true;
}

bool A64Emitter::EmitBody(hir::HIRBuilder* builder,
                          EmitFunctionInfo& func_info) {
  auto block = builder->first_block();
  while (block) {
    auto instr = block->instr_head;
    while (instr) {
      const hir::Instr* new_tail = nullptr;
      if (!SelectSequence(this, instr, &new_tail)) {
        XELOGE("ARM64: Unimplemented HIR opcode: {}",
               GetOpcodeName(instr->GetOpcodeNum()));
        // Emit NOP for unimplemented opcodes to keep going
        asm__.NOP();
      }
      instr = instr->next;
    }
    block = block->next;
  }
  return true;
}

}  // namespace a64
}  // namespace backend
}  // namespace cpu
}  // namespace xe
