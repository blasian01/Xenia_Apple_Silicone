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
#include "xenia/cpu/hir/label.h"
#include "xenia/cpu/processor.h"
#include "xenia/base/math.h"

namespace xe {
namespace cpu {
namespace backend {
namespace a64 {

// Map virtual register indices to physical ARM64 registers.
const uint32_t A64Emitter::gpr_reg_map_[GPR_COUNT] = {
    X21, X22, X23, X24, X25, X26, X27, X28,
};
const uint32_t A64Emitter::vreg_reg_map_[VREG_COUNT] = {
    V16, V17, V18, V19, V20, V21, V22, V23,
    V24, V25, V26, V27, V28, V29, V30, V31,
};

A64Emitter::A64Emitter(A64Backend* backend)
    : processor_(backend->processor()),
      backend_(backend),
      code_cache_(backend->code_cache()) {}

A64Emitter::~A64Emitter() {
  for (auto* label : label_storage_) {
    delete label;
  }
}

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
  asm__.MOV64(kScratch0, v.low);
  asm__.MOV64(kScratch1, v.high);
  asm__.STR(kScratch0, X29, StackLayout::GUEST_SCRATCH);
  asm__.STR(kScratch1, X29, StackLayout::GUEST_SCRATCH + 8);
  asm__.LDR_Q(vd, X29, StackLayout::GUEST_SCRATCH);
}

a64::Label* A64Emitter::GetLabel(uint32_t hir_label_id) {
  auto it = label_map_.find(hir_label_id);
  if (it != label_map_.end()) {
    return it->second;
  }
  auto* label = new a64::Label();
  label_map_[hir_label_id] = label;
  label_storage_.push_back(label);
  return label;
}

void A64Emitter::BindLabel(uint32_t hir_label_id) {
  auto* label = GetLabel(hir_label_id);
  asm__.Bind(label);
}

void A64Emitter::EmitPrologue(size_t stack_size) {
  stack_size_ = stack_size;
  size_t total = StackLayout::HOST_FRAME_SAVE_SIZE +
                 StackLayout::GUEST_STACK_SIZE + stack_size;
  total = (total + 15) & ~15;  // 16-byte align

  asm__.STP_pre(X29, X30, SP, -(int32_t)total);
  asm__.ADD(X29, SP, StackLayout::HOST_FRAME_SAVE_SIZE);
  asm__.STR(X2, X29, StackLayout::GUEST_RET_ADDR);
  asm__.MOVZ(kScratch0, 0);
  asm__.STR(kScratch0, X29, StackLayout::GUEST_CALL_RET_ADDR);

  // Guest functions may call other translated functions, so preserve the
  // allocatable register file across nested BLR / tail-call boundaries.
  asm__.STP(X21, X22, X29, StackLayout::GUEST_GPR_SAVE + 0x00);
  asm__.STP(X23, X24, X29, StackLayout::GUEST_GPR_SAVE + 0x10);
  asm__.STP(X25, X26, X29, StackLayout::GUEST_GPR_SAVE + 0x20);
  asm__.STP(X27, X28, X29, StackLayout::GUEST_GPR_SAVE + 0x30);

  asm__.STP_Q(V16, V17, X29, StackLayout::GUEST_VREG_SAVE + 0x00);
  asm__.STP_Q(V18, V19, X29, StackLayout::GUEST_VREG_SAVE + 0x20);
  asm__.STP_Q(V20, V21, X29, StackLayout::GUEST_VREG_SAVE + 0x40);
  asm__.STP_Q(V22, V23, X29, StackLayout::GUEST_VREG_SAVE + 0x60);
  asm__.STP_Q(V24, V25, X29, StackLayout::GUEST_VREG_SAVE + 0x80);
  asm__.STP_Q(V26, V27, X29, StackLayout::GUEST_VREG_SAVE + 0xA0);
  asm__.STP_Q(V28, V29, X29, StackLayout::GUEST_VREG_SAVE + 0xC0);
  asm__.STP_Q(V30, V31, X29, StackLayout::GUEST_VREG_SAVE + 0xE0);
}

void A64Emitter::EmitEpilogue(size_t stack_size) {
  size_t total = StackLayout::HOST_FRAME_SAVE_SIZE +
                 StackLayout::GUEST_STACK_SIZE + stack_size;
  total = (total + 15) & ~15;

  asm__.LDP_Q(V16, V17, X29, StackLayout::GUEST_VREG_SAVE + 0x00);
  asm__.LDP_Q(V18, V19, X29, StackLayout::GUEST_VREG_SAVE + 0x20);
  asm__.LDP_Q(V20, V21, X29, StackLayout::GUEST_VREG_SAVE + 0x40);
  asm__.LDP_Q(V22, V23, X29, StackLayout::GUEST_VREG_SAVE + 0x60);
  asm__.LDP_Q(V24, V25, X29, StackLayout::GUEST_VREG_SAVE + 0x80);
  asm__.LDP_Q(V26, V27, X29, StackLayout::GUEST_VREG_SAVE + 0xA0);
  asm__.LDP_Q(V28, V29, X29, StackLayout::GUEST_VREG_SAVE + 0xC0);
  asm__.LDP_Q(V30, V31, X29, StackLayout::GUEST_VREG_SAVE + 0xE0);

  asm__.LDP(X21, X22, X29, StackLayout::GUEST_GPR_SAVE + 0x00);
  asm__.LDP(X23, X24, X29, StackLayout::GUEST_GPR_SAVE + 0x10);
  asm__.LDP(X25, X26, X29, StackLayout::GUEST_GPR_SAVE + 0x20);
  asm__.LDP(X27, X28, X29, StackLayout::GUEST_GPR_SAVE + 0x30);

  asm__.LDP(X29, X30, SP, 0);
  asm__.ADD(SP, SP, (int32_t)total);
  asm__.RET();
}

void A64Emitter::MarkSourceOffset(const hir::Instr* i) {
  SourceMapEntry entry;
  entry.guest_address = static_cast<uint32_t>(i->src1.offset);
  entry.hir_offset = uint32_t(i->block->ordinal << 16) | i->ordinal;
  entry.code_offset = static_cast<uint32_t>(asm__.offset());
  source_map_entries_.push_back(entry);
}

bool A64Emitter::Emit(GuestFunction* function, hir::HIRBuilder* builder,
                      uint32_t debug_info_flags, FunctionDebugInfo* debug_info,
                      void** out_code_address, size_t* out_code_size,
                      std::vector<SourceMapEntry>* out_source_map) {
  asm__.Reset();
  debug_info_ = debug_info;
  debug_info_flags_ = debug_info_flags;
  current_guest_function_ = function->address();
  source_map_entries_.clear();

  // Clear label map from previous emission
  label_map_.clear();
  for (auto* label : label_storage_) {
    delete label;
  }
  label_storage_.clear();

  // Pre-create labels for all HIR blocks
  auto block = builder->first_block();
  while (block) {
    auto hir_label = block->label_head;
    while (hir_label) {
      GetLabel(hir_label->id);  // creates if not exists
      hir_label = hir_label->next;
    }
    block = block->next;
  }

  // Lay out local slots after the fixed guest stack header, mirroring x64.
  size_t stack_offset = StackLayout::GUEST_STACK_SIZE;
  for (auto* slot : builder->locals()) {
    const size_t type_size = hir::GetTypeSize(slot->type);
    stack_offset = xe::align(stack_offset, type_size);
    slot->set_constant(static_cast<uint32_t>(stack_offset));
    stack_offset += type_size;
  }
  stack_offset -= StackLayout::GUEST_STACK_SIZE;
  stack_size_ = xe::align(stack_offset, static_cast<size_t>(16));

  // Emit function info
  EmitFunctionInfo func_info = {};
  size_t prolog_start = asm__.offset();

  // Emit prologue
  EmitPrologue(stack_size_);
  func_info.prolog_stack_alloc_offset = asm__.offset() - prolog_start;
  func_info.code_size.prolog = asm__.offset() - prolog_start;
  func_info.stack_size = StackLayout::HOST_FRAME_SAVE_SIZE +
                         StackLayout::GUEST_STACK_SIZE + stack_size_;

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
  *out_source_map = source_map_entries_;

  return true;
}

bool A64Emitter::EmitBody(hir::HIRBuilder* builder,
                          EmitFunctionInfo& func_info) {
  auto block = builder->first_block();
  while (block) {
    // Bind all labels for this block
    auto hir_label = block->label_head;
    while (hir_label) {
      BindLabel(hir_label->id);
      hir_label = hir_label->next;
    }

    // Process instructions
    auto instr = block->instr_head;
    while (instr) {
      const hir::Instr* new_tail = nullptr;
      if (!SelectSequence(this, instr, &new_tail)) {
        XELOGE("ARM64: Unimplemented HIR opcode: {}",
               GetOpcodeName(instr->GetOpcodeNum()));
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
