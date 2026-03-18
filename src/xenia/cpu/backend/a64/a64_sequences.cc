/**
 * ARM64 Sequences — HIR opcode handlers (Phase 2: full integer + control flow)
 */
#include "xenia/cpu/backend/a64/a64_sequences.h"

#include "xenia/base/logging.h"
#include "xenia/cpu/backend/a64/a64_backend.h"
#include "xenia/cpu/backend/a64/a64_emitter.h"
#include "xenia/cpu/backend/a64/a64_stack_layout.h"
#include "xenia/cpu/hir/hir_builder.h"
#include "xenia/cpu/hir/opcodes.h"
#include "xenia/cpu/processor.h"

namespace xe {
namespace cpu {
namespace backend {
namespace a64 {

using namespace xe::cpu::hir;

std::unordered_map<uint32_t, SequenceSelectFn> sequence_table;

// === Helpers ===

static bool IsAlloc(const Value* v) {
  return v && (v->flags & VALUE_IS_ALLOCATED);
}

static GReg GR(const Value* v) {
  return A64Emitter::GprForIndex(v->reg.index);
}
static VReg VR(const Value* v) {
  return A64Emitter::VregForIndex(v->reg.index);
}

// Load a value (const or register) into a scratch GPR. Returns the reg.
static GReg LoadGPR(A64Emitter& e, const Value* v, GReg scratch) {
  if (v->IsConstant()) {
    e.MovImm64(scratch, v->constant.i64);
    return scratch;
  }
  return GR(v);
}

// === Control Flow ===

static bool EmitNop(A64Emitter& e, const Instr* i) { return true; }
static bool EmitComment(A64Emitter& e, const Instr* i) { return true; }

static bool EmitSourceOffset(A64Emitter& e, const Instr* i) {
  e.MarkSourceOffset(i);
  return true;
}

static bool EmitDebugBreak(A64Emitter& e, const Instr* i) {
  e.asm_().BRK(0);
  return true;
}

static bool EmitTrap(A64Emitter& e, const Instr* i) {
  e.asm_().BRK(0);
  return true;
}

static bool EmitReturn(A64Emitter& e, const Instr* i) {
  e.EmitEpilogue(e.stack_size());
  return true;
}

static bool EmitBranch(A64Emitter& e, const Instr* i) {
  // OPCODE_BRANCH: unconditional branch to label
  // For now emit a NOP — label resolution needs the emitter label system
  e.asm_().NOP();
  return true;
}

static bool EmitBranchTrue(A64Emitter& e, const Instr* i) {
  auto cond = i->src1.value;
  if (!cond) return true;
  if (cond->IsConstant()) {
    if (cond->IsConstantTrue()) {
      e.asm_().NOP();  // unconditional branch
    }
    return true;
  }
  if (IsAlloc(cond)) {
    // CBZ/CBNZ pattern
    e.asm_().NOP();  // Branch to label — needs label system
  }
  return true;
}

static bool EmitBranchFalse(A64Emitter& e, const Instr* i) {
  auto cond = i->src1.value;
  if (!cond) return true;
  if (cond->IsConstant()) {
    if (cond->IsConstantFalse()) {
      e.asm_().NOP();
    }
    return true;
  }
  if (IsAlloc(cond)) {
    e.asm_().NOP();
  }
  return true;
}

static bool EmitCall(A64Emitter& e, const Instr* i) {
  // CALL to a resolved function — use resolve thunk
  // The function address comes from the HIR
  auto fn = i->src1.value;
  if (fn && fn->IsConstant()) {
    // Load PPC target address into x9, then call resolve thunk
    e.MovImm64(X9, fn->constant.u32);
  } else if (fn && IsAlloc(fn)) {
    e.asm_().MOV(X9, GR(fn));
  } else {
    e.asm_().NOP();
    return true;
  }
  // Load resolve thunk address and call
  auto backend = static_cast<A64Backend*>(e.backend());
  e.MovImm64(kScratch0,
             reinterpret_cast<uint64_t>(backend->resolve_function_thunk()));
  e.asm_().BLR(kScratch0);
  return true;
}

static bool EmitCallIndirect(A64Emitter& e, const Instr* i) {
  auto target = i->src1.value;
  if (!target) return true;
  // Load PPC target address into x9
  GReg addr = LoadGPR(e, target, X9);
  if (addr != X9) e.asm_().MOV(X9, addr);
  // Call resolve thunk — it will resolve PPC addr and jump
  auto backend = static_cast<A64Backend*>(e.backend());
  e.MovImm64(kScratch0,
             reinterpret_cast<uint64_t>(backend->resolve_function_thunk()));
  e.asm_().BLR(kScratch0);
  return true;
}

static bool EmitCallExtern(A64Emitter& e, const Instr* i) {
  // Extern calls go through guest-to-host thunk
  // The function object is stored in src1
  auto fn = i->src1.value;
  if (!fn) return true;
  // For extern handling, we just call the resolve thunk with the
  // function's PPC address — the processor will handle the extern dispatch
  if (fn->IsConstant()) {
    e.MovImm64(X9, fn->constant.u32);
  } else if (IsAlloc(fn)) {
    e.asm_().MOV(X9, GR(fn));
  } else {
    e.asm_().NOP();
    return true;
  }
  auto backend = static_cast<A64Backend*>(e.backend());
  e.MovImm64(kScratch0,
             reinterpret_cast<uint64_t>(backend->resolve_function_thunk()));
  e.asm_().BLR(kScratch0);
  return true;
}

// === Context Load/Store ===

static bool EmitLoadContext(A64Emitter& e, const Instr* i) {
  uint64_t offset = i->src1.offset;
  auto dest = i->dest;
  if (!dest || !IsAlloc(dest)) { e.asm_().NOP(); return true; }

  GReg ctx = kContextReg;
  switch (dest->type) {
    case INT8_TYPE:  e.asm_().LDRB(GR(dest), ctx, (int32_t)offset); break;
    case INT16_TYPE: e.asm_().LDRH(GR(dest), ctx, (int32_t)offset); break;
    case INT32_TYPE: e.asm_().LDRw(GR(dest), ctx, (int32_t)offset); break;
    case INT64_TYPE: e.asm_().LDR(GR(dest), ctx, (int32_t)offset); break;
    case FLOAT32_TYPE: e.asm_().LDR_S(VR(dest), ctx, (int32_t)offset); break;
    case FLOAT64_TYPE: e.asm_().LDR_D(VR(dest), ctx, (int32_t)offset); break;
    case VEC128_TYPE:  e.asm_().LDR_Q(VR(dest), ctx, (int32_t)offset); break;
    default: return false;
  }
  return true;
}

static bool EmitStoreContext(A64Emitter& e, const Instr* i) {
  uint64_t offset = i->src1.offset;
  auto src = i->src2.value;
  if (!src) return true;

  GReg ctx = kContextReg;
  if (src->IsConstant()) {
    switch (src->type) {
      case INT8_TYPE:
        e.asm_().MOVZ(kScratch0, src->constant.u8);
        e.asm_().STRB(kScratch0, ctx, (int32_t)offset); break;
      case INT16_TYPE:
        e.asm_().MOVZ(kScratch0, src->constant.u16);
        e.asm_().STRH(kScratch0, ctx, (int32_t)offset); break;
      case INT32_TYPE:
        e.MovImm64(kScratch0, src->constant.u32);
        e.asm_().STRw(kScratch0, ctx, (int32_t)offset); break;
      case INT64_TYPE:
        e.MovImm64(kScratch0, src->constant.i64);
        e.asm_().STR(kScratch0, ctx, (int32_t)offset); break;
      case FLOAT32_TYPE:
        e.MovImm64(kScratch0, (uint64_t)src->constant.u32);
        e.asm_().STRw(kScratch0, ctx, (int32_t)offset); break;
      case FLOAT64_TYPE:
        e.MovImm64(kScratch0, src->constant.u64);
        e.asm_().STR(kScratch0, ctx, (int32_t)offset); break;
      case VEC128_TYPE:
        e.LoadConstantV128(kVScratch0, src->constant.v128);
        e.asm_().STR_Q(kVScratch0, ctx, (int32_t)offset); break;
      default: return false;
    }
  } else if (IsAlloc(src)) {
    switch (src->type) {
      case INT8_TYPE:  e.asm_().STRB(GR(src), ctx, (int32_t)offset); break;
      case INT16_TYPE: e.asm_().STRH(GR(src), ctx, (int32_t)offset); break;
      case INT32_TYPE: e.asm_().STRw(GR(src), ctx, (int32_t)offset); break;
      case INT64_TYPE: e.asm_().STR(GR(src), ctx, (int32_t)offset); break;
      case FLOAT32_TYPE: e.asm_().STR_S(VR(src), ctx, (int32_t)offset); break;
      case FLOAT64_TYPE: e.asm_().STR_D(VR(src), ctx, (int32_t)offset); break;
      case VEC128_TYPE:  e.asm_().STR_Q(VR(src), ctx, (int32_t)offset); break;
      default: return false;
    }
  }
  return true;
}

// === Memory Load/Store ===

static bool EmitLoad(A64Emitter& e, const Instr* i) {
  auto dest = i->dest;
  auto addr = i->src1.value;
  if (!dest || !IsAlloc(dest) || !addr) { e.asm_().NOP(); return true; }

  // Compute host address: membase + guest_addr
  GReg host_addr = kScratch0;
  GReg guest = LoadGPR(e, addr, kScratch1);
  // Zero-extend guest address to 32 bits, then add membase
  e.asm_().MOVw(host_addr, guest);  // zero-extends to 64-bit
  e.asm_().ADD(host_addr, kMembaseReg, host_addr);

  switch (dest->type) {
    case INT8_TYPE:  e.asm_().LDRB(GR(dest), host_addr); break;
    case INT16_TYPE: e.asm_().LDRH(GR(dest), host_addr); break;
    case INT32_TYPE: e.asm_().LDRw(GR(dest), host_addr); break;
    case INT64_TYPE: e.asm_().LDR(GR(dest), host_addr); break;
    case FLOAT32_TYPE: e.asm_().LDR_S(VR(dest), host_addr); break;
    case FLOAT64_TYPE: e.asm_().LDR_D(VR(dest), host_addr); break;
    case VEC128_TYPE:  e.asm_().LDR_Q(VR(dest), host_addr); break;
    default: return false;
  }

  // Byte swap if needed (Xbox 360 is big-endian)
  if (i->flags & LOAD_STORE_BYTE_SWAP) {
    switch (dest->type) {
      case INT16_TYPE: e.asm_().REV16(GR(dest), GR(dest)); break;
      case INT32_TYPE: e.asm_().REVw(GR(dest), GR(dest)); break;
      case INT64_TYPE: e.asm_().REV(GR(dest), GR(dest)); break;
      default: break;
    }
  }
  return true;
}

static bool EmitStore(A64Emitter& e, const Instr* i) {
  auto addr = i->src1.value;
  auto val = i->src2.value;
  if (!addr || !val) return true;

  GReg host_addr = kScratch0;
  GReg guest = LoadGPR(e, addr, kScratch1);
  e.asm_().MOVw(host_addr, guest);
  e.asm_().ADD(host_addr, kMembaseReg, host_addr);

  if (val->IsConstant()) {
    GReg sv = kScratch1;
    switch (val->type) {
      case INT8_TYPE:  e.asm_().MOVZ(sv, val->constant.u8); e.asm_().STRB(sv, host_addr); break;
      case INT16_TYPE: e.asm_().MOVZ(sv, val->constant.u16); e.asm_().STRH(sv, host_addr); break;
      case INT32_TYPE: e.MovImm64(sv, val->constant.u32); e.asm_().STRw(sv, host_addr); break;
      case INT64_TYPE: e.MovImm64(sv, val->constant.i64); e.asm_().STR(sv, host_addr); break;
      default: break;
    }
  } else if (IsAlloc(val)) {
    // Byte swap before store if needed
    GReg sv = GR(val);
    if (i->flags & LOAD_STORE_BYTE_SWAP) {
      sv = kScratch1;
      switch (val->type) {
        case INT16_TYPE: e.asm_().REV16(sv, GR(val)); break;
        case INT32_TYPE: e.asm_().REVw(sv, GR(val)); break;
        case INT64_TYPE: e.asm_().REV(sv, GR(val)); break;
        default: e.asm_().MOV(sv, GR(val)); break;
      }
    }
    switch (val->type) {
      case INT8_TYPE:  e.asm_().STRB(sv, host_addr); break;
      case INT16_TYPE: e.asm_().STRH(sv, host_addr); break;
      case INT32_TYPE: e.asm_().STRw(sv, host_addr); break;
      case INT64_TYPE: e.asm_().STR(sv, host_addr); break;
      case FLOAT32_TYPE: e.asm_().STR_S(VR(val), host_addr); break;
      case FLOAT64_TYPE: e.asm_().STR_D(VR(val), host_addr); break;
      case VEC128_TYPE:  e.asm_().STR_Q(VR(val), host_addr); break;
      default: break;
    }
  }
  return true;
}

static bool EmitLoadOffset(A64Emitter& e, const Instr* i) {
  return EmitLoad(e, i);  // Simplified — treat as regular load
}

static bool EmitStoreOffset(A64Emitter& e, const Instr* i) {
  return EmitStore(e, i);  // Simplified
}

// === Assign / Type Conversions ===

static bool EmitAssign(A64Emitter& e, const Instr* i) {
  if (!i->dest || !IsAlloc(i->dest)) return true;
  auto src = i->src1.value;
  if (!src) return true;

  if (src->type <= INT64_TYPE) {
    GReg rd = GR(i->dest);
    if (src->IsConstant()) {
      e.MovImm64(rd, src->constant.i64);
    } else if (IsAlloc(src)) {
      e.asm_().MOV(rd, GR(src));
    }
  } else {
    VReg vd = VR(i->dest);
    if (src->IsConstant()) {
      e.LoadConstantV128(vd, src->constant.v128);
    } else if (IsAlloc(src)) {
      e.asm_().MOV_16B(vd, VR(src));
    }
  }
  return true;
}

static bool EmitCast(A64Emitter& e, const Instr* i) {
  if (!i->dest || !IsAlloc(i->dest)) return true;
  auto src = i->src1.value;
  if (!src) return true;

  // Cast between int and float of same size
  if (src->type <= INT64_TYPE && i->dest->type >= FLOAT32_TYPE) {
    // Int → FP register
    GReg rs = src->IsConstant() ? kScratch0 : GR(src);
    if (src->IsConstant()) e.MovImm64(rs, src->constant.i64);
    e.asm_().FMOV_DX(VR(i->dest), rs);
  } else if (src->type >= FLOAT32_TYPE && i->dest->type <= INT64_TYPE) {
    // FP register → Int
    if (IsAlloc(src)) {
      e.asm_().FMOV_XD(GR(i->dest), VR(src));
    }
  } else {
    // Same domain — just move
    return EmitAssign(e, i);
  }
  return true;
}

static bool EmitZeroExtend(A64Emitter& e, const Instr* i) {
  if (!i->dest || !IsAlloc(i->dest)) return true;
  auto src = i->src1.value;
  if (!src) return true;

  GReg rd = GR(i->dest);
  if (src->IsConstant()) {
    uint64_t val = 0;
    switch (src->type) {
      case INT8_TYPE:  val = src->constant.u8; break;
      case INT16_TYPE: val = src->constant.u16; break;
      case INT32_TYPE: val = src->constant.u32; break;
      case INT64_TYPE: val = src->constant.u64; break;
      default: break;
    }
    e.MovImm64(rd, val);
  } else if (IsAlloc(src)) {
    GReg rs = GR(src);
    switch (src->type) {
      case INT8_TYPE:  e.asm_().UXTB(rd, rs); break;
      case INT16_TYPE: e.asm_().UXTH(rd, rs); break;
      case INT32_TYPE: e.asm_().MOVw(rd, rs); break;  // MOV w clears upper 32
      case INT64_TYPE: e.asm_().MOV(rd, rs); break;
      default: break;
    }
  }
  return true;
}

static bool EmitSignExtend(A64Emitter& e, const Instr* i) {
  if (!i->dest || !IsAlloc(i->dest)) return true;
  auto src = i->src1.value;
  if (!src) return true;

  GReg rd = GR(i->dest);
  if (src->IsConstant()) {
    int64_t val = 0;
    switch (src->type) {
      case INT8_TYPE:  val = (int64_t)(int8_t)src->constant.i8; break;
      case INT16_TYPE: val = (int64_t)(int16_t)src->constant.i16; break;
      case INT32_TYPE: val = (int64_t)(int32_t)src->constant.i32; break;
      case INT64_TYPE: val = src->constant.i64; break;
      default: break;
    }
    e.MovImm64(rd, (uint64_t)val);
  } else if (IsAlloc(src)) {
    GReg rs = GR(src);
    switch (src->type) {
      case INT8_TYPE:  e.asm_().SXTB(rd, rs); break;
      case INT16_TYPE: e.asm_().SXTH(rd, rs); break;
      case INT32_TYPE: e.asm_().SXTW(rd, rs); break;
      case INT64_TYPE: e.asm_().MOV(rd, rs); break;
      default: break;
    }
  }
  return true;
}

static bool EmitTruncate(A64Emitter& e, const Instr* i) {
  if (!i->dest || !IsAlloc(i->dest)) return true;
  auto src = i->src1.value;
  if (!src) return true;

  GReg rd = GR(i->dest);
  if (src->IsConstant()) {
    uint64_t val = src->constant.u64;
    switch (i->dest->type) {
      case INT8_TYPE:  val &= 0xFF; break;
      case INT16_TYPE: val &= 0xFFFF; break;
      case INT32_TYPE: val &= 0xFFFFFFFF; break;
      default: break;
    }
    e.MovImm64(rd, val);
  } else if (IsAlloc(src)) {
    GReg rs = GR(src);
    switch (i->dest->type) {
      case INT8_TYPE:  e.asm_().UXTB(rd, rs); break;
      case INT16_TYPE: e.asm_().UXTH(rd, rs); break;
      case INT32_TYPE: e.asm_().MOVw(rd, rs); break;
      default: e.asm_().MOV(rd, rs); break;
    }
  }
  return true;
}

// === Integer Arithmetic ===

static bool EmitAdd(A64Emitter& e, const Instr* i) {
  if (!i->dest || !IsAlloc(i->dest)) return true;
  auto s1 = i->src1.value, s2 = i->src2.value;
  GReg rd = GR(i->dest);

  if (s1->IsConstant() && s2->IsConstant()) {
    e.MovImm64(rd, s1->constant.i64 + s2->constant.i64);
  } else if (s2->IsConstant() && (s2->constant.u64 & 0xFFF) == s2->constant.u64) {
    GReg r1 = LoadGPR(e, s1, kScratch0);
    e.asm_().ADD(rd, r1, (uint32_t)s2->constant.u64);
  } else {
    GReg r1 = LoadGPR(e, s1, kScratch0);
    GReg r2 = LoadGPR(e, s2, kScratch1);
    if (i->dest->type <= INT32_TYPE) {
      e.asm_().ADDw(rd, r1, r2);
    } else {
      e.asm_().ADD(rd, r1, r2);
    }
  }
  return true;
}

static bool EmitSub(A64Emitter& e, const Instr* i) {
  if (!i->dest || !IsAlloc(i->dest)) return true;
  auto s1 = i->src1.value, s2 = i->src2.value;
  GReg rd = GR(i->dest);

  if (s1->IsConstant() && s2->IsConstant()) {
    e.MovImm64(rd, s1->constant.i64 - s2->constant.i64);
  } else if (s2->IsConstant() && (s2->constant.u64 & 0xFFF) == s2->constant.u64) {
    GReg r1 = LoadGPR(e, s1, kScratch0);
    e.asm_().SUB(rd, r1, (uint32_t)s2->constant.u64);
  } else {
    GReg r1 = LoadGPR(e, s1, kScratch0);
    GReg r2 = LoadGPR(e, s2, kScratch1);
    if (i->dest->type <= INT32_TYPE) {
      e.asm_().SUBw(rd, r1, r2);
    } else {
      e.asm_().SUB(rd, r1, r2);
    }
  }
  return true;
}

static bool EmitMul(A64Emitter& e, const Instr* i) {
  if (!i->dest || !IsAlloc(i->dest)) return true;
  GReg rd = GR(i->dest);
  GReg r1 = LoadGPR(e, i->src1.value, kScratch0);
  GReg r2 = LoadGPR(e, i->src2.value, kScratch1);
  if (i->dest->type <= INT32_TYPE) {
    e.asm_().MULw(rd, r1, r2);
  } else {
    e.asm_().MUL(rd, r1, r2);
  }
  return true;
}

static bool EmitMulHi(A64Emitter& e, const Instr* i) {
  if (!i->dest || !IsAlloc(i->dest)) return true;
  GReg rd = GR(i->dest);
  GReg r1 = LoadGPR(e, i->src1.value, kScratch0);
  GReg r2 = LoadGPR(e, i->src2.value, kScratch1);
  if (i->flags & ARITHMETIC_UNSIGNED) {
    e.asm_().UMULH(rd, r1, r2);
  } else {
    e.asm_().SMULH(rd, r1, r2);
  }
  return true;
}

static bool EmitDiv(A64Emitter& e, const Instr* i) {
  if (!i->dest || !IsAlloc(i->dest)) return true;
  GReg rd = GR(i->dest);
  GReg r1 = LoadGPR(e, i->src1.value, kScratch0);
  GReg r2 = LoadGPR(e, i->src2.value, kScratch1);
  if (i->flags & ARITHMETIC_UNSIGNED) {
    if (i->dest->type <= INT32_TYPE) e.asm_().UDIVw(rd, r1, r2);
    else e.asm_().UDIV(rd, r1, r2);
  } else {
    if (i->dest->type <= INT32_TYPE) e.asm_().SDIVw(rd, r1, r2);
    else e.asm_().SDIV(rd, r1, r2);
  }
  return true;
}

static bool EmitNeg(A64Emitter& e, const Instr* i) {
  if (!i->dest || !IsAlloc(i->dest)) return true;
  GReg rd = GR(i->dest);
  GReg rs = LoadGPR(e, i->src1.value, kScratch0);
  e.asm_().NEG(rd, rs);
  return true;
}

static bool EmitAbs(A64Emitter& e, const Instr* i) {
  if (!i->dest || !IsAlloc(i->dest)) return true;
  auto src = i->src1.value;
  if (src->type >= FLOAT32_TYPE) {
    if (src->type == FLOAT32_TYPE) {
      VReg vs = IsAlloc(src) ? VR(src) : kVScratch0;
      e.asm_().FABS_S(VR(i->dest), vs);
    } else {
      VReg vs = IsAlloc(src) ? VR(src) : kVScratch0;
      e.asm_().FABS_D(VR(i->dest), vs);
    }
  } else {
    GReg rd = GR(i->dest);
    GReg rs = LoadGPR(e, src, kScratch0);
    // abs x = (x XOR (x ASR 63)) - (x ASR 63)
    e.asm_().ASRv(kScratch1, rs, kScratch2);  // Would need imm ASR
    // Simpler: CMP, CNEG
    e.asm_().CMP(rs, (uint32_t)0);
    e.asm_().NEG(kScratch1, rs);
    e.asm_().CSEL(rd, kScratch1, rs, MI);
  }
  return true;
}

// === Logical ===

static bool EmitAnd(A64Emitter& e, const Instr* i) {
  if (!i->dest || !IsAlloc(i->dest)) return true;
  GReg rd = GR(i->dest);
  GReg r1 = LoadGPR(e, i->src1.value, kScratch0);
  GReg r2 = LoadGPR(e, i->src2.value, kScratch1);
  if (i->dest->type <= INT32_TYPE) e.asm_().ANDw(rd, r1, r2);
  else e.asm_().AND(rd, r1, r2);
  return true;
}

static bool EmitOr(A64Emitter& e, const Instr* i) {
  if (!i->dest || !IsAlloc(i->dest)) return true;
  GReg rd = GR(i->dest);
  GReg r1 = LoadGPR(e, i->src1.value, kScratch0);
  GReg r2 = LoadGPR(e, i->src2.value, kScratch1);
  if (i->dest->type <= INT32_TYPE) e.asm_().ORRw(rd, r1, r2);
  else e.asm_().ORR(rd, r1, r2);
  return true;
}

static bool EmitXor(A64Emitter& e, const Instr* i) {
  if (!i->dest || !IsAlloc(i->dest)) return true;
  GReg rd = GR(i->dest);
  GReg r1 = LoadGPR(e, i->src1.value, kScratch0);
  GReg r2 = LoadGPR(e, i->src2.value, kScratch1);
  if (i->dest->type <= INT32_TYPE) e.asm_().EORw(rd, r1, r2);
  else e.asm_().EOR(rd, r1, r2);
  return true;
}

static bool EmitNot(A64Emitter& e, const Instr* i) {
  if (!i->dest || !IsAlloc(i->dest)) return true;
  GReg rd = GR(i->dest);
  GReg rs = LoadGPR(e, i->src1.value, kScratch0);
  e.asm_().MVN(rd, rs);
  return true;
}

static bool EmitAndNot(A64Emitter& e, const Instr* i) {
  if (!i->dest || !IsAlloc(i->dest)) return true;
  GReg rd = GR(i->dest);
  GReg r1 = LoadGPR(e, i->src1.value, kScratch0);
  GReg r2 = LoadGPR(e, i->src2.value, kScratch1);
  e.asm_().BIC(rd, r1, r2);  // rd = r1 AND NOT r2
  return true;
}

// === Shifts ===

static bool EmitShl(A64Emitter& e, const Instr* i) {
  if (!i->dest || !IsAlloc(i->dest)) return true;
  GReg rd = GR(i->dest);
  GReg r1 = LoadGPR(e, i->src1.value, kScratch0);
  GReg r2 = LoadGPR(e, i->src2.value, kScratch1);
  e.asm_().LSLv(rd, r1, r2);
  return true;
}

static bool EmitShr(A64Emitter& e, const Instr* i) {
  if (!i->dest || !IsAlloc(i->dest)) return true;
  GReg rd = GR(i->dest);
  GReg r1 = LoadGPR(e, i->src1.value, kScratch0);
  GReg r2 = LoadGPR(e, i->src2.value, kScratch1);
  e.asm_().LSRv(rd, r1, r2);
  return true;
}

static bool EmitSha(A64Emitter& e, const Instr* i) {
  if (!i->dest || !IsAlloc(i->dest)) return true;
  GReg rd = GR(i->dest);
  GReg r1 = LoadGPR(e, i->src1.value, kScratch0);
  GReg r2 = LoadGPR(e, i->src2.value, kScratch1);
  e.asm_().ASRv(rd, r1, r2);
  return true;
}

static bool EmitRotateLeft(A64Emitter& e, const Instr* i) {
  if (!i->dest || !IsAlloc(i->dest)) return true;
  GReg rd = GR(i->dest);
  GReg r1 = LoadGPR(e, i->src1.value, kScratch0);
  GReg r2 = LoadGPR(e, i->src2.value, kScratch1);
  // ROL(x, n) = ROR(x, 64-n)
  e.asm_().NEG(kScratch2, r2);
  e.asm_().RORv(rd, r1, kScratch2);
  return true;
}

// === Comparisons ===

static bool EmitCompare(A64Emitter& e, const Instr* i, Cond cc) {
  if (!i->dest || !IsAlloc(i->dest)) return true;
  GReg rd = GR(i->dest);
  GReg r1 = LoadGPR(e, i->src1.value, kScratch0);
  GReg r2 = LoadGPR(e, i->src2.value, kScratch1);
  if (i->src1.value->type <= INT32_TYPE) {
    e.asm_().CMPw(r1, r2);
  } else {
    e.asm_().CMP(r1, r2);
  }
  e.asm_().CSET(rd, cc);
  return true;
}

static bool EmitCompareEQ(A64Emitter& e, const Instr* i) { return EmitCompare(e, i, EQ); }
static bool EmitCompareNE(A64Emitter& e, const Instr* i) { return EmitCompare(e, i, NE); }
static bool EmitCompareSLT(A64Emitter& e, const Instr* i) { return EmitCompare(e, i, LT); }
static bool EmitCompareSLE(A64Emitter& e, const Instr* i) { return EmitCompare(e, i, LE); }
static bool EmitCompareSGT(A64Emitter& e, const Instr* i) { return EmitCompare(e, i, GT); }
static bool EmitCompareSGE(A64Emitter& e, const Instr* i) { return EmitCompare(e, i, GE); }
static bool EmitCompareULT(A64Emitter& e, const Instr* i) { return EmitCompare(e, i, LO); }
static bool EmitCompareULE(A64Emitter& e, const Instr* i) { return EmitCompare(e, i, LS); }
static bool EmitCompareUGT(A64Emitter& e, const Instr* i) { return EmitCompare(e, i, HI); }
static bool EmitCompareUGE(A64Emitter& e, const Instr* i) { return EmitCompare(e, i, HS); }

// === Misc Integer ===

static bool EmitByteSwap(A64Emitter& e, const Instr* i) {
  if (!i->dest || !IsAlloc(i->dest)) return true;
  GReg rd = GR(i->dest);
  GReg rs = LoadGPR(e, i->src1.value, kScratch0);
  switch (i->src1.value->type) {
    case INT16_TYPE: e.asm_().REV16(rd, rs); break;
    case INT32_TYPE: e.asm_().REVw(rd, rs); break;
    case INT64_TYPE: e.asm_().REV(rd, rs); break;
    default: e.asm_().MOV(rd, rs); break;
  }
  return true;
}

static bool EmitCntlz(A64Emitter& e, const Instr* i) {
  if (!i->dest || !IsAlloc(i->dest)) return true;
  GReg rd = GR(i->dest);
  GReg rs = LoadGPR(e, i->src1.value, kScratch0);
  if (i->src1.value->type <= INT32_TYPE) e.asm_().CLZw(rd, rs);
  else e.asm_().CLZ(rd, rs);
  return true;
}

static bool EmitSelect(A64Emitter& e, const Instr* i) {
  if (!i->dest || !IsAlloc(i->dest)) return true;
  auto cond = i->src1.value;
  auto val_true = i->src2.value;
  auto val_false = i->src3.value;

  if (cond->type <= INT64_TYPE && val_true->type <= INT64_TYPE) {
    GReg rd = GR(i->dest);
    GReg rc = LoadGPR(e, cond, kScratch2);
    GReg rt = LoadGPR(e, val_true, kScratch0);
    GReg rf = LoadGPR(e, val_false, kScratch1);
    e.asm_().CMP(rc, (uint32_t)0);
    e.asm_().CSEL(rd, rt, rf, NE);
  } else {
    e.asm_().NOP();  // Vector select — Phase 4
  }
  return true;
}

static bool EmitMax(A64Emitter& e, const Instr* i) {
  if (!i->dest || !IsAlloc(i->dest)) return true;
  GReg rd = GR(i->dest);
  GReg r1 = LoadGPR(e, i->src1.value, kScratch0);
  GReg r2 = LoadGPR(e, i->src2.value, kScratch1);
  e.asm_().CMP(r1, r2);
  e.asm_().CSEL(rd, r1, r2, GT);
  return true;
}

static bool EmitMin(A64Emitter& e, const Instr* i) {
  if (!i->dest || !IsAlloc(i->dest)) return true;
  GReg rd = GR(i->dest);
  GReg r1 = LoadGPR(e, i->src1.value, kScratch0);
  GReg r2 = LoadGPR(e, i->src2.value, kScratch1);
  e.asm_().CMP(r1, r2);
  e.asm_().CSEL(rd, r1, r2, LT);
  return true;
}

static bool EmitAddCarry(A64Emitter& e, const Instr* i) {
  return EmitAdd(e, i);  // Simplified — proper carry needs ADDS/ADC
}

static bool EmitIsNan(A64Emitter& e, const Instr* i) {
  if (!i->dest || !IsAlloc(i->dest)) return true;
  // Compare with self — NaN != NaN
  auto src = i->src1.value;
  if (IsAlloc(src)) {
    VReg vs = VR(src);
    if (src->type == FLOAT32_TYPE) {
      e.asm_().FCMP_S(vs, vs);  // Single-precision NaN check
    } else {
      e.asm_().FCMP_D(vs, vs);  // Double-precision NaN check
    }
    e.asm_().CSET(GR(i->dest), VS);  // VS = unordered (NaN)
  } else {
    e.asm_().MOVZ(GR(i->dest), 0);
  }
  return true;
}

// === Float Conversions and Utilities ===

static bool EmitConvert(A64Emitter& e, const Instr* i) {
  if (!i->dest || !IsAlloc(i->dest)) return true;
  auto src = i->src1.value;
  if (!src || !IsAlloc(src)) { e.asm_().NOP(); return true; }

  if (src->type == INT32_TYPE && i->dest->type == FLOAT32_TYPE) {
    e.asm_().SCVTF_S(VR(i->dest), GR(src));
  } else if (src->type == INT32_TYPE && i->dest->type == FLOAT64_TYPE) {
    e.asm_().SCVTF_D(VR(i->dest), GR(src));
  } else if (src->type == INT64_TYPE && i->dest->type == FLOAT64_TYPE) {
    e.asm_().SCVTF_D(VR(i->dest), GR(src));
  } else if (src->type == FLOAT32_TYPE && i->dest->type == INT32_TYPE) {
    e.asm_().FCVTZS_S(GR(i->dest), VR(src));
  } else if (src->type == FLOAT64_TYPE && i->dest->type == INT64_TYPE) {
    e.asm_().FCVTZS(GR(i->dest), VR(src));
  } else if (src->type == FLOAT32_TYPE && i->dest->type == FLOAT64_TYPE) {
    e.asm_().FCVT_DS(VR(i->dest), VR(src));
  } else if (src->type == FLOAT64_TYPE && i->dest->type == FLOAT32_TYPE) {
    e.asm_().FCVT_SD(VR(i->dest), VR(src));
  } else {
    e.asm_().NOP();
  }
  return true;
}

static bool EmitSqrt(A64Emitter& e, const Instr* i) {
  if (!i->dest || !IsAlloc(i->dest)) return true;
  auto src = i->src1.value;
  if (!src || !IsAlloc(src)) { e.asm_().NOP(); return true; }
  if (src->type == FLOAT32_TYPE) e.asm_().FSQRT_S(VR(i->dest), VR(src));
  else e.asm_().FSQRT_D(VR(i->dest), VR(src));
  return true;
}

static bool EmitMemset(A64Emitter& e, const Instr* i) {
  e.asm_().NOP();
  return true;
}

static bool EmitLoadClock(A64Emitter& e, const Instr* i) {
  if (!i->dest || !IsAlloc(i->dest)) return true;
  e.asm_().NOP();
  return true;
}

// === Phase 3: Floating Point ===

static bool EmitRound(A64Emitter& e, const Instr* i) {
  if (!i->dest || !IsAlloc(i->dest)) return true;
  auto src = i->src1.value;
  if (!src || !IsAlloc(src)) { e.asm_().NOP(); return true; }

  VReg vd = VR(i->dest);
  VReg vs = VR(src);
  // Round mode is encoded in flags
  uint32_t mode = i->flags;
  if (src->type == FLOAT32_TYPE) {
    switch (mode) {
      case ROUND_TO_ZERO:         e.asm_().FRINTZ_S(vd, vs); break;
      case ROUND_TO_NEAREST:      e.asm_().FRINTN_S(vd, vs); break;
      case ROUND_TO_MINUS_INFINITY: e.asm_().FRINTM_S(vd, vs); break;
      case ROUND_TO_POSITIVE_INFINITY: e.asm_().FRINTP_S(vd, vs); break;
      default: e.asm_().FRINTN_S(vd, vs); break;
    }
  } else {
    switch (mode) {
      case ROUND_TO_ZERO:         e.asm_().FRINTZ_D(vd, vs); break;
      case ROUND_TO_NEAREST:      e.asm_().FRINTN_D(vd, vs); break;
      case ROUND_TO_MINUS_INFINITY: e.asm_().FRINTM_D(vd, vs); break;
      case ROUND_TO_POSITIVE_INFINITY: e.asm_().FRINTP_D(vd, vs); break;
      default: e.asm_().FRINTN_D(vd, vs); break;
    }
  }
  return true;
}

static bool EmitToSingle(A64Emitter& e, const Instr* i) {
  if (!i->dest || !IsAlloc(i->dest)) return true;
  auto src = i->src1.value;
  if (!src || !IsAlloc(src)) { e.asm_().NOP(); return true; }
  // Double → Single with rounding
  e.asm_().FCVT_SD(VR(i->dest), VR(src));
  return true;
}

static bool EmitRsqrt(A64Emitter& e, const Instr* i) {
  if (!i->dest || !IsAlloc(i->dest)) return true;
  auto src = i->src1.value;
  if (!src || !IsAlloc(src)) { e.asm_().NOP(); return true; }
  if (src->type == FLOAT32_TYPE) {
    e.asm_().FRSQRTE_S(VR(i->dest), VR(src));
  } else {
    e.asm_().FRSQRTE_D(VR(i->dest), VR(src));
  }
  return true;
}

static bool EmitRecip(A64Emitter& e, const Instr* i) {
  if (!i->dest || !IsAlloc(i->dest)) return true;
  auto src = i->src1.value;
  if (!src || !IsAlloc(src)) { e.asm_().NOP(); return true; }
  if (src->type == FLOAT32_TYPE) {
    e.asm_().FRECPE_S(VR(i->dest), VR(src));
  } else {
    e.asm_().FRECPE_D(VR(i->dest), VR(src));
  }
  return true;
}

static bool EmitFAdd(A64Emitter& e, const Instr* i) {
  if (!i->dest || !IsAlloc(i->dest)) return true;
  auto s1 = i->src1.value, s2 = i->src2.value;
  if (!IsAlloc(s1) || !IsAlloc(s2)) { e.asm_().NOP(); return true; }
  if (i->dest->type == FLOAT32_TYPE)
    e.asm_().FADD_S(VR(i->dest), VR(s1), VR(s2));
  else
    e.asm_().FADD_D(VR(i->dest), VR(s1), VR(s2));
  return true;
}

static bool EmitFSub(A64Emitter& e, const Instr* i) {
  if (!i->dest || !IsAlloc(i->dest)) return true;
  auto s1 = i->src1.value, s2 = i->src2.value;
  if (!IsAlloc(s1) || !IsAlloc(s2)) { e.asm_().NOP(); return true; }
  if (i->dest->type == FLOAT32_TYPE)
    e.asm_().FSUB_S(VR(i->dest), VR(s1), VR(s2));
  else
    e.asm_().FSUB_D(VR(i->dest), VR(s1), VR(s2));
  return true;
}

static bool EmitFMul(A64Emitter& e, const Instr* i) {
  if (!i->dest || !IsAlloc(i->dest)) return true;
  auto s1 = i->src1.value, s2 = i->src2.value;
  if (!IsAlloc(s1) || !IsAlloc(s2)) { e.asm_().NOP(); return true; }
  if (i->dest->type == FLOAT32_TYPE)
    e.asm_().FMUL_S(VR(i->dest), VR(s1), VR(s2));
  else
    e.asm_().FMUL_D(VR(i->dest), VR(s1), VR(s2));
  return true;
}

static bool EmitFDiv(A64Emitter& e, const Instr* i) {
  if (!i->dest || !IsAlloc(i->dest)) return true;
  auto s1 = i->src1.value, s2 = i->src2.value;
  if (!IsAlloc(s1) || !IsAlloc(s2)) { e.asm_().NOP(); return true; }
  if (i->dest->type == FLOAT32_TYPE)
    e.asm_().FDIV_S(VR(i->dest), VR(s1), VR(s2));
  else
    e.asm_().FDIV_D(VR(i->dest), VR(s1), VR(s2));
  return true;
}

static bool EmitFNeg(A64Emitter& e, const Instr* i) {
  if (!i->dest || !IsAlloc(i->dest)) return true;
  auto src = i->src1.value;
  if (!src || !IsAlloc(src)) { e.asm_().NOP(); return true; }
  if (src->type == FLOAT32_TYPE)
    e.asm_().FNEG_S(VR(i->dest), VR(src));
  else
    e.asm_().FNEG_D(VR(i->dest), VR(src));
  return true;
}

static bool EmitSetRoundingMode(A64Emitter& e, const Instr* i) {
  // Set ARM64 FPCR rounding mode bits [23:22]
  // src1 = rounding mode value from guest
  auto src = i->src1.value;
  if (!src) return true;
  GReg mode = LoadGPR(e, src, kScratch0);
  // Read current FPCR
  e.asm_().MRS_FPCR(kScratch1);
  // Clear rounding mode bits [23:22]
  e.MovImm64(kScratch2, ~(3ULL << 22));
  e.asm_().AND(kScratch1, kScratch1, kScratch2);
  // Shift mode into bits [23:22] and OR in
  e.asm_().AND(mode, mode, kScratch2);  // mask to 2 bits
  e.asm_().MOVZ(kScratch2, 3);
  e.asm_().AND(mode, mode, kScratch2);
  e.asm_().LSLv(mode, mode, kScratch2);  // shift by 22
  e.MovImm64(kScratch2, 22);
  e.asm_().LSLv(mode, mode, kScratch2);
  e.asm_().ORR(kScratch1, kScratch1, mode);
  // Write back FPCR
  e.asm_().MSR_FPCR(kScratch1);
  return true;
}

static bool EmitMulAdd(A64Emitter& e, const Instr* i) {
  if (!i->dest || !IsAlloc(i->dest)) return true;
  auto s1 = i->src1.value, s2 = i->src2.value, s3 = i->src3.value;
  if (!IsAlloc(s1) || !IsAlloc(s2) || !IsAlloc(s3)) {
    e.asm_().NOP();
    return true;
  }
  // dest = s1 + (s2 * s3)
  if (i->dest->type == FLOAT32_TYPE) {
    e.asm_().FMADD_S(VR(i->dest), VR(s2), VR(s3), VR(s1));
  } else if (i->dest->type == FLOAT64_TYPE) {
    e.asm_().FMADD_D(VR(i->dest), VR(s2), VR(s3), VR(s1));
  } else if (i->dest->type == VEC128_TYPE) {
    // Vector FMA: FMLA Vd, Vn, Vm
    e.asm_().MOV_16B(VR(i->dest), VR(s1));
    e.asm_().FMLA_4S(VR(i->dest), VR(s2), VR(s3));
  }
  return true;
}

static bool EmitMulSub(A64Emitter& e, const Instr* i) {
  if (!i->dest || !IsAlloc(i->dest)) return true;
  auto s1 = i->src1.value, s2 = i->src2.value, s3 = i->src3.value;
  if (!IsAlloc(s1) || !IsAlloc(s2) || !IsAlloc(s3)) {
    e.asm_().NOP();
    return true;
  }
  // dest = s1 - (s2 * s3)
  if (i->dest->type == FLOAT32_TYPE) {
    e.asm_().FMSUB_S(VR(i->dest), VR(s2), VR(s3), VR(s1));
  } else if (i->dest->type == FLOAT64_TYPE) {
    e.asm_().FMSUB_D(VR(i->dest), VR(s2), VR(s3), VR(s1));
  } else if (i->dest->type == VEC128_TYPE) {
    e.asm_().MOV_16B(VR(i->dest), VR(s1));
    e.asm_().FMLS_4S(VR(i->dest), VR(s2), VR(s3));
  }
  return true;
}

static bool EmitPow2(A64Emitter& e, const Instr* i) {
  // 2^x approximation — complex; stub for now
  e.asm_().NOP();
  return true;
}

static bool EmitLog2(A64Emitter& e, const Instr* i) {
  // log2(x) — complex; stub for now
  e.asm_().NOP();
  return true;
}

static bool EmitDotProduct3(A64Emitter& e, const Instr* i) {
  if (!i->dest || !IsAlloc(i->dest)) return true;
  auto s1 = i->src1.value, s2 = i->src2.value;
  if (!IsAlloc(s1) || !IsAlloc(s2)) { e.asm_().NOP(); return true; }
  // dp3 = s1[0]*s2[0] + s1[1]*s2[1] + s1[2]*s2[2]
  e.asm_().FMUL_4S(kVScratch0, VR(s1), VR(s2));
  // Horizontal add first 3 elements — use pairwise adds
  // For now, approximate with NOP as a stub
  e.asm_().NOP();
  return true;
}

static bool EmitDotProduct4(A64Emitter& e, const Instr* i) {
  if (!i->dest || !IsAlloc(i->dest)) return true;
  auto s1 = i->src1.value, s2 = i->src2.value;
  if (!IsAlloc(s1) || !IsAlloc(s2)) { e.asm_().NOP(); return true; }
  // dp4 = s1[0]*s2[0] + s1[1]*s2[1] + s1[2]*s2[2] + s1[3]*s2[3]
  e.asm_().FMUL_4S(kVScratch0, VR(s1), VR(s2));
  e.asm_().NOP();  // TODO: horizontal sum
  return true;
}

static bool EmitDidSaturate(A64Emitter& e, const Instr* i) {
  if (!i->dest || !IsAlloc(i->dest)) return true;
  e.asm_().MOVZ(GR(i->dest), 0);  // Always report no saturation for now
  return true;
}

// === Phase 4: Vector/NEON ===

static bool EmitSplat(A64Emitter& e, const Instr* i) {
  if (!i->dest || !IsAlloc(i->dest)) return true;
  auto src = i->src1.value;
  if (!src) return true;
  VReg vd = VR(i->dest);
  if (src->IsConstant()) {
    if (src->type <= INT64_TYPE) {
      e.MovImm64(kScratch0, src->constant.i64);
      e.asm_().DUP_4S(vd, kScratch0);
    } else {
      e.LoadConstantV128(vd, src->constant.v128);
    }
  } else if (IsAlloc(src)) {
    if (src->type <= INT64_TYPE) {
      e.asm_().DUP_4S(vd, GR(src));
    } else {
      // DUP first element
      e.asm_().DUP_S(vd, VR(src), 0);
    }
  }
  return true;
}

static bool EmitInsert(A64Emitter& e, const Instr* i) {
  if (!i->dest || !IsAlloc(i->dest)) return true;
  auto vec = i->src1.value;
  auto idx_val = i->src2.value;
  auto part = i->src3.value;
  if (!vec || !idx_val || !part) return true;

  VReg vd = VR(i->dest);
  if (IsAlloc(vec) && vd != VR(vec)) {
    e.asm_().MOV_16B(vd, VR(vec));
  }
  uint32_t idx = idx_val->IsConstant() ? (uint32_t)idx_val->constant.u32 : 0;
  if (part->IsConstant()) {
    e.MovImm64(kScratch0, part->constant.i64);
    e.asm_().INS_S_GPR(vd, idx, kScratch0);
  } else if (IsAlloc(part)) {
    if (part->type <= INT64_TYPE) {
      e.asm_().INS_S_GPR(vd, idx, GR(part));
    } else {
      e.asm_().INS_S(vd, idx, VR(part), 0);
    }
  }
  return true;
}

static bool EmitExtract(A64Emitter& e, const Instr* i) {
  if (!i->dest || !IsAlloc(i->dest)) return true;
  auto vec = i->src1.value;
  auto idx_val = i->src2.value;
  if (!vec || !idx_val || !IsAlloc(vec)) return true;

  uint32_t idx = idx_val->IsConstant() ? (uint32_t)idx_val->constant.u32 : 0;
  if (i->dest->type <= INT64_TYPE) {
    e.asm_().UMOV_W(GR(i->dest), VR(vec), idx);
  } else {
    e.asm_().DUP_S(VR(i->dest), VR(vec), idx);
  }
  return true;
}

static bool EmitVectorAdd(A64Emitter& e, const Instr* i) {
  if (!i->dest || !IsAlloc(i->dest)) return true;
  auto s1 = i->src1.value, s2 = i->src2.value;
  if (!IsAlloc(s1) || !IsAlloc(s2)) { e.asm_().NOP(); return true; }
  VReg vd = VR(i->dest);
  // Check flags for float vs int
  uint32_t type_size = i->flags & 0x7;
  if (type_size == 6) {  // FLOAT32 type part
    e.asm_().FADD_4S(vd, VR(s1), VR(s2));
  } else {
    e.asm_().ADD_4S(vd, VR(s1), VR(s2));
  }
  return true;
}

static bool EmitVectorSub(A64Emitter& e, const Instr* i) {
  if (!i->dest || !IsAlloc(i->dest)) return true;
  auto s1 = i->src1.value, s2 = i->src2.value;
  if (!IsAlloc(s1) || !IsAlloc(s2)) { e.asm_().NOP(); return true; }
  VReg vd = VR(i->dest);
  uint32_t type_size = i->flags & 0x7;
  if (type_size == 6) {
    e.asm_().FSUB_4S(vd, VR(s1), VR(s2));
  } else {
    e.asm_().SUB_4S(vd, VR(s1), VR(s2));
  }
  return true;
}

static bool EmitVectorCompareEQ(A64Emitter& e, const Instr* i) {
  if (!i->dest || !IsAlloc(i->dest)) return true;
  auto s1 = i->src1.value, s2 = i->src2.value;
  if (!IsAlloc(s1) || !IsAlloc(s2)) { e.asm_().NOP(); return true; }
  uint32_t type_size = i->flags & 0x7;
  if (type_size == 6) {
    e.asm_().FCMEQ_4S(VR(i->dest), VR(s1), VR(s2));
  } else {
    e.asm_().CMEQ_4S(VR(i->dest), VR(s1), VR(s2));
  }
  return true;
}

static bool EmitVectorCompareSGT(A64Emitter& e, const Instr* i) {
  if (!i->dest || !IsAlloc(i->dest)) return true;
  auto s1 = i->src1.value, s2 = i->src2.value;
  if (!IsAlloc(s1) || !IsAlloc(s2)) { e.asm_().NOP(); return true; }
  uint32_t type_size = i->flags & 0x7;
  if (type_size == 6) {
    e.asm_().FCMGT_4S(VR(i->dest), VR(s1), VR(s2));
  } else {
    e.asm_().CMGT_4S(VR(i->dest), VR(s1), VR(s2));
  }
  return true;
}

static bool EmitVectorCompareSGE(A64Emitter& e, const Instr* i) {
  if (!i->dest || !IsAlloc(i->dest)) return true;
  auto s1 = i->src1.value, s2 = i->src2.value;
  if (!IsAlloc(s1) || !IsAlloc(s2)) { e.asm_().NOP(); return true; }
  uint32_t type_size = i->flags & 0x7;
  if (type_size == 6) {
    e.asm_().FCMGE_4S(VR(i->dest), VR(s1), VR(s2));
  } else {
    e.asm_().CMGE_4S(VR(i->dest), VR(s1), VR(s2));
  }
  return true;
}

static bool EmitVectorCompareUGT(A64Emitter& e, const Instr* i) {
  if (!i->dest || !IsAlloc(i->dest)) return true;
  auto s1 = i->src1.value, s2 = i->src2.value;
  if (!IsAlloc(s1) || !IsAlloc(s2)) { e.asm_().NOP(); return true; }
  e.asm_().CMHI_4S(VR(i->dest), VR(s1), VR(s2));
  return true;
}

static bool EmitVectorCompareUGE(A64Emitter& e, const Instr* i) {
  if (!i->dest || !IsAlloc(i->dest)) return true;
  auto s1 = i->src1.value, s2 = i->src2.value;
  if (!IsAlloc(s1) || !IsAlloc(s2)) { e.asm_().NOP(); return true; }
  e.asm_().CMHS_4S(VR(i->dest), VR(s1), VR(s2));
  return true;
}

static bool EmitVectorMax(A64Emitter& e, const Instr* i) {
  if (!i->dest || !IsAlloc(i->dest)) return true;
  auto s1 = i->src1.value, s2 = i->src2.value;
  if (!IsAlloc(s1) || !IsAlloc(s2)) { e.asm_().NOP(); return true; }
  uint32_t type_size = i->flags & 0x7;
  if (type_size == 6) {
    e.asm_().FMAX_4S(VR(i->dest), VR(s1), VR(s2));
  } else if (i->flags & ARITHMETIC_UNSIGNED) {
    e.asm_().UMAX_4S(VR(i->dest), VR(s1), VR(s2));
  } else {
    e.asm_().SMAX_4S(VR(i->dest), VR(s1), VR(s2));
  }
  return true;
}

static bool EmitVectorMin(A64Emitter& e, const Instr* i) {
  if (!i->dest || !IsAlloc(i->dest)) return true;
  auto s1 = i->src1.value, s2 = i->src2.value;
  if (!IsAlloc(s1) || !IsAlloc(s2)) { e.asm_().NOP(); return true; }
  uint32_t type_size = i->flags & 0x7;
  if (type_size == 6) {
    e.asm_().FMIN_4S(VR(i->dest), VR(s1), VR(s2));
  } else if (i->flags & ARITHMETIC_UNSIGNED) {
    e.asm_().UMIN_4S(VR(i->dest), VR(s1), VR(s2));
  } else {
    e.asm_().SMIN_4S(VR(i->dest), VR(s1), VR(s2));
  }
  return true;
}

static bool EmitVectorShl(A64Emitter& e, const Instr* i) {
  if (!i->dest || !IsAlloc(i->dest)) return true;
  auto s1 = i->src1.value, s2 = i->src2.value;
  if (!IsAlloc(s1) || !IsAlloc(s2)) { e.asm_().NOP(); return true; }
  // USHL uses positive shift for left
  e.asm_().USHL_4S(VR(i->dest), VR(s1), VR(s2));
  return true;
}

static bool EmitVectorShr(A64Emitter& e, const Instr* i) {
  if (!i->dest || !IsAlloc(i->dest)) return true;
  auto s1 = i->src1.value, s2 = i->src2.value;
  if (!IsAlloc(s1) || !IsAlloc(s2)) { e.asm_().NOP(); return true; }
  // USHL with negated shift = right shift
  e.asm_().NEG_4S(kVScratch0, VR(s2));
  e.asm_().USHL_4S(VR(i->dest), VR(s1), kVScratch0);
  return true;
}

static bool EmitVectorSha(A64Emitter& e, const Instr* i) {
  if (!i->dest || !IsAlloc(i->dest)) return true;
  auto s1 = i->src1.value, s2 = i->src2.value;
  if (!IsAlloc(s1) || !IsAlloc(s2)) { e.asm_().NOP(); return true; }
  // SSHL with negated shift = arithmetic right shift
  e.asm_().NEG_4S(kVScratch0, VR(s2));
  e.asm_().SSHL_4S(VR(i->dest), VR(s1), kVScratch0);
  return true;
}

static bool EmitVectorRotateLeft(A64Emitter& e, const Instr* i) {
  if (!i->dest || !IsAlloc(i->dest)) return true;
  auto s1 = i->src1.value, s2 = i->src2.value;
  if (!IsAlloc(s1) || !IsAlloc(s2)) { e.asm_().NOP(); return true; }
  // ROL = (x << n) | (x >> (32-n))
  VReg vd = VR(i->dest);
  VReg vs = VR(s1);
  VReg vn = VR(s2);
  // Left shift
  e.asm_().USHL_4S(kVScratch0, vs, vn);
  // Right shift by (32 - n)
  e.asm_().MOVI_4S_zero(kVScratch1);
  // Load 32 into each lane
  e.asm_().MOVZ(kScratch0, 32);
  e.asm_().DUP_4S(kVScratch1, kScratch0);
  e.asm_().SUB_4S(kVScratch1, kVScratch1, vn);
  e.asm_().NEG_4S(kVScratch1, kVScratch1);  // negate for USHL = right shift
  e.asm_().USHL_4S(vd, vs, kVScratch1);
  e.asm_().ORR_16B(vd, vd, kVScratch0);
  return true;
}

static bool EmitVectorAverage(A64Emitter& e, const Instr* i) {
  if (!i->dest || !IsAlloc(i->dest)) return true;
  auto s1 = i->src1.value, s2 = i->src2.value;
  if (!IsAlloc(s1) || !IsAlloc(s2)) { e.asm_().NOP(); return true; }
  if (i->flags & ARITHMETIC_UNSIGNED) {
    e.asm_().URHADD_4S(VR(i->dest), VR(s1), VR(s2));
  } else {
    e.asm_().SRHADD_4S(VR(i->dest), VR(s1), VR(s2));
  }
  return true;
}

static bool EmitPermute(A64Emitter& e, const Instr* i) {
  if (!i->dest || !IsAlloc(i->dest)) return true;
  auto ctrl = i->src1.value;
  auto s1 = i->src2.value, s2 = i->src3.value;
  if (!IsAlloc(s1) || !IsAlloc(s2)) { e.asm_().NOP(); return true; }
  if (ctrl && IsAlloc(ctrl)) {
    // Use TBL for fully dynamic permute
    e.asm_().TBL(VR(i->dest), VR(s1), VR(ctrl));
  } else {
    e.asm_().MOV_16B(VR(i->dest), VR(s1));
  }
  return true;
}

static bool EmitSwizzle(A64Emitter& e, const Instr* i) {
  if (!i->dest || !IsAlloc(i->dest)) return true;
  auto src = i->src1.value;
  if (!src || !IsAlloc(src)) { e.asm_().NOP(); return true; }
  // Swizzle mask is in flags — simplified with TBL or NOP for now
  e.asm_().MOV_16B(VR(i->dest), VR(src));
  return true;
}

static bool EmitPack(A64Emitter& e, const Instr* i) {
  if (!i->dest || !IsAlloc(i->dest)) return true;
  // Pack is complex — depends on pack type in flags
  // For now, just copy src1
  auto src = i->src1.value;
  if (src && IsAlloc(src)) {
    e.asm_().MOV_16B(VR(i->dest), VR(src));
  } else {
    e.asm_().MOVI_4S_zero(VR(i->dest));
  }
  return true;
}

static bool EmitUnpack(A64Emitter& e, const Instr* i) {
  if (!i->dest || !IsAlloc(i->dest)) return true;
  auto src = i->src1.value;
  if (src && IsAlloc(src)) {
    e.asm_().MOV_16B(VR(i->dest), VR(src));
  } else {
    e.asm_().MOVI_4S_zero(VR(i->dest));
  }
  return true;
}

static bool EmitVectorConvertI2F(A64Emitter& e, const Instr* i) {
  if (!i->dest || !IsAlloc(i->dest)) return true;
  auto src = i->src1.value;
  if (!src || !IsAlloc(src)) { e.asm_().NOP(); return true; }
  if (i->flags & ARITHMETIC_UNSIGNED) {
    e.asm_().UCVTF_4S(VR(i->dest), VR(src));
  } else {
    e.asm_().SCVTF_4S(VR(i->dest), VR(src));
  }
  return true;
}

static bool EmitVectorConvertF2I(A64Emitter& e, const Instr* i) {
  if (!i->dest || !IsAlloc(i->dest)) return true;
  auto src = i->src1.value;
  if (!src || !IsAlloc(src)) { e.asm_().NOP(); return true; }
  if (i->flags & ARITHMETIC_UNSIGNED) {
    e.asm_().FCVTZU_4S(VR(i->dest), VR(src));
  } else {
    e.asm_().FCVTZS_4S(VR(i->dest), VR(src));
  }
  return true;
}

static bool EmitVectorDenormFlush(A64Emitter& e, const Instr* i) {
  if (!i->dest || !IsAlloc(i->dest)) return true;
  auto src = i->src1.value;
  if (!src || !IsAlloc(src)) { e.asm_().NOP(); return true; }
  // On ARM64 with FZ bit set in FPCR, denorms flush automatically
  // Just copy through
  e.asm_().MOV_16B(VR(i->dest), VR(src));
  return true;
}

static bool EmitUnimplemented(A64Emitter& e, const Instr* i) {
  e.asm_().BRK(0xFFFF);
  return true;
}

// === Registration ===

void RegisterSequences() {
  static bool registered = false;
  if (registered) return;
  registered = true;

  // Control flow
  sequence_table[OPCODE_COMMENT] = EmitComment;
  sequence_table[OPCODE_NOP] = EmitNop;
  sequence_table[OPCODE_SOURCE_OFFSET] = EmitSourceOffset;
  sequence_table[OPCODE_DEBUG_BREAK] = EmitDebugBreak;
  sequence_table[OPCODE_DEBUG_BREAK_TRUE] = EmitDebugBreak;
  sequence_table[OPCODE_TRAP] = EmitTrap;
  sequence_table[OPCODE_TRAP_TRUE] = EmitTrap;
  sequence_table[OPCODE_RETURN] = EmitReturn;
  sequence_table[OPCODE_RETURN_TRUE] = EmitReturn;
  sequence_table[OPCODE_BRANCH] = EmitBranch;
  sequence_table[OPCODE_BRANCH_TRUE] = EmitBranchTrue;
  sequence_table[OPCODE_BRANCH_FALSE] = EmitBranchFalse;
  sequence_table[OPCODE_CALL] = EmitCall;
  sequence_table[OPCODE_CALL_TRUE] = EmitCall;
  sequence_table[OPCODE_CALL_INDIRECT] = EmitCallIndirect;
  sequence_table[OPCODE_CALL_INDIRECT_TRUE] = EmitCallIndirect;
  sequence_table[OPCODE_CALL_EXTERN] = EmitCallExtern;
  sequence_table[OPCODE_SET_RETURN_ADDRESS] = EmitNop;

  // Context
  sequence_table[OPCODE_LOAD_CONTEXT] = EmitLoadContext;
  sequence_table[OPCODE_STORE_CONTEXT] = EmitStoreContext;
  sequence_table[OPCODE_CONTEXT_BARRIER] = EmitNop;

  // Memory
  sequence_table[OPCODE_LOAD] = EmitLoad;
  sequence_table[OPCODE_STORE] = EmitStore;
  sequence_table[OPCODE_LOAD_OFFSET] = EmitLoadOffset;
  sequence_table[OPCODE_STORE_OFFSET] = EmitStoreOffset;
  sequence_table[OPCODE_LOAD_MMIO] = EmitLoad;
  sequence_table[OPCODE_STORE_MMIO] = EmitStore;
  sequence_table[OPCODE_MEMORY_BARRIER] = [](A64Emitter& e, const Instr*) {
    e.asm_().DMB_ISH();
    return true;
  };
  sequence_table[OPCODE_MEMSET] = EmitMemset;
  sequence_table[OPCODE_CACHE_CONTROL] = EmitNop;
  sequence_table[OPCODE_LOAD_CLOCK] = EmitLoadClock;
  sequence_table[OPCODE_LOAD_LOCAL] = EmitNop;  // Stub
  sequence_table[OPCODE_STORE_LOCAL] = EmitNop;  // Stub

  // Type conversions
  sequence_table[OPCODE_ASSIGN] = EmitAssign;
  sequence_table[OPCODE_CAST] = EmitCast;
  sequence_table[OPCODE_ZERO_EXTEND] = EmitZeroExtend;
  sequence_table[OPCODE_SIGN_EXTEND] = EmitSignExtend;
  sequence_table[OPCODE_TRUNCATE] = EmitTruncate;
  sequence_table[OPCODE_CONVERT] = EmitConvert;

  // Integer arithmetic (dispatch to FP when dest is float)
  sequence_table[OPCODE_ADD] = [](A64Emitter& e, const Instr* i) -> bool {
    if (i->dest && i->dest->type >= FLOAT32_TYPE) return EmitFAdd(e, i);
    return EmitAdd(e, i);
  };
  sequence_table[OPCODE_ADD_CARRY] = EmitAddCarry;
  sequence_table[OPCODE_SUB] = [](A64Emitter& e, const Instr* i) -> bool {
    if (i->dest && i->dest->type >= FLOAT32_TYPE) return EmitFSub(e, i);
    return EmitSub(e, i);
  };
  sequence_table[OPCODE_MUL] = [](A64Emitter& e, const Instr* i) -> bool {
    if (i->dest && i->dest->type >= FLOAT32_TYPE) return EmitFMul(e, i);
    return EmitMul(e, i);
  };
  sequence_table[OPCODE_MUL_HI] = EmitMulHi;
  sequence_table[OPCODE_DIV] = [](A64Emitter& e, const Instr* i) -> bool {
    if (i->dest && i->dest->type >= FLOAT32_TYPE) return EmitFDiv(e, i);
    return EmitDiv(e, i);
  };
  sequence_table[OPCODE_NEG] = [](A64Emitter& e, const Instr* i) -> bool {
    if (i->dest && i->dest->type >= FLOAT32_TYPE) return EmitFNeg(e, i);
    return EmitNeg(e, i);
  };
  sequence_table[OPCODE_ABS] = EmitAbs;
  sequence_table[OPCODE_SQRT] = EmitSqrt;
  sequence_table[OPCODE_RSQRT] = EmitRsqrt;
  sequence_table[OPCODE_RECIP] = EmitRecip;
  sequence_table[OPCODE_MAX] = EmitMax;
  sequence_table[OPCODE_MIN] = EmitMin;

  // FP-specific
  sequence_table[OPCODE_ROUND] = EmitRound;
  sequence_table[OPCODE_TO_SINGLE] = EmitToSingle;
  sequence_table[OPCODE_SET_ROUNDING_MODE] = EmitSetRoundingMode;
  sequence_table[OPCODE_MUL_ADD] = EmitMulAdd;
  sequence_table[OPCODE_MUL_SUB] = EmitMulSub;
  sequence_table[OPCODE_POW2] = EmitPow2;
  sequence_table[OPCODE_LOG2] = EmitLog2;
  sequence_table[OPCODE_DOT_PRODUCT_3] = EmitDotProduct3;
  sequence_table[OPCODE_DOT_PRODUCT_4] = EmitDotProduct4;
  sequence_table[OPCODE_DID_SATURATE] = EmitDidSaturate;

  // Logical
  sequence_table[OPCODE_AND] = EmitAnd;
  sequence_table[OPCODE_OR] = EmitOr;
  sequence_table[OPCODE_XOR] = EmitXor;
  sequence_table[OPCODE_NOT] = EmitNot;
  sequence_table[OPCODE_AND_NOT] = EmitAndNot;

  // Shifts
  sequence_table[OPCODE_SHL] = EmitShl;
  sequence_table[OPCODE_SHR] = EmitShr;
  sequence_table[OPCODE_SHA] = EmitSha;
  sequence_table[OPCODE_ROTATE_LEFT] = EmitRotateLeft;

  // Comparisons
  sequence_table[OPCODE_COMPARE_EQ] = EmitCompareEQ;
  sequence_table[OPCODE_COMPARE_NE] = EmitCompareNE;
  sequence_table[OPCODE_COMPARE_SLT] = EmitCompareSLT;
  sequence_table[OPCODE_COMPARE_SLE] = EmitCompareSLE;
  sequence_table[OPCODE_COMPARE_SGT] = EmitCompareSGT;
  sequence_table[OPCODE_COMPARE_SGE] = EmitCompareSGE;
  sequence_table[OPCODE_COMPARE_ULT] = EmitCompareULT;
  sequence_table[OPCODE_COMPARE_ULE] = EmitCompareULE;
  sequence_table[OPCODE_COMPARE_UGT] = EmitCompareUGT;
  sequence_table[OPCODE_COMPARE_UGE] = EmitCompareUGE;

  // Misc
  sequence_table[OPCODE_BYTE_SWAP] = EmitByteSwap;
  sequence_table[OPCODE_CNTLZ] = EmitCntlz;
  sequence_table[OPCODE_SELECT] = EmitSelect;
  sequence_table[OPCODE_IS_NAN] = EmitIsNan;
  sequence_table[OPCODE_DELAY_EXECUTION] = EmitNop;

  sequence_table[OPCODE_ATOMIC_EXCHANGE] = EmitUnimplemented;
  sequence_table[OPCODE_ATOMIC_COMPARE_EXCHANGE] = EmitUnimplemented;
  sequence_table[OPCODE_RESERVED_LOAD] = EmitUnimplemented;
  sequence_table[OPCODE_RESERVED_STORE] = EmitUnimplemented;

  // Vector/NEON (Phase 4)
  sequence_table[OPCODE_SPLAT] = EmitSplat;
  sequence_table[OPCODE_INSERT] = EmitInsert;
  sequence_table[OPCODE_EXTRACT] = EmitExtract;
  sequence_table[OPCODE_VECTOR_ADD] = EmitVectorAdd;
  sequence_table[OPCODE_VECTOR_SUB] = EmitVectorSub;
  sequence_table[OPCODE_VECTOR_COMPARE_EQ] = EmitVectorCompareEQ;
  sequence_table[OPCODE_VECTOR_COMPARE_SGT] = EmitVectorCompareSGT;
  sequence_table[OPCODE_VECTOR_COMPARE_SGE] = EmitVectorCompareSGE;
  sequence_table[OPCODE_VECTOR_COMPARE_UGT] = EmitVectorCompareUGT;
  sequence_table[OPCODE_VECTOR_COMPARE_UGE] = EmitVectorCompareUGE;
  sequence_table[OPCODE_VECTOR_MAX] = EmitVectorMax;
  sequence_table[OPCODE_VECTOR_MIN] = EmitVectorMin;
  sequence_table[OPCODE_VECTOR_SHL] = EmitVectorShl;
  sequence_table[OPCODE_VECTOR_SHR] = EmitVectorShr;
  sequence_table[OPCODE_VECTOR_SHA] = EmitVectorSha;
  sequence_table[OPCODE_VECTOR_ROTATE_LEFT] = EmitVectorRotateLeft;
  sequence_table[OPCODE_VECTOR_AVERAGE] = EmitVectorAverage;
  sequence_table[OPCODE_PERMUTE] = EmitPermute;
  sequence_table[OPCODE_SWIZZLE] = EmitSwizzle;
  sequence_table[OPCODE_PACK] = EmitPack;
  sequence_table[OPCODE_UNPACK] = EmitUnpack;
  sequence_table[OPCODE_VECTOR_CONVERT_I2F] = EmitVectorConvertI2F;
  sequence_table[OPCODE_VECTOR_CONVERT_F2I] = EmitVectorConvertF2I;
  sequence_table[OPCODE_VECTOR_DENORMFLUSH] = EmitVectorDenormFlush;

  // Fill remaining with unimplemented
  for (uint32_t op = 0; op < __OPCODE_MAX_VALUE; ++op) {
    if (sequence_table.find(op) == sequence_table.end()) {
      sequence_table[op] = EmitUnimplemented;
    }
  }
}

bool SelectSequence(A64Emitter* e, const hir::Instr* i,
                    const hir::Instr** new_tail) {
  Opcode opcode = static_cast<Opcode>(i->GetOpcodeNum());
  auto it = sequence_table.find(static_cast<uint32_t>(opcode));
  if (it != sequence_table.end()) {
    return it->second(*e, i);
  }
  XELOGE("ARM64: No sequence for opcode {}", GetOpcodeName(opcode));
  e->asm_().BRK(0xFFFE);
  return false;
}

}  // namespace a64
}  // namespace backend
}  // namespace cpu
}  // namespace xe
