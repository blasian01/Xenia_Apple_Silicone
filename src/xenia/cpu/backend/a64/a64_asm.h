/**
 * ARM64 Instruction Assembler for Xenia JIT Backend
 * Encodes ARM64 instructions as 32-bit words.
 */
#ifndef XENIA_CPU_BACKEND_A64_A64_ASM_H_
#define XENIA_CPU_BACKEND_A64_A64_ASM_H_

#include <cassert>
#include <cstdint>
#include <cstring>
#include <vector>

namespace xe {
namespace cpu {
namespace backend {
namespace a64 {

// ARM64 general-purpose registers
enum GReg : uint32_t {
  X0 = 0, X1, X2, X3, X4, X5, X6, X7,
  X8, X9, X10, X11, X12, X13, X14, X15,
  X16, X17, X18, X19, X20, X21, X22, X23,
  X24, X25, X26, X27, X28, X29, X30, XZR = 31, SP = 31,
};

// ARM64 NEON/FP registers
enum VReg : uint32_t {
  V0 = 0, V1, V2, V3, V4, V5, V6, V7,
  V8, V9, V10, V11, V12, V13, V14, V15,
  V16, V17, V18, V19, V20, V21, V22, V23,
  V24, V25, V26, V27, V28, V29, V30, V31,
};

// Condition codes
enum Cond : uint32_t {
  EQ = 0, NE, CS, CC, MI, PL, VS, VC,
  HI, LS, GE, LT, GT, LE, AL, NV,
  HS = CS, LO = CC,
};

// Shift types
enum Shift : uint32_t { LSL = 0, LSR = 1, ASR = 2, ROR = 3 };

// Extend types
enum Extend : uint32_t {
  UXTB = 0, UXTH = 1, UXTW = 2, UXTX = 3,
  SXTB = 4, SXTH = 5, SXTW = 6, SXTX = 7,
};

// NEON arrangement specifiers
enum Arrangement : uint32_t {
  B8 = 0, B16 = 1, H4 = 2, H8 = 3,
  S2 = 4, S4 = 5, D1 = 6, D2 = 7,
};

// Label for forward/backward references
struct Label {
  int32_t offset = -1;  // Byte offset in code buffer, -1 if unbound
  struct Patch { uint32_t code_offset; uint8_t type; };
  std::vector<Patch> patches;

  bool bound() const { return offset >= 0; }
};

// ARM64 code emitter
class A64Asm {
 public:
  static constexpr size_t kDefaultCapacity = 1024 * 1024;  // 1MB

  A64Asm() { code_.reserve(kDefaultCapacity / 4); }

  void Reset() { code_.clear(); }
  const uint32_t* code() const { return code_.data(); }
  size_t code_size() const { return code_.size() * 4; }
  size_t offset() const { return code_.size() * 4; }

  // === Label management ===
  void Bind(Label* label) {
    assert(!label->bound());
    label->offset = static_cast<int32_t>(offset());
    for (auto& patch : label->patches) {
      PatchBranch(patch.code_offset, label->offset, patch.type);
    }
    label->patches.clear();
  }

  // === Data Processing (Immediate) ===

  // ADD/SUB immediate (64-bit)
  void ADD(GReg rd, GReg rn, uint32_t imm12, bool shift12 = false) {
    Emit(0x91000000 | (shift12 ? (1u << 22) : 0) |
         ((imm12 & 0xFFF) << 10) | (rn << 5) | rd);
  }
  void SUB(GReg rd, GReg rn, uint32_t imm12, bool shift12 = false) {
    Emit(0xD1000000 | (shift12 ? (1u << 22) : 0) |
         ((imm12 & 0xFFF) << 10) | (rn << 5) | rd);
  }
  void ADDS(GReg rd, GReg rn, uint32_t imm12) {
    Emit(0xB1000000 | ((imm12 & 0xFFF) << 10) | (rn << 5) | rd);
  }
  void SUBS(GReg rd, GReg rn, uint32_t imm12) {
    Emit(0xF1000000 | ((imm12 & 0xFFF) << 10) | (rn << 5) | rd);
  }
  // CMP = SUBS XZR
  void CMP(GReg rn, uint32_t imm12) { SUBS(XZR, rn, imm12); }

  // ADD/SUB shifted register (64-bit)
  void ADD(GReg rd, GReg rn, GReg rm, Shift sh = LSL, uint32_t amt = 0) {
    Emit(0x8B000000 | (sh << 22) | (rm << 16) | ((amt & 0x3F) << 10) |
         (rn << 5) | rd);
  }
  void SUB(GReg rd, GReg rn, GReg rm, Shift sh = LSL, uint32_t amt = 0) {
    Emit(0xCB000000 | (sh << 22) | (rm << 16) | ((amt & 0x3F) << 10) |
         (rn << 5) | rd);
  }
  void SUBS(GReg rd, GReg rn, GReg rm, Shift sh = LSL, uint32_t amt = 0) {
    Emit(0xEB000000 | (sh << 22) | (rm << 16) | ((amt & 0x3F) << 10) |
         (rn << 5) | rd);
  }
  void CMP(GReg rn, GReg rm) { SUBS(XZR, rn, rm); }
  void NEG(GReg rd, GReg rm) { SUB(rd, XZR, rm); }

  // 32-bit variants
  void ADDw(GReg rd, GReg rn, uint32_t imm12) {
    Emit(0x11000000 | ((imm12 & 0xFFF) << 10) | (rn << 5) | rd);
  }
  void SUBw(GReg rd, GReg rn, uint32_t imm12) {
    Emit(0x51000000 | ((imm12 & 0xFFF) << 10) | (rn << 5) | rd);
  }
  void ADDw(GReg rd, GReg rn, GReg rm, Shift sh = LSL, uint32_t amt = 0) {
    Emit(0x0B000000 | (sh << 22) | (rm << 16) | ((amt & 0x3F) << 10) |
         (rn << 5) | rd);
  }
  void SUBw(GReg rd, GReg rn, GReg rm, Shift sh = LSL, uint32_t amt = 0) {
    Emit(0x4B000000 | (sh << 22) | (rm << 16) | ((amt & 0x3F) << 10) |
         (rn << 5) | rd);
  }
  void SUBSw(GReg rd, GReg rn, GReg rm) {
    Emit(0x6B000000 | (rm << 16) | (rn << 5) | rd);
  }
  void CMPw(GReg rn, GReg rm) { SUBSw(XZR, rn, rm); }

  // Multiply
  void MUL(GReg rd, GReg rn, GReg rm) { MADD(rd, rn, rm, XZR); }
  void MADD(GReg rd, GReg rn, GReg rm, GReg ra) {
    Emit(0x9B000000 | (rm << 16) | (ra << 10) | (rn << 5) | rd);
  }
  void MSUB(GReg rd, GReg rn, GReg rm, GReg ra) {
    Emit(0x9B008000 | (rm << 16) | (ra << 10) | (rn << 5) | rd);
  }
  void SMULH(GReg rd, GReg rn, GReg rm) {
    Emit(0x9B407C00 | (rm << 16) | (rn << 5) | rd);
  }
  void UMULH(GReg rd, GReg rn, GReg rm) {
    Emit(0x9BC07C00 | (rm << 16) | (rn << 5) | rd);
  }
  void MULw(GReg rd, GReg rn, GReg rm) {
    Emit(0x1B007C00 | (rm << 16) | (rn << 5) | rd);
  }
  void SDIV(GReg rd, GReg rn, GReg rm) {
    Emit(0x9AC00C00 | (rm << 16) | (rn << 5) | rd);
  }
  void UDIV(GReg rd, GReg rn, GReg rm) {
    Emit(0x9AC00800 | (rm << 16) | (rn << 5) | rd);
  }
  void SDIVw(GReg rd, GReg rn, GReg rm) {
    Emit(0x1AC00C00 | (rm << 16) | (rn << 5) | rd);
  }
  void UDIVw(GReg rd, GReg rn, GReg rm) {
    Emit(0x1AC00800 | (rm << 16) | (rn << 5) | rd);
  }

  // Logical (shifted register, 64-bit)
  void AND(GReg rd, GReg rn, GReg rm, Shift sh = LSL, uint32_t amt = 0) {
    Emit(0x8A000000 | (sh << 22) | (rm << 16) | ((amt & 0x3F) << 10) |
         (rn << 5) | rd);
  }
  void ORR(GReg rd, GReg rn, GReg rm, Shift sh = LSL, uint32_t amt = 0) {
    Emit(0xAA000000 | (sh << 22) | (rm << 16) | ((amt & 0x3F) << 10) |
         (rn << 5) | rd);
  }
  void EOR(GReg rd, GReg rn, GReg rm, Shift sh = LSL, uint32_t amt = 0) {
    Emit(0xCA000000 | (sh << 22) | (rm << 16) | ((amt & 0x3F) << 10) |
         (rn << 5) | rd);
  }
  void ORN(GReg rd, GReg rn, GReg rm, Shift sh = LSL, uint32_t amt = 0) {
    Emit(0xAA200000 | (sh << 22) | (rm << 16) | ((amt & 0x3F) << 10) |
         (rn << 5) | rd);
  }
  void BIC(GReg rd, GReg rn, GReg rm) {
    Emit(0x8A200000 | (rm << 16) | (rn << 5) | rd);
  }
  void MVN(GReg rd, GReg rm) { ORN(rd, XZR, rm); }
  void MOV(GReg rd, GReg rm) { ORR(rd, XZR, rm); }

  // 32-bit logical
  void ANDw(GReg rd, GReg rn, GReg rm) {
    Emit(0x0A000000 | (rm << 16) | (rn << 5) | rd);
  }
  void ORRw(GReg rd, GReg rn, GReg rm) {
    Emit(0x2A000000 | (rm << 16) | (rn << 5) | rd);
  }
  void EORw(GReg rd, GReg rn, GReg rm) {
    Emit(0x4A000000 | (rm << 16) | (rn << 5) | rd);
  }
  void MOVw(GReg rd, GReg rm) { ORRw(rd, XZR, rm); }

  // Shifts
  void LSLv(GReg rd, GReg rn, GReg rm) {
    Emit(0x9AC02000 | (rm << 16) | (rn << 5) | rd);
  }
  void LSRv(GReg rd, GReg rn, GReg rm) {
    Emit(0x9AC02400 | (rm << 16) | (rn << 5) | rd);
  }
  void ASRv(GReg rd, GReg rn, GReg rm) {
    Emit(0x9AC02800 | (rm << 16) | (rn << 5) | rd);
  }
  void RORv(GReg rd, GReg rn, GReg rm) {
    Emit(0x9AC02C00 | (rm << 16) | (rn << 5) | rd);
  }

  // Bit manipulation
  void CLZ(GReg rd, GReg rn) { Emit(0xDAC01000 | (rn << 5) | rd); }
  void CLZw(GReg rd, GReg rn) { Emit(0x5AC01000 | (rn << 5) | rd); }
  void REV(GReg rd, GReg rn) { Emit(0xDAC00C00 | (rn << 5) | rd); }
  void REVw(GReg rd, GReg rn) { Emit(0x5AC00800 | (rn << 5) | rd); }
  void REV16(GReg rd, GReg rn) { Emit(0xDAC00400 | (rn << 5) | rd); }
  void REV32(GReg rd, GReg rn) { Emit(0xDAC00800 | (rn << 5) | rd); }
  void RBIT(GReg rd, GReg rn) { Emit(0xDAC00000 | (rn << 5) | rd); }

  // Immediate arithmetic (12-bit unsigned immediate)
  void ADD_imm(GReg rd, GReg rn, uint32_t imm12) {
    Emit(0x91000000 | ((imm12 & 0xFFF) << 10) | (rn << 5) | rd);
  }
  void SUB_imm(GReg rd, GReg rn, uint32_t imm12) {
    Emit(0xD1000000 | ((imm12 & 0xFFF) << 10) | (rn << 5) | rd);
  }
  void ADD_immw(GReg rd, GReg rn, uint32_t imm12) {
    Emit(0x11000000 | ((imm12 & 0xFFF) << 10) | (rn << 5) | rd);
  }

  // Immediate bitfield (LSL/LSR via UBFM/SBFM)
  void LSL_imm(GReg rd, GReg rn, uint32_t shift) {
    // LSL Xd, Xn, #shift = UBFM Xd, Xn, #(64-shift), #(63-shift)
    uint32_t immr = (64 - shift) & 0x3F;
    uint32_t imms = (63 - shift) & 0x3F;
    Emit(0xD3400000 | (immr << 16) | (imms << 10) | (rn << 5) | rd);
  }
  void LSR_imm(GReg rd, GReg rn, uint32_t shift) {
    // LSR Xd, Xn, #shift = UBFM Xd, Xn, #shift, #63
    Emit(0xD340FC00 | ((shift & 0x3F) << 16) | (rn << 5) | rd);
  }
  void LSR_immw(GReg rd, GReg rn, uint32_t shift) {
    // LSR Wd, Wn, #shift = UBFM Wd, Wn, #shift, #31
    Emit(0x53007C00 | ((shift & 0x1F) << 16) | (rn << 5) | rd);
  }

  // AND with immediate (logical immediate encoding)
  // Simplified: only common patterns (0xFF, 0xFFFF, etc.)
  void AND_imm(GReg rd, GReg rn, uint64_t imm) {
    // For simple masks, use MOV + AND register
    // This is a simplified version for common cases
    if (imm == 0xFF) {
      UXTB(rd, rn);  // Zero-extend byte
    } else if (imm == 0xFFFF) {
      UXTH(rd, rn);  // Zero-extend halfword
    } else if (imm == 0xFFFFFFFF) {
      MOVw(rd, rn);  // 32-bit move (zero-extends)
    } else {
      // General case: load immediate and AND
      // Use scratch approach for now — caller should handle this
      // Emit a NOP as fallback (should not be reached for common cases)
      NOP();
    }
  }

  // FMOV between GPR and FP (see also FMOV_WS/SW/XD/DX below)
  // FMOV_D: alias for FMOV_DX below

  // Exclusive access (see LDXR/STXR in System section below)
  // Data memory barrier (see DMB_ISH in System section below)
  void DMB_ISHST() { Emit(0xD50332BF); }  // DMB ISHST

  // Move wide
  void MOVZ(GReg rd, uint16_t imm, uint32_t shift = 0) {
    Emit(0xD2800000 | ((shift / 16) << 21) | (uint32_t(imm) << 5) | rd);
  }
  void MOVK(GReg rd, uint16_t imm, uint32_t shift = 0) {
    Emit(0xF2800000 | ((shift / 16) << 21) | (uint32_t(imm) << 5) | rd);
  }
  void MOVN(GReg rd, uint16_t imm, uint32_t shift = 0) {
    Emit(0x92800000 | ((shift / 16) << 21) | (uint32_t(imm) << 5) | rd);
  }

  // Load 64-bit immediate via MOVZ/MOVK sequence
  void MOV64(GReg rd, uint64_t imm) {
    if (imm == 0) { MOV(rd, XZR); return; }
    bool first = true;
    for (int i = 0; i < 4; i++) {
      uint16_t part = (imm >> (i * 16)) & 0xFFFF;
      if (part != 0 || (i == 0 && imm == part)) {
        if (first) { MOVZ(rd, part, i * 16); first = false; }
        else { MOVK(rd, part, i * 16); }
      }
    }
    if (first) MOVZ(rd, 0, 0);
  }

  // Conditional select
  void CSEL(GReg rd, GReg rn, GReg rm, Cond cc) {
    Emit(0x9A800000 | (rm << 16) | (cc << 12) | (rn << 5) | rd);
  }
  void CSET(GReg rd, Cond cc) {
    CSEL(rd, XZR, XZR, static_cast<Cond>(cc ^ 1));
    // Actually CSINC: rd = (cc) ? 1 : 0
    code_.back() = 0x9A9F0000 | ((cc ^ 1) << 12) | (XZR << 5) | rd |
                   (XZR << 16);
  }

  // Sign/Zero extend
  void SXTB(GReg rd, GReg rn) {
    Emit(0x93401C00 | (rn << 5) | rd);  // SBFM x, x, #0, #7
  }
  void SXTH(GReg rd, GReg rn) {
    Emit(0x93403C00 | (rn << 5) | rd);  // SBFM x, x, #0, #15
  }
  void SXTW(GReg rd, GReg rn) {
    Emit(0x93407C00 | (rn << 5) | rd);  // SBFM x, x, #0, #31
  }
  void UXTB(GReg rd, GReg rn) {
    Emit(0x53001C00 | (rn << 5) | rd);  // UBFM w, w, #0, #7
  }
  void UXTH(GReg rd, GReg rn) {
    Emit(0x53003C00 | (rn << 5) | rd);  // UBFM w, w, #0, #15
  }

  // === Branches ===
  void B(Label* label) { EmitBranch(0x14000000, label, 0); }
  void BL(Label* label) { EmitBranch(0x94000000, label, 0); }
  void B(Cond cc, Label* label) {
    EmitBranch(0x54000000 | cc, label, 1);
  }
  void CBZ(GReg rt, Label* label) {
    EmitBranch(0xB4000000 | rt, label, 2);
  }
  void CBNZ(GReg rt, Label* label) {
    EmitBranch(0xB5000000 | rt, label, 2);
  }
  void BR(GReg rn) { Emit(0xD61F0000 | (rn << 5)); }
  void BLR(GReg rn) { Emit(0xD63F0000 | (rn << 5)); }
  void RET(GReg rn = X30) { Emit(0xD65F0000 | (rn << 5)); }

  // === Load/Store ===

  // LDR/STR unsigned offset (64-bit)
  void LDR(GReg rt, GReg rn, int32_t offset = 0) {
    assert((offset & 7) == 0 && offset >= 0 && offset < 32768);
    Emit(0xF9400000 | (((offset / 8) & 0xFFF) << 10) | (rn << 5) | rt);
  }
  void STR(GReg rt, GReg rn, int32_t offset = 0) {
    assert((offset & 7) == 0 && offset >= 0 && offset < 32768);
    Emit(0xF9000000 | (((offset / 8) & 0xFFF) << 10) | (rn << 5) | rt);
  }
  // 32-bit load/store
  void LDRw(GReg rt, GReg rn, int32_t offset = 0) {
    assert((offset & 3) == 0 && offset >= 0);
    Emit(0xB9400000 | (((offset / 4) & 0xFFF) << 10) | (rn << 5) | rt);
  }
  void STRw(GReg rt, GReg rn, int32_t offset = 0) {
    assert((offset & 3) == 0 && offset >= 0);
    Emit(0xB9000000 | (((offset / 4) & 0xFFF) << 10) | (rn << 5) | rt);
  }
  // 16-bit / 8-bit loads
  void LDRH(GReg rt, GReg rn, int32_t offset = 0) {
    assert((offset & 1) == 0 && offset >= 0);
    Emit(0x79400000 | (((offset / 2) & 0xFFF) << 10) | (rn << 5) | rt);
  }
  void LDRB(GReg rt, GReg rn, int32_t offset = 0) {
    assert(offset >= 0);
    Emit(0x39400000 | ((offset & 0xFFF) << 10) | (rn << 5) | rt);
  }
  void STRH(GReg rt, GReg rn, int32_t offset = 0) {
    assert((offset & 1) == 0 && offset >= 0);
    Emit(0x79000000 | (((offset / 2) & 0xFFF) << 10) | (rn << 5) | rt);
  }
  void STRB(GReg rt, GReg rn, int32_t offset = 0) {
    assert(offset >= 0);
    Emit(0x39000000 | ((offset & 0xFFF) << 10) | (rn << 5) | rt);
  }
  // Sign-extending loads
  void LDRSW(GReg rt, GReg rn, int32_t offset = 0) {
    assert((offset & 3) == 0 && offset >= 0);
    Emit(0xB9800000 | (((offset / 4) & 0xFFF) << 10) | (rn << 5) | rt);
  }
  void LDRSH(GReg rt, GReg rn, int32_t offset = 0) {
    assert((offset & 1) == 0 && offset >= 0);
    Emit(0x79800000 | (((offset / 2) & 0xFFF) << 10) | (rn << 5) | rt);
  }
  void LDRSB(GReg rt, GReg rn, int32_t offset = 0) {
    assert(offset >= 0);
    Emit(0x39800000 | ((offset & 0xFFF) << 10) | (rn << 5) | rt);
  }

  // LDR/STR register offset
  void LDR(GReg rt, GReg rn, GReg rm, Extend ext = UXTX, uint32_t s = 0) {
    Emit(0xF8600800 | (rm << 16) | (ext << 13) | (s << 12) | (rn << 5) | rt);
  }
  void STR(GReg rt, GReg rn, GReg rm, Extend ext = UXTX, uint32_t s = 0) {
    Emit(0xF8200800 | (rm << 16) | (ext << 13) | (s << 12) | (rn << 5) | rt);
  }

  // LDP/STP (pair, 64-bit)
  void STP(GReg rt1, GReg rt2, GReg rn, int32_t offset) {
    assert((offset & 7) == 0 && offset >= -512 && offset < 512);
    Emit(0xA9000000 | ((((offset / 8)) & 0x7F) << 15) | (rt2 << 10) |
         (rn << 5) | rt1);
  }
  void LDP(GReg rt1, GReg rt2, GReg rn, int32_t offset) {
    assert((offset & 7) == 0 && offset >= -512 && offset < 512);
    Emit(0xA9400000 | ((((offset / 8)) & 0x7F) << 15) | (rt2 << 10) |
         (rn << 5) | rt1);
  }
  // STP/LDP pre-index
  void STP_pre(GReg rt1, GReg rt2, GReg rn, int32_t offset) {
    assert((offset & 7) == 0);
    Emit(0xA9800000 | ((((offset / 8)) & 0x7F) << 15) | (rt2 << 10) |
         (rn << 5) | rt1);
  }
  void LDP_post(GReg rt1, GReg rt2, GReg rn, int32_t offset) {
    assert((offset & 7) == 0);
    Emit(0xA8C00000 | ((((offset / 8)) & 0x7F) << 15) | (rt2 << 10) |
         (rn << 5) | rt1);
  }

  // NEON load/store (128-bit Q register)
  void LDR_Q(VReg vt, GReg rn, int32_t offset = 0) {
    assert((offset & 15) == 0 && offset >= 0);
    Emit(0x3DC00000 | (((offset / 16) & 0xFFF) << 10) | (rn << 5) | vt);
  }
  void STR_Q(VReg vt, GReg rn, int32_t offset = 0) {
    assert((offset & 15) == 0 && offset >= 0);
    Emit(0x3D800000 | (((offset / 16) & 0xFFF) << 10) | (rn << 5) | vt);
  }
  // NEON STP/LDP (128-bit pair)
  void STP_Q(VReg vt1, VReg vt2, GReg rn, int32_t offset) {
    assert((offset & 15) == 0);
    Emit(0xAD000000 | ((((offset / 16)) & 0x7F) << 15) | (vt2 << 10) |
         (rn << 5) | vt1);
  }
  void LDP_Q(VReg vt1, VReg vt2, GReg rn, int32_t offset) {
    assert((offset & 15) == 0);
    Emit(0xAD400000 | ((((offset / 16)) & 0x7F) << 15) | (vt2 << 10) |
         (rn << 5) | vt1);
  }
  // NEON 64-bit scalar load/store
  void LDR_D(VReg vt, GReg rn, int32_t offset = 0) {
    assert((offset & 7) == 0 && offset >= 0);
    Emit(0xFD400000 | (((offset / 8) & 0xFFF) << 10) | (rn << 5) | vt);
  }
  void STR_D(VReg vt, GReg rn, int32_t offset = 0) {
    assert((offset & 7) == 0 && offset >= 0);
    Emit(0xFD000000 | (((offset / 8) & 0xFFF) << 10) | (rn << 5) | vt);
  }
  // NEON 32-bit scalar load/store
  void LDR_S(VReg vt, GReg rn, int32_t offset = 0) {
    assert((offset & 3) == 0 && offset >= 0);
    Emit(0xBD400000 | (((offset / 4) & 0xFFF) << 10) | (rn << 5) | vt);
  }
  void STR_S(VReg vt, GReg rn, int32_t offset = 0) {
    assert((offset & 3) == 0 && offset >= 0);
    Emit(0xBD000000 | (((offset / 4) & 0xFFF) << 10) | (rn << 5) | vt);
  }

  // === Scalar FP ===
  void FADD_D(VReg vd, VReg vn, VReg vm) {
    Emit(0x1E602800 | (vm << 16) | (vn << 5) | vd);
  }
  void FSUB_D(VReg vd, VReg vn, VReg vm) {
    Emit(0x1E603800 | (vm << 16) | (vn << 5) | vd);
  }
  void FMUL_D(VReg vd, VReg vn, VReg vm) {
    Emit(0x1E600800 | (vm << 16) | (vn << 5) | vd);
  }
  void FDIV_D(VReg vd, VReg vn, VReg vm) {
    Emit(0x1E601800 | (vm << 16) | (vn << 5) | vd);
  }
  void FSQRT_D(VReg vd, VReg vn) {
    Emit(0x1E61C000 | (vn << 5) | vd);
  }
  void FABS_D(VReg vd, VReg vn) { Emit(0x1E60C000 | (vn << 5) | vd); }
  void FNEG_D(VReg vd, VReg vn) { Emit(0x1E614000 | (vn << 5) | vd); }
  void FMOV_D(VReg vd, VReg vn) { Emit(0x1E604000 | (vn << 5) | vd); }
  void FCMP_D(VReg vn, VReg vm) {
    Emit(0x1E602000 | (vm << 16) | (vn << 5));
  }

  // Single precision FP
  void FADD_S(VReg vd, VReg vn, VReg vm) {
    Emit(0x1E202800 | (vm << 16) | (vn << 5) | vd);
  }
  void FSUB_S(VReg vd, VReg vn, VReg vm) {
    Emit(0x1E203800 | (vm << 16) | (vn << 5) | vd);
  }
  void FMUL_S(VReg vd, VReg vn, VReg vm) {
    Emit(0x1E200800 | (vm << 16) | (vn << 5) | vd);
  }
  void FDIV_S(VReg vd, VReg vn, VReg vm) {
    Emit(0x1E201800 | (vm << 16) | (vn << 5) | vd);
  }
  void FSQRT_S(VReg vd, VReg vn) { Emit(0x1E21C000 | (vn << 5) | vd); }
  void FABS_S(VReg vd, VReg vn) { Emit(0x1E20C000 | (vn << 5) | vd); }
  void FNEG_S(VReg vd, VReg vn) { Emit(0x1E214000 | (vn << 5) | vd); }
  void FMOV_S(VReg vd, VReg vn) { Emit(0x1E204000 | (vn << 5) | vd); }
  void FCMP_S(VReg vn, VReg vm) {
    Emit(0x1E202000 | (vm << 16) | (vn << 5));
  }

  // Scalar FP reciprocal/rsqrt estimate
  void FRECPE_S(VReg vd, VReg vn) { Emit(0x5EA1D800 | (vn << 5) | vd); }
  void FRECPE_D(VReg vd, VReg vn) { Emit(0x5EE1D800 | (vn << 5) | vd); }
  void FRSQRTE_S(VReg vd, VReg vn) { Emit(0x7EA1D800 | (vn << 5) | vd); }
  void FRSQRTE_D(VReg vd, VReg vn) { Emit(0x7EE1D800 | (vn << 5) | vd); }

  // Fused multiply-add/sub: FMADD Vd = Va + (Vn * Vm)
  void FMADD_S(VReg vd, VReg vn, VReg vm, VReg va) {
    Emit(0x1F000000 | (vm << 16) | (va << 10) | (vn << 5) | vd);
  }
  void FMADD_D(VReg vd, VReg vn, VReg vm, VReg va) {
    Emit(0x1F400000 | (vm << 16) | (va << 10) | (vn << 5) | vd);
  }
  void FMSUB_S(VReg vd, VReg vn, VReg vm, VReg va) {
    Emit(0x1F008000 | (vm << 16) | (va << 10) | (vn << 5) | vd);
  }
  void FMSUB_D(VReg vd, VReg vn, VReg vm, VReg va) {
    Emit(0x1F408000 | (vm << 16) | (va << 10) | (vn << 5) | vd);
  }

  // FP rounding (scalar single)
  void FRINTN_S(VReg vd, VReg vn) { Emit(0x1E244000 | (vn << 5) | vd); }
  void FRINTP_S(VReg vd, VReg vn) { Emit(0x1E24C000 | (vn << 5) | vd); }
  void FRINTM_S(VReg vd, VReg vn) { Emit(0x1E254000 | (vn << 5) | vd); }
  void FRINTZ_S(VReg vd, VReg vn) { Emit(0x1E25C000 | (vn << 5) | vd); }
  void FRINTA_S(VReg vd, VReg vn) { Emit(0x1E264000 | (vn << 5) | vd); }

  // FP rounding (scalar double)
  void FRINTN_D(VReg vd, VReg vn) { Emit(0x1E644000 | (vn << 5) | vd); }
  void FRINTP_D(VReg vd, VReg vn) { Emit(0x1E64C000 | (vn << 5) | vd); }
  void FRINTM_D(VReg vd, VReg vn) { Emit(0x1E654000 | (vn << 5) | vd); }
  void FRINTZ_D(VReg vd, VReg vn) { Emit(0x1E65C000 | (vn << 5) | vd); }
  void FRINTA_D(VReg vd, VReg vn) { Emit(0x1E664000 | (vn << 5) | vd); }

  // FP conversions
  void FCVT_DS(VReg vd, VReg vn) { Emit(0x1E22C000 | (vn << 5) | vd); }
  void FCVT_SD(VReg vd, VReg vn) { Emit(0x1E624000 | (vn << 5) | vd); }
  // Int → FP
  void SCVTF_D(VReg vd, GReg rn) { Emit(0x9E620000 | (rn << 5) | vd); }
  void UCVTF_D(VReg vd, GReg rn) { Emit(0x9E630000 | (rn << 5) | vd); }
  void SCVTF_S(VReg vd, GReg rn) { Emit(0x1E220000 | (rn << 5) | vd); }
  void UCVTF_S(VReg vd, GReg rn) { Emit(0x1E230000 | (rn << 5) | vd); }
  void SCVTF_Sw(VReg vd, GReg rn) { Emit(0x1E220000 | (rn << 5) | vd); }
  // FP → Int
  void FCVTZS(GReg rd, VReg vn) { Emit(0x9E780000 | (vn << 5) | rd); }
  void FCVTZU(GReg rd, VReg vn) { Emit(0x9E790000 | (vn << 5) | rd); }
  void FCVTZS_S(GReg rd, VReg vn) { Emit(0x1E380000 | (vn << 5) | rd); }
  void FCVTZU_S(GReg rd, VReg vn) { Emit(0x1E390000 | (vn << 5) | rd); }
  // GP <-> FP moves
  void FMOV_XD(GReg rd, VReg vn) { Emit(0x9E660000 | (vn << 5) | rd); }
  void FMOV_DX(VReg vd, GReg rn) { Emit(0x9E670000 | (rn << 5) | vd); }
  void FMOV_WS(GReg rd, VReg vn) { Emit(0x1E260000 | (vn << 5) | rd); }
  void FMOV_SW(VReg vd, GReg rn) { Emit(0x1E270000 | (rn << 5) | vd); }

  // System: MSR/MRS for FPCR (rounding mode)
  void MRS_FPCR(GReg rt) { Emit(0xD53B4400 | rt); }
  void MSR_FPCR(GReg rt) { Emit(0xD51B4400 | rt); }

  // === NEON Vector ===
  // 4S (32-bit x 4)
  void ADD_4S(VReg vd, VReg vn, VReg vm) {
    Emit(0x4EA08400 | (vm << 16) | (vn << 5) | vd);
  }
  void SUB_4S(VReg vd, VReg vn, VReg vm) {
    Emit(0x6EA08400 | (vm << 16) | (vn << 5) | vd);
  }
  void MUL_4S(VReg vd, VReg vn, VReg vm) {
    Emit(0x4EA09C00 | (vm << 16) | (vn << 5) | vd);
  }
  void AND_16B(VReg vd, VReg vn, VReg vm) {
    Emit(0x4E201C00 | (vm << 16) | (vn << 5) | vd);
  }
  void ORR_16B(VReg vd, VReg vn, VReg vm) {
    Emit(0x4EA01C00 | (vm << 16) | (vn << 5) | vd);
  }
  void EOR_16B(VReg vd, VReg vn, VReg vm) {
    Emit(0x6E201C00 | (vm << 16) | (vn << 5) | vd);
  }
  void BIC_16B(VReg vd, VReg vn, VReg vm) {
    Emit(0x4E601C00 | (vm << 16) | (vn << 5) | vd);
  }
  void NOT_16B(VReg vd, VReg vn) {
    Emit(0x6E205800 | (vn << 5) | vd);
  }
  void MOV_16B(VReg vd, VReg vn) { ORR_16B(vd, vn, vn); }

  // NEON float (4S)
  void FADD_4S(VReg vd, VReg vn, VReg vm) {
    Emit(0x4E20D400 | (vm << 16) | (vn << 5) | vd);
  }
  void FSUB_4S(VReg vd, VReg vn, VReg vm) {
    Emit(0x4EA0D400 | (vm << 16) | (vn << 5) | vd);
  }
  void FMUL_4S(VReg vd, VReg vn, VReg vm) {
    Emit(0x6E20DC00 | (vm << 16) | (vn << 5) | vd);
  }
  void FMLA_4S(VReg vd, VReg vn, VReg vm) {
    Emit(0x4E20CC00 | (vm << 16) | (vn << 5) | vd);
  }
  void FMLS_4S(VReg vd, VReg vn, VReg vm) {
    Emit(0x4EA0CC00 | (vm << 16) | (vn << 5) | vd);
  }
  void FMAX_4S(VReg vd, VReg vn, VReg vm) {
    Emit(0x4E20F400 | (vm << 16) | (vn << 5) | vd);
  }
  void FMIN_4S(VReg vd, VReg vn, VReg vm) {
    Emit(0x4EA0F400 | (vm << 16) | (vn << 5) | vd);
  }
  void FABS_4S(VReg vd, VReg vn) { Emit(0x4EA0F800 | (vn << 5) | vd); }
  void FNEG_4S(VReg vd, VReg vn) { Emit(0x6EA0F800 | (vn << 5) | vd); }
  void FSQRT_4S(VReg vd, VReg vn) { Emit(0x6EA1F800 | (vn << 5) | vd); }
  void FRECPE_4S(VReg vd, VReg vn) { Emit(0x4EA1D800 | (vn << 5) | vd); }
  void FRSQRTE_4S(VReg vd, VReg vn) { Emit(0x6EA1D800 | (vn << 5) | vd); }

  // NEON compare (4S)
  void CMEQ_4S(VReg vd, VReg vn, VReg vm) {
    Emit(0x6EA08C00 | (vm << 16) | (vn << 5) | vd);
  }
  void CMGT_4S(VReg vd, VReg vn, VReg vm) {
    Emit(0x4EA03400 | (vm << 16) | (vn << 5) | vd);
  }
  void CMGE_4S(VReg vd, VReg vn, VReg vm) {
    Emit(0x4EA03C00 | (vm << 16) | (vn << 5) | vd);
  }
  void CMHI_4S(VReg vd, VReg vn, VReg vm) {
    Emit(0x6EA03400 | (vm << 16) | (vn << 5) | vd);
  }
  void CMHS_4S(VReg vd, VReg vn, VReg vm) {
    Emit(0x6EA03C00 | (vm << 16) | (vn << 5) | vd);
  }
  void FCMEQ_4S(VReg vd, VReg vn, VReg vm) {
    Emit(0x4E20E400 | (vm << 16) | (vn << 5) | vd);
  }
  void FCMGT_4S(VReg vd, VReg vn, VReg vm) {
    Emit(0x6EA0E400 | (vm << 16) | (vn << 5) | vd);
  }
  void FCMGE_4S(VReg vd, VReg vn, VReg vm) {
    Emit(0x6E20E400 | (vm << 16) | (vn << 5) | vd);
  }

  // NEON shifts
  void SHL_4S(VReg vd, VReg vn, uint32_t shift) {
    assert(shift < 32);
    Emit(0x4F205400 | ((32 + shift) << 16) | (vn << 5) | vd);
  }
  void USHR_4S(VReg vd, VReg vn, uint32_t shift) {
    assert(shift > 0 && shift <= 32);
    Emit(0x6F200400 | ((64 - shift) << 16) | (vn << 5) | vd);
  }
  void SSHR_4S(VReg vd, VReg vn, uint32_t shift) {
    assert(shift > 0 && shift <= 32);
    Emit(0x4F200400 | ((64 - shift) << 16) | (vn << 5) | vd);
  }

  // NEON permute/extract
  void EXT_16B(VReg vd, VReg vn, VReg vm, uint32_t index) {
    Emit(0x6E000000 | (vm << 16) | ((index & 0xF) << 11) | (vn << 5) | vd);
  }
  void TBL(VReg vd, VReg vn, VReg vm) {
    Emit(0x4E000000 | (vm << 16) | (vn << 5) | vd);
  }
  void DUP_4S(VReg vd, GReg rn) {
    Emit(0x4E040C00 | (rn << 5) | vd);
  }
  void DUP_S(VReg vd, VReg vn, uint32_t idx) {
    Emit(0x4E040400 | ((idx * 4 + 4) << 16) | (vn << 5) | vd);
  }

  // NEON variable shifts (4S) — shift amount per-lane
  void USHL_4S(VReg vd, VReg vn, VReg vm) {
    Emit(0x6EA04400 | (vm << 16) | (vn << 5) | vd);
  }
  void SSHL_4S(VReg vd, VReg vn, VReg vm) {
    Emit(0x4EA04400 | (vm << 16) | (vn << 5) | vd);
  }
  void NEG_4S(VReg vd, VReg vn) {
    Emit(0x6EA0B800 | (vn << 5) | vd);
  }

  // NEON integer min/max (4S)
  void SMAX_4S(VReg vd, VReg vn, VReg vm) {
    Emit(0x4EA06400 | (vm << 16) | (vn << 5) | vd);
  }
  void SMIN_4S(VReg vd, VReg vn, VReg vm) {
    Emit(0x4EA06C00 | (vm << 16) | (vn << 5) | vd);
  }
  void UMAX_4S(VReg vd, VReg vn, VReg vm) {
    Emit(0x6EA06400 | (vm << 16) | (vn << 5) | vd);
  }
  void UMIN_4S(VReg vd, VReg vn, VReg vm) {
    Emit(0x6EA06C00 | (vm << 16) | (vn << 5) | vd);
  }

  // NEON rounding halving add (unsigned average)
  void URHADD_4S(VReg vd, VReg vn, VReg vm) {
    Emit(0x6E201400 | (vm << 16) | (vn << 5) | vd);
  }
  void SRHADD_4S(VReg vd, VReg vn, VReg vm) {
    Emit(0x4E201400 | (vm << 16) | (vn << 5) | vd);
  }

  // NEON vector int<->float conversions
  void SCVTF_4S(VReg vd, VReg vn) { Emit(0x4E21D800 | (vn << 5) | vd); }
  void UCVTF_4S(VReg vd, VReg vn) { Emit(0x6E21D800 | (vn << 5) | vd); }
  void FCVTZS_4S(VReg vd, VReg vn) { Emit(0x4EA1B800 | (vn << 5) | vd); }
  void FCVTZU_4S(VReg vd, VReg vn) { Emit(0x6EA1B800 | (vn << 5) | vd); }

  // NEON vector rounding (4S)
  void FRINTN_4S(VReg vd, VReg vn) { Emit(0x4E218800 | (vn << 5) | vd); }
  void FRINTZ_4S(VReg vd, VReg vn) { Emit(0x4EA19800 | (vn << 5) | vd); }
  void FRINTM_4S(VReg vd, VReg vn) { Emit(0x4E219800 | (vn << 5) | vd); }
  void FRINTP_4S(VReg vd, VReg vn) { Emit(0x4EA18800 | (vn << 5) | vd); }

  // NEON permute (zip/unzip for pack/unpack)
  void ZIP1_4S(VReg vd, VReg vn, VReg vm) {
    Emit(0x4E803800 | (vm << 16) | (vn << 5) | vd);
  }
  void ZIP2_4S(VReg vd, VReg vn, VReg vm) {
    Emit(0x4E807800 | (vm << 16) | (vn << 5) | vd);
  }
  void UZP1_4S(VReg vd, VReg vn, VReg vm) {
    Emit(0x4E801800 | (vm << 16) | (vn << 5) | vd);
  }
  void UZP2_4S(VReg vd, VReg vn, VReg vm) {
    Emit(0x4E805800 | (vm << 16) | (vn << 5) | vd);
  }
  void REV64_4S(VReg vd, VReg vn) {
    Emit(0x4EA00800 | (vn << 5) | vd);
  }

  // NEON immediate: MOVI Vd.4S, #0 (zero a register)
  void MOVI_4S_zero(VReg vd) { Emit(0x4F000400 | vd); }

  // 32-bit element INS/UMOV
  void INS_S(VReg vd, uint32_t dst_idx, VReg vn, uint32_t src_idx) {
    // INS Vd.S[dst], Vn.S[src]
    uint32_t imm5 = (dst_idx << 3) | 0x4;  // size=S (0b100 at bits [2:0])
    uint32_t imm4 = src_idx << 2;
    Emit(0x6E000400 | (imm5 << 16) | (imm4 << 11) | (vn << 5) | vd);
  }
  void UMOV_W(GReg rd, VReg vn, uint32_t idx) {
    // UMOV Wd, Vn.S[idx]
    uint32_t imm5 = (idx << 3) | 0x4;
    Emit(0x0E003C00 | (imm5 << 16) | (vn << 5) | rd);
  }
  void INS_S_GPR(VReg vd, uint32_t idx, GReg rn) {
    // INS Vd.S[idx], Wn
    uint32_t imm5 = (idx << 3) | 0x4;
    Emit(0x4E001C00 | (imm5 << 16) | (rn << 5) | vd);
  }

  // NEON <-> GPR moves
  void UMOV(GReg rd, VReg vn, uint32_t idx) {
    // Move 64-bit from vector lane
    Emit(0x4E083C00 | ((idx * 8 + 8) << 16) | (vn << 5) | rd);
  }
  void INS(VReg vd, uint32_t idx, GReg rn) {
    Emit(0x4E081C00 | ((idx * 8 + 8) << 16) | (rn << 5) | vd);
  }

  // === System ===
  void NOP() { Emit(0xD503201F); }
  void BRK(uint16_t imm = 0) { Emit(0xD4200000 | (uint32_t(imm) << 5)); }
  void DMB_ISH() { Emit(0xD5033BBF); }
  void DSB_ISH() { Emit(0xD5033B9F); }
  void ISB() { Emit(0xD5033FDF); }

  // Address generation
  void ADR(GReg rd, int32_t offset) {
    uint32_t immlo = (offset & 3) << 29;
    uint32_t immhi = ((offset >> 2) & 0x7FFFF) << 5;
    Emit(0x10000000 | immlo | immhi | rd);
  }

  // Atomic (for reserved load/store)
  void LDXR(GReg rt, GReg rn) { Emit(0xC85F7C00 | (rn << 5) | rt); }
  void STXR(GReg rs, GReg rt, GReg rn) {
    Emit(0xC8007C00 | (rs << 16) | (rn << 5) | rt);
  }
  void LDXRw(GReg rt, GReg rn) { Emit(0x885F7C00 | (rn << 5) | rt); }
  void STXRw(GReg rs, GReg rt, GReg rn) {
    Emit(0x88007C00 | (rs << 16) | (rn << 5) | rt);
  }

  // Raw emit
  void Emit(uint32_t instr) { code_.push_back(instr); }

 private:
  // Branch types: 0=B/BL(26-bit), 1=B.cond(19-bit), 2=CBZ/CBNZ(19-bit)
  void EmitBranch(uint32_t base, Label* label, uint8_t type) {
    if (label->bound()) {
      int32_t rel = label->offset - static_cast<int32_t>(offset());
      ApplyBranchImm(base, rel, type);
    } else {
      label->patches.push_back({static_cast<uint32_t>(offset()), type});
      Emit(base);  // Will be patched
    }
  }

  void ApplyBranchImm(uint32_t base, int32_t rel, uint8_t type) {
    if (type == 0) {
      // B/BL: 26-bit signed offset in words
      Emit(base | (((rel >> 2) & 0x3FFFFFF)));
    } else {
      // B.cond, CBZ, CBNZ: 19-bit signed offset in words
      Emit(base | ((((rel >> 2) & 0x7FFFF)) << 5));
    }
  }

  void PatchBranch(uint32_t code_offset, int32_t target, uint8_t type) {
    int32_t rel = target - static_cast<int32_t>(code_offset);
    uint32_t idx = code_offset / 4;
    uint32_t base = code_[idx];
    if (type == 0) {
      code_[idx] = (base & ~0x3FFFFFF) | ((rel >> 2) & 0x3FFFFFF);
    } else {
      code_[idx] = (base & ~(0x7FFFF << 5)) | (((rel >> 2) & 0x7FFFF) << 5);
    }
  }

  std::vector<uint32_t> code_;
};

}  // namespace a64
}  // namespace backend
}  // namespace cpu
}  // namespace xe

#endif  // XENIA_CPU_BACKEND_A64_A64_ASM_H_
