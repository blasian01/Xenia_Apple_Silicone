/**
 * ARM64 Sequences — HIR opcode handlers (Phase 2: full integer + control flow)
 */
#include "xenia/cpu/backend/a64/a64_sequences.h"

#include <atomic>
#include <cmath>
#include <cstring>

#include "xenia/base/logging.h"
#include "xenia/base/math.h"
#include "xenia/cpu/backend/a64/a64_backend.h"
#include "xenia/cpu/backend/a64/a64_code_cache.h"
#include "xenia/cpu/backend/a64/a64_emitter.h"
#include "xenia/cpu/backend/a64/a64_function.h"
#include "xenia/cpu/backend/a64/a64_stack_layout.h"
#include "xenia/cpu/function.h"
#include "xenia/cpu/hir/hir_builder.h"
#include "xenia/cpu/hir/label.h"
#include "xenia/cpu/hir/opcodes.h"
#include "xenia/cpu/mmio_handler.h"
#include "xenia/cpu/ppc/ppc_context.h"
#include "xenia/cpu/processor.h"

namespace xe {
namespace cpu {
namespace backend {
namespace a64 {

using namespace xe::cpu::hir;

std::unordered_map<uint32_t, SequenceSelectFn> sequence_table;
thread_local const Instr* current_instr_for_logging = nullptr;
thread_local A64Emitter* current_emitter_for_logging = nullptr;

enum : uint32_t {
  kA64BackendHasReserveBit = 0,
};

static constexpr uint32_t kA64BackendHasReserveMask =
    1u << kA64BackendHasReserveBit;
static constexpr uintptr_t kMinimumLikelyHostPointer = 0x100000000ull;
static constexpr uint32_t kTrackedContextGuestFunction = 0x820B8E78;
static constexpr uint64_t kTrackedContextOffsetR3 =
    offsetof(ppc::PPCContext, r[3]);
static constexpr uint64_t kTrackedContextOffsetR11 =
    offsetof(ppc::PPCContext, r[11]);
static constexpr uint64_t kTrackedContextOffsetR28 =
    offsetof(ppc::PPCContext, r[28]);
static constexpr uint64_t kTrackedContextOffsetR30 =
    offsetof(ppc::PPCContext, r[30]);
static constexpr uint64_t kTrackedCountFieldStageRawLoad = 0;
static constexpr uint64_t kTrackedCountFieldStageByteSwap = 1;

static bool ShouldLogPossibleGuestReturnMismatch(uint32_t guest_function) {
  switch (guest_function) {
    case 0x820AF3F0:
    case 0x820AF414:
    case 0x820AF438:
    case 0x820AF4B0:
    case 0x820AFFD8:
      return true;
    default:
      return false;
  }
}

static bool ShouldLogPreparedReturnAddress(uint32_t guest_function) {
  switch (guest_function) {
    case 0x820AF430:
      return true;
    default:
      return false;
  }
}

static bool ShouldTraceContextAccess(uint32_t guest_function, uint64_t offset) {
  if (guest_function != kTrackedContextGuestFunction) {
    return false;
  }
  switch (offset) {
    case kTrackedContextOffsetR3:
    case kTrackedContextOffsetR11:
    case kTrackedContextOffsetR30:
      return true;
    default:
      return false;
  }
}

static uint64_t EncodeTrackedContextAccess(uint64_t offset, bool is_store) {
  return (offset & 0xFFFFull) | (is_store ? (1ull << 16) : 0);
}

static const char* TrackedContextOffsetName(uint64_t offset) {
  switch (offset) {
    case kTrackedContextOffsetR3:
      return "r3";
    case kTrackedContextOffsetR11:
      return "r11";
    case kTrackedContextOffsetR30:
      return "r30";
    default:
      return "unknown";
  }
}

// === Helpers ===

static bool IsLikelyHostValuePointer(const Value* v) {
  return reinterpret_cast<uintptr_t>(v) >= kMinimumLikelyHostPointer;
}

static void LogInvalidValuePointer(const char* usage, const Value* v) {
  const auto* instr = current_instr_for_logging;
  const uint32_t guest_fn =
      current_emitter_for_logging
          ? current_emitter_for_logging->current_guest_function()
          : 0;
  XELOGE("ARM64: invalid Value* {:016X} in {} for guest {:08X} opcode={}",
         reinterpret_cast<uintptr_t>(v), usage, guest_fn,
         instr ? GetOpcodeName(instr->GetOpcodeNum()) : "<null>");
}

static bool ValueIsConstant(const Value* v, const char* usage) {
  if (!v) {
    return false;
  }
  if (!IsLikelyHostValuePointer(v)) {
    LogInvalidValuePointer(usage, v);
    return false;
  }
  return v->IsConstant();
}

#define VALUE_IS_CONSTANT(v) ValueIsConstant((v), #v)

static bool IsAlloc(const Value* v) {
  if (v && !IsLikelyHostValuePointer(v)) {
    LogInvalidValuePointer("IsAlloc", v);
    return false;
  }
  return v && !v->IsConstant() && v->reg.set && v->reg.index >= 0;
}

static void LogBadRegisterUse(const char* reg_class, const Value* v) {
  if (v && !IsLikelyHostValuePointer(v)) {
    LogInvalidValuePointer(reg_class, v);
    return;
  }
  const auto* instr = current_instr_for_logging;
  const uint32_t guest_fn =
      current_emitter_for_logging
          ? current_emitter_for_logging->current_guest_function()
          : 0;
  XELOGE(
      "ARM64: invalid {} use in guest {:08X}: value v{} type={} reg_set={} "
      "reg_index={} def={} current_opcode={}",
      reg_class, guest_fn, v ? v->ordinal : UINT32_C(0),
      v ? static_cast<int>(v->type) : static_cast<int>(MAX_TYPENAME),
      (v && v->reg.set) ? v->reg.set->name : "<null>",
      v ? v->reg.index : -1,
      (v && v->def) ? GetOpcodeName(v->def->GetOpcodeNum()) : "<null>",
      instr ? GetOpcodeName(instr->GetOpcodeNum()) : "<null>");
}

static GReg GR(const Value* v) {
  if (!v || !v->reg.set || v->reg.index < 0 || v->reg.index >= GPR_COUNT) {
    LogBadRegisterUse("GPR", v);
    return kScratch0;
  }
  return A64Emitter::GprForIndex(v->reg.index);
}
static VReg VR(const Value* v) {
  if (!v || !v->reg.set || v->reg.index < 0 || v->reg.index >= VREG_COUNT) {
    LogBadRegisterUse("VREG", v);
    return kVScratch0;
  }
  return A64Emitter::VregForIndex(v->reg.index);
}

static GReg FrameBaseForOffset(A64Emitter& e, int32_t offset, GReg scratch,
                               int32_t* out_offset) {
  if (offset >= 0 && offset < 4096) {
    *out_offset = offset;
    return X29;
  }
  e.MovImm64(scratch, static_cast<uint32_t>(offset));
  e.asm_().ADD(scratch, X29, scratch);
  *out_offset = 0;
  return scratch;
}

// Load a value (const or register) into a scratch GPR. Returns the reg.
static GReg LoadGPR(A64Emitter& e, const Value* v, GReg scratch) {
  if (!v) {
    e.asm_().MOVZ(scratch, 0);
    return scratch;
  }
  if (!IsLikelyHostValuePointer(v)) {
    LogInvalidValuePointer("LoadGPR", v);
    e.asm_().MOVZ(scratch, 0);
    return scratch;
  }
  if (VALUE_IS_CONSTANT(v)) {
    e.MovImm64(scratch, v->constant.i64);
    return scratch;
  }
  return GR(v);
}

static GReg ComputeHostAddress(A64Emitter& e, const Value* guest_address,
                               const Value* offset, GReg host_addr,
                               GReg scratch) {
  GReg guest = LoadGPR(e, guest_address, host_addr);
  e.asm_().MOVw(host_addr, guest);

  if (offset) {
    GReg guest_offset = LoadGPR(e, offset, scratch);
    e.asm_().ADDw(host_addr, host_addr, guest_offset);
  }

  e.asm_().ADD(host_addr, kMembaseReg, host_addr);
  return host_addr;
}

// Loads a v128 operand into a register, materializing constants into the
// given scratch.
static VReg LoadVec128(A64Emitter& e, const Value* v, VReg scratch) {
  if (VALUE_IS_CONSTANT(v)) {
    e.LoadConstantV128(scratch, v->constant.v128);
    return scratch;
  }
  return VR(v);
}

static const vec128_t kByteSwapMaskVec128 =
    vec128b(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15);

static void EmitByteSwapVector128(A64Emitter& e, VReg dest, VReg src) {
  e.LoadConstantV128(kVScratch2, kByteSwapMaskVec128);
  e.asm_().TBL(dest, src, kVScratch2);
}

static void EmitLoadFloat32Memory(A64Emitter& e, VReg dest, GReg host_addr,
                                  bool byte_swap) {
  if (!byte_swap) {
    e.asm_().LDR_S(dest, host_addr);
    return;
  }
  e.asm_().LDRw(kScratch1, host_addr);
  e.asm_().REVw(kScratch1, kScratch1);
  e.asm_().FMOV_SW(dest, kScratch1);
}

static void EmitLoadFloat64Memory(A64Emitter& e, VReg dest, GReg host_addr,
                                  bool byte_swap) {
  if (!byte_swap) {
    e.asm_().LDR_D(dest, host_addr);
    return;
  }
  e.asm_().LDR(kScratch1, host_addr);
  e.asm_().REV(kScratch1, kScratch1);
  e.asm_().FMOV_DX(dest, kScratch1);
}

static void EmitLoadVector128Memory(A64Emitter& e, VReg dest, GReg host_addr,
                                    bool byte_swap) {
  e.asm_().LDR_Q(dest, host_addr);
  if (byte_swap) {
    EmitByteSwapVector128(e, dest, dest);
  }
}

static void EmitStoreHalfWordMemory(A64Emitter& e, GReg host_addr, GReg value,
                                    bool byte_swap) {
  GReg store_reg = value;
  if (byte_swap) {
    if (store_reg != kScratch2) {
      e.asm_().MOVw(kScratch2, store_reg);
      store_reg = kScratch2;
    }
    e.asm_().REV16(store_reg, store_reg);
  }
  e.asm_().STRH(store_reg, host_addr);
}

static void EmitStoreWordMemory(A64Emitter& e, GReg host_addr, GReg value,
                                bool byte_swap) {
  GReg store_reg = value;
  if (byte_swap) {
    if (store_reg != kScratch2) {
      e.asm_().MOVw(kScratch2, store_reg);
      store_reg = kScratch2;
    }
    e.asm_().REVw(store_reg, store_reg);
  }
  e.asm_().STRw(store_reg, host_addr);
}

static void EmitStoreDoubleWordMemory(A64Emitter& e, GReg host_addr,
                                      GReg value, bool byte_swap) {
  GReg store_reg = value;
  if (byte_swap) {
    if (store_reg != kScratch2) {
      e.asm_().MOV(kScratch2, store_reg);
      store_reg = kScratch2;
    }
    e.asm_().REV(store_reg, store_reg);
  }
  e.asm_().STR(store_reg, host_addr);
}

static void EmitStoreFloat32Memory(A64Emitter& e, GReg host_addr, GReg value,
                                   bool byte_swap) {
  EmitStoreWordMemory(e, host_addr, value, byte_swap);
}

static void EmitStoreFloat64Memory(A64Emitter& e, GReg host_addr, GReg value,
                                   bool byte_swap) {
  EmitStoreDoubleWordMemory(e, host_addr, value, byte_swap);
}

static void EmitStoreVector128Memory(A64Emitter& e, GReg host_addr, VReg value,
                                     bool byte_swap) {
  GReg preserved_host_addr = host_addr;
  if (byte_swap &&
      (preserved_host_addr == kScratch0 || preserved_host_addr == kScratch1)) {
    e.asm_().MOV(kScratch2, preserved_host_addr);
    preserved_host_addr = kScratch2;
  }

  VReg store_reg = value;
  if (byte_swap) {
    if (store_reg != kVScratch0) {
      e.asm_().MOV_16B(kVScratch0, store_reg);
      store_reg = kVScratch0;
    }
    EmitByteSwapVector128(e, store_reg, store_reg);
  }
  e.asm_().STR_Q(store_reg, preserved_host_addr);
}

extern "C" uint64_t A64UndefinedCallExtern(void* raw_context,
                                           uint64_t function_ptr,
                                           uint64_t unused) {
  auto function = reinterpret_cast<Function*>(function_ptr);
  XELOGE("undefined extern call to {:08X} {}", function->address(),
         function->name());
  return 0;
}

static A64BackendContext* A64GetBackendContext(void* raw_context) {
  return reinterpret_cast<A64BackendContext*>(
      reinterpret_cast<uint8_t*>(raw_context) - sizeof(A64BackendContext));
}

static void A64ReleaseCachedReservation(A64BackendContext* backend_context) {
  if (!(backend_context->flags & kA64BackendHasReserveMask) ||
      !backend_context->cached_reserve_offset) {
    backend_context->flags &= ~kA64BackendHasReserveMask;
    return;
  }

  auto reserve_word =
      reinterpret_cast<uint64_t*>(backend_context->cached_reserve_offset);
  uint64_t reserve_mask =
      uint64_t(1) << (backend_context->cached_reserve_bit & 63);
  __atomic_fetch_and(reserve_word, ~reserve_mask, __ATOMIC_ACQ_REL);
  backend_context->flags &= ~kA64BackendHasReserveMask;
}

static bool A64TryAcquireReservation(A64BackendContext* backend_context,
                                     uint32_t guest_address) {
  A64ReleaseCachedReservation(backend_context);
  if (!backend_context->reserve_helper_) {
    return false;
  }

  uint32_t reserve_index = guest_address >> RESERVE_BLOCK_SHIFT;
  auto reserve_word =
      &backend_context->reserve_helper_->blocks[reserve_index >> 6];
  uint32_t reserve_bit = reserve_index & 63;
  uint64_t reserve_mask = uint64_t(1) << reserve_bit;
  uint64_t previous =
      __atomic_fetch_or(reserve_word, reserve_mask, __ATOMIC_ACQ_REL);

  backend_context->cached_reserve_offset =
      reinterpret_cast<uint64_t>(reserve_word);
  backend_context->cached_reserve_bit = reserve_bit;

  bool acquired = (previous & reserve_mask) == 0;
  if (acquired) {
    backend_context->flags |= kA64BackendHasReserveMask;
  } else {
    backend_context->flags &= ~kA64BackendHasReserveMask;
  }
  return acquired;
}

template <typename T>
static T A64ReservedLoadImpl(void* raw_context, uint64_t guest_address_u64) {
  auto* ppc_context = reinterpret_cast<ppc::PPCContext*>(raw_context);
  auto* backend_context = A64GetBackendContext(raw_context);
  uint32_t guest_address = static_cast<uint32_t>(guest_address_u64);

  A64TryAcquireReservation(backend_context, guest_address);

  auto* host_ptr =
      reinterpret_cast<T*>(ppc_context->virtual_membase + guest_address);
  T value = __atomic_load_n(host_ptr, __ATOMIC_ACQUIRE);
  backend_context->cached_reserve_value_ = static_cast<uint64_t>(value);
  return value;
}

template <typename T>
static uint64_t A64ReservedStoreImpl(void* raw_context,
                                     uint64_t guest_address_u64,
                                     uint64_t value_u64) {
  auto* ppc_context = reinterpret_cast<ppc::PPCContext*>(raw_context);
  auto* backend_context = A64GetBackendContext(raw_context);
  uint32_t guest_address = static_cast<uint32_t>(guest_address_u64);

  if (!(backend_context->flags & kA64BackendHasReserveMask) ||
      !backend_context->reserve_helper_ ||
      !backend_context->cached_reserve_offset) {
    return 0;
  }

  auto* cached_reserve_word =
      reinterpret_cast<uint64_t*>(backend_context->cached_reserve_offset);
  uint32_t cached_reserve_bit = backend_context->cached_reserve_bit & 63;
  uint64_t cached_reserve_mask = uint64_t(1) << cached_reserve_bit;

  uint32_t current_reserve_index = guest_address >> RESERVE_BLOCK_SHIFT;
  auto* current_reserve_word =
      &backend_context->reserve_helper_->blocks[current_reserve_index >> 6];
  uint32_t current_reserve_bit = current_reserve_index & 63;

  bool success = false;
  if (cached_reserve_word == current_reserve_word &&
      cached_reserve_bit == current_reserve_bit) {
    auto* host_ptr =
        reinterpret_cast<T*>(ppc_context->virtual_membase + guest_address);
    T expected = static_cast<T>(backend_context->cached_reserve_value_);
    T desired = static_cast<T>(value_u64);
    success = __atomic_compare_exchange_n(host_ptr, &expected, desired, false,
                                          __ATOMIC_ACQ_REL, __ATOMIC_ACQUIRE);
  }

  __atomic_fetch_and(cached_reserve_word, ~cached_reserve_mask,
                     __ATOMIC_ACQ_REL);
  backend_context->flags &= ~kA64BackendHasReserveMask;
  return success ? 1 : 0;
}

extern "C" uint64_t A64ReservedLoad32(void* raw_context, uint64_t guest_address,
                                      uint64_t unused) {
  return A64ReservedLoadImpl<uint32_t>(raw_context, guest_address);
}

extern "C" uint64_t A64ReservedLoad64(void* raw_context, uint64_t guest_address,
                                      uint64_t unused) {
  return A64ReservedLoadImpl<uint64_t>(raw_context, guest_address);
}

extern "C" uint64_t A64ReservedStore32(void* raw_context,
                                       uint64_t guest_address,
                                       uint64_t value) {
  return A64ReservedStoreImpl<uint32_t>(raw_context, guest_address, value);
}

extern "C" uint64_t A64ReservedStore64(void* raw_context,
                                       uint64_t guest_address,
                                       uint64_t value) {
  return A64ReservedStoreImpl<uint64_t>(raw_context, guest_address, value);
}

extern "C" uint64_t A64LoadMmio(void* raw_context, uint64_t mmio_range_ptr,
                                uint64_t guest_address) {
  auto* mmio_range = reinterpret_cast<MMIORange*>(mmio_range_ptr);
  if (!mmio_range || !mmio_range->read) {
    return 0;
  }
  return mmio_range->read(raw_context, mmio_range->callback_context,
                          static_cast<uint32_t>(guest_address));
}

extern "C" uint64_t A64StoreMmio(void* raw_context, uint64_t mmio_range_ptr,
                                 uint64_t guest_address_and_value) {
  auto* mmio_range = reinterpret_cast<MMIORange*>(mmio_range_ptr);
  if (!mmio_range || !mmio_range->write) {
    return 0;
  }

  uint32_t guest_address = static_cast<uint32_t>(guest_address_and_value >> 32);
  uint32_t value = static_cast<uint32_t>(guest_address_and_value);
  mmio_range->write(raw_context, mmio_range->callback_context, guest_address,
                    value);
  return 0;
}

template <typename CopyFn>
static uint64_t A64AccessVectorSide(void* raw_context, uint64_t guest_address,
                                    CopyFn copy_fn) {
  auto* ppc_context = reinterpret_cast<ppc::PPCContext*>(raw_context);
  auto* backend_context = A64GetBackendContext(raw_context);
  uint8_t* scratch =
      reinterpret_cast<uint8_t*>(backend_context->helper_scratch);
  uint8_t* host_ptr =
      ppc_context->virtual_membase + static_cast<uint32_t>(guest_address);
  uint32_t eb = static_cast<uint32_t>(guest_address) & 0xF;
  copy_fn(scratch, host_ptr, eb);
  return 0;
}

extern "C" uint64_t A64LoadVectorLeft(void* raw_context, uint64_t guest_address,
                                      uint64_t unused) {
  return A64AccessVectorSide(
      raw_context, guest_address,
      [](uint8_t* scratch, uint8_t* host_ptr, uint32_t eb) {
        std::memset(scratch, 0, 16);
        std::memcpy(scratch, host_ptr, 16 - eb);
      });
}

extern "C" uint64_t A64LoadVectorRight(void* raw_context,
                                       uint64_t guest_address,
                                       uint64_t unused) {
  return A64AccessVectorSide(
      raw_context, guest_address,
      [](uint8_t* scratch, uint8_t* host_ptr, uint32_t eb) {
        std::memset(scratch, 0, 16);
        if (eb) {
          std::memcpy(scratch + (16 - eb), host_ptr, eb);
        }
      });
}

extern "C" uint64_t A64StoreVectorLeft(void* raw_context,
                                       uint64_t guest_address,
                                       uint64_t unused) {
  return A64AccessVectorSide(
      raw_context, guest_address,
      [](uint8_t* scratch, uint8_t* host_ptr, uint32_t eb) {
        std::memcpy(host_ptr, scratch, 16 - eb);
      });
}

extern "C" uint64_t A64StoreVectorRight(void* raw_context,
                                        uint64_t guest_address,
                                        uint64_t unused) {
  return A64AccessVectorSide(
      raw_context, guest_address,
      [](uint8_t* scratch, uint8_t* host_ptr, uint32_t eb) {
        if (eb) {
          std::memcpy(host_ptr, scratch + (16 - eb), eb);
        }
      });
}

// Whole-vector bit shifts (PPC vsl/vsr). The value is exchanged through
// helper_scratch; the shift amount (0-7 bits) arrives in arg0. Semantics
// mirror the x64 backend's EmulateShlV128/EmulateShrV128, including the ^3
// byte swizzle of the guest vector layout.
extern "C" uint64_t A64ShlV128(void* raw_context, uint64_t shamt,
                               uint64_t unused) {
  auto* backend_context = A64GetBackendContext(raw_context);
  auto* value = reinterpret_cast<vec128_t*>(backend_context->helper_scratch);
  const uint32_t n = static_cast<uint32_t>(shamt) & 0x7;
  vec128_t v = *value;
  for (int idx = 0; idx < 15; ++idx) {
    v.u8[idx ^ 0x3] = static_cast<uint8_t>(
        (v.u8[idx ^ 0x3] << n) | (v.u8[(idx + 1) ^ 0x3] >> (8 - n)));
  }
  v.u8[15 ^ 0x3] = static_cast<uint8_t>(v.u8[15 ^ 0x3] << n);
  *value = v;
  return 0;
}

extern "C" uint64_t A64ShrV128(void* raw_context, uint64_t shamt,
                               uint64_t unused) {
  auto* backend_context = A64GetBackendContext(raw_context);
  auto* value = reinterpret_cast<vec128_t*>(backend_context->helper_scratch);
  const uint32_t n = static_cast<uint32_t>(shamt) & 0x7;
  vec128_t v = *value;
  for (int idx = 15; idx > 0; --idx) {
    v.u8[idx ^ 0x3] = static_cast<uint8_t>(
        (v.u8[idx ^ 0x3] >> n) | (v.u8[(idx - 1) ^ 0x3] << (8 - n)));
  }
  v.u8[0 ^ 0x3] = static_cast<uint8_t>(v.u8[0 ^ 0x3] >> n);
  *value = v;
  return 0;
}

// PACK/UNPACK conversions exchanged through helper_scratch. Semantics match
// the x64 backend's emulation paths (see x64_seq_vector.cc) and the
// expectations in src/xenia/cpu/testing/{pack,unpack}_test.cc.
extern "C" uint64_t A64PackD3DCOLOR(void* raw_context, uint64_t arg0,
                                    uint64_t arg1) {
  auto* backend_context = A64GetBackendContext(raw_context);
  auto* value = reinterpret_cast<vec128_t*>(backend_context->helper_scratch);
  const vec128_t v = *value;
  auto to_byte = [](float f) -> uint32_t {
    // Valid inputs sit in [3.0, 3.0 + 255/256]; the packed byte is the low
    // mantissa byte of the clamped value.
    float lo = 3.0f;
    uint32_t hi_bits = 0x404000FF;
    float hi;
    std::memcpy(&hi, &hi_bits, sizeof(hi));
    f = std::min(std::max(f, lo), hi);
    uint32_t bits;
    std::memcpy(&bits, &f, sizeof(bits));
    return bits & 0xFF;
  };
  vec128_t r = vec128b(0);
  r.u32[3] = (to_byte(v.f32[3]) << 24) | (to_byte(v.f32[0]) << 16) |
             (to_byte(v.f32[1]) << 8) | to_byte(v.f32[2]);
  *value = r;
  return 0;
}

extern "C" uint64_t A64UnpackD3DCOLOR(void* raw_context, uint64_t arg0,
                                      uint64_t arg1) {
  auto* backend_context = A64GetBackendContext(raw_context);
  auto* value = reinterpret_cast<vec128_t*>(backend_context->helper_scratch);
  const uint32_t t = value->u32[3];
  vec128_t r;
  r.u32[0] = 0x3F800000 | ((t >> 16) & 0xFF);  // R
  r.u32[1] = 0x3F800000 | ((t >> 8) & 0xFF);   // G
  r.u32[2] = 0x3F800000 | (t & 0xFF);          // B
  r.u32[3] = 0x3F800000 | ((t >> 24) & 0xFF);  // A
  *value = r;
  return 0;
}

extern "C" uint64_t A64PackFLOAT16_2(void* raw_context, uint64_t arg0,
                                     uint64_t arg1) {
  auto* backend_context = A64GetBackendContext(raw_context);
  auto* value = reinterpret_cast<vec128_t*>(backend_context->helper_scratch);
  const vec128_t v = *value;
  vec128_t r = vec128b(0);
  r.u32[3] = (uint32_t(float_to_xenos_half(v.f32[0])) << 16) |
             float_to_xenos_half(v.f32[1]);
  *value = r;
  return 0;
}

extern "C" uint64_t A64UnpackFLOAT16_2(void* raw_context, uint64_t arg0,
                                       uint64_t arg1) {
  auto* backend_context = A64GetBackendContext(raw_context);
  auto* value = reinterpret_cast<vec128_t*>(backend_context->helper_scratch);
  const uint32_t t = value->u32[3];
  vec128_t r;
  r.f32[0] = xenos_half_to_float(uint16_t(t >> 16));
  r.f32[1] = xenos_half_to_float(uint16_t(t & 0xFFFF));
  r.f32[2] = 0.0f;
  r.f32[3] = 1.0f;
  *value = r;
  return 0;
}

extern "C" uint64_t A64PackFLOAT16_4(void* raw_context, uint64_t arg0,
                                     uint64_t arg1) {
  auto* backend_context = A64GetBackendContext(raw_context);
  auto* value = reinterpret_cast<vec128_t*>(backend_context->helper_scratch);
  const vec128_t v = *value;
  vec128_t r = vec128b(0);
  // Round-to-nearest-even, matching the x64 path.
  r.u16[5] = float_to_xenos_half(v.f32[0], false, true);
  r.u16[4] = float_to_xenos_half(v.f32[1], false, true);
  r.u16[7] = float_to_xenos_half(v.f32[2], false, true);
  r.u16[6] = float_to_xenos_half(v.f32[3], false, true);
  *value = r;
  return 0;
}

extern "C" uint64_t A64UnpackFLOAT16_4(void* raw_context, uint64_t arg0,
                                       uint64_t arg1) {
  auto* backend_context = A64GetBackendContext(raw_context);
  auto* value = reinterpret_cast<vec128_t*>(backend_context->helper_scratch);
  const vec128_t v = *value;
  vec128_t r;
  r.f32[0] = xenos_half_to_float(v.u16[5]);
  r.f32[1] = xenos_half_to_float(v.u16[4]);
  r.f32[2] = xenos_half_to_float(v.u16[7]);
  r.f32[3] = xenos_half_to_float(v.u16[6]);
  *value = r;
  return 0;
}

extern "C" uint64_t A64PackSHORT_2(void* raw_context, uint64_t arg0,
                                   uint64_t arg1) {
  auto* backend_context = A64GetBackendContext(raw_context);
  auto* value = reinterpret_cast<vec128_t*>(backend_context->helper_scratch);
  const vec128_t v = *value;
  auto to_short = [](float f) -> uint32_t {
    // Values are pre-biased around 3.0: n = (f - 3.0) * 2^22, saturated to
    // [-32767, 32767] (note the asymmetric lower bound, per vpkd3d).
    double n = (double(f) - 3.0) * 4194304.0;
    n = std::min(std::max(n, -32767.0), 32767.0);
    return uint32_t(int32_t(std::llround(n))) & 0xFFFF;
  };
  vec128_t r = vec128b(0);
  r.u32[3] = (to_short(v.f32[0]) << 16) | to_short(v.f32[1]);
  *value = r;
  return 0;
}

extern "C" uint64_t A64UnpackSHORT_2(void* raw_context, uint64_t arg0,
                                     uint64_t arg1) {
  auto* backend_context = A64GetBackendContext(raw_context);
  auto* value = reinterpret_cast<vec128_t*>(backend_context->helper_scratch);
  const uint32_t t = value->u32[3];
  vec128_t r;
  r.f32[0] = 3.0f + float(int16_t(t >> 16)) * (1.0f / 4194304.0f);
  r.f32[1] = 3.0f + float(int16_t(t & 0xFFFF)) * (1.0f / 4194304.0f);
  r.f32[2] = 0.0f;
  r.f32[3] = 1.0f;
  *value = r;
  return 0;
}

// POW2/LOG2 (PPC vexptefp/vlogefp and scalar HIR ops) via C helpers.
// arg0 selects the operand shape: 0 = f32 lane 0, 1 = f64 lane 0, 2 = all
// four f32 lanes.
extern "C" uint64_t A64Pow2(void* raw_context, uint64_t shape,
                            uint64_t unused) {
  auto* backend_context = A64GetBackendContext(raw_context);
  auto* value = reinterpret_cast<vec128_t*>(backend_context->helper_scratch);
  switch (shape) {
    case 0:
      value->f32[0] = std::exp2(value->f32[0]);
      break;
    case 1: {
      double d;
      std::memcpy(&d, &value->u64[0], sizeof(d));
      d = std::exp2(d);
      std::memcpy(&value->u64[0], &d, sizeof(d));
      break;
    }
    default:
      for (int i = 0; i < 4; ++i) {
        value->f32[i] = std::exp2(value->f32[i]);
      }
      break;
  }
  return 0;
}

extern "C" uint64_t A64Log2(void* raw_context, uint64_t shape,
                            uint64_t unused) {
  auto* backend_context = A64GetBackendContext(raw_context);
  auto* value = reinterpret_cast<vec128_t*>(backend_context->helper_scratch);
  switch (shape) {
    case 0:
      value->f32[0] = std::log2(value->f32[0]);
      break;
    case 1: {
      double d;
      std::memcpy(&d, &value->u64[0], sizeof(d));
      d = std::log2(d);
      std::memcpy(&value->u64[0], &d, sizeof(d));
      break;
    }
    default:
      for (int i = 0; i < 4; ++i) {
        value->f32[i] = std::log2(value->f32[i]);
      }
      break;
  }
  return 0;
}

extern "C" uint64_t A64LogPossibleGuestReturnMismatch(void* raw_context,
                                                      uint64_t guest_function,
                                                      uint64_t packed_values) {
  auto* guest_context = reinterpret_cast<ppc::PPCContext_s*>(raw_context);
  const uint32_t target = static_cast<uint32_t>(packed_values);
  const uint32_t guest_ret = static_cast<uint32_t>(packed_values >> 32);
  XELOGI(
      "ARM64: possible-return mismatch guest={:08X} target={:08X} "
      "guest_ret={:08X} ctx_lr={:08X} r1={:08X}",
      static_cast<uint32_t>(guest_function), target, guest_ret,
      static_cast<uint32_t>(guest_context->lr),
      static_cast<uint32_t>(guest_context->r[1]));
  return 0;
}

extern "C" uint64_t A64LogPreparedReturnAddress(void* raw_context,
                                                uint64_t guest_function,
                                                uint64_t packed_values) {
  auto* guest_context = reinterpret_cast<ppc::PPCContext_s*>(raw_context);
  const uint32_t prepared_ret = static_cast<uint32_t>(packed_values);
  const bool is_tail = (packed_values >> 32) != 0;
  XELOGI(
      "ARM64: prepared-return guest={:08X} kind={} prepared_ret={:08X} "
      "ctx_lr={:08X} r1={:08X}",
      static_cast<uint32_t>(guest_function), is_tail ? "tail" : "call",
      prepared_ret, static_cast<uint32_t>(guest_context->lr),
      static_cast<uint32_t>(guest_context->r[1]));
  return 0;
}

extern "C" uint64_t A64LogZeroIndirectGuestTarget(void* raw_context,
                                                  uint64_t guest_function,
                                                  uint64_t target_address) {
  auto* guest_context = reinterpret_cast<ppc::PPCContext_s*>(raw_context);
  XELOGE(
      "ARM64: zero indirect target guest={:08X} target={:08X} ctx_lr={:08X} "
      "ctx_ctr={:08X} r1={:08X} r13={:08X}",
      static_cast<uint32_t>(guest_function), static_cast<uint32_t>(target_address),
      static_cast<uint32_t>(guest_context->lr),
      static_cast<uint32_t>(guest_context->ctr),
      static_cast<uint32_t>(guest_context->r[1]),
      static_cast<uint32_t>(guest_context->r[13]));
  return 0;
}

extern "C" uint64_t A64LogTrackedContextAccess(void* raw_context,
                                               uint64_t encoded_access,
                                               uint64_t value) {
  static std::atomic<uint64_t> access_log_count{0};
  const uint64_t count = ++access_log_count;
  const uint64_t offset = encoded_access & 0xFFFFull;
  const bool is_store = (encoded_access & (1ull << 16)) != 0;
  const uint32_t low32 = static_cast<uint32_t>(value);
  if (count > 96 && low32 != 0) {
    return 0;
  }

  auto* guest_context = reinterpret_cast<ppc::PPCContext_s*>(raw_context);
  XELOGI(
      "ARM64: ctx-trace #{} guest={:08X} kind={} {} value={:016X} "
      "low32={:08X} ctx_lr={:08X}",
      count, kTrackedContextGuestFunction, is_store ? "store" : "load",
      TrackedContextOffsetName(offset), value, low32,
      static_cast<uint32_t>(guest_context->lr));
  return 0;
}

static void MaybeEmitTrackedContextAccess(A64Emitter& e, uint64_t offset,
                                          GReg value_reg, bool is_store) {
  if (!ShouldTraceContextAccess(e.current_guest_function(), offset)) {
    return;
  }

  auto backend = static_cast<A64Backend*>(e.backend());
  e.MovImm64(X0, reinterpret_cast<uint64_t>(&A64LogTrackedContextAccess));
  e.MovImm64(X1, EncodeTrackedContextAccess(offset, is_store));
  if (value_reg != X2) {
    e.asm_().ORR(X2, value_reg, value_reg);
  }
  e.MovImm64(kScratch0,
             reinterpret_cast<uint64_t>(backend->guest_to_host_thunk()));
  e.asm_().BLR(kScratch0);
}

extern "C" uint64_t A64LogTrackedLoopSetup(void* raw_context,
                                           uint64_t count_value,
                                           uint64_t unused) {
  auto* guest_context = reinterpret_cast<ppc::PPCContext_s*>(raw_context);
  const uint32_t r27 = static_cast<uint32_t>(guest_context->r[27]);
  const uint32_t r29 = static_cast<uint32_t>(guest_context->r[29]);
  const uint32_t r30 = static_cast<uint32_t>(guest_context->r[30]);
  const uint32_t r31 = static_cast<uint32_t>(guest_context->r[31]);

  uint32_t outer_offset_raw = 0;
  uint32_t outer_offset = 0;
  if (auto* ptr = guest_context->TranslateVirtual<const uint32_t*>(r27)) {
    outer_offset_raw = *ptr;
    outer_offset = xe::byte_swap(outer_offset_raw);
  }

  const uint32_t sub_base = r31 + outer_offset;
  uint32_t field380_raw = 0;
  uint32_t field380 = 0;
  if (auto* ptr =
          guest_context->TranslateVirtual<const uint32_t*>(sub_base + 380)) {
    field380_raw = *ptr;
    field380 = xe::byte_swap(field380_raw);
  }

  uint32_t field384_raw = 0;
  uint32_t field384 = 0;
  if (auto* ptr =
          guest_context->TranslateVirtual<const uint32_t*>(sub_base + 384)) {
    field384_raw = *ptr;
    field384 = xe::byte_swap(field384_raw);
  }

  XELOGI(
      "ARM64: loop-setup guest={:08X} lr={:08X} r27={:08X} offset_raw={:08X} "
      "offset={:08X} sub_base={:08X} field380_raw={:08X} field380={:08X} "
      "field384_raw={:08X} field384={:08X} count={:08X} r29={:08X} "
      "r30={:08X} r31={:08X}",
      kTrackedContextGuestFunction, static_cast<uint32_t>(guest_context->lr),
      r27, outer_offset_raw, outer_offset, sub_base, field380_raw, field380,
      field384_raw, field384, static_cast<uint32_t>(count_value), r29, r30,
      r31);
  return 0;
}

static bool ShouldTraceLoopSetupStore(A64Emitter& e, const Instr* i,
                                      uint64_t offset) {
  if (e.current_guest_function() != kTrackedContextGuestFunction ||
      offset != kTrackedContextOffsetR28) {
    return false;
  }
  auto src = i->src2.value;
  if (!src || !src->def ||
      src->def->GetOpcodeNum() != OPCODE_LOAD_CONTEXT) {
    return false;
  }
  return src->def->src1.offset == offsetof(ppc::PPCContext, r[11]);
}

static void MaybeEmitTrackedLoopSetup(A64Emitter& e, GReg count_reg) {
  auto backend = static_cast<A64Backend*>(e.backend());
  e.MovImm64(X0, reinterpret_cast<uint64_t>(&A64LogTrackedLoopSetup));
  if (count_reg != X1) {
    e.asm_().ORR(X1, count_reg, count_reg);
  }
  e.asm_().MOVZ(X2, 0);
  e.MovImm64(kScratch0,
             reinterpret_cast<uint64_t>(backend->guest_to_host_thunk()));
  e.asm_().BLR(kScratch0);
}

extern "C" uint64_t A64LogTrackedCountField(void* raw_context, uint64_t stage,
                                            uint64_t value) {
  auto* guest_context = reinterpret_cast<ppc::PPCContext_s*>(raw_context);
  const uint32_t r11 = static_cast<uint32_t>(guest_context->r[11]);
  uint32_t mem_raw = 0;
  uint32_t mem_swapped = 0;
  if (auto* ptr = guest_context->TranslateVirtual<const uint32_t*>(r11 + 384)) {
    mem_raw = *ptr;
    mem_swapped = xe::byte_swap(mem_raw);
  }
  XELOGI(
      "ARM64: count-field guest={:08X} stage={} value={:08X} r11={:08X} "
      "mem_raw={:08X} mem_swapped={:08X} lr={:08X}",
      kTrackedContextGuestFunction,
      stage == kTrackedCountFieldStageRawLoad ? "raw" : "swap",
      static_cast<uint32_t>(value), r11, mem_raw, mem_swapped,
      static_cast<uint32_t>(guest_context->lr));
  return 0;
}

static bool ShouldTraceCountFieldLoadOffset(A64Emitter& e, const Instr* i) {
  if (e.current_guest_function() != kTrackedContextGuestFunction ||
      i->dest->type != INT32_TYPE || !i->src2.value ||
      !VALUE_IS_CONSTANT(i->src2.value) ||
      i->src2.value->constant.u64 != 384) {
    return false;
  }
  return true;
}

static bool ShouldTraceCountFieldByteSwap(A64Emitter& e, const Instr* i) {
  if (e.current_guest_function() != kTrackedContextGuestFunction ||
      !i->src1.value || !i->src1.value->def ||
      i->src1.value->def->GetOpcodeNum() != OPCODE_LOAD_OFFSET) {
    return false;
  }
  auto* load = i->src1.value->def;
  if (!load->src2.value || !VALUE_IS_CONSTANT(load->src2.value) ||
      load->src2.value->constant.u64 != 384) {
    return false;
  }
  return true;
}

static void MaybeEmitTrackedCountField(A64Emitter& e, uint64_t stage,
                                       GReg value_reg) {
  auto backend = static_cast<A64Backend*>(e.backend());
  e.MovImm64(X0, reinterpret_cast<uint64_t>(&A64LogTrackedCountField));
  e.MovImm64(X1, stage);
  if (value_reg != X2) {
    e.asm_().ORR(X2, value_reg, value_reg);
  }
  e.MovImm64(kScratch0,
             reinterpret_cast<uint64_t>(backend->guest_to_host_thunk()));
  e.asm_().BLR(kScratch0);
}

static bool ShouldSkipConditionalOp(const Value* cond) {
  if (!cond) {
    return true;
  }
  if (VALUE_IS_CONSTANT(cond)) {
    return cond->IsConstantFalse();
  }
  return !IsAlloc(cond);
}

static bool MaybeEmitConditionalSkip(A64Emitter& e, const Value* cond,
                                     a64::Label* skip) {
  if (!cond || VALUE_IS_CONSTANT(cond)) {
    return false;
  }
  e.asm_().CBZ(GR(cond), skip);
  return true;
}

static void EmitTailBranch(A64Emitter& e, GReg target_reg) {
  size_t total = StackLayout::HOST_FRAME_SAVE_SIZE +
                 StackLayout::GUEST_STACK_SIZE + e.stack_size();
  total = (total + 15) & ~size_t(15);

  e.asm_().LDP_Q(V16, V17, X29, StackLayout::GUEST_VREG_SAVE + 0x00);
  e.asm_().LDP_Q(V18, V19, X29, StackLayout::GUEST_VREG_SAVE + 0x20);
  e.asm_().LDP_Q(V20, V21, X29, StackLayout::GUEST_VREG_SAVE + 0x40);
  e.asm_().LDP_Q(V22, V23, X29, StackLayout::GUEST_VREG_SAVE + 0x60);
  e.asm_().LDP_Q(V24, V25, X29, StackLayout::GUEST_VREG_SAVE + 0x80);
  e.asm_().LDP_Q(V26, V27, X29, StackLayout::GUEST_VREG_SAVE + 0xA0);
  e.asm_().LDP_Q(V28, V29, X29, StackLayout::GUEST_VREG_SAVE + 0xC0);
  e.asm_().LDP_Q(V30, V31, X29, StackLayout::GUEST_VREG_SAVE + 0xE0);

  e.asm_().LDP(X21, X22, X29, StackLayout::GUEST_GPR_SAVE + 0x00);
  e.asm_().LDP(X23, X24, X29, StackLayout::GUEST_GPR_SAVE + 0x10);
  e.asm_().LDP(X25, X26, X29, StackLayout::GUEST_GPR_SAVE + 0x20);
  e.asm_().LDP(X27, X28, X29, StackLayout::GUEST_GPR_SAVE + 0x30);

  e.asm_().LDP(X29, X30, SP, 0);
  e.asm_().ADD(SP, SP, static_cast<int32_t>(total));
  e.asm_().BR(target_reg);
}

static void EmitCallOrTailBranch(A64Emitter& e, const Instr* i,
                                 GReg target_reg) {
  if (i->flags & CALL_TAIL) {
    EmitTailBranch(e, target_reg);
  } else {
    e.asm_().BLR(target_reg);
  }
}

static void PrepareGuestCallReturnAddress(A64Emitter& e, const Instr* i) {
  const bool is_tail = (i->flags & CALL_TAIL) != 0;
  if (is_tail) {
    e.asm_().LDR(X2, X29, StackLayout::GUEST_RET_ADDR);
  } else {
    e.asm_().LDR(X2, X29, StackLayout::GUEST_CALL_RET_ADDR);
  }

  if (!ShouldLogPreparedReturnAddress(e.current_guest_function())) {
    return;
  }

  e.asm_().STR(X2, X29, StackLayout::GUEST_SCRATCH + 0x18);
  e.MovImm64(X0, reinterpret_cast<uint64_t>(&A64LogPreparedReturnAddress));
  e.MovImm64(X1, e.current_guest_function());
  e.MovImm64(kScratch2, is_tail ? 1ull << 32 : 0);
  e.asm_().ORR(X2, X2, kScratch2);
  auto backend = static_cast<A64Backend*>(e.backend());
  e.MovImm64(kScratch0,
             reinterpret_cast<uint64_t>(backend->guest_to_host_thunk()));
  e.asm_().BLR(kScratch0);
  e.asm_().LDR(X2, X29, StackLayout::GUEST_SCRATCH + 0x18);
}

static void MaybeEmitPossibleGuestReturn(A64Emitter& e, const Instr* i,
                                         GReg target_guest_address) {
  if (!(i->flags & CALL_POSSIBLE_RETURN)) {
    return;
  }

  const bool log_mismatch =
      ShouldLogPossibleGuestReturnMismatch(e.current_guest_function());
  auto backend = static_cast<A64Backend*>(e.backend());
  e.asm_().LDR(kScratch1, X29, StackLayout::GUEST_RET_ADDR);
  e.asm_().CMPw(target_guest_address, kScratch1);
  a64::Label not_return;
  e.asm_().B(NE, &not_return);
  e.EmitEpilogue(e.stack_size());
  e.asm_().Bind(&not_return);

  if (!log_mismatch) {
    return;
  }

  e.asm_().STR(target_guest_address, X29, StackLayout::GUEST_SCRATCH + 0x00);
  e.asm_().STR(kScratch1, X29, StackLayout::GUEST_SCRATCH + 0x08);
  e.asm_().LSL_imm(kScratch2, kScratch1, 32);
  e.asm_().ORR(kScratch2, kScratch2, target_guest_address);
  e.MovImm64(X0,
             reinterpret_cast<uint64_t>(&A64LogPossibleGuestReturnMismatch));
  e.MovImm64(X1, e.current_guest_function());
  e.asm_().MOV(X2, kScratch2);
  e.MovImm64(kScratch0,
             reinterpret_cast<uint64_t>(backend->guest_to_host_thunk()));
  e.asm_().BLR(kScratch0);
  e.asm_().LDR(target_guest_address, X29, StackLayout::GUEST_SCRATCH + 0x00);
  e.asm_().LDR(kScratch1, X29, StackLayout::GUEST_SCRATCH + 0x08);
}

static bool EmitCallExternFunction(A64Emitter& e, const Instr* i,
                                   const Function* function) {
  if (!function) {
    return true;
  }

  auto backend = static_cast<A64Backend*>(e.backend());
  uint64_t target = reinterpret_cast<uint64_t>(&A64UndefinedCallExtern);
  uint64_t arg0 = reinterpret_cast<uint64_t>(function);
  uint64_t arg1 = 0;
  bool load_kernel_state = false;

  switch (function->behavior()) {
    case Function::Behavior::kBuiltin: {
      auto builtin = static_cast<const BuiltinFunction*>(function);
      if (builtin->handler()) {
        target = reinterpret_cast<uint64_t>(builtin->handler());
        arg0 = reinterpret_cast<uint64_t>(builtin->arg0());
        arg1 = reinterpret_cast<uint64_t>(builtin->arg1());
      }
      break;
    }
    case Function::Behavior::kExtern: {
      auto extern_function = static_cast<const GuestFunction*>(function);
      if (extern_function->extern_handler()) {
        target = reinterpret_cast<uint64_t>(extern_function->extern_handler());
        load_kernel_state = true;
      }
      break;
    }
    default:
      break;
  }

  e.MovImm64(X0, target);
  if (load_kernel_state) {
    e.asm_().LDR(X1, kContextReg, offsetof(ppc::PPCContext, kernel_state));
  } else {
    e.MovImm64(X1, arg0);
  }
  e.MovImm64(X2, arg1);
  e.MovImm64(kScratch0,
             reinterpret_cast<uint64_t>(backend->guest_to_host_thunk()));
  EmitCallOrTailBranch(e, i, kScratch0);
  return true;
}

static bool EmitGuestFunctionCall(A64Emitter& e, const Instr* i,
                                  const Function* function) {
  if (!function) {
    return true;
  }

  if (function->behavior() == Function::Behavior::kBuiltin ||
      function->behavior() == Function::Behavior::kExtern) {
    return EmitCallExternFunction(e, i, function);
  }

  PrepareGuestCallReturnAddress(e, i);

  auto guest_function = static_cast<const GuestFunction*>(function);
  if (guest_function->machine_code()) {
    e.MovImm64(kScratch0,
               reinterpret_cast<uint64_t>(guest_function->machine_code()));
    EmitCallOrTailBranch(e, i, kScratch0);
    return true;
  }

  auto a64_function = static_cast<const A64Function*>(guest_function);
  e.MovImm64(kScratch0,
             reinterpret_cast<uint64_t>(a64_function->machine_code_slot()));
  e.asm_().LDR(kScratch0, kScratch0, 0);

  a64::Label unresolved;
  if (i->flags & CALL_TAIL) {
    e.asm_().CBZ(kScratch0, &unresolved);
    EmitCallOrTailBranch(e, i, kScratch0);
    e.asm_().Bind(&unresolved);
    e.MovImm64(X9, function->address());
    auto backend = static_cast<A64Backend*>(e.backend());
    e.MovImm64(kScratch1,
               reinterpret_cast<uint64_t>(backend->resolve_function_thunk()));
    EmitCallOrTailBranch(e, i, kScratch1);
    return true;
  }

  a64::Label done;
  e.asm_().CBZ(kScratch0, &unresolved);
  EmitCallOrTailBranch(e, i, kScratch0);
  e.asm_().B(&done);
  e.asm_().Bind(&unresolved);
  e.MovImm64(X9, function->address());
  auto backend = static_cast<A64Backend*>(e.backend());
  e.MovImm64(kScratch1,
             reinterpret_cast<uint64_t>(backend->resolve_function_thunk()));
  EmitCallOrTailBranch(e, i, kScratch1);
  e.asm_().Bind(&done);
  return true;
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

static bool EmitDebugBreakTrue(A64Emitter& e, const Instr* i) {
  auto cond = i->src1.value;
  if (ShouldSkipConditionalOp(cond)) {
    return true;
  }
  a64::Label skip;
  if (MaybeEmitConditionalSkip(e, cond, &skip)) {
    e.asm_().BRK(0);
    e.asm_().Bind(&skip);
    return true;
  }
  e.asm_().BRK(0);
  return true;
}

static bool EmitTrap(A64Emitter& e, const Instr* i) {
  e.asm_().BRK(0);
  return true;
}

static bool EmitTrapTrue(A64Emitter& e, const Instr* i) {
  auto cond = i->src1.value;
  if (ShouldSkipConditionalOp(cond)) {
    return true;
  }
  a64::Label skip;
  if (MaybeEmitConditionalSkip(e, cond, &skip)) {
    e.asm_().BRK(0);
    e.asm_().Bind(&skip);
    return true;
  }
  e.asm_().BRK(0);
  return true;
}

static bool EmitReturn(A64Emitter& e, const Instr* i) {
  e.EmitEpilogue(e.stack_size());
  return true;
}

static bool EmitReturnTrue(A64Emitter& e, const Instr* i) {
  auto cond = i->src1.value;
  if (ShouldSkipConditionalOp(cond)) {
    return true;
  }
  a64::Label skip;
  if (MaybeEmitConditionalSkip(e, cond, &skip)) {
    e.EmitEpilogue(e.stack_size());
    e.asm_().Bind(&skip);
    return true;
  }
  e.EmitEpilogue(e.stack_size());
  return true;
}

static bool EmitBranch(A64Emitter& e, const Instr* i) {
  // OPCODE_BRANCH: unconditional branch to HIR label
  auto* hir_label = i->src1.label;
  if (hir_label) {
    auto* target = e.GetLabel(hir_label->id);
    e.asm_().B(target);
  }
  return true;
}

static bool EmitBranchTrue(A64Emitter& e, const Instr* i) {
  auto cond = i->src1.value;
  auto* hir_label = i->src2.label;
  if (!cond || !hir_label) return true;
  auto* target = e.GetLabel(hir_label->id);
  if (VALUE_IS_CONSTANT(cond)) {
    if (cond->IsConstantTrue()) {
      e.asm_().B(target);
    }
    return true;
  }
  if (IsAlloc(cond)) {
    e.asm_().CBNZ(GR(cond), target);
  }
  return true;
}

static bool EmitBranchFalse(A64Emitter& e, const Instr* i) {
  auto cond = i->src1.value;
  auto* hir_label = i->src2.label;
  if (!cond || !hir_label) return true;
  auto* target = e.GetLabel(hir_label->id);
  if (VALUE_IS_CONSTANT(cond)) {
    if (cond->IsConstantFalse()) {
      e.asm_().B(target);
    }
    return true;
  }
  if (IsAlloc(cond)) {
    e.asm_().CBZ(GR(cond), target);
  }
  return true;
}

static bool EmitCall(A64Emitter& e, const Instr* i) {
  return EmitGuestFunctionCall(e, i, i->src1.symbol);
}

static bool EmitCallTrue(A64Emitter& e, const Instr* i) {
  auto cond = i->src1.value;
  if (ShouldSkipConditionalOp(cond)) {
    return true;
  }
  a64::Label skip;
  if (MaybeEmitConditionalSkip(e, cond, &skip)) {
    auto result = EmitGuestFunctionCall(e, i, i->src2.symbol);
    e.asm_().Bind(&skip);
    return result;
  }
  return EmitGuestFunctionCall(e, i, i->src2.symbol);
}

// Branches to a guest target held in X9, taking the indirection-table fast
// path when the target's slot is populated (32-bit offset into the JIT
// region, 0 = unresolved) and falling back to the resolve thunk otherwise.
// The resolve path patches the slot, so each target resolves at most once.
static void EmitIndirectCallTarget(A64Emitter& e, const Instr* i) {
  auto backend = static_cast<A64Backend*>(e.backend());
  auto code_cache = backend->code_cache();
  if (!code_cache->has_indirection_table()) {
    e.MovImm64(kScratch1,
               reinterpret_cast<uint64_t>(backend->resolve_function_thunk()));
    EmitCallOrTailBranch(e, i, kScratch1);
    return;
  }

  a64::Label resolve, done;
  // NOTE: X9 (kScratch0) holds the guest target and must survive into the
  // resolve path, so only kScratch1/kScratch2 are usable here.
  // Only [0x80000000, 0xA0000000) has table pages: (target >> 29) == 4.
  e.asm_().LSR_imm(kScratch1, X9, 29);
  e.asm_().SUB_imm(kScratch1, kScratch1, 4);
  e.asm_().CBNZ(kScratch1, &resolve);
  // slot = table_base + (target - 0x80000000); the bias is folded into the
  // constant (unsigned wrap-around is fine).
  e.MovImm64(kScratch2,
             reinterpret_cast<uint64_t>(code_cache->indirection_table_base()) -
                 0x80000000ull);
  e.asm_().LDRw_reg(kScratch2, kScratch2, X9);
  e.asm_().CBZ(kScratch2, &resolve);
  e.MovImm64(kScratch1, code_cache->execute_base_address());
  e.asm_().ADD(kScratch1, kScratch1, kScratch2);
  EmitCallOrTailBranch(e, i, kScratch1);
  e.asm_().B(&done);
  e.asm_().Bind(&resolve);
  e.MovImm64(kScratch1,
             reinterpret_cast<uint64_t>(backend->resolve_function_thunk()));
  EmitCallOrTailBranch(e, i, kScratch1);
  e.asm_().Bind(&done);
}

static bool EmitCallIndirect(A64Emitter& e, const Instr* i) {
  auto target = i->src1.value;
  if (!target) return true;

  GReg addr = LoadGPR(e, target, X9);
  if (addr != X9) {
    e.asm_().MOVw(X9, addr);
  }
  a64::Label nonzero_target;
  e.asm_().CBNZ(X9, &nonzero_target);
  e.asm_().STR(X9, X29, StackLayout::GUEST_SCRATCH + 0x00);
  e.MovImm64(X0, reinterpret_cast<uint64_t>(&A64LogZeroIndirectGuestTarget));
  e.MovImm64(X1, e.current_guest_function());
  e.MovImm64(X2, 0);
  auto backend = static_cast<A64Backend*>(e.backend());
  e.MovImm64(kScratch0,
             reinterpret_cast<uint64_t>(backend->guest_to_host_thunk()));
  e.asm_().BLR(kScratch0);
  e.asm_().LDR(X9, X29, StackLayout::GUEST_SCRATCH + 0x00);
  e.asm_().Bind(&nonzero_target);
  MaybeEmitPossibleGuestReturn(e, i, X9);

  PrepareGuestCallReturnAddress(e, i);

  // X9 holds the guest target; branch via the indirection table.
  EmitIndirectCallTarget(e, i);
  return true;
}

static bool EmitCallIndirectTrue(A64Emitter& e, const Instr* i) {
  auto cond = i->src1.value;
  if (ShouldSkipConditionalOp(cond)) {
    return true;
  }
  a64::Label skip;
  if (MaybeEmitConditionalSkip(e, cond, &skip)) {
    auto target = i->src2.value;
    if (target) {
      GReg addr = LoadGPR(e, target, X9);
      if (addr != X9) {
        e.asm_().MOVw(X9, addr);
      }
      a64::Label nonzero_target;
      e.asm_().CBNZ(X9, &nonzero_target);
      e.asm_().STR(X9, X29, StackLayout::GUEST_SCRATCH + 0x00);
      e.MovImm64(X0, reinterpret_cast<uint64_t>(&A64LogZeroIndirectGuestTarget));
      e.MovImm64(X1, e.current_guest_function());
      e.MovImm64(X2, 0);
      auto backend = static_cast<A64Backend*>(e.backend());
      e.MovImm64(kScratch0,
                 reinterpret_cast<uint64_t>(backend->guest_to_host_thunk()));
      e.asm_().BLR(kScratch0);
      e.asm_().LDR(X9, X29, StackLayout::GUEST_SCRATCH + 0x00);
      e.asm_().Bind(&nonzero_target);
      MaybeEmitPossibleGuestReturn(e, i, X9);
      PrepareGuestCallReturnAddress(e, i);
      EmitIndirectCallTarget(e, i);
    }
    e.asm_().Bind(&skip);
    return true;
  }
  auto target = i->src2.value;
  if (!target) {
    return true;
  }
  GReg addr = LoadGPR(e, target, X9);
  if (addr != X9) {
    e.asm_().MOVw(X9, addr);
  }
  a64::Label nonzero_target;
  e.asm_().CBNZ(X9, &nonzero_target);
  e.asm_().STR(X9, X29, StackLayout::GUEST_SCRATCH + 0x00);
  e.MovImm64(X0, reinterpret_cast<uint64_t>(&A64LogZeroIndirectGuestTarget));
  e.MovImm64(X1, e.current_guest_function());
  e.MovImm64(X2, 0);
  auto backend = static_cast<A64Backend*>(e.backend());
  e.MovImm64(kScratch0,
             reinterpret_cast<uint64_t>(backend->guest_to_host_thunk()));
  e.asm_().BLR(kScratch0);
  e.asm_().LDR(X9, X29, StackLayout::GUEST_SCRATCH + 0x00);
  e.asm_().Bind(&nonzero_target);
  MaybeEmitPossibleGuestReturn(e, i, X9);
  PrepareGuestCallReturnAddress(e, i);
  EmitIndirectCallTarget(e, i);
  return true;
}

static bool EmitCallExtern(A64Emitter& e, const Instr* i) {
  return EmitCallExternFunction(e, i, i->src1.symbol);
}

static bool EmitSetReturnAddress(A64Emitter& e, const Instr* i) {
  auto address = i->src1.value;
  if (!address) {
    return true;
  }

  if (VALUE_IS_CONSTANT(address)) {
    e.MovImm64(kScratch0, address->constant.u64);
    e.asm_().STR(kScratch0, X29, StackLayout::GUEST_CALL_RET_ADDR);
    return true;
  }

  if (!IsAlloc(address)) {
    return true;
  }

  e.asm_().STR(GR(address), X29, StackLayout::GUEST_CALL_RET_ADDR);
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
    case INT64_TYPE:
      e.asm_().LDR(GR(dest), ctx, (int32_t)offset);
      MaybeEmitTrackedContextAccess(e, offset, GR(dest), false);
      break;
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
  if (VALUE_IS_CONSTANT(src)) {
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
        e.asm_().STR(kScratch0, ctx, (int32_t)offset);
        MaybeEmitTrackedContextAccess(e, offset, kScratch0, true);
        if (ShouldTraceLoopSetupStore(e, i, offset)) {
          MaybeEmitTrackedLoopSetup(e, kScratch0);
        }
        break;
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
      case INT64_TYPE:
        e.asm_().STR(GR(src), ctx, (int32_t)offset);
        MaybeEmitTrackedContextAccess(e, offset, GR(src), true);
        if (ShouldTraceLoopSetupStore(e, i, offset)) {
          MaybeEmitTrackedLoopSetup(e, GR(src));
        }
        break;
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

  GReg host_addr =
      ComputeHostAddress(e, addr, nullptr, kScratch0, kScratch1);
  const bool byte_swap = (i->flags & LOAD_STORE_BYTE_SWAP) != 0;

  switch (dest->type) {
    case INT8_TYPE:  e.asm_().LDRB(GR(dest), host_addr); break;
    case INT16_TYPE: e.asm_().LDRH(GR(dest), host_addr); break;
    case INT32_TYPE: e.asm_().LDRw(GR(dest), host_addr); break;
    case INT64_TYPE: e.asm_().LDR(GR(dest), host_addr); break;
    case FLOAT32_TYPE:
      EmitLoadFloat32Memory(e, VR(dest), host_addr, byte_swap);
      return true;
    case FLOAT64_TYPE:
      EmitLoadFloat64Memory(e, VR(dest), host_addr, byte_swap);
      return true;
    case VEC128_TYPE:
      EmitLoadVector128Memory(e, VR(dest), host_addr, byte_swap);
      return true;
    default: return false;
  }

  // Byte swap if needed (Xbox 360 is big-endian)
  if (byte_swap) {
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

  GReg host_addr =
      ComputeHostAddress(e, addr, nullptr, kScratch0, kScratch1);
  const bool byte_swap = (i->flags & LOAD_STORE_BYTE_SWAP) != 0;

  if (VALUE_IS_CONSTANT(val)) {
    switch (val->type) {
      case INT8_TYPE:
        e.asm_().MOVZ(kScratch1, val->constant.u8);
        e.asm_().STRB(kScratch1, host_addr);
        return true;
      case INT16_TYPE:
        e.asm_().MOVZ(kScratch1, val->constant.u16);
        EmitStoreHalfWordMemory(e, host_addr, kScratch1, byte_swap);
        return true;
      case INT32_TYPE:
        e.MovImm64(kScratch1, val->constant.u32);
        EmitStoreWordMemory(e, host_addr, kScratch1, byte_swap);
        return true;
      case INT64_TYPE:
        e.MovImm64(kScratch1, val->constant.i64);
        EmitStoreDoubleWordMemory(e, host_addr, kScratch1, byte_swap);
        return true;
      case FLOAT32_TYPE:
        e.MovImm64(kScratch1, val->constant.u32);
        EmitStoreFloat32Memory(e, host_addr, kScratch1, byte_swap);
        return true;
      case FLOAT64_TYPE:
        e.MovImm64(kScratch1, val->constant.u64);
        EmitStoreFloat64Memory(e, host_addr, kScratch1, byte_swap);
        return true;
      case VEC128_TYPE:
        e.asm_().MOV(kScratch2, host_addr);
        e.LoadConstantV128(kVScratch0, val->constant.v128);
        EmitStoreVector128Memory(e, kScratch2, kVScratch0, byte_swap);
        return true;
      default:
        return false;
    }
  } else if (IsAlloc(val)) {
    switch (val->type) {
      case INT8_TYPE:
        e.asm_().STRB(GR(val), host_addr);
        return true;
      case INT16_TYPE:
        EmitStoreHalfWordMemory(e, host_addr, GR(val), byte_swap);
        return true;
      case INT32_TYPE:
        EmitStoreWordMemory(e, host_addr, GR(val), byte_swap);
        return true;
      case INT64_TYPE:
        EmitStoreDoubleWordMemory(e, host_addr, GR(val), byte_swap);
        return true;
      case FLOAT32_TYPE:
        e.asm_().FMOV_WS(kScratch1, VR(val));
        EmitStoreFloat32Memory(e, host_addr, kScratch1, byte_swap);
        return true;
      case FLOAT64_TYPE:
        e.asm_().FMOV_XD(kScratch1, VR(val));
        EmitStoreFloat64Memory(e, host_addr, kScratch1, byte_swap);
        return true;
      case VEC128_TYPE:
        EmitStoreVector128Memory(e, host_addr, VR(val), byte_swap);
        return true;
      default:
        return false;
    }
  }
  return true;
}

static bool EmitLoadMmio(A64Emitter& e, const Instr* i) {
  auto dest = i->dest;
  if (!dest || !IsAlloc(dest)) {
    return true;
  }

  e.MovImm64(X1, i->src1.offset);
  e.MovImm64(X2, static_cast<uint32_t>(i->src2.offset));
  e.MovImm64(X0, reinterpret_cast<uint64_t>(&A64LoadMmio));

  auto backend = static_cast<A64Backend*>(e.backend());
  e.MovImm64(kScratch0,
             reinterpret_cast<uint64_t>(backend->guest_to_host_thunk()));
  e.asm_().BLR(kScratch0);

  switch (dest->type) {
    case INT8_TYPE:
      e.asm_().UXTB(GR(dest), X0);
      break;
    case INT16_TYPE:
      e.asm_().UXTH(GR(dest), X0);
      break;
    case INT32_TYPE:
      e.asm_().MOVw(GR(dest), X0);
      break;
    case INT64_TYPE:
      e.asm_().MOV(GR(dest), X0);
      break;
    case FLOAT32_TYPE:
      e.asm_().FMOV_SW(VR(dest), X0);
      break;
    default:
      return false;
  }
  return true;
}

static bool EmitStoreMmio(A64Emitter& e, const Instr* i) {
  auto val = i->src3.value;
  if (!val) {
    return true;
  }

  e.MovImm64(X1, i->src1.offset);

  if (VALUE_IS_CONSTANT(val)) {
    switch (val->type) {
      case INT8_TYPE:
        e.MovImm64(X2, val->constant.u8);
        break;
      case INT16_TYPE:
        e.MovImm64(X2, val->constant.u16);
        break;
      case INT32_TYPE:
      case FLOAT32_TYPE:
        e.MovImm64(X2, val->constant.u32);
        break;
      case INT64_TYPE:
      case FLOAT64_TYPE:
        e.MovImm64(X2, static_cast<uint32_t>(val->constant.u64));
        break;
      default:
        return false;
    }
  } else {
    switch (val->type) {
      case INT8_TYPE:
      case INT16_TYPE:
      case INT32_TYPE:
      case INT64_TYPE:
        e.asm_().MOVw(X2, LoadGPR(e, val, X2));
        break;
      case FLOAT32_TYPE:
        if (!IsAlloc(val)) {
          return true;
        }
        e.asm_().FMOV_WS(X2, VR(val));
        break;
      case FLOAT64_TYPE:
        if (!IsAlloc(val)) {
          return true;
        }
        e.asm_().FMOV_XD(X2, VR(val));
        e.asm_().MOVw(X2, X2);
        break;
      default:
        return false;
    }
  }

  e.MovImm64(kScratch1, uint64_t(i->src2.offset) << 32);
  e.asm_().ORR(X2, X2, kScratch1);

  e.MovImm64(X0, reinterpret_cast<uint64_t>(&A64StoreMmio));
  auto backend = static_cast<A64Backend*>(e.backend());
  e.MovImm64(kScratch0,
             reinterpret_cast<uint64_t>(backend->guest_to_host_thunk()));
  e.asm_().BLR(kScratch0);
  return true;
}

static bool EmitLoadVectorSide(A64Emitter& e, const Instr* i,
                               uint64_t helper_address) {
  auto dest = i->dest;
  auto addr_val = i->src1.value;
  if (!dest || !IsAlloc(dest) || !addr_val) {
    return true;
  }

  GReg guest_addr = VALUE_IS_CONSTANT(addr_val) ? X1 : LoadGPR(e, addr_val, X1);
  if (VALUE_IS_CONSTANT(addr_val)) {
    e.MovImm64(X1, static_cast<uint32_t>(addr_val->constant.u64));
  } else if (guest_addr != X1) {
    e.asm_().MOVw(X1, guest_addr);
  } else {
    e.asm_().MOVw(X1, X1);
  }
  e.MovImm64(X2, 0);
  e.MovImm64(X0, helper_address);

  auto backend = static_cast<A64Backend*>(e.backend());
  e.MovImm64(kScratch0,
             reinterpret_cast<uint64_t>(backend->guest_to_host_thunk()));
  e.asm_().BLR(kScratch0);

  e.asm_().SUB(kScratch1, kContextReg,
               static_cast<uint32_t>(sizeof(A64BackendContext)));
  e.asm_().LDR_Q(VR(dest), kScratch1,
                 static_cast<int32_t>(
                     offsetof(A64BackendContext, helper_scratch)));
  return true;
}

static bool EmitStoreVectorSide(A64Emitter& e, const Instr* i,
                                uint64_t helper_address) {
  auto addr_val = i->src1.value;
  auto val = i->src2.value;
  if (!addr_val || !val) {
    return true;
  }

  e.asm_().SUB(kScratch1, kContextReg,
               static_cast<uint32_t>(sizeof(A64BackendContext)));
  int32_t scratch_offset =
      static_cast<int32_t>(offsetof(A64BackendContext, helper_scratch));
  if (VALUE_IS_CONSTANT(val)) {
    e.LoadConstantV128(kVScratch0, val->constant.v128);
    e.asm_().STR_Q(kVScratch0, kScratch1, scratch_offset);
  } else if (IsAlloc(val)) {
    e.asm_().STR_Q(VR(val), kScratch1, scratch_offset);
  } else {
    return true;
  }

  GReg guest_addr = VALUE_IS_CONSTANT(addr_val) ? X1 : LoadGPR(e, addr_val, X1);
  if (VALUE_IS_CONSTANT(addr_val)) {
    e.MovImm64(X1, static_cast<uint32_t>(addr_val->constant.u64));
  } else if (guest_addr != X1) {
    e.asm_().MOVw(X1, guest_addr);
  } else {
    e.asm_().MOVw(X1, X1);
  }
  e.MovImm64(X2, 0);
  e.MovImm64(X0, helper_address);

  auto backend = static_cast<A64Backend*>(e.backend());
  e.MovImm64(kScratch0,
             reinterpret_cast<uint64_t>(backend->guest_to_host_thunk()));
  e.asm_().BLR(kScratch0);
  return true;
}

static bool EmitLoadVectorLeft(A64Emitter& e, const Instr* i) {
  return EmitLoadVectorSide(e, i,
                            reinterpret_cast<uint64_t>(&A64LoadVectorLeft));
}

static bool EmitLoadVectorRight(A64Emitter& e, const Instr* i) {
  return EmitLoadVectorSide(e, i,
                            reinterpret_cast<uint64_t>(&A64LoadVectorRight));
}

static bool EmitStoreVectorLeft(A64Emitter& e, const Instr* i) {
  return EmitStoreVectorSide(e, i,
                             reinterpret_cast<uint64_t>(&A64StoreVectorLeft));
}

static bool EmitStoreVectorRight(A64Emitter& e, const Instr* i) {
  return EmitStoreVectorSide(e, i,
                             reinterpret_cast<uint64_t>(&A64StoreVectorRight));
}

static bool EmitLoadLocal(A64Emitter& e, const Instr* i) {
  auto dest = i->dest;
  auto slot = i->src1.value;
  if (!dest || !IsAlloc(dest) || !slot || !VALUE_IS_CONSTANT(slot)) {
    e.asm_().NOP();
    return true;
  }

  int32_t local_offset = static_cast<int32_t>(slot->constant.u32);
  GReg base = FrameBaseForOffset(e, local_offset, kScratch2, &local_offset);

  switch (dest->type) {
    case INT8_TYPE: e.asm_().LDRB(GR(dest), base, local_offset); break;
    case INT16_TYPE: e.asm_().LDRH(GR(dest), base, local_offset); break;
    case INT32_TYPE: e.asm_().LDRw(GR(dest), base, local_offset); break;
    case INT64_TYPE: e.asm_().LDR(GR(dest), base, local_offset); break;
    case FLOAT32_TYPE: e.asm_().LDR_S(VR(dest), base, local_offset); break;
    case FLOAT64_TYPE: e.asm_().LDR_D(VR(dest), base, local_offset); break;
    case VEC128_TYPE: e.asm_().LDR_Q(VR(dest), base, local_offset); break;
    default: return false;
  }
  return true;
}

static bool EmitStoreLocal(A64Emitter& e, const Instr* i) {
  auto slot = i->src1.value;
  auto val = i->src2.value;
  if (!slot || !VALUE_IS_CONSTANT(slot) || !val) {
    e.asm_().NOP();
    return true;
  }

  int32_t local_offset = static_cast<int32_t>(slot->constant.u32);
  GReg base = FrameBaseForOffset(e, local_offset, kScratch2, &local_offset);

  if (VALUE_IS_CONSTANT(val)) {
    switch (val->type) {
      case INT8_TYPE:
        e.asm_().MOVZ(kScratch0, val->constant.u8);
        e.asm_().STRB(kScratch0, base, local_offset);
        break;
      case INT16_TYPE:
        e.asm_().MOVZ(kScratch0, val->constant.u16);
        e.asm_().STRH(kScratch0, base, local_offset);
        break;
      case INT32_TYPE:
        e.MovImm64(kScratch0, val->constant.u32);
        e.asm_().STRw(kScratch0, base, local_offset);
        break;
      case INT64_TYPE:
        e.MovImm64(kScratch0, val->constant.u64);
        e.asm_().STR(kScratch0, base, local_offset);
        break;
      case FLOAT32_TYPE:
        e.MovImm64(kScratch0, val->constant.u32);
        e.asm_().STRw(kScratch0, base, local_offset);
        break;
      case FLOAT64_TYPE:
        e.MovImm64(kScratch0, val->constant.u64);
        e.asm_().STR(kScratch0, base, local_offset);
        break;
      case VEC128_TYPE:
        e.LoadConstantV128(kVScratch0, val->constant.v128);
        e.asm_().STR_Q(kVScratch0, base, local_offset);
        break;
      default:
        return false;
    }
    return true;
  }

  if (!IsAlloc(val)) {
    e.asm_().NOP();
    return true;
  }

  switch (val->type) {
    case INT8_TYPE: e.asm_().STRB(GR(val), base, local_offset); break;
    case INT16_TYPE: e.asm_().STRH(GR(val), base, local_offset); break;
    case INT32_TYPE: e.asm_().STRw(GR(val), base, local_offset); break;
    case INT64_TYPE: e.asm_().STR(GR(val), base, local_offset); break;
    case FLOAT32_TYPE: e.asm_().STR_S(VR(val), base, local_offset); break;
    case FLOAT64_TYPE: e.asm_().STR_D(VR(val), base, local_offset); break;
    case VEC128_TYPE: e.asm_().STR_Q(VR(val), base, local_offset); break;
    default: return false;
  }
  return true;
}

static bool EmitLoadOffset(A64Emitter& e, const Instr* i) {
  auto dest = i->dest;
  auto addr = i->src1.value;
  auto offset = i->src2.value;
  if (!dest || !IsAlloc(dest) || !addr || !offset) {
    e.asm_().NOP();
    return true;
  }

  GReg host_addr = ComputeHostAddress(e, addr, offset, kScratch0, kScratch1);
  const bool byte_swap = (i->flags & LOAD_STORE_BYTE_SWAP) != 0;

  switch (dest->type) {
    case INT8_TYPE:
      e.asm_().LDRB(GR(dest), host_addr);
      break;
    case INT16_TYPE:
      e.asm_().LDRH(GR(dest), host_addr);
      break;
    case INT32_TYPE:
      e.asm_().LDRw(GR(dest), host_addr);
      if (ShouldTraceCountFieldLoadOffset(e, i)) {
        MaybeEmitTrackedCountField(e, kTrackedCountFieldStageRawLoad,
                                   GR(dest));
      }
      break;
    case INT64_TYPE:
      e.asm_().LDR(GR(dest), host_addr);
      break;
    case FLOAT32_TYPE:
      EmitLoadFloat32Memory(e, VR(dest), host_addr, byte_swap);
      return true;
    case FLOAT64_TYPE:
      EmitLoadFloat64Memory(e, VR(dest), host_addr, byte_swap);
      return true;
    case VEC128_TYPE:
      EmitLoadVector128Memory(e, VR(dest), host_addr, byte_swap);
      return true;
    default:
      return false;
  }

  if (byte_swap) {
    switch (dest->type) {
      case INT16_TYPE:
        e.asm_().REV16(GR(dest), GR(dest));
        break;
      case INT32_TYPE:
        e.asm_().REVw(GR(dest), GR(dest));
        break;
      case INT64_TYPE:
        e.asm_().REV(GR(dest), GR(dest));
        break;
      default:
        break;
    }
  }
  return true;
}

static bool EmitStoreOffset(A64Emitter& e, const Instr* i) {
  auto addr = i->src1.value;
  auto offset = i->src2.value;
  auto val = i->src3.value;
  if (!addr || !offset || !val) {
    return true;
  }

  GReg host_addr = ComputeHostAddress(e, addr, offset, kScratch0, kScratch1);
  const bool byte_swap = (i->flags & LOAD_STORE_BYTE_SWAP) != 0;

  if (VALUE_IS_CONSTANT(val)) {
    switch (val->type) {
      case INT8_TYPE:
        e.asm_().MOVZ(kScratch1, val->constant.u8);
        e.asm_().STRB(kScratch1, host_addr);
        return true;
      case INT16_TYPE:
        e.asm_().MOVZ(kScratch1, val->constant.u16);
        EmitStoreHalfWordMemory(e, host_addr, kScratch1, byte_swap);
        return true;
      case INT32_TYPE:
        e.MovImm64(kScratch1, val->constant.u32);
        EmitStoreWordMemory(e, host_addr, kScratch1, byte_swap);
        return true;
      case INT64_TYPE:
        e.MovImm64(kScratch1, val->constant.i64);
        EmitStoreDoubleWordMemory(e, host_addr, kScratch1, byte_swap);
        return true;
      case FLOAT32_TYPE:
        e.MovImm64(kScratch1, val->constant.u32);
        EmitStoreFloat32Memory(e, host_addr, kScratch1, byte_swap);
        return true;
      case FLOAT64_TYPE:
        e.MovImm64(kScratch1, val->constant.u64);
        EmitStoreFloat64Memory(e, host_addr, kScratch1, byte_swap);
        return true;
      case VEC128_TYPE:
        e.asm_().MOV(kScratch2, host_addr);
        e.LoadConstantV128(kVScratch0, val->constant.v128);
        EmitStoreVector128Memory(e, kScratch2, kVScratch0, byte_swap);
        return true;
      default:
        return false;
    }
  } else if (IsAlloc(val)) {
    switch (val->type) {
      case INT8_TYPE:
        e.asm_().STRB(GR(val), host_addr);
        return true;
      case INT16_TYPE:
        EmitStoreHalfWordMemory(e, host_addr, GR(val), byte_swap);
        return true;
      case INT32_TYPE:
        EmitStoreWordMemory(e, host_addr, GR(val), byte_swap);
        return true;
      case INT64_TYPE:
        EmitStoreDoubleWordMemory(e, host_addr, GR(val), byte_swap);
        return true;
      case FLOAT32_TYPE:
        e.asm_().FMOV_WS(kScratch1, VR(val));
        EmitStoreFloat32Memory(e, host_addr, kScratch1, byte_swap);
        return true;
      case FLOAT64_TYPE:
        e.asm_().FMOV_XD(kScratch1, VR(val));
        EmitStoreFloat64Memory(e, host_addr, kScratch1, byte_swap);
        return true;
      case VEC128_TYPE:
        EmitStoreVector128Memory(e, host_addr, VR(val), byte_swap);
        return true;
      default:
        return false;
    }
  }
  return true;
}

// === Assign / Type Conversions ===

static bool EmitAssign(A64Emitter& e, const Instr* i) {
  if (!i->dest || !IsAlloc(i->dest)) return true;
  auto src = i->src1.value;
  if (!src) return true;

  if (src->type <= INT64_TYPE) {
    GReg rd = GR(i->dest);
    if (VALUE_IS_CONSTANT(src)) {
      e.MovImm64(rd, src->constant.i64);
    } else if (IsAlloc(src)) {
      e.asm_().MOV(rd, GR(src));
    }
  } else {
    VReg vd = VR(i->dest);
    if (VALUE_IS_CONSTANT(src)) {
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
    GReg rs = VALUE_IS_CONSTANT(src) ? kScratch0 : GR(src);
    if (VALUE_IS_CONSTANT(src)) e.MovImm64(rs, src->constant.i64);
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
  if (VALUE_IS_CONSTANT(src)) {
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
  if (VALUE_IS_CONSTANT(src)) {
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
  if (VALUE_IS_CONSTANT(src)) {
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

  if (VALUE_IS_CONSTANT(s1) && VALUE_IS_CONSTANT(s2)) {
    e.MovImm64(rd, s1->constant.i64 + s2->constant.i64);
  } else if (VALUE_IS_CONSTANT(s2) &&
             (s2->constant.u64 & 0xFFF) == s2->constant.u64) {
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

  if (VALUE_IS_CONSTANT(s1) && VALUE_IS_CONSTANT(s2)) {
    e.MovImm64(rd, s1->constant.i64 - s2->constant.i64);
  } else if (VALUE_IS_CONSTANT(s2) &&
             (s2->constant.u64 & 0xFFF) == s2->constant.u64) {
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

// Whole-vector shifts route through a C helper: the value goes out and comes
// back via helper_scratch, the bit count (0-7) via arg0.
static bool EmitVec128ShiftViaHelper(A64Emitter& e, const Instr* i,
                                     uint64_t helper_address) {
  auto val = i->src1.value;
  auto amt = i->src2.value;
  if (!i->dest || !IsAlloc(i->dest) || !val || !amt) return true;

  e.asm_().SUB(kScratch1, kContextReg,
               static_cast<uint32_t>(sizeof(A64BackendContext)));
  const int32_t scratch_offset =
      static_cast<int32_t>(offsetof(A64BackendContext, helper_scratch));
  if (VALUE_IS_CONSTANT(val)) {
    e.LoadConstantV128(kVScratch0, val->constant.v128);
    e.asm_().SUB(kScratch1, kContextReg,
                 static_cast<uint32_t>(sizeof(A64BackendContext)));
    e.asm_().STR_Q(kVScratch0, kScratch1, scratch_offset);
  } else if (IsAlloc(val)) {
    e.asm_().STR_Q(VR(val), kScratch1, scratch_offset);
  } else {
    return true;
  }

  GReg amt_reg = LoadGPR(e, amt, X1);
  if (amt_reg != X1) {
    e.asm_().MOV(X1, amt_reg);
  }
  e.MovImm64(X2, 0);
  e.MovImm64(X0, helper_address);
  auto backend = static_cast<A64Backend*>(e.backend());
  e.MovImm64(kScratch0,
             reinterpret_cast<uint64_t>(backend->guest_to_host_thunk()));
  e.asm_().BLR(kScratch0);

  e.asm_().SUB(kScratch1, kContextReg,
               static_cast<uint32_t>(sizeof(A64BackendContext)));
  e.asm_().LDR_Q(VR(i->dest), kScratch1, scratch_offset);
  return true;
}

// Shift semantics follow the x64 backend: the shift amount is masked to
// [0, 31] for 8/16/32-bit operands and [0, 63] for 64-bit ones. Sub-word
// results are kept zero-extended per the register convention.

static bool EmitShl(A64Emitter& e, const Instr* i) {
  if (!i->dest || !IsAlloc(i->dest)) return true;
  if (i->dest->type == VEC128_TYPE) {
    return EmitVec128ShiftViaHelper(e, i,
                                    reinterpret_cast<uint64_t>(&A64ShlV128));
  }
  GReg rd = GR(i->dest);
  GReg r1 = LoadGPR(e, i->src1.value, kScratch0);
  GReg r2 = LoadGPR(e, i->src2.value, kScratch1);
  switch (i->dest->type) {
    case INT8_TYPE:
      e.asm_().LSLVw(rd, r1, r2);
      e.asm_().UXTB(rd, rd);
      break;
    case INT16_TYPE:
      e.asm_().LSLVw(rd, r1, r2);
      e.asm_().UXTH(rd, rd);
      break;
    case INT32_TYPE:
      e.asm_().LSLVw(rd, r1, r2);
      break;
    default:
      e.asm_().LSLv(rd, r1, r2);
      break;
  }
  return true;
}

static bool EmitShr(A64Emitter& e, const Instr* i) {
  if (!i->dest || !IsAlloc(i->dest)) return true;
  if (i->dest->type == VEC128_TYPE) {
    return EmitVec128ShiftViaHelper(e, i,
                                    reinterpret_cast<uint64_t>(&A64ShrV128));
  }
  GReg rd = GR(i->dest);
  GReg r1 = LoadGPR(e, i->src1.value, kScratch0);
  GReg r2 = LoadGPR(e, i->src2.value, kScratch1);
  // Sub-word values are zero-extended, so a 32-bit logical shift right
  // cannot pull in bits beyond the operand width.
  if (i->dest->type <= INT32_TYPE) {
    e.asm_().LSRVw(rd, r1, r2);
  } else {
    e.asm_().LSRv(rd, r1, r2);
  }
  return true;
}

static bool EmitSha(A64Emitter& e, const Instr* i) {
  if (!i->dest || !IsAlloc(i->dest)) return true;
  GReg rd = GR(i->dest);
  GReg r1 = LoadGPR(e, i->src1.value, kScratch0);
  GReg r2 = LoadGPR(e, i->src2.value, kScratch1);
  switch (i->dest->type) {
    case INT8_TYPE:
      e.asm_().SXTB(kScratch2, r1);
      e.asm_().ASRVw(rd, kScratch2, r2);
      e.asm_().UXTB(rd, rd);
      break;
    case INT16_TYPE:
      e.asm_().SXTH(kScratch2, r1);
      e.asm_().ASRVw(rd, kScratch2, r2);
      e.asm_().UXTH(rd, rd);
      break;
    case INT32_TYPE:
      e.asm_().ASRVw(rd, r1, r2);
      break;
    default:
      e.asm_().ASRv(rd, r1, r2);
      break;
  }
  return true;
}

static bool EmitRotateLeft(A64Emitter& e, const Instr* i) {
  if (!i->dest || !IsAlloc(i->dest)) return true;
  GReg rd = GR(i->dest);
  GReg r1 = LoadGPR(e, i->src1.value, kScratch0);
  GReg r2 = LoadGPR(e, i->src2.value, kScratch1);
  switch (i->dest->type) {
    case INT8_TYPE:
    case INT16_TYPE: {
      // No hardware sub-word rotate: (x << n) | (x >> (width - n)),
      // with n taken mod width. x is zero-extended per convention.
      const uint32_t width = i->dest->type == INT8_TYPE ? 8 : 16;
      e.asm_().MOVZ(kScratch2, width - 1);
      e.asm_().ANDw(kScratch2, r2, kScratch2);  // n %= width
      e.asm_().LSLVw(kScratch1, r1, kScratch2);  // r2 dead from here on
      e.asm_().NEGw(kScratch2, kScratch2);
      e.asm_().ADD_immw(kScratch2, kScratch2, width);  // width - n
      e.asm_().LSRVw(kScratch2, r1, kScratch2);  // n==0 shifts by width -> 0
      e.asm_().ORRw(rd, kScratch1, kScratch2);
      if (i->dest->type == INT8_TYPE) {
        e.asm_().UXTB(rd, rd);
      } else {
        e.asm_().UXTH(rd, rd);
      }
      break;
    }
    case INT32_TYPE:
      // ROL32(x, n) = ROR32(x, 32-n); RORVw masks the amount mod 32.
      e.asm_().NEGw(kScratch2, r2);
      e.asm_().RORVw(rd, r1, kScratch2);
      break;
    default:
      // ROL64(x, n) = ROR64(x, 64-n)
      e.asm_().NEG(kScratch2, r2);
      e.asm_().RORv(rd, r1, kScratch2);
      break;
  }
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
  if (i->src1.value && i->src1.value->type == VEC128_TYPE) {
    // v128 byte swap reverses bytes within each 32-bit word.
    if (VALUE_IS_CONSTANT(i->src1.value)) {
      e.LoadConstantV128(kVScratch0, i->src1.value->constant.v128);
      e.asm_().REV32_16B(VR(i->dest), kVScratch0);
    } else {
      e.asm_().REV32_16B(VR(i->dest), VR(i->src1.value));
    }
    return true;
  }
  GReg rd = GR(i->dest);
  GReg rs = LoadGPR(e, i->src1.value, kScratch0);
  switch (i->src1.value->type) {
    case INT16_TYPE: e.asm_().REV16(rd, rs); break;
    case INT32_TYPE:
      e.asm_().REVw(rd, rs);
      if (ShouldTraceCountFieldByteSwap(e, i)) {
        MaybeEmitTrackedCountField(e, kTrackedCountFieldStageByteSwap, rd);
      }
      break;
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
  if (!i->dest || !IsAlloc(i->dest)) return true;

  auto s1 = i->src1.value;
  auto s2 = i->src2.value;
  auto carry_in = i->src3.value;
  GReg rd = GR(i->dest);
  GReg r1 = LoadGPR(e, s1, kScratch0);
  GReg r2 = LoadGPR(e, s2, kScratch1);

  uint32_t carry_imm = 0;
  GReg carry_reg = kScratch2;
  if (carry_in) {
    if (VALUE_IS_CONSTANT(carry_in)) {
      carry_imm = carry_in->constant.u64 ? 1u : 0u;
    } else {
      GReg carry_src = LoadGPR(e, carry_in, kScratch2);
      if (carry_in->type <= INT32_TYPE) {
        e.asm_().CMPw(carry_src, XZR);
      } else {
        e.asm_().CMP(carry_src, XZR);
      }
      e.asm_().CSET(carry_reg, NE);
    }
  }

  if (i->dest->type <= INT32_TYPE) {
    e.asm_().ADDw(rd, r1, r2);
    if (carry_imm) {
      e.asm_().ADDw(rd, rd, carry_imm);
    } else if (carry_in) {
      e.asm_().ADDw(rd, rd, carry_reg);
    }

    if (i->dest->type == INT8_TYPE) {
      e.asm_().UXTB(rd, rd);
    } else if (i->dest->type == INT16_TYPE) {
      e.asm_().UXTH(rd, rd);
    }
  } else {
    e.asm_().ADD(rd, r1, r2);
    if (carry_imm) {
      e.asm_().ADD(rd, rd, carry_imm);
    } else if (carry_in) {
      e.asm_().ADD(rd, rd, carry_reg);
    }
  }

  return true;
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
  // src1 = PPC FPSCR[RN]: 0=nearest, 1=toward zero, 2=+inf, 3=-inf.
  // ARM FPCR RMode (bits [23:22]): 0=nearest, 1=+inf, 2=-inf, 3=zero.
  // The mapping (0->0, 1->3, 2->1, 3->2) is packed into 0x9C, two bits per
  // guest mode.
  auto src = i->src1.value;
  if (!src) return true;
  GReg mode = LoadGPR(e, src, kScratch0);
  e.asm_().MOVZ(kScratch2, 3);
  e.asm_().ANDw(kScratch1, mode, kScratch2);      // g = mode & 3
  e.asm_().LSL_imm(kScratch1, kScratch1, 1);      // g * 2
  e.asm_().MOVZ(kScratch0, 0x9C);                 // mode no longer needed
  e.asm_().LSRv(kScratch0, kScratch0, kScratch1); // table >> (g*2)
  e.asm_().ANDw(kScratch0, kScratch0, kScratch2); // ARM RMode
  e.asm_().LSL_imm(kScratch0, kScratch0, 22);
  e.asm_().MRS_FPCR(kScratch1);
  e.MovImm64(kScratch2, ~(3ULL << 22));
  e.asm_().AND(kScratch1, kScratch1, kScratch2);
  e.asm_().ORR(kScratch1, kScratch1, kScratch0);
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

static bool EmitVec128UnaryViaHelper(A64Emitter& e, const Instr* i,
                                     const Value* src, uint64_t helper_address,
                                     uint64_t arg0);

// POW2/LOG2 route through C helpers (std::exp2 / std::log2), matching the
// x64 backend's emulation path. The shape selector (arg0) picks f32 lane 0,
// f64 lane 0, or all four f32 lanes.
static uint64_t Pow2Log2Shape(TypeName type) {
  switch (type) {
    case FLOAT32_TYPE: return 0;
    case FLOAT64_TYPE: return 1;
    default: return 2;
  }
}

static bool EmitPow2(A64Emitter& e, const Instr* i) {
  if (!i->dest || !IsAlloc(i->dest)) return true;
  auto src = i->src1.value;
  if (!src) { e.asm_().NOP(); return true; }
  return EmitVec128UnaryViaHelper(e, i, src,
                                  reinterpret_cast<uint64_t>(&A64Pow2),
                                  Pow2Log2Shape(i->dest->type));
}

static bool EmitLog2(A64Emitter& e, const Instr* i) {
  if (!i->dest || !IsAlloc(i->dest)) return true;
  auto src = i->src1.value;
  if (!src) { e.asm_().NOP(); return true; }
  return EmitVec128UnaryViaHelper(e, i, src,
                                  reinterpret_cast<uint64_t>(&A64Log2),
                                  Pow2Log2Shape(i->dest->type));
}

static bool EmitDotProduct3(A64Emitter& e, const Instr* i) {
  if (!i->dest || !IsAlloc(i->dest)) return true;
  auto s1 = i->src1.value, s2 = i->src2.value;
  if (!IsAlloc(s1) || !IsAlloc(s2)) { e.asm_().NOP(); return true; }
  VReg vd = VR(i->dest);
  // dp3 = s1[0]*s2[0] + s1[1]*s2[1] + s1[2]*s2[2]
  e.asm_().FMUL_4S(kVScratch0, VR(s1), VR(s2));  // element-wise multiply
  // Zero out lane 3 before summing
  e.asm_().MOVI_4S_zero(kVScratch1);
  e.asm_().INS_S(kVScratch0, 3, kVScratch1, 0);  // zero lane 3
  // Horizontal pairwise add: [a+b, c+0, ...] then [a+b+c+0, ...]
  e.asm_().FADD_4S(kVScratch1, kVScratch0, kVScratch0);  // won't work for hadd
  // Use EXT + FADD pattern for horizontal sum
  e.asm_().EXT_16B(kVScratch1, kVScratch0, kVScratch0, 4);
  e.asm_().FADD_4S(kVScratch0, kVScratch0, kVScratch1);
  e.asm_().EXT_16B(kVScratch1, kVScratch0, kVScratch0, 8);
  e.asm_().FADD_4S(kVScratch0, kVScratch0, kVScratch1);
  // Splat result to all lanes
  e.asm_().DUP_S(vd, kVScratch0, 0);
  return true;
}

static bool EmitDotProduct4(A64Emitter& e, const Instr* i) {
  if (!i->dest || !IsAlloc(i->dest)) return true;
  auto s1 = i->src1.value, s2 = i->src2.value;
  if (!IsAlloc(s1) || !IsAlloc(s2)) { e.asm_().NOP(); return true; }
  VReg vd = VR(i->dest);
  // dp4 = sum of element-wise products
  e.asm_().FMUL_4S(kVScratch0, VR(s1), VR(s2));
  // Horizontal sum via EXT+FADD
  e.asm_().EXT_16B(kVScratch1, kVScratch0, kVScratch0, 4);
  e.asm_().FADD_4S(kVScratch0, kVScratch0, kVScratch1);
  e.asm_().EXT_16B(kVScratch1, kVScratch0, kVScratch0, 8);
  e.asm_().FADD_4S(kVScratch0, kVScratch0, kVScratch1);
  e.asm_().DUP_S(vd, kVScratch0, 0);
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
  if (VALUE_IS_CONSTANT(src)) {
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

// INSERT/EXTRACT indices are guest-order element numbers; within each
// 32-bit word the guest byte order is reversed relative to the host, so
// byte lanes swizzle with ^3 and halfword lanes with ^1 (words map 1:1).

// Computes the host byte offset of a guest element into `out` (clobbers
// kScratch2). `idx` is the guest index register.
static void EmitElementOffset(A64Emitter& e, GReg out, GReg idx,
                              TypeName part_type) {
  switch (part_type) {
    case INT8_TYPE:
      e.asm_().MOVZ(out, 0xF);
      e.asm_().ANDw(out, idx, out);
      e.asm_().MOVZ(kScratch2, 3);
      e.asm_().EORw(out, out, kScratch2);
      break;
    case INT16_TYPE:
      e.asm_().MOVZ(out, 0x7);
      e.asm_().ANDw(out, idx, out);
      e.asm_().MOVZ(kScratch2, 1);
      e.asm_().EORw(out, out, kScratch2);
      e.asm_().LSL_imm(out, out, 1);
      break;
    default:  // INT32/FLOAT32
      e.asm_().MOVZ(out, 0x3);
      e.asm_().ANDw(out, idx, out);
      e.asm_().LSL_imm(out, out, 2);
      break;
  }
}

static bool EmitInsert(A64Emitter& e, const Instr* i) {
  if (!i->dest || !IsAlloc(i->dest)) return true;
  auto vec = i->src1.value;
  auto idx_val = i->src2.value;
  auto part = i->src3.value;
  if (!vec || !idx_val || !part) return true;

  VReg vd = VR(i->dest);
  if (VALUE_IS_CONSTANT(vec)) {
    e.LoadConstantV128(vd, vec->constant.v128);
  } else if (IsAlloc(vec) && vd != VR(vec)) {
    e.asm_().MOV_16B(vd, VR(vec));
  }

  if (VALUE_IS_CONSTANT(idx_val)) {
    const uint32_t idx = (uint32_t)idx_val->constant.u8;
    GReg part_reg;
    if (VALUE_IS_CONSTANT(part)) {
      e.MovImm64(kScratch0, part->constant.i64);
      part_reg = kScratch0;
    } else if (IsAlloc(part) && part->type <= INT64_TYPE) {
      part_reg = GR(part);
    } else if (IsAlloc(part)) {
      e.asm_().INS_S(vd, idx & 3, VR(part), 0);
      return true;
    } else {
      return true;
    }
    switch (part->type) {
      case INT8_TYPE:
        e.asm_().INS_B_GPR(vd, (idx ^ 3) & 15, part_reg);
        break;
      case INT16_TYPE:
        e.asm_().INS_H_GPR(vd, (idx ^ 1) & 7, part_reg);
        break;
      default:
        e.asm_().INS_S_GPR(vd, idx & 3, part_reg);
        break;
    }
    return true;
  }

  // Dynamic index: spill, patch the element in memory, reload.
  e.asm_().STR_Q(vd, X29, StackLayout::GUEST_SCRATCH);
  GReg idx = LoadGPR(e, idx_val, kScratch0);
  EmitElementOffset(e, kScratch1, idx, part->type);
  e.asm_().ADD(kScratch1, X29, kScratch1);
  GReg part_reg;
  if (VALUE_IS_CONSTANT(part)) {
    e.MovImm64(kScratch2, part->constant.i64);
    part_reg = kScratch2;
  } else if (IsAlloc(part) && part->type <= INT64_TYPE) {
    part_reg = GR(part);
  } else {
    return true;
  }
  switch (part->type) {
    case INT8_TYPE:
      e.asm_().STRB(part_reg, kScratch1, StackLayout::GUEST_SCRATCH);
      break;
    case INT16_TYPE:
      e.asm_().STRH(part_reg, kScratch1, StackLayout::GUEST_SCRATCH);
      break;
    default:
      e.asm_().STRw(part_reg, kScratch1, StackLayout::GUEST_SCRATCH);
      break;
  }
  e.asm_().LDR_Q(vd, X29, StackLayout::GUEST_SCRATCH);
  return true;
}

static bool EmitExtract(A64Emitter& e, const Instr* i) {
  if (!i->dest || !IsAlloc(i->dest)) return true;
  auto vec = i->src1.value;
  auto idx_val = i->src2.value;
  if (!vec || !idx_val) return true;
  VReg vs = LoadVec128(e, vec, kVScratch0);

  if (VALUE_IS_CONSTANT(idx_val)) {
    const uint32_t idx = (uint32_t)idx_val->constant.u8;
    switch (i->dest->type) {
      case INT8_TYPE:
        e.asm_().UMOV_B(GR(i->dest), vs, (idx ^ 3) & 15);
        return true;
      case INT16_TYPE:
        e.asm_().UMOV_H(GR(i->dest), vs, (idx ^ 1) & 7);
        return true;
      case INT32_TYPE:
      case INT64_TYPE:
        e.asm_().UMOV_W(GR(i->dest), vs, idx & 3);
        return true;
      case FLOAT32_TYPE:
        e.asm_().DUP_S(VR(i->dest), vs, idx & 3);
        return true;
      default:
        return false;
    }
  }

  // Dynamic index: spill and load the element from memory.
  e.asm_().STR_Q(vs, X29, StackLayout::GUEST_SCRATCH);
  GReg idx = LoadGPR(e, idx_val, kScratch0);
  EmitElementOffset(e, kScratch1, idx, i->dest->type);
  e.asm_().ADD(kScratch1, X29, kScratch1);
  switch (i->dest->type) {
    case INT8_TYPE:
      e.asm_().LDRB(GR(i->dest), kScratch1, StackLayout::GUEST_SCRATCH);
      return true;
    case INT16_TYPE:
      e.asm_().LDRH(GR(i->dest), kScratch1, StackLayout::GUEST_SCRATCH);
      return true;
    case INT32_TYPE:
    case INT64_TYPE:
      e.asm_().LDRw(GR(i->dest), kScratch1, StackLayout::GUEST_SCRATCH);
      return true;
    default:
      return false;
  }
}

// VECTOR_ADD/SUB/AVERAGE pack flags as part_type | (arithmetic_flags << 8);
// VECTOR_MAX/MIN pack them the other way around (see hir_builder.cc).
static bool EmitVectorAdd(A64Emitter& e, const Instr* i) {
  if (!i->dest || !IsAlloc(i->dest)) return true;
  auto s1 = i->src1.value, s2 = i->src2.value;
  if (!s1 || !s2) { e.asm_().NOP(); return true; }
  VReg v1 = LoadVec128(e, s1, kVScratch0);
  VReg v2 = LoadVec128(e, s2, kVScratch1);
  VReg vd = VR(i->dest);
  const TypeName part_type = static_cast<TypeName>(i->flags & 0xFF);
  const uint32_t arith = i->flags >> 8;
  const bool is_unsigned = !!(arith & ARITHMETIC_UNSIGNED);
  const bool saturate = !!(arith & ARITHMETIC_SATURATE);
  switch (part_type) {
    case INT8_TYPE:
      if (saturate) {
        if (is_unsigned) e.asm_().UQADD_16B(vd, v1, v2);
        else e.asm_().SQADD_16B(vd, v1, v2);
      } else {
        e.asm_().ADD_16B(vd, v1, v2);
      }
      return true;
    case INT16_TYPE:
      if (saturate) {
        if (is_unsigned) e.asm_().UQADD_8H(vd, v1, v2);
        else e.asm_().SQADD_8H(vd, v1, v2);
      } else {
        e.asm_().ADD_8H(vd, v1, v2);
      }
      return true;
    case INT32_TYPE:
      if (saturate) {
        if (is_unsigned) e.asm_().UQADD_4S(vd, v1, v2);
        else e.asm_().SQADD_4S(vd, v1, v2);
      } else {
        e.asm_().ADD_4S(vd, v1, v2);
      }
      return true;
    case FLOAT32_TYPE:
      e.asm_().FADD_4S(vd, v1, v2);
      return true;
    default:
      return false;
  }
}

static bool EmitVectorSub(A64Emitter& e, const Instr* i) {
  if (!i->dest || !IsAlloc(i->dest)) return true;
  auto s1 = i->src1.value, s2 = i->src2.value;
  if (!s1 || !s2) { e.asm_().NOP(); return true; }
  VReg v1 = LoadVec128(e, s1, kVScratch0);
  VReg v2 = LoadVec128(e, s2, kVScratch1);
  VReg vd = VR(i->dest);
  const TypeName part_type = static_cast<TypeName>(i->flags & 0xFF);
  const uint32_t arith = i->flags >> 8;
  const bool is_unsigned = !!(arith & ARITHMETIC_UNSIGNED);
  const bool saturate = !!(arith & ARITHMETIC_SATURATE);
  switch (part_type) {
    case INT8_TYPE:
      if (saturate) {
        if (is_unsigned) e.asm_().UQSUB_16B(vd, v1, v2);
        else e.asm_().SQSUB_16B(vd, v1, v2);
      } else {
        e.asm_().SUB_16B(vd, v1, v2);
      }
      return true;
    case INT16_TYPE:
      if (saturate) {
        if (is_unsigned) e.asm_().UQSUB_8H(vd, v1, v2);
        else e.asm_().SQSUB_8H(vd, v1, v2);
      } else {
        e.asm_().SUB_8H(vd, v1, v2);
      }
      return true;
    case INT32_TYPE:
      if (saturate) {
        if (is_unsigned) e.asm_().UQSUB_4S(vd, v1, v2);
        else e.asm_().SQSUB_4S(vd, v1, v2);
      } else {
        e.asm_().SUB_4S(vd, v1, v2);
      }
      return true;
    case FLOAT32_TYPE:
      e.asm_().FSUB_4S(vd, v1, v2);
      return true;
    default:
      return false;
  }
}

// VECTOR_COMPARE_* flags are the bare part type.
enum class VectorCmp { EQ, SGT, SGE, UGT, UGE };

static bool EmitVectorCompareCommon(A64Emitter& e, const Instr* i,
                                    VectorCmp cmp) {
  if (!i->dest || !IsAlloc(i->dest)) return true;
  auto s1 = i->src1.value, s2 = i->src2.value;
  if (!s1 || !s2) { e.asm_().NOP(); return true; }
  VReg v1 = LoadVec128(e, s1, kVScratch0);
  VReg v2 = LoadVec128(e, s2, kVScratch1);
  VReg vd = VR(i->dest);
  switch (static_cast<TypeName>(i->flags)) {
    case INT8_TYPE:
      switch (cmp) {
        case VectorCmp::EQ:  e.asm_().CMEQ_16B(vd, v1, v2); return true;
        case VectorCmp::SGT: e.asm_().CMGT_16B(vd, v1, v2); return true;
        case VectorCmp::SGE: e.asm_().CMGE_16B(vd, v1, v2); return true;
        case VectorCmp::UGT: e.asm_().CMHI_16B(vd, v1, v2); return true;
        case VectorCmp::UGE: e.asm_().CMHS_16B(vd, v1, v2); return true;
      }
      return false;
    case INT16_TYPE:
      switch (cmp) {
        case VectorCmp::EQ:  e.asm_().CMEQ_8H(vd, v1, v2); return true;
        case VectorCmp::SGT: e.asm_().CMGT_8H(vd, v1, v2); return true;
        case VectorCmp::SGE: e.asm_().CMGE_8H(vd, v1, v2); return true;
        case VectorCmp::UGT: e.asm_().CMHI_8H(vd, v1, v2); return true;
        case VectorCmp::UGE: e.asm_().CMHS_8H(vd, v1, v2); return true;
      }
      return false;
    case INT32_TYPE:
      switch (cmp) {
        case VectorCmp::EQ:  e.asm_().CMEQ_4S(vd, v1, v2); return true;
        case VectorCmp::SGT: e.asm_().CMGT_4S(vd, v1, v2); return true;
        case VectorCmp::SGE: e.asm_().CMGE_4S(vd, v1, v2); return true;
        case VectorCmp::UGT: e.asm_().CMHI_4S(vd, v1, v2); return true;
        case VectorCmp::UGE: e.asm_().CMHS_4S(vd, v1, v2); return true;
      }
      return false;
    case FLOAT32_TYPE:
      switch (cmp) {
        case VectorCmp::EQ:  e.asm_().FCMEQ_4S(vd, v1, v2); return true;
        case VectorCmp::SGT: e.asm_().FCMGT_4S(vd, v1, v2); return true;
        case VectorCmp::SGE: e.asm_().FCMGE_4S(vd, v1, v2); return true;
        default: return false;  // no unsigned float compares
      }
    default:
      return false;
  }
}

static bool EmitVectorCompareEQ(A64Emitter& e, const Instr* i) {
  return EmitVectorCompareCommon(e, i, VectorCmp::EQ);
}
static bool EmitVectorCompareSGT(A64Emitter& e, const Instr* i) {
  return EmitVectorCompareCommon(e, i, VectorCmp::SGT);
}
static bool EmitVectorCompareSGE(A64Emitter& e, const Instr* i) {
  return EmitVectorCompareCommon(e, i, VectorCmp::SGE);
}
static bool EmitVectorCompareUGT(A64Emitter& e, const Instr* i) {
  return EmitVectorCompareCommon(e, i, VectorCmp::UGT);
}
static bool EmitVectorCompareUGE(A64Emitter& e, const Instr* i) {
  return EmitVectorCompareCommon(e, i, VectorCmp::UGE);
}

static bool EmitVectorMinMaxCommon(A64Emitter& e, const Instr* i, bool is_max) {
  if (!i->dest || !IsAlloc(i->dest)) return true;
  auto s1 = i->src1.value, s2 = i->src2.value;
  if (!s1 || !s2) { e.asm_().NOP(); return true; }
  VReg v1 = LoadVec128(e, s1, kVScratch0);
  VReg v2 = LoadVec128(e, s2, kVScratch1);
  VReg vd = VR(i->dest);
  // VECTOR_MAX/MIN: flags = arithmetic_flags | (part_type << 8)
  const TypeName part_type = static_cast<TypeName>(i->flags >> 8);
  const bool is_unsigned = !!(i->flags & ARITHMETIC_UNSIGNED);
  switch (part_type) {
    case INT8_TYPE:
      if (is_unsigned) {
        if (is_max) e.asm_().UMAX_16B(vd, v1, v2);
        else e.asm_().UMIN_16B(vd, v1, v2);
      } else {
        if (is_max) e.asm_().SMAX_16B(vd, v1, v2);
        else e.asm_().SMIN_16B(vd, v1, v2);
      }
      return true;
    case INT16_TYPE:
      if (is_unsigned) {
        if (is_max) e.asm_().UMAX_8H(vd, v1, v2);
        else e.asm_().UMIN_8H(vd, v1, v2);
      } else {
        if (is_max) e.asm_().SMAX_8H(vd, v1, v2);
        else e.asm_().SMIN_8H(vd, v1, v2);
      }
      return true;
    case INT32_TYPE:
      if (is_unsigned) {
        if (is_max) e.asm_().UMAX_4S(vd, v1, v2);
        else e.asm_().UMIN_4S(vd, v1, v2);
      } else {
        if (is_max) e.asm_().SMAX_4S(vd, v1, v2);
        else e.asm_().SMIN_4S(vd, v1, v2);
      }
      return true;
    case FLOAT32_TYPE:
      if (is_max) e.asm_().FMAX_4S(vd, v1, v2);
      else e.asm_().FMIN_4S(vd, v1, v2);
      return true;
    default:
      return false;
  }
}

static bool EmitVectorMax(A64Emitter& e, const Instr* i) {
  return EmitVectorMinMaxCommon(e, i, true);
}

static bool EmitVectorMin(A64Emitter& e, const Instr* i) {
  return EmitVectorMinMaxCommon(e, i, false);
}

// Per-element shifts. PPC (vslb/vslh/vslw etc.) masks each lane's shift
// amount to the element width; NEON *SHL produces 0 for out-of-range
// amounts, so the mask is required. Negative amounts shift right.
static bool EmitVectorShiftCommon(A64Emitter& e, const Instr* i, bool right,
                                  bool arithmetic) {
  if (!i->dest || !IsAlloc(i->dest)) return true;
  auto s1 = i->src1.value, s2 = i->src2.value;
  if (!s1 || !s2) { e.asm_().NOP(); return true; }
  VReg v1 = LoadVec128(e, s1, kVScratch0);
  VReg v2 = LoadVec128(e, s2, kVScratch1);
  VReg vd = VR(i->dest);
  switch (i->flags) {
    case INT8_TYPE:
      e.asm_().MOVI_16B(kVScratch2, 7);
      e.asm_().AND_16B(kVScratch2, v2, kVScratch2);
      if (right) e.asm_().NEG_16B(kVScratch2, kVScratch2);
      if (arithmetic) e.asm_().SSHL_16B(vd, v1, kVScratch2);
      else e.asm_().USHL_16B(vd, v1, kVScratch2);
      return true;
    case INT16_TYPE:
      e.asm_().MOVI_8H(kVScratch2, 15);
      e.asm_().AND_16B(kVScratch2, v2, kVScratch2);
      if (right) e.asm_().NEG_8H(kVScratch2, kVScratch2);
      if (arithmetic) e.asm_().SSHL_8H(vd, v1, kVScratch2);
      else e.asm_().USHL_8H(vd, v1, kVScratch2);
      return true;
    case INT32_TYPE:
      e.asm_().MOVI_4S(kVScratch2, 31);
      e.asm_().AND_16B(kVScratch2, v2, kVScratch2);
      if (right) e.asm_().NEG_4S(kVScratch2, kVScratch2);
      if (arithmetic) e.asm_().SSHL_4S(vd, v1, kVScratch2);
      else e.asm_().USHL_4S(vd, v1, kVScratch2);
      return true;
    default:
      return false;
  }
}

static bool EmitVectorShl(A64Emitter& e, const Instr* i) {
  return EmitVectorShiftCommon(e, i, false, false);
}

static bool EmitVectorShr(A64Emitter& e, const Instr* i) {
  return EmitVectorShiftCommon(e, i, true, false);
}

static bool EmitVectorSha(A64Emitter& e, const Instr* i) {
  return EmitVectorShiftCommon(e, i, true, true);
}

static bool EmitVectorRotateLeft(A64Emitter& e, const Instr* i) {
  if (!i->dest || !IsAlloc(i->dest)) return true;
  auto s1 = i->src1.value, s2 = i->src2.value;
  if (!s1 || !s2) { e.asm_().NOP(); return true; }
  VReg v1 = LoadVec128(e, s1, kVScratch0);
  VReg v2 = LoadVec128(e, s2, kVScratch1);
  VReg vd = VR(i->dest);
  // ROL(x, n) = (x << n) | (x >> (width - n)), n taken mod width.
  // The right shift is a USHL by (n - width), which is 0 for n == 0.
  // NOT(width-1 splat) is a splat of -width, so n - width comes from an ADD.
  switch (i->flags) {
    case INT8_TYPE:
      e.asm_().MOVI_16B(kVScratch2, 7);
      e.asm_().AND_16B(kVScratch1, v2, kVScratch2);   // n (v2 dead after)
      e.asm_().NOT_16B(kVScratch2, kVScratch2);       // -8 splat
      e.asm_().ADD_16B(kVScratch2, kVScratch1, kVScratch2);  // n - 8
      e.asm_().USHL_16B(kVScratch1, v1, kVScratch1);  // x << n
      e.asm_().USHL_16B(kVScratch2, v1, kVScratch2);  // x >> (8 - n)
      e.asm_().ORR_16B(vd, kVScratch1, kVScratch2);
      return true;
    case INT16_TYPE:
      e.asm_().MOVI_8H(kVScratch2, 15);
      e.asm_().AND_16B(kVScratch1, v2, kVScratch2);
      e.asm_().NOT_16B(kVScratch2, kVScratch2);
      e.asm_().ADD_8H(kVScratch2, kVScratch1, kVScratch2);
      e.asm_().USHL_8H(kVScratch1, v1, kVScratch1);
      e.asm_().USHL_8H(kVScratch2, v1, kVScratch2);
      e.asm_().ORR_16B(vd, kVScratch1, kVScratch2);
      return true;
    case INT32_TYPE:
      e.asm_().MOVI_4S(kVScratch2, 31);
      e.asm_().AND_16B(kVScratch1, v2, kVScratch2);
      e.asm_().NOT_16B(kVScratch2, kVScratch2);
      e.asm_().ADD_4S(kVScratch2, kVScratch1, kVScratch2);
      e.asm_().USHL_4S(kVScratch1, v1, kVScratch1);
      e.asm_().USHL_4S(kVScratch2, v1, kVScratch2);
      e.asm_().ORR_16B(vd, kVScratch1, kVScratch2);
      return true;
    default:
      return false;
  }
}

static bool EmitVectorAverage(A64Emitter& e, const Instr* i) {
  if (!i->dest || !IsAlloc(i->dest)) return true;
  auto s1 = i->src1.value, s2 = i->src2.value;
  if (!s1 || !s2) { e.asm_().NOP(); return true; }
  VReg v1 = LoadVec128(e, s1, kVScratch0);
  VReg v2 = LoadVec128(e, s2, kVScratch1);
  VReg vd = VR(i->dest);
  // Same flag packing as VECTOR_ADD.
  const TypeName part_type = static_cast<TypeName>(i->flags & 0xFF);
  const bool is_unsigned = !!((i->flags >> 8) & ARITHMETIC_UNSIGNED);
  switch (part_type) {
    case INT8_TYPE:
      if (is_unsigned) e.asm_().URHADD_16B(vd, v1, v2);
      else e.asm_().SRHADD_16B(vd, v1, v2);
      return true;
    case INT16_TYPE:
      if (is_unsigned) e.asm_().URHADD_8H(vd, v1, v2);
      else e.asm_().SRHADD_8H(vd, v1, v2);
      return true;
    case INT32_TYPE:
      if (is_unsigned) e.asm_().URHADD_4S(vd, v1, v2);
      else e.asm_().SRHADD_4S(vd, v1, v2);
      return true;
    default:
      return false;
  }
}

static const vec128_t kLoadVectorShlTable[16] = {
    vec128b(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15),
    vec128b(1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16),
    vec128b(2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17),
    vec128b(3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18),
    vec128b(4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19),
    vec128b(5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20),
    vec128b(6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21),
    vec128b(7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22),
    vec128b(8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23),
    vec128b(9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24),
    vec128b(10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24,
            25),
    vec128b(11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25,
            26),
    vec128b(12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26,
            27),
    vec128b(13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27,
            28),
    vec128b(14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28,
            29),
    vec128b(15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29,
            30),
};

static const vec128_t kLoadVectorShrTable[16] = {
    vec128b(16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31),
    vec128b(15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30),
    vec128b(14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29),
    vec128b(13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28),
    vec128b(12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27),
    vec128b(11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26),
    vec128b(10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25),
    vec128b(9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24),
    vec128b(8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23),
    vec128b(7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22),
    vec128b(6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21),
    vec128b(5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20),
    vec128b(4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19),
    vec128b(3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18),
    vec128b(2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17),
    vec128b(1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16),
};

template <size_t TableSize>
static bool EmitLoadVectorTable(A64Emitter& e, const Instr* i,
                                const vec128_t (&table)[TableSize]) {
  static_assert(TableSize == 16, "load-vector tables must have 16 entries");
  auto dest = i->dest;
  auto sh = i->src1.value;
  if (!dest || !IsAlloc(dest)) {
    return true;
  }

  if (!sh) {
    e.LoadConstantV128(VR(dest), table[0]);
    return true;
  }

  if (VALUE_IS_CONSTANT(sh)) {
    e.LoadConstantV128(VR(dest), table[sh->constant.u8 & 0xF]);
    return true;
  }

  GReg shift = LoadGPR(e, sh, kScratch0);
  if (shift != kScratch0) {
    e.asm_().UXTB(kScratch0, shift);
  } else {
    e.asm_().UXTB(kScratch0, kScratch0);
  }

  a64::Label case_labels[TableSize];
  a64::Label done;

  e.asm_().CBZ(kScratch0, &case_labels[0]);
  for (uint32_t index = 1; index < TableSize; ++index) {
    e.asm_().CMP(kScratch0, index);
    e.asm_().B(EQ, &case_labels[index]);
  }

  e.LoadConstantV128(VR(dest), table[0]);
  e.asm_().B(&done);

  for (uint32_t index = 0; index < TableSize; ++index) {
    e.asm_().Bind(&case_labels[index]);
    e.LoadConstantV128(VR(dest), table[index]);
    if (index + 1 != TableSize) {
      e.asm_().B(&done);
    }
  }

  e.asm_().Bind(&done);
  return true;
}

static bool EmitLoadVectorShl(A64Emitter& e, const Instr* i) {
  return EmitLoadVectorTable(e, i, kLoadVectorShlTable);
}

static bool EmitLoadVectorShr(A64Emitter& e, const Instr* i) {
  return EmitLoadVectorTable(e, i, kLoadVectorShrTable);
}

static bool EmitPermute(A64Emitter& e, const Instr* i) {
  if (!i->dest || !IsAlloc(i->dest)) return true;
  auto ctrl = i->src1.value;
  auto s1 = i->src2.value, s2 = i->src3.value;
  if (!ctrl || !s1 || !s2) { e.asm_().NOP(); return true; }
  VReg vd = VR(i->dest);

  // Sources go into V0/V1 — TBL2 requires a consecutive register pair.
  if (VALUE_IS_CONSTANT(s1)) {
    e.LoadConstantV128(kVScratch0, s1->constant.v128);
  } else {
    e.asm_().MOV_16B(kVScratch0, VR(s1));
  }
  if (VALUE_IS_CONSTANT(s2)) {
    e.LoadConstantV128(kVScratch1, s2->constant.v128);
  } else {
    e.asm_().MOV_16B(kVScratch1, VR(s2));
  }

  if (i->flags == INT32_TYPE) {
    // Word-granularity select; control must be a constant (as on x64).
    if (!VALUE_IS_CONSTANT(ctrl)) return false;
    const uint32_t mask = ctrl->constant.u32;
    vec128_t idx = vec128b(0);
    for (uint32_t w = 0; w < 4; ++w) {
      const uint32_t lane = (mask >> (w * 8)) & 0x3;
      const uint32_t sel = (mask >> (w * 8 + 2)) & 0x1;
      for (uint32_t b = 0; b < 4; ++b) {
        idx.u8[w * 4 + b] = static_cast<uint8_t>(sel * 16 + lane * 4 + b);
      }
    }
    e.LoadConstantV128(kVScratch2, idx);
    e.asm_().TBL2(vd, kVScratch0, kVScratch2);
    return true;
  }

  if (i->flags == INT8_TYPE) {
    // vperm: each guest control byte selects from the 32-byte guest
    // concatenation. In host layout both the control byte position and the
    // source byte position carry the ^3 word swizzle, so the host table
    // index is (control & 0x1F) ^ 3.
    if (VALUE_IS_CONSTANT(ctrl)) {
      vec128_t idx = ctrl->constant.v128;
      for (int b = 0; b < 16; ++b) {
        idx.u8[b] = (idx.u8[b] & 0x1F) ^ 0x3;
      }
      e.LoadConstantV128(kVScratch2, idx);
    } else {
      e.asm_().MOVI_16B(kVScratch2, 0x1F);
      e.asm_().AND_16B(kVScratch2, VR(ctrl), kVScratch2);
      e.asm_().MOVI_16B(vd, 0x3);  // s1/s2 already copied; vd is free
      e.asm_().EOR_16B(kVScratch2, kVScratch2, vd);
    }
    e.asm_().TBL2(vd, kVScratch0, kVScratch2);
    return true;
  }

  return false;
}

static bool EmitSwizzle(A64Emitter& e, const Instr* i) {
  if (!i->dest || !IsAlloc(i->dest)) return true;
  auto src = i->src1.value;
  if (!src) { e.asm_().NOP(); return true; }
  VReg vs = LoadVec128(e, src, kVScratch0);
  VReg vd = VR(i->dest);
  // The swizzle mask lives in src2.offset — each 2 bits selects a source
  // word (flags carries the part type, which is always 32-bit here).
  const uint32_t swizzle = static_cast<uint32_t>(i->src2.offset);
  vec128_t idx = vec128b(0);
  for (uint32_t w = 0; w < 4; ++w) {
    const uint32_t lane = (swizzle >> (w * 2)) & 0x3;
    for (uint32_t b = 0; b < 4; ++b) {
      idx.u8[w * 4 + b] = static_cast<uint8_t>(lane * 4 + b);
    }
  }
  e.LoadConstantV128(kVScratch2, idx);
  e.asm_().TBL(vd, vs, kVScratch2);
  return true;
}

// Runs a v128 -> v128 conversion through a C helper: src goes out via
// helper_scratch, the result comes back the same way.
static bool EmitVec128UnaryViaHelper(A64Emitter& e, const Instr* i,
                                     const Value* src, uint64_t helper_address,
                                     uint64_t arg0) {
  if (!i->dest || !IsAlloc(i->dest) || !src) return true;
  const int32_t scratch_offset =
      static_cast<int32_t>(offsetof(A64BackendContext, helper_scratch));
  VReg src_reg = LoadVec128(e, src, kVScratch0);
  e.asm_().SUB(kScratch1, kContextReg,
               static_cast<uint32_t>(sizeof(A64BackendContext)));
  e.asm_().STR_Q(src_reg, kScratch1, scratch_offset);
  e.MovImm64(X1, arg0);
  e.MovImm64(X2, 0);
  e.MovImm64(X0, helper_address);
  auto backend = static_cast<A64Backend*>(e.backend());
  e.MovImm64(kScratch0,
             reinterpret_cast<uint64_t>(backend->guest_to_host_thunk()));
  e.asm_().BLR(kScratch0);
  e.asm_().SUB(kScratch1, kContextReg,
               static_cast<uint32_t>(sizeof(A64BackendContext)));
  e.asm_().LDR_Q(VR(i->dest), kScratch1, scratch_offset);
  return true;
}

static bool EmitPack(A64Emitter& e, const Instr* i) {
  if (!i->dest || !IsAlloc(i->dest)) return true;
  auto src1 = i->src1.value;
  VReg vd = VR(i->dest);
  uint32_t pack_mode = i->flags & PACK_TYPE_MODE;

  if (!src1) {
    e.asm_().MOVI_4S_zero(vd);
    return true;
  }

  switch (pack_mode) {
    case PACK_TYPE_D3DCOLOR:
      return EmitVec128UnaryViaHelper(
          e, i, src1, reinterpret_cast<uint64_t>(&A64PackD3DCOLOR), 0);
    case PACK_TYPE_FLOAT16_2:
      return EmitVec128UnaryViaHelper(
          e, i, src1, reinterpret_cast<uint64_t>(&A64PackFLOAT16_2), 0);
    case PACK_TYPE_FLOAT16_4:
      return EmitVec128UnaryViaHelper(
          e, i, src1, reinterpret_cast<uint64_t>(&A64PackFLOAT16_4), 0);
    case PACK_TYPE_SHORT_2:
      return EmitVec128UnaryViaHelper(
          e, i, src1, reinterpret_cast<uint64_t>(&A64PackSHORT_2), 0);
    case PACK_TYPE_8_IN_16:
    case PACK_TYPE_16_IN_32:
    case PACK_TYPE_SHORT_4:
    case PACK_TYPE_UINT_2101010:
    case PACK_TYPE_ULONG_4202020:
    default:
      // TODO: unimplemented pack modes — copy through for now
      if (IsAlloc(src1)) {
        e.asm_().MOV_16B(vd, VR(src1));
      } else {
        e.asm_().MOVI_4S_zero(vd);
      }
      break;
  }
  return true;
}

static bool EmitUnpack(A64Emitter& e, const Instr* i) {
  if (!i->dest || !IsAlloc(i->dest)) return true;
  auto src = i->src1.value;
  VReg vd = VR(i->dest);
  uint32_t pack_mode = i->flags & PACK_TYPE_MODE;

  if (!src) {
    e.asm_().MOVI_4S_zero(vd);
    return true;
  }

  switch (pack_mode) {
    case PACK_TYPE_D3DCOLOR:
      return EmitVec128UnaryViaHelper(
          e, i, src, reinterpret_cast<uint64_t>(&A64UnpackD3DCOLOR), 0);
    case PACK_TYPE_FLOAT16_2:
      return EmitVec128UnaryViaHelper(
          e, i, src, reinterpret_cast<uint64_t>(&A64UnpackFLOAT16_2), 0);
    case PACK_TYPE_FLOAT16_4:
      return EmitVec128UnaryViaHelper(
          e, i, src, reinterpret_cast<uint64_t>(&A64UnpackFLOAT16_4), 0);
    case PACK_TYPE_SHORT_2:
      return EmitVec128UnaryViaHelper(
          e, i, src, reinterpret_cast<uint64_t>(&A64UnpackSHORT_2), 0);
    case PACK_TYPE_8_IN_16:
    case PACK_TYPE_16_IN_32:
    case PACK_TYPE_SHORT_4:
    case PACK_TYPE_UINT_2101010:
    case PACK_TYPE_ULONG_4202020:
    default:
      // TODO: unimplemented unpack modes — copy through for now
      if (IsAlloc(src)) {
        e.asm_().MOV_16B(vd, VR(src));
      } else {
        e.asm_().MOVI_4S_zero(vd);
      }
      break;
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

// === Atomics ===

static bool EmitAtomicExchange(A64Emitter& e, const Instr* i) {
  // ATOMIC_EXCHANGE: atomically swap *addr with new_value, return old
  // src1 = address (in guest memory), src2 = new value
  if (!i->dest || !IsAlloc(i->dest)) return true;
  auto addr_val = i->src1.value;
  auto new_val = i->src2.value;
  if (!addr_val || !new_val) return true;

  GReg addr_reg = IsAlloc(addr_val) ? GR(addr_val) : kScratch0;
  if (!IsAlloc(addr_val) && VALUE_IS_CONSTANT(addr_val)) {
    e.MovImm64(kScratch0, addr_val->constant.u64);
  }
  GReg new_reg = IsAlloc(new_val) ? GR(new_val) : kScratch1;
  if (!IsAlloc(new_val) && VALUE_IS_CONSTANT(new_val)) {
    e.MovImm64(kScratch1, new_val->constant.u64);
  }

  GReg dest = GR(i->dest);
  // Compute host address: membase + guest_addr
  e.asm_().ADD(kScratch2, kMembaseReg, addr_reg);

  // LDXR/STXR loop
  Label retry;
  e.asm_().Bind(&retry);
  if (i->dest->type <= INT32_TYPE) {
    e.asm_().LDXRw(dest, kScratch2);
    e.asm_().STXRw(X12, new_reg, kScratch2);
  } else {
    e.asm_().LDXR(dest, kScratch2);
    e.asm_().STXR(X12, new_reg, kScratch2);
  }
  e.asm_().CBNZ(X12, &retry);  // retry if store failed
  e.asm_().DMB_ISH();  // memory barrier
  return true;
}

static bool EmitAtomicCompareExchange(A64Emitter& e, const Instr* i) {
  // ATOMIC_COMPARE_EXCHANGE: if *addr == expected, store desired; return old
  // src1 = address, src2 = expected, src3 = desired
  if (!i->dest || !IsAlloc(i->dest)) return true;
  auto addr_val = i->src1.value;
  auto expected_val = i->src2.value;
  auto desired_val = i->src3.value;
  if (!addr_val || !expected_val || !desired_val) return true;

  GReg addr_reg = IsAlloc(addr_val) ? GR(addr_val) : kScratch0;
  if (!IsAlloc(addr_val) && VALUE_IS_CONSTANT(addr_val)) {
    e.MovImm64(kScratch0, addr_val->constant.u64);
  }
  GReg expected = IsAlloc(expected_val) ? GR(expected_val) : kScratch1;
  if (!IsAlloc(expected_val) && VALUE_IS_CONSTANT(expected_val)) {
    e.MovImm64(kScratch1, expected_val->constant.u64);
  }
  GReg desired = IsAlloc(desired_val) ? GR(desired_val) : kScratch2;
  if (!IsAlloc(desired_val) && VALUE_IS_CONSTANT(desired_val)) {
    e.MovImm64(kScratch2, desired_val->constant.u64);
  }

  GReg dest = GR(i->dest);
  // Compute host address
  e.asm_().ADD(X13, kMembaseReg, addr_reg);

  Label retry, done;
  e.asm_().Bind(&retry);
  if (i->dest->type <= INT32_TYPE) {
    e.asm_().LDXRw(dest, X13);
    e.asm_().CMP(dest, expected);
  } else {
    e.asm_().LDXR(dest, X13);
    e.asm_().CMP(dest, expected);
  }
  e.asm_().B(NE, &done);  // if not equal, skip store
  if (i->dest->type <= INT32_TYPE) {
    e.asm_().STXRw(X12, desired, X13);
  } else {
    e.asm_().STXR(X12, desired, X13);
  }
  e.asm_().CBNZ(X12, &retry);  // retry if exclusive store failed
  e.asm_().Bind(&done);
  e.asm_().DMB_ISH();
  return true;
}

static bool EmitReservedLoad(A64Emitter& e, const Instr* i) {
  if (!i->dest || !IsAlloc(i->dest)) return true;
  auto addr_val = i->src1.value;
  if (!addr_val) return true;

  GReg guest_addr = VALUE_IS_CONSTANT(addr_val) ? X1 : LoadGPR(e, addr_val, X1);
  if (VALUE_IS_CONSTANT(addr_val)) {
    e.MovImm64(X1, static_cast<uint32_t>(addr_val->constant.u64));
  } else if (guest_addr != X1) {
    e.asm_().MOVw(X1, guest_addr);
  } else {
    e.asm_().MOVw(X1, X1);
  }

  e.MovImm64(X2, 0);
  e.MovImm64(
      X0, reinterpret_cast<uint64_t>(i->dest->type <= INT32_TYPE
                                         ? &A64ReservedLoad32
                                         : &A64ReservedLoad64));

  auto backend = static_cast<A64Backend*>(e.backend());
  e.MovImm64(kScratch0,
             reinterpret_cast<uint64_t>(backend->guest_to_host_thunk()));
  e.asm_().BLR(kScratch0);

  if (i->dest->type <= INT32_TYPE) {
    e.asm_().MOVw(GR(i->dest), X0);
  } else {
    e.asm_().MOV(GR(i->dest), X0);
  }
  return true;
}

static bool EmitReservedStore(A64Emitter& e, const Instr* i) {
  if (!i->dest || !IsAlloc(i->dest)) return true;
  auto addr_val = i->src1.value;
  auto value_val = i->src2.value;
  if (!addr_val || !value_val) return true;

  GReg guest_addr = VALUE_IS_CONSTANT(addr_val) ? X1 : LoadGPR(e, addr_val, X1);
  if (VALUE_IS_CONSTANT(addr_val)) {
    e.MovImm64(X1, static_cast<uint32_t>(addr_val->constant.u64));
  } else if (guest_addr != X1) {
    e.asm_().MOVw(X1, guest_addr);
  } else {
    e.asm_().MOVw(X1, X1);
  }

  bool is_32bit_store = value_val->type <= INT32_TYPE;
  GReg store_value = VALUE_IS_CONSTANT(value_val)
                         ? X2
                         : LoadGPR(e, value_val, X2);
  if (VALUE_IS_CONSTANT(value_val)) {
    if (is_32bit_store) {
      e.MovImm64(X2, static_cast<uint32_t>(value_val->constant.u64));
    } else {
      e.MovImm64(X2, value_val->constant.u64);
    }
  } else if (is_32bit_store && store_value != X2) {
    e.asm_().MOVw(X2, store_value);
  } else if (is_32bit_store) {
    e.asm_().MOVw(X2, X2);
  } else if (store_value != X2) {
    e.asm_().MOV(X2, store_value);
  }

  e.MovImm64(X0, reinterpret_cast<uint64_t>(is_32bit_store
                                                ? &A64ReservedStore32
                                                : &A64ReservedStore64));

  auto backend = static_cast<A64Backend*>(e.backend());
  e.MovImm64(kScratch0,
             reinterpret_cast<uint64_t>(backend->guest_to_host_thunk()));
  e.asm_().BLR(kScratch0);
  e.asm_().MOVw(GR(i->dest), X0);
  return true;
}

static bool EmitUnimplemented(A64Emitter& e, const Instr* i) {
  static std::atomic<int> unimpl_count{0};
  int c = ++unimpl_count;
  if (c <= 50 || (c % 500) == 0) {
    XELOGW("ARM64: UNIMPLEMENTED opcode #{}: {}",
           c, GetOpcodeName(i->GetOpcodeNum()));
  }
  // Emit NOP instead of BRK so the thread continues
  // (wrong results but won't trap/hang)
  e.asm_().NOP();
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
  sequence_table[OPCODE_DEBUG_BREAK_TRUE] = EmitDebugBreakTrue;
  sequence_table[OPCODE_TRAP] = EmitTrap;
  sequence_table[OPCODE_TRAP_TRUE] = EmitTrapTrue;
  sequence_table[OPCODE_RETURN] = EmitReturn;
  sequence_table[OPCODE_RETURN_TRUE] = EmitReturnTrue;
  sequence_table[OPCODE_BRANCH] = EmitBranch;
  sequence_table[OPCODE_BRANCH_TRUE] = EmitBranchTrue;
  sequence_table[OPCODE_BRANCH_FALSE] = EmitBranchFalse;
  sequence_table[OPCODE_CALL] = EmitCall;
  sequence_table[OPCODE_CALL_TRUE] = EmitCallTrue;
  sequence_table[OPCODE_CALL_INDIRECT] = EmitCallIndirect;
  sequence_table[OPCODE_CALL_INDIRECT_TRUE] = EmitCallIndirectTrue;
  sequence_table[OPCODE_CALL_EXTERN] = EmitCallExtern;
  sequence_table[OPCODE_SET_RETURN_ADDRESS] = EmitSetReturnAddress;

  // Context
  sequence_table[OPCODE_LOAD_CONTEXT] = EmitLoadContext;
  sequence_table[OPCODE_STORE_CONTEXT] = EmitStoreContext;
  sequence_table[OPCODE_CONTEXT_BARRIER] = EmitNop;

  // Memory
  sequence_table[OPCODE_LOAD] = EmitLoad;
  sequence_table[OPCODE_STORE] = EmitStore;
  sequence_table[OPCODE_LOAD_OFFSET] = EmitLoadOffset;
  sequence_table[OPCODE_STORE_OFFSET] = EmitStoreOffset;
  sequence_table[OPCODE_LOAD_MMIO] = EmitLoadMmio;
  sequence_table[OPCODE_STORE_MMIO] = EmitStoreMmio;
  sequence_table[OPCODE_LVL] = EmitLoadVectorLeft;
  sequence_table[OPCODE_LVR] = EmitLoadVectorRight;
  sequence_table[OPCODE_STVL] = EmitStoreVectorLeft;
  sequence_table[OPCODE_STVR] = EmitStoreVectorRight;
  sequence_table[OPCODE_MEMORY_BARRIER] = [](A64Emitter& e, const Instr*) {
    e.asm_().DMB_ISH();
    return true;
  };
  sequence_table[OPCODE_MEMSET] = EmitMemset;
  sequence_table[OPCODE_CACHE_CONTROL] = EmitNop;
  sequence_table[OPCODE_LOAD_CLOCK] = EmitLoadClock;
  sequence_table[OPCODE_LOAD_LOCAL] = EmitLoadLocal;
  sequence_table[OPCODE_STORE_LOCAL] = EmitStoreLocal;

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

  sequence_table[OPCODE_ATOMIC_EXCHANGE] = EmitAtomicExchange;
  sequence_table[OPCODE_ATOMIC_COMPARE_EXCHANGE] = EmitAtomicCompareExchange;
  sequence_table[OPCODE_RESERVED_LOAD] = EmitReservedLoad;
  sequence_table[OPCODE_RESERVED_STORE] = EmitReservedStore;

  // Vector/NEON (Phase 4)
  sequence_table[OPCODE_LOAD_VECTOR_SHL] = EmitLoadVectorShl;
  sequence_table[OPCODE_LOAD_VECTOR_SHR] = EmitLoadVectorShr;
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
    current_instr_for_logging = i;
    current_emitter_for_logging = e;
    bool result = it->second(*e, i);
    current_instr_for_logging = nullptr;
    current_emitter_for_logging = nullptr;
    return result;
  }
  static std::atomic<int> missing_count{0};
  int c = ++missing_count;
  if (c <= 50 || (c % 500) == 0) {
    XELOGE("ARM64: No sequence for opcode #{}: {}", c, GetOpcodeName(opcode));
  }
  // Emit NOP instead of BRK so the thread continues
  e->asm_().NOP();
  return false;
}

}  // namespace a64
}  // namespace backend
}  // namespace cpu
}  // namespace xe
