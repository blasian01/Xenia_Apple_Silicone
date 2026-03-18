/**
 ******************************************************************************
 * Xenia : Xbox 360 Emulator Research Project                                 *
 ******************************************************************************
 * Copyright 2025 Ben Vanik. All rights reserved.                             *
 * Released under the BSD license - see LICENSE in the root for more details. *
 ******************************************************************************
 */

#include "xenia/base/exception_handler.h"

#include <signal.h>
#include <cstdint>

#include "xenia/base/assert.h"
#include "xenia/base/host_thread_context.h"
#include "xenia/base/logging.h"
#include "xenia/base/math.h"
#include "xenia/base/platform.h"

// macOS uses a different mcontext_t layout than Linux.
// On x86_64 macOS: mcontext->__ss has general regs, mcontext->__fs has FPU/XMM.
// On ARM64 macOS: mcontext->__ss has general regs, mcontext->__ns has NEON.

namespace xe {

bool signal_handlers_installed_ = false;
struct sigaction original_sigill_handler_;
struct sigaction original_sigsegv_handler_;
struct sigaction original_sigbus_handler_;

constexpr size_t kMaxHandlerCount = 8;
std::pair<ExceptionHandler::Handler, void*> handlers_[kMaxHandlerCount];

static void ExceptionHandlerCallback(int signal_number, siginfo_t* signal_info,
                                     void* signal_context) {
  ucontext_t* ucontext = reinterpret_cast<ucontext_t*>(signal_context);
  mcontext_t mcontext = ucontext->uc_mcontext;

  HostThreadContext thread_context;

#if XE_ARCH_AMD64
  // macOS x86_64: __darwin_x86_thread_state64
  thread_context.rip = uint64_t(mcontext->__ss.__rip);
  thread_context.eflags = uint32_t(mcontext->__ss.__rflags);
  thread_context.rax = uint64_t(mcontext->__ss.__rax);
  thread_context.rcx = uint64_t(mcontext->__ss.__rcx);
  thread_context.rdx = uint64_t(mcontext->__ss.__rdx);
  thread_context.rbx = uint64_t(mcontext->__ss.__rbx);
  thread_context.rsp = uint64_t(mcontext->__ss.__rsp);
  thread_context.rbp = uint64_t(mcontext->__ss.__rbp);
  thread_context.rsi = uint64_t(mcontext->__ss.__rsi);
  thread_context.rdi = uint64_t(mcontext->__ss.__rdi);
  thread_context.r8 = uint64_t(mcontext->__ss.__r8);
  thread_context.r9 = uint64_t(mcontext->__ss.__r9);
  thread_context.r10 = uint64_t(mcontext->__ss.__r10);
  thread_context.r11 = uint64_t(mcontext->__ss.__r11);
  thread_context.r12 = uint64_t(mcontext->__ss.__r12);
  thread_context.r13 = uint64_t(mcontext->__ss.__r13);
  thread_context.r14 = uint64_t(mcontext->__ss.__r14);
  thread_context.r15 = uint64_t(mcontext->__ss.__r15);
  // XMM registers are in the float state
  std::memcpy(thread_context.xmm_registers, &mcontext->__fs.__fpu_xmm0,
              sizeof(thread_context.xmm_registers));
#elif XE_ARCH_ARM64
  // macOS ARM64: __darwin_arm_thread_state64
  std::memcpy(thread_context.x, mcontext->__ss.__x,
              sizeof(thread_context.x));
  thread_context.sp = uint64_t(mcontext->__ss.__sp);
  thread_context.pc = uint64_t(mcontext->__ss.__pc);
  thread_context.pstate = uint32_t(mcontext->__ss.__cpsr);
  // NEON/SIMD registers
  thread_context.fpsr = mcontext->__ns.__fpsr;
  thread_context.fpcr = mcontext->__ns.__fpcr;
  std::memcpy(thread_context.v, mcontext->__ns.__v,
              sizeof(thread_context.v));
#endif  // XE_ARCH

  Exception ex;
  switch (signal_number) {
    case SIGILL:
      ex.InitializeIllegalInstruction(&thread_context);
      break;
    case SIGSEGV:
    case SIGBUS: {
      // macOS uses SIGBUS for some memory access violations
      Exception::AccessViolationOperation access_violation_operation;
#if XE_ARCH_AMD64
      // On macOS x86_64, we can check the error code in __es.__err
      // Bit 1 of the error code indicates write (1) vs read (0)
      constexpr uint64_t kX86PageFaultErrorCodeWrite = UINT64_C(1) << 1;
      access_violation_operation =
          (uint64_t(mcontext->__es.__err) & kX86PageFaultErrorCodeWrite)
              ? Exception::AccessViolationOperation::kWrite
              : Exception::AccessViolationOperation::kRead;
#elif XE_ARCH_ARM64
      // On macOS ARM64, we need to check the ESR (Exception Syndrome Register)
      // via the far/esr fields in the exception state
      access_violation_operation =
          Exception::AccessViolationOperation::kUnknown;
      // Try to determine from ESR if available
      uint64_t esr = mcontext->__es.__esr;
      if (((esr >> 26) & 0b111110) == 0b100100) {
        access_violation_operation =
            (esr & (UINT64_C(1) << 6))
                ? Exception::AccessViolationOperation::kWrite
                : Exception::AccessViolationOperation::kRead;
      }
#else
      access_violation_operation =
          Exception::AccessViolationOperation::kUnknown;
#endif  // XE_ARCH
      ex.InitializeAccessViolation(
          &thread_context, reinterpret_cast<uint64_t>(signal_info->si_addr),
          access_violation_operation);
    } break;
    default:
      assert_unhandled_case(signal_number);
  }

  for (size_t i = 0; i < xe::countof(handlers_) && handlers_[i].first; ++i) {
    if (handlers_[i].first(&ex, handlers_[i].second)) {
      // Exception handled - write back modified registers.
#if XE_ARCH_AMD64
      mcontext->__ss.__rip = thread_context.rip;
      mcontext->__ss.__rflags = thread_context.eflags;
      uint32_t modified_register_index;
      static constexpr uint64_t* __darwin_x86_regs[] = {nullptr};
      // Write back modified integer registers
      uint16_t modified_int_registers_remaining = ex.modified_int_registers();
      while (xe::bit_scan_forward(modified_int_registers_remaining,
                                  &modified_register_index)) {
        modified_int_registers_remaining &=
            ~(UINT16_C(1) << modified_register_index);
        switch (modified_register_index) {
          case 0: mcontext->__ss.__rax = thread_context.rax; break;
          case 1: mcontext->__ss.__rcx = thread_context.rcx; break;
          case 2: mcontext->__ss.__rdx = thread_context.rdx; break;
          case 3: mcontext->__ss.__rbx = thread_context.rbx; break;
          case 4: mcontext->__ss.__rsp = thread_context.rsp; break;
          case 5: mcontext->__ss.__rbp = thread_context.rbp; break;
          case 6: mcontext->__ss.__rsi = thread_context.rsi; break;
          case 7: mcontext->__ss.__rdi = thread_context.rdi; break;
          case 8: mcontext->__ss.__r8 = thread_context.r8; break;
          case 9: mcontext->__ss.__r9 = thread_context.r9; break;
          case 10: mcontext->__ss.__r10 = thread_context.r10; break;
          case 11: mcontext->__ss.__r11 = thread_context.r11; break;
          case 12: mcontext->__ss.__r12 = thread_context.r12; break;
          case 13: mcontext->__ss.__r13 = thread_context.r13; break;
          case 14: mcontext->__ss.__r14 = thread_context.r14; break;
          case 15: mcontext->__ss.__r15 = thread_context.r15; break;
        }
      }
      // Write back modified XMM registers
      uint16_t modified_xmm_registers_remaining = ex.modified_xmm_registers();
      while (xe::bit_scan_forward(modified_xmm_registers_remaining,
                                  &modified_register_index)) {
        modified_xmm_registers_remaining &=
            ~(UINT16_C(1) << modified_register_index);
        std::memcpy(
            reinterpret_cast<uint8_t*>(&mcontext->__fs.__fpu_xmm0) +
                modified_register_index * sizeof(vec128_t),
            &thread_context.xmm_registers[modified_register_index],
            sizeof(vec128_t));
      }
#elif XE_ARCH_ARM64
      uint32_t modified_register_index;
      uint32_t modified_x_registers_remaining = ex.modified_x_registers();
      while (xe::bit_scan_forward(modified_x_registers_remaining,
                                  &modified_register_index)) {
        modified_x_registers_remaining &=
            ~(UINT32_C(1) << modified_register_index);
        mcontext->__ss.__x[modified_register_index] =
            thread_context.x[modified_register_index];
      }
      mcontext->__ss.__sp = thread_context.sp;
      mcontext->__ss.__pc = thread_context.pc;
      mcontext->__ss.__cpsr = thread_context.pstate;
      mcontext->__ns.__fpsr = thread_context.fpsr;
      mcontext->__ns.__fpcr = thread_context.fpcr;
      uint32_t modified_v_registers_remaining = ex.modified_v_registers();
      while (xe::bit_scan_forward(modified_v_registers_remaining,
                                  &modified_register_index)) {
        modified_v_registers_remaining &=
            ~(UINT32_C(1) << modified_register_index);
        std::memcpy(&mcontext->__ns.__v[modified_register_index],
                    &thread_context.v[modified_register_index],
                    sizeof(vec128_t));
      }
#endif  // XE_ARCH
      return;
    }
  }
}

void ExceptionHandler::Install(Handler fn, void* data) {
  if (!signal_handlers_installed_) {
    struct sigaction signal_handler;

    std::memset(&signal_handler, 0, sizeof(signal_handler));
    signal_handler.sa_sigaction = ExceptionHandlerCallback;
    signal_handler.sa_flags = SA_SIGINFO;

    if (sigaction(SIGILL, &signal_handler, &original_sigill_handler_) != 0) {
      assert_always("Failed to install new SIGILL handler");
    }
    if (sigaction(SIGSEGV, &signal_handler, &original_sigsegv_handler_) != 0) {
      assert_always("Failed to install new SIGSEGV handler");
    }
    // macOS also uses SIGBUS for memory access violations
    if (sigaction(SIGBUS, &signal_handler, &original_sigbus_handler_) != 0) {
      assert_always("Failed to install new SIGBUS handler");
    }
    signal_handlers_installed_ = true;
  }

  for (size_t i = 0; i < xe::countof(handlers_); ++i) {
    if (!handlers_[i].first) {
      handlers_[i].first = fn;
      handlers_[i].second = data;
      return;
    }
  }
  assert_always("Too many exception handlers installed");
}

void ExceptionHandler::Uninstall(Handler fn, void* data) {
  for (size_t i = 0; i < xe::countof(handlers_); ++i) {
    if (handlers_[i].first == fn && handlers_[i].second == data) {
      for (; i < xe::countof(handlers_) - 1; ++i) {
        handlers_[i] = handlers_[i + 1];
      }
      handlers_[i].first = nullptr;
      handlers_[i].second = nullptr;
      break;
    }
  }

  bool has_any = false;
  for (size_t i = 0; i < xe::countof(handlers_); ++i) {
    if (handlers_[i].first) {
      has_any = true;
      break;
    }
  }
  if (!has_any) {
    if (signal_handlers_installed_) {
      if (sigaction(SIGILL, &original_sigill_handler_, NULL) != 0) {
        assert_always("Failed to restore original SIGILL handler");
      }
      if (sigaction(SIGSEGV, &original_sigsegv_handler_, NULL) != 0) {
        assert_always("Failed to restore original SIGSEGV handler");
      }
      if (sigaction(SIGBUS, &original_sigbus_handler_, NULL) != 0) {
        assert_always("Failed to restore original SIGBUS handler");
      }
      signal_handlers_installed_ = false;
    }
  }
}

}  // namespace xe
