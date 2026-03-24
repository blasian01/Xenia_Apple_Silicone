/**
 * ARM64 JIT Function Implementation
 */
#include "xenia/cpu/backend/a64/a64_function.h"

#include "xenia/cpu/backend/a64/a64_backend.h"
#include "xenia/base/logging.h"
#include "xenia/cpu/processor.h"
#include "xenia/cpu/thread_state.h"

namespace xe {
namespace cpu {
namespace backend {
namespace a64 {

A64Function::A64Function(Module* module, uint32_t address)
    : GuestFunction(module, address) {}

A64Function::~A64Function() {
  // machine_code_ is freed by code cache.
}

void A64Function::Setup(uint8_t* machine_code, size_t machine_code_length) {
  machine_code_ = machine_code;
  machine_code_length_ = machine_code_length;
}

bool A64Function::CallImpl(ThreadState* thread_state, uint32_t return_address) {
  auto backend =
      reinterpret_cast<A64Backend*>(thread_state->processor()->backend());
  auto thunk = backend->host_to_guest_thunk();
  auto* context = thread_state->context();
  XELOGI(
      "ARM64: CallImpl guest={:08X} machine_code={:p} thunk={:p} r1={:08X} "
      "r13={:08X} lr={:08X} ctr={:08X}",
      address(), (void*)machine_code_, (void*)thunk, uint32_t(context->r[1]),
      uint32_t(context->r[13]), uint32_t(context->lr), uint32_t(context->ctr));
  thunk(machine_code_, thread_state->context(),
        reinterpret_cast<void*>(uintptr_t(return_address)));
  XELOGI("ARM64: CallImpl returned from guest={:08X}", address());
  return true;
}

}  // namespace a64
}  // namespace backend
}  // namespace cpu
}  // namespace xe
