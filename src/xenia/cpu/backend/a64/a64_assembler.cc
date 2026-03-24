/**
 * ARM64 Assembler Implementation
 */
#include "xenia/cpu/backend/a64/a64_assembler.h"

#include "xenia/base/logging.h"
#include "xenia/cpu/backend/a64/a64_backend.h"
#include "xenia/cpu/backend/a64/a64_emitter.h"
#include "xenia/cpu/backend/a64/a64_function.h"

namespace xe {
namespace cpu {
namespace backend {
namespace a64 {

A64Assembler::A64Assembler(A64Backend* backend)
    : Assembler(backend), a64_backend_(backend) {}

A64Assembler::~A64Assembler() = default;

bool A64Assembler::Initialize() {
  Assembler::Initialize();
  emitter_ = std::make_unique<A64Emitter>(a64_backend_);
  return true;
}

void A64Assembler::Reset() {
  Assembler::Reset();
  string_buffer_.Reset();
}

bool A64Assembler::Assemble(GuestFunction* function,
                            hir::HIRBuilder* builder,
                            uint32_t debug_info_flags,
                            std::unique_ptr<FunctionDebugInfo> debug_info) {
  Reset();

  void* machine_code = nullptr;
  size_t code_size = 0;
  std::vector<SourceMapEntry> source_map;

  if (!emitter_->Emit(function, builder, debug_info_flags,
                      debug_info.get(), &machine_code, &code_size,
                      &source_map)) {
    XELOGE("ARM64: Failed to emit function {:08X}", function->address());
    return false;
  }

  function->set_debug_info(std::move(debug_info));

  // Setup the function with the emitted code
  auto a64_fn = static_cast<A64Function*>(function);
  a64_fn->Setup(reinterpret_cast<uint8_t*>(machine_code), code_size);

  // Copy source map
  function->source_map() = std::move(source_map);
  if (debug_info && (debug_info_flags & DebugInfoFlags::kDebugInfoAllDisasm)) {
    debug_info->Dump();
  }

  return true;
}

}  // namespace a64
}  // namespace backend
}  // namespace cpu
}  // namespace xe
