/**
 * ARM64 Code Cache Implementation
 * Uses MAP_JIT on Apple Silicon for W^X compatible executable memory.
 */
#include "xenia/cpu/backend/a64/a64_code_cache.h"

#include <cstring>
#include <sys/mman.h>
#include "xenia/base/logging.h"

#ifdef __APPLE__
#include <libkern/OSCacheControl.h>
#include <pthread.h>
#endif

namespace xe {
namespace cpu {
namespace backend {
namespace a64 {

class PosixA64CodeCache : public A64CodeCache {
 public:
  PosixA64CodeCache() = default;
  ~PosixA64CodeCache() override;
  bool Initialize() override;

 private:
  void* jit_region_ = nullptr;
  size_t jit_region_size_ = 0;
};

std::unique_ptr<A64CodeCache> A64CodeCache::Create() {
  return std::make_unique<PosixA64CodeCache>();
}

PosixA64CodeCache::~PosixA64CodeCache() {
  if (jit_region_) {
    munmap(jit_region_, jit_region_size_);
    jit_region_ = nullptr;
  }
}

bool PosixA64CodeCache::Initialize() {
  // On macOS ARM64, we cannot use fixed-address file mappings.
  // Instead, use mmap with MAP_JIT for executable memory.
  // This bypasses the base class Initialize() entirely.

  // 64MB is plenty for JIT code — most games use < 10MB
  jit_region_size_ = 64 * 1024 * 1024;

  // Allocate JIT-capable memory using MAP_JIT on Apple Silicon
  int flags = MAP_PRIVATE | MAP_ANONYMOUS;
#ifdef __APPLE__
  flags |= MAP_JIT;
#endif

  jit_region_ = mmap(nullptr, jit_region_size_,
                     PROT_READ | PROT_WRITE | PROT_EXEC,
                     flags, -1, 0);
  if (jit_region_ == MAP_FAILED) {
    XELOGE("ARM64: Failed to mmap JIT region ({} bytes): {}",
           jit_region_size_, strerror(errno));
    jit_region_ = nullptr;
    return false;
  }

  XELOGI("ARM64: JIT code cache allocated at {:p} ({} MB)",
         jit_region_, jit_region_size_ / (1024 * 1024));

  // Set up the code cache pointers that the base class expects
  generated_code_execute_base_ = reinterpret_cast<uint8_t*>(jit_region_);
  generated_code_write_base_ = generated_code_execute_base_;

  // Reserve the indirection table at an arbitrary address — the canonical
  // 0x80000000 host address sits inside PAGEZERO on macOS. Slots hold 32-bit
  // offsets into the JIT region (0 = unresolved). The whole table is mapped
  // read-write: pages are lazily zero-filled on first touch, so fast-path
  // loads can never fault regardless of the guest target, and untouched
  // pages cost no real memory.
  void* table = mmap(nullptr, kIndirectionTableSize, PROT_READ | PROT_WRITE,
                     MAP_PRIVATE | MAP_ANONYMOUS | MAP_NORESERVE, -1, 0);
  if (table == MAP_FAILED) {
    XELOGW(
        "ARM64: failed to reserve indirection table ({}); indirect calls "
        "will always go through the resolve thunk",
        strerror(errno));
    indirection_table_base_ = nullptr;
  } else {
    indirection_table_base_ = reinterpret_cast<uint8_t*>(table);
    XELOGI("ARM64: indirection table reserved at {:p} ({} MB)", table,
           kIndirectionTableSize / (1024 * 1024));
  }

  // Initialize the generated code offset
  generated_code_offset_ = 0;

  // Don't fill entire region with BRK — just leave it zeroed.
  // Individual code regions will be filled as needed.

  return true;
}

bool A64CodeCache::Initialize() {
  // Delegate to platform-specific implementation
  return true;
}

void A64CodeCache::FillCode(void* write_address, size_t size) {
  // Fill with BRK #1 instructions (0xD4200020)
  uint32_t* p = reinterpret_cast<uint32_t*>(write_address);
  size_t count = size / 4;
  for (size_t i = 0; i < count; ++i) {
    p[i] = 0xD4200020;  // BRK #1
  }
}

void A64CodeCache::FlushCodeRange(void* address, size_t size) {
#ifdef __APPLE__
  // On Apple Silicon, flush data cache and invalidate instruction cache.
  sys_dcache_flush(address, size);
  sys_icache_invalidate(address, size);
#else
  // GCC/Linux: use built-in
  __builtin___clear_cache(reinterpret_cast<char*>(address),
                          reinterpret_cast<char*>(address) + size);
#endif
}

void A64CodeCache::OnCodePlaced(uint32_t guest_address,
                                GuestFunction* function_info,
                                void* code_execute_address,
                                size_t code_size) {
  // No VTune on ARM64; no-op.
}

}  // namespace a64
}  // namespace backend
}  // namespace cpu
}  // namespace xe
