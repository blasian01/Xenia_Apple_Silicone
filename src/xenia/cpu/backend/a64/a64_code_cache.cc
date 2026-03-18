/**
 * ARM64 Code Cache Implementation
 */
#include "xenia/cpu/backend/a64/a64_code_cache.h"

#include <cstring>
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
  ~PosixA64CodeCache() override = default;
  bool Initialize() override;
};

std::unique_ptr<A64CodeCache> A64CodeCache::Create() {
  return std::make_unique<PosixA64CodeCache>();
}

bool PosixA64CodeCache::Initialize() {
  return A64CodeCache::Initialize();
}

bool A64CodeCache::Initialize() {
  return CodeCacheBase<A64CodeCache>::Initialize();
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
