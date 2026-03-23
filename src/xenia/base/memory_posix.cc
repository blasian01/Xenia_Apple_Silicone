/**
 ******************************************************************************
 * Xenia : Xbox 360 Emulator Research Project                                 *
 ******************************************************************************
 * Copyright 2025 Ben Vanik. All rights reserved.                             *
 * Released under the BSD license - see LICENSE in the root for more details. *
 ******************************************************************************
 */

#include "xenia/base/memory.h"

#include <fcntl.h>
#include <sys/mman.h>
#include <unistd.h>
#include <algorithm>
#include <cstddef>
#include <cstdlib>
#include <fstream>
#include <mutex>
#include <sstream>

#include "xenia/base/math.h"
#include "xenia/base/platform.h"
#include "xenia/base/string.h"

#if XE_PLATFORM_ANDROID
#include <dlfcn.h>
#include <linux/ashmem.h>
#include <string.h>
#include <sys/ioctl.h>

#include "xenia/base/main_android.h"
#endif

#if XE_PLATFORM_MAC
#include <mach/mach.h>
#include <mach/mach_vm.h>
#include <sys/sysctl.h>
#endif

namespace xe {
namespace memory {

#if XE_PLATFORM_ANDROID
// May be null if no dynamically loaded functions are required.
static void* libandroid_;
// API 26+.
static int (*android_ASharedMemory_create_)(const char* name, size_t size);

void AndroidInitialize() {
  if (xe::GetAndroidApiLevel() >= 26) {
    libandroid_ = dlopen("libandroid.so", RTLD_NOW);
    assert_not_null(libandroid_);
    if (libandroid_) {
      android_ASharedMemory_create_ =
          reinterpret_cast<decltype(android_ASharedMemory_create_)>(
              dlsym(libandroid_, "ASharedMemory_create"));
      assert_not_null(android_ASharedMemory_create_);
    }
  }
}

void AndroidShutdown() {
  android_ASharedMemory_create_ = nullptr;
  if (libandroid_) {
    dlclose(libandroid_);
    libandroid_ = nullptr;
  }
}
#endif

size_t page_size() { return getpagesize(); }
size_t allocation_granularity() { return page_size(); }

uint32_t ToPosixProtectFlags(PageAccess access) {
  switch (access) {
    case PageAccess::kNoAccess:
      return PROT_NONE;
    case PageAccess::kReadOnly:
      return PROT_READ;
    case PageAccess::kReadWrite:
      return PROT_READ | PROT_WRITE;
    case PageAccess::kExecuteReadOnly:
      return PROT_READ | PROT_EXEC;
    case PageAccess::kExecuteReadWrite:
      return PROT_READ | PROT_WRITE | PROT_EXEC;
    default:
      assert_unhandled_case(access);
      return PROT_NONE;
  }
}

PageAccess ToXeniaProtectFlags(const char* protection) {
  if (protection[0] == 'r' && protection[1] == 'w' && protection[2] == 'x') {
    return PageAccess::kExecuteReadWrite;
  }
  if (protection[0] == 'r' && protection[1] == '-' && protection[2] == 'x') {
    return PageAccess::kExecuteReadOnly;
  }
  if (protection[0] == 'r' && protection[1] == 'w' && protection[2] == '-') {
    return PageAccess::kReadWrite;
  }
  if (protection[0] == 'r' && protection[1] == '-' && protection[2] == '-') {
    return PageAccess::kReadOnly;
  }
  return PageAccess::kNoAccess;
}

bool IsWritableExecutableMemorySupported() {
#if XE_PLATFORM_MAC
  // macOS with hardened runtime enforces W^X. Under Rosetta 2 on Apple
  // Silicon, MAP_JIT allows toggling between writable and executable.
  // For x86_64 native or Rosetta, MAP_JIT + pthread_jit_write_protect_np
  // is the supported path.
  return true;
#else
  return true;
#endif
}

struct MappedFileRange {
  uintptr_t region_begin;
  uintptr_t region_end;
};

std::vector<MappedFileRange> mapped_file_ranges;
std::mutex g_mapped_file_ranges_mutex;

// Track shm file names for cleanup on exit
std::vector<std::string> g_shm_file_names;
std::mutex g_shm_file_names_mutex;
static bool g_cleanup_handlers_installed = false;

#if !XE_PLATFORM_ANDROID
static void CleanupAtExit() {
  for (const auto& name : g_shm_file_names) {
    shm_unlink(name.c_str());
  }
}

static void InstallCleanupHandlers() {
  if (g_cleanup_handlers_installed) {
    return;
  }
  g_cleanup_handlers_installed = true;

  std::atexit(CleanupAtExit);
  std::at_quick_exit(CleanupAtExit);
}
#endif  // !XE_PLATFORM_ANDROID

void* AllocFixed(void* base_address, size_t length,
                 AllocationType allocation_type, PageAccess access) {
  // mmap does not support reserve / commit, so ignore allocation_type.
  uint32_t prot = ToPosixProtectFlags(access);
  int flags = MAP_PRIVATE | MAP_ANONYMOUS;

  if (base_address != nullptr) {
    if (allocation_type == AllocationType::kCommit) {
      if (Protect(base_address, length, access)) {
        return base_address;
      }
      return nullptr;
    }
#if XE_PLATFORM_MAC
    // macOS does not have MAP_FIXED_NOREPLACE. Use MAP_FIXED but first check
    // if the region is available using mach_vm_region.
    mach_vm_address_t check_addr = (mach_vm_address_t)base_address;
    mach_vm_size_t check_size = 0;
    vm_region_basic_info_data_64_t info;
    mach_msg_type_number_t info_count = VM_REGION_BASIC_INFO_COUNT_64;
    mach_port_t object_name;
    kern_return_t kr = mach_vm_region(
        mach_task_self(), &check_addr, &check_size,
        VM_REGION_BASIC_INFO_64, (vm_region_info_t)&info,
        &info_count, &object_name);
    if (kr == KERN_SUCCESS && check_addr <= (mach_vm_address_t)base_address &&
        check_addr + check_size > (mach_vm_address_t)base_address) {
      // Region is already mapped.
      return nullptr;
    }
    flags |= MAP_FIXED;
#else
    flags |= MAP_FIXED_NOREPLACE;
#endif
  }

  void* result = mmap(base_address, length, prot, flags, -1, 0);

  if (result != MAP_FAILED) {
    return result;
  }
  return nullptr;
}

bool DeallocFixed(void* base_address, size_t length,
                  DeallocationType deallocation_type) {
  const auto region_begin = reinterpret_cast<uintptr_t>(base_address);
  const uintptr_t region_end =
      reinterpret_cast<uintptr_t>(base_address) + length;

  std::lock_guard guard(g_mapped_file_ranges_mutex);
  for (const auto& mapped_range : mapped_file_ranges) {
    if (region_begin >= mapped_range.region_begin &&
        region_end <= mapped_range.region_end) {
      switch (deallocation_type) {
        case DeallocationType::kDecommit:
          return Protect(base_address, length, PageAccess::kNoAccess);
        case DeallocationType::kRelease:
          return false;
        default:
          assert_unhandled_case(deallocation_type);
      }
    }
  }

  switch (deallocation_type) {
    case DeallocationType::kDecommit:
      return Protect(base_address, length, PageAccess::kNoAccess);
    case DeallocationType::kRelease:
      return munmap(base_address, length) == 0;
    default:
      assert_unhandled_case(deallocation_type);
  }
}

bool Protect(void* base_address, size_t length, PageAccess access,
             PageAccess* out_old_access) {
  if (out_old_access) {
    size_t length_copy = length;
    QueryProtect(base_address, length_copy, *out_old_access);
  }

  uint32_t prot = ToPosixProtectFlags(access);
  
  // Apple Silicon uses 16KB pages, but Xenia guest memory (Xbox) uses 4K pages.
  // mprotect requires the address to be aligned to the host page size.
  uintptr_t page_mask = page_size() - 1;
  uintptr_t addr = reinterpret_cast<uintptr_t>(base_address);
  uintptr_t aligned_addr = addr & ~page_mask;
  size_t aligned_len = ((addr + length + page_mask) & ~page_mask) - aligned_addr;

  if (mprotect((void*)aligned_addr, aligned_len, prot) == 0) {
    return true;
  }
  fprintf(stderr, "mprotect failed for address %p (aligned %p), length %zu (aligned %zu), prot %u: %s\n", base_address, (void*)aligned_addr, length, aligned_len, prot, strerror(errno));
  return false;
}

bool QueryProtect(void* base_address, size_t& length, PageAccess& access_out) {
#if XE_PLATFORM_MAC
  // Use Mach VM APIs on macOS instead of /proc/self/maps
  mach_vm_address_t addr = (mach_vm_address_t)base_address;
  mach_vm_size_t size = 0;
  vm_region_basic_info_data_64_t info;
  mach_msg_type_number_t info_count = VM_REGION_BASIC_INFO_COUNT_64;
  mach_port_t object_name;

  kern_return_t kr = mach_vm_region(
      mach_task_self(), &addr, &size,
      VM_REGION_BASIC_INFO_64, (vm_region_info_t)&info,
      &info_count, &object_name);

  if (kr != KERN_SUCCESS) {
    return false;
  }

  // Convert Mach VM protection to Xenia PageAccess
  bool readable = (info.protection & VM_PROT_READ) != 0;
  bool writable = (info.protection & VM_PROT_WRITE) != 0;
  bool executable = (info.protection & VM_PROT_EXECUTE) != 0;

  if (readable && writable && executable) {
    access_out = PageAccess::kExecuteReadWrite;
  } else if (readable && executable) {
    access_out = PageAccess::kExecuteReadOnly;
  } else if (readable && writable) {
    access_out = PageAccess::kReadWrite;
  } else if (readable) {
    access_out = PageAccess::kReadOnly;
  } else {
    access_out = PageAccess::kNoAccess;
  }

  length = size - ((mach_vm_address_t)base_address - addr);
  return true;
#else
  // Linux: read /proc/self/maps
  std::ifstream memory_maps;
  memory_maps.open("/proc/self/maps", std::ios_base::in);
  std::string maps_entry_string;

  while (std::getline(memory_maps, maps_entry_string)) {
    std::stringstream entry_stream(maps_entry_string);
    uintptr_t map_region_begin, map_region_end;
    char separator;
    char protection[5];

    entry_stream >> std::hex >> map_region_begin >> separator >>
        map_region_end >> protection;

    if (map_region_begin <= reinterpret_cast<uintptr_t>(base_address) &&
        map_region_end > reinterpret_cast<uintptr_t>(base_address)) {
      length = map_region_end - reinterpret_cast<uintptr_t>(base_address);

      access_out = ToXeniaProtectFlags(protection);

      while (std::getline(memory_maps, maps_entry_string)) {
        std::stringstream next_entry_stream(maps_entry_string);
        uintptr_t next_map_region_begin, next_map_region_end;
        char next_protection[5];

        next_entry_stream >> std::hex >> next_map_region_begin >> separator >>
            next_map_region_end >> next_protection;
        if (map_region_end == next_map_region_begin &&
            access_out == ToXeniaProtectFlags(next_protection)) {
          length =
              next_map_region_end - reinterpret_cast<uintptr_t>(base_address);
          continue;
        }
        break;
      }

      memory_maps.close();
      return true;
    }
  }

  memory_maps.close();
  return false;
#endif
}

FileMappingHandle CreateFileMappingHandle(const std::filesystem::path& path,
                                          size_t length, PageAccess access,
                                          bool commit) {
#if XE_PLATFORM_ANDROID
  // TODO(Triang3l): Check if memfd can be used instead on API 30+.
  if (android_ASharedMemory_create_) {
    int sharedmem_fd = android_ASharedMemory_create_(path.c_str(), length);
    return sharedmem_fd >= 0 ? sharedmem_fd : kFileMappingHandleInvalid;
  }

  // Use /dev/ashmem on API versions below 26, which added ASharedMemory.
  // /dev/ashmem was disabled on API 29 for apps targeting it.
  // https://chromium.googlesource.com/chromium/src/+/master/third_party/ashmem/ashmem-dev.c
  int ashmem_fd = open("/" ASHMEM_NAME_DEF, O_RDWR);
  if (ashmem_fd < 0) {
    return kFileMappingHandleInvalid;
  }
  char ashmem_name[ASHMEM_NAME_LEN];
  strlcpy(ashmem_name, path.c_str(), xe::countof(ashmem_name));
  if (ioctl(ashmem_fd, ASHMEM_SET_NAME, ashmem_name) < 0 ||
      ioctl(ashmem_fd, ASHMEM_SET_SIZE, length) < 0) {
    close(ashmem_fd);
    return kFileMappingHandleInvalid;
  }
  return ashmem_fd;
#else
  int oflag;
  switch (access) {
    case PageAccess::kNoAccess:
      oflag = 0;
      break;
    case PageAccess::kReadOnly:
    case PageAccess::kExecuteReadOnly:
      oflag = O_RDONLY;
      break;
    case PageAccess::kReadWrite:
    case PageAccess::kExecuteReadWrite:
      oflag = O_RDWR;
      break;
    default:
      assert_always();
      return kFileMappingHandleInvalid;
  }
  oflag |= O_CREAT;
  auto full_path = "/" / path;
  int ret = shm_open(full_path.c_str(), oflag, 0777);
  if (ret < 0) {
    return kFileMappingHandleInvalid;
  }
  // macOS uses ftruncate (off_t is 64-bit natively), Linux has ftruncate64
#if XE_PLATFORM_MAC
  if (ftruncate(ret, length) < 0) {
#else
  if (ftruncate64(ret, length) < 0) {
#endif
    close(ret);
    shm_unlink(full_path.c_str());
    return kFileMappingHandleInvalid;
  }
  // Track for cleanup on abnormal exit and install cleanup handlers
  {
    std::lock_guard guard(g_shm_file_names_mutex);
    g_shm_file_names.push_back(full_path.string());
  }
  InstallCleanupHandlers();
  return ret;
#endif
}

void CloseFileMappingHandle(FileMappingHandle handle,
                            const std::filesystem::path& path) {
  close(handle);
#if !XE_PLATFORM_ANDROID
  auto full_path = "/" / path;
  shm_unlink(full_path.c_str());
  // Remove from tracking
  {
    std::lock_guard guard(g_shm_file_names_mutex);
    auto it = std::find(g_shm_file_names.begin(), g_shm_file_names.end(),
                        full_path.string());
    if (it != g_shm_file_names.end()) {
      g_shm_file_names.erase(it);
    }
  }
#endif
}

void* MapFileView(FileMappingHandle handle, void* base_address, size_t length,
                  PageAccess access, size_t file_offset) {
  uint32_t prot = ToPosixProtectFlags(access);

  int flags = MAP_SHARED;
  if (base_address != nullptr) {
#if XE_PLATFORM_MAC
    // macOS does not have MAP_FIXED_NOREPLACE. We must check if the region
    // is available before using MAP_FIXED, otherwise we silently overwrite
    // existing mappings (system libraries, frameworks, etc.)
    mach_vm_address_t check_addr = (mach_vm_address_t)base_address;
    mach_vm_size_t check_size = 0;
    vm_region_basic_info_data_64_t info;
    mach_msg_type_number_t info_count = VM_REGION_BASIC_INFO_COUNT_64;
    mach_port_t object_name;
    kern_return_t kr = mach_vm_region(
        mach_task_self(), &check_addr, &check_size,
        VM_REGION_BASIC_INFO_64, (vm_region_info_t)&info,
        &info_count, &object_name);
    if (kr == KERN_SUCCESS) {
      // Check if any existing mapping overlaps our requested range
      uintptr_t req_begin = (uintptr_t)base_address;
      uintptr_t req_end = req_begin + length;
      uintptr_t map_begin = (uintptr_t)check_addr;
      uintptr_t map_end = map_begin + (uintptr_t)check_size;
      if (map_begin < req_end && map_end > req_begin) {
        // Region is already partially or fully mapped - don't overwrite.
        return nullptr;
      }
    }
    flags |= MAP_FIXED;
#else
    flags |= MAP_FIXED_NOREPLACE;
#endif
  }

  void* result = mmap(base_address, length, prot, flags, handle, file_offset);

  if (result != MAP_FAILED) {
    std::lock_guard guard(g_mapped_file_ranges_mutex);
    mapped_file_ranges.push_back(
        {reinterpret_cast<uintptr_t>(result),
         reinterpret_cast<uintptr_t>(result) + length});
    return result;
  }

  return nullptr;
}

bool UnmapFileView(FileMappingHandle handle, void* base_address,
                   size_t length) {
  std::lock_guard guard(g_mapped_file_ranges_mutex);
  for (auto mapped_range = mapped_file_ranges.begin();
       mapped_range != mapped_file_ranges.end();) {
    if (mapped_range->region_begin ==
            reinterpret_cast<uintptr_t>(base_address) &&
        mapped_range->region_end ==
            reinterpret_cast<uintptr_t>(base_address) + length) {
      mapped_file_ranges.erase(mapped_range);
      return munmap(base_address, length) == 0;
    }
    ++mapped_range;
  }
  // TODO: Implement partial file unmapping.
  assert_always("Error: Partial unmapping of files not yet supported.");
  return munmap(base_address, length) == 0;
}

}  // namespace memory
}  // namespace xe
