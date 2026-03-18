/**
 ******************************************************************************
 * Xenia : Xbox 360 Emulator Research Project                                 *
 ******************************************************************************
 * Copyright 2025 Ben Vanik. All rights reserved.                             *
 * Released under the BSD license - see LICENSE in the root for more details. *
 ******************************************************************************
 */

#include "xenia/ui/windowed_app_context_mac.h"

#include <SDL2/SDL.h>
#include <SDL2/SDL_vulkan.h>

#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <limits.h>
#include <mach-o/dyld.h>

#include "xenia/base/logging.h"

namespace xe {
namespace ui {

namespace {

std::filesystem::path GetBundledPath(const std::filesystem::path& relative_path) {
  char executable_path[PATH_MAX];
  uint32_t executable_path_capacity = sizeof(executable_path);
  if (_NSGetExecutablePath(executable_path, &executable_path_capacity) != 0) {
    return {};
  }

  std::filesystem::path bundle_path = executable_path;
  return bundle_path.parent_path().parent_path() / relative_path;
}

}  // namespace

MacWindowedAppContext::MacWindowedAppContext() = default;

MacWindowedAppContext::~MacWindowedAppContext() {
  if (vulkan_library_loaded_) {
    SDL_Vulkan_UnloadLibrary();
  }
  SDL_Quit();
}

bool MacWindowedAppContext::Initialize() {
  if (SDL_Init(SDL_INIT_VIDEO | SDL_INIT_EVENTS | SDL_INIT_GAMECONTROLLER) !=
      0) {
    XELOGE("Failed to initialize SDL: {}", SDL_GetError());
    return false;
  }

  std::filesystem::path icd_path =
      GetBundledPath("Resources/vulkan/icd.d/MoltenVK_icd.json");
  if (!icd_path.empty() && std::filesystem::exists(icd_path)) {
    std::string icd_path_string = icd_path.string();
    setenv("VK_DRIVER_FILES", icd_path_string.c_str(), 1);
    setenv("VK_ICD_FILENAMES", icd_path_string.c_str(), 1);
  }

  std::filesystem::path bundled_vulkan_loader_path =
      GetBundledPath("Frameworks/libvulkan.1.dylib");
  if (!bundled_vulkan_loader_path.empty() &&
      std::filesystem::exists(bundled_vulkan_loader_path)) {
    if (SDL_Vulkan_LoadLibrary(bundled_vulkan_loader_path.string().c_str()) ==
        0) {
      vulkan_library_loaded_ = true;
    } else {
      XELOGW("Failed to preload bundled Vulkan loader: {}", SDL_GetError());
    }
  }

  return true;
}

void MacWindowedAppContext::NotifyUILoopOfPendingFunctions() {
  // Push a user event to wake up the SDL event loop.
  SDL_Event event;
  event.type = SDL_USEREVENT;
  event.user.code = 0;
  event.user.data1 = nullptr;
  event.user.data2 = nullptr;
  SDL_PushEvent(&event);
}

void MacWindowedAppContext::PlatformQuitFromUIThread() {
  is_running_ = false;
}

void MacWindowedAppContext::ProcessPendingFunctions() {
  ExecutePendingFunctionsFromUIThread();
}

}  // namespace ui
}  // namespace xe
