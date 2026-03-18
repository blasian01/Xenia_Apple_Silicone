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

#include "xenia/base/logging.h"

namespace xe {
namespace ui {

MacWindowedAppContext::MacWindowedAppContext() = default;

MacWindowedAppContext::~MacWindowedAppContext() {
  SDL_Quit();
}

bool MacWindowedAppContext::Initialize() {
  if (SDL_Init(SDL_INIT_VIDEO | SDL_INIT_EVENTS | SDL_INIT_GAMECONTROLLER) !=
      0) {
    XELOGE("Failed to initialize SDL: {}", SDL_GetError());
    return false;
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
