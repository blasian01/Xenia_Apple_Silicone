/**
 ******************************************************************************
 * Xenia : Xbox 360 Emulator Research Project                                 *
 ******************************************************************************
 * Copyright 2025 Ben Vanik. All rights reserved.                             *
 * Released under the BSD license - see LICENSE in the root for more details. *
 ******************************************************************************
 */

#ifndef XENIA_UI_WINDOW_MAC_H_
#define XENIA_UI_WINDOW_MAC_H_

#include <SDL2/SDL.h>

#include <memory>
#include <string>

#include "xenia/ui/surface_mac.h"
#include "xenia/ui/window.h"
#include "xenia/ui/windowed_app_context_mac.h"

namespace xe {
namespace ui {

class MacWindow : public Window {
 public:
  MacWindow(WindowedAppContext& app_context, const std::string& title,
            uint32_t width, uint32_t height);
  ~MacWindow() override;

  // Window overrides
  uint32_t GetMediumDpi() const override { return 96; }

  SDL_Window* sdl_window() const { return sdl_window_; }

  bool OpenImpl() override;
  void RequestCloseImpl() override;

  std::unique_ptr<Surface> CreateSurfaceImpl(
      Surface::TypeFlags type_flags) override;

  void RequestPaintImpl() override;

 private:
  SDL_Window* sdl_window_ = nullptr;
  void* metal_layer_ = nullptr;
};

}  // namespace ui
}  // namespace xe

#endif  // XENIA_UI_WINDOW_MAC_H_
