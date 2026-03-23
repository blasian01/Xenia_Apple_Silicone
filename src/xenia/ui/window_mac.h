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

  static void DispatchSDLEvent(const SDL_Event& event);

  // Window overrides
  uint32_t GetMediumDpi() const override { return 96; }

  SDL_Window* sdl_window() const { return sdl_window_; }

  bool OpenImpl() override;
  void RequestCloseImpl() override;

  std::unique_ptr<Surface> CreateSurfaceImpl(
      Surface::TypeFlags type_flags) override;

  void ApplyNewMainMenu(MenuItem* old_main_menu) override;
  void CompleteMainMenuItemsUpdateImpl() override;

  void RequestPaintImpl() override;

 private:
  void DestroyWindowResources();
  void HandleWindowEvent(const SDL_WindowEvent& event);
  void HandleDropEvent(const SDL_DropEvent& event);
  void HandlePaintRequest();
  VirtualKey TranslateSDLKey(const SDL_Keysym& keysym);

  uint32_t window_id_ = 0;
  SDL_Window* sdl_window_ = nullptr;
  void* metal_view_ = nullptr;
};

}  // namespace ui
}  // namespace xe

#endif  // XENIA_UI_WINDOW_MAC_H_
