/**
 ******************************************************************************
 * Xenia : Xbox 360 Emulator Research Project                                 *
 ******************************************************************************
 * Copyright 2025 Ben Vanik. All rights reserved.                             *
 * Released under the BSD license - see LICENSE in the root for more details. *
 ******************************************************************************
 */

#include "xenia/ui/window_mac.h"

#include <SDL2/SDL.h>
#include <SDL2/SDL_metal.h>
#include <SDL2/SDL_vulkan.h>

#include "xenia/base/logging.h"
#include "xenia/ui/surface_mac.h"

namespace xe {
namespace ui {

std::unique_ptr<Window> Window::Create(WindowedAppContext& app_context,
                                       const std::string_view title,
                                       uint32_t desired_logical_width,
                                       uint32_t desired_logical_height) {
  return std::make_unique<MacWindow>(app_context, std::string(title),
                                     desired_logical_width,
                                     desired_logical_height);
}

std::unique_ptr<ui::MenuItem> MenuItem::Create(Type type,
                                               const std::string& text,
                                               const std::string& hotkey,
                                               std::function<void()> callback) {
  // Basic MenuItem — macOS does not use native NSMenu wrappers at this stage.
  return std::unique_ptr<MenuItem>(
      new MenuItem(type, text, hotkey, std::move(callback)));
}

MacWindow::MacWindow(WindowedAppContext& app_context, const std::string& title,
                     uint32_t width, uint32_t height)
    : Window(app_context, title, width, height) {}

MacWindow::~MacWindow() {
  if (sdl_window_) {
    SDL_DestroyWindow(sdl_window_);
    sdl_window_ = nullptr;
  }
}

bool MacWindow::OpenImpl() {
  uint32_t flags = SDL_WINDOW_VULKAN | SDL_WINDOW_RESIZABLE |
                   SDL_WINDOW_ALLOW_HIGHDPI | SDL_WINDOW_SHOWN;

  sdl_window_ = SDL_CreateWindow(
      GetTitle().c_str(), SDL_WINDOWPOS_CENTERED, SDL_WINDOWPOS_CENTERED,
      static_cast<int>(GetDesiredLogicalWidth()),
      static_cast<int>(GetDesiredLogicalHeight()), flags);

  if (!sdl_window_) {
    XELOGE("Failed to create SDL window: {}", SDL_GetError());
    return false;
  }

  // Get the Metal layer from SDL's Vulkan window
  metal_layer_ = SDL_Metal_CreateView(sdl_window_);
  if (!metal_layer_) {
    XELOGE("Failed to create Metal view from SDL window: {}", SDL_GetError());
    return false;
  }

  return true;
}

void MacWindow::RequestCloseImpl() {
  if (sdl_window_) {
    SDL_Event event;
    event.type = SDL_QUIT;
    SDL_PushEvent(&event);
  }
}

std::unique_ptr<Surface> MacWindow::CreateSurfaceImpl(
    Surface::TypeFlags type_flags) {
  if (!(type_flags & Surface::kTypeFlag_MacMetalLayer)) {
    return nullptr;
  }

  int w, h;
  SDL_Vulkan_GetDrawableSize(sdl_window_, &w, &h);

  return std::make_unique<MacSurface>(metal_layer_,
                                      static_cast<uint32_t>(w),
                                      static_cast<uint32_t>(h));
}

void MacWindow::RequestPaintImpl() {
  // Trigger a redraw through the SDL event loop
  SDL_Event event;
  event.type = SDL_USEREVENT;
  event.user.code = 1;  // Paint request
  event.user.data1 = nullptr;
  event.user.data2 = nullptr;
  SDL_PushEvent(&event);
}

}  // namespace ui
}  // namespace xe
