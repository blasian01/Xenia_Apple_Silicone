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
#include <SDL2/SDL_video.h>
#include <objc/runtime.h>
#include <objc/message.h>

#include <unordered_map>

#include "xenia/base/logging.h"
#include "xenia/ui/surface_mac.h"

#import <AppKit/AppKit.h>
#import <QuartzCore/QuartzCore.h>

namespace xe {
namespace ui {

class MacMenuItem;

}  // namespace ui
}  // namespace xe

@interface XeniaMenuItemTarget : NSObject
@property(nonatomic, assign) xe::ui::MacMenuItem* menuItem;
- (void)onMenuItemAction:(id)sender;
@end

namespace xe {
namespace ui {

namespace {

constexpr int kSDLUserEventPaintRequest = 1;

std::unordered_map<uint32_t, MacWindow*>& GetMacWindowRegistry() {
  static std::unordered_map<uint32_t, MacWindow*> registry;
  return registry;
}

MacWindow* GetMacWindowByID(uint32_t window_id) {
  auto& registry = GetMacWindowRegistry();
  auto it = registry.find(window_id);
  return it == registry.end() ? nullptr : it->second;
}

}  // namespace

class MacMenuItem : public MenuItem {
 public:
  MacMenuItem(Type type, const std::string& text, const std::string& hotkey,
              std::function<void()> callback)
      : MenuItem(type, text, hotkey, std::move(callback)) {
    
    NSString* title = [NSString stringWithUTF8String:text.c_str()];
    title = [title stringByReplacingOccurrencesOfString:@"&" withString:@""];
    
    if (type == Type::kSeparator) {
      ns_item_ = [[NSMenuItem separatorItem] retain];
    } else {
      ns_item_ = [[NSMenuItem alloc] initWithTitle:title action:nil keyEquivalent:@""];
      if (type == Type::kPopup) {
        NSMenu* submenu = [[NSMenu alloc] initWithTitle:title];
        [ns_item_ setSubmenu:submenu];
        [submenu release];
      } else {
        target_ = [[XeniaMenuItemTarget alloc] init];
        target_.menuItem = this;
        [ns_item_ setTarget:target_];
        [ns_item_ setAction:@selector(onMenuItemAction:)];
        if (!hotkey.empty()) {
            NSString* hk = [NSString stringWithUTF8String:hotkey.c_str()];
            hk = [[hk lowercaseString] stringByReplacingOccurrencesOfString:@"ctrl+" withString:@""];
            if ([hk length] == 1) {
                [ns_item_ setKeyEquivalent:hk];
            }
        }
      }
    }
  }

  ~MacMenuItem() override {
    if (target_) {
      target_.menuItem = nullptr;
      [target_ release];
    }
    if (ns_item_.menu) {
      [ns_item_.menu removeItem:ns_item_];
    }
    [ns_item_ release];
  }

  void OnChildAdded(MenuItem* generic_child_item) override {
    auto child_item = static_cast<MacMenuItem*>(generic_child_item);
    if (ns_item_.submenu && child_item) {
        if (child_item->ns_item().menu) {
            [child_item->ns_item().menu removeItem:child_item->ns_item()];
        }
        [ns_item_.submenu addItem:child_item->ns_item()];
    }
  }

  void OnChildRemoved(MenuItem* generic_child_item) override {
    auto child_item = static_cast<MacMenuItem*>(generic_child_item);
    if (ns_item_.submenu && child_item) {
        [ns_item_.submenu removeItem:child_item->ns_item()];
    }
  }

  void SetEnabled(bool enabled) override {
    [ns_item_ setEnabled:enabled ? YES : NO];
  }

  void OnSelectedPublic() { OnSelected(); }

  NSMenuItem* ns_item() const { return ns_item_; }
  const std::vector<MenuItemPtr>& get_children() const { return children_; }

 private:
  NSMenuItem* ns_item_ = nil;
  XeniaMenuItemTarget* target_ = nil;
};

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
  return std::unique_ptr<MenuItem>(
      new MacMenuItem(type, text, hotkey, std::move(callback)));
}

}  // namespace ui
}  // namespace xe

@implementation XeniaMenuItemTarget
- (void)onMenuItemAction:(id)sender {
  if (_menuItem) {
    _menuItem->OnSelectedPublic();
  }
}
@end

namespace xe {
namespace ui {

MacWindow::MacWindow(WindowedAppContext& app_context, const std::string& title,
                     uint32_t width, uint32_t height)
    : Window(app_context, title, width, height) {}

MacWindow::~MacWindow() {
  DestroyWindowResources();
}

void MacWindow::DispatchSDLEvent(const SDL_Event& event) {
  switch (event.type) {
    case SDL_WINDOWEVENT: {
      if (auto* window = GetMacWindowByID(event.window.windowID)) {
        window->HandleWindowEvent(event.window);
      }
    } break;
    case SDL_DROPFILE: {
      if (auto* window = GetMacWindowByID(event.drop.windowID)) {
        window->HandleDropEvent(event.drop);
      }
    } break;
    case SDL_USEREVENT: {
      if (event.user.code != kSDLUserEventPaintRequest || !event.user.data1) {
        break;
      }
      const auto window_id =
          static_cast<uint32_t>(reinterpret_cast<uintptr_t>(event.user.data1));
      if (auto* window = GetMacWindowByID(window_id)) {
        window->HandlePaintRequest();
      }
    } break;
    default:
      break;
  }
}

void MacWindow::DestroyWindowResources() {
  if (window_id_) {
    GetMacWindowRegistry().erase(window_id_);
    window_id_ = 0;
  }
  if (metal_view_) {
    SDL_Metal_DestroyView(metal_view_);
    metal_view_ = nullptr;
  }
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

  window_id_ = SDL_GetWindowID(sdl_window_);
  GetMacWindowRegistry()[window_id_] = this;

  metal_view_ = SDL_Metal_CreateView(sdl_window_);
  if (!metal_view_) {
    XELOGE("Failed to create Metal view from SDL window: {}", SDL_GetError());
    DestroyWindowResources();
    return false;
  }

  {
    WindowDestructionReceiver destruction_receiver(this);
    int drawable_width = 0;
    int drawable_height = 0;
    SDL_Vulkan_GetDrawableSize(sdl_window_, &drawable_width, &drawable_height);
    OnActualSizeUpdate(drawable_width > 0 ? uint32_t(drawable_width) : 0,
                       drawable_height > 0 ? uint32_t(drawable_height) : 0,
                       destruction_receiver);
    if (destruction_receiver.IsWindowDestroyed()) {
      return true;
    }
    OnFocusUpdate(SDL_GetKeyboardFocus() == sdl_window_, destruction_receiver);
    if (destruction_receiver.IsWindowDestroyed()) {
      return true;
    }
  }

  if (GetMainMenu()) {
    CompleteMainMenuItemsUpdateImpl();
  }

  return true;
}

void MacWindow::RequestCloseImpl() {
  if (!sdl_window_) {
    return;
  }

  WindowDestructionReceiver destruction_receiver(this);
  OnBeforeClose(destruction_receiver);
  if (destruction_receiver.IsWindowDestroyed()) {
    return;
  }

  DestroyWindowResources();
  OnAfterClose();
}

void MacWindow::ApplyNewMainMenu(MenuItem* old_main_menu) {
  CompleteMainMenuItemsUpdateImpl();
}

void MacWindow::CompleteMainMenuItemsUpdateImpl() {
  MenuItem* root = GetMainMenu();
  if (!root) {
    [NSApp setMainMenu:nil];
    return;
  }

  MacMenuItem* mac_root = static_cast<MacMenuItem*>(root);
  
  NSMenu* current_main_menu = [NSApp mainMenu];
  NSMenuItem* app_menu_item = nil;
  if (current_main_menu && [current_main_menu numberOfItems] > 0) {
    app_menu_item = [[current_main_menu itemAtIndex:0] retain];
    [current_main_menu removeItem:app_menu_item];
  }

  NSMenu* new_main_menu = [[NSMenu alloc] initWithTitle:@"Main Menu"];
  
  if (app_menu_item) {
    [new_main_menu addItem:app_menu_item];
    [app_menu_item release];
  } else {
    // Keep Xenia menu fallback
    NSMenuItem* app_item = [[NSMenuItem alloc] initWithTitle:@"Xenia" action:nil keyEquivalent:@""];
    NSMenu* app_menu = [[NSMenu alloc] initWithTitle:@"Xenia"];
    [app_item setSubmenu:app_menu];
    [app_menu addItemWithTitle:@"Quit Xenia" action:@selector(terminate:) keyEquivalent:@"q"];
    [new_main_menu addItem:app_item];
    [app_item release];
    [app_menu release];
  }

  for (auto& child_ptr : mac_root->get_children()) {
    MacMenuItem* child = static_cast<MacMenuItem*>(child_ptr.get());
    if (child && child->ns_item()) {
      if (child->ns_item().menu) {
        [child->ns_item().menu removeItem:child->ns_item()];
      }
      [new_main_menu addItem:child->ns_item()];
    }
  }

  [NSApp setMainMenu:new_main_menu];
  [new_main_menu release];
}

std::unique_ptr<Surface> MacWindow::CreateSurfaceImpl(
    Surface::TypeFlags type_flags) {
  if (!(type_flags & Surface::kTypeFlag_MacMetalLayer)) {
    return nullptr;
  }

  int w, h;
  SDL_Vulkan_GetDrawableSize(sdl_window_, &w, &h);

  void* layer = SDL_Metal_GetLayer(metal_view_);

  // MoltenVK 1.2+ has a bug where it attempts to query the layer's delegate (which is the SDL_cocoametalview),
  // casts it to an NSView, and then sends the `-delegate` selector to it, causing an unrecognized selector crash.
  // We bypass this entirely by simply unsetting the delegate of the CAMetalLayer before handing it to Vulkan.
  CALayer* mtlLayer = (CALayer*)layer;
  mtlLayer.delegate = nil;

  return std::make_unique<MacSurface>(layer,
                                      static_cast<uint32_t>(w),
                                      static_cast<uint32_t>(h));
}

void MacWindow::RequestPaintImpl() {
  SDL_Event event;
  event.type = SDL_USEREVENT;
  event.user.code = kSDLUserEventPaintRequest;
  event.user.data1 = reinterpret_cast<void*>(uintptr_t(window_id_));
  event.user.data2 = nullptr;
  SDL_PushEvent(&event);
}

void MacWindow::HandleWindowEvent(const SDL_WindowEvent& event) {
  switch (event.event) {
    case SDL_WINDOWEVENT_CLOSE: {
      RequestClose();
    } break;
    case SDL_WINDOWEVENT_EXPOSED: {
      HandlePaintRequest();
    } break;
    case SDL_WINDOWEVENT_SIZE_CHANGED:
    case SDL_WINDOWEVENT_RESIZED: {
      WindowDestructionReceiver destruction_receiver(this);
      int drawable_width = 0;
      int drawable_height = 0;
      SDL_Vulkan_GetDrawableSize(sdl_window_, &drawable_width, &drawable_height);
      OnActualSizeUpdate(drawable_width > 0 ? uint32_t(drawable_width) : 0,
                         drawable_height > 0 ? uint32_t(drawable_height) : 0,
                         destruction_receiver);
    } break;
    case SDL_WINDOWEVENT_FOCUS_GAINED:
    case SDL_WINDOWEVENT_FOCUS_LOST: {
      WindowDestructionReceiver destruction_receiver(this);
      OnFocusUpdate(event.event == SDL_WINDOWEVENT_FOCUS_GAINED,
                    destruction_receiver);
    } break;
    default:
      break;
  }
}

void MacWindow::HandleDropEvent(const SDL_DropEvent& event) {
  if (!event.file) {
    return;
  }
  FileDropEvent drop_event(this, event.file);
  WindowDestructionReceiver destruction_receiver(this);
  OnFileDrop(drop_event, destruction_receiver);
  SDL_free(event.file);
}

void MacWindow::HandlePaintRequest() { OnPaint(); }

}  // namespace ui
}  // namespace xe
