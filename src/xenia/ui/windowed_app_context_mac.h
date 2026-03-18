/**
 ******************************************************************************
 * Xenia : Xbox 360 Emulator Research Project                                 *
 ******************************************************************************
 * Copyright 2025 Ben Vanik. All rights reserved.                             *
 * Released under the BSD license - see LICENSE in the root for more details. *
 ******************************************************************************
 */

#ifndef XENIA_UI_WINDOWED_APP_CONTEXT_MAC_H_
#define XENIA_UI_WINDOWED_APP_CONTEXT_MAC_H_

#include <deque>
#include <functional>
#include <mutex>
#include <utility>

#include "xenia/ui/windowed_app_context.h"

namespace xe {
namespace ui {

class MacWindowedAppContext final : public WindowedAppContext {
 public:
  MacWindowedAppContext();
  ~MacWindowedAppContext();

  bool Initialize();

  // WindowedAppContext implementation
  void NotifyUILoopOfPendingFunctions() override;
  void PlatformQuitFromUIThread() override;

  // Process pending UI thread functions. Called from the event loop.
  void ProcessPendingFunctions();

  bool is_running() const { return is_running_; }

 private:
  bool is_running_ = true;
};

}  // namespace ui
}  // namespace xe

#endif  // XENIA_UI_WINDOWED_APP_CONTEXT_MAC_H_
