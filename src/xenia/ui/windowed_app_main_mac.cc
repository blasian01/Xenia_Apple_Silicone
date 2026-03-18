/**
 ******************************************************************************
 * Xenia : Xbox 360 Emulator Research Project                                 *
 ******************************************************************************
 * Copyright 2025 Ben Vanik. All rights reserved.                             *
 * Released under the BSD license - see LICENSE in the root for more details. *
 ******************************************************************************
 */

#include <SDL2/SDL.h>

#include <cstdio>
#include <cstdlib>
#include <memory>

#include "xenia/base/cvar.h"
#include "xenia/base/logging.h"
#include "xenia/ui/windowed_app.h"
#include "xenia/ui/window_mac.h"
#include "xenia/ui/windowed_app_context_mac.h"

extern "C" int main(int argc, char** argv) {
  int result;

  {
    xe::ui::MacWindowedAppContext app_context;
    if (!app_context.Initialize()) {
      return EXIT_FAILURE;
    }

    std::unique_ptr<xe::ui::WindowedApp> app =
        xe::ui::GetWindowedAppCreator()(app_context);

    cvar::ParseLaunchArguments(argc, argv,
                               app->GetPositionalOptionsUsage(),
                               app->GetPositionalOptions());

    // Initialize logging. Needs parsed cvars.
    xe::InitializeLogging(app->GetName());

    if (app->OnInitialize()) {
      while (app_context.is_running()) {
        SDL_Event event;
        if (SDL_WaitEventTimeout(&event, 16)) {
          do {
            if (event.type == SDL_QUIT) {
              app_context.QuitFromUIThread();
              break;
            }
            xe::ui::MacWindow::DispatchSDLEvent(event);
          } while (app_context.is_running() && SDL_PollEvent(&event));
        }

        app_context.ProcessPendingFunctions();
      }
      result = EXIT_SUCCESS;
    } else {
      result = EXIT_FAILURE;
    }

    app->InvokeOnDestroy();
  }

  // Logging may still be needed in the destructors.
  xe::ShutdownLogging();

  return result;
}
