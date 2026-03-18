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
#include "xenia/ui/windowed_app_context_mac.h"

extern "C" int main(int argc, char** argv) {
  int result;

  {
    xe::ui::MacWindowedAppContext app_context;

    std::unique_ptr<xe::ui::WindowedApp> app =
        xe::ui::GetWindowedAppCreator()(app_context);

    cvar::ParseLaunchArguments(argc, argv,
                               app->GetPositionalOptionsUsage(),
                               app->GetPositionalOptions());

    // Initialize logging. Needs parsed cvars.
    xe::InitializeLogging(app->GetName());

    if (app->OnInitialize()) {
      // SDL event loop
      bool running = true;
      while (running) {
        SDL_Event event;
        while (SDL_PollEvent(&event)) {
          switch (event.type) {
            case SDL_QUIT:
              running = false;
              break;
            case SDL_WINDOWEVENT:
              if (event.window.event == SDL_WINDOWEVENT_CLOSE) {
                running = false;
              }
              break;
            default:
              break;
          }
        }

        // Process any pending UI thread functions
        app_context.ExecutePendingFunctionsFromUIThread();

        // Small sleep to prevent 100% CPU usage
        SDL_Delay(1);
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
