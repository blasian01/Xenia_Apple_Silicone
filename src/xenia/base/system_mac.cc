/**
 ******************************************************************************
 * Xenia : Xbox 360 Emulator Research Project                                 *
 ******************************************************************************
 * Copyright 2025 Ben Vanik. All rights reserved.                             *
 * Released under the BSD license - see LICENSE in the root for more details. *
 ******************************************************************************
 */

#include "xenia/base/system.h"

#include <cstdlib>
#include <string>

#include "xenia/base/logging.h"
#include "xenia/base/string.h"

namespace xe {

void LaunchWebBrowser(const std::string_view url) {
  std::string command = "open \"" + std::string(url) + "\"";
  system(command.c_str());
}

void LaunchFileExplorer(const std::filesystem::path& path) {
  std::string command = "open \"" + path.string() + "\"";
  system(command.c_str());
}

bool SetProcessPriorityClass(const uint32_t priority_class) {
  // No-op on macOS (same as Linux).
  return true;
}

bool IsUseNexusForGameBarEnabled() {
  // Xbox Game Bar is a Windows-only concept.
  return false;
}

void ShowSimpleMessageBox(SimpleMessageBoxType type, std::string_view message) {
  // Use osascript to display a native macOS dialog.
  std::string escaped;
  escaped.reserve(message.size());
  for (char c : message) {
    if (c == '"') {
      escaped += "\\\"";
    } else if (c == '\\') {
      escaped += "\\\\";
    } else {
      escaped += c;
    }
  }

  const char* icon = "note";
  switch (type) {
    case SimpleMessageBoxType::Warning:
      icon = "caution";
      break;
    case SimpleMessageBoxType::Error:
      icon = "stop";
      break;
    default:
      break;
  }

  std::string command = "osascript -e 'display dialog \"" + escaped +
                        "\" with icon " + icon +
                        " buttons {\"OK\"} default button \"OK\"' "
                        ">/dev/null 2>&1 &";
  system(command.c_str());
}

}  // namespace xe
