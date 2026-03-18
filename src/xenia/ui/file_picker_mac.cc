/**
 ******************************************************************************
 * Xenia : Xbox 360 Emulator Research Project                                 *
 ******************************************************************************
 * Copyright 2025 Ben Vanik. All rights reserved.                             *
 * Released under the BSD license - see LICENSE in the root for more details. *
 ******************************************************************************
 */

#include "xenia/ui/file_picker.h"

#include <cstdlib>
#include <string>

#include "xenia/base/logging.h"

namespace xe {
namespace ui {

class MacFilePicker : public FilePicker {
 public:
  MacFilePicker() = default;

  bool Show(Window* parent_window = nullptr) override {
    std::string script;
    if (mode() == Mode::kOpen) {
      script = "osascript -e 'POSIX path of (choose file";
      if (!title().empty()) {
        script += " with prompt \"" + title() + "\"";
      }
      if (!extensions().empty()) {
        script += " of type {";
        bool first = true;
        for (const auto& ext_pair : extensions()) {
          // ext_pair.second contains the extension pattern like "*.xex"
          if (!first) script += ", ";
          first = false;
          // Strip leading *. if present
          std::string ext = ext_pair.second;
          if (ext.size() > 2 && ext[0] == '*' && ext[1] == '.') {
            ext = ext.substr(2);
          }
          script += "\"" + ext + "\"";
        }
        script += "}";
      }
      script += ")'";
    } else {
      script = "osascript -e 'POSIX path of (choose file name";
      if (!title().empty()) {
        script += " with prompt \"" + title() + "\"";
      }
      script += ")'";
    }

    FILE* pipe = popen(script.c_str(), "r");
    if (!pipe) return false;

    char buffer[4096];
    std::string result;
    while (fgets(buffer, sizeof(buffer), pipe) != nullptr) {
      result += buffer;
    }
    int status = pclose(pipe);

    if (status != 0 || result.empty()) {
      return false;
    }

    // Remove trailing newline
    if (!result.empty() && result.back() == '\n') {
      result.pop_back();
    }

    std::vector<std::filesystem::path> files;
    files.push_back(std::filesystem::path(result));
    set_selected_files(std::move(files));
    return true;
  }
};

std::unique_ptr<FilePicker> FilePicker::Create() {
  return std::make_unique<MacFilePicker>();
}

}  // namespace ui
}  // namespace xe
