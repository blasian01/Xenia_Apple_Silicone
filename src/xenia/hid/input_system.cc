/**
 ******************************************************************************
 * Xenia : Xbox 360 Emulator Research Project                                 *
 ******************************************************************************
 * Copyright 2013 Ben Vanik. All rights reserved.                             *
 * Released under the BSD license - see LICENSE in the root for more details. *
 ******************************************************************************
 */

#include "xenia/base/logging.h"

#include "xenia/hid/input_system.h"

#include <atomic>
#include <chrono>
#include <filesystem>
#include <fstream>
#include <mutex>
#include <sstream>
#include <string>
#include <thread>

#include "xenia/base/cvar.h"
#include "xenia/base/profiling.h"
#include "xenia/hid/hid_flags.h"
#include "xenia/hid/input_driver.h"
#include "xenia/kernel/util/shim_utils.h"

#ifdef XE_PLATFORM_WIN32
#include "xenia/hid/portal/hardware_portal.h"
#endif  // XE_PLATFORM_WIN32

namespace xe {
namespace hid {

DEFINE_bool(vibration, true, "Toggle controller vibration.", "HID");

DEFINE_double(left_stick_deadzone_percentage, 0.0,
              "Defines deadzone level for left stick. Allowed range [0.0-1.0].",
              "HID");
DEFINE_double(
    right_stick_deadzone_percentage, 0.0,
    "Defines deadzone level for right stick. Allowed range [0.0-1.0].", "HID");

DEFINE_path(
    input_script_file, "",
    "Debug aid: poll this file for controller commands and apply them to "
    "the first connected controller of user 0 (the keyboard driver counts). "
    "The file is consumed (deleted) once read. Commands are separated by "
    "whitespace and run in order: 'button[+button...][:milliseconds]' holds "
    "the buttons (a b x y start back guide lb rb lt rt ls rs up down left "
    "right lsup lsdown lsleft lsright rsup rsdown rsleft rsright) and "
    "'wait:milliseconds' pauses.",
    "HID");

namespace {

// Lets automated runs drive a title without a controller or window focus.
class ScriptedInput {
 public:
  static ScriptedInput& Get() {
    static ScriptedInput instance;
    return instance;
  }

  // ORs the currently scripted state into the gamepad state.
  void Apply(X_INPUT_STATE* state) {
    std::call_once(thread_started_, [this]() {
      std::thread([this]() { Run(); }).detach();
    });
    std::lock_guard<std::mutex> lock(mutex_);
    state->packet_number += packet_number_;
    state->gamepad.buttons = state->gamepad.buttons | current_.buttons;
    if (current_.left_trigger) {
      state->gamepad.left_trigger = current_.left_trigger;
    }
    if (current_.right_trigger) {
      state->gamepad.right_trigger = current_.right_trigger;
    }
    if (current_.thumb_lx) state->gamepad.thumb_lx = current_.thumb_lx;
    if (current_.thumb_ly) state->gamepad.thumb_ly = current_.thumb_ly;
    if (current_.thumb_rx) state->gamepad.thumb_rx = current_.thumb_rx;
    if (current_.thumb_ry) state->gamepad.thumb_ry = current_.thumb_ry;
  }

 private:
  struct State {
    uint16_t buttons = 0;
    uint8_t left_trigger = 0;
    uint8_t right_trigger = 0;
    int16_t thumb_lx = 0;
    int16_t thumb_ly = 0;
    int16_t thumb_rx = 0;
    int16_t thumb_ry = 0;
  };

  static bool ParseButton(const std::string& name, State& state) {
    static const struct {
      const char* name;
      uint16_t button;
    } kButtons[] = {
        {"a", X_INPUT_GAMEPAD_A},
        {"b", X_INPUT_GAMEPAD_B},
        {"x", X_INPUT_GAMEPAD_X},
        {"y", X_INPUT_GAMEPAD_Y},
        {"start", X_INPUT_GAMEPAD_START},
        {"back", X_INPUT_GAMEPAD_BACK},
        {"guide", X_INPUT_GAMEPAD_GUIDE},
        {"lb", X_INPUT_GAMEPAD_LEFT_SHOULDER},
        {"rb", X_INPUT_GAMEPAD_RIGHT_SHOULDER},
        {"ls", X_INPUT_GAMEPAD_LEFT_THUMB},
        {"rs", X_INPUT_GAMEPAD_RIGHT_THUMB},
        {"up", X_INPUT_GAMEPAD_DPAD_UP},
        {"down", X_INPUT_GAMEPAD_DPAD_DOWN},
        {"left", X_INPUT_GAMEPAD_DPAD_LEFT},
        {"right", X_INPUT_GAMEPAD_DPAD_RIGHT},
    };
    for (const auto& entry : kButtons) {
      if (name == entry.name) {
        state.buttons |= entry.button;
        return true;
      }
    }
    if (name == "lt") {
      state.left_trigger = 0xFF;
    } else if (name == "rt") {
      state.right_trigger = 0xFF;
    } else if (name == "lsup") {
      state.thumb_ly = INT16_MAX;
    } else if (name == "lsdown") {
      state.thumb_ly = INT16_MIN;
    } else if (name == "lsleft") {
      state.thumb_lx = INT16_MIN;
    } else if (name == "lsright") {
      state.thumb_lx = INT16_MAX;
    } else if (name == "rsup") {
      state.thumb_ry = INT16_MAX;
    } else if (name == "rsdown") {
      state.thumb_ry = INT16_MIN;
    } else if (name == "rsleft") {
      state.thumb_rx = INT16_MIN;
    } else if (name == "rsright") {
      state.thumb_rx = INT16_MAX;
    } else {
      return false;
    }
    return true;
  }

  void SetState(const State& state) {
    std::lock_guard<std::mutex> lock(mutex_);
    current_ = state;
    ++packet_number_;
  }

  void Run() {
    const std::filesystem::path path = cvars::input_script_file;
    while (true) {
      std::this_thread::sleep_for(std::chrono::milliseconds(30));
      std::error_code error_code;
      if (!std::filesystem::exists(path, error_code)) {
        continue;
      }
      std::stringstream commands;
      {
        std::ifstream file(path);
        commands << file.rdbuf();
      }
      std::filesystem::remove(path, error_code);

      std::string command;
      while (commands >> command) {
        uint32_t duration_ms = 200;
        size_t colon = command.find(':');
        if (colon != std::string::npos) {
          duration_ms = uint32_t(std::strtoul(command.c_str() + colon + 1,
                                              nullptr, 10));
          command.resize(colon);
        }
        State state;
        bool valid = true;
        if (command != "wait") {
          std::stringstream names(command);
          std::string name;
          while (std::getline(names, name, '+')) {
            valid &= ParseButton(name, state);
          }
        }
        if (!valid) {
          XELOGW("input_script_file: unknown command \"{}\"", command);
          continue;
        }
        SetState(state);
        std::this_thread::sleep_for(std::chrono::milliseconds(duration_ms));
        if (command != "wait") {
          // Release between commands so repeated presses register.
          SetState(State());
          std::this_thread::sleep_for(std::chrono::milliseconds(120));
        }
      }
    }
  }

  std::once_flag thread_started_;
  std::mutex mutex_;
  State current_;
  uint32_t packet_number_ = 0;
};

}  // namespace

InputSystem::InputSystem(xe::ui::Window* window) : window_(window) {
#ifdef XE_PLATFORM_WIN32
  portal_ = std::make_unique<HardwarePortal>();
#endif  // XE_PLATFORM_WIN32
}

InputSystem::~InputSystem() = default;

X_STATUS InputSystem::Setup() { return X_STATUS_SUCCESS; }

void InputSystem::AddDriver(std::unique_ptr<InputDriver> driver) {
  drivers_.push_back(std::move(driver));
}

void InputSystem::UpdateUsedSlot(InputDriver* driver, uint8_t slot,
                                 bool connected) {
  if (slot == XUserIndexAny) {
    XELOGW("{} received requrest for slot any! Unsupported", __func__);
    return;
  }

  // Do not report passthrough as a controller.
  if (driver && driver->GetInputType() == InputType::Keyboard) {
    return;
  }

  if (connected_slots.test(slot) == connected) {
    // No state change, so nothing to do.
    return;
  }

  XELOGI(controller_slot_state_change_message[connected].c_str(), slot);
  connected_slots.flip(slot);
  if (kernel::kernel_state()) {
    kernel::kernel_state()->BroadcastNotification(
        kXNotificationSystemInputDevicesChanged, 0);
  }

  if (driver) {
    X_INPUT_CAPABILITIES capabilities = {};
    const X_RESULT result = driver->GetCapabilities(slot, 0, &capabilities);
    if (result != X_STATUS_SUCCESS) {
      return;
    }

    controllers_max_joystick_value[slot] = {
        {capabilities.gamepad.thumb_lx, capabilities.gamepad.thumb_ly},
        {capabilities.gamepad.thumb_rx, capabilities.gamepad.thumb_ry}};
  }
}

std::vector<InputDriver*> InputSystem::FilterDrivers(uint32_t flags) {
  std::vector<InputDriver*> filtered_drivers;
  for (auto& driver : drivers_) {
    if (driver->GetInputType() == InputType::None) {
      continue;
    }

    if ((flags & driver->GetInputType()) != 0) {
      filtered_drivers.push_back(driver.get());
    }
  }
  return filtered_drivers;
}

X_RESULT InputSystem::GetCapabilities(uint32_t user_index, uint32_t flags,
                                      X_INPUT_CAPABILITIES* out_caps) {
  SCOPE_profile_cpu_f("hid");

  std::vector<InputDriver*> filtered_drivers = FilterDrivers(flags);

  for (auto& driver : filtered_drivers) {
    X_RESULT result = driver->GetCapabilities(user_index, flags, out_caps);
    if (result == X_ERROR_SUCCESS) {
      return result;
    }
  }
  return X_ERROR_DEVICE_NOT_CONNECTED;
}

X_RESULT InputSystem::GetState(uint32_t user_index, uint32_t flags,
                               X_INPUT_STATE* out_state) {
  SCOPE_profile_cpu_f("hid");

  std::vector<InputDriver*> filtered_drivers = FilterDrivers(flags);
  if (filtered_drivers.empty()) {
    return X_ERROR_DEVICE_NOT_CONNECTED;
  }

  for (auto& driver : filtered_drivers) {
    X_RESULT result = driver->GetState(user_index, out_state);
    if (result == X_ERROR_SUCCESS) {
      UpdateUsedSlot(driver, user_index, true);
      if (user_index == 0 && !cvars::input_script_file.empty()) {
        ScriptedInput::Get().Apply(out_state);
      }
      AdjustDeadzoneLevels(user_index, &out_state->gamepad);

      if (out_state->gamepad.buttons != 0) {
        last_used_slot = user_index;
      }
      return result;
    }
  }
  UpdateUsedSlot(nullptr, user_index, false);
  return X_ERROR_DEVICE_NOT_CONNECTED;
}

X_RESULT InputSystem::SetState(uint32_t user_index,
                               X_INPUT_VIBRATION* vibration) {
  SCOPE_profile_cpu_f("hid");
  X_INPUT_VIBRATION modified_vibration = ModifyVibrationLevel(vibration);
  for (auto& driver : drivers_) {
    X_RESULT result = driver->SetState(user_index, &modified_vibration);
    if (result == X_ERROR_SUCCESS) {
      return result;
    }
  }
  return X_ERROR_DEVICE_NOT_CONNECTED;
}

X_RESULT InputSystem::GetKeystroke(uint32_t user_index, uint32_t flags,
                                   X_INPUT_KEYSTROKE* out_keystroke) {
  SCOPE_profile_cpu_f("hid");

  std::vector<InputDriver*> filtered_drivers = FilterDrivers(flags);

  bool any_connected = false;
  for (auto& driver : filtered_drivers) {
    // connected_slots
    X_RESULT result = driver->GetKeystroke(user_index, flags, out_keystroke);
    if (result == X_ERROR_INVALID_PARAMETER ||
        result == X_ERROR_DEVICE_NOT_CONNECTED) {
      continue;
    }

    any_connected = true;

    if (result == X_ERROR_SUCCESS) {
      last_used_slot = user_index;
      return result;
    }

    if (result == X_ERROR_EMPTY) {
      continue;
    }
  }
  return any_connected ? X_ERROR_EMPTY : X_ERROR_DEVICE_NOT_CONNECTED;
}

bool InputSystem::GetVibrationCvar() { return cvars::vibration; }

void InputSystem::ToggleVibration() {
  OVERRIDE_bool(vibration, !cvars::vibration);
  // Send instant update to vibration state to prevent awaiting for next tick.
  X_INPUT_VIBRATION vibration = X_INPUT_VIBRATION();

  for (uint8_t user_index = 0; user_index < XUserMaxUserCount; user_index++) {
    SetState(user_index, &vibration);
  }
}

void InputSystem::AdjustDeadzoneLevels(const uint8_t slot,
                                       X_INPUT_GAMEPAD* gamepad) {
  if (slot > XUserMaxUserCount) {
    return;
  }

  // Left stick
  if (cvars::left_stick_deadzone_percentage > 0.0 &&
      cvars::left_stick_deadzone_percentage < 1.0) {
    const double deadzone_lx_percentage =
        controllers_max_joystick_value[slot].first.first *
        cvars::left_stick_deadzone_percentage;
    const double deadzone_ly_percentage =
        controllers_max_joystick_value[slot].first.second *
        cvars::left_stick_deadzone_percentage;

    const double theta = std::atan2(static_cast<double>(gamepad->thumb_ly),
                                    static_cast<double>(gamepad->thumb_lx));

    const double deadzone_y_value = std::sin(theta) * deadzone_ly_percentage;
    const double deadzone_x_value = std::cos(theta) * deadzone_lx_percentage;

    if (gamepad->thumb_ly > -deadzone_y_value &&
        gamepad->thumb_ly < deadzone_y_value) {
      gamepad->thumb_ly = 0;
    }

    if (gamepad->thumb_lx > -deadzone_x_value &&
        gamepad->thumb_lx < deadzone_x_value) {
      gamepad->thumb_lx = 0;
    }
  }

  // Right stick
  if (cvars::right_stick_deadzone_percentage > 0.0 &&
      cvars::right_stick_deadzone_percentage < 1.0) {
    const double deadzone_rx_percentage =
        controllers_max_joystick_value[slot].second.first *
        cvars::right_stick_deadzone_percentage;
    const double deadzone_ry_percentage =
        controllers_max_joystick_value[slot].second.second *
        cvars::right_stick_deadzone_percentage;

    const double theta = std::atan2(static_cast<double>(gamepad->thumb_ry),
                                    static_cast<double>(gamepad->thumb_rx));

    const double deadzone_y_value = std::sin(theta) * deadzone_ry_percentage;
    const double deadzone_x_value = std::cos(theta) * deadzone_rx_percentage;

    if (gamepad->thumb_ry > -deadzone_y_value &&
        gamepad->thumb_ry < deadzone_y_value) {
      gamepad->thumb_ry = 0;
    }

    if (gamepad->thumb_rx > -deadzone_x_value &&
        gamepad->thumb_rx < deadzone_x_value) {
      gamepad->thumb_rx = 0;
    }
  }
}

X_INPUT_VIBRATION InputSystem::ModifyVibrationLevel(
    X_INPUT_VIBRATION* vibration) {
  X_INPUT_VIBRATION modified_vibration = *vibration;
  if (cvars::vibration) {
    return modified_vibration;
  }

  // TODO(Gliniak): Use modifier instead of boolean value.
  modified_vibration.left_motor_speed = 0;
  modified_vibration.right_motor_speed = 0;
  return modified_vibration;
}
std::unique_lock<xe_unlikely_mutex> InputSystem::lock() {
  return std::unique_lock<xe_unlikely_mutex>{lock_};
}
}  // namespace hid
}  // namespace xe
