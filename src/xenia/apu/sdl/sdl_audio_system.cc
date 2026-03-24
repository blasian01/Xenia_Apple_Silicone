/**
 ******************************************************************************
 * Xenia : Xbox 360 Emulator Research Project                                 *
 ******************************************************************************
 * Copyright 2020 Ben Vanik. All rights reserved.                             *
 * Released under the BSD license - see LICENSE in the root for more details. *
 ******************************************************************************
 */

#include "xenia/apu/sdl/sdl_audio_system.h"

#include "xenia/apu/apu_flags.h"
#include "xenia/apu/sdl/sdl_audio_driver.h"
#include "xenia/base/logging.h"

namespace xe {
namespace apu {
namespace sdl {

namespace {

class SilentAudioDriver final : public AudioDriver {
 public:
  explicit SilentAudioDriver(xe::threading::Semaphore* semaphore)
      : semaphore_(semaphore) {}

  bool Initialize() override { return true; }
  void Shutdown() override {}

  void SubmitFrame(float* samples) override {
    auto ret = semaphore_->Release(1, nullptr);
    assert_true(ret);
  }

  void Pause() override {}
  void Resume() override {}
  void SetVolume(float volume) override {}

 private:
  xe::threading::Semaphore* semaphore_;
};

}  // namespace

std::unique_ptr<AudioSystem> SDLAudioSystem::Create(cpu::Processor* processor) {
  return std::make_unique<SDLAudioSystem>(processor);
}

SDLAudioSystem::SDLAudioSystem(cpu::Processor* processor)
    : AudioSystem(processor) {}

SDLAudioSystem::~SDLAudioSystem() {}

void SDLAudioSystem::Initialize() { AudioSystem::Initialize(); }

X_STATUS SDLAudioSystem::CreateDriver(size_t index,
                                      xe::threading::Semaphore* semaphore,
                                      AudioDriver** out_driver) {
  assert_not_null(out_driver);
  auto driver = std::make_unique<SDLAudioDriver>(semaphore);
  if (!driver->Initialize()) {
    driver->Shutdown();
    XELOGW(
        "SDLAudioSystem::CreateDriver: falling back to silent driver for "
        "client index {}",
        index);
    auto silent_driver = std::make_unique<SilentAudioDriver>(semaphore);
    if (!silent_driver->Initialize()) {
      return X_STATUS_UNSUCCESSFUL;
    }
    *out_driver = silent_driver.release();
    return X_STATUS_SUCCESS;
  }

  *out_driver = driver.release();
  return X_STATUS_SUCCESS;
}

AudioDriver* SDLAudioSystem::CreateDriver(xe::threading::Semaphore* semaphore,
                                          uint32_t frequency, uint32_t channels,
                                          bool need_format_conversion) {
  return new SDLAudioDriver(semaphore, frequency, channels,
                            need_format_conversion);
}

void SDLAudioSystem::DestroyDriver(AudioDriver* driver) {
  assert_not_null(driver);
  if (auto sdldriver = dynamic_cast<SDLAudioDriver*>(driver)) {
    sdldriver->Shutdown();
    delete sdldriver;
    return;
  }

  driver->Shutdown();
  delete driver;
}

}  // namespace sdl
}  // namespace apu
}  // namespace xe
