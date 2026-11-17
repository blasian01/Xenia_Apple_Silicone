/**
 ******************************************************************************
 * Xenia : Xbox 360 Emulator Research Project                                 *
 ******************************************************************************
 * Copyright 2021 Ben Vanik. All rights reserved.                             *
 * Released under the BSD license - see LICENSE in the root for more details. *
 ******************************************************************************
 */

#include <sys/time.h>

#include "xenia/base/assert.h"
#include "xenia/base/clock.h"

namespace xe {

uint64_t Clock::host_tick_frequency_platform() {
  // host_tick_count_platform() always counts nanoseconds, so the frequency is
  // 1 GHz regardless of the clock's resolution. Deriving it from clock_getres()
  // only works where the resolution is 1 ns: Apple Silicon reports 42 ns
  // (the Mach timebase), which made the guest clock run 42 times too fast.
  return 1000000000ull;
}

uint64_t Clock::host_tick_count_platform() {
  timespec tp;
  int error = clock_gettime(CLOCK_MONOTONIC_RAW, &tp);
  assert_zero(error);

  return tp.tv_nsec + tp.tv_sec * 1000000000ull;
}

uint64_t Clock::QueryHostSystemTime() {
  // https://docs.microsoft.com/en-us/windows/win32/sysinfo/converting-a-time-t-value-to-a-file-time
  constexpr uint64_t seconds_per_day = 3600 * 24;
  // Don't forget the 89 leap days.
  constexpr uint64_t seconds_1601_to_1970 =
      ((369 * 365 + 89) * seconds_per_day);

  timeval now;
  int error = gettimeofday(&now, nullptr);
  assert_zero(error);

  // NT systems use 100ns intervals.
  return static_cast<uint64_t>(
      (static_cast<int64_t>(now.tv_sec) + seconds_1601_to_1970) * 10000000ull +
      now.tv_usec * 10);
}

uint64_t Clock::QueryHostUptimeMillis() {
  return host_tick_count_platform() * 1000 / host_tick_frequency_platform();
}

uint64_t Clock::QueryHostInterruptTime() {
  // Interrupt time is in 100 ns units, like on Windows.
  return host_tick_count_platform() / 100;
}
}  // namespace xe
