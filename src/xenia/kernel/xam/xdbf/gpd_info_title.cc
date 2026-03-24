/**
 ******************************************************************************
 * Xenia : Xbox 360 Emulator Research Project                                 *
 ******************************************************************************
 * Copyright 2024 Xenia Canary. All rights reserved.                          *
 * Released under the BSD license - see LICENSE in the root for more details. *
 ******************************************************************************
 */

#include "xenia/kernel/xam/xdbf/gpd_info_title.h"

#include <cstring>
#include <ranges>

namespace xe {
namespace kernel {
namespace xam {

namespace {

constexpr size_t kAchievementHeaderSize = sizeof(X_XDBF_GPD_ACHIEVEMENT);

bool GetAchievementStringOffset(const Entry* entry, size_t string_index,
                                size_t* out_offset) {
  if (!entry || entry->data.size() < kAchievementHeaderSize) {
    return false;
  }

  size_t offset = kAchievementHeaderSize;
  for (size_t string_i = 0; string_i < string_index; ++string_i) {
    if (offset >= entry->data.size()) {
      return false;
    }

    const size_t remaining_bytes = entry->data.size() - offset;
    const size_t max_chars = remaining_bytes / sizeof(char16_t);
    bool found_terminator = false;
    for (size_t char_i = 0; char_i < max_chars; ++char_i) {
      uint16_t raw_char = 0;
      std::memcpy(&raw_char, entry->data.data() + offset +
                                 char_i * sizeof(char16_t),
                  sizeof(raw_char));
      if (xe::byte_swap(raw_char) == 0) {
        offset += (char_i + 1) * sizeof(char16_t);
        found_terminator = true;
        break;
      }
    }
    if (!found_terminator) {
      return false;
    }
  }

  if (offset > entry->data.size()) {
    return false;
  }

  *out_offset = offset;
  return true;
}

std::u16string ReadAchievementString(const Entry* entry, size_t offset) {
  if (!entry || offset >= entry->data.size()) {
    return {};
  }

  const size_t remaining_bytes = entry->data.size() - offset;
  const size_t max_chars = remaining_bytes / sizeof(char16_t);
  std::u16string result;
  result.reserve(max_chars);

  for (size_t char_i = 0; char_i < max_chars; ++char_i) {
    uint16_t raw_char = 0;
    std::memcpy(&raw_char, entry->data.data() + offset +
                               char_i * sizeof(char16_t),
                sizeof(raw_char));
    char16_t swapped_char = static_cast<char16_t>(xe::byte_swap(raw_char));
    if (swapped_char == 0) {
      return result;
    }
    result.push_back(swapped_char);
  }

  return {};
}

}  // namespace

X_XDBF_GPD_ACHIEVEMENT* GpdInfoTitle::GetAchievementEntry(const uint32_t id) {
  Entry* entry = GetEntry(static_cast<uint16_t>(GpdSection::kAchievement), id);

  if (!entry || entry->data.size() < kAchievementHeaderSize) {
    return nullptr;
  }

  return reinterpret_cast<X_XDBF_GPD_ACHIEVEMENT*>(entry->data.data());
}

const char16_t* GpdInfoTitle::GetAchievementTitlePtr(const uint32_t id) {
  const Entry* entry =
      GetEntry(static_cast<uint16_t>(GpdSection::kAchievement), id);
  size_t offset = 0;
  if (!GetAchievementStringOffset(entry, 0, &offset)) {
    return nullptr;
  }
  return reinterpret_cast<const char16_t*>(entry->data.data() + offset);
}

const char16_t* GpdInfoTitle::GetAchievementDescriptionPtr(const uint32_t id) {
  const Entry* entry =
      GetEntry(static_cast<uint16_t>(GpdSection::kAchievement), id);
  size_t offset = 0;
  if (!GetAchievementStringOffset(entry, 1, &offset)) {
    return nullptr;
  }
  return reinterpret_cast<const char16_t*>(entry->data.data() + offset);
}

const char16_t* GpdInfoTitle::GetAchievementUnachievedDescriptionPtr(
    const uint32_t id) {
  const Entry* entry =
      GetEntry(static_cast<uint16_t>(GpdSection::kAchievement), id);
  size_t offset = 0;
  if (!GetAchievementStringOffset(entry, 2, &offset)) {
    return nullptr;
  }
  return reinterpret_cast<const char16_t*>(entry->data.data() + offset);
}

std::u16string GpdInfoTitle::GetAchievementTitle(const uint32_t id) {
  const Entry* entry =
      GetEntry(static_cast<uint16_t>(GpdSection::kAchievement), id);
  size_t offset = 0;
  if (!GetAchievementStringOffset(entry, 0, &offset)) {
    return {};
  }
  return ReadAchievementString(entry, offset);
}

std::u16string GpdInfoTitle::GetAchievementDescription(const uint32_t id) {
  const Entry* entry =
      GetEntry(static_cast<uint16_t>(GpdSection::kAchievement), id);
  size_t offset = 0;
  if (!GetAchievementStringOffset(entry, 1, &offset)) {
    return {};
  }
  return ReadAchievementString(entry, offset);
}

std::u16string GpdInfoTitle::GetAchievementUnachievedDescription(
    const uint32_t id) {
  const Entry* entry =
      GetEntry(static_cast<uint16_t>(GpdSection::kAchievement), id);
  size_t offset = 0;
  if (!GetAchievementStringOffset(entry, 2, &offset)) {
    return {};
  }
  return ReadAchievementString(entry, offset);
}

std::vector<uint32_t> GpdInfoTitle::GetAchievementsIds() const {
  std::vector<uint32_t> ids;

  auto achievements =
      entries_ | std::views::filter([](const auto& entry) {
        return !IsSyncEntry(&entry);
      }) |
      std::views::filter([](const auto& entry) {
        return IsEntryOfSection(&entry, GpdSection::kAchievement);
      });

  for (const auto& achievement : achievements) {
    ids.push_back(static_cast<uint32_t>(achievement.info.id));
  }
  return ids;
}

void GpdInfoTitle::AddAchievement(const AchievementDetails* header) {
  Entry* entry =
      GetEntry(static_cast<uint16_t>(GpdSection::kAchievement), header->id);

  if (entry) {
    return;
  }

  X_XDBF_GPD_ACHIEVEMENT internal_info;
  internal_info.magic = sizeof(X_XDBF_GPD_ACHIEVEMENT);
  internal_info.id = header->id;
  internal_info.image_id = header->image_id;
  internal_info.gamerscore = header->gamerscore;
  internal_info.flags = header->flags;
  internal_info.unlock_time = 0;

  const uint32_t strings_size =
      static_cast<uint32_t>(string_util::size_in_bytes(header->label) +
                            string_util::size_in_bytes(header->description) +
                            string_util::size_in_bytes(header->unachieved));

  const uint32_t entry_size = sizeof(X_XDBF_GPD_ACHIEVEMENT) + strings_size;

  Entry new_entry(header->id, static_cast<uint16_t>(GpdSection::kAchievement),
                  entry_size);

  uint8_t* write_ptr = new_entry.data.data();
  memcpy(write_ptr, &internal_info, sizeof(X_XDBF_GPD_ACHIEVEMENT));

  write_ptr += sizeof(X_XDBF_GPD_ACHIEVEMENT);

  string_util::copy_and_swap_truncating(reinterpret_cast<char16_t*>(write_ptr),
                                        header->label,
                                        header->label.length() + 1);

  write_ptr += string_util::size_in_bytes(header->label);

  string_util::copy_and_swap_truncating(reinterpret_cast<char16_t*>(write_ptr),
                                        header->description,
                                        header->description.length() + 1);

  write_ptr += string_util::size_in_bytes(header->description);

  string_util::copy_and_swap_truncating(reinterpret_cast<char16_t*>(write_ptr),
                                        header->unachieved,
                                        header->unachieved.length() + 1);

  UpsertEntry(&new_entry);
}

uint32_t GpdInfoTitle::GetTotalGamerscore() {
  const auto ids = GetAchievementsIds();

  uint32_t gamerscore = 0;
  for (const auto id : ids) {
    gamerscore += GetAchievementEntry(id)->gamerscore;
  }

  return gamerscore;
}
uint32_t GpdInfoTitle::GetGamerscore() {
  const auto ids = GetAchievementsIds();
  uint32_t gamerscore = 0;
  for (const auto id : ids) {
    const auto entry = GetAchievementEntry(id);
    if (entry->is_achievement_unlocked()) {
      gamerscore += GetAchievementEntry(id)->gamerscore;
    }
  }
  return gamerscore;
}

uint32_t GpdInfoTitle::GetAchievementCount() {
  return static_cast<uint32_t>(GetAchievementsIds().size());
}

uint32_t GpdInfoTitle::GetUnlockedAchievementCount() {
  const auto ids = GetAchievementsIds();
  uint32_t count = 0;
  for (const auto id : ids) {
    const auto entry = GetAchievementEntry(id);
    if (entry->is_achievement_unlocked()) {
      count += 1;
    }
  }
  return count;
}

}  // namespace xam
}  // namespace kernel
}  // namespace xe
