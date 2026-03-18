/**
 ******************************************************************************
 * Xenia : Xbox 360 Emulator Research Project                                 *
 ******************************************************************************
 * Copyright 2025 Ben Vanik. All rights reserved.                             *
 * Released under the BSD license - see LICENSE in the root for more details. *
 ******************************************************************************
 */

#ifndef XENIA_UI_SURFACE_MAC_H_
#define XENIA_UI_SURFACE_MAC_H_

#include "xenia/ui/surface.h"

namespace xe {
namespace ui {

class MacSurface : public Surface {
 public:
  // metal_layer is a CAMetalLayer* (passed as void* to avoid ObjC in header)
  MacSurface(void* metal_layer, uint32_t width, uint32_t height)
      : metal_layer_(metal_layer), width_(width), height_(height) {}

  TypeIndex GetType() const override { return kTypeIndex_MacMetalLayer; }

  void* metal_layer() const { return metal_layer_; }

  void SetSize(uint32_t width, uint32_t height) {
    width_ = width;
    height_ = height;
  }

 protected:
  bool GetSizeImpl(uint32_t& width_out, uint32_t& height_out) const override {
    width_out = width_;
    height_out = height_;
    return width_ > 0 && height_ > 0;
  }

 private:
  void* metal_layer_;
  uint32_t width_ = 0;
  uint32_t height_ = 0;
};

}  // namespace ui
}  // namespace xe

#endif  // XENIA_UI_SURFACE_MAC_H_
