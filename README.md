<p align="center">
    <a href="https://github.com/xenia-canary/xenia-canary/tree/canary_experimental/assets/icon">
        <img height="256px" src="https://raw.githubusercontent.com/xenia-canary/xenia/master/assets/icon/256.png" />
    </a>
</p>

<h1 align="center">Xenia — Native macOS (Apple Silicon) Build</h1>

## 👋 About This Fork

Hey, my name is **Bronson**. I grew up on the Xbox 360 — it was the console that defined my childhood. Halo 4, Castle Miner Z, Bo2, you name it. Those games mean a lot to me.

Fast forward to today and I'm in a position where I only have an Apple Mac. Xenia is the best Xbox 360 emulator out there, but it only officially supports Windows. Rather than accept that, I decided to try and build a **native macOS application** for Xenia on Apple Silicon.

This fork contains all the changes needed to compile and run Xenia natively on **macOS arm64 (Apple Silicon)** — no Rosetta, no Wine, no CrossOver. Just a native `.app` bundle you can drop into your Applications folder.

> **⚠️ Important Disclaimer:** This is an experimental port. With the help of my monkey brain and AI I am trying to get this fully working with a proper DMG built as well as a working x64 CPU JIT backend for ARM64. I am only working on this in my free time so things might not work :/

---

## 🛠️ Building from Source

### Prerequisites

Make sure you have these installed on your Mac:

- **Xcode Command Line Tools** (or full Xcode)
  ```bash
  xcode-select --install
  ```
- **CMake** (3.20 or later)
  ```bash
  brew install cmake
  ```
- **Python 3** (for code generation scripts)
  ```bash
  brew install python3
  ```
- **SDL2** (for input and audio)
  ```bash
  brew install sdl2
  ```
- **Vulkan SDK + MoltenVK** (required for GPU rendering — translates Vulkan → Metal)
  ```bash
  brew install molten-vk
  brew install --cask vulkan-sdk
  ```
  > MoltenVK is what makes Vulkan work on macOS by translating it to Apple's Metal API. **Without it, the app will launch but immediately fail with "Failed to load Vulkan Portability library".**

### Clone the Repo

```bash
git clone --recursive https://github.com/blasian01/Xenia_Apple_Silicone.git
cd Xenia_Apple_Silicone
```

> **Note:** The `--recursive` flag is important — Xenia has a lot of submodules (FFmpeg, SDL2, Capstone, etc.).

If you already cloned without `--recursive`, run:
```bash
git submodule update --init --recursive
```

### Build

```bash
# Configure for arm64
cmake -S . -B build-macos -DCMAKE_BUILD_TYPE=Release -DCMAKE_OSX_ARCHITECTURES=arm64

# Build (uses all available CPU cores)
cmake --build build-macos -j$(sysctl -n hw.ncpu)
```

The build takes a few minutes. When it's done, the app bundle is at:
```
build/bin/macOS/Xenia.app
```

### First Launch Fix

macOS will block the app because it's not signed by an Apple Developer account. You **must** run these commands before opening for the first time:

```bash
# Clear quarantine flag and ad-hoc sign the app
xattr -cr build/bin/macOS/Xenia.app
codesign --force --deep --sign - build/bin/macOS/Xenia.app
```

Then open it:
```bash
open build/bin/macOS/Xenia.app
```

If macOS still shows a "damaged or incomplete" warning, right-click the app → **Open** → click **Open** in the dialog.

### Package as DMG (Optional)

To create a distributable `.dmg` file:

```bash
hdiutil create -srcfolder build/bin/macOS/Xenia.app -volname "Xenia" -format UDZO build/Xenia.dmg
```

This creates `build/Xenia.dmg`. Open it and drag `Xenia.app` to your Applications folder. **Remember to run the `xattr -cr` and `codesign` commands on the copy in Applications too.**

### Verify the Build

```bash
# Confirm it's a native arm64 binary
file build/bin/macOS/Xenia.app/Contents/MacOS/xenia-app
# Should output: Mach-O 64-bit executable arm64
```

---

## 🎮 Getting Games Running

### What You Need

1. **Xbox 360 game disc images** — You need to dump your own games from discs you legally own. Xenia supports:
   - **ISO** files (raw disc images)
   - **XBLA / GOD** packages (Xbox Live Arcade / Games on Demand)
   - **XEX** files (raw executables extracted from game packages)

2. **Launch the app** — Open `Xenia.app` from your Applications folder (or run it directly from the build output).

3. **Open a game** — Use `File > Open` and select your game file (`.iso`, `.xex`, or extracted game folder).

### Game Compatibility

Check the [Xenia Canary Game Compatibility List](https://github.com/xenia-canary/game-compatibility/issues) to see which games are known to work. Keep in mind that this macOS port may have additional limitations beyond the base compatibility list due to the ARM64 JIT backend and MoltenVK GPU translation layer.

### Tips

- **MoltenVK is required**: The GPU rendering pipeline is Vulkan → MoltenVK → Metal. Without MoltenVK installed, the app window won't open.
- **First launch**: The first time you open a game it may take longer as shaders are compiled and cached.
- **Performance**: Native arm64 means no translation overhead for the emulator itself. The ARM64 JIT backend compiles Xbox 360 PPC instructions directly to native ARM64 code on the fly.
- **Logs**: If something goes wrong, check `~/Library/Application Support/Xenia/` for logs, or run the app from Terminal to see console output.
- **GPU acceleration**: M1/M2/M3/M4 GPU hardware is used automatically through the Metal API via MoltenVK.

---

## 🔧 What Was Changed

This fork makes the following changes to build and run on macOS arm64:

### ARM64 JIT Backend (New)
- **Complete ARM64 code generator** — 100+ HIR opcode handlers translating Xbox 360 PPC instructions to native ARM64 machine code
- **150+ ARM64 instruction encodings** including integer, FP, NEON vector, and atomic operations
- **Runtime infrastructure** — resolve-function thunk for on-demand JIT compilation, host↔guest thunks, guest trampolines
- **Branch label system** for intra-function control flow (B/CBNZ/CBZ to HIR labels)
- **Atomic operations** using ARM64 exclusive access (LDXR/STXR) with memory barriers

### Platform Adaptation
- **Platform stubs** for macOS (message boxes, file pickers, window management, system functions)
- **Architecture guards** to exclude x86-specific code (AVX intrinsics, x64 JIT backend, SSE defines) on arm64
- **ARM64 NEON compatibility** fixes for `constexpr` issues with AppleClang
- **FFmpeg aarch64 stubs** so the audio/video libraries link without needing NEON assembly compilation
- **CMake build system** updates for macOS app bundle packaging, framework linking, and Vulkan/MoltenVK integration
- **Duplicate symbol resolution** between `_posix` and `_mac` platform files
- **Linker fixes** for large PPC opcode tables with unaligned pointers

---

## 📋 Original Project

This is a fork of [Xenia Canary](https://github.com/xenia-canary/xenia-canary), the experimental branch of the Xenia Xbox 360 emulator. All credit for the incredible emulator goes to the Xenia team and contributors.

- [Xenia Canary Wiki](https://github.com/xenia-canary/xenia-canary/wiki)
- [Discord](https://discord.gg/Q9mxZf9)
- [FAQ](https://github.com/xenia-canary/xenia-canary/wiki/FAQ)

## Disclaimer

The goal of this project is to experiment, research, and educate on the topic of emulation of modern devices and operating systems. **It is not for enabling illegal activity**. All information is obtained via reverse engineering of legally purchased devices and games and information made public on the internet.
