<p align="center">
    <a href="https://github.com/xenia-canary/xenia-canary/tree/canary_experimental/assets/icon">
        <img height="256px" src="https://raw.githubusercontent.com/xenia-canary/xenia/master/assets/icon/256.png" />
    </a>
</p>

<h1 align="center">Xenia — Native macOS (Apple Silicon) Build</h1>

## 👋 About This Fork

Hey, my name is **Bronson**. I grew up on the Xbox 360 — it was the console that defined my childhood. Halo 3, Gears of War, Forza, you name it. Those games mean a lot to me.

Fast forward to today and I'm in a position where I only have an Apple Mac. Xenia is the best Xbox 360 emulator out there, but it only officially supports Windows. Rather than accept that, I decided to try and build a **native macOS application** for Xenia on Apple Silicon.

This fork contains all the changes needed to compile and run Xenia natively on **macOS arm64 (Apple Silicon)** — no Rosetta, no Wine, no CrossOver. Just a native `.app` bundle you can drop into your Applications folder.

> **⚠️ Important Disclaimer:** This is an experimental port. The x64 CPU JIT backend (which translates Xbox 360 PowerPC code to x86) is excluded on arm64 since it can't run on ARM hardware. The app will build and launch, but running actual Xbox 360 games will require a future ARM64 JIT backend. This is a work in progress!

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
- **Vulkan SDK** (for the graphics backend)
  - Download from [LunarG](https://vulkan.lunarg.com/sdk/home#mac) or install via:
  ```bash
  brew install --cask vulkan-sdk
  ```

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

### Package as DMG

To create a distributable `.dmg` file:

```bash
cmake --build build-macos --target xenia-dmg
```

This creates `build/Xenia.dmg`. Open it and drag `Xenia.app` to your Applications folder.

### Verify the Build

```bash
# Confirm it's a native arm64 binary
file build/bin/macOS/Xenia.app/Contents/MacOS/Xenia
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

Check the [Xenia Canary Game Compatibility List](https://github.com/xenia-canary/game-compatibility/issues) to see which games are known to work. Keep in mind that this macOS port may have additional limitations beyond the base compatibility list due to the Vulkan backend on MoltenVK.

### Tips

- **Vulkan drivers**: macOS uses [MoltenVK](https://github.com/KhronosGroup/MoltenVK) to translate Vulkan to Metal. Make sure the Vulkan SDK is installed.
- **First launch**: The first time you open a game it may take longer as shaders are compiled and cached.
- **Performance**: Native arm64 means no translation overhead for the emulator itself, but GPU-intensive games may still be demanding.
- **Logs**: If something goes wrong, check the log output in the terminal for error details.

---

## 🔧 What Was Changed

This fork makes the following changes to build on macOS arm64:

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
