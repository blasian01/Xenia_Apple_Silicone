#!/bin/bash
# Disassembles a raw ARM64 code dump produced with XENIA_A64_DUMP_CODE.
LLVM_BIN="${LLVM_BIN:-/opt/homebrew/opt/llvm/bin}"
xxd -p -c 4 "$1" | sed -E 's/(..)(..)(..)(..)/0x\1 0x\2 0x\3 0x\4/' |
  "$LLVM_BIN/llvm-mc" --disassemble -triple=aarch64 -mattr=+neon,+fp-armv8 2>&1 | grep -v "^\s*\.text"
