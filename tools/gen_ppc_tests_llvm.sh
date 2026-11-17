#!/bin/bash
# Generates the PPC instruction test binaries (src/xenia/cpu/ppc/testing/bin)
# with LLVM tools instead of the custom binutils build that `xb gentests`
# needs. Xbox 360-specific opcodes (VMX128 etc.) that llvm-mc doesn't know are
# encoded by tools/ppc_vmx128_asm.py. Anything that still fails to assemble
# is listed in bin/_unassembled.txt.
#
# Usage: tools/gen_ppc_tests_llvm.sh   (from the repository root)
set -u
set -o pipefail
LLVM_BIN="${LLVM_BIN:-/opt/homebrew/opt/llvm/bin}"
SRC=src/xenia/cpu/ppc/testing
OUT="$SRC/bin"
mkdir -p "$OUT"
ok=0
failed=()
for s in "$SRC"/*.s; do
  name=$(basename "$s" .s)
  # llvm-mc wants bare register numbers on ELF targets; strip the r/f/v/cr
  # prefixes on code lines (annotation comments are read from the original).
  if sed -E '/m[ft]spr/{s/([ ,])xer([ ,]|$)/\11\2/;s/([ ,])lr([ ,]|$)/\18\2/;s/([ ,])ctr([ ,]|$)/\19\2/;}' "$s" | python3 tools/ppc_vmx128_asm.py tools/ppc-instructions.xml | sed -E '/^[[:space:]]*#/!s/(^|[^A-Za-z0-9_.])(cr|r|f|v)([0-9]+)([^A-Za-z0-9_:]|$)/\1\3\4/g;/^[[:space:]]*#/!s/(^|[^A-Za-z0-9_.])(cr|r|f|v)([0-9]+)([^A-Za-z0-9_:]|$)/\1\3\4/g' |
     "$LLVM_BIN/llvm-mc" -triple=powerpc64-unknown-elf -mcpu=a2 \
       -mattr=+altivec,+vsx -filetype=obj -o "$OUT/$name.o" 2>"$OUT/$name.err" &&
     "$LLVM_BIN/llvm-objcopy" -O binary -j .text "$OUT/$name.o" "$OUT/$name.bin" &&
     "$LLVM_BIN/llvm-nm" --numeric-sort "$OUT/$name.o" > "$OUT/$name.map"; then
    ok=$((ok + 1))
    rm -f "$OUT/$name.err"
  else
    failed+=("$name")
    rm -f "$OUT/$name.bin" "$OUT/$name.map"
  fi
  rm -f "$OUT/$name.o"
done
echo "assembled: $ok"
printf '%s\n' ${failed[@]+"${failed[@]}"} > "$OUT/_unassembled.txt"
