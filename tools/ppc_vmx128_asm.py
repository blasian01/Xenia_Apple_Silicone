#!/usr/bin/env python3
"""Rewrites Xbox 360-specific PPC instructions as raw .long directives.

llvm-mc does not know the VMX128 extension (nor lvlx/lvrx/stvlx/stvrx or
dcbz128), so tools/gen_ppc_tests_llvm.sh pipes each test source through this
filter first. Encodings come from tools/ppc-instructions.xml; field layouts
from src/xenia/cpu/ppc/ppc_instr.h.

Usage: ppc_vmx128_asm.py <ppc-instructions.xml> < in.s > out.s
"""
import re
import sys
import xml.etree.ElementTree as ET

EXTRA_X_FORM = {
    'lvlx', 'lvlxl', 'lvrx', 'lvrxl', 'stvlx', 'stvlxl', 'stvrx', 'stvrxl',
}


def load_table(path):
    table = {}
    for insn in ET.parse(path).getroot().iter('insn'):
        mnem = insn.get('mnem')
        if '128' in mnem or mnem in EXTRA_X_FORM:
            table[mnem] = (int(insn.get('opcode'), 16), insn.get('form'))
    return table


def num(tok):
    tok = tok.strip()
    m = re.fullmatch(r'(?:cr|r|f|v)(\d+)', tok)
    if m:
        return int(m.group(1))
    return int(tok, 0)


def vd128(v):
    return ((v & 0x1F) << 21) | (((v >> 5) & 0x3) << 2)


def va128(v):
    return ((v & 0x1F) << 16) | (((v >> 5) & 0x1) << 5) | (((v >> 6) & 0x1) << 10)


def vb128(v):
    return ((v & 0x1F) << 11) | ((v >> 5) & 0x3)


def encode(mnem, rc, ops, table):
    opcode, form = table[mnem]
    o = [num(x) for x in ops]
    if form == 'X':
        return opcode | (o[0] << 21) | (o[1] << 16) | (o[2] << 11)
    if form == 'DCBZ':
        return opcode | (o[0] << 16) | (o[1] << 11)
    if form == 'VX128_1':
        return opcode | vd128(o[0]) | (o[1] << 16) | (o[2] << 11)
    if form in ('VX128', 'VX128_2', 'VX128_R'):
        if mnem == 'vmaddcfp128':  # vD, vA, vD, vB
            word = opcode | vd128(o[0]) | va128(o[1]) | vb128(o[3])
        elif len(o) == 2:  # vD, vB
            word = opcode | vd128(o[0]) | vb128(o[1])
        else:
            word = opcode | vd128(o[0]) | va128(o[1]) | vb128(o[2])
        if mnem == 'vperm128':
            word |= (o[3] & 0x7) << 6
        if form == 'VX128_R' and rc:
            word |= 1 << 6
        return word
    if form == 'VX128_3':
        if mnem == 'vspltisw128':  # vD, SIMM
            return opcode | vd128(o[0]) | ((o[1] & 0x1F) << 16)
        word = opcode | vd128(o[0]) | vb128(o[1])
        if len(o) > 2:
            word |= (o[2] & 0x1F) << 16
        return word
    if form == 'VX128_4':
        if mnem == 'vpkd3d128':  # vD, vB, type, pack, shift
            imm = ((o[2] & 0x7) << 2) | (o[3] & 0x3)
            z = o[4]
        else:  # vrlimi128 vD, vB, IMM, z
            imm, z = o[2], o[3]
        return (opcode | vd128(o[0]) | vb128(o[1]) | ((imm & 0x1F) << 16) |
                ((z & 0x3) << 6))
    if form == 'VX128_5':
        return (opcode | vd128(o[0]) | va128(o[1]) | vb128(o[2]) |
                ((o[3] & 0xF) << 6))
    if form == 'VX128_P':
        return (opcode | vd128(o[0]) | vb128(o[1]) | ((o[2] & 0x1F) << 16) |
                (((o[2] >> 5) & 0x7) << 6))
    raise ValueError('unhandled form %s for %s' % (form, mnem))


def main():
    table = load_table(sys.argv[1])
    line_re = re.compile(r'^(\s+)([a-z0-9]+)(\.?)\s+([^#]*?)\s*$')
    for line in sys.stdin:
        m = line_re.match(line)
        if m and m.group(2) == 'mcrxr':
            # Removed from the ISA in v3.0, so llvm-mc rejects it.
            word = 0x7C000400 | (num(m.group(4)) << 23)
            sys.stdout.write('%s.long 0x%08X  # %s\n' %
                             (m.group(1), word, line.strip()))
        elif m and m.group(2) == 'li':
            # binutils accepts 32-bit two's complement immediates here.
            reg, imm = [x.strip() for x in m.group(4).split(',')]
            value = int(imm, 0)
            if value > 0x7FFF and (value & 0xFFFF8000) == 0xFFFF8000:
                value -= 1 << 32
            sys.stdout.write('%sli %s, %d\n' % (m.group(1), reg, value))
        elif m and m.group(2) in table:
            ops = [x for x in m.group(4).split(',') if x.strip()]
            word = encode(m.group(2), bool(m.group(3)), ops, table)
            sys.stdout.write('%s.long 0x%08X  # %s\n' %
                             (m.group(1), word, line.strip()))
        else:
            sys.stdout.write(line)


if __name__ == '__main__':
    main()
