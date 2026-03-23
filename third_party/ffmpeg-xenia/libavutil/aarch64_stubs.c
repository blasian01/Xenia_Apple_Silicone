/*
 * aarch64_stubs.c — stub implementations for aarch64 FFmpeg functions.
 * These provide the symbols that FFmpeg's generic C code calls
 * when aarch64 is detected, but without requiring NEON assembly.
 * The init functions are no-ops, so the generic C fallbacks are used.
 */

#include <stddef.h>
#include "libavutil/cpu.h"
#include "libavutil/float_dsp.h"
#include "libavutil/tx_priv.h"

int ff_get_cpu_flags_aarch64(void) {
    return 0;  /* No NEON optimizations reported */
}

void ff_float_dsp_init_aarch64(AVFloatDSPContext *fdsp) {
    /* No-op: use generic C implementations */
}

/* Empty codelet list */
const FFTXCodelet * const ff_tx_codelet_list_float_aarch64[] = {
    NULL,
};

size_t ff_get_cpu_max_align_aarch64(void) {
    return 16;  /* NEON is always available on Apple Silicon */
}
