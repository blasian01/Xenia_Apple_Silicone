/*
 * aarch64_stubs.c — stub for aarch64 FFmpeg libavcodec functions.
 */

#include "libavcodec/mpegaudiodsp.h"

void ff_mpadsp_init_aarch64(MPADSPContext *s) {
    /* No-op: use generic C implementations */
}
