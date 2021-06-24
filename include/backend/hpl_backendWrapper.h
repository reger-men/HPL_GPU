#pragma once

#include <stdint.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

enum HPL_target {Default, CPUx, HIP};
void HPL_malloc(void** ptr, size_t size, enum HPL_target tr);

#ifdef __cplusplus
}
#endif