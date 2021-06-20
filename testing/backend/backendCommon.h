/*#pragma once

// Choose the backend
#if defined(TARGET_DEVICE_CPU)
#include "backend_cpu.h"
#elif defined(TARGET_DEVICE_HIP)
#include "backend_hip.h"
#elif defined(TARGET_DEVICE_CUDA)
#include "backend_cuda.h"
#endif



namespace HPL {
    int  init();
    void malloc();
    void memcpy();
    void memset();

    template<typename... Ts, typename... Fns>
    __device__ constexpr auto dispatch(Fns&&... fns);
}*/