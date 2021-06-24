#pragma once

#include <stdio.h>
#include <stdlib.h>
#include "backend/hpl_backendCPU.h"

/*namespace HPL {
    int  init();
    void malloc(void** ptr, size_t size){printf("malloc");};
    void memcpy();
    void memset();

    template<typename... Ts, typename... Fns>
    constexpr auto dispatch(Fns&&... fns);
}*/


#ifdef __cplusplus

namespace HPL
{
  template<typename Func, typename... Args> 
  constexpr auto dispatch(Func&& func, Args... args) {
    func(args...);
  }
};

#endif

