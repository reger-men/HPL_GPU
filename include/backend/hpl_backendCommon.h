#pragma once

#include <stdio.h>
#include <stdlib.h>
#include "backend/hpl_backendCPU.h"
#include "backend/hpl_backendHIP.h"

#ifdef __cplusplus

namespace HPL
{
  template<typename Func, typename... Args> 
  constexpr auto dispatch(Func&& func, Args... args) {
    func(args...);
  }
};

#endif

