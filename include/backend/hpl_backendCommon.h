#pragma once

#include <stdio.h>
#include <stdlib.h>
#include "backend/hpl_backendWrapper.h"
#include "backend/hpl_backendCPU.h"
#include "backend/hpl_backendHIP.h"

#ifdef __cplusplus

namespace HPL
{
  template<typename Func, typename... Args> 
  constexpr auto dispatch(Func&& func, Args... args) {
    return func(args...);
  }
};

#endif


void HPL_piplen(HPL_T_panel*, const int, const int*, int*, int*);
void HPL_plindx(HPL_T_panel*, const int, const int*, int*, int*, int*, int*, int*, int*, int*);
int  HPL_scatterv(double*, const int*, const int*, const int, int, MPI_Comm);
int  HPL_allgatherv(double*, const int, const int*, const int*, MPI_Comm);
int  HPL_pdpanel_bcast(HPL_T_panel*);
