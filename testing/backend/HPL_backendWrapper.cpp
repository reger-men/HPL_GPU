
#include "backend/hpl_backendCommon.h"
#include "backend/hpl_backendWrapper.h"
#include <iostream>

extern "C" {
   
   void HPL_malloc(void** ptr, size_t size, HPL_target tr)
   {
      switch(tr) {
         case CPUx :
            HPL::dispatch(CPU::malloc, ptr, size);
            break;
         case HIP:
            printf("NOT IMPLEMENTED!!\n");
            break;
         default:
            HPL::dispatch(CPU::malloc, ptr, size);
      }
   }
}