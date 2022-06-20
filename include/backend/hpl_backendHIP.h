#pragma once

#include <hip/hip_runtime.h>
#if defined(HPLHIP_USE_ROCRAND)
#include <rocrand.h>
#endif
#include <rocblas.h>

#include <cstdio>
#include <cstdlib>
#include <map>
#include <string>
#include <list>

extern "C" {
#include "hpl_pmatgen.h"
#include "hpl_panel.h"
#include "hpl_pgesv.h"
}


#define HIP_CHECK_ERROR(error)                        \
    if(error != hipSuccess)                           \
    {                                                 \
        fprintf(stderr,                               \
                "hip error: '%s'(%d) at %s:%d\n",     \
                hipGetErrorString(error),             \
                error,                                \
                __FILE__,                             \
                __LINE__);                            \
        exit(EXIT_FAILURE);                           \
    }

#define ROCRAND_CHECK_STATUS(status)                  \
  {                                                   \
    rocrand_status _status = status;                  \
    if(_status != ROCRAND_STATUS_SUCCESS) {           \
        fprintf(stderr,                               \
                "rocRAND error: (%d) at %s:%d\n",     \
                _status,                              \
                __FILE__,                             \
                __LINE__);                            \
        exit(EXIT_FAILURE);                           \
    }                                                 \
  }

#define ROCBLAS_CHECK_STATUS(status)                  \
    if(status != rocblas_status_success)              \
    {                                                 \
        fprintf(stderr,                               \
                "rocBLAS error: '%s'(%d) at %s:%d\n", \
                rocblas_status_to_string(status),     \
                status,                               \
                __FILE__,                             \
                __LINE__);                            \
        exit(EXIT_FAILURE);                           \
    }


#define GPUError(...) GPUInfo(__VA_ARGS__)

// #define HPL_PRINT_INFO
#ifdef HPL_PRINT_INFO
#define GPUInfo(msg, ...)                             \
{                                                     \
  printf("INFO\t %-15s\t" msg "\n", ##__VA_ARGS__);   \
}
#else
#define GPUInfo(msg, ...)
#endif


enum PDLASWP_OP{
  SU0, SU1, SU2,  // start update for look-ahead, update1, update2
  CU0, CU1, CU2,  // communication for look-ahead, update1, update2
  EU0, EU1, EU2   // end update for look-ahead, update1, update2
};

enum SWP_PHASE {
  SWP_START, SWP_COMM, SWP_END, SWP_NO
};

namespace HIP {
    void init(size_t);
    void release();

    void malloc(void**, size_t);
    void host_malloc(void**, size_t, unsigned int);
    void free(void **);
    void host_free(void **);
    void panel_new(HPL_T_grid *, HPL_T_palg *, const int, const int, const int, HPL_T_pmat *,
                   const int, const int, const int, HPL_T_panel **);
    void panel_init(HPL_T_grid *, HPL_T_palg *, const int,
                    const int, const int, HPL_T_pmat *,
                    const int, const int, const int, HPL_T_panel *);
    void panel_send_to_host(HPL_T_panel *);
    void panel_send_to_device(HPL_T_panel *);
    int panel_free(HPL_T_panel *);
    int panel_disp(HPL_T_panel**);
    void matgen(const HPL_T_grid *, const int, const int, const int, double *, const int, const int);
    int pdmatgen(HPL_T_test* TEST, HPL_T_grid* GRID, HPL_T_palg* ALGO,  HPL_T_pmat* mat,const int N, const int NB);
    void pdmatfree(HPL_T_pmat* mat);
    void event_record(enum HPL_EVENT, const HPL_T_UPD);
    void event_synchronize(enum HPL_EVENT, const HPL_T_UPD);
    void stream_synchronize(enum HPL_STREAM);
    void stream_wait_event(enum HPL_STREAM, enum HPL_EVENT);
    float elapsedTime(const HPL_T_UPD);
    void device_sync();
    int bcast_ibcst(HPL_T_panel*, int*);
/*
*  ----------------------------------------------------------------------
*  - BLAS ---------------------------------------------------------------
*  ----------------------------------------------------------------------
*/ 

    int  idamax(const int, const double *, const int);
    void daxpy(const int, const double, const double *, const int, double *, 
                    const int);
    void dscal(const int, const double, double *, const int);
    void dswap(const int, double *, const int, double *, const int);

    void dger( const enum HPL_ORDER, const int, const int, const double, const double *,
                  const int, double *, const int, double *, const int);




    void trsm( const enum HPL_ORDER, const enum HPL_SIDE, 
                const enum HPL_UPLO, const enum HPL_TRANS, 
                const enum HPL_DIAG, const int, const int, 
                const double, const double *, const int, double *, const int);
    void trsv(const enum HPL_ORDER, const enum HPL_UPLO,
                const enum HPL_TRANS, const enum HPL_DIAG,
                const int, const double *, const int,
                double *, const int);                  
    void dgemm(const enum HPL_ORDER, const enum HPL_TRANS, 
                const enum HPL_TRANS, const int, const int, const int, 
                const double, const double *, const int, 
                const double *, const int, const double, double *, 
                const int);
    void dgemv(const enum HPL_ORDER, const enum HPL_TRANS, const int, const int,
                const double, const double *, const int, const double *, const int,
                const double, double *, const int);
    void copy(const int, const double *, const int, double *, const int);                
    void acpy(const int, const int, const double *, const int, double *, const int);                
    void atcpy(const int, const int, const double *, const int,
                double *, const int);  

    void move_data(double *, const double *, const size_t, const int);
    void move_data_2d(void*, size_t, const void*, size_t, size_t, size_t, const int);


    void gPrintMat(const int, const int, const int, const double*);
    double pdlange(const HPL_T_grid*, const HPL_T_NORM, const int, const int, const int, const double*, const int);
    void HPL_dlaswp00N(const int, const int, double*, const int, const int*);
    void HPL_dlaswp01T(const int, const int, double*, const int,double*, const int, const int*);
    void HPL_dlaswp02T(const int, const int, double*, const int, const int*, const int*);
    void HPL_dlaswp03T(const int, const int, double*, const int, double*,
                        const int, const int*);
    void HPL_dlaswp04T(const int, const int, double*, const int, double*,const int, const int*);
    void HPL_dlaswp10N(const int, const int, double*, const int, const int*); 
    void HPL_set_zero(const int N, double* __restrict__ X);

    void pdlaswp_set_var(HPL_T_panel* PANEL, double* &dU, double* &U, int &ldu, double* &dW, double* &W, int &ldw, int &n, double* &dA, const HPL_T_UPD UPD); 
    void HPL_pdlaswp_hip(HPL_T_panel* PANEL, const HPL_T_UPD UPD, const SWP_PHASE phase);
    void HPL_pdlaswp_hip(HPL_T_panel* PANEL, int icurcol, std::list<PDLASWP_OP> op_vec);
    // BLAS members
    namespace {
      rocblas_handle _handle;
      static char host_name[MPI_MAX_PROCESSOR_NAME];
      hipStream_t computeStream, dataStream, pdlaswpStream;
      hipEvent_t panelUpdate;
      hipEvent_t panelCopy, swapDataTransfer;
      hipEvent_t panelSendToHost, panelSendToDevice;
      hipEvent_t pdlaswpStart_1, pdlaswpStart_2;
      hipEvent_t pdlaswpFinish_1, pdlaswpFinish_2;
      hipEvent_t L1Transfer, L2Transfer;
      hipEvent_t swapStartEvent[HPL_N_UPD], update[HPL_N_UPD];
      hipEvent_t swapUCopyEvent[HPL_N_UPD], swapWCopyEvent[HPL_N_UPD];
      hipEvent_t dgemmStart[HPL_N_UPD], dgemmStop[HPL_N_UPD];
      std::map<int, const char*> _memcpyKind;
    }
}
