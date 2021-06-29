#pragma once

#include <stdint.h>
#include <stddef.h>

#define DO_NOTHING()

#ifdef __cplusplus
extern "C" {
#endif


#include "hpl_grid.h"
#include "hpl_blas.h"
#include "hpl_panel.h"

enum HPL_TARGET {T_DEFAULT, T_CPU, T_HIP};

void HPL_btinit(size_t, enum HPL_TARGET);

void HPL_bmalloc(void**, size_t, enum HPL_TARGET);

void HPL_bfree(void **, enum HPL_TARGET);
void HPL_bpanel_free(HPL_T_panel *, enum HPL_TARGET);
void HPL_bpanel_disp(HPL_T_panel**, enum HPL_TARGET);

void HPL_bmatgen(const HPL_T_grid *, const int, const int,
                 const int, double *, const int,
                 const int, enum HPL_TARGET);

/*
*  ----------------------------------------------------------------------
*  - BLAS ---------------------------------------------------------------
*  ----------------------------------------------------------------------
*/ 

int  HPL_bidamax(const int, const double *, const int, enum HPL_TARGET);
void HPL_bdaxpy(const int, const double, const double *, const int, double *, 
                const int, enum HPL_TARGET);
void HPL_bdscal(const int, const double, double *, const int, enum HPL_TARGET);
void HPL_bdswap(const int, double *, const int, double *, const int, enum HPL_TARGET);

void HPL_bdger( const enum HPL_ORDER, const int, const int, const double, const double *,
               const int, double *, const int, double *, const int, enum HPL_TARGET);







void HPL_btrsm(const enum HPL_ORDER, const enum HPL_SIDE, 
                const enum HPL_UPLO, const enum HPL_TRANS, 
                const enum HPL_DIAG, const int, const int, 
                const double, const double *, const int, double *, 
                const int, enum HPL_TARGET);

void HPL_btrsv(const enum HPL_ORDER, const enum HPL_UPLO,
                const enum HPL_TRANS, const enum HPL_DIAG,
                const int, const double *, const int,
                double *, const int, enum HPL_TARGET);                

void HPL_bdgemm(const enum HPL_ORDER, const enum HPL_TRANS, const enum HPL_TRANS,
                const int, const int, const int, const double, const double *,
                const int, const double *, const int, const double, double *, 
                const int, enum HPL_TARGET);

void HPL_bdgemv(const enum HPL_ORDER, const enum HPL_TRANS, const int, const int,
                const double, const double *, const int, const double *, const int,
                const double, double *, const int, enum HPL_TARGET);

void HPL_bcopy(const int, const double *, const int, double *, const int, enum HPL_TARGET);

void HPL_bacpy(const int, const int, const double *, const int, double *, const int, enum HPL_TARGET);

void HPL_batcpy(const int, const int, const double *, const int,
                double *, const int, enum HPL_TARGET);

#ifdef __cplusplus
}
#endif