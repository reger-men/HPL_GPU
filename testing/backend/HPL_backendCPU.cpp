#include "backend/hpl_backendCPU.h"

void CPU::malloc(void** ptr, size_t size)
{
    CPUInfo("---\t[Allocate] \tmemory of size %ld on \tCPU", size);
    *ptr = std::malloc(size);
}

void CPU::free(void** ptr)
{
    std::free(*ptr);
}

int CPU::panel_free(HPL_T_panel *ptr)
{
    CPUInfo("---\t[Deallocate] \tthe panel resources on \tCPU\n");
    return HPL_pdpanel_free( ptr );
}

int CPU::panel_disp(HPL_T_panel **ptr)
{
    CPUInfo("---\t[Deallocate] \tthe panel structure on \t\tCPU\n");
    return HPL_pdpanel_disp( ptr );
}

void CPU::matgen(const HPL_T_grid *GRID, const int M, const int N,
                 const int NB, double *A, const int LDA,
                 const int ISEED)
{
    CPUInfo("---\t[Generate matrix] \twith A of R=%d:C=%d on \tCPU", M, N);
    HPL_pdmatgen(GRID, M, N, NB, A, LDA, ISEED);
}

int CPU::idamax(const int N, const double *DX, const int INCX)
{
    CPUInfo("---\t[DMAX] \twith X of R=%d on \tCPU", N);
    HPL_idamax( N, DX, INCX );
}

void CPU::daxpy(const int N, const double DA, const double *DX, const int INCX, double *DY, 
                const int INCY)
{
    CPUInfo("---\t[DAXPY] \twith X of R=%d on \tCPU", N);
    HPL_daxpy(N, DA, DX, INCX, DY, INCY);
}

void CPU::dscal(const int N, const double DA, double *DX, const int INCX)
{
    CPUInfo("---\t[DSCAL] \twith X of R=%d on \t\tCPU", N);
    HPL_dscal(N, DA, DX, INCX);
}

void CPU::dswap(const int N, double *DX, const int INCX, double *DY, const int INCY)
{    
    CPUInfo("---\t[DSWAP] \twith X of R=%d on \t\t\tCPU", N);
    HPL_dswap(N, DX, INCX, DY, INCY);
}

void CPU::dger( const enum HPL_ORDER ORDER, const int M, const int N, const double ALPHA, const double *X,
               const int INCX, double *Y, const int INCY, double *A, const int LDA)
{
    CPUInfo("---\t[DGER] \t\twith A of R=%d:C=%d on \t\tCPU", M, N);
    HPL_dger(ORDER, M, N, ALPHA, X, INCX, Y, INCY, A, LDA);
}

void CPU::trsm( const enum HPL_ORDER ORDER, const enum HPL_SIDE SIDE, 
                const enum HPL_UPLO UPLO, const enum HPL_TRANS TRANSA, 
                const enum HPL_DIAG DIAG, const int M, const int N, 
                const double ALPHA, const double *A, const int LDA, double *B, const int LDB)
{
    CPUInfo("---\t[TRSM] \t\twith B of R=%d:C=%d on \t\tCPU", M, N);
    HPL_dtrsm(ORDER, SIDE, UPLO, TRANSA, DIAG, M, N, ALPHA, A, LDA, B, LDB);
}

void CPU::trsv(const enum HPL_ORDER ORDER, const enum HPL_UPLO UPLO,
                const enum HPL_TRANS TRANSA, const enum HPL_DIAG DIAG,
                const int N, const double *A, const int LDA,
                double *X, const int INCX)
{  
    CPUInfo("---\t[TRSV] \t\twith A of R=%d on \t\tCPU", N);
    HPL_dtrsv( ORDER, UPLO, TRANSA, DIAG, N, A, LDA, X, INCX);
}

void CPU::dgemm(const enum HPL_ORDER ORDER, const enum HPL_TRANS TRANSA, 
                const enum HPL_TRANS TRANSB, const int M, const int N, const int K, 
                const double ALPHA, const double *A, const int LDA, 
                const double *B, const int LDB, const double BETA, double *C, 
                const int LDC)
{
    CPUInfo("---\t[DGEMM] \twith C of R=%d:C=%d on \tCPU", M, N);
    HPL_dgemm(ORDER, TRANSA, TRANSB, M, N, K, ALPHA, A, LDA, B, LDB, BETA, C, LDC);
}

void CPU::dgemv(const enum HPL_ORDER ORDER, const enum HPL_TRANS TRANS, const int M, const int N,
                const double ALPHA, const double *A, const int LDA, const double *X, const int INCX,
                const double BETA, double *Y, const int INCY)
{
    CPUInfo("---\t[DGEMV] \twith A of R=%d:C=%d on \tCPU", M, N);
    HPL_dgemv(ORDER, TRANS, M, N, ALPHA, A, LDA, X, INCX, BETA, Y, INCY);
}

void CPU::copy(const int N, const double *X, const int INCX, double *Y, const int INCY)
{
    CPUInfo("---\t[COPY] \t\tVector with X of R=%d on \tCPU", N);
    HPL_dcopy( N, X, INCX, Y, INCY);
}

void CPU::acpy(const int M, const int N, const double *A, const int LDA,
                  double *B, const int LDB)
{
    CPUInfo("---\t[LACOPY] \twith A of R=%d:C=%d on \tCPU", M, N);
    HPL_dlacpy(M, N, A, LDA, B, LDB);
}

void CPU::atcpy(const int M, const int N, const double *A, const int LDA,
                double *B, const int LDB)
{
    CPUInfo("---\t[LATCOPY] \twith A of R=%d:C=%d on \tCPU", M, N);
    HPL_dlatcpy(M, N, A, LDA, B, LDB);
}