#include "backend/hpl_backendCPU.h"

void CPU::malloc(void** ptr, size_t size)
{
    CPUInfo("allocate memory on CPU of size %ld", size);
    *ptr = std::malloc(size);
}

void CPU::matgen(const HPL_T_grid *GRID, const int M, const int N,
                 const int NB, double *A, const int LDA,
                 const int ISEED)
{
    CPUInfo("generate matrix on CPU");
    HPL_pdmatgen(GRID, M, N, NB, A, LDA, ISEED);
}

void CPU::trsm( const enum HPL_ORDER ORDER, const enum HPL_SIDE SIDE, 
                const enum HPL_UPLO UPLO, const enum HPL_TRANS TRANSA, 
                const enum HPL_DIAG DIAG, const int M, const int N, 
                const double ALPHA, const double *A, const int LDA, double *B, const int LDB)
{
    CPUInfo("TRSM on CPU");
    HPL_dtrsm(ORDER, SIDE, UPLO, TRANSA, DIAG, M, N, ALPHA, A, LDA, B, LDB);
}

void CPU::dgemm(const enum HPL_ORDER ORDER, const enum HPL_TRANS TRANSA, 
                const enum HPL_TRANS TRANSB, const int M, const int N, const int K, 
                const double ALPHA, const double *A, const int LDA, 
                const double *B, const int LDB, const double BETA, double *C, 
                const int LDC)
{
    CPUInfo("DGEMM on CPU");
    HPL_dgemm(ORDER, TRANSA, TRANSB, M, N, K, ALPHA, A, LDA, B, LDB, BETA, C, LDC);
}

void CPU::dgemv(const enum HPL_ORDER ORDER, const enum HPL_TRANS TRANS, const int M, const int N,
                const double ALPHA, const double *A, const int LDA, const double *X, const int INCX,
                const double BETA, double *Y, const int INCY)
{
    CPUInfo("DGEMV on CPU");
    HPL_dgemv(ORDER, TRANS, M, N, ALPHA, A, LDA, X, INCX, BETA, Y, INCY);
}

void CPU::acpy(const int M, const int N, const double *A, const int LDA,
                  double *B, const int LDB)
{
    CPUInfo("A copy on CPU");
    HPL_dlacpy(M, N, A, LDA, B, LDB);
}

void CPU::atcpy(const int M, const int N, const double *A, const int LDA,
                double *B, const int LDB)
{
    CPUInfo("A transpose copy on CPU");
    HPL_dlatcpy(M, N, A, LDA, B, LDB);
}