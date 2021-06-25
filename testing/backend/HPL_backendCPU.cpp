#include "backend/hpl_backendCPU.h"

void CPU::malloc(void** ptr, size_t size)
{
    printf("allocate memory on CPU of size %ld\n", size);
    *ptr = std::malloc(size);
}

void CPU::matgen(const HPL_T_grid *GRID, const int M, const int N,
                 const int NB, double *A, const int LDA,
                 const int ISEED)
{
    printf("generate matrix on CPU\n");
    HPL_pdmatgen(GRID, M, N, NB, A, LDA, ISEED);
}

void CPU::trsm( const enum HPL_ORDER ORDER, const enum HPL_SIDE SIDE, 
                const enum HPL_UPLO UPLO, const enum HPL_TRANS TRANSA, 
                const enum HPL_DIAG DIAG, const int M, const int N, 
                const double ALPHA, const double *A, const int LDA, double *B, const int LDB)
{
    printf("TRSM on CPU\n");
    HPL_dtrsm(ORDER, SIDE, UPLO, TRANSA, DIAG, M, N, ALPHA, A, LDA, B, LDB);
}

void CPU::dgemm(const enum HPL_ORDER ORDER, const enum HPL_TRANS TRANSA, 
                const enum HPL_TRANS TRANSB, const int M, const int N, const int K, 
                const double ALPHA, const double *A, const int LDA, 
                const double *B, const int LDB, const double BETA, double *C, 
                const int LDC)
{
    printf("DGEMM on CPU\n");
    HPL_dgemm(ORDER, TRANSA, TRANSB, M, N, K, ALPHA, A, LDA, B, LDB, BETA, C, LDC);
}

void CPU::atcpy(const int M, const int N, const double *A, const int LDA,
                double *B, const int LDB)
{
    printf("A transpose copy on CPU\n");
    HPL_dlatcpy(M, N, A, LDA, B, LDB);
}