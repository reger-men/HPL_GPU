#include "backend/hpl_backendCPU.h"
#include <hpl.h>

void CPU::malloc(void** ptr, size_t size)
{
    CPUInfo("%-25s %-12ld (B) \t%-5s", "[Allocate]", "Memory of size",  size, "CPU");
    #if 0
    *ptr = std::malloc(size);
    #endif
    hipHostMalloc(ptr, size, hipHostMallocDefault);

}

void CPU::free(void** ptr)
{
    //std::free(*ptr);
    hipHostFree(*ptr);
}

int CPU::panel_free(HPL_T_panel *ptr)
{
    CPUInfo("%-40s \t%-5s", "[Deallocate]", "Panel resources", "CPU");
    if( ptr->WORK  ) HIP_CHECK_ERROR(hipHostFree( ptr->WORK  ));
    if( ptr->IWORK ) HIP_CHECK_ERROR(hipHostFree( ptr->IWORK ));
    return( MPI_SUCCESS );
    // return HPL_pdpanel_free( ptr );
}

int CPU::panel_disp(HPL_T_panel **ptr)
{
    CPUInfo("%-40s \t%-5s", "[Deallocate]", "Panel structure", "CPU");
    return HPL_pdpanel_disp( ptr );
}

void printMat(const int M, const int N, const int LDA, const double *A)
{
    // Last row is the vector b
    for(int y=0;y<M+1; y++){
        for(int x=0;x<N-1; x++){
            int index = x+y*LDA;
            printf("%-4d:%-8lf\t", index, A[index]);
        }
        printf("\n");
    }
}

void CPU::matgen(const HPL_T_grid *GRID, const int M, const int N,
                 const int NB, double *A, const int LDA,
                 const int ISEED)
{
    CPUInfo("%-25s %-8d%-8d \t%-5s", "[Generate matrix]", "With A of (R:C)", M, N, "CPU");
    HPL_pdmatgen(GRID, M, N, NB, A, LDA, ISEED);
    //printMat(M,N,LDA,A);
}

int CPU::idamax(const int N, const double *DX, const int INCX)
{
    CPUInfo("%-25s %-17d \t%-5s", "[IDAMAX]", "With X of (R)", N, "CPU");
    return HPL_idamax( N, DX, INCX );
}

void CPU::daxpy(const int N, const double DA, const double *DX, const int INCX, double *DY, 
                const int INCY)
{
    CPUInfo("%-25s %-17d \t%-5s", "[DAXPY]", "With X of (R)", N, "CPU");
    HPL_daxpy(N, DA, DX, INCX, DY, INCY);
}

void CPU::dscal(const int N, const double DA, double *DX, const int INCX)
{
    CPUInfo("%-25s %-17d \t%-5s", "[DSCAL]", "With X of (R)", N, "CPU");
    HPL_dscal(N, DA, DX, INCX);
}

void CPU::dswap(const int N, double *DX, const int INCX, double *DY, const int INCY)
{    
    CPUInfo("%-25s %-17d \t%-5s", "[DSWAP]", "With X of (R)", N, "CPU");
    HPL_dswap(N, DX, INCX, DY, INCY);
}

void CPU::dger( const enum HPL_ORDER ORDER, const int M, const int N, const double ALPHA, const double *X,
               const int INCX, double *Y, const int INCY, double *A, const int LDA)
{
    CPUInfo("%-25s %-8d%-8d \t%-5s", "[DGER]", "With A of (R:C)", M, N, "CPU");
    HPL_dger(ORDER, M, N, ALPHA, X, INCX, Y, INCY, A, LDA);
}

void CPU::trsm( const enum HPL_ORDER ORDER, const enum HPL_SIDE SIDE, 
                const enum HPL_UPLO UPLO, const enum HPL_TRANS TRANSA, 
                const enum HPL_DIAG DIAG, const int M, const int N, 
                const double ALPHA, const double *A, const int LDA, double *B, const int LDB)
{
    CPUInfo("%-25s %-8d%-8d \t%-5s", "[TRSM]", "With B of (R:C)", M, N, "CPU");
    HPL_dtrsm(ORDER, SIDE, UPLO, TRANSA, DIAG, M, N, ALPHA, A, LDA, B, LDB);
}

void CPU::trsv(const enum HPL_ORDER ORDER, const enum HPL_UPLO UPLO,
                const enum HPL_TRANS TRANSA, const enum HPL_DIAG DIAG,
                const int N, const double *A, const int LDA,
                double *X, const int INCX)
{  
    CPUInfo("%-25s %-17d \t%-5s", "[TRSV]", "With A of (R)", N, "CPU");
    HPL_dtrsv( ORDER, UPLO, TRANSA, DIAG, N, A, LDA, X, INCX);
}

void CPU::dgemm(const enum HPL_ORDER ORDER, const enum HPL_TRANS TRANSA, 
                const enum HPL_TRANS TRANSB, const int M, const int N, const int K, 
                const double ALPHA, const double *A, const int LDA, 
                const double *B, const int LDB, const double BETA, double *C, 
                const int LDC)
{
    CPUInfo("%-25s %-8d%-8d \t%-5s", "[DGEMM]", "With C of (R:C)", M, N, "CPU");
    HPL_dgemm(ORDER, TRANSA, TRANSB, M, N, K, ALPHA, A, LDA, B, LDB, BETA, C, LDC);
}

void CPU::dgemv(const enum HPL_ORDER ORDER, const enum HPL_TRANS TRANS, const int M, const int N,
                const double ALPHA, const double *A, const int LDA, const double *X, const int INCX,
                const double BETA, double *Y, const int INCY)
{
    CPUInfo("%-25s %-8d%-8d \t%-5s", "[DGEMV]", "With A of (R:C)", M, N, "CPU");
    HPL_dgemv(ORDER, TRANS, M, N, ALPHA, A, LDA, X, INCX, BETA, Y, INCY);
}

void CPU::copy(const int N, const double *X, const int INCX, double *Y, const int INCY)
{
    CPUInfo("%-25s %-17d \t%-5s", "[COPY]", "With X of (R)", N, "CPU");
    HPL_dcopy( N, X, INCX, Y, INCY);
}

void CPU::acpy(const int M, const int N, const double *A, const int LDA,
                  double *B, const int LDB)
{
    CPUInfo("%-25s %-8d%-8d \t%-5s", "[LACOPY]", "With A of (R:C)", M, N, "CPU");
    HPL_dlacpy(M, N, A, LDA, B, LDB);
}

void CPU::atcpy(const int M, const int N, const double *A, const int LDA,
                double *B, const int LDB)
{
    CPUInfo("%-25s %-8d%-8d \t%-5s", "[LATCOPY]", "With A of (R:C)", M, N, "CPU");
    HPL_dlatcpy(M, N, A, LDA, B, LDB);
}