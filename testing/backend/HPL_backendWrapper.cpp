#pragma once

#include "backend/hpl_backendWrapper.h"
#include "backend/hpl_backendCommon.h"



extern "C" {

   /*
   * Initialize the Target device
   */
   void HPL_btinit(size_t num_devices, enum HPL_TARGET TR)
   {
      switch(TR) {
         case T_CPU :
            DO_NOTHING();
            break;
         case T_HIP:
            HPL::dispatch(HIP::init, num_devices);
            break;
         default:
            DO_NOTHING();
      }
   }
   
   /*
   * Allocate memory
   */
   void HPL_bmalloc(void** ptr, size_t size, HPL_TARGET TR)
   {
      switch(TR) {
         case T_CPU :
            HPL::dispatch(CPU::malloc, ptr, size);
            break;
         case T_HIP:
            HPL::dispatch(HIP::malloc, ptr, size);
            break;
         default:
            HPL::dispatch(CPU::malloc, ptr, size);
      }
   }

   /*
   * Matrix generator
   */
   void HPL_bmatgen(const HPL_T_grid *GRID, const int M, const int N,
                 const int NB, double *A, const int LDA,
                 const int ISEED, HPL_TARGET TR)
   {
      switch(TR) {
         case T_CPU :
            HPL::dispatch(CPU::matgen, GRID, M, N, NB, A, LDA, ISEED);
            break;
         case T_HIP:
            HPL::dispatch(HIP::matgen, GRID, M, N, NB, A, LDA, ISEED);
            break;
         default:
            HPL::dispatch(CPU::matgen, GRID, M, N, NB, A, LDA, ISEED);
      }
   }

   /*
   * Triangular Solver Matrix
   */
   void HPL_btrsm(const enum HPL_ORDER ORDER, const enum HPL_SIDE SIDE, 
                const enum HPL_UPLO UPLO, const enum HPL_TRANS TRANSA, 
                const enum HPL_DIAG DIAG, const int M, const int N, 
                const double ALPHA, const double *A, const int LDA, 
                double *B, const int LDB, HPL_TARGET TR)
   {
      switch(TR) {
         case T_CPU :
            HPL::dispatch(CPU::trsm, ORDER, SIDE, UPLO, TRANSA, DIAG, M, N, ALPHA, A, LDA, B, LDB);
            break;
         case T_HIP:
            printf("NOT IMPLEMENTED!!\n");
            break;
         default:
            HPL::dispatch(CPU::trsm, ORDER, SIDE, UPLO, TRANSA, DIAG, M, N, ALPHA, A, LDA, B, LDB);
      }
   }

   /*
   * Douple precision GEMM OP
   */
   void HPL_bdgemm(const enum HPL_ORDER ORDER, const enum HPL_TRANS TRANSA, 
                const enum HPL_TRANS TRANSB, const int M, const int N, const int K, 
                const double ALPHA, const double *A, const int LDA, 
                const double *B, const int LDB, const double BETA, double *C, 
                const int LDC, enum HPL_TARGET TR)
   {
      switch(TR) {
         case T_CPU :
            HPL::dispatch(CPU::dgemm, ORDER, TRANSA, TRANSB, M, N, K, ALPHA, A, LDA, B, LDB, BETA, C, LDC);
            break;
         case T_HIP:
            printf("NOT IMPLEMENTED!!\n");
            break;
         default:
            HPL::dispatch(CPU::dgemm, ORDER, TRANSA, TRANSB, M, N, K, ALPHA, A, LDA, B, LDB, BETA, C, LDC);
      }
   }

   /*
   * Douple precision GEMV OP
   */
   void HPL_bdgemv(const enum HPL_ORDER ORDER, const enum HPL_TRANS TRANS, const int M, const int N,
                const double ALPHA, const double *A, const int LDA, const double *X, const int INCX,
                const double BETA, double *Y, const int INCY, enum HPL_TARGET TR)
   {
      switch(TR) {
         case T_CPU :
            HPL::dispatch(CPU::dgemv, ORDER, TRANS, M, N, ALPHA, A, LDA, X, INCX, BETA, Y, INCY);
            break;
         case T_HIP:
            printf("NOT IMPLEMENTED!!\n");
            break;
         default:
            HPL::dispatch(CPU::dgemv, ORDER, TRANS, M, N, ALPHA, A, LDA, X, INCX, BETA, Y, INCY);
      }   
   }

   /*
   * copies array A into an array B.
   */
   void HPL_bacpy(const int M, const int N, const double *A, const int LDA,
                  double *B, const int LDB, enum HPL_TARGET TR)
   {
      switch(TR) {
         case T_CPU :
            HPL::dispatch(CPU::acpy, M, N, A, LDA, B, LDB);
            break;
         case T_HIP:
            printf("NOT IMPLEMENTED!!\n");
            break;
         default:
            HPL::dispatch(CPU::acpy, M, N, A, LDA, B, LDB);
      }
   }

   /*
   * Copies the transpose of an array A into an array B
   */
   void HPL_batcpy(const int M, const int N, const double *A, const int LDA,
                double *B, const int LDB, enum HPL_TARGET TR)
   {
      switch(TR) {
         case T_CPU :
            HPL::dispatch(CPU::atcpy, M, N, A, LDA, B, LDB);
            break;
         case T_HIP:
            printf("NOT IMPLEMENTED!!\n");
            break;
         default:
            HPL::dispatch(CPU::atcpy, M, N, A, LDA, B, LDB);
      }
   }
} //extern "C"