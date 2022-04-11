
#include <hpl.h>


extern "C" {
   /*
   * Initialize the Target device
   */
   void HPL_BE_init(size_t num_devices, enum HPL_TARGET TR)
   {
      switch(TR) {
         case T_CPU:
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
   void HPL_BE_malloc(void** ptr, size_t size, HPL_TARGET TR)
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
   * Deallocate the memory of ptr
   */
   void HPL_BE_free(void** ptr, enum HPL_TARGET TR)
   {
      switch(TR) {
         case T_CPU :
            HPL::dispatch(CPU::free, ptr);
            break;
         case T_HIP:
            HPL::dispatch(HIP::free, ptr);
            break;
         default:
            HPL::dispatch(CPU::free, ptr);
      }
   }

    /*
   * Allocate the panel resources
   */
   void HPL_BE_panel_new(HPL_T_grid *GRID, HPL_T_palg *ALGO, const int M, const int N, const int JB,
                         HPL_T_pmat *A, const int IA, const int JA, const int TAG, HPL_T_panel **PANEL, HPL_TARGET TR)
   {
      switch(TR) {
         case T_CPU :
            HPL::dispatch(CPU::panel_new, GRID, ALGO, M, N, JB, A, IA, JA, TAG, PANEL);
            break;
         case T_HIP:
            HPL::dispatch(HIP::panel_new, GRID, ALGO, M, N, JB, A, IA, JA, TAG, PANEL);
            break;
         default:
            HPL::dispatch(CPU::panel_new, GRID, ALGO, M, N, JB, A, IA, JA, TAG, PANEL);
      }
   }
    
    /*
   * Initialize the panel resources
   */
   void HPL_BE_panel_init(HPL_T_grid *GRID, HPL_T_palg *ALGO, const int M, const int N, const int JB,
                         HPL_T_pmat *A, const int IA, const int JA, const int TAG, HPL_T_panel *PANEL, HPL_TARGET TR)
   {
      switch(TR) {
         case T_CPU :
            HPL::dispatch(CPU::panel_init, GRID, ALGO, M, N, JB, A, IA, JA, TAG, PANEL);
            break;
         case T_HIP:
            HPL::dispatch(HIP::panel_init, GRID, ALGO, M, N, JB, A, IA, JA, TAG, PANEL);
            break;
         default:
            HPL::dispatch(CPU::panel_init, GRID, ALGO, M, N, JB, A, IA, JA, TAG, PANEL);
      }
   }

   /*
   * Deallocate the panel resources
   */
   void HPL_BE_panel_free(HPL_T_panel *ptr, HPL_TARGET TR)
   {
      switch(TR) {
         case T_CPU :
            HPL::dispatch(CPU::panel_free, ptr);
            break;
         case T_HIP:
            HPL::dispatch(HIP::panel_free, ptr);
            break;
         default:
            HPL::dispatch(CPU::panel_free, ptr);
      }
   }

   void HPL_BE_panel_send_to_device(HPL_T_panel* PANEL, HPL_TARGET TR)
   {
      switch(TR) {
         case T_CPU :
            DO_NOTHING();
            break;
         case T_HIP:
            HPL::dispatch(HIP::panel_send_to_device, PANEL);
            break;
         default:
            DO_NOTHING();
      }
   }

   void HPL_BE_panel_send_to_host(HPL_T_panel* PANEL, HPL_TARGET TR)
   {
      switch(TR) {
         case T_CPU :
            DO_NOTHING();
            break;
         case T_HIP:
            HPL::dispatch(HIP::panel_send_to_host, PANEL);
            break;
         default:
            DO_NOTHING();
      }
   }

   /*
   * Deallocates  the  panel  structure  and  resources.
   */
   void HPL_BE_panel_disp(HPL_T_panel **ptr, HPL_TARGET TR)
   {
      switch(TR) {
         case T_CPU :
            HPL::dispatch(CPU::panel_disp, ptr);
            break;
         case T_HIP:
            HPL::dispatch(HIP::panel_disp, ptr);
            break;
         default:
            HPL::dispatch(CPU::panel_disp, ptr);
      }
   }

   void HPL_BE_device_sync(HPL_TARGET TR)
   {
      switch(TR) {
         case T_CPU :
            DO_NOTHING();
            break;
         case T_HIP:
            HPL::dispatch(HIP::device_sync);
            break;
         default:
            DO_NOTHING();
      }
   }
   void HPL_BE_event_record(HPL_EVENT event, HPL_TARGET TR)
   {
      switch(TR) {
         case T_CPU :
            DO_NOTHING();
            break;
         case T_HIP:
            HPL::dispatch(HIP::event_record, event);
            break;
         default:
            DO_NOTHING();
      }
   }
   void HPL_BE_stream_sync(enum HPL_STREAM stream, enum HPL_TARGET TR)
   {
      switch(TR) {
         case T_CPU :
            DO_NOTHING();
            break;
         case T_HIP:
            HPL::dispatch(HIP::one_stream_sync, stream);
            break;
         default:
            DO_NOTHING();
      }
   }
   void HPL_BE_set_stream_handle(enum HPL_UPDATE_FLAG flag, enum HPL_TARGET TR)
   {
      switch(TR) {
         case T_CPU :
            DO_NOTHING();
            break;
         case T_HIP:
            HPL::dispatch(HIP::set_stream_handle, flag);
            break;
         default:
            DO_NOTHING();
      }
   }
   /*
   * Matrix generator
   */
   void HPL_BE_dmatgen(const HPL_T_grid *GRID, const int M, const int N,
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
*  ----------------------------------------------------------------------
*  - BLAS ---------------------------------------------------------------
*  ----------------------------------------------------------------------
*/ 

   /*
   * Find the index of the first element having maximum absolute value.
   */
   int  HPL_BE_idamax(const int N, const double *DX, const int INCX, enum HPL_TARGET TR)
   {
      switch(TR) {
         case T_CPU :
            return HPL::dispatch(CPU::idamax, N, DX, INCX);
            break;
         case T_HIP:
            return HPL::dispatch(HIP::idamax, N, DX, INCX);
            break;
         default:
            return HPL::dispatch(CPU::idamax, N, DX, INCX);
      }
   }

   /*
   * Scale the vector x by alpha and adds it to y.
   */
   void HPL_BE_daxpy(const int N, const double DA, const double *DX, const int INCX, double *DY, 
                const int INCY, enum HPL_TARGET TR)
   {
      switch(TR) {
         case T_CPU :
            HPL::dispatch(CPU::daxpy, N, DA, DX, INCX, DY, INCY);
            break;
         case T_HIP:
            HPL::dispatch(HIP::daxpy, N, DA, DX, INCX, DY, INCY);
            break;
         default:
            HPL::dispatch(CPU::daxpy, N, DA, DX, INCX, DY, INCY);
      }
   }

   /*
   * Scale the vector x by alpha
   */
   void HPL_BE_dscal(const int N, const double DA, double *DX, const int INCX, enum HPL_TARGET TR)
   {
      switch(TR) {
         case T_CPU :
            HPL::dispatch(CPU::dscal, N, DA, DX, INCX);
            break;
         case T_HIP:
            HPL::dispatch(HIP::dscal, N, DA, DX, INCX);
            break;
         default:
            HPL::dispatch(CPU::dscal, N, DA, DX, INCX);
      }
   }

   /*
   * Swap the vectors x and y.
   */
   void HPL_BE_dswap(const int N, double *DX, const int INCX, double *DY, const int INCY, enum HPL_TARGET TR)
   {
      switch(TR) {
         case T_CPU :
            HPL::dispatch(CPU::dswap, N, DX, INCX, DY, INCY);
            break;
         case T_HIP:
            HPL::dispatch(HIP::dswap, N, DX, INCX, DY, INCY);
            break;
         default:
            HPL::dispatch(CPU::dswap, N, DX, INCX, DY, INCY);
      }
   }

   /*
   * Perform the rank 1 operation
   */
   void HPL_BE_dger( const enum HPL_ORDER ORDER, const int M, const int N, const double ALPHA, const double *X,
               const int INCX, double *Y, const int INCY, double *A, const int LDA, enum HPL_TARGET TR)
   {
      switch(TR) {
         case T_CPU :
            HPL::dispatch(CPU::dger, ORDER, M, N, ALPHA, X, INCX, Y, INCY, A, LDA);
            break;
         case T_HIP:
            HPL::dispatch(HIP::dger, ORDER, M, N, ALPHA, X, INCX, Y, INCY, A, LDA);
            break;
         default:
            HPL::dispatch(CPU::dger, ORDER, M, N, ALPHA, X, INCX, Y, INCY, A, LDA);
      }
   }

   /*
   * Triangular Solver Matrix
   */
   void HPL_BE_dtrsm(const enum HPL_ORDER ORDER, const enum HPL_SIDE SIDE, 
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
            HPL::dispatch(HIP::trsm, ORDER, SIDE, UPLO, TRANSA, DIAG, M, N, ALPHA, A, LDA, B, LDB);
            break;
         default:
            HPL::dispatch(CPU::trsm, ORDER, SIDE, UPLO, TRANSA, DIAG, M, N, ALPHA, A, LDA, B, LDB);
      }
   }

   /*
   * Triangular Solver Vector
   */
   void HPL_BE_dtrsv(const enum HPL_ORDER ORDER, const enum HPL_UPLO UPLO,
                const enum HPL_TRANS TRANSA, const enum HPL_DIAG DIAG,
                const int N, const double *A, const int LDA,
                double *X, const int INCX, HPL_TARGET TR)
   {
      switch(TR) {
         case T_CPU :
            HPL::dispatch(CPU::trsv, ORDER, UPLO, TRANSA, DIAG, N, A, LDA, X, INCX);
            break;
         case T_HIP:
            HPL::dispatch(HIP::trsv, ORDER, UPLO, TRANSA, DIAG, N, A, LDA, X, INCX);
            break;
         default:
            HPL::dispatch(CPU::trsv, ORDER, UPLO, TRANSA, DIAG, N, A, LDA, X, INCX);
      }
   }

   /*
   * Douple precision GEMM OP
   */
   void HPL_BE_dgemm(const enum HPL_ORDER ORDER, const enum HPL_TRANS TRANSA, 
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
            HPL::dispatch(HIP::dgemm, ORDER, TRANSA, TRANSB, M, N, K, ALPHA, A, LDA, B, LDB, BETA, C, LDC);
            break;
         default:
            HPL::dispatch(CPU::dgemm, ORDER, TRANSA, TRANSB, M, N, K, ALPHA, A, LDA, B, LDB, BETA, C, LDC);
      }
   }

   /*
   * Douple precision GEMV OP
   */
   void HPL_BE_dgemv(const enum HPL_ORDER ORDER, const enum HPL_TRANS TRANS, const int M, const int N,
                const double ALPHA, const double *A, const int LDA, const double *X, const int INCX,
                const double BETA, double *Y, const int INCY, enum HPL_TARGET TR)
   {
      switch(TR) {
         case T_CPU :
            HPL::dispatch(CPU::dgemv, ORDER, TRANS, M, N, ALPHA, A, LDA, X, INCX, BETA, Y, INCY);
            break;
         case T_HIP:
            HPL::dispatch(HIP::dgemv, ORDER, TRANS, M, N, ALPHA, A, LDA, X, INCX, BETA, Y, INCY);
            break;
         default:
            HPL::dispatch(CPU::dgemv, ORDER, TRANS, M, N, ALPHA, A, LDA, X, INCX, BETA, Y, INCY);
      }   
   }

   /*
   * Copies the vector x into the vector y
   */
   void HPL_BE_dcopy(const int N, const double *X, const int INCX, double *Y, const int INCY, enum HPL_TARGET TR)
   {
      switch(TR) {
         case T_CPU :
            HPL::dispatch(CPU::copy, N, X, INCX, Y, INCY);
            break;
         case T_HIP:
            HPL::dispatch(HIP::copy, N, X, INCX, Y, INCY);
            break;
         default:
            HPL::dispatch(CPU::copy, N, X, INCX, Y, INCY);
      }
   }

   /*
   * copies array A into an array B.
   */
   void HPL_BE_dlacpy(const int M, const int N, const double *A, const int LDA,
                  double *B, const int LDB, enum HPL_TARGET TR)
   {
      switch(TR) {
         case T_CPU :
            HPL::dispatch(CPU::acpy, M, N, A, LDA, B, LDB);
            break;
         case T_HIP:
            HPL::dispatch(HIP::acpy, M, N, A, LDA, B, LDB);
            break;
         default:
            HPL::dispatch(CPU::acpy, M, N, A, LDA, B, LDB);
      }
   }

   /*
   * Copies the transpose of an array A into an array B
   */
   void HPL_BE_dlatcpy(const int M, const int N, const double *A, const int LDA,
                double *B, const int LDB, enum HPL_TARGET TR)
   {
      switch(TR) {
         case T_CPU :
            HPL::dispatch(CPU::atcpy, M, N, A, LDA, B, LDB);
            break;
         case T_HIP:
            HPL::dispatch(HIP::atcpy, M, N, A, LDA, B, LDB);
            break;
         default:
            HPL::dispatch(CPU::atcpy, M, N, A, LDA, B, LDB);
      }
   }

   void HPL_BE_move_data(double *DST, const double *SRC, const size_t SIZE, 
                     enum HPL_MOVE_DIRECTION KIND, enum HPL_TARGET TR)
   {
      switch(TR) {
         case T_CPU :
            DO_NOTHING();
            break;
         case T_HIP:
            HPL::dispatch(HIP::move_data, DST, SRC, SIZE, (int)KIND);
            break;
         default:
            DO_NOTHING();
      }
   }
   void HPL_BE_dlaswp00N(const int M, const int N, double * A, const int LDA, const int * IPIV, enum HPL_TARGET TR) 
   {
      switch(TR) {
         case T_CPU :
            HPL::dispatch(CPU::dlaswp00N, M, N, A, LDA, IPIV);
            break;
         case T_HIP:
            HPL::dispatch(HIP::dlaswp00N, M, N, A, LDA, IPIV);
            break;
         default:
            HPL::dispatch(CPU::dlaswp00N, M, N, A, LDA, IPIV);
      }
   }
} //extern "C"
