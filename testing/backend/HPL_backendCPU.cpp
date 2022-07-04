#include "backend/hpl_backendCPU.h"

void CPU::malloc(void** ptr, size_t size)
{
    CPUInfo("%-25s %-12ld (B) \t%-5s", "[Allocate]", "Memory of size",  size, "CPU");
    *ptr = std::malloc(size);
}

void CPU::free(void** ptr)
{
    std::free(*ptr);
}

void CPU::panel_new(HPL_T_grid *GRID, HPL_T_palg *ALGO, const int M, const int N, const int JB, HPL_T_pmat *A,
               const int IA, const int JA, const int TAG, HPL_T_panel **PANEL)
{
    CPUInfo("%-40s \t%-5s", "[Allocate]", "Panel new", "CPU");
    HPL_pdpanel_new(GRID, ALGO, M, N, JB, A, IA, JA, TAG, PANEL);
}

void CPU::panel_init(HPL_T_grid *GRID, HPL_T_palg *ALGO, const int M, const int N, const int JB, 
                        HPL_T_pmat *A, const int IA, const int JA, const int TAG, HPL_T_panel *PANEL)
{
    CPUInfo("%-40s \t%-5s", "[Allocate]", "Panel init", "CPU");
    HPL_pdpanel_init(GRID, ALGO, M, N, JB, A, IA, JA, TAG, PANEL);
}

int CPU::panel_free(HPL_T_panel *ptr)
{
    CPUInfo("%-40s \t%-5s", "[Deallocate]", "Panel resources", "CPU");
    return HPL_pdpanel_free( ptr );
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

void CPU::dlaswp00N(const int M, const int N, double * A, const int LDA, const int * IPIV)
{
    CPUInfo("%-25s %-8d%-8d \t%-5s", "[DLASWP00N]", "With A of (R:C)", M, N, "CPU");
    HPL_dlaswp00N( M, N, A, LDA, IPIV);
}

double CPU::pdlange(const HPL_T_grid* GRID, const HPL_T_NORM  NORM, const int M, const int N, const int NB, const double* A, const int LDA)
{
    CPUInfo("%-25s %-8d%-8d \t%-5s", "[PDLANGE]", "With A of (R:C)", M, N, "CPU");
    return HPL_pdlange(GRID, NORM, M, N, NB, A, LDA);
}

void CPU::pdmatfree(void* mat)
{
    CPUInfo("%-40s \t%-5s", "[Deallocate]", "Matrix structure", "CPU");
    CPU::free((void**)&mat);
}

void CPU::HPL_dgemm_omp(const enum HPL_ORDER ORDER, const enum HPL_TRANS TRANSA, const enum HPL_TRANS TRANSB,
                   const int M, const int N, const int K,
                   const double ALPHA, const double* A, const int LDA,
                   const double* B, const int LDB, const double BETA,
                   double* C, const int LDC, 
                   const int NB, const int II,
                   const int thread_rank, const int thread_size) 
{
  int tile = 0;
  if(tile % thread_size == thread_rank) {
    const int mm = Mmin(NB - II, M);
    HPL_dgemm(ORDER, TRANSA, TRANSB, mm, N, K, ALPHA, A, LDA, B, LDB, BETA, C, LDC);
  }
  ++tile;
  int i = NB - II;
  for(; i < M; i += NB) {
    if(tile % thread_size == thread_rank) {
      const int mm = Mmin(NB, M - i);
      HPL_dgemm(ORDER, TRANSA, TRANSB, mm, N, K, ALPHA, A + i, LDA, B, LDB, BETA, C + i, LDC);
    }
    ++tile;
  }
}

void CPU::HPL_dlacpy(const int M, const int N, const double* A, const int LDA, double* B, const int LDB) {
  int j;

  if((M <= 0) || (N <= 0)) return;

  for(j = 0; j < N; j++, A += LDA, B += LDB) HPL_dcopy(M, A, 1, B, 1);
}

void CPU::HPL_pdmxswp(HPL_T_panel* PANEL, const int M, const int II, const int JJ, double* WORK) {

  double *     A0, *Wmx, *Wwork;
  HPL_T_grid*  grid;
  MPI_Comm     comm;
  int cnt_, cnt0, i, icurrow, lda, myrow, n0;

#ifdef HPL_DETAILED_TIMING
  HPL_ptimer(HPL_TIMING_MXSWP);
#endif
  grid  = PANEL->grid;
  comm    = grid->col_comm;
  myrow = grid->myrow;
  n0      = PANEL->jb;
  int NB  = PANEL->nb;
  icurrow = PANEL->prow;
  cnt0 = 4 + 2*NB;

  A0    = (Wmx = WORK + 4) + NB;
  Wwork = WORK + cnt0;

  if(M > 0) {
    lda = PANEL->lda;

    HPL_dcopy(n0, Mptr(PANEL->A, II + (int)(WORK[1]), 0, lda), lda, Wmx, 1);
    if(myrow == icurrow) {
      HPL_dcopy(n0, Mptr(PANEL->A, II, 0, lda), lda, A0, 1);
    } else {
      for(i = 0; i < n0; i++) A0[i] = HPL_rzero;
    }
  } else {
    for(i = 0; i < n0; i++) A0[i] = HPL_rzero;
    for(i = 0; i < n0; i++) Wmx[i] = HPL_rzero;
  }

  /* Perform swap-broadcast */
  CPU::HPL_all_reduce_dmxswp(WORK, cnt0, icurrow, comm, Wwork);

  (PANEL->ipiv)[JJ] = (int)WORK[2];
#ifdef HPL_DETAILED_TIMING
  HPL_ptimer(HPL_TIMING_MXSWP);
#endif
}

void CPU::HPL_dlocswpN(HPL_T_panel* PANEL, const int II, const int JJ, double* WORK) {

  double  gmax;
  double *A1, *A2, *L, *Wr0, *Wmx;
  int     ilindx, lda, myrow, n0;

  myrow = PANEL->grid->myrow;
  n0    = PANEL->jb;
  int NB = PANEL->nb;
  lda   = PANEL->lda;

  Wr0     = (Wmx = WORK + 4) + NB;
  Wmx[JJ] = gmax = WORK[0];

  /*
   * Replicated swap and copy of the current (new) row of A into L1
   */
  L = Mptr(PANEL->L1, JJ, 0, n0);
  /*
   * If the pivot is non-zero ...
   */
  if(gmax != HPL_rzero) {
    /*
     * and if I own the current row of A ...
     */
    if(myrow == PANEL->prow) {
      /*
       * and if I also own the row to be swapped with the current row of A ...
       */
      if(myrow == (int)(WORK[3])) {
        /*
         * and if the current row of A is not to swapped with itself ...
         */
        if((ilindx = (int)(WORK[1])) != 0) {
          /*
           * then copy the max row into L1 and locally swap the 2 rows of A.
           */
          A1 = Mptr(PANEL->A, II, 0, lda);
          A2 = Mptr(A1, ilindx, 0, lda);

          HPL_dcopy(n0, Wmx, 1, L, n0);
          HPL_dcopy(n0, Wmx, 1, A1, lda);
          HPL_dcopy(n0, Wr0, 1, A2, lda);

        } else {
          /*
           * otherwise the current row of  A  is swapped with itself, so just
           * copy the current of A into L1.
           */
          *Mptr(PANEL->A, II, JJ, lda) = gmax;

          HPL_dcopy(n0, Wmx, 1, L, n0);
        }

      } else {
        /*
         * otherwise, the row to be swapped with the current row of A is in Wmx,
         * so copy Wmx into L1 and A.
         */
        A1 = Mptr(PANEL->A, II, 0, lda);

        HPL_dcopy(n0, Wmx, 1, L, n0);
        HPL_dcopy(n0, Wmx, 1, A1, lda);
      }

    } else {
      /*
       * otherwise I do not own the current row of A, so copy the max row  Wmx
       * into L1.
       */
      HPL_dcopy(n0, Wmx, 1, L, n0);

      /*
       * and if I own the max row, overwrite it with the current row Wr0.
       */
      if(myrow == (int)(WORK[3])) {
        A2 = Mptr(PANEL->A, II + (size_t)(WORK[1]), 0, lda);

        HPL_dcopy(n0, Wr0, 1, A2, lda);
      }
    }
  } else {
    /*
     * Otherwise the max element in the current column is zero,  simply copy
     * the current row Wr0 into L1. The matrix is singular.
     */
    HPL_dcopy(n0, Wr0, 1, L, n0);

    /*
     * set INFO.
     */
    if(*(PANEL->DINFO) == 0.0) *(PANEL->DINFO) = (double)(PANEL->ia + JJ + 1);
  }
}

void CPU::HPL_dscal_omp(const int N, const double ALPHA, double* X, const int INCX, const int NB, const int II, const int thread_rank, const int thread_size) 
{
  int tile = 0;
  if(tile % thread_size == thread_rank) {
    const int nn = Mmin(NB - II, N);
    HPL_dscal(nn, ALPHA, X, INCX);
  }
  ++tile;
  int i = NB - II;
  for(; i < N; i += NB) {
    if(tile % thread_size == thread_rank) {
      const int nn = Mmin(NB, N - i);
      HPL_dscal(nn, ALPHA, X + i * INCX, INCX);
    }
    ++tile;
  }
}

void CPU::HPL_daxpy_omp(const int N, const double ALPHA, const double* X, const int INCX, double* Y, const int INCY, const int NB, const int II, const int thread_rank, const int thread_size) 
{
  int tile = 0;
  if(tile % thread_size == thread_rank) {
    const int nn = Mmin(NB - II, N);
    HPL_daxpy(nn, ALPHA, X, INCX, Y, INCY);
  }
  ++tile;
  int i = NB - II;
  for(; i < N; i += NB) {
    if(tile % thread_size == thread_rank) {
      const int nn = Mmin(NB, N - i);
      HPL_daxpy(nn, ALPHA, X + i * INCX, INCX, Y + i * INCY, INCY);
    }
    ++tile;
  }
}

void CPU::HPL_dger_omp(const enum HPL_ORDER ORDER, const int M, const int N,
                  const double ALPHA, const double* X, const int INCX, double* Y, const int INCY, 
                  double* A, const int LDA,
                  const int NB, const int II, const int thread_rank, const int thread_size) {

  int tile = 0;
  if(tile % thread_size == thread_rank) {
    const int mm = Mmin(NB - II, M);
    HPL_dger(ORDER, mm, N, ALPHA, X, INCX, Y, INCY, A, LDA);
  }
  ++tile;
  int i = NB - II;
  for(; i < M; i += NB) {
    if(tile % thread_size == thread_rank) {
      const int mm = Mmin(NB, M - i);
      HPL_dger(ORDER, mm, N, ALPHA, X + i * INCX, INCX, Y, INCY, A + i, LDA);
    }
    ++tile;
  }
}


void CPU::HPL_idamax_omp(const int N, const double* X, const int INCX, const int NB, const int II, const int thread_rank, const int thread_size, int* max_index, double* max_value) 
{
  max_index[thread_rank] = 0;
  max_value[thread_rank] = 0.0;

  if(N < 1) return;

  int tile = 0;
  if(tile % thread_size == thread_rank) {
    const int nn           = Mmin(NB - II, N);
    max_index[thread_rank] = HPL_idamax(nn, X, INCX);
    max_value[thread_rank] = X[max_index[thread_rank] * INCX];
  }
  ++tile;
  int i = NB - II;
  for(; i < N; i += NB) {
    if(tile % thread_size == thread_rank) {
      const int nn  = Mmin(NB, N - i);
      const int idm = HPL_idamax(nn, X + i * INCX, INCX);
      if(abs(X[(idm + i) * INCX]) > abs(max_value[thread_rank])) {
        max_value[thread_rank] = X[(idm + i) * INCX];
        max_index[thread_rank] = idm + i;
      }
    }
    ++tile;
  }

#pragma omp barrier

  // finish reduction
  if(thread_rank == 0) {
    for(int rank = 1; rank < thread_size; ++rank) {
      if(abs(max_value[rank]) > abs(max_value[0])) {
        max_value[0] = max_value[rank];
        max_index[0] = max_index[rank];
      }
    }
  }
}
MPI_Op HPL_DMXSWP;
MPI_Datatype PDFACT_ROW;

/* Swap-broadcast comparison function usable in MPI_Allreduce */
void HPL_dmxswp(void* invec, void* inoutvec, int* len,
                MPI_Datatype* datatype) {

  assert(*datatype == PDFACT_ROW);
  assert(*len == 1);

  int N;
  MPI_Type_size(PDFACT_ROW, &N);

  double* Wwork = static_cast<double*>(invec);
  double* WORK  = static_cast<double*>(inoutvec);

  const int jb = ((N/sizeof(double))-4)/2;

  //check max column value and overwirte row if new max is found
  const double gmax = Mabs(WORK[0]);
  const double tmp1 = Mabs(Wwork[0]);
  if((tmp1 > gmax) || ((tmp1 == gmax) && (Wwork[3] < WORK[3]))) {
    HPL_dcopy(jb+4, Wwork, 1, WORK, 1);
  }

  // Add the input top row to the inout top row.
  HPL_daxpy(jb, 1.0, Wwork+jb+4, 1, WORK+jb+4, 1);

}

const int max_req = 128;
MPI_Request reqs[max_req];
int req_idx = 0;

void CPU::HPL_all_reduce_dmxswp(double* BUFFER, const int COUNT, const int ROOT, MPI_Comm COMM, double* WORK) 
{

#if 0
  MPI_Op_create(HPL_dmxswp, true, &HPL_DMXSWP);
  MPI_Request req;
   (void) MPI_Iallreduce(MPI_IN_PLACE, BUFFER, 1, PDFACT_ROW, HPL_DMXSWP, COMM, &req);
  MPI_Wait(&req, MPI_STATUS_IGNORE);

#else
  double       gmax, tmp1;
  double *     A0, *Wmx;
  unsigned int hdim, ip2, ip2_, ipow, k, mask;
  int Np2, cnt_, cnt0, i, icurrow, mydist, mydis_, myrow, n0, nprow,
      partner, rcnt, root, scnt, size_;

  MPI_Comm_rank(COMM, &myrow);
  MPI_Comm_size(COMM, &nprow);

  hdim = 0;
  ip2  = 1;
  k    = nprow;
  while(k > 1) {
    k >>= 1;
    ip2 <<= 1;
    hdim++;
  }

  n0      = (COUNT-4)/2;
  icurrow = ROOT;
  Np2     = (int)((size_ = nprow - ip2) != 0);
  mydist  = MModSub(myrow, icurrow, nprow);

  cnt0 = (cnt_ = n0 + 4) + n0;
  A0    = (Wmx = BUFFER + 4) + n0;

  if((Np2 != 0) && ((partner = (int)((unsigned int)(mydist) ^ ip2)) < nprow)) {
    if((mydist & ip2) != 0) {
      if(mydist == (int)(ip2)) {
        int mpartner = MModAdd(partner, icurrow, nprow);
        MPI_Sendrecv(BUFFER, cnt_, MPI_DOUBLE, mpartner, MSGID_BEGIN_PFACT, A0, n0, MPI_DOUBLE, mpartner, MSGID_BEGIN_PFACT, COMM, MPI_STATUS_IGNORE);
      }
      else {
        MPI_Isend(BUFFER, cnt_, MPI_DOUBLE, MModAdd(partner, icurrow, nprow), MSGID_BEGIN_PFACT, COMM, &reqs[req_idx++]);
      }
    } else {
      if(mydist == 0) {
        int mpartner = MModAdd(partner, icurrow, nprow);
        MPI_Sendrecv(A0, n0, MPI_DOUBLE, mpartner, MSGID_BEGIN_PFACT, WORK, cnt_, MPI_DOUBLE, mpartner, MSGID_BEGIN_PFACT, COMM, MPI_STATUS_IGNORE);
      }
      else {
        MPI_Irecv(WORK, cnt_, MPI_DOUBLE, MModAdd(partner, icurrow, nprow), MSGID_BEGIN_PFACT, COMM, &reqs[req_idx++]);
      }

      tmp1 = Mabs(WORK[0]);
      gmax = Mabs(BUFFER[0]);
      if((tmp1 > gmax) || ((tmp1 == gmax) && (WORK[3] < BUFFER[3]))) {
        HPL_dcopy(cnt_, WORK, 1, BUFFER, 1);
      }
    }
  }

  if(mydist < (int)(ip2)) {
    k    = 0;
    ipow = 1;

    while(k < hdim) {
      if(((unsigned int)(mydist) >> (k + 1)) == 0) {
        if(((unsigned int)(mydist) >> k) == 0) {
          scnt = cnt0;
          rcnt = cnt_;
        } else {
          scnt = cnt_;
          rcnt = cnt0;
        }
      } else {
        scnt = rcnt = cnt_;
      }

      partner = (int)((unsigned int)(mydist) ^ ipow);
      int mpartner = MModAdd(partner, icurrow, nprow);
      MPI_Sendrecv(BUFFER, scnt, MPI_DOUBLE, mpartner, MSGID_BEGIN_PFACT, WORK, rcnt, MPI_DOUBLE, mpartner, MSGID_BEGIN_PFACT, COMM, MPI_STATUS_IGNORE);

      tmp1 = Mabs(WORK[0]);
      gmax = Mabs(BUFFER[0]);
      if((tmp1 > gmax) || ((tmp1 == gmax) && (WORK[3] < BUFFER[3]))) {
        HPL_dcopy((rcnt == cnt0 ? cnt0 : cnt_), WORK, 1, BUFFER, 1);
      } else if(rcnt == cnt0) {
        HPL_dcopy(n0, WORK + cnt_, 1, A0, 1);
      }

      ipow <<= 1;
      k++;
    }
  } else if(size_ > 1) {
    k    = (unsigned int)(size_)-1;
    ip2_ = mask = 1;
    while(k > 1) {
      k >>= 1;
      ip2_ <<= 1;
      mask <<= 1;
      mask++;
    }

    root   = MModAdd(icurrow, (int)(ip2), nprow);
    mydis_ = MModSub(myrow, root, nprow);

    do {
      mask ^= ip2_;
      if((mydis_ & mask) == 0) {
        partner = (int)(mydis_ ^ ip2_);
        if((mydis_ & ip2_) != 0) {
          MPI_Irecv(A0, n0, MPI_DOUBLE, MModAdd(root, partner, nprow), MSGID_BEGIN_PFACT, COMM, &reqs[req_idx++]);
        } else if(partner < size_) {
          MPI_Isend(A0, n0, MPI_DOUBLE, MModAdd(root, partner, nprow), MSGID_BEGIN_PFACT, COMM, &reqs[req_idx++]);
        }
      }
      ip2_ >>= 1;
    } while(ip2_ > 0);
  }
  if((Np2 != 0) && ((partner = (int)((unsigned int)(mydist) ^ ip2)) < nprow)) {
    if((mydist & ip2) != 0) {
      MPI_Irecv(BUFFER, cnt_, MPI_DOUBLE, MModAdd(partner, icurrow, nprow), MSGID_BEGIN_PFACT, COMM, &reqs[req_idx++]);
    } else {
      MPI_Isend(BUFFER, cnt_, MPI_DOUBLE, MModAdd(partner, icurrow, nprow), MSGID_BEGIN_PFACT, COMM, &reqs[req_idx++]);
    }
  }

  MPI_Waitall(req_idx, reqs, MPI_STATUSES_IGNORE);
  req_idx = 0;
#endif
}

void CPU::HPL_set_zero(const int N, double* __restrict__ X) {
  for( int tmp1=0; tmp1 < N; tmp1++ ) X[tmp1] = HPL_rzero; 
}