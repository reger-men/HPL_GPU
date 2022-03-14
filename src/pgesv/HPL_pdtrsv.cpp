/* 
 * -- High Performance Computing Linpack Benchmark (HPL)                
 *    HPL - 2.3 - December 2, 2018                          
 *    Antoine P. Petitet                                                
 *    University of Tennessee, Knoxville                                
 *    Innovative Computing Laboratory                                 
 *    (C) Copyright 2000-2008 All Rights Reserved                       
 *                                                                      
 * -- Copyright notice and Licensing terms:                             
 *                                                                      
 * Redistribution  and  use in  source and binary forms, with or without
 * modification, are  permitted provided  that the following  conditions
 * are met:                                                             
 *                                                                      
 * 1. Redistributions  of  source  code  must retain the above copyright
 * notice, this list of conditions and the following disclaimer.        
 *                                                                      
 * 2. Redistributions in binary form must reproduce  the above copyright
 * notice, this list of conditions,  and the following disclaimer in the
 * documentation and/or other materials provided with the distribution. 
 *                                                                      
 * 3. All  advertising  materials  mentioning  features  or  use of this
 * software must display the following acknowledgement:                 
 * This  product  includes  software  developed  at  the  University  of
 * Tennessee, Knoxville, Innovative Computing Laboratory.             
 *                                                                      
 * 4. The name of the  University,  the name of the  Laboratory,  or the
 * names  of  its  contributors  may  not  be used to endorse or promote
 * products  derived   from   this  software  without  specific  written
 * permission.                                                          
 *                                                                      
 * -- Disclaimer:                                                       
 *                                                                      
 * THIS  SOFTWARE  IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
 * ``AS IS'' AND ANY EXPRESS OR IMPLIED WARRANTIES,  INCLUDING,  BUT NOT
 * LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
 * A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE UNIVERSITY
 * OR  CONTRIBUTORS  BE  LIABLE FOR ANY  DIRECT,  INDIRECT,  INCIDENTAL,
 * SPECIAL,  EXEMPLARY,  OR  CONSEQUENTIAL DAMAGES  (INCLUDING,  BUT NOT
 * LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
 * DATA OR PROFITS; OR BUSINESS INTERRUPTION)  HOWEVER CAUSED AND ON ANY
 * THEORY OF LIABILITY, WHETHER IN CONTRACT,  STRICT LIABILITY,  OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE. 
 * ---------------------------------------------------------------------
 */ 
/*
 * Include files
 */
#include "hpl.h"

#ifdef STDC_HEADERS
void HPL_pdtrsv
(
   HPL_T_grid *                     GRID,
   HPL_T_pmat *                     AMAT
)
#else
void HPL_pdtrsv
( GRID, AMAT )
   HPL_T_grid *                     GRID;
   HPL_T_pmat *                     AMAT;
#endif
{
/* 
 * Purpose
 * =======
 *
 * HPL_pdtrsv solves an upper triangular system of linear equations.
 *  
 * The rhs is the last column of the N by N+1 matrix A. The solve starts
 * in the process  column owning the  Nth  column of A, so the rhs b may
 * need to be moved one process column to the left at the beginning. The
 * routine therefore needs  a column  vector in every process column but
 * the one owning  b. The result is  replicated in all process rows, and
 * returned in XR, i.e. XR is of size nq = LOCq( N ) in all processes.
 *  
 * The algorithm uses decreasing one-ring broadcast in process rows  and
 * columns  implemented  in terms of  synchronous communication point to
 * point primitives.  The  lookahead of depth 1 is used to minimize  the
 * critical path. This entire operation is essentially ``latency'' bound
 * and an estimate of its running time is given by:
 *  
 *    (move rhs) lat + N / ( P bdwth ) +            
 *    (solve)    ((N / NB)-1) 2 (lat + NB / bdwth) +
 *               gam2 N^2 / ( P Q ),                
 *  
 * where  gam2   is an estimate of the   Level 2 BLAS rate of execution.
 * There are  N / NB  diagonal blocks. One must exchange  2  messages of
 * length NB to compute the next  NB  entries of the vector solution, as
 * well as performing a total of N^2 floating point operations.
 *
 * Arguments
 * =========
 *
 * GRID    (local input)                 HPL_T_grid *
 *         On entry,  GRID  points  to the data structure containing the
 *         process grid information.
 *
 * AMAT    (local input/output)          HPL_T_pmat *
 *         On entry,  AMAT  points  to the data structure containing the
 *         local array information.
 *
 * ---------------------------------------------------------------------
 */ 
/*
 * .. Local Variables ..
 */
   MPI_Comm                   Ccomm, Rcomm;
   double                     * A=NULL, * Aprev=NULL, * Aptr, * XC=NULL,
                              * XR=NULL, * Xd=NULL, * Xdprev=NULL,
                              * W=NULL;
#ifdef ROCM
   double                     * dA=NULL, * dAprev=NULL, * dAptr, * dXC=NULL,
                              * dXR=NULL, * dXd=NULL, * dXdprev=NULL,
                              * dW=NULL;
#endif
   int                        Alcol, Alrow, Anpprev, Anp, Anq, Bcol,
                              Cmsgid, GridIsNotPx1, GridIsNot1xQ, Rmsgid,
                              Wfr=0, colprev, kb, kbprev, lda, mycol,
                              myrow, n, n1, n1p, n1pprev=0, nb, npcol,
                              nprow, rowprev, tmp1, tmp2;
/* ..
 * .. Executable Statements ..
 */
#ifdef HPL_DETAILED_TIMING
   HPL_ptimer( HPL_TIMING_PTRSV );
#endif
   if( ( n = AMAT->n ) <= 0 ) return;
   nb = AMAT->nb; lda = AMAT->ld; A = AMAT->A; XR = AMAT->X;

#ifdef ROCM
   // rocblas_set_stream(handle, 0);
   rocblas_handle handle;
   rocblas_create_handle(&handle);
   rocblas_set_pointer_mode(handle, rocblas_pointer_mode_host);
   dA = AMAT->d_A; dXR = AMAT->d_X;
   // hipMemcpy(AMAT->A, AMAT->dA, (AMAT->n+1)*AMAT->ld*sizeof(double), hipMemcpyDeviceToHost);
#endif
   (void) HPL_grid_info( GRID, &nprow, &npcol, &myrow, &mycol );
   Rcomm = GRID->row_comm; Rmsgid = MSGID_BEGIN_PTRSV;
   Ccomm = GRID->col_comm; Cmsgid = MSGID_BEGIN_PTRSV + 1;
   GridIsNot1xQ = ( nprow > 1 ); GridIsNotPx1 = ( npcol > 1 );
/*
 * Move the rhs in the process column owning the last column of A.
 */
   Mnumroc( Anp, n, nb, nb, myrow, 0, nprow );
   Mnumroc( Anq, n, nb, nb, mycol, 0, npcol );

   tmp1  = ( n - 1 ) / nb;
   Alrow = tmp1 - ( tmp1 / nprow ) * nprow;
   Alcol = tmp1 - ( tmp1 / npcol ) * npcol;
   kb    = n    - tmp1 * nb;

   Aptr = (double *)(A); XC = Mptr( Aptr, 0, Anq, lda );
#ifdef ROCM
   dAptr = (double *)(dA); dXC = Mptr( dAptr, 0, Anq, lda );
#endif
   Mindxg2p( n, nb, nb, Bcol, 0, npcol );

   if( ( Anp > 0 ) && ( Alcol != Bcol ) )
   {
      if( mycol == Bcol  )
      { 
#ifdef ROCM
         hipMemcpy(XC, dXC, Anp*sizeof(double), hipMemcpyDeviceToHost);
#endif
         (void) HPL_send( XC, Anp, Alcol, Rmsgid, Rcomm ); 
      }
      else if( mycol == Alcol )
      { 
         (void) HPL_recv( XC, Anp, Bcol,  Rmsgid, Rcomm ); 
#ifdef ROCM
         hipMemcpy(dXC, XC, Anp*sizeof(double), hipMemcpyHostToDevice);
#endif
      }
   }
   Rmsgid = ( Rmsgid + 2 >
              MSGID_END_PTRSV ? MSGID_BEGIN_PTRSV : Rmsgid + 2 );
   if( mycol != Alcol )
   {
#ifdef ROCM
      hipMemset(dXC, 0, Anp*sizeof(double));
#endif
      for( tmp1=0; tmp1 < Anp; tmp1++ ) XC[tmp1] = HPL_rzero; 
   }
/*
 * Set up lookahead
 */
   n1 = ( npcol - 1 ) * nb; n1 = Mmax( n1, nb );
   if( Anp > 0 )
   {
#ifdef ROCM
      hipHostMalloc((void**)&W, (size_t)(Mmin( n1, Anp )) * sizeof( double ), 0);
      hipMalloc((void**)&dW, (size_t)(Mmin( n1, Anp )) * sizeof( double ));
      if( W == NULL || dW == NULL)
#else
      //Adil
      HPL_BE_malloc((void**)&W, (size_t)(Mmin( n1, Anp )) * sizeof( double ), T_TEMPO);
      /*W = (double*)malloc( (size_t)(Mmin( n1, Anp )) * sizeof( double ) );*/
      if( W == NULL )
#endif
      { HPL_pabort( __LINE__, "HPL_pdtrsv", "Memory allocation failed" ); }
      Wfr = 1;
   }

   Anpprev = Anp; Xdprev = XR; Aprev = Aptr = Mptr( Aptr, 0, Anq, lda );
#ifdef ROCM
   dXdprev = dXR; dAprev = dAptr = Mptr( dAptr, 0, Anq, lda );
#endif
   tmp1    = n - kb; tmp1 -= ( tmp2 = Mmin( tmp1, n1 ) );
   MnumrocI( n1pprev, tmp2, Mmax( 0, tmp1 ), nb, nb, myrow, 0, nprow );

   if( myrow == Alrow ) { Anpprev = ( Anp -= kb ); }
   if( mycol == Alcol )
   {
      Aprev = ( Aptr -= lda * kb ); Anq -= kb; Xdprev = ( Xd = XR + Anq );
#ifdef ROCM
      dAprev = ( dAptr -= lda * kb ); dXdprev = ( dXd = dXR + Anq );
#endif
      if( myrow == Alrow )
      {
#ifdef ROCM
         const double one = 1.0;
         rocblas_dtrsm(handle, rocblas_side_left, rocblas_fill_upper,
                      rocblas_operation_none, rocblas_diagonal_non_unit,
                       kb, 1, &one, dAptr+Anp, lda, dXC+Anp, kb);
         rocblas_dcopy(handle, kb, dXC+Anp, 1, dXd, 1 );
#else
         //Adil
         HPL_BE_dtrsv(HplColumnMajor, HplUpper, HplNoTrans, HplNonUnit,
                    kb, Aptr+Anp, lda, XC+Anp, 1, T_TEMPO);
         /*HPL_dtrsv( HplColumnMajor, HplUpper, HplNoTrans, HplNonUnit,
                    kb, Aptr+Anp, lda, XC+Anp, 1 );*/

         //Adil
         HPL_BE_dcopy(kb, XC+Anp, 1, Xd, 1, T_TEMPO);                    
         /*HPL_dcopy( kb, XC+Anp, 1, Xd, 1 );*/
#endif
      }
   }

   rowprev = Alrow; Alrow = MModSub1( Alrow, nprow );
   colprev = Alcol; Alcol = MModSub1( Alcol, npcol );
   kbprev  = kb; n -= kb;
   tmp1    = n - ( kb = nb ); tmp1 -= ( tmp2 = Mmin( tmp1, n1 ) );
   MnumrocI( n1p, tmp2, Mmax( 0, tmp1 ), nb, nb, myrow, 0, nprow );
/*
 * Start the operations
 */
   while( n > 0 )
   {
      if( mycol == Alcol ) { 
         Aptr -= lda * kb; Anq -= kb; Xd = XR + Anq; 
#ifdef ROCM
         dAptr -= lda * kb; dXd = dXR + Anq;
#endif
      }
      if( myrow == Alrow ) { Anp -= kb; }
/*
 * Broadcast  (decreasing-ring)  of  previous solution block in previous
 * process column,  compute  partial update of current block and send it
 * to current process column.
 */
      if( mycol == colprev )
      {
/*
 * Send previous solution block in process row above
 */
         if( myrow == rowprev )
         {
            if( GridIsNot1xQ ) {
#ifdef ROCM
               hipMemcpy(Xdprev, dXdprev, kbprev*sizeof(double), hipMemcpyDeviceToHost);
#endif
               (void) HPL_send( Xdprev, kbprev, MModSub1( myrow, nprow ),
                                Cmsgid, Ccomm );
            }
         }
         else
         {
            (void) HPL_recv( Xdprev, kbprev, MModAdd1( myrow, nprow ),
                             Cmsgid, Ccomm );
#ifdef ROCM
            hipMemcpy(dXdprev, Xdprev, kbprev*sizeof(double), hipMemcpyHostToDevice);
#endif
         } 
/*
 * Compute partial update of previous solution block and send it to cur-
 * rent column
 */
         if( n1pprev > 0 )
         {
            tmp1 = Anpprev - n1pprev;
#ifdef ROCM
            const double one =  1.0;
            const double mone = -1.0;
            rocblas_dgemv(handle, rocblas_operation_none, n1pprev, kbprev,
                         &mone, dAprev+tmp1, lda, dXdprev, 1, &one,
                         dXC+tmp1, 1 );
#else
            //Adil
            HPL_BE_dgemv( HplColumnMajor, HplNoTrans, n1pprev, kbprev,
                       -HPL_rone, Aprev+tmp1, lda, Xdprev, 1, HPL_rone,
                       XC+tmp1, 1, T_DEFAULT);
            /*HPL_dgemv( HplColumnMajor, HplNoTrans, n1pprev, kbprev,
                       -HPL_rone, Aprev+tmp1, lda, Xdprev, 1, HPL_rone,
                       XC+tmp1, 1 );*/
#endif
            if( GridIsNotPx1 ) {
#ifdef ROCM
               hipMemcpy(XC+tmp1, dXC+tmp1, n1pprev*sizeof(double), hipMemcpyDeviceToHost);
#endif
               (void) HPL_send( XC+tmp1, n1pprev, Alcol, Rmsgid, Rcomm );
            }
         }
/*
 * Finish  the (decreasing-ring) broadcast of the solution block in pre-
 * vious process column
 */
         if( ( myrow != rowprev ) &&
             ( myrow != MModAdd1( rowprev, nprow ) ) ) {
#ifdef ROCM
            hipMemcpy(Xdprev, dXdprev, kbprev*sizeof(double), hipMemcpyDeviceToHost);
#endif
            (void) HPL_send( Xdprev, kbprev, MModSub1( myrow, nprow ),
                             Cmsgid, Ccomm );
             }
      }
      else if( mycol == Alcol )
      {
/*
 * Current  column  receives  and accumulates partial update of previous
 * solution block
 */
         if( n1pprev > 0 )
         {
            (void) HPL_recv( W, n1pprev, colprev, Rmsgid, Rcomm );
#ifdef ROCM
            hipMemcpy(dW, W, n1pprev*sizeof(double), hipMemcpyHostToDevice);
            const double one = 1.0;
            rocblas_daxpy(handle, n1pprev, &one, dW, 1, dXC+Anpprev-n1pprev, 1);
#else
            //Adil
            HPL_BE_daxpy(n1pprev, HPL_rone, W, 1, XC+Anpprev-n1pprev, 1, T_DEFAULT);
            /*HPL_daxpy( n1pprev, HPL_rone, W, 1, XC+Anpprev-n1pprev, 1 );*/
#endif
         }
      }
/*
 * Solve current diagonal block 
 */
      if( ( mycol == Alcol ) && ( myrow == Alrow ) )
      {
#ifdef ROCM
         const double one = 1.0;
         rocblas_dtrsm(handle, rocblas_side_left, rocblas_fill_upper,
                      rocblas_operation_none, rocblas_diagonal_non_unit,
                       kb, 1, &one, dAptr+Anp, lda, dXC+Anp, kb);
         rocblas_dcopy(handle, kb, dXC+Anp, 1, dXR+Anq, 1 );
#else
         //Adil
         HPL_BE_dtrsv(HplColumnMajor, HplUpper, HplNoTrans, HplNonUnit,
                    kb, Aptr+Anp, lda, XC+Anp, 1, T_DEFAULT);
         /*HPL_dtrsv( HplColumnMajor, HplUpper, HplNoTrans, HplNonUnit,
                    kb, Aptr+Anp, lda, XC+Anp, 1 );*/

         //Adil
         HPL_BE_dcopy(kb, XC+Anp, 1, XR+Anq, 1, T_DEFAULT);                    
         /*HPL_dcopy( kb, XC+Anp, 1, XR+Anq, 1 );*/
#endif
      }
/*
*  Finish previous update
*/
      if( ( mycol == colprev ) && ( ( tmp1 = Anpprev - n1pprev ) > 0 ) ) {
#ifdef ROCM
         const double one =  1.0;
         const double mone = -1.0;
         rocblas_dgemv(handle, rocblas_operation_none, tmp1, kbprev,
                      &mone, dAprev, lda, dXdprev, 1, &one,
                      dXC, 1 );
#else
         //Adil
         HPL_BE_dgemv( HplColumnMajor, HplNoTrans, tmp1, kbprev, -HPL_rone,
                    Aprev, lda, Xdprev, 1, HPL_rone, XC, 1, T_DEFAULT);
         /*HPL_dgemv( HplColumnMajor, HplNoTrans, tmp1, kbprev, -HPL_rone,
                    Aprev, lda, Xdprev, 1, HPL_rone, XC, 1 );*/
#endif
      }
/*
*  Save info of current step and update info for the next step
*/
      if( mycol == Alcol ) { 
         Xdprev   = Xd; Aprev = Aptr; 
#ifdef ROCM
         dXdprev   = dXd; dAprev = dAptr;
#endif
      }
      if( myrow == Alrow ) { Anpprev -= kb; }
      rowprev = Alrow; colprev = Alcol;
      n1pprev = n1p;   kbprev  = kb; n -= kb;
      Alrow = MModSub1( Alrow, nprow ); Alcol = MModSub1( Alcol, npcol );
      tmp1  = n - ( kb = nb ); tmp1 -= ( tmp2 = Mmin( tmp1, n1 ) );
      MnumrocI( n1p, tmp2, Mmax( 0, tmp1 ), nb, nb, myrow, 0, nprow );

      Rmsgid = ( Rmsgid+2 > MSGID_END_PTRSV ? 
                 MSGID_BEGIN_PTRSV   : Rmsgid+2 );
      Cmsgid = ( Cmsgid+2 > MSGID_END_PTRSV ?
                 MSGID_BEGIN_PTRSV+1 : Cmsgid+2 );
   }
/*
 * Replicate last solution block
 */
   if( mycol == colprev ) {
#ifdef ROCM
      hipMemcpy(XR, dXR, kbprev*sizeof(double), hipMemcpyDeviceToHost);
#endif
      (void) HPL_broadcast( (void *)(XR), kbprev, HPL_DOUBLE, rowprev,
                            Ccomm );
#ifdef ROCM
      hipMemcpy(dXR, XR, kbprev*sizeof(double), hipMemcpyHostToDevice);
#endif
   }

#ifdef ROCM
   hipDeviceSynchronize();
   if( Wfr  ) {
      hipHostFree( W  );
      hipFree( dW  );
   }
   // hipMemcpy(AMAT->dX, AMAT->X, (AMAT->n)*sizeof(double), hipMemcpyHostToDevice);
#else
   //Adil
   if( Wfr ) HPL_BE_free((void**)&W, T_TEMPO);
   /*if( Wfr  ) free( W  );*/
#endif
#ifdef HPL_DETAILED_TIMING
   HPL_ptimer( HPL_TIMING_PTRSV );
#endif
/*
 * End of HPL_pdtrsv
 */
}
