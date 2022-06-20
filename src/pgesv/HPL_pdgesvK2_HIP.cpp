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
void HPL_pdgesvK2_HIP
(
   HPL_T_grid *                     GRID,
   HPL_T_palg *                     ALGO,
   HPL_T_pmat *                     A
)
#else
void HPL_pdgesvK2_HIP
( GRID, ALGO, A )
   HPL_T_grid *                     GRID;
   HPL_T_palg *                     ALGO;
   HPL_T_pmat *                     A;
#endif
{
/* 
 * Purpose
 * =======
 *
 * HPL_pdgesvK2 factors a N+1-by-N matrix using LU factorization with row
 * partial pivoting.  The main algorithm  is the "right looking" variant
 * with look-ahead.  The  lower  triangular factor is left unpivoted and
 * the pivots are not returned. The right hand side is the N+1 column of
 * the coefficient matrix.
 *
 * Arguments
 * =========
 *
 * GRID    (local input)                 HPL_T_grid *
 *         On entry,  GRID  points  to the data structure containing the
 *         process grid information.
 *
 * ALGO    (global input)                HPL_T_palg *
 *         On entry,  ALGO  points to  the data structure containing the
 *         algorithmic parameters.
 *
 * A       (local input/output)          HPL_T_pmat *
 *         On entry, A points to the data structure containing the local
 *         array information.
 *
 * ---------------------------------------------------------------------
 */ 
/*
 * .. Local Variables ..
 */
   HPL_T_panel                * p, * * panel = NULL;
   HPL_T_UPD_FUN              HPL_pdupdate; 
   int                        N, depth, icurcol=0, j, jb, jj=0, jstart,
                              k, mycol, n, nb, nn, npcol, nq,
                              tag=MSGID_BEGIN_FACT, test=HPL_KEEP_TESTING;
#ifdef HPL_PROGRESS_REPORT
   double start_time, time, gflops;
#endif
/* ..
 * .. Executable Statements ..
 */
   mycol = GRID->mycol; npcol        = GRID->npcol;
   depth = ALGO->depth; HPL_pdupdate = ALGO->upfun;
   N     = A->n;        nb           = A->nb;

   if( N <= 0 ) return;

#ifdef HPL_PROGRESS_REPORT
   start_time = HPL_timer_walltime();
#endif

/*
 * Allocate a panel list of length depth + 1 (depth >= 1)
 */
   //Adil
   panel = (HPL_T_panel**)malloc((size_t)(depth + 1) * sizeof(HPL_T_panel*));
   if( panel == NULL )
   { HPL_pabort( __LINE__, "HPL_pdgesvK2", "Memory allocation failed" ); }
/*
 * Create and initialize the first depth panels
 */
   nq = HPL_numroc( N+1, nb, nb, mycol, 0, npcol ); nn = N; jstart = 0;

   for( k = 0; k < depth; k++ )
   {
      jb = Mmin( nn, nb );
      HPL_BE_panel_new( GRID, ALGO, nn, nn+1, jb, A, jstart, jstart,
                       tag, &panel[k], T_HIP);
      nn -= jb; jstart += jb;
      if( mycol == icurcol ) { jj += jb; nq -= jb; }
      icurcol = MModAdd1( icurcol, npcol );
      tag     = MNxtMgid( tag, MSGID_BEGIN_FACT, MSGID_END_FACT );
   }
/*
 * Create last depth+1 panel
 */
   HPL_BE_panel_new( GRID, ALGO, nn, nn+1, Mmin( nn, nb ), A, jstart,
                    jstart, tag, &panel[depth], T_HIP);
   tag = MNxtMgid( tag, MSGID_BEGIN_FACT, MSGID_END_FACT );
/*
 * Initialize the lookahead - Factor jstart columns: panel[0..depth-1]
 */
   for( k = 0, j = 0; k < depth; k++ )
   {
      jb = jstart - j; jb = Mmin( jb, nb ); j += jb;
/*
 * Factor and broadcast k-th panel
 */
      HPL_BE_panel_send_to_host( panel[k], T_HIP );
      HPL_BE_stream_synchronize(HPL_DATASTREAM, T_HIP);

      HPL_pdfact(         panel[k] );

      HPL_BE_panel_send_to_device( panel[k], T_HIP );
      HPL_BE_stream_synchronize(HPL_DATASTREAM, T_HIP);

      HPL_pdpanel_bcast(panel[0]); 

      // pre-perform the row swapping
      HIP::HPL_pdlaswp_hip(panel[0], icurcol, {SU2, SU0, SU1, CU2, CU0});
/*
 * Partial update of the depth-k-1 panels in front of me
 */
   }
/*
 * Main loop over the remaining columns of A
 */
  float  smallDgemmTime, largeDgemm1Time, largeDgemm2Time;
  double smallDgemmGflops, pdfactGflops, largeDgemm1Gflops, largeDgemm2Gflops;
  double stepStart, stepEnd, pdfactStart, pdfactEnd;
   for( j = jstart; j < N; j += nb )
   {
      stepStart = MPI_Wtime();
      n = N - j; jb = Mmin( n, nb );
/*
 * Initialize current panel - Finish latest update, Factor and broadcast
 * current panel
 */
      (void)HPL_BE_panel_free(panel[depth], T_HIP);
      HPL_BE_panel_init(GRID, ALGO, n, n+1, jb, A, j, j, tag, panel[depth], T_HIP);
      if( mycol == icurcol )
      {
         // finish the row swapping of the look-ahead part
         HIP::HPL_pdlaswp_hip(panel[0], icurcol, {EU0});
         HPL_pdupdate(HPL_LOOK_AHEAD, NULL, NULL, panel[0], panel[0]->nu0);

         // when the look ahead update is finished, copy back the current panel
         HPL_BE_stream_wait_event(HPL_DATASTREAM, UPDATE_LOOK_AHEAD, T_HIP);
         HPL_BE_panel_send_to_host(panel[1], T_HIP);

         // finish the row swapping of second part
         HIP::HPL_pdlaswp_hip(panel[0], icurcol, {EU2});
         HPL_pdupdate(HPL_UPD_2, NULL, NULL, panel[0], panel[0]->nu2);

         //wait for the panel to arrive
         HPL_BE_stream_synchronize(HPL_DATASTREAM, T_HIP);

#ifdef ROCM
#ifdef HPL_PROGRESS_REPORT
      const int curr  = (panel[0]->grid->myrow == panel[0]->prow ? 1 : 0);
      const int mp    = panel[0]->mp - (curr != 0 ? jb : 0);

      //compute the GFLOPs of the look ahead update DGEMM
      smallDgemmTime = HIP::elapsedTime(HPL_LOOK_AHEAD);
      smallDgemmGflops = (2.0 * mp * jb * jb) / (1000.0 * 1000.0 * smallDgemmTime);
#endif
#endif

         pdfactStart = MPI_Wtime();
         HPL_pdfact(panel[depth]); /* factor current panel */
         pdfactEnd = MPI_Wtime();

#ifdef HPL_PROGRESS_REPORT
      pdfactGflops =
          (((double) panel[0]->mp)*jb*jb - (1.0/3.0)*jb*jb*jb - 0.5*jb*jb) / ((1000.0 * 1000.0 * 1000.0)*(pdfactEnd - pdfactStart));
#endif

         // send the panel back to device before bcast
         HPL_BE_panel_send_to_device(panel[1], T_HIP);
         HPL_BE_stream_synchronize(HPL_DATASTREAM, T_HIP);
      }
      else { 
         nn = 0;
         // finish the row swapping of second part
         HIP::HPL_pdlaswp_hip(panel[0], icurcol, {EU2});
         HPL_pdupdate(HPL_UPD_2, NULL, NULL, panel[0], panel[0]->nu2);
      }
     
    HPL_pdpanel_bcast(panel[1]);

    // start local row swapping for second part
    HIP::HPL_pdlaswp_hip(panel[1], icurcol, {SU2});

    // swap and finish the row swapping for the first part
    HIP::HPL_pdlaswp_hip(panel[0], icurcol, {CU1, EU1});

    // update for the first part
    HPL_pdupdate(HPL_UPD_1, NULL, NULL, panel[0], panel[0]->nu1);

    if(mycol == icurcol) {
      jj += jb;
      nq -= jb;
    }
    icurcol = MModAdd1(icurcol, npcol);
    tag     = MNxtMgid(tag, MSGID_BEGIN_FACT, MSGID_END_FACT);

    HIP::HPL_pdlaswp_hip(panel[1], icurcol, {SU0, SU1, CU2, CU0});

    //wait here for the updates to compete
    HPL_BE_device_sync(T_HIP);
    stepEnd = MPI_Wtime();

/*
 * Circular  of the panel pointers:
 * xtmp = x[0]; for( k=0; k < depth; k++ ) x[k] = x[k+1]; x[d] = xtmp;
 *
 * Go to next process row and column - update the message ids for broadcast
 */
#ifdef HPL_PROGRESS_REPORT
#ifdef ROCM
    const int curr  = (panel[0]->grid->myrow == panel[0]->prow ? 1 : 0);
    const int mp    = panel[0]->mp - (curr != 0 ? jb : 0);

    largeDgemm1Time = 0.0;
    largeDgemm2Time = 0.0;
    if (panel[0]->nu1) {
      largeDgemm1Time = HIP::elapsedTime(HPL_UPD_1);
      largeDgemm1Gflops = (2.0 * mp * jb * (panel[0]->nu1)) / (1000.0 * 1000.0 * (largeDgemm1Time));
    }
    if (panel[0]->nu2) {
      largeDgemm2Time = HIP::elapsedTime(HPL_UPD_2);
      largeDgemm2Gflops = (2.0 * mp * jb * (panel[0]->nu2)) / (1000.0 * 1000.0 * (largeDgemm2Time));
    }

    /* if this is process 0,0 and not the first panel */
    if(GRID->myrow == 0 && mycol == 0 && j > 0) {
      time   = HPL_ptimer_walltime() - start_time;
      gflops = 2.0 * (N * (double)N * N - n * (double)n * n) / 3.0 /
               (time > 0.0 ? time : 1.e-6) / 1.e9;
      printf("Column=%09d (%4.1f%%) ", j, j * 100.0 / N);
      printf("Step Time(s)=%9.7f ", stepEnd-stepStart);

      if (panel[0]->nu0) {
        printf("Small DGEMM Gflops=%9.3e ", smallDgemmGflops);
        printf("pdfact Gflops=%9.3e ", pdfactGflops);
      } else {
        printf("Small DGEMM Gflops=--------- ");
        printf("pdfact Gflops=--------- ");
      }
      if (panel[0]->nu2) {
        printf("DGEMM1 Gflops=%9.3e ", largeDgemm2Gflops);
      } else {
        printf("DGEMM1 Gflops=--------- ");
      }

      if (panel[0]->nu1) {
        printf("DGEMM2 Gflops=%9.3e ", largeDgemm1Gflops);
      } else {
        printf("DGEMM2 Gflops=--------- ");
      }

      printf("Overall Gflops=%9.3e\n", gflops);
    }
#endif
#endif
     p = panel[0];
     panel[0] = panel[1];
     panel[1] = p;
   }
/*
 * Clean-up: Finish updates - release panels and panel list
 */
   nn = HPL_numrocI( 1, N, nb, nb, mycol, 0, npcol );
   for( k = 0; k < depth; k++ )
   {
      HIP::HPL_pdlaswp_hip(panel[k], HPL_LOOK_AHEAD, SWP_END);
      HPL_pdupdate(HPL_LOOK_AHEAD, NULL, NULL, panel[k], nn);
      HIP::HPL_pdlaswp_hip(panel[k], HPL_UPD_2, SWP_END);
      HPL_pdupdate(HPL_UPD_2, NULL, NULL, panel[k], nn);

      HPL_BE_device_sync(T_HIP);
      HPL_BE_panel_disp(  &panel[k], T_HIP);
   }
   HPL_BE_panel_disp( &panel[depth], T_HIP);
   if( panel ) free(panel);
/*
 * End of HPL_pdgesvK2
 */
}
