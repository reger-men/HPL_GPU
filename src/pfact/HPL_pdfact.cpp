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
#ifdef PDFACT_OMP
#include <omp.h>
double max_value[128];
int    max_index[128];
#endif 
#ifdef STDC_HEADERS
void HPL_pdfact
(
   HPL_T_panel *                    PANEL
)
#else
void HPL_pdfact
( PANEL )
   HPL_T_panel *                    PANEL;
#endif
{
/* 
 * Purpose
 * =======
 *
 * HPL_pdfact recursively factorizes a  1-dimensional  panel of columns.
 * The  RPFACT  function pointer specifies the recursive algorithm to be
 * used, either Crout, Left- or Right looking.  NBMIN allows to vary the
 * recursive stopping criterium in terms of the number of columns in the
 * panel, and  NDIV allows to specify the number of subpanels each panel
 * should be divided into. Usuallly a value of 2 will be chosen. Finally
 * PFACT is a function pointer specifying the non-recursive algorithm to
 * to be used on at most NBMIN columns. One can also choose here between
 * Crout, Left- or Right looking.  Empirical tests seem to indicate that
 * values of 4 or 8 for NBMIN give the best results.
 *  
 * Bi-directional  exchange  is  used  to  perform  the  swap::broadcast
 * operations  at once  for one column in the panel.  This  results in a
 * lower number of slightly larger  messages than usual.  On P processes
 * and assuming bi-directional links,  the running time of this function
 * can be approximated by (when N is equal to N0):                      
 *  
 *    N0 * log_2( P ) * ( lat + ( 2*N0 + 4 ) / bdwth ) +
 *    N0^2 * ( M - N0/3 ) * gam2-3
 *  
 * where M is the local number of rows of  the panel, lat and bdwth  are
 * the latency and bandwidth of the network for  double  precision  real
 * words, and  gam2-3  is  an estimate of the  Level 2 and Level 3  BLAS
 * rate of execution. The  recursive  algorithm  allows indeed to almost
 * achieve  Level 3 BLAS  performance  in the panel factorization.  On a
 * large  number of modern machines,  this  operation is however latency
 * bound,  meaning  that its cost can  be estimated  by only the latency
 * portion N0 * log_2(P) * lat.  Mono-directional links will double this
 * communication cost.
 *
 * Arguments
 * =========
 *
 * PANEL   (local input/output)          HPL_T_panel *
 *         On entry,  PANEL  points to the data structure containing the
 *         panel information.
 *
 * ---------------------------------------------------------------------
 */ 
/*
 * .. Local Variables ..
 */
   void                       * vptr = NULL;
   int                        align, jb;
/* ..
 * .. Executable Statements ..
 */
   jb = PANEL->jb; PANEL->n -= jb; PANEL->ja += jb;

   if( ( PANEL->grid->mycol != PANEL->pcol ) || ( jb <= 0 ) ) return;
#ifdef HPL_DETAILED_TIMING
   HPL_ptimer( HPL_TIMING_RPFACT );
#endif
   align = PANEL->algo->align;

/*
 * Factor the panel - Update the panel pointers, the buffer is allocated when initializing panel
 */

#ifdef ROCM
#ifdef PDFACT_OMP
#pragma omp parallel shared(max_value, max_index)
  {
    const int thread_rank = omp_get_thread_num();
    const int thread_size = omp_get_num_threads();
    assert(thread_size <= 128);

   PANEL->algo->rffun( PANEL, PANEL->mp, jb, 0, PANEL->fWORK );
  } 
#else 
   PANEL->algo->rffun( PANEL, PANEL->mp, jb, 0, PANEL->fWORK );
#endif
#else
   //Adil
   HPL_BE_malloc((void**)&vptr, ( (size_t)(align) + (size_t)(((4+((unsigned int)(jb) << 1)) << 1) )) * sizeof(double), T_TEMPO);
   /*vptr  = (void *)malloc( ( (size_t)(align) + (size_t)(((4+((unsigned int)(jb) << 1)) << 1) )) * sizeof(double) );*/
   if( vptr == NULL )
   { HPL_pabort( __LINE__, "HPL_pdfact", "Memory allocation failed" ); }
/*
 * Factor the panel - Update the panel pointers
 */
   PANEL->algo->rffun( PANEL, PANEL->mp, jb, 0, (double *)HPL_PTR( vptr,
                       ((size_t)(align) * sizeof(double) ) ) );

   //Adil
   if( vptr ) HPL_BE_free((void**)&vptr, T_TEMPO);
   //if( vptr ) free( vptr );
#endif

#ifdef ROCM
   PANEL->dA   = Mptr( PANEL->dA, 0, jb, PANEL->dlda );
#else
   PANEL->A   = Mptr( PANEL->A, 0, jb, PANEL->lda );
#endif
   PANEL->nq -= jb; PANEL->jj += jb;
#ifdef HPL_DETAILED_TIMING
   HPL_ptimer( HPL_TIMING_RPFACT );
#endif
/*
 * End of HPL_pdfact
 */
}
