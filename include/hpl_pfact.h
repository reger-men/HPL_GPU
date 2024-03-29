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
 */ 
#ifndef HPL_PFACT_H
#define HPL_PFACT_H
/*
 * ---------------------------------------------------------------------
 * Include files
 * ---------------------------------------------------------------------
 */
#include "hpl_misc.h"
#include "hpl_blas.h"
#include "hpl_gesv.h"

#include "hpl_pmisc.h"
#include "hpl_pauxil.h"
#include "hpl_panel.h"

extern double max_value[128];
extern int    max_index[128];
/*
 * ---------------------------------------------------------------------
 * #typedefs and data structures
 * ---------------------------------------------------------------------
 */
typedef void (*HPL_T_PFA_FUN)
(  HPL_T_panel *,   const int,       const int,       const int,
   double * );
typedef void (*HPL_T_RFA_FUN)
(  HPL_T_panel *,   const int,       const int,       const int,
   double * );

#ifndef ROCM
typedef void (*HPL_T_UPD_FUN)
(  HPL_T_panel *,   int *,           HPL_T_panel *,   const int ); 
#endif
/*
 * ---------------------------------------------------------------------
 * Function prototypes
 * ---------------------------------------------------------------------
 */
void                             HPL_dlocmax
STDC_ARGS( (
   HPL_T_panel *,
   const int,
   const int,
   const int,
   double *
) );

void                             HPL_dlocswpN
STDC_ARGS( (
   HPL_T_panel *,
   const int,
   const int,
   double *
) );
void                             HPL_dlocswpT
STDC_ARGS( (
   HPL_T_panel *,
   const int,
   const int,
   double *
) );
void                             HPL_pdmxswp
STDC_ARGS( (
   HPL_T_panel *,
   const int,
   const int,
   const int,
   double *
) );

void                             HPL_pdpancrN
STDC_ARGS( (
   HPL_T_panel *,
   const int,
   const int,
   const int,
   double *
) );
void                             HPL_pdpancrT
STDC_ARGS( (
   HPL_T_panel *,
   const int,
   const int,
   const int,
   double *
) );
void                             HPL_pdpanllN
STDC_ARGS( (
   HPL_T_panel *,
   const int,
   const int,
   const int,
   double *
) );
void                             HPL_pdpanllT
STDC_ARGS( (
   HPL_T_panel *,
   const int,
   const int,
   const int,
   double *
) );
void                             HPL_pdpanrlN
STDC_ARGS( (
   HPL_T_panel *,
   const int,
   const int,
   const int,
   double *
) );
void                             HPL_pdpanrlT
STDC_ARGS( (
   HPL_T_panel *,
   const int,
   const int,
   const int,
   double *
) );

void                             HPL_pdrpancrN
STDC_ARGS( (
   HPL_T_panel *,
   const int,
   const int,
   const int,
   double *
) );
void                             HPL_pdrpancrT
STDC_ARGS( (
   HPL_T_panel *,
   const int,
   const int,
   const int,
   double *
) );
void                             HPL_pdrpanllN
STDC_ARGS( (
   HPL_T_panel *,
   const int,
   const int,
   const int,
   double *
) );
void                             HPL_pdrpanllT
STDC_ARGS( (
   HPL_T_panel *,
   const int,
   const int,
   const int,
   double *
) );
void                             HPL_pdrpanrlN
STDC_ARGS( (
   HPL_T_panel *,
   const int,
   const int,
   const int,
   double *
) );
void                             HPL_pdrpanrlT
STDC_ARGS( (
   HPL_T_panel *,
   const int,
   const int,
   const int,
   double *
) );

void                             HPL_pdfact
STDC_ARGS( (
   HPL_T_panel *
) );
 
#endif
/*
 * End of hpl_pfact.h
 */
