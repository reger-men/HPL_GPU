#  
#  -- High Performance Computing Linpack Benchmark (HPL)                
#     HPL - 2.3 - December 2, 2018                          
#     Antoine P. Petitet                                                
#     University of Tennessee, Knoxville                                
#     Innovative Computing Laboratory                                 
#     (C) Copyright 2000-2008 All Rights Reserved                       
#                                                                       
#  -- Copyright notice and Licensing terms:                             
#                                                                       
#  Redistribution  and  use in  source and binary forms, with or without
#  modification, are  permitted provided  that the following  conditions
#  are met:                                                             
#                                                                       
#  1. Redistributions  of  source  code  must retain the above copyright
#  notice, this list of conditions and the following disclaimer.        
#                                                                       
#  2. Redistributions in binary form must reproduce  the above copyright
#  notice, this list of conditions,  and the following disclaimer in the
#  documentation and/or other materials provided with the distribution. 
#                                                                       
#  3. All  advertising  materials  mentioning  features  or  use of this
#  software must display the following acknowledgement:                 
#  This  product  includes  software  developed  at  the  University  of
#  Tennessee, Knoxville, Innovative Computing Laboratory.             
#                                                                       
#  4. The name of the  University,  the name of the  Laboratory,  or the
#  names  of  its  contributors  may  not  be used to endorse or promote
#  products  derived   from   this  software  without  specific  written
#  permission.                                                          
#                                                                       
#  -- Disclaimer:                                                       
#                                                                       
#  THIS  SOFTWARE  IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
#  ``AS IS'' AND ANY EXPRESS OR IMPLIED WARRANTIES,  INCLUDING,  BUT NOT
#  LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
#  A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE UNIVERSITY
#  OR  CONTRIBUTORS  BE  LIABLE FOR ANY  DIRECT,  INDIRECT,  INCIDENTAL,
#  SPECIAL,  EXEMPLARY,  OR  CONSEQUENTIAL DAMAGES  (INCLUDING,  BUT NOT
#  LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
#  DATA OR PROFITS; OR BUSINESS INTERRUPTION)  HOWEVER CAUSED AND ON ANY
#  THEORY OF LIABILITY, WHETHER IN CONTRACT,  STRICT LIABILITY,  OR TORT
#  (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
#  OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE. 
# ######################################################################
#  
include Make.inc
#
# ######################################################################
#
INCdep           = \
   $(INCdir)/hpl_misc.h   $(INCdir)/hpl_blas.h   $(INCdir)/hpl_auxil.h \
   $(INCdir)/hpl_pmisc.h  $(INCdir)/hpl_grid.h   $(INCdir)/hpl_comm.h  \
   $(INCdir)/hpl_pauxil.h $(INCdir)/hpl_panel.h  $(INCdir)/hpl_pfact.h \
   $(INCdir)/hpl_pgesv.h
#
## Object files ########################################################
#
HPL_pgeobj       = \
   HPL_pipid.o            HPL_plindx0.o          HPL_pdlaswp00N.o       \
   HPL_pdlaswp00T.o       HPL_perm.o             HPL_logsort.o          \
   HPL_plindx10.o         HPL_plindx1.o          HPL_spreadN.o          \
   HPL_spreadT.o          HPL_rollN.o            HPL_rollT.o            \
   HPL_equil.o            HPL_pdlaswp01N.o       HPL_pdlaswp01T.o       \
   HPL_pdupdateNN.o       HPL_pdupdateNT.o       HPL_pdupdateTN.o       \
   HPL_pdupdateTT.o       HPL_pdtrsv.o           HPL_pdgesv0.o          \
   HPL_pdgesvK1.o         HPL_pdgesvK2.o         HPL_pdgesv.o
#
## Targets #############################################################
#
all     : lib 
#
lib     : lib.grd
#
lib.grd : $(HPL_pgeobj)
	$(ARCHIVER) $(ARFLAGS) $(HPLlib) $(HPL_pgeobj)
	$(RANLIB) $(HPLlib)
	$(TOUCH) lib.grd
#
# ######################################################################
#
HPL_pipid.o            : ../HPL_pipid.cpp            $(INCdep)
	$(CC) -x c -o $@ -c $(CCFLAGS)  ../HPL_pipid.cpp
HPL_plindx0.o          : ../HPL_plindx0.cpp          $(INCdep)
	$(CC) -x c -o $@ -c $(CCFLAGS)  ../HPL_plindx0.cpp
HPL_pdlaswp00N.o       : ../HPL_pdlaswp00N.cpp       $(INCdep)
	$(CC) -x c -o $@ -c $(CCFLAGS)  ../HPL_pdlaswp00N.cpp
HPL_pdlaswp00T.o       : ../HPL_pdlaswp00T.cpp       $(INCdep)
	$(CC) -x c -o $@ -c $(CCFLAGS)  ../HPL_pdlaswp00T.cpp
HPL_perm.o             : ../HPL_perm.cpp             $(INCdep)
	$(CC) -x c -o $@ -c $(CCFLAGS)  ../HPL_perm.cpp
HPL_logsort.o          : ../HPL_logsort.cpp          $(INCdep)
	$(CC) -x c -o $@ -c $(CCFLAGS)  ../HPL_logsort.cpp
HPL_plindx10.o         : ../HPL_plindx10.cpp         $(INCdep)
	$(CC) -x c -o $@ -c $(CCFLAGS)  ../HPL_plindx10.cpp
HPL_plindx1.o          : ../HPL_plindx1.cpp          $(INCdep)
	$(CC) -x c -o $@ -c $(CCFLAGS)  ../HPL_plindx1.cpp
HPL_spreadN.o          : ../HPL_spreadN.cpp          $(INCdep)
	$(CC) -x c -o $@ -c $(CCFLAGS)  ../HPL_spreadN.cpp
HPL_spreadT.o          : ../HPL_spreadT.cpp          $(INCdep)
	$(CC) -x c -o $@ -c $(CCFLAGS)  ../HPL_spreadT.cpp
HPL_rollN.o            : ../HPL_rollN.cpp            $(INCdep)
	$(CC) -x c -o $@ -c $(CCFLAGS)  ../HPL_rollN.cpp
HPL_rollT.o            : ../HPL_rollT.cpp            $(INCdep)
	$(CC) -x c -o $@ -c $(CCFLAGS)  ../HPL_rollT.cpp
HPL_equil.o            : ../HPL_equil.cpp            $(INCdep)
	$(CC) -x c -o $@ -c $(CCFLAGS)  ../HPL_equil.cpp
HPL_pdlaswp01N.o       : ../HPL_pdlaswp01N.cpp       $(INCdep)
	$(CC) -x c -o $@ -c $(CCFLAGS)  ../HPL_pdlaswp01N.cpp
HPL_pdlaswp01T.o       : ../HPL_pdlaswp01T.cpp       $(INCdep)
	$(CC) -x c -o $@ -c $(CCFLAGS)  ../HPL_pdlaswp01T.cpp
HPL_pdupdateNN.o       : ../HPL_pdupdateNN.cpp       $(INCdep)
	$(CC) -x c -o $@ -c $(CCFLAGS)  ../HPL_pdupdateNN.cpp
HPL_pdupdateNT.o       : ../HPL_pdupdateNT.cpp       $(INCdep)
	$(CC) -x c -o $@ -c $(CCFLAGS)  ../HPL_pdupdateNT.cpp
HPL_pdupdateTN.o       : ../HPL_pdupdateTN.cpp       $(INCdep)
	$(CC) -x c -o $@ -c $(CCFLAGS)  ../HPL_pdupdateTN.cpp
HPL_pdupdateTT.o       : ../HPL_pdupdateTT.cpp       $(INCdep)
	$(CC) -x c -o $@ -c $(CCFLAGS)  ../HPL_pdupdateTT.cpp
HPL_pdtrsv.o           : ../HPL_pdtrsv.cpp           $(INCdep)
	$(CC) -x c -o $@ -c $(CCFLAGS)  ../HPL_pdtrsv.cpp
HPL_pdgesv0.o          : ../HPL_pdgesv0.cpp          $(INCdep)
	$(CC) -x c -o $@ -c $(CCFLAGS)  ../HPL_pdgesv0.cpp
HPL_pdgesvK1.o         : ../HPL_pdgesvK1.cpp         $(INCdep)
	$(CC) -x c -o $@ -c $(CCFLAGS)  ../HPL_pdgesvK1.cpp
HPL_pdgesvK2.o         : ../HPL_pdgesvK2.cpp         $(INCdep)
	$(CC) -x c -o $@ -c $(CCFLAGS)  ../HPL_pdgesvK2.cpp
HPL_pdgesv.o           : ../HPL_pdgesv.cpp           $(INCdep)
	$(CC) -x c -o $@ -c $(CCFLAGS)  ../HPL_pdgesv.cpp
#
# ######################################################################
#
clean            :
	$(RM) *.o *.grd
#
# ######################################################################
