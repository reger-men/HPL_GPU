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
   $(INCdir)/hpl_misc.h  $(INCdir)/hpl_blas.h   $(INCdir)/hpl_auxil.h \
   $(INCdir)/hpl_pmisc.h $(INCdir)/hpl_pauxil.h $(INCdir)/hpl_pfact.h
#
## Object files ########################################################
#
HPL_pfaobj       = \
   HPL_dlocmax.o          HPL_dlocswpN.o         HPL_dlocswpT.o         \
   HPL_pdmxswp.o          HPL_pdpancrN.o         HPL_pdpancrT.o         \
   HPL_pdpanllN.o         HPL_pdpanllT.o         HPL_pdpanrlN.o         \
   HPL_pdpanrlT.o         HPL_pdrpanllN.o        HPL_pdrpanllT.o        \
   HPL_pdrpancrN.o        HPL_pdrpancrT.o        HPL_pdrpanrlN.o        \
   HPL_pdrpanrlT.o        HPL_pdfact.o
#
## Targets #############################################################
#
all              : lib 
#
lib              : lib.grd
#
lib.grd          : $(HPL_pfaobj)
	$(ARCHIVER) $(ARFLAGS) $(HPLlib) $(HPL_pfaobj)
	$(RANLIB) $(HPLlib)
	$(TOUCH) lib.grd
#
# ######################################################################
#
HPL_dlocmax.o          : ../HPL_dlocmax.cpp          $(INCdep)
	$(CC) -x c -o $@ -c $(CCFLAGS)  ../HPL_dlocmax.cpp
HPL_dlocswpN.o         : ../HPL_dlocswpN.cpp         $(INCdep)
	$(CC) -x c -o $@ -c $(CCFLAGS)  ../HPL_dlocswpN.cpp
HPL_dlocswpT.o         : ../HPL_dlocswpT.cpp         $(INCdep)
	$(CC) -x c -o $@ -c $(CCFLAGS)  ../HPL_dlocswpT.cpp
HPL_pdmxswp.o          : ../HPL_pdmxswp.cpp          $(INCdep)
	$(CC) -x c -o $@ -c $(CCFLAGS)  ../HPL_pdmxswp.cpp
HPL_pdpancrN.o         : ../HPL_pdpancrN.cpp         $(INCdep)
	$(CC) -x c -o $@ -c $(CCFLAGS)  ../HPL_pdpancrN.cpp
HPL_pdpancrT.o         : ../HPL_pdpancrT.cpp         $(INCdep)
	$(CC) -x c -o $@ -c $(CCFLAGS)  ../HPL_pdpancrT.cpp
HPL_pdpanllN.o         : ../HPL_pdpanllN.cpp         $(INCdep)
	$(CC) -x c -o $@ -c $(CCFLAGS)  ../HPL_pdpanllN.cpp
HPL_pdpanllT.o         : ../HPL_pdpanllT.cpp         $(INCdep)
	$(CC) -x c -o $@ -c $(CCFLAGS)  ../HPL_pdpanllT.cpp
HPL_pdpanrlN.o         : ../HPL_pdpanrlN.cpp         $(INCdep)
	$(CC) -x c -o $@ -c $(CCFLAGS)  ../HPL_pdpanrlN.cpp
HPL_pdpanrlT.o         : ../HPL_pdpanrlT.cpp         $(INCdep)
	$(CC) -x c -o $@ -c $(CCFLAGS)  ../HPL_pdpanrlT.cpp
HPL_pdrpanllN.o        : ../HPL_pdrpanllN.cpp        $(INCdep)
	$(CC) -x c -o $@ -c $(CCFLAGS)  ../HPL_pdrpanllN.cpp
HPL_pdrpanllT.o        : ../HPL_pdrpanllT.cpp        $(INCdep)
	$(CC) -x c -o $@ -c $(CCFLAGS)  ../HPL_pdrpanllT.cpp
HPL_pdrpancrN.o        : ../HPL_pdrpancrN.cpp        $(INCdep)
	$(CC) -x c -o $@ -c $(CCFLAGS)  ../HPL_pdrpancrN.cpp
HPL_pdrpancrT.o        : ../HPL_pdrpancrT.cpp        $(INCdep)
	$(CC) -x c -o $@ -c $(CCFLAGS)  ../HPL_pdrpancrT.cpp
HPL_pdrpanrlN.o        : ../HPL_pdrpanrlN.cpp        $(INCdep)
	$(CC) -x c -o $@ -c $(CCFLAGS)  ../HPL_pdrpanrlN.cpp
HPL_pdrpanrlT.o        : ../HPL_pdrpanrlT.cpp        $(INCdep)
	$(CC) -x c -o $@ -c $(CCFLAGS)  ../HPL_pdrpanrlT.cpp
HPL_pdfact.o           : ../HPL_pdfact.cpp           $(INCdep)
	$(CC) -x c -o $@ -c $(CCFLAGS)  ../HPL_pdfact.cpp
#
# ######################################################################
#
clean            :
	$(RM) *.o *.grd
#
# ######################################################################
