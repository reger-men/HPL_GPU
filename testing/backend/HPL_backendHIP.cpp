
#include <hpl.h>



void HIP::init(const HPL_T_grid* GRID)
{
    int rank, size, count, namelen; 
    size_t bytes;
    char (*host_names)[MPI_MAX_PROCESSOR_NAME];
    int nprow, npcol, myrow, mycol;


    (void)HPL_grid_info(GRID, &nprow, &npcol, &myrow, &mycol);

    MPI_Comm_rank( MPI_COMM_WORLD, &rank );
    MPI_Comm_size( MPI_COMM_WORLD, &size );
    MPI_Get_processor_name(host_name,&namelen);

    bytes = size * sizeof(char[MPI_MAX_PROCESSOR_NAME]);
    host_names = (char (*)[MPI_MAX_PROCESSOR_NAME])std::malloc(bytes);


    strcpy(host_names[rank], host_name);

    for (int n=0; n < size; n++){
        MPI_Bcast(&(host_names[n]),MPI_MAX_PROCESSOR_NAME, MPI_CHAR, n, MPI_COMM_WORLD);
    }

    int localRank = GRID->local_mycol + GRID->local_myrow * GRID->local_npcol;
    int localSize = GRID->local_npcol * GRID->local_nprow;


    hipDeviceProp_t hipDeviceProp;
    HIP_CHECK_ERROR(hipGetDeviceCount(&count));
    //TODO: set dynamic device id
    int device_id = localRank % count; 
    HIP_CHECK_ERROR(hipSetDevice(device_id));

    // Get device properties
    HIP_CHECK_ERROR(hipGetDeviceProperties(&hipDeviceProp, device_id));
    printf ("Assigning device %d on node %s to rank %d \n", localRank % count,  host_name, rank);


    GPUInfo("%-25s %-12s \t%-5s", "[Device]", "Using HIP Device",  hipDeviceProp.name, "With Properties:");
    GPUInfo("%-25s %-20lld", "[GlobalMem]", "Total Global Memory",  (unsigned long long int)hipDeviceProp.totalGlobalMem);
    GPUInfo("%-25s %-20lld", "[SharedMem]", "Shared Memory Per Block", (unsigned long long int)hipDeviceProp.sharedMemPerBlock);
    GPUInfo("%-25s %-20d", "[Regs]", "Registers Per Block", hipDeviceProp.regsPerBlock);
    GPUInfo("%-25s %-20d", "[WarpSize]", "WaveFront Size", hipDeviceProp.warpSize);
    GPUInfo("%-25s %-20d", "[MaxThreads]", "Max Threads Per Block", hipDeviceProp.maxThreadsPerBlock);
    GPUInfo("%-25s %-4d %-4d %-4d", "[MaxThreadsDim]", "Max Threads Dimension", hipDeviceProp.maxThreadsDim[0], hipDeviceProp.maxThreadsDim[1], hipDeviceProp.maxThreadsDim[2]);
    GPUInfo("%-25s %-4d %-4d %-4d", "[MaxGridSize]", "Max Grid Size", hipDeviceProp.maxGridSize[0], hipDeviceProp.maxGridSize[1], hipDeviceProp.maxGridSize[2]);
    GPUInfo("%-25s %-20lld", "[ConstMem]", "Total Constant Memory",  (unsigned long long int)hipDeviceProp.totalConstMem);
    GPUInfo("%-25s %-20d", "[Major]", "Major", hipDeviceProp.major);
    GPUInfo("%-25s %-20d", "[Minor]", "Minor", hipDeviceProp.minor);
    GPUInfo("%-25s %-20d", "[ClkRate]", "Clock Rate", hipDeviceProp.memoryClockRate);
    GPUInfo("%-25s %-20d", "[#CUs]", "Multi Processor Count", hipDeviceProp.multiProcessorCount);
    GPUInfo("%-25s %-20d", "[PCIBusID]", "PCI Bus ID", hipDeviceProp.pciBusID);
    GPUInfo("----------------------------------------", "----------------------------------------");


    //Init ROCBlas
    rocblas_initialize();
    ROCBLAS_CHECK_STATUS(rocblas_create_handle(&_handle));

    rocblas_set_pointer_mode(_handle, rocblas_pointer_mode_host);
    HIP_CHECK_ERROR(hipStreamCreate(&computeStream));
    HIP_CHECK_ERROR(hipStreamCreate(&dataStream));
    HIP_CHECK_ERROR(hipStreamCreate(&pdlaswpStream));
    
    ROCBLAS_CHECK_STATUS(rocblas_set_stream(_handle, computeStream));
    
    HIP_CHECK_ERROR(hipEventCreate(&panelUpdate));
    HIP_CHECK_ERROR(hipEventCreate(&panelCopy));
    HIP_CHECK_ERROR(hipEventCreate(&swapDataTransfer));
    HIP_CHECK_ERROR(hipEventCreate(&L1Transfer));
    HIP_CHECK_ERROR(hipEventCreate(&L2Transfer));

    HIP_CHECK_ERROR(hipEventCreate(&panelSendToHost));
    HIP_CHECK_ERROR(hipEventCreate(&panelSendToDevice));

    HIP_CHECK_ERROR(hipEventCreate(&pdlaswpStart_1));
    HIP_CHECK_ERROR(hipEventCreate(&pdlaswpFinish_1));
    HIP_CHECK_ERROR(hipEventCreate(&pdlaswpStart_2));
    HIP_CHECK_ERROR(hipEventCreate(&pdlaswpFinish_2));

    HIP_CHECK_ERROR(hipEventCreate(swapStartEvent + HPL_LOOK_AHEAD));
    HIP_CHECK_ERROR(hipEventCreate(swapStartEvent + HPL_UPD_1));
    HIP_CHECK_ERROR(hipEventCreate(swapStartEvent + HPL_UPD_2));

    HIP_CHECK_ERROR(hipEventCreate(swapUCopyEvent + HPL_LOOK_AHEAD));
    HIP_CHECK_ERROR(hipEventCreate(swapUCopyEvent + HPL_UPD_1));
    HIP_CHECK_ERROR(hipEventCreate(swapUCopyEvent + HPL_UPD_2));

    HIP_CHECK_ERROR(hipEventCreate(swapWCopyEvent + HPL_LOOK_AHEAD));
    HIP_CHECK_ERROR(hipEventCreate(swapWCopyEvent + HPL_UPD_1));
    HIP_CHECK_ERROR(hipEventCreate(swapWCopyEvent + HPL_UPD_2));

    HIP_CHECK_ERROR(hipEventCreate(update + HPL_LOOK_AHEAD));
    HIP_CHECK_ERROR(hipEventCreate(update + HPL_UPD_1));
    HIP_CHECK_ERROR(hipEventCreate(update + HPL_UPD_2));

    HIP_CHECK_ERROR(hipEventCreate(dgemmStart + HPL_LOOK_AHEAD));
    HIP_CHECK_ERROR(hipEventCreate(dgemmStart + HPL_UPD_1));
    HIP_CHECK_ERROR(hipEventCreate(dgemmStart + HPL_UPD_2));

    HIP_CHECK_ERROR(hipEventCreate(dgemmStop + HPL_LOOK_AHEAD));
    HIP_CHECK_ERROR(hipEventCreate(dgemmStop + HPL_UPD_1));
    HIP_CHECK_ERROR(hipEventCreate(dgemmStop + HPL_UPD_2));

    _memcpyKind[0] = "H2H";
    _memcpyKind[1] = "H2D";
    _memcpyKind[2] = "D2H";
    _memcpyKind[3] = "D2D";
    _memcpyKind[4] = "DEFAULT";
}

void HIP::release()
{
    ROCBLAS_CHECK_STATUS(rocblas_destroy_handle(_handle));
    HIP_CHECK_ERROR(hipStreamDestroy(computeStream));
    HIP_CHECK_ERROR(hipStreamDestroy(dataStream));
}

void HIP::malloc(void** ptr, size_t size)
{
    GPUInfo("%-25s %-12ld (B) \t%-5s", "[Allocate]", "Memory of size",  size, "HIP");
    HIP_CHECK_ERROR(hipMalloc(ptr, size));
}

void HIP::host_malloc(void** ptr, size_t size, unsigned int flag)
{
    GPUInfo("%-25s %-12ld (B) \t%-5s", "[Allocate]", "Host memory of size",  size, "HIP");
    HIP_CHECK_ERROR(hipHostMalloc(ptr, size, flag));
}

void HIP::free(void** ptr)
{
    HIP_CHECK_ERROR(hipFree(*ptr));
}

void HIP::host_free(void **ptr)
{
    HIP_CHECK_ERROR(hipHostFree(*ptr));
}

void HIP::panel_new(HPL_T_grid *GRID, HPL_T_palg *ALGO, const int M, const int N, const int JB, HPL_T_pmat *A,
               const int IA, const int JA, const int TAG, HPL_T_panel **PANEL)
{
    HPL_T_panel                * p = NULL;
    if( !( p = (HPL_T_panel *)std::malloc( sizeof( HPL_T_panel ) ) ) )
    {
        HPL_pabort( __LINE__, "HPL_pdpanel_new", "Memory allocation failed" );
    }

    p->max_work_size = 0;
    p->max_iwork_size = 0;
    p->free_work_now = 0;
    p->max_fwork_size = 0;
    p->A = NULL;
    p->WORK = NULL;
    p->IWORK = NULL;
    p->IWORK2 = NULL;
    p->fWORK = NULL;
    HIP::panel_init( GRID, ALGO, M, N, JB, A, IA, JA, TAG, p );
    *PANEL = p;
}




void HIP::panel_send_to_host(HPL_T_panel *PANEL)
{
    int jb = PANEL->jb;

    if( ( PANEL->grid->mycol != PANEL->pcol ) || ( jb <= 0 ) ) return;
    if(PANEL->mp > 0)
    HIP_CHECK_ERROR(hipMemcpy2DAsync(PANEL->A,  PANEL->lda*sizeof(double),
                    PANEL->dA, PANEL->dlda*sizeof(double),
                    PANEL->mp*sizeof(double), jb,
                    hipMemcpyDeviceToHost, dataStream));
    HIP_CHECK_ERROR(hipEventRecord(panelCopy, dataStream));
}

// Only for P=1
void HPL_unroll_ipiv(const int mp, const int jb, int* ipiv, int* ipiv_ex, int* upiv)
{
  for(int i = 0; i < mp; i++) { upiv[i] = i; } // initialize ids for the swapping
  for(int i = 0; i < jb; i++) {                // swap ids
    int id = upiv[i];
    upiv[i] = upiv[ipiv[i]];
    upiv[ipiv[i]] = id;
  }

  for(int i = 0; i < jb; i++) { ipiv_ex[i] = -1; }

  int cnt = 0;
  for(int i = jb; i < mp; i++) { // find swapped ids outside of panel
    if(upiv[i] < jb) { ipiv_ex[upiv[i]] = i; }
  }
}

void HIP::panel_send_to_device(HPL_T_panel *PANEL)
{
  double *A, *dA;
  int jb, i, ml2;

  jb = PANEL->jb;

  if (jb <= 0)
     return;

  // only the root column copies to device
  if (PANEL->grid->mycol == PANEL->pcol) {

    if(PANEL->grid->nprow == 1) {

      // unroll pivoting and send to device now
      int* ipiv    = PANEL->ipiv;
      int* ipiv_ex = PANEL->ipiv + jb;
      int* upiv    = PANEL->IWORK + jb; // scratch space

      for(i = 0; i < jb; i++) { ipiv[i] -= PANEL->ii; } // shift
      HPL_unroll_ipiv(PANEL->mp, jb, ipiv, ipiv_ex, upiv);

      int* dipiv    = PANEL->dipiv;
      int* dipiv_ex = PANEL->dipiv + jb;

      HIP_CHECK_ERROR(hipMemcpy2DAsync(dipiv, jb * sizeof(int),
                       upiv, jb * sizeof(int),
                       jb * sizeof(int), 1,
                       hipMemcpyHostToDevice, dataStream));
      HIP_CHECK_ERROR(hipMemcpy2DAsync(dipiv_ex, jb * sizeof(int),
                       ipiv_ex, jb * sizeof(int),
                       jb * sizeof(int), 1,
                       hipMemcpyHostToDevice, dataStream));
    } 
    else {
      int k;
      int *iflag, *ipl, *ipID, *ipA, *iplen, *ipmap, *ipmapm1, *upiv, *iwork, 
          *lindxU, *lindxA, *lindxAU, *permU, *permU_ex, *ipiv, 
          *dlindxU, *dlindxA, *dlindxAU, *dpermU, *dpermU_ex, *dipiv;

      k = (int)((unsigned int)(jb) << 1);
      iflag = PANEL->IWORK; ipl = iflag + 1; ipID = ipl + 1; ipA = ipID + ((unsigned int)(k) << 1); 
      iplen = ipA + 1; ipmap = iplen + PANEL->grid->nprow + 1; ipmapm1 = ipmap + PANEL->grid->nprow; 
      upiv = ipmapm1 + PANEL->grid->nprow; iwork = upiv + PANEL->mp;

      lindxU = PANEL->lindxU; lindxA = PANEL->lindxA; lindxAU = PANEL->lindxAU; 
      permU = PANEL->permU; permU_ex = permU + jb; ipiv = PANEL->ipiv;
      dlindxU = PANEL->dlindxU; dlindxA = PANEL->dlindxA; dlindxAU = PANEL->dlindxAU; 
      dpermU = PANEL->dpermU; dpermU_ex = dpermU + jb; dipiv = PANEL->dipiv;

      if(*iflag == -1) { /* no index arrays have been computed so far */
        HPL_pipid(PANEL, ipl, ipID);
        HPL_plindx(PANEL, *ipl, ipID, ipA, lindxU, lindxAU, lindxA, iplen, permU, iwork);
        *iflag = 1;
      }

      int N = Mmax(*ipA, jb);
      if(N > 0) {
        HIP_CHECK_ERROR(hipMemcpy2DAsync(dlindxA, k * sizeof(int), lindxA, k * sizeof(int), N * sizeof(int), 1, hipMemcpyHostToDevice, dataStream));
        HIP_CHECK_ERROR(hipMemcpy2DAsync(dlindxAU, k * sizeof(int), lindxAU, k * sizeof(int), N * sizeof(int), 1, hipMemcpyHostToDevice, dataStream));
      }

      HIP_CHECK_ERROR(hipMemcpyAsync(dlindxU, lindxU, jb * sizeof(int), hipMemcpyHostToDevice, dataStream));

      HIP_CHECK_ERROR(hipMemcpy2DAsync(dpermU, jb * sizeof(int), permU, jb * sizeof(int), jb * sizeof(int), 1, hipMemcpyHostToDevice, dataStream));
      HIP_CHECK_ERROR(hipMemcpy2DAsync(dipiv, jb * sizeof(int), ipiv, jb * sizeof(int), jb * sizeof(int), 1, hipMemcpyHostToDevice, dataStream));
    }
  }

  //record when the swap data will arrive
  HIP_CHECK_ERROR(hipEventRecord(swapDataTransfer, dataStream));

  // copy A and/or L2
  if(PANEL->grid->mycol == PANEL->pcol) {
    // copy L1
    HIP_CHECK_ERROR(hipMemcpy2DAsync(PANEL->dL1, jb * sizeof(double),
                     PANEL->L1, jb * sizeof(double),
                     jb * sizeof(double), jb,
                     hipMemcpyHostToDevice, dataStream));

    //record when L1 will arrive
    HIP_CHECK_ERROR(hipEventRecord(L1Transfer, dataStream));

    if(PANEL->grid->npcol > 1) { // L2 is its own array
      if(PANEL->grid->myrow == PANEL->prow) {
        HIP_CHECK_ERROR(hipMemcpy2DAsync(Mptr(PANEL->dA, 0, -jb, PANEL->dlda), PANEL->dlda * sizeof(double),
                         Mptr(PANEL->A, 0, 0, PANEL->lda), PANEL->lda * sizeof(double),
                         jb * sizeof(double), jb,
                         hipMemcpyHostToDevice, dataStream));

        if((PANEL->mp - jb) > 0)
          HIP_CHECK_ERROR(hipMemcpy2DAsync(PANEL->dL2, PANEL->dldl2 * sizeof(double),
                           Mptr(PANEL->A, jb, 0, PANEL->lda), PANEL->lda * sizeof(double),
                           (PANEL->mp - jb) * sizeof(double), jb,
                           hipMemcpyHostToDevice, dataStream));
      } 
      else {
        if((PANEL->mp) > 0)
          HIP_CHECK_ERROR(hipMemcpy2DAsync(PANEL->dL2, PANEL->dldl2 * sizeof(double),
                           Mptr(PANEL->A, 0, 0, PANEL->lda), PANEL->lda * sizeof(double),
                           PANEL->mp * sizeof(double), jb,
                           hipMemcpyHostToDevice, dataStream));
      }
    } 
    else {
      if(PANEL->mp > 0)
        HIP_CHECK_ERROR(hipMemcpy2DAsync(Mptr(PANEL->dA, 0, -jb, PANEL->dlda), PANEL->dlda * sizeof(double),
                         Mptr(PANEL->A, 0, 0, PANEL->lda), PANEL->lda * sizeof(double),
                         PANEL->mp * sizeof(double), jb,
                         hipMemcpyHostToDevice, dataStream));
    }
    //record when L2 will arrive
    HIP_CHECK_ERROR(hipEventRecord(L2Transfer, dataStream));
  }
}

int HIP::panel_free(HPL_T_panel *PANEL)
{
    GPUInfo("%-40s \t%-5s", "[Deallocate]", "Panel resources", "HIP");
    if(PANEL->pmat->info == 0) PANEL->pmat->info = *(PANEL->DINFO);
    if (PANEL->free_work_now == 1) 
    {
        if(PANEL->WORK) HIP_CHECK_ERROR(hipHostFree(PANEL->WORK));
        if(PANEL->dWORK) HIP_CHECK_ERROR(hipFree(PANEL->dWORK));
        PANEL->max_work_size = 0;
        if(PANEL->IWORK) std::free(PANEL->IWORK);
        if(PANEL->fWORK) HIP_CHECK_ERROR(hipHostFree(PANEL->fWORK));
        PANEL->max_iwork_size = 0;
        PANEL->max_fwork_size = 0;
  }

  return HPL_SUCCESS;
}

void HIP::gPrintMat(const int M, const int N, const int LDA, const double *A)
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

int HIP::pdmatgen(HPL_T_test* TEST, HPL_T_grid* GRID, HPL_T_palg* ALGO,  HPL_T_pmat* mat,const int N, const int NB)
{
  int ii, ip2, im4096;
  int mycol, myrow, npcol, nprow, nq, info[3];
  (void)HPL_grid_info(GRID, &nprow, &npcol, &myrow, &mycol);

  mat->n    = N; mat->nb   = NB; mat->info = 0; mat->dN = N;
  mat->mp   = HPL_numroc(N, NB, NB, myrow, 0, nprow);
  nq        = HPL_numroc(N, NB, NB, mycol, 0, npcol);
  /*
   * Allocate matrix, right-hand-side, and vector solution x. [ A | b ] is
   * N by N+1.  One column is added in every process column for the solve.
   * The  result  however  is stored in a 1 x N vector replicated in every
   * process row. In every process, A is lda * (nq+1), x is 1 * nq and the
   * workspace is mp.
   *
   * Ensure that lda is a multiple of ALIGN and not a power of 2, and not
   * a multiple of 4096 bytes
   */
  mat->ld = Mmax(1, mat->mp);
  mat->ld = ((mat->ld + 95) / 128) * 128 + 32; /*pad*/

  mat->nq = nq + 1;

  mat->d_A = nullptr;
  mat->d_X = nullptr;

  mat->dW = nullptr;
  mat->W  = nullptr;
  /*
   * Allocate dynamic memory
   */

  // allocate on device
  size_t numbytes = ((size_t)(mat->ld) * (size_t)(mat->nq)) * sizeof(double);

#ifdef HPL_VERBOSE_PRINT
  if((myrow == 0) && (mycol == 0)) {
    printf("dA: Allocating %g GBs of storage on GPU...",
           ((double)numbytes) / (1024 * 1024 * 1024));
    fflush(stdout);
  }
#endif 

  HIP_CHECK_ERROR(hipMalloc(&(mat->d_A), numbytes));

  /*Check matrix allocation is valid*/
  if (mat->d_A==NULL) {
    char host_name[MPI_MAX_PROCESSOR_NAME];
    int rank, namelen;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Get_processor_name(host_name, &namelen);

    printf("Matrix allocation on node %s, rank %d, failed. \n", host_name, rank);
  }
  info[0] = (mat->d_A == NULL); info[1] = myrow; info[2] = mycol;
  (void)HPL_all_reduce((void*)(info), 3, HPL_INT, HPL_max, GRID->all_comm);
  if(info[0] != 0) {
    HPL_pwarn(TEST->outfp, __LINE__, "HPL_pdmatgen", "[%d,%d] %s", info[1], info[2], "Device memory allocation failed for A and b. Skip.");
    return HPL_FAILURE;
  }
#ifdef HPL_VERBOSE_PRINT
  if((myrow == 0) && (mycol == 0)) printf("done.\n");
#endif
  // seperate space for X vector
  HIP_CHECK_ERROR(hipMalloc(&(mat->d_X), mat->nq * sizeof(double)));

  /*Check vector allocation is valid*/
  info[0] = (mat->d_X == NULL); info[1] = myrow; info[2] = mycol;
  (void)HPL_all_reduce((void*)(info), 3, HPL_INT, HPL_max, GRID->all_comm);
  if(info[0] != 0) {
    HPL_pwarn(TEST->outfp, __LINE__, "HPL_pdmatgen", "[%d,%d] %s", info[1], info[2], "Device memory allocation failed for x. Skip.");
    return HPL_FAILURE;
  }

  int Anp;
  Mnumroc(Anp, mat->n, mat->nb, mat->nb, myrow, 0, nprow);

  /*Need space for a column of panels for pdfact on CPU*/
  size_t A_hostsize = mat->ld * mat->nb * sizeof(double);

#ifdef HPL_VERBOSE_PRINT
  if((myrow == 0) && (mycol == 0)) {
    printf("A: Allocating %g GBs of storage on CPU...",
           ((double)A_hostsize) / (1024 * 1024 * 1024));
    fflush(stdout);
  }
#endif
  HIP_CHECK_ERROR(hipHostMalloc((void**)&(mat->A), A_hostsize));

  /*Check workspace allocation is valid*/
  info[0] = (mat->A == NULL); info[1] = myrow; info[2] = mycol;
  (void)HPL_all_reduce((void*)(info), 3, HPL_INT, HPL_max, GRID->all_comm);
  if(info[0] != 0) {
    HPL_pwarn(TEST->outfp, __LINE__, "HPL_pdmatgen", "[%d,%d] %s", info[1], info[2], "Host memory allocation failed for host A. Skip.");
    return HPL_FAILURE;
  }
#ifdef HPL_VERBOSE_PRINT
  if((myrow == 0) && (mycol == 0)) printf("done.\n");
#endif
  size_t dworkspace_size = 0;
  size_t workspace_size  = 0;

  /*pdtrsv needs two vectors for B and W (and X on host) */
  dworkspace_size = Mmax(2 * Anp * sizeof(double), dworkspace_size);
  workspace_size  = Mmax((2 * Anp + nq) * sizeof(double), workspace_size);

  /*Scratch space for rows in pdlaswp (with extra space for padding) */
  dworkspace_size = Mmax((nq + mat->nb + 256) * mat->nb * sizeof(double), dworkspace_size);
  workspace_size = Mmax((nq + mat->nb + 256) * mat->nb * sizeof(double), workspace_size);

#ifdef HPL_VERBOSE_PRINT
  if((myrow == 0) && (mycol == 0)) {
    printf("dW: Allocating %g GBs of storage on GPU...",
           ((double)dworkspace_size) / (1024 * 1024 * 1024));
    fflush(stdout);
  }
#endif
  HIP_CHECK_ERROR(hipMalloc((void**)&(mat->dW), dworkspace_size));

  /*Check workspace allocation is valid*/
  info[0] = (mat->dW == NULL);
  info[1] = myrow;
  info[2] = mycol;
  (void)HPL_all_reduce((void*)(info), 3, HPL_INT, HPL_max, GRID->all_comm);
  if(info[0] != 0) {
    HPL_pwarn(TEST->outfp, __LINE__, "HPL_pdmatgen", "[%d,%d] %s", info[1], info[2], "Device memory allocation failed for workspace. Skip.");
    return HPL_FAILURE;
  }
#ifdef HPL_VERBOSE_PRINT
  if((myrow == 0) && (mycol == 0)) printf("done.\n");

  if((myrow == 0) && (mycol == 0)) {
    printf("W:Allocating %g GBs of storage on CPU...",
           ((double)workspace_size) / (1024 * 1024 * 1024));
    fflush(stdout);
  }
#endif
  HIP_CHECK_ERROR(hipHostMalloc((void**)&(mat->W), workspace_size));

  /*Check workspace allocation is valid*/
  info[0] = (mat->W == NULL);
  info[1] = myrow;
  info[2] = mycol;
  (void)HPL_all_reduce((void*)(info), 3, HPL_INT, HPL_max, GRID->all_comm);
  if(info[0] != 0) {
    HPL_pwarn(TEST->outfp, __LINE__, "HPL_pdmatgen", "[%d,%d] %s", info[1], info[2], "Host memory allocation failed for workspace. Skip.");
    return HPL_FAILURE;
  }
#ifdef HPL_VERBOSE_PRINT
  if((myrow == 0) && (mycol == 0)) printf("done.\n");
#endif 
  return HPL_SUCCESS;
}

void HIP::pdmatfree(HPL_T_pmat* mat) {

  if(mat->d_A) {hipFree(mat->d_A); mat->d_A=nullptr;}
  if(mat->d_X) {hipFree(mat->d_X); mat->d_X=nullptr;}
  if(mat->dW) {hipFree(mat->dW); mat->dW=nullptr;}

  if(mat->A) {hipHostFree(mat->A); mat->A=nullptr;}
  if(mat->W) {hipHostFree(mat->W); mat->W=nullptr;}

}

int HIP::panel_disp(HPL_T_panel **PANEL)
{
    GPUInfo("%-40s \t%-5s", "[Deallocate]", "Panel structure", "HIP");
    (*PANEL)->pmat->n = (*PANEL)->pmat->dN;
    (*PANEL)->free_work_now = 1;
    int err = HIP::panel_free(*PANEL);
    if (*PANEL) free(*PANEL);
    *PANEL = NULL;
    return( err );
}


void HIP::matgen(const HPL_T_grid *GRID, const int M, const int N,
                 const int NB, double *A, const int LDA,
                 const int ISEED)
{
    GPUInfo("%-25s %-8d%-8d \t%-5s", "[Generate matrix]", "With A of (R:C)", M, N, "HIP");
    int mp, mycol, myrow, npcol, nprow, nq;
    (void) HPL_grid_info( GRID, &nprow, &npcol, &myrow, &mycol );
    
    Mnumroc( mp, M, NB, NB, myrow, 0, nprow );
    Mnumroc( nq, N, NB, NB, mycol, 0, npcol );

    if( ( mp <= 0 ) || ( nq <= 0 ) ) return;
    mp = (mp<LDA) ? LDA : mp;
    
    unsigned long long pos1 = myrow*nq + mycol*mp*M;
    rocrand_generator generator;
    rocrand_create_generator(&generator, ROCRAND_RNG_PSEUDO_DEFAULT);
    rocrand_set_seed(generator, ISEED);
    rocrand_set_offset(generator, pos1);

    rocrand_generate_uniform_double(generator,A, ((size_t)mp)*nq);
    HIP_CHECK_ERROR(hipDeviceSynchronize());

    rocrand_destroy_generator(generator);
}

void HIP::event_record(enum HPL_EVENT _event, const HPL_T_UPD UPD){
    switch (_event)
    {
    case HPL_PANEL_COPY:
      HIP_CHECK_ERROR(hipEventRecord(panelCopy, dataStream));
      break;

    case HPL_PANEL_UPDATE:
      HIP_CHECK_ERROR(hipEventRecord(panelUpdate, computeStream));
      break;

    case HPL_RS_1:
      HIP_CHECK_ERROR(hipEventRecord(pdlaswpFinish_1, pdlaswpStream));
      break;

    case HPL_RS_2:
      HIP_CHECK_ERROR(hipEventRecord(pdlaswpFinish_2, pdlaswpStream));
      break;

    case DGEMMSTART:
      HIP_CHECK_ERROR(hipEventRecord(dgemmStart[UPD], computeStream));
      break;

    case DGEMMSTOP:
      HIP_CHECK_ERROR(hipEventRecord(dgemmStop[UPD], computeStream));
      break;

    case UPDATE:
      HIP_CHECK_ERROR(hipEventRecord(update[UPD], computeStream));
      break; 
    
    case SWAPSTART:
      HIP_CHECK_ERROR(hipEventRecord(swapStartEvent[UPD], computeStream));
      break;

    default:
      break;
    }
}

void HIP::event_synchronize(enum HPL_EVENT _event, const HPL_T_UPD UPD) {
    switch (_event)
    {
    case HPL_PANEL_COPY:
      HIP_CHECK_ERROR(hipEventSynchronize(panelCopy));        
      break;

    case HPL_PANEL_UPDATE:
      HIP_CHECK_ERROR(hipEventSynchronize(panelUpdate));
      break;

    case HPL_RS_1:
      HIP_CHECK_ERROR(hipEventSynchronize(pdlaswpFinish_1));
      break;

    case HPL_RS_2:
      HIP_CHECK_ERROR(hipEventSynchronize(pdlaswpFinish_2));
      break;

    case SWAPSTART:
      HIP_CHECK_ERROR(hipEventSynchronize(swapStartEvent[UPD]));
      break;

    default:
      break;
    }
}

void HIP::stream_synchronize(enum HPL_STREAM _stream) {
    switch (_stream)
    {
    case HPL_COMPUTESTREAM:
      HIP_CHECK_ERROR(hipStreamSynchronize(computeStream));        
      break;
    case HPL_DATASTREAM:
      HIP_CHECK_ERROR(hipStreamSynchronize(dataStream));    
      break;    
    case HPL_PDLASWPSTREAM:
      HIP_CHECK_ERROR(hipStreamSynchronize(pdlaswpStream));        
      break;
    default:
      break;
    }
}

void HIP::stream_wait_event(enum HPL_STREAM _stream, enum HPL_EVENT _event){
    switch (_event)
    {
    case HPL_PANEL_UPDATE:
      HIP_CHECK_ERROR(hipStreamWaitEvent(pdlaswpStream, panelUpdate, 0));        
      break;

    case HPL_RS_1:
      HIP_CHECK_ERROR(hipStreamWaitEvent(computeStream, pdlaswpFinish_1, 0));        
      break;

    case HPL_RS_2:
      HIP_CHECK_ERROR(hipStreamWaitEvent(computeStream, pdlaswpFinish_2, 0));        
      break;
    
    case UPDATE_LOOK_AHEAD:
      HIP_CHECK_ERROR(hipStreamWaitEvent(dataStream, update[HPL_LOOK_AHEAD], 0));  
      break;

    case L1TRANSFER:
      HIP_CHECK_ERROR(hipStreamWaitEvent(computeStream, L1Transfer, 0));  
      break;

    case L2TRANSFER:
      HIP_CHECK_ERROR(hipStreamWaitEvent(computeStream, L2Transfer, 0));  
      break;   

    case SWAPDATATRANSFER:
      HIP_CHECK_ERROR(hipStreamWaitEvent(computeStream, swapDataTransfer, 0));
      break;

    default:
      break;
    }
}

float HIP::elapsedTime(const HPL_T_UPD UPD){
    float time = 0.f;
    HIP_CHECK_ERROR(hipEventElapsedTime(&time, dgemmStart[UPD], dgemmStop[UPD]));
    return time;
}

int HIP::idamax(const int N, const double *DX, const int INCX)
{
    GPUInfo("%-25s %-17d \t%-5s", "[IDAMAX]", "With X of (R)", N, "HIP");
    rocblas_int result;
    ROCBLAS_CHECK_STATUS(rocblas_idamax(_handle, N, DX, INCX, &result));
    return result;
}

void HIP::daxpy(const int N, const double DA, const double *DX, const int INCX, double *DY, 
                const int INCY)
{
    GPUInfo("%-25s %-17d \t%-5s", "[DAXPY]", "With X of (R)", N, "HIP");
    ROCBLAS_CHECK_STATUS(rocblas_daxpy(_handle, N, &DA, DX, INCX, DY, INCY));
}

void HIP::dscal(const int N, const double DA, double *DX, const int INCX)
{
    GPUInfo("%-25s %-17d \t%-5s", "[DSCAL]", "With X of (R)", N, "HIP");
    ROCBLAS_CHECK_STATUS(rocblas_dscal(_handle, N, &DA, DX, INCX));
}

void HIP::dswap(const int N, double *DX, const int INCX, double *DY, const int INCY)
{    
    GPUInfo("%-25s %-17d \t%-5s", "[DSWAP]", "With X of (R)", N, "HIP");
    ROCBLAS_CHECK_STATUS(rocblas_dswap(_handle, N, DX, INCX, DY, INCY));
}

void HIP::dger( const enum HPL_ORDER ORDER, const int M, const int N, const double ALPHA, const double *X,
               const int INCX, double *Y, const int INCY, double *A, const int LDA)
{
    GPUInfo("%-25s %-8d%-8d \t%-5s", "[DGER]", "With A of (R:C)", M, N, "HIP");
    //rocBLAS uses column-major storage for 2D arrays
    ROCBLAS_CHECK_STATUS(rocblas_dger(_handle, M, N, &ALPHA, X, INCX, Y, INCY, A, LDA));
}

void HIP::trsm( const enum HPL_ORDER ORDER, const enum HPL_SIDE SIDE, 
                const enum HPL_UPLO UPLO, const enum HPL_TRANS TRANSA, 
                const enum HPL_DIAG DIAG, const int M, const int N, 
                const double ALPHA, const double *A, const int LDA, double *B, const int LDB)
{
    GPUInfo("%-25s %-8d%-8d \t%-5s", "[DTRSM]", "With C of (R:C)", M, N, "HIP");
#if 1
    //rocBLAS uses column-major storage for 2D arrays
    ROCBLAS_CHECK_STATUS(rocblas_dtrsm(_handle, (rocblas_side)SIDE, (rocblas_fill)UPLO, (rocblas_operation)TRANSA, 
                  (rocblas_diagonal)DIAG, M, N, &ALPHA, A, LDA, B, LDB));
#else
    double * d_A, * d_B;
    HIP::malloc((void**)&d_A, LDA*M*sizeof(double));
    HIP::malloc((void**)&d_B, LDB*N*sizeof(double));

    HIP::move_data(d_A, A, LDA*M*sizeof(double), 1);
    HIP::move_data(d_B, B, LDB*N*sizeof(double), 1);
    
    ROCBLAS_CHECK_STATUS(rocblas_dtrsm(_handle, (rocblas_side)SIDE, (rocblas_fill)UPLO, (rocblas_operation)TRANSA, 
                  (rocblas_diagonal)DIAG, M, N, &ALPHA, d_A, LDA, d_B, LDB));
                  
    HIP::move_data(B, d_B, LDB*N*sizeof(double), 2);

    HIP::free((void**)&d_A);
    HIP::free((void**)&d_B);
#endif    
}

void HIP::trsv(const enum HPL_ORDER ORDER, const enum HPL_UPLO UPLO,
                const enum HPL_TRANS TRANSA, const enum HPL_DIAG DIAG,
                const int N, const double *A, const int LDA,
                double *X, const int INCX)
{ 
    GPUInfo("%-25s %-17d \t%-5s", "[TRSV]", "With A of (R)", N, "HIP");
    //rocBLAS uses column-major storage for 2D arrays
    ROCBLAS_CHECK_STATUS(rocblas_dtrsv(_handle, (rocblas_fill)UPLO, (rocblas_operation)TRANSA,
                    (rocblas_diagonal)DIAG, N, A, LDA, X, INCX));
}

void HIP::dgemm(const enum HPL_ORDER ORDER, const enum HPL_TRANS TRANSA, 
                const enum HPL_TRANS TRANSB, const int M, const int N, const int K, 
                const double ALPHA, const double *A, const int LDA, 
                const double *B, const int LDB, const double BETA, double *C, 
                const int LDC)
{
    GPUInfo("%-25s %-8d%-8d%-8d \t%-5s", "[DGEMM]", "With C of (R:C)", M, N, K, "HIP");
#if 1
    //rocBLAS uses column-major storage for 2D arrays
    ROCBLAS_CHECK_STATUS(rocblas_dgemm(_handle, (rocblas_operation)TRANSA, (rocblas_operation)TRANSB, 
                         M, N, K, &ALPHA, A, LDA, B, LDB, &BETA, C, LDC));
#else
    double                    * d_A, * d_B, * d_C;
    HIP::malloc((void**)&d_A, LDA*K*sizeof(double));
    HIP::malloc((void**)&d_B, LDB*N*sizeof(double));
    HIP::malloc((void**)&d_C, LDC*N*sizeof(double));

    HIP::move_data(d_A, A, LDA*K*sizeof(double), 1);
    HIP::move_data(d_B, B, LDB*N*sizeof(double), 1);
    HIP::move_data(d_C, C, LDC*N*sizeof(double), 1);

    ROCBLAS_CHECK_STATUS(rocblas_dgemm(_handle, (rocblas_operation)TRANSA, (rocblas_operation)TRANSB, 
                         M, N, K, &ALPHA, d_A, LDA, d_B, LDB, &BETA, d_C, LDC));

    HIP::move_data(C, d_C, LDC*N*sizeof(double), 2);

    HIP::free((void**)&d_A);
    HIP::free((void**)&d_B);
    HIP::free((void**)&d_C);
#endif

}

void HIP::dgemv(const enum HPL_ORDER ORDER, const enum HPL_TRANS TRANS, const int M, const int N,
                const double ALPHA, const double *A, const int LDA, const double *X, const int INCX,
                const double BETA, double *Y, const int INCY)
{
    GPUInfo("%-25s %-8d%-8d \t%-5s", "[DGEMV]", "With A of (R:C)", M, N, "HIP");
    //rocBLAS uses column-major storage for 2D arrays
    ROCBLAS_CHECK_STATUS(rocblas_dgemv(_handle, (rocblas_operation)TRANS,M, N, &ALPHA, A, LDA, X, INCX, &BETA, Y, INCY));
}

/*
*  ----------------------------------------------------------------------
*  - COPY ---------------------------------------------------------------
*  ----------------------------------------------------------------------
*/ 
void HIP::copy(const int N, const double *X, const int INCX, double *Y, const int INCY)
{
    GPUInfo("%-25s %-17d \t%-5s", "[COPY]", "With X of (R)", N, "HIP");
    ROCBLAS_CHECK_STATUS(rocblas_dcopy(_handle, N, X, INCX, Y, INCY));
}

__global__ void 
_dlacpy(const int M, const int N, const double *A, const int LDA,
        double *B, const int LDB)
{
 
}

/*
* Copies an array A into an array B.
*/
void HIP::acpy(const int M, const int N, const double *A, const int LDA,
                  double *B, const int LDB)
{
    GPUInfo("%-25s %-8d%-8d \t%-5s", "[LACOPY]", "With A of (R:C)", M, N, "HIP");
    dim3 block_size(64, 1);
    dim3 grid_size((M+64-1)/64, (N+64-1)/64);
    _dlacpy<<<block_size, grid_size, 0, 0>>>(M, N, A, LDA, B, LDB);
}

#define TILE_DIM 64
#define BLOCK_ROWS 16

// Transpose kernel
__global__ void 
_dlatcpy(const int M, const int N, const double* __restrict__ A, const int LDA, double* __restrict__ B, const int LDB) 
{
  __shared__ double s_tile[TILE_DIM][TILE_DIM + 1];

  int I = blockIdx.x * TILE_DIM + threadIdx.y;
  int J = blockIdx.y * TILE_DIM + threadIdx.x;

  if(J < N) {
    if(I + 0 < M)
      s_tile[threadIdx.y + 0][threadIdx.x] = A[((size_t)I + 0) * LDA + J];
    if(I + 16 < M)
      s_tile[threadIdx.y + 16][threadIdx.x] = A[((size_t)I + 16) * LDA + J];
    if(I + 32 < M)
      s_tile[threadIdx.y + 32][threadIdx.x] = A[((size_t)I + 32) * LDA + J];
    if(I + 48 < M)
      s_tile[threadIdx.y + 48][threadIdx.x] = A[((size_t)I + 48) * LDA + J];
  }

  I = blockIdx.x * TILE_DIM + threadIdx.x;
  J = blockIdx.y * TILE_DIM + threadIdx.y;

  __syncthreads();

  if(I < M) {
    if(J + 0 < N)
      B[I + ((size_t)J + 0) * LDB] = s_tile[threadIdx.x][threadIdx.y + 0];
    if(J + 16 < N)
      B[I + ((size_t)J + 16) * LDB] = s_tile[threadIdx.x][threadIdx.y + 16];
    if(J + 32 < N)
      B[I + ((size_t)J + 32) * LDB] = s_tile[threadIdx.x][threadIdx.y + 32];
    if(J + 48 < N)
      B[I + ((size_t)J + 48) * LDB] = s_tile[threadIdx.x][threadIdx.y + 48];
  }
}

/*
* Copies the transpose of an array A into an array B.
*/
void HIP::atcpy(const int M, const int N, const double *A, const int LDA, double *B, const int LDB)
{
    GPUInfo("%-25s %-8d%-8d \t%-5s", "[LATCOPY]", "With A of (R:C)", M, N, "HIP");
    hipStream_t stream;
    ROCBLAS_CHECK_STATUS(rocblas_get_stream(_handle, &stream));
    dim3 grid_size((M + TILE_DIM - 1) / TILE_DIM, (N + TILE_DIM - 1) / TILE_DIM);
    dim3 block_size(TILE_DIM, BLOCK_ROWS);
    hipLaunchKernelGGL((_dlatcpy), grid_size, block_size, 0, stream, M, N, A, LDA, B, LDB);
}

void HIP::move_data(double *DST, const double *SRC, const size_t SIZE, const int KIND)
{
    char title[25] = "[MOVE_"; strcat(title,_memcpyKind[KIND]); strcat(title,"]");
    GPUInfo("%-25s %-12ld (B) \t%-5s", title, "Memory of size",  SIZE, "HIP");
    HIP_CHECK_ERROR(hipMemcpy(DST, SRC, SIZE, (hipMemcpyKind)KIND));
}

void HIP::move_data_2d(void* dst, size_t dpitch, const void* src, size_t spitch, size_t width, size_t height, const int KIND)
{
    char title[25] = "[MOVE2D_"; strcat(title,_memcpyKind[KIND]); strcat(title,"]");
    GPUInfo("%-25s %-12ld (B) \t%-5s", title, "Memory of size",  SIZE, "HIP");
    HIP_CHECK_ERROR(hipMemcpy2D(dst, dpitch, src, spitch, width, height, (hipMemcpyKind)KIND));
}


void HIP::device_sync() {
    HIP_CHECK_ERROR(hipDeviceSynchronize());
}

int HIP::bcast_ibcst(HPL_T_panel* PANEL, int* IFLAG) {

  double *L2ptr;
#ifdef ROCM
  L2ptr = PANEL->dL2;
#else
  L2ptr = PANEL->L2;
#endif

  if(PANEL == NULL) {
    return HPL_SUCCESS;
  }
  if(PANEL->grid->npcol <= 1) {
    return HPL_SUCCESS;
  }

  MPI_Comm comm  = PANEL->grid->row_comm;
  int root  = PANEL->pcol;

  if(PANEL->len <= 0) return HPL_SUCCESS;
  int ierr = MPI_Ibcast(L2ptr, PANEL->len, MPI_DOUBLE, root, comm, &bcast_req);
  *IFLAG = ((ierr == MPI_SUCCESS ? HPL_SUCCESS : HPL_FAILURE));
  return *IFLAG;
}

int HIP::bwait_ibcast(HPL_T_panel* PANEL) {
  int ierr;
  ierr = MPI_Wait(&bcast_req, MPI_STATUS_IGNORE);
  return ((ierr == MPI_SUCCESS ? HPL_SUCCESS : HPL_FAILURE));
}

void HIP::HPL_pdlaswp_hip(HPL_T_panel* PANEL, const HPL_T_UPD UPD, const SWP_PHASE phase) {
  double *U, *W;
  double *dA, *dU, *dW;
  int *ipID, *iplen, *ipcounts, *ipoffsets, *iwork, *lindxU = NULL, *lindxA = NULL, *lindxAU, *permU;
  int *dlindxU = NULL, *dlindxA = NULL, *dlindxAU, *dpermU, *dpermU_ex;
  int  icurrow, *iflag, *ipA, *ipl, jb, k, lda, myrow, n, nprow, LDU, LDW; 
  MPI_Comm comm;

  /*
   * Retrieve parameters from the PANEL data structure
   */
  n = PANEL->n; jb = PANEL->jb;
  nprow = PANEL->grid->nprow; myrow = PANEL->grid->myrow;
  comm = PANEL->grid->col_comm; icurrow = PANEL->prow;
  iflag = PANEL->IWORK;
  dA = PANEL->dA; lda = PANEL->dlda;
  PANEL->pmat->n = PANEL->dldl1;

  // Quick return if we're 1xQ
  if(phase != SWP_END && nprow == 1) return;

  pdlaswp_set_var(PANEL, dU, U, LDU, dW, W, LDW, n, dA, UPD);

  /* Quick return if there is nothing to do */
  if((n <= 0) || (jb <= 0)) return;

  // Quick swapping if P==1
  if (phase == SWP_END && nprow == 1) {
    // wait for swapping data to arrive
    HPL_BE_stream_wait_event(HPL_COMPUTESTREAM, SWAPDATATRANSFER, HPL_TR);

    HIP::HPL_dlaswp00N(jb, n, dA, lda, PANEL->dipiv);
    return;
  }

  /*
   * Compute ipID (if not already done for this panel). lindxA and lindxAU
   * are of length at most 2*jb - iplen is of size nprow+1, ipmap, ipmapm1
   * are of size nprow,  permU is of length jb, and  this function needs a
   * workspace of size max( 2 * jb (plindx1), nprow+1(equil)):
   * 1(iflag) + 1(ipl) + 1(ipA) + 9*jb + 3*nprow + 1 + MAX(2*jb,nprow+1)
   * i.e. 4 + 9*jb + 3*nprow + max(2*jb, nprow+1);
   */
  k = (int)((unsigned int)(jb) << 1);
  ipl = iflag + 1;
  ipID = ipl + 1;
  ipA = ipID + ((unsigned int)(k) << 1);
  iplen = ipA + 1;
  ipcounts = iplen + nprow + 1;
  ipoffsets = ipcounts + nprow;
  iwork = ipoffsets + nprow;

  if (phase == SWP_START) {
    if(*iflag == -1) {/* no index arrays have been computed so far */
        // get the ipivs on the host after the Bcast
        if(PANEL->grid->mycol != PANEL->pcol) {
          HIP_CHECK_ERROR(hipMemcpy2DAsync(PANEL->ipiv, PANEL->jb * sizeof(int),
                          PANEL->dipiv, PANEL->jb * sizeof(int),
                          PANEL->jb * sizeof(int), 1,
                          hipMemcpyDeviceToHost, HIP::dataStream));
        }
        HPL_BE_stream_synchronize(HPL_DATASTREAM, HPL_TR);

        // compute spreading info
        HPL_pipid(PANEL, ipl, ipID);
        HPL_plindx(PANEL, *ipl, ipID, ipA, PANEL->lindxU, PANEL->lindxAU, PANEL->lindxA, iplen, PANEL->permU, iwork);
        *iflag = 1;
    }

      /*
      * For i in [0..2*jb),  lindxA[i] is the offset in A of a row that ulti-
      * mately goes to U( :, lindxAU[i] ).  In each rank, we directly pack
      * into U, otherwise we pack into workspace. The  first
      * entry of each column packed in workspace is in fact the row or column
      * offset in U where it should go to.
      */
      if(myrow == icurrow) {
        // copy needed rows of A into U
        HIP::HPL_dlaswp01T(jb, n, dA, lda, dU, LDU, PANEL->dlindxU);
        // record the evernt when packing completes
        HIP::event_record(SWAPSTART, UPD);
      } else {
        // copy needed rows from A into U(:, iplen[myrow])
        HIP::HPL_dlaswp03T(iplen[myrow + 1] - iplen[myrow], n, dA, lda, Mptr(dU, 0, iplen[myrow], LDU), LDU, PANEL->dlindxU);
        // record the event when packing completes
        HIP::event_record(SWAPSTART, UPD);
      }
  }
  else if (phase == SWP_COMM) {
    /* Set MPI message counts and offsets */
    ipcounts[0] = (iplen[1] - iplen[0]) * LDU;
    ipoffsets[0] = 0;
    PANEL->pmat->n = PANEL->dldl1;
    // if (phase == SWP_END)
    //   PANEL->pmat->n = PANEL->pmat->dN;
    for(int i = 1; i < nprow; ++i) {
      ipcounts[i] = (iplen[i + 1] - iplen[i]) * LDU;
      ipoffsets[i] = ipcounts[i - 1] + ipoffsets[i - 1];
    }

    if(myrow == icurrow) {
      HIP::event_synchronize(SWAPSTART, UPD);
      // Send rows info to other ranks
      HPL_scatterv(dU, ipcounts, ipoffsets, ipcounts[myrow], icurrow, comm);
      // All gather dU (gather + broadcast)
      HPL_allgatherv(dU, ipcounts[myrow], ipcounts, ipoffsets, comm);
    } else {
      // Wait for dU to be ready
      HIP::event_synchronize(SWAPSTART, UPD);
      // Receive rows from icurrow into dW
      HPL_scatterv(dW, ipcounts, ipoffsets, ipcounts[myrow], icurrow, comm);
      // All gather dU
      HPL_allgatherv(dU, ipcounts[myrow], ipcounts, ipoffsets, comm);
    }
  }
  else if (phase == SWP_END) {
    if(myrow == icurrow) {
        // Swap rows local to A on device
        HIP::HPL_dlaswp02T(*ipA, n, dA, lda, PANEL->dlindxAU, PANEL->dlindxA);
    } else {
        // Queue inserting recieved rows in W into A on device
        HIP::HPL_dlaswp04T(iplen[myrow + 1] - iplen[myrow], n, dA, lda, dW, LDW, PANEL->dlindxU);
    }
    /* Permute U in every process row */
    HIP::HPL_dlaswp10N(n, jb, dU, LDU, PANEL->dpermU);

  }
}

#define BLOCK_SIZE_PDLANGE 512
#define GRID_SIZE_PDLANGE 512

__global__ void normA_1(const int N, const int M, const double* __restrict__ A, const int LDA, double* __restrict__ normAtmp) {
  __shared__ double s_norm[BLOCK_SIZE_PDLANGE];

  const int t  = threadIdx.x, i  = blockIdx.x;
  size_t id = i * BLOCK_SIZE_PDLANGE + t;

  s_norm[t] = 0.0;
  for(; id < (size_t)N * M; id += gridDim.x * BLOCK_SIZE_PDLANGE) {
    const int m = id % M, n = id / M;
    const double Anm = fabs(A[n + ((size_t)m) * LDA]);

    s_norm[t] = (Anm > s_norm[t]) ? Anm : s_norm[t];
  }
  __syncthreads();

  for(int k = BLOCK_SIZE_PDLANGE / 2; k > 0; k /= 2) {
    if(t < k) {
      s_norm[t] = (s_norm[t + k] > s_norm[t]) ? s_norm[t + k] : s_norm[t];
    }
    __syncthreads();
  }

  if(t == 0) normAtmp[i] = s_norm[0];
}

__global__ void normA_2(const int N, double* __restrict__ normAtmp) {
  __shared__ double s_norm[BLOCK_SIZE_PDLANGE];

  const int t = threadIdx.x;

  s_norm[t] = 0.0;
  for(size_t id = t; id < N; id += BLOCK_SIZE_PDLANGE) {
    const double Anm = normAtmp[id];
    s_norm[t] = (Anm > s_norm[t]) ? Anm : s_norm[t];
  }
  __syncthreads();

  for(int k = BLOCK_SIZE_PDLANGE / 2; k > 0; k /= 2) {
    if(t < k) {
      s_norm[t] = (s_norm[t + k] > s_norm[t]) ? s_norm[t + k] : s_norm[t];
    }
    __syncthreads();
  }

  if(t == 0) normAtmp[0] = s_norm[0];
}

__global__ void norm1(const int N, const int M, const double* __restrict__ A, const int LDA, double* __restrict__ work) {

  __shared__ double s_norm1[BLOCK_SIZE_PDLANGE];

  const int t = threadIdx.x, n = blockIdx.x;

  s_norm1[t] = 0.0;
  for(size_t id = t; id < M; id += BLOCK_SIZE_PDLANGE) {
    s_norm1[t] += fabs(A[id + n * ((size_t)LDA)]);
  }

  __syncthreads();

  for(int k = BLOCK_SIZE_PDLANGE / 2; k > 0; k /= 2) {
    if(t < k) { s_norm1[t] += s_norm1[t + k]; }
    __syncthreads();
  }

  if(t == 0) work[n] = s_norm1[0];
}

__global__ void norminf(const int N, const int M, const double* __restrict__ A, const int LDA, double* __restrict__ work) {
  const int t  = threadIdx.x, b  = blockIdx.x;
  const size_t id = b * BLOCK_SIZE_PDLANGE + t; // row id

  if(id < M) {
    double norm = 0.0;
    for(size_t i = 0; i < N; i++) { norm += fabs(A[id + i * ((size_t)LDA)]); }
    work[id] = norm;
  }
}

double HIP::pdlange(const HPL_T_grid* GRID, const HPL_T_NORM  NORM, const int M, const int N, const int NB, const double* A, const int LDA) {

  double s, v0 = HPL_rzero, *work = NULL, *dwork = NULL;
  MPI_Comm Acomm, Ccomm, Rcomm;
  int ii, jj, mp, mycol, myrow, npcol, nprow, nq;

  (void)HPL_grid_info(GRID, &nprow, &npcol, &myrow, &mycol);
  Rcomm = GRID->row_comm; Ccomm = GRID->col_comm; Acomm = GRID->all_comm;

  Mnumroc(mp, M, NB, NB, myrow, 0, nprow);
  Mnumroc(nq, N, NB, NB, mycol, 0, npcol);

  if(Mmin(M, N) == 0) {
    return (v0);
  }
  else if(NORM == HPL_NORM_A) {
    if((nq > 0) && (mp > 0)) {
      if(nq == 1) { // column vector
        int id;
        ROCBLAS_CHECK_STATUS(rocblas_idamax(_handle, mp, A, 1, &id));
        HIP_CHECK_ERROR(hipMemcpy(&v0, A + id - 1, 1 * sizeof(double), hipMemcpyDeviceToHost));
      }
      else if(mp == 1) { // row vector
        int id;
        ROCBLAS_CHECK_STATUS(rocblas_idamax(_handle, nq, A, LDA, &id));
        HIP_CHECK_ERROR(hipMemcpy(&v0, A + ((size_t)id * LDA), 1 * sizeof(double), hipMemcpyDeviceToHost));
      }
      else {
        // custom reduction kernels
        HIP_CHECK_ERROR(hipMalloc(&dwork, GRID_SIZE_PDLANGE * sizeof(double)));

        size_t grid_size = (nq * mp + BLOCK_SIZE_PDLANGE - 1) / BLOCK_SIZE_PDLANGE;
        grid_size        = (grid_size < GRID_SIZE_PDLANGE) ? grid_size : GRID_SIZE_PDLANGE;

        hipLaunchKernelGGL((normA_1), dim3(grid_size), dim3(BLOCK_SIZE_PDLANGE), 0, 0,
                           nq, mp, A, LDA, dwork);
        hipLaunchKernelGGL((normA_2), dim3(1), dim3(BLOCK_SIZE_PDLANGE), 0, 0, grid_size, dwork);

        HIP_CHECK_ERROR(hipMemcpy(&v0, dwork, 1 * sizeof(double), hipMemcpyDeviceToHost));
        HIP_CHECK_ERROR(hipFree(dwork));
      }
    }
    (void)HPL_reduce((void*)(&v0), 1, HPL_DOUBLE, HPL_max, 0, Acomm);
  }
  else if(NORM == HPL_NORM_1) {
    /* Find norm_1( A ).  */
    if(nq > 0) {
      work = (double*)std::malloc((size_t)(nq) * sizeof(double));
      if(work == NULL) {
        HPL_pabort(__LINE__, "HPL_pdlange", "Memory allocation failed");
      }

      if(nq == 1) { // column vector
        ROCBLAS_CHECK_STATUS(rocblas_dasum(_handle, mp, A, 1, work));
      }
      else {
        HIP_CHECK_ERROR(hipMalloc(&dwork, nq * sizeof(double)));
        hipLaunchKernelGGL(
            (norm1), dim3(nq), dim3(BLOCK_SIZE_PDLANGE), 0, 0, nq, mp, A, LDA, dwork);
        HIP_CHECK_ERROR(hipMemcpy(work, dwork, nq * sizeof(double), hipMemcpyDeviceToHost));
      }
      /* Find sum of global matrix columns, store on row 0 of process grid */
      (void)HPL_reduce((void*)(work), nq, HPL_DOUBLE, HPL_sum, 0, Ccomm);
      /* Find maximum sum of columns for 1-norm */
      if(myrow == 0) {
        v0 = work[HPL_idamax(nq, work, 1)];
        v0 = Mabs(v0);
      }
      if(work) std::free(work);
      if(dwork) HIP_CHECK_ERROR(hipFree(dwork));
    }
    /* Find max in row 0, store result in process (0,0) */
    if(myrow == 0)
      (void)HPL_reduce((void*)(&v0), 1, HPL_DOUBLE, HPL_max, 0, Rcomm);
  }
  else if(NORM == HPL_NORM_I) {
    /* Find norm_inf( A ) */
    if(mp > 0) {
      work = (double*)std::malloc((size_t)(mp) * sizeof(double));
      if(work == NULL) {
        HPL_pabort(__LINE__, "HPL_pdlange", "Memory allocation failed");
      }

      if(mp == 1) { // row vector
        ROCBLAS_CHECK_STATUS(rocblas_dasum(_handle, nq, A, LDA, work));
      }
      else {
        HIP_CHECK_ERROR(hipMalloc(&dwork, mp * sizeof(double)));

        size_t grid_size = (mp + BLOCK_SIZE_PDLANGE - 1) / BLOCK_SIZE_PDLANGE;
        hipLaunchKernelGGL((norminf), dim3(grid_size), dim3(BLOCK_SIZE_PDLANGE), 0, 0,
                           nq, mp, A, LDA, dwork);
        HIP_CHECK_ERROR(hipMemcpy(work, dwork, mp * sizeof(double), hipMemcpyDeviceToHost));
      }

      (void)HPL_reduce((void*)(work), mp, HPL_DOUBLE, HPL_sum, 0, Rcomm);
      /* Find maximum sum of rows for inf-norm */
      if(mycol == 0) {
        v0 = work[HPL_idamax(mp, work, 1)];
        v0 = Mabs(v0);
      }
      if(work) std::free(work);
      if(dwork) HIP_CHECK_ERROR(hipFree(dwork));
    }
    /* Find max in column 0, store result in process (0,0) */
    if(mycol == 0)
      (void)HPL_reduce((void*)(&v0), 1, HPL_DOUBLE, HPL_max, 0, Ccomm);
  }
  /* Broadcast answer to every process in the grid */
  (void)HPL_broadcast((void*)(&v0), 1, HPL_DOUBLE, 0, Acomm);

  return (v0);
}

#define BLOCK_SIZE_00N 512

__global__ void _dlaswp00N(const int N, const int M, double* __restrict__ A, const int LDA, const int* __restrict__ IPIV) {

  __shared__ double s_An_init[2048];
  __shared__ double s_An_ipiv[2048];

  const int m = threadIdx.x;
  const int n = blockIdx.x;

  // read in block column
  for(int i = m; i < M; i += blockDim.x)
    s_An_init[i] = A[i + n * ((size_t)LDA)];

  __syncthreads();

  // local block
  for(int i = m; i < M; i += blockDim.x) {
    const int ip = IPIV[i];

    if(ip < M) { // local swap
      s_An_ipiv[i] = s_An_init[ip];
    } else { // non local swap
      s_An_ipiv[i] = A[ip + n * ((size_t)LDA)];
    }
  }
  __syncthreads();

  // write out local block
  for(int i = m; i < M; i += blockDim.x)
    A[i + n * ((size_t)LDA)] = s_An_ipiv[i];

  // remaining swaps in column
  for(int i = m; i < M; i += blockDim.x) {
    const int ip_ex = IPIV[i + M];

    if(ip_ex > -1)
      A[ip_ex + n * ((size_t)LDA)] = s_An_init[i];
  }
}

// Row swapping for P==1
void HIP::HPL_dlaswp00N(const int M, const int N, double* A, const int LDA, const int* IPIV) {

  if((M <= 0) || (N <= 0)) return;

  hipStream_t stream;
  ROCBLAS_CHECK_STATUS(rocblas_get_stream(_handle, &stream));
  int grid_size = N;
  hipLaunchKernelGGL((_dlaswp00N), dim3(grid_size), dim3(BLOCK_SIZE_00N), 0, stream, N, M, A, LDA, IPIV);
}


#define TILE_DIM_01T 32
#define BLOCK_ROWS_01T 8

/* Build U matrix from rows of A */
__global__ void _dlaswp01T(const int M, const int N, double* __restrict__ A, const int LDA, double* __restrict__ U, const int LDU, const int* __restrict__ LINDXU) {

  __shared__ double s_U[TILE_DIM_01T][TILE_DIM_01T + 1];

  const int m = threadIdx.x + TILE_DIM_01T * blockIdx.x;
  const int n = threadIdx.y + TILE_DIM_01T * blockIdx.y;

  if(m < M) {
    const int ipa  = LINDXU[m];

    // Save to LDS to reduce global memory operation
    s_U[threadIdx.x][threadIdx.y + 0] =
        (n + 0 < N) ? A[ipa + (n + 0) * ((size_t)LDA)] : 0.0;
    s_U[threadIdx.x][threadIdx.y + 8] =
        (n + 8 < N) ? A[ipa + (n + 8) * ((size_t)LDA)] : 0.0;
    s_U[threadIdx.x][threadIdx.y + 16] =
        (n + 16 < N) ? A[ipa + (n + 16) * ((size_t)LDA)] : 0.0;
    s_U[threadIdx.x][threadIdx.y + 24] =
        (n + 24 < N) ? A[ipa + (n + 24) * ((size_t)LDA)] : 0.0;
  }

  __syncthreads();

  const int um = threadIdx.y + TILE_DIM_01T * blockIdx.x;
  const int un = threadIdx.x + TILE_DIM_01T * blockIdx.y;

  if(un < N) {
    // write out chunks of U
    if((um + 0) < M)
      U[un + (um + 0) * ((size_t)LDU)] = s_U[threadIdx.y + 0][threadIdx.x];
    if((um + 8) < M)
      U[un + (um + 8) * ((size_t)LDU)] = s_U[threadIdx.y + 8][threadIdx.x];
    if((um + 16) < M)
      U[un + (um + 16) * ((size_t)LDU)] = s_U[threadIdx.y + 16][threadIdx.x];
    if((um + 24) < M)
      U[un + (um + 24) * ((size_t)LDU)] = s_U[threadIdx.y + 24][threadIdx.x];
  }
}

void HIP::HPL_dlaswp01T(const int M, const int N, double* A, const int LDA, double* U, const int LDU, const int* LINDXU) {

  if((M <= 0) || (N <= 0)) return;

  hipStream_t stream;
  ROCBLAS_CHECK_STATUS(rocblas_get_stream(_handle, &stream));
  dim3 grid_size((M + TILE_DIM_01T - 1) / TILE_DIM_01T, (N + TILE_DIM_01T - 1) / TILE_DIM_01T);
  dim3 block_size(TILE_DIM_01T, BLOCK_ROWS_01T);
  hipLaunchKernelGGL((_dlaswp01T), grid_size, block_size, 0, stream, M, N, A, LDA, U, LDU, LINDXU);
}

/* Perform any local row swaps of A */
__global__ void _dlaswp02T(const int M, const int N, double* __restrict__ A, const int LDA, const int* __restrict__ LINDXAU, const int* __restrict__ LINDXA) {

  const int n = blockIdx.x, m = threadIdx.x;

  const int srow = LINDXAU[m]; //src row
  const int drow  = LINDXA[m];  //dst row

  const double An = A[srow + n * ((size_t)LDA)];

  __syncthreads();

  A[drow + n * ((size_t)LDA)] = An;
}

void HIP::HPL_dlaswp02T(const int M, const int N, double* A, const int LDA, const int* LINDXAU, const int* LINDXA) {

  if((M <= 0) || (N <= 0)) return;

  hipStream_t stream;
  ROCBLAS_CHECK_STATUS(rocblas_get_stream(_handle, &stream));
  dim3 grid_size(N), block_size(M);
  hipLaunchKernelGGL((_dlaswp02T), N, M, 0, stream, M, N, A, LDA, LINDXAU, LINDXA);
}

#define TILE_DIM_03T 32
#define BLOCK_ROWS_03T 8

/* Build W matrix from rows of A */
__global__ void _dlaswp03T(const int M, const int N, double* __restrict__ A, const int LDA, double* __restrict__ W, const int LDW, const int* __restrict__ LINDXU) {

  __shared__ double s_W[TILE_DIM_03T][TILE_DIM_03T + 1];

  const int m = threadIdx.x + TILE_DIM_03T * blockIdx.x;
  const int n = threadIdx.y + TILE_DIM_03T * blockIdx.y;

  if(m < M) {
    const int ipa = LINDXU[m];

    // Save to LDS to reduce global memory operation
    s_W[threadIdx.x][threadIdx.y + 0] =
        (n + 0 < N) ? A[ipa + (n + 0) * ((size_t)LDA)] : 0.0;
    s_W[threadIdx.x][threadIdx.y + 8] =
        (n + 8 < N) ? A[ipa + (n + 8) * ((size_t)LDA)] : 0.0;
    s_W[threadIdx.x][threadIdx.y + 16] =
        (n + 16 < N) ? A[ipa + (n + 16) * ((size_t)LDA)] : 0.0;
    s_W[threadIdx.x][threadIdx.y + 24] =
        (n + 24 < N) ? A[ipa + (n + 24) * ((size_t)LDA)] : 0.0;
  }

  __syncthreads();

  const int wm = threadIdx.y + TILE_DIM_03T * blockIdx.x;
  const int wn = threadIdx.x + TILE_DIM_03T * blockIdx.y;

  if(wn < N) {
    // write out chunks of W
    if((wm + 0) < M)
      W[wn + (wm + 0) * ((size_t)LDW)] = s_W[threadIdx.y + 0][threadIdx.x];
    if((wm + 8) < M)
      W[wn + (wm + 8) * ((size_t)LDW)] = s_W[threadIdx.y + 8][threadIdx.x];
    if((wm + 16) < M)
      W[wn + (wm + 16) * ((size_t)LDW)] = s_W[threadIdx.y + 16][threadIdx.x];
    if((wm + 24) < M)
      W[wn + (wm + 24) * ((size_t)LDW)] = s_W[threadIdx.y + 24][threadIdx.x];
  }
}

void HIP::HPL_dlaswp03T(const int M, const int N, double* A, const int LDA, double* W, const int LDW, const int* LINDXU) {

  if((M <= 0) || (N <= 0)) return;
  hipStream_t stream;
  ROCBLAS_CHECK_STATUS(rocblas_get_stream(_handle, &stream));
  dim3 grid_size((M + TILE_DIM_03T - 1) / TILE_DIM_03T, (N + TILE_DIM_03T - 1) / TILE_DIM_03T);
  dim3 block_size(TILE_DIM_03T, BLOCK_ROWS_03T);
  hipLaunchKernelGGL((_dlaswp03T), grid_size, block_size, 0, stream, M, N, A, LDA, W, LDW, LINDXU);
}

#define TILE_DIM_04T 32
#define BLOCK_ROWS_04T 8

static __global__ void _dlaswp04T(const int M, const int N, double* __restrict__ A, const int LDA, double* __restrict__ W, const int LDW, const int* __restrict__ LINDXU) {

  __shared__ double s_W[TILE_DIM_04T][TILE_DIM_04T + 1];

  const int am = threadIdx.x + TILE_DIM_04T * blockIdx.x;
  const int an = threadIdx.y + TILE_DIM_04T * blockIdx.y;

  const int wm = threadIdx.y + TILE_DIM_04T * blockIdx.x;
  const int wn = threadIdx.x + TILE_DIM_04T * blockIdx.y;

  if(wn < N) {
    s_W[threadIdx.y + 0][threadIdx.x] =
        (wm + 0 < M) ? W[wn + (wm + 0) * ((size_t)LDW)] : 0.0;
    s_W[threadIdx.y + 8][threadIdx.x] =
        (wm + 8 < M) ? W[wn + (wm + 8) * ((size_t)LDW)] : 0.0;
    s_W[threadIdx.y + 16][threadIdx.x] =
        (wm + 16 < M) ? W[wn + (wm + 16) * ((size_t)LDW)] : 0.0;
    s_W[threadIdx.y + 24][threadIdx.x] =
        (wm + 24 < M) ? W[wn + (wm + 24) * ((size_t)LDW)] : 0.0;
  }

  __syncthreads();

  if(am < M) {
    const int aip = LINDXU[am];
    if((an + 0) < N)
      A[aip + (an + 0) * ((size_t)LDA)] = s_W[threadIdx.x][threadIdx.y + 0];
    if((an + 8) < N)
      A[aip + (an + 8) * ((size_t)LDA)] = s_W[threadIdx.x][threadIdx.y + 8];
    if((an + 16) < N)
      A[aip + (an + 16) * ((size_t)LDA)] = s_W[threadIdx.x][threadIdx.y + 16];
    if((an + 24) < N)
      A[aip + (an + 24) * ((size_t)LDA)] = s_W[threadIdx.x][threadIdx.y + 24];
  }
}

void HIP::HPL_dlaswp04T(const int M, const int N, double* A, const int LDA, double* W, const int LDW, const int* LINDXU) {
  if((M <= 0) || (N <= 0)) return;
  hipStream_t stream;
  ROCBLAS_CHECK_STATUS(rocblas_get_stream(_handle, &stream)); 
  dim3 grid_size((M + TILE_DIM_04T - 1) / TILE_DIM_04T, (N + TILE_DIM_04T - 1) / TILE_DIM_04T);
  dim3 block_size(TILE_DIM_04T, BLOCK_ROWS_04T);
  hipLaunchKernelGGL((_dlaswp04T), grid_size, block_size, 0, stream, M, N, A, LDA, W, LDW, LINDXU);
}

__global__ void _dlaswp10N(const int M, const int N, double* __restrict__ A, const int LDA, const int* __restrict__ IPIV) {

  const int m = threadIdx.x + blockDim.x * blockIdx.x;

  if (m < M) { 
    for (int i = 0; i < N; i++) {
      const int ip = IPIV[i];
      if (ip != i) {
        // swap rows
        const double Ai = A[m + i * ((size_t)LDA)];
        const double Aip = A[m + ip * ((size_t)LDA)];
        A[m + i * ((size_t)LDA)] = Aip;
        A[m + ip * ((size_t)LDA)] = Ai;
      }
    }
  }
}

void HIP::HPL_dlaswp10N(const int M, const int N, double* A, const int LDA, const int* IPIV) {
  if((M <= 0) || (N <= 0)) return;

  hipStream_t stream;
  ROCBLAS_CHECK_STATUS(rocblas_get_stream(_handle, &stream));

  const int block_size_10N = 512;

  dim3 grid_size((M + block_size_10N - 1) / block_size_10N);
  hipLaunchKernelGGL((_dlaswp10N), grid_size, dim3(block_size_10N), 0, stream, M, N, A, LDA, IPIV);
}

__global__ void setZero(const int N, double* __restrict__ X) {
  const int t = threadIdx.x, b = blockIdx.x;
  const size_t id = b * blockDim.x + t; // row id

  if(id < N)
    X[id] = 0.0;
}

void HIP::HPL_set_zero(const int N, double* __restrict__ X) {
    const int block_size = 512;
    hipLaunchKernelGGL((setZero), dim3((N + block_size - 1) / block_size), dim3(block_size), 0, HIP::computeStream, N, X);
}



// Setting the matrix section and phase of pdupdate
void HIP::HPL_pdlaswp_hip(HPL_T_panel* PANEL, int icurcol, std::list<PDLASWP_OP> op_list) {
  HPL_T_UPD UPD;
  SWP_PHASE phase;
  for (auto it = op_list.begin(); it != op_list.end(); ++it) {
    const PDLASWP_OP op = *it;
    if (op == SU0 || op == SU1 || op == SU2)  phase = SWP_START;
    else if (op == CU0 || op == CU1 || op == CU2) phase = SWP_COMM;
    else if (op == EU0 || op == EU1 || op == EU2) phase = SWP_END;
    else  phase = SWP_NO;
    
    if (op == SU0 || op == CU0 || op == EU0)  UPD = HPL_LOOK_AHEAD;
    else if (op == SU1 || op == CU1 || op == EU1) UPD = HPL_UPD_1;
    else if (op == SU2 || op == CU2 || op == EU2) UPD = HPL_UPD_2;
    else UPD = HPL_N_UPD;

    if (UPD == HPL_LOOK_AHEAD && PANEL->grid->mycol != icurcol)
      continue;
    else
      HPL_pdlaswp_hip(PANEL, UPD, phase);
  }
}

void HIP::pdlaswp_set_var(HPL_T_panel* PANEL, double* &dU, double* &U, int &ldu, double* &dW, double* &W, int &ldw, int &n, double* &dA, const HPL_T_UPD UPD) {
  switch (UPD) {
  case HPL_LOOK_AHEAD:
    dU = PANEL->dU; U = PANEL->U; ldu = PANEL->ldu0;
    dW = PANEL->dW; W = PANEL->W; ldw = PANEL->ldu0;
    n = PANEL->nu0;
    break;
  case HPL_UPD_1:
    dU = PANEL->dU1;  U = PANEL->U1;  ldu = PANEL->ldu1;
    dW = PANEL->dW1;  W = PANEL->W1;  ldw = PANEL->ldu1;
    n = PANEL->nu1;
    dA = Mptr(dA, 0, PANEL->nu0, PANEL->dlda);
    break;
  case HPL_UPD_2:
    dU = PANEL->dU2;  U = PANEL->U2;  ldu = PANEL->ldu2;
    dW = PANEL->dW2;  W = PANEL->W2;  ldw = PANEL->ldu2;
    n = PANEL->nu2;
    dA = Mptr(dA, 0, PANEL->nu0 + PANEL->nu1, PANEL->dlda);
    break;
  default:
    break;
  }
}

void HIP::panel_init(HPL_T_grid *GRID, HPL_T_palg *ALGO, const int M, const int N, const int JB,
                     HPL_T_pmat *A, const int IA, const int JA, const int TAG, HPL_T_panel *PANEL)
{
  size_t dalign;
  int icurcol, icurrow, ii, itmp1, jj, lwork,
      ml2, mp, mycol, myrow, nb, npcol, nprow,
      nq, nu, ldu;
  /* ..
   * .. Executable Statements ..
   */
  PANEL->grid = GRID; /* ptr to the process grid */
  PANEL->algo = ALGO; /* ptr to the algo parameters */
  PANEL->pmat = A;    /* ptr to the local array info */

  myrow = GRID->myrow;
  mycol = GRID->mycol;
  nprow = GRID->nprow;
  npcol = GRID->npcol;
  nb = A->nb;

  HPL_infog2l(IA, JA, nb, nb, nb, nb, 0, 0, myrow, mycol,
              nprow, npcol, &ii, &jj, &icurrow, &icurcol);
  mp = HPL_numrocI(M, IA, nb, nb, myrow, 0, nprow);
  nq = HPL_numrocI(N, JA, nb, nb, mycol, 0, npcol);

  const int inxtcol = MModAdd1(icurcol, npcol);
  const int inxtrow = MModAdd1(icurrow, nprow);

  /* ptr to trailing part of A */
  PANEL->A = A->A;
  PANEL->dA = Mptr((double *)(A->d_A), ii, jj, A->ld);

  /*
   * Workspace pointers are initialized to NULL.
   */
  PANEL->L2 = nullptr;
  PANEL->dL2 = nullptr;
  PANEL->L1 = nullptr;
  PANEL->dL1 = nullptr;
  PANEL->DINFO = nullptr;
  PANEL->U = nullptr;
  PANEL->dU = nullptr;
  PANEL->W = nullptr;
  PANEL->dW = nullptr;
  PANEL->U1 = nullptr;
  PANEL->dU1 = nullptr;
  PANEL->W1 = nullptr;
  PANEL->dW1 = nullptr;
  PANEL->U2 = nullptr;
  PANEL->dU2 = nullptr;
  PANEL->W2 = nullptr;
  PANEL->dW2 = nullptr;
  /*
   * Local lengths, indexes process coordinates
   */
  PANEL->nb = nb;           /* distribution blocking factor */
  PANEL->jb = JB;           /* panel width */
  PANEL->m = M;             /* global # of rows of trailing part of A */
  PANEL->n = N;             /* global # of cols of trailing part of A */
  PANEL->ia = IA;           /* global row index of trailing part of A */
  PANEL->ja = JA;           /* global col index of trailing part of A */
  PANEL->mp = mp;           /* local # of rows of trailing part of A */
  PANEL->nq = nq;           /* local # of cols of trailing part of A */
  PANEL->ii = ii;           /* local row index of trailing part of A */
  PANEL->jj = jj;           /* local col index of trailing part of A */
  PANEL->lda = Mmax(1, mp); /* local leading dim of array A */
  PANEL->dlda = A->ld;      /* local leading dim of array A */
  PANEL->prow = icurrow;    /* proc row owning 1st row of trailing A */
  PANEL->pcol = icurcol;    /* proc col owning 1st col of trailing A */
  PANEL->msgid = TAG;       /* message id to be used for panel bcast */
  /*
   * Initialize  ldl2 and len to temporary dummy values and Update tag for
   * next panel
   */
  PANEL->ldl2 = 0;  /* local leading dim of array L2 */
  PANEL->dldl2 = 0; /* local leading dim of array L2 */
  PANEL->dldl1 = 1.02 * A->dN; // padding
  PANEL->len = 0;   /* length of the buffer to broadcast */
  PANEL->nu0 = 0;
  PANEL->nu1 = 0;
  PANEL->nu2 = 0;
  PANEL->ldu0 = 0;
  PANEL->ldu1 = 0;
  PANEL->ldu2 = 0;

  
  /*Split fraction*/
  const double fraction = 0.6;

  if ((double)M / A->dN > 0.97) {
    HPL_ptimer_boot();
    HPL_ptimer( 0 );
  }
  dalign = ALGO->align * sizeof(double);
  size_t lpiv = (5 * JB * sizeof(int) + sizeof(double) - 1) / (sizeof(double));

  if (npcol == 1) /* P x 1 process grid */
  {               /* space for L1, DPIV, DINFO */
    lwork = ALGO->align + (PANEL->len = JB * JB + lpiv) + 1;
    nu = Mmax(0, nq - JB);
    ldu = nu + 256; /*extra space for padding*/
    lwork += JB * ldu;

    if (PANEL->max_work_size < (size_t)(lwork) * sizeof(double))
    {
      if (PANEL->WORK)
      {
        HIP_CHECK_ERROR(hipFree(PANEL->dWORK));
        HIP_CHECK_ERROR(hipHostFree(PANEL->WORK));
      }
      size_t numbytes = (size_t)(lwork) * sizeof(double);

      if (hipMalloc((void **)&(PANEL->dWORK), numbytes) != HIP_SUCCESS ||
          hipHostMalloc((void **)&(PANEL->WORK), numbytes, hipHostMallocDefault) != HIP_SUCCESS)
      {
        HPL_pabort(__LINE__, "HPL_pdpanel_init", "Memory allocation failed");
      }
      PANEL->max_work_size = (size_t)(lwork) * sizeof(double);

#ifdef HPL_VERBOSE_PRINT
      if ((myrow == 0) && (mycol == 0))
      {
        printf("Allocating %g GBs of storage on CPU...",
               ((double)numbytes) / (1024 * 1024 * 1024));
        fflush(stdout);

        printf("done.\n");
        printf("Allocating %g GBs of storage on GPU...",
               ((double)numbytes) / (1024 * 1024 * 1024));
        fflush(stdout);
        printf("done.\n");
      }
#endif
    }
    /*
     * Initialize the pointers of the panel structure  -  Always re-use A in
     * the only process column
     */
    PANEL->ldl2 = Mmax(1, mp);
    PANEL->dldl2 = A->ld;
    PANEL->dL2 = PANEL->dA + (myrow == icurrow ? JB : 0);
    PANEL->L2 = PANEL->A + (myrow == icurrow ? JB : 0);
    PANEL->U = (double *)PANEL->WORK;
    PANEL->dU = (double *)PANEL->dWORK;
    PANEL->L1 = (double *)PANEL->WORK + (JB * Mmax(0, ldu));
    PANEL->dL1 = (double *)PANEL->dWORK + (JB * Mmax(0, ldu));
    PANEL->W = A->W;
    PANEL->dW = A->dW;

    if (nprow == 1)
    {
      PANEL->nu0 = Mmin(JB, nu);
      PANEL->ldu0 = PANEL->nu0;

      PANEL->nu1 = 0;
      PANEL->ldu1 = 0;

      PANEL->nu2 = nu - PANEL->nu0;
      PANEL->ldu2 = ((PANEL->nu2 + 95) / 128) * 128 + 32; /*pad*/

      PANEL->U1 = PANEL->U + PANEL->ldu0 * JB;
      PANEL->dU1 = PANEL->dU + PANEL->ldu0 * JB;
      PANEL->U2 = PANEL->U1 + PANEL->ldu1 * JB;
      PANEL->dU2 = PANEL->dU1 + PANEL->ldu1 * JB;

      PANEL->permU = (int *)(PANEL->L1 + JB * JB);
      PANEL->dpermU = (int *)(PANEL->dL1 + JB * JB);
      PANEL->ipiv = PANEL->permU + JB;
      PANEL->dipiv = PANEL->dpermU + JB;

      PANEL->DINFO = (double *)(PANEL->ipiv + 2 * JB);
      PANEL->dDINFO = (double *)(PANEL->dipiv + 2 * JB);
    }
    else
    {
      const int NSplit = Mmax(0, ((((int)(A->nq * fraction)) / nb) * nb));
      PANEL->nu0 = Mmin(JB, nu);
      PANEL->ldu0 = PANEL->nu0;

      PANEL->nu2 = Mmin(nu - PANEL->nu0, NSplit);
      PANEL->ldu2 = ((PANEL->nu2 + 95) / 128) * 128 + 32; /*pad*/

      PANEL->nu1 = nu - PANEL->nu0 - PANEL->nu2;
      PANEL->ldu1 = ((PANEL->nu1 + 95) / 128) * 128 + 32; /*pad*/

      PANEL->U1 = PANEL->U + PANEL->ldu0 * JB;
      PANEL->dU1 = PANEL->dU + PANEL->ldu0 * JB;
      PANEL->U2 = PANEL->U1 + PANEL->ldu1 * JB;
      PANEL->dU2 = PANEL->dU1 + PANEL->ldu1 * JB;

      PANEL->W1 = PANEL->W + PANEL->ldu0 * JB;
      PANEL->dW1 = PANEL->dW + PANEL->ldu0 * JB;
      PANEL->W2 = PANEL->W1 + PANEL->ldu1 * JB;
      PANEL->dW2 = PANEL->dW1 + PANEL->ldu1 * JB;

      PANEL->lindxA = (int *)(PANEL->L1 + JB * JB);
      PANEL->dlindxA = (int *)(PANEL->dL1 + JB * JB);
      PANEL->lindxAU = PANEL->lindxA + JB;
      PANEL->dlindxAU = PANEL->dlindxA + JB;
      PANEL->lindxU = PANEL->lindxAU + JB;
      PANEL->dlindxU = PANEL->dlindxAU + JB;
      PANEL->permU = PANEL->lindxU + JB;
      PANEL->dpermU = PANEL->dlindxU + JB;

      // Put ipiv array at the end
      PANEL->dipiv = PANEL->dpermU + JB;
      PANEL->ipiv = PANEL->permU + JB;

      PANEL->DINFO = ((double *)PANEL->lindxA) + lpiv;
      PANEL->dDINFO = ((double *)PANEL->dlindxA) + lpiv;
    }

    *(PANEL->DINFO) = 0.0;
  }
  else  // for ncol != 1
  { /* space for L2, L1, DPIV */
    ml2 = (myrow == icurrow ? mp - JB : mp);
    ml2 = Mmax(0, ml2);
    ml2 = ((ml2 + 95) / 128) * 128 + 32; /*pad*/
    itmp1 = JB * JB + lpiv;              // L1, integer arrays
    PANEL->len = ml2 * JB + itmp1;

    lwork = ALGO->align + PANEL->len + 1;

    nu = Mmax(0, (mycol == icurcol ? nq - JB : nq));
    ldu = nu + 256; /*extra space for potential padding*/

    // if( nprow > 1 )                                 /* space for U */
    {
      lwork += JB * ldu;
    }
    if (PANEL->max_work_size < (size_t)(lwork) * sizeof(double))
    {
      if (PANEL->WORK)
      {
        HIP_CHECK_ERROR(hipFree(PANEL->dWORK));
        HIP_CHECK_ERROR(hipHostFree(PANEL->WORK));
      }
      size_t numbytes = (size_t)(lwork) * sizeof(double);

      if (hipMalloc((void **)&(PANEL->dWORK), numbytes) != HIP_SUCCESS ||
          hipHostMalloc((void **)&(PANEL->WORK), numbytes, hipHostMallocDefault) != HIP_SUCCESS)
      {
        HPL_pabort(__LINE__, "HPL_pdpanel_init", "Memory allocation failed");
      }
      PANEL->max_work_size = (size_t)(lwork) * sizeof(double);
#ifdef HPL_VERBOSE_PRINT
      if ((myrow == 0) && (mycol == 0))
      {
        printf("Allocating %g GBs of storage on CPU...",
               ((double)numbytes) / (1024 * 1024 * 1024));
        fflush(stdout);
        printf("done.\n");
        printf("Allocating %g GBs of storage on GPU...",
               ((double)numbytes) / (1024 * 1024 * 1024));
        fflush(stdout);
        printf("done.\n");
      }
#endif
    }
    /*
     * Initialize the pointers of the panel structure - Re-use A in the cur-
     * rent process column when HPL_COPY_L is not defined.
     */
    PANEL->U = (double *)PANEL->WORK;
    PANEL->dU = (double *)PANEL->dWORK;

    PANEL->W = A->W;
    PANEL->dW = A->dW;

    PANEL->L2 = (double *)PANEL->WORK + (JB * Mmax(0, ldu));
    PANEL->dL2 = (double *)PANEL->dWORK + (JB * Mmax(0, ldu));
    PANEL->L1 = PANEL->L2 + ml2 * JB;
    PANEL->dL1 = PANEL->dL2 + ml2 * JB;
    PANEL->ldl2 = Mmax(1, ml2);
    PANEL->dldl2 = Mmax(1, ml2);

    if (nprow == 1)
    {
      PANEL->nu0 = (mycol == inxtcol) ? Mmin(JB, nu) : 0;
      PANEL->ldu0 = PANEL->nu0;

      PANEL->nu1 = 0;
      PANEL->ldu1 = 0;

      PANEL->nu2 = nu - PANEL->nu0;
      PANEL->ldu2 = ((PANEL->nu2 + 95) / 128) * 128 + 32; /*pad*/

      PANEL->U1 = PANEL->U + PANEL->ldu0 * JB;
      PANEL->dU1 = PANEL->dU + PANEL->ldu0 * JB;
      PANEL->U2 = PANEL->U1 + PANEL->ldu1 * JB;
      PANEL->dU2 = PANEL->dU1 + PANEL->ldu1 * JB;

      PANEL->permU = (int *)(PANEL->L1 + JB * JB);
      PANEL->dpermU = (int *)(PANEL->dL1 + JB * JB);
      PANEL->ipiv = PANEL->permU + JB;
      PANEL->dipiv = PANEL->dpermU + JB;

      PANEL->DINFO = (double *)(PANEL->ipiv + 2 * JB);
      PANEL->dDINFO = (double *)(PANEL->dipiv + 2 * JB);
    }
    else
    {
      const int NSplit = Mmax(0, ((((int)(A->nq * fraction)) / nb) * nb));
      PANEL->nu0 = (mycol == inxtcol) ? Mmin(JB, nu) : 0;
      PANEL->ldu0 = PANEL->nu0;

      PANEL->nu2 = Mmin(nu - PANEL->nu0, NSplit);
      PANEL->ldu2 = ((PANEL->nu2 + 95) / 128) * 128 + 32; /*pad*/

      PANEL->nu1 = nu - PANEL->nu0 - PANEL->nu2;
      PANEL->ldu1 = ((PANEL->nu1 + 95) / 128) * 128 + 32; /*pad*/

      PANEL->U1 = PANEL->U + PANEL->ldu0 * JB;
      PANEL->dU1 = PANEL->dU + PANEL->ldu0 * JB;
      PANEL->U2 = PANEL->U1 + PANEL->ldu1 * JB;
      PANEL->dU2 = PANEL->dU1 + PANEL->ldu1 * JB;

      PANEL->W1 = PANEL->W + PANEL->ldu0 * JB;
      PANEL->dW1 = PANEL->dW + PANEL->ldu0 * JB;
      PANEL->W2 = PANEL->W1 + PANEL->ldu1 * JB;
      PANEL->dW2 = PANEL->dW1 + PANEL->ldu1 * JB;

      PANEL->lindxA = (int *)(PANEL->L1 + JB * JB);
      PANEL->dlindxA = (int *)(PANEL->dL1 + JB * JB);
      PANEL->lindxAU = PANEL->lindxA + JB;
      PANEL->dlindxAU = PANEL->dlindxA + JB;
      PANEL->lindxU = PANEL->lindxAU + JB;
      PANEL->dlindxU = PANEL->dlindxAU + JB;
      PANEL->permU = PANEL->lindxU + JB;
      PANEL->dpermU = PANEL->dlindxU + JB;

      // Put ipiv array at the end
      PANEL->ipiv = PANEL->permU + JB;
      PANEL->dipiv = PANEL->dpermU + JB;

      PANEL->DINFO = ((double *)PANEL->lindxA) + lpiv;
      PANEL->dDINFO = ((double *)PANEL->dlindxA) + lpiv;
    }

    *(PANEL->DINFO) = 0.0;
  }

  if (nprow == 1)
  {
    lwork = mp + JB;
  }
  else
  {
    itmp1 = (JB << 1);
    lwork = nprow + 1;
    itmp1 = Mmax(itmp1, lwork);
    lwork = mp + 4 + (5 * JB) + (3 * nprow) + itmp1;
  }

  if (PANEL->max_iwork_size < (size_t)(lwork) * sizeof(int))
  {
    if (PANEL->IWORK)
    {
      std::free(PANEL->IWORK);
    }
    size_t numbytes = (size_t)(lwork) * sizeof(int);
    PANEL->IWORK = (int *)std::malloc(numbytes);
    if (PANEL->IWORK == NULL)
    {
      HPL_pabort(__LINE__, "HPL_pdpanel_init", "Panel Host Integer Memory allocation failed");
    }
    PANEL->max_iwork_size = (size_t)(lwork) * sizeof(int);
  }
  if (lwork)
    *(PANEL->IWORK) = -1;

  /* ensure the temp buffer in HPL_pdfact is allocated once*/
  lwork = (size_t)(((4 + ((unsigned int)(JB) << 1)) << 1));
  if (PANEL->max_fwork_size < (size_t)(lwork) * sizeof(double))
  {
    if (PANEL->fWORK)
    {
      HIP_CHECK_ERROR(hipHostFree(PANEL->fWORK));
    }
    size_t numbytes = (size_t)(lwork) * sizeof(double);

    HIP_CHECK_ERROR(hipHostMalloc((void **)&PANEL->fWORK, numbytes));
    if (PANEL->fWORK == NULL)
    {
      HPL_pabort(__LINE__, "HPL_pdpanel_init", "Panel Host pdfact Scratch Memory allocation failed");
    }
    PANEL->max_fwork_size = (size_t)(lwork) * sizeof(double);
  }
  /*
   * End of HPL_pdpanel_init
   */
}

