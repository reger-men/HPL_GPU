
#include <hpl.h>



void HIP::init(size_t num_gpus)
{
    int rank, size, count, namelen; 
    size_t bytes;
    char (*host_names)[MPI_MAX_PROCESSOR_NAME];


    MPI_Comm_rank( MPI_COMM_WORLD, &rank );
    MPI_Comm_size( MPI_COMM_WORLD, &size );
    printf("rank = %d,  size = %d\n", rank, size);
    MPI_Get_processor_name(host_name,&namelen);

    bytes = size * sizeof(char[MPI_MAX_PROCESSOR_NAME]);
    host_names = (char (*)[MPI_MAX_PROCESSOR_NAME])std::malloc(bytes);


    strcpy(host_names[rank], host_name);

    for (int n=0; n < size; n++){
        MPI_Bcast(&(host_names[n]),MPI_MAX_PROCESSOR_NAME, MPI_CHAR, n, MPI_COMM_WORLD);
    }
    int localRank = 0;
    for (int n = 0; n < rank; n++){
        if (!strcmp(host_name, host_names[n])) localRank++;
    }
    int localSize = 0;
    for (int n = 0; n < size; n++){
        if (!strcmp(host_name, host_names[n])) localSize++;
    }


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
    ROCBLAS_CHECK_STATUS(rocblas_create_handle(&small_handle));
    ROCBLAS_CHECK_STATUS(rocblas_create_handle(&large_handle));
    rocblas_set_pointer_mode(_handle, rocblas_pointer_mode_host);
    rocblas_set_pointer_mode(small_handle, rocblas_pointer_mode_host);
    rocblas_set_pointer_mode(large_handle, rocblas_pointer_mode_host); 
    HIP_CHECK_ERROR(hipStreamCreate(&computeStream));
    HIP_CHECK_ERROR(hipStreamCreate(&dataStream));
    HIP_CHECK_ERROR(hipStreamCreate(&pdlaswpStream));
    
    ROCBLAS_CHECK_STATUS(rocblas_set_stream(_handle, computeStream));
    
    HIP_CHECK_ERROR(hipEventCreate(&panelUpdate));
    HIP_CHECK_ERROR(hipEventCreate(&panelCopy));

    HIP_CHECK_ERROR(hipEventCreate(&panelSendToHost));
    HIP_CHECK_ERROR(hipEventCreate(&panelSendToDevice));

    HIP_CHECK_ERROR(hipEventCreate(&pdlaswpStart_1));
    HIP_CHECK_ERROR(hipEventCreate(&pdlaswpStop_1));
    HIP_CHECK_ERROR(hipEventCreate(&pdlaswpStart_2));
    HIP_CHECK_ERROR(hipEventCreate(&pdlaswpStop_2));
    HIP_CHECK_ERROR(hipEventCreate(&pdlaswpStart_3));
    HIP_CHECK_ERROR(hipEventCreate(&pdlaswpStop_3));
    HIP_CHECK_ERROR(hipEventCreate(&dtrsmStart));
    HIP_CHECK_ERROR(hipEventCreate(&dtrsmStop));
    HIP_CHECK_ERROR(hipEventCreate(&dgemmStart));
    HIP_CHECK_ERROR(hipEventCreate(&dgemmStop));

    _memcpyKind[0] = "H2H";
    _memcpyKind[1] = "H2D";
    _memcpyKind[2] = "D2H";
    _memcpyKind[3] = "D2D";
    _memcpyKind[4] = "DEFAULT";
}

void HIP::release()
{
    ROCBLAS_CHECK_STATUS(rocblas_destroy_handle(_handle));
    ROCBLAS_CHECK_STATUS(rocblas_destroy_handle(small_handle));
    ROCBLAS_CHECK_STATUS(rocblas_destroy_handle(large_handle));
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
    p->WORK = NULL;
    p->IWORK = NULL;
    p->IWORK2 = NULL;
    p->fWORK  = NULL;
    HIP::panel_init( GRID, ALGO, M, N, JB, A, IA, JA, TAG, p );
    *PANEL = p;
}

void HIP::panel_init(HPL_T_grid *GRID, HPL_T_palg *ALGO, const int M, const int N, const int JB, 
                        HPL_T_pmat *A, const int IA, const int JA, const int TAG, HPL_T_panel *PANEL)
{
    size_t                     dalign;
    int                        icurcol, icurrow, ii, itmp1, jj, lwork,
                                ml2, mp, mycol, myrow, nb, npcol, nprow,
                                nq, nu, ldu;
    /* ..
    * .. Executable Statements ..
    */
    PANEL->grid    = GRID;                  /* ptr to the process grid */
    PANEL->algo    = ALGO;               /* ptr to the algo parameters */
    PANEL->pmat    = A;                 /* ptr to the local array info */

    myrow = GRID->myrow; mycol = GRID->mycol;
    nprow = GRID->nprow; npcol = GRID->npcol; nb = A->nb;

    HPL_infog2l( IA, JA, nb, nb, nb, nb, 0, 0, myrow, mycol,
                    nprow, npcol, &ii, &jj, &icurrow, &icurcol );
    mp = HPL_numrocI( M, IA, nb, nb, myrow, 0, nprow );
    nq = HPL_numrocI( N, JA, nb, nb, mycol, 0, npcol );
                                            /* ptr to trailing part of A */

    PANEL->A       = Mptr( (double *)(A->A), ii, jj, A->ld );
    PANEL->dA      = Mptr( (double *)(A->d_A), ii, jj, A->ld );

    /*
    * Workspace pointers are initialized to NULL.
    */
    // PANEL->WORK    = NULL;
    PANEL->L2      = NULL;
    PANEL->dL2     = NULL;
    PANEL->L1      = NULL;
    PANEL->dL1     = NULL;
    PANEL->DPIV    = NULL;
    PANEL->DINFO   = NULL;
    PANEL->U       = NULL;
    PANEL->dU      = NULL;
    // PANEL->IWORK   = NULL;
    /*
    * Local lengths, indexes process coordinates
    */
    PANEL->nb      = nb;               /* distribution blocking factor */
    PANEL->jb      = JB;                                /* panel width */
    PANEL->m       = M;      /* global # of rows of trailing part of A */
    PANEL->n       = N;      /* global # of cols of trailing part of A */
    PANEL->ia      = IA;     /* global row index of trailing part of A */
    PANEL->ja      = JA;     /* global col index of trailing part of A */
    PANEL->mp      = mp;      /* local # of rows of trailing part of A */
    PANEL->nq      = nq;      /* local # of cols of trailing part of A */
    PANEL->ii      = ii;      /* local row index of trailing part of A */
    PANEL->jj      = jj;      /* local col index of trailing part of A */
    PANEL->lda     = A->ld;            /* local leading dim of array A */
    PANEL->prow    = icurrow; /* proc row owning 1st row of trailing A */
    PANEL->pcol    = icurcol; /* proc col owning 1st col of trailing A */
    PANEL->msgid   = TAG;     /* message id to be used for panel bcast */
    /*
    * Initialize  ldl2 and len to temporary dummy values and Update tag for
    * next panel
    */
    PANEL->ldl2    = 0;               /* local leading dim of array L2 */
    PANEL->len     = 0;           /* length of the buffer to broadcast */
    /*
    * Figure out the exact amount of workspace  needed by the factorization
    * and the update - Allocate that space - Finish the panel data structu-
    * re initialization.
    *
    * L1:    JB x JB in all processes
    * DPIV:  JB      in all processes
    * DINFO: 1       in all processes
    *
    * We make sure that those three arrays are contiguous in memory for the
    * later panel broadcast.  We  also  choose  to put this amount of space
    * right  after  L2 (when it exist) so that one can receive a contiguous
    * buffer.
    */
    dalign = ALGO->align * sizeof( double );
    size_t lpiv = (6 * JB * sizeof(int) + sizeof(double) - 1) / (sizeof(double));
    size_t ipivlen = (JB * sizeof(int) + sizeof(double) - 1) / (sizeof(double));

    if( npcol == 1 )                             /* P x 1 process grid */
    {                                     /* space for L1, DPIV, DINFO */
        lwork = ALGO->align + ( PANEL->len = JB * JB + JB + 1 );
        // if( nprow > 1 )                                 /* space for U */
        { nu = nq - JB; ldu = nu + 256; lwork += JB * Mmax( 0, nu ); }
        

        if(PANEL->max_work_size<(size_t)(lwork) * sizeof( double ))
        {
            if( PANEL->WORK  )
            {
            hipFree( PANEL->dWORK);
            hipHostFree( PANEL->WORK);
            }
            // size_t numbytes = (((size_t)((size_t)(lwork) * sizeof( double )) + (size_t)4095)/(size_t)4096)*(size_t)4096;
            size_t numbytes = (size_t)(lwork) *sizeof( double );


            if(hipMalloc((void**)&(PANEL->dWORK),numbytes)!=HIP_SUCCESS ||
            hipHostMalloc((void**)&(PANEL->WORK),numbytes, hipHostMallocDefault)!=HIP_SUCCESS)
            {
                HPL_pabort( __LINE__, "HPL_pdpanel_init",
                            "Memory allocation failed" );
            }
            PANEL->max_work_size = (size_t)(lwork) * sizeof( double );
        }

    /*
    * Initialize the pointers of the panel structure  -  Always re-use A in
    * the only process column
    */
        PANEL->ldl2  = A->ld;
        PANEL->dL2   = PANEL->dA + ( myrow == icurrow ? JB : 0 );
        PANEL->L2    = PANEL->A + ( myrow == icurrow ? JB : 0 );
        PANEL->dL1   = (double *)HPL_PTR( PANEL->dWORK, dalign );
        PANEL->dDPIV = (double *)HPL_PTR( PANEL->dWORK, dalign ) + JB * JB;
        PANEL->L1    = (double *)HPL_PTR( PANEL->WORK, dalign );

        PANEL->dlindxA  = (int*)(PANEL->dL1 + JB * JB);
        PANEL->lindxA   = (int*)(PANEL->L1 + JB * JB);
        PANEL->dlindxAU = PANEL->dlindxA + 2 * JB;
        PANEL->lindxAU  = PANEL->lindxA + 2 * JB;
        PANEL->dpermU   = PANEL->dlindxAU + 2 * JB;
        PANEL->permU    = PANEL->lindxAU + 2 * JB;

        // Put ipiv array at the end
        PANEL->dipiv = PANEL->dpermU + JB;
        PANEL->ipiv  = PANEL->permU + JB;

        PANEL->DINFO  = ((double*)PANEL->lindxA) + lpiv + ipivlen;
        PANEL->dDINFO = ((double*)PANEL->dlindxA) + lpiv + ipivlen;
        *(PANEL->DINFO) = 0.0;
        PANEL->U     = (  PANEL->DINFO + 1);
        PANEL->dU    = ( PANEL->dDINFO + 1);
    }
    else
    {                                        /* space for L2, L1, DPIV */
        ml2 = ( myrow == icurrow ? mp - JB : mp ); ml2 = Mmax( 0, ml2 );
        PANEL->len = ml2*JB + ( itmp1 = JB*JB + lpiv + ipivlen ); 
    #ifdef HPL_COPY_L
        lwork = ALGO->align + PANEL->len + 1;
    #else
        lwork = ALGO->align + ( mycol == icurcol ? itmp1 : PANEL->len ) + 1;
    #endif

        // if( nprow > 1 )                                 /* space for U */
        {
            nu = ( mycol == icurcol ? nq - JB : nq );
            lwork += JB * Mmax( 0, nu );
        }
        if(PANEL->max_work_size<(size_t)(lwork) * sizeof( double ))
        {
            if( PANEL->WORK  )
            {
            hipFree( PANEL->dWORK);
            hipHostFree( PANEL->WORK);
            }
            // size_t numbytes = (((size_t)((size_t)(lwork) * sizeof( double )) + (size_t)4095)/(size_t)4096)*(size_t)4096;
            size_t numbytes = (size_t)(lwork) *sizeof( double );


            if(hipMalloc((void**)&(PANEL->dWORK),numbytes)!=HIP_SUCCESS ||
            hipHostMalloc((void**)&(PANEL->WORK),numbytes, hipHostMallocDefault)!=HIP_SUCCESS)
            {
                HPL_pabort( __LINE__, "HPL_pdpanel_init",
                            "Memory allocation failed" );
            }
            PANEL->max_work_size = (size_t)(lwork) * sizeof( double );
        }
    
    /*
    * Initialize the pointers of the panel structure - Re-use A in the cur-
    * rent process column when HPL_COPY_L is not defined.
    */
    #ifdef HPL_COPY_L
        PANEL->dL2   = (double *)HPL_PTR( PANEL->dWORK, dalign );
        PANEL->dL1   = PANEL->dL2 + ml2 * JB;
        PANEL->L2    = (double *)HPL_PTR( PANEL->WORK, dalign );
        PANEL->L1    = PANEL->L2 + ml2 * JB;
        PANEL->ldl2  = Mmax( 1, ml2 );
    #else
        if( mycol == icurcol )
        {
            PANEL->L2   = PANEL->A + ( myrow == icurrow ? JB : 0 );
            PANEL->dL2  = PANEL->dA + ( myrow == icurrow ? JB : 0 );
            PANEL->ldl2 = A->ld;
            PANEL->L1   = (double *)HPL_PTR( PANEL->WORK, dalign );
            PANEL->dL1   = (double *)HPL_PTR( PANEL->dWORK, dalign );
        }
        else
        {
            PANEL->dL2   = (double *)HPL_PTR( PANEL->dWORK, dalign );
            PANEL->dL1   = PANEL->dL2 + ml2 * JB;

            PANEL->L2   = (double *)HPL_PTR( PANEL->WORK, dalign );
            PANEL->L1   = PANEL->L2 + ml2 * JB;
            PANEL->ldl2 = Mmax( 1, ml2 );
        }
    #endif
        PANEL->dlindxA  = (int*)(PANEL->dL1 + JB * JB);
        PANEL->lindxA   = (int*)(PANEL->L1 + JB * JB);
        PANEL->dlindxAU = PANEL->dlindxA + 2 * JB;
        PANEL->lindxAU  = PANEL->lindxA + 2 * JB;
        PANEL->dpermU   = PANEL->dlindxAU + 2 * JB;
        PANEL->permU    = PANEL->lindxAU + 2 * JB;

        PANEL->dipiv = PANEL->dpermU + JB;
        PANEL->ipiv  = PANEL->permU + JB;

        PANEL->DINFO  = ((double*)PANEL->lindxA) + lpiv + ipivlen;
        PANEL->dDINFO = ((double*)PANEL->dlindxA) + lpiv + ipivlen;
        *(PANEL->DINFO) = 0.0;
        PANEL->U     = (  PANEL->DINFO + 1  );
        PANEL->dU    = ( PANEL->dDINFO + 1 );
    }


    if( nprow == 1 ) { lwork = mp + JB; }
    else
    {
        itmp1 = (JB << 1); lwork = nprow + 1; itmp1 = Mmax( itmp1, lwork );
        lwork = mp + 4 + (5 * JB) + (3 * nprow) + itmp1;
    }


        if(PANEL->max_iwork_size<(size_t)(lwork) * sizeof( int ))
        {
        if( PANEL->IWORK  )
        {
            hipFree( PANEL->dIWORK);
            hipHostFree( PANEL->IWORK);
        }
        // size_t numbytes = (((size_t)((size_t)(lwork) * sizeof( double )) + (size_t)4095)/(size_t)4096)*(size_t)4096;
        size_t numbytes = (size_t)(lwork) *sizeof( int );

        if(hipMalloc((void**)&(PANEL->dIWORK),numbytes)!=HIP_SUCCESS ||
            hipHostMalloc((void**)&(PANEL->IWORK),numbytes, hipHostMallocDefault)!=HIP_SUCCESS)
        {
            HPL_pabort( __LINE__, "HPL_pdpanel_init",
                        "Memory allocation failed" );
        }
        PANEL->max_iwork_size = (size_t)(lwork) * sizeof( int );

        if (PANEL->IWORK2)
            std::free(PANEL->IWORK2);

        PANEL->IWORK2 = (int *)std::malloc( (size_t)(mp) * sizeof( int ) );
        }
    

    if (lwork)
        *(PANEL->IWORK) = -1;

    /* ensure the temp buffer in HPL_pdfact is allocated once*/
    lwork = (size_t)(PANEL->algo->align) + (size_t)(((4+((unsigned int)(PANEL->jb) << 1)) << 1) );
    if(PANEL->max_fwork_size < (size_t)(lwork) * sizeof(double)) {
        if(PANEL->fWORK) { hipHostFree(PANEL->fWORK); }
        size_t numbytes = (size_t)(lwork) * sizeof(double);

        hipHostMalloc((void**)&PANEL->fWORK, numbytes);
        if(PANEL->fWORK == NULL) {
        HPL_pabort(__LINE__,
                    "HPL_pdpanel_init",
                    "Panel Host pdfact Scratch Memory allocation failed");
        }
        PANEL->max_fwork_size = (size_t)(lwork) * sizeof(double);
    }
    /*
    * End of HPL_pdpanel_init
    */    
}

void HIP::panel_send_to_host(HPL_T_panel *PANEL)
{
    int jb = PANEL->jb;
    
    if( ( PANEL->grid->mycol != PANEL->pcol ) || ( jb <= 0 ) ) return;
    hipMemcpy2DAsync(PANEL->A,  PANEL->lda*sizeof(double),
                    PANEL->dA, PANEL->lda*sizeof(double),
                    PANEL->mp*sizeof(double), jb,
                    hipMemcpyDeviceToHost, dataStream);
}

void HIP::panel_send_to_device(HPL_T_panel *PANEL)
{
    double *A, *dA;
    int jb, i, ml2;
    static int equil = -1;
    /* ..
     * .. Executable Statements ..
     */
    jb = PANEL->jb;

    if (jb <= 0)
        return;

    // copy A and/or L2
    if (PANEL->grid->mycol == PANEL->pcol)
    { // L2 reuses A
        if (PANEL->grid->npcol > 1) {
            if (PANEL->grid->myrow == PANEL->prow) {
                hipMemcpy2DAsync(Mptr(PANEL->dA, 0, -jb, PANEL->lda), PANEL->lda * sizeof(double),
                        Mptr(PANEL->A, 0, 0, PANEL->lda), PANEL->lda * sizeof(double),
                        jb * sizeof(double), jb,
                        hipMemcpyHostToDevice, dataStream);

            if((PANEL->mp - jb) > 0)
                    hipMemcpy2DAsync(PANEL->dL2, PANEL->ldl2 * sizeof(double),
                            Mptr(PANEL->A, jb, 0, PANEL->lda), PANEL->lda * sizeof(double),
                            (PANEL->mp - jb) * sizeof(double), jb,
                            hipMemcpyHostToDevice, dataStream);
        }
        else {
            if(PANEL->mp > 0)
                    hipMemcpy2DAsync(PANEL->dL2, PANEL->ldl2 * sizeof(double),
                           Mptr(PANEL->A, 0, 0, PANEL->lda), PANEL->lda * sizeof(double),
                           PANEL->mp * sizeof(double), jb,
                           hipMemcpyHostToDevice, dataStream);
        }
    }
        else {
            if(PANEL->mp > 0)
                hipMemcpy2DAsync(Mptr(PANEL->dA, 0, -jb, PANEL->lda), PANEL->lda * sizeof(double),
                        Mptr(PANEL->A, 0, 0, PANEL->lda), PANEL->lda * sizeof(double),
                        PANEL->mp * sizeof(double), jb,
                        hipMemcpyHostToDevice, dataStream);
        }
        // copy L1
        hipMemcpy2DAsync(PANEL->dL1, jb * sizeof(double),
                     PANEL->L1, jb * sizeof(double),
                     jb * sizeof(double), jb,
                     hipMemcpyHostToDevice, dataStream);
    }
    
    if (PANEL->grid->mycol == PANEL->pcol) {
        if (PANEL->grid->nprow == 1) {
        // unroll pivoting and send to device
        int* ipiv    = PANEL->ipiv;
        int* ipiv_ex = PANEL->ipiv + jb;
        int* dipiv    = PANEL->dipiv;
        int* dipiv_ex = PANEL->dipiv + jb;

        int *upiv = PANEL->IWORK2;

        for (i = 0; i < jb; i++)
        {
            ipiv[i] = (int)(PANEL->ipiv[i]) - PANEL->ii;
        } // shift
        for (i = 0; i < PANEL->mp; i++)
        {
            upiv[i] = i;
        } // initialize ids
        for (i = 0; i < jb; i++)
        { // swap ids
            int id = upiv[i];
            upiv[i] = upiv[ipiv[i]];
            upiv[ipiv[i]] = id;
        }

        for (i = 0; i < jb; i++)
        {
            ipiv_ex[i] = -1;
        }

        int cnt = 0;
        for (i = jb; i < PANEL->mp; i++)
        { // find swapped ids outside of panel
            if (upiv[i] < jb)
            {
                ipiv_ex[upiv[i]] = i;
            }
        }

        hipMemcpy2DAsync(dipiv, jb * sizeof(int),
                        upiv, jb * sizeof(int),
                        jb * sizeof(int), 1,
                        hipMemcpyHostToDevice, dataStream);
        hipMemcpy2DAsync(dipiv_ex, jb * sizeof(int),
                        ipiv_ex, jb * sizeof(int),
                        jb * sizeof(int), 1,
                        hipMemcpyHostToDevice, dataStream);
        }
        else {
            if(equil == -1) equil = PANEL->algo->equil;

            // to broadcast row-swapping information between column processes
            int  k       = (int)((unsigned int)(jb) << 1);
            int* iflag   = PANEL->IWORK;
            int* ipl     = iflag + 1;
            int* ipID    = ipl + 1;
            int* ipA     = ipID + ((unsigned int)(k) << 1);
            int* iplen   = ipA + 1;
            int* ipmap   = iplen + PANEL->grid->nprow + 1;
            int* ipmapm1 = ipmap + PANEL->grid->nprow;
            int* upiv    = ipmapm1 + PANEL->grid->nprow;
            int* iwork   = upiv + PANEL->mp;

            int* lindxA   = PANEL->lindxA;
            int* lindxAU  = PANEL->lindxAU;
            int* permU    = PANEL->permU;
            int* permU_ex = permU + jb;
            int* ipiv     = PANEL->ipiv;

            int* dlindxA   = PANEL->dlindxA;
            int* dlindxAU  = PANEL->dlindxAU;
            int* dpermU    = PANEL->dpermU;
            int* dpermU_ex = dpermU + jb;
            int* dipiv     = PANEL->dipiv;

            if(*iflag == -1) /* no index arrays have been computed so far */
            {
                HPL_pipid(PANEL, ipl, ipID);
                HPL_plindx1(PANEL, *ipl, ipID, ipA, lindxA, lindxAU, iplen, ipmap, ipmapm1, permU, iwork);
                *iflag = 1;
            } else if(*iflag == 0) /* HPL_pdlaswp00N called before: reuse ipID */
            {
                HPL_plindx1(PANEL, *ipl, ipID, ipA, lindxA, lindxAU, iplen, ipmap, ipmapm1, permU, iwork);
                *iflag = 1;
            } else if((*iflag == 1) && (equil != 0)) { /* HPL_pdlaswp01N was call before only re-compute IPLEN, IPMAP */
                HPL_plindx10(PANEL, *ipl, ipID, iplen, ipmap, ipmapm1);
                *iflag = 1;
            }

            int N = Mmax(*ipA, jb);
            if(N > 0) {
                hipMemcpy2DAsync(dlindxA, k * sizeof(int), lindxA, k * sizeof(int), N * sizeof(int), 1, hipMemcpyHostToDevice, dataStream);
                hipMemcpy2DAsync(dlindxAU, k * sizeof(int), lindxAU, k * sizeof(int), N * sizeof(int), 1, hipMemcpyHostToDevice, dataStream);
            }

            hipMemcpy2DAsync(dpermU, jb * sizeof(int), permU, jb * sizeof(int), jb * sizeof(int), 1, hipMemcpyHostToDevice, dataStream);
            hipMemcpy2DAsync(dipiv, jb * sizeof(int), ipiv, jb * sizeof(int), jb * sizeof(int), 1, hipMemcpyHostToDevice, dataStream);
        }
    }
}

int HIP::panel_free(HPL_T_panel *PANEL)
{
    GPUInfo("%-40s \t%-5s", "[Deallocate]", "Panel resources", "HIP");
    if (PANEL->free_work_now == 1)
    {
        if (PANEL->WORK)
        {
            HIP_CHECK_ERROR(hipFree(PANEL->dWORK));
            HIP_CHECK_ERROR(hipHostFree(PANEL->WORK));
            PANEL->max_work_size = 0;
        }
        if (PANEL->IWORK)
        {
            HIP_CHECK_ERROR(hipFree(PANEL->dIWORK));
            HIP_CHECK_ERROR(hipHostFree(PANEL->IWORK));
            PANEL->max_iwork_size = 0;
        }
        if(PANEL->fWORK)
        {
            HIP_CHECK_ERROR(hipHostFree(PANEL->fWORK));
            PANEL->max_fwork_size = 0;
        }
    }
    return (MPI_SUCCESS);
}

int HIP::panel_disp(HPL_T_panel **PANEL)
{
    GPUInfo("%-40s \t%-5s", "[Deallocate]", "Panel structure", "HIP");
    int err = HIP::panel_free(*PANEL);
    (*PANEL)->free_work_now = 1;
    // if(*ptr) HIP_CHECK_ERROR(hipFree( ptr ));
    if (*PANEL) free(*PANEL);
    *PANEL = NULL;
    return( err );
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
    hipDeviceSynchronize();

    rocrand_destroy_generator(generator);
}

void HIP::event_record(enum HPL_EVENT _event){
    switch (_event)
    {
    case HPL_PANEL_COPY:
    HIP_CHECK_ERROR(hipEventRecord(panelCopy, dataStream));
    break;

    case HPL_PANEL_UPDATE:
    HIP_CHECK_ERROR(hipEventRecord(panelUpdate, computeStream));
    break;

    case HPL_RS_1:
    HIP_CHECK_ERROR(hipEventRecord(pdlaswpStop_1, pdlaswpStream));
    break;

    case HPL_RS_2:
    HIP_CHECK_ERROR(hipEventRecord(pdlaswpStop_2, pdlaswpStream));
    break;

    case HPL_RS_3:
    HIP_CHECK_ERROR(hipEventRecord(pdlaswpStop_3, pdlaswpStream));
    break;

    default:
    break;
    }
}

void HIP::event_synchronize(enum HPL_EVENT _event){
    switch (_event)
    {
    case HPL_PANEL_COPY:
    HIP_CHECK_ERROR(hipEventSynchronize(panelCopy));        
    break;

    case HPL_PANEL_UPDATE:
    HIP_CHECK_ERROR(hipEventSynchronize(panelUpdate));
    break;

    case HPL_RS_1:
    HIP_CHECK_ERROR(hipEventSynchronize(pdlaswpStop_1));
    break;

    case HPL_RS_2:
    HIP_CHECK_ERROR(hipEventSynchronize(pdlaswpStop_2));
    break;

    case HPL_RS_3:
    HIP_CHECK_ERROR(hipEventSynchronize(pdlaswpStop_3));
    break;
    
    default:
    break;
    }
   
}

void HIP::stream_synchronize(enum HPL_STREAM _stream){
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
    HIP_CHECK_ERROR(hipStreamWaitEvent(computeStream, pdlaswpStop_1, 0));        
    break;

    case HPL_RS_2:
    HIP_CHECK_ERROR(hipStreamWaitEvent(computeStream, pdlaswpStop_2, 0));        
    break;

    case HPL_RS_3:
    HIP_CHECK_ERROR(hipStreamWaitEvent(computeStream, pdlaswpStop_3, 0));        
    break;

    default:
    break;
    }
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
    hipEventRecord(dgemmStart, computeStream);
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

__global__ void 
_dlatcpy(const int M, const int N, const double* __restrict__ A, const int LDA,
         double* __restrict__ B, const int LDB) 
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
void HIP::atcpy(const int M, const int N, const double *A, const int LDA,
                double *B, const int LDB)
{
    GPUInfo("%-25s %-8d%-8d \t%-5s", "[LATCOPY]", "With A of (R:C)", M, N, "HIP");
    hipStream_t stream;
    rocblas_get_stream(_handle, &stream);
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

#define BLOCK_SIZE 512

__global__ void _dlaswp00N(const int N, const int M,
                     double* __restrict__ A,
                     const int LDA,
                     const int* __restrict__ IPIV) {
                    //  const int* IPIV) {

   __shared__ double s_An_init[512];
   __shared__ double s_An_ipiv[512];

   const int m = threadIdx.x;
   const int n = blockIdx.x;

   //read in block column
   for (int i=m;i<M;i+=blockDim.x)
      s_An_init[i] = A[i+n*((size_t)LDA)];

   __syncthreads();

   //local block
   for (int i=m;i<M;i+=blockDim.x) {
      const int ip = IPIV[i];

      if (ip<M) { //local swap
         s_An_ipiv[i] = s_An_init[ip];
      } else { //non local swap
         s_An_ipiv[i] = A[ip+n*((size_t)LDA)];
      }
   }
   __syncthreads();

   //write out local block
   for (int i=m;i<M;i+=blockDim.x)
      A[i+n*((size_t)LDA)] = s_An_ipiv[i];

   //remaining swaps in column
   for (int i=m;i<M;i+=blockDim.x) {
      const int ip_ex = IPIV[i+M];

      if (ip_ex>-1) {
         A[ip_ex+n*((size_t)LDA)] = s_An_init[i];
      }
   }
}

void HIP::dlaswp00N(const int M, const int N, double * A, const int LDA, const int * IPIV)
{
    GPUInfo("%-25s %-8d%-8d \t%-5s", "[DLASWP00N]", "With A of (R:C)", M, N, "HIP");
    hipStream_t stream;
    rocblas_get_stream(_handle, &stream);

    hipEvent_t start, stop;
    float elapsedTime = 0.f; 

    const int block_size = 512, grid_size = N;
    
    hipLaunchKernelGGL(_dlaswp00N, dim3(grid_size), dim3(block_size), 0, stream,
                                      N, M, A, LDA, IPIV);

}

void HIP::device_sync() {
    HIP_CHECK_ERROR(hipDeviceSynchronize());
}


void HIP::pdlaswp(HPL_T_panel *PANEL, const int NN){
    double * Aptr, * L1ptr, * L2ptr, * Uptr, * dpiv;
    int * ipiv;
    int i, jb, lda, mp, n, nb, nq0, nn;
    nb = PANEL->nb;
    jb = PANEL->jb;
    n = PANEL->nq; 
    lda = PANEL->lda;
    if( NN >= 0 ) n = Mmin( NN, n );
    Aptr = PANEL->dA;       L2ptr = PANEL->dL2;   L1ptr = PANEL->dL1;
    dpiv  = PANEL->DPIV;    ipiv  = PANEL->dipiv;
    mp   = PANEL->mp - jb;  nq0   = 0;       nn = n - nq0;

    const int block_size = 512, grid_size = nn;
    hipStreamWaitEvent(pdlaswpStream, dgemmStart, 0);
    hipLaunchKernelGGL(_dlaswp00N, dim3(grid_size), dim3(block_size), 0, pdlaswpStream,
                                      nn, jb, Aptr, lda, ipiv);
}

void HIP::binit_ibcst(HPL_T_panel* PANEL, int &result) {

    result = HPL_SUCCESS;
}

#define _M_BUFF (void*)(PANEL->dL2)
#define _M_COUNT PANEL->len
#define _M_TYPE MPI_DOUBLE

static MPI_Request request  = MPI_REQUEST_NULL;
static MPI_Request request2 = MPI_REQUEST_NULL;

void HIP::bcast_ibcst(HPL_T_panel* PANEL, int* IFLAG, int &result) {
  MPI_Comm comm;
  int      ierr, ierr2, go, next, msgid, prev, rank, root, size;

  if(PANEL == NULL) {
    *IFLAG = HPL_SUCCESS;
    result = HPL_SUCCESS;
    return;
  }
  if((size = PANEL->grid->npcol) <= 1) {
    *IFLAG = HPL_SUCCESS;
    result = HPL_SUCCESS;
    return;
  }

  rank  = PANEL->grid->mycol;
  comm  = PANEL->grid->row_comm;
  root  = PANEL->pcol;
  msgid = PANEL->msgid;

  ierr  = MPI_Ibcast(_M_BUFF, _M_COUNT, _M_TYPE, root, comm, &request);
  /*
   * If the message was received and being forwarded,  return HPL_SUCCESS.
   * If an error occured in an MPI call, return HPL_FAILURE.
   */
  *IFLAG = (ierr == MPI_SUCCESS ? HPL_SUCCESS : HPL_FAILURE);

    result = *IFLAG;
}

void HIP::bwait_ibcst(HPL_T_panel* PANEL, int &result) {
  int ierr1, ierr2;

  if(PANEL == NULL) { result = HPL_SUCCESS; return; }
  if(PANEL->grid->npcol <= 1) { result = HPL_SUCCESS; return; }

  ierr1 = MPI_Wait(&request, MPI_STATUS_IGNORE);

  result = (ierr1 == MPI_SUCCESS? HPL_SUCCESS : HPL_FAILURE);
}

#define TILE_DIM_01T 32
#define BLOCK_ROWS_01T 8

/* Build U matrix from rows of A */
__global__ void dlaswp01T_1(const int M, const int N,
                            double* __restrict__ A, const int LDA,
                            double* __restrict__ U, const int LDU,
                            const int* __restrict__ LINDXA, const int* __restrict__ LINDXAU) {

  __shared__ double s_U[TILE_DIM_01T][TILE_DIM_01T + 1];

  const int m = threadIdx.x + TILE_DIM_01T * blockIdx.x;
  const int n = threadIdx.y + TILE_DIM_01T * blockIdx.y;

  if(m < M) {
    const int ipa  = LINDXA[m];
    const int ipau = LINDXAU[m];

    if(ipau >= 0) { // row will swap into U
      // save in LDS for the moment
      // possible cache-hits if ipas are close
      s_U[threadIdx.x][threadIdx.y + 0] =
          (n + 0 < N) ? A[ipa + (n + 0) * ((size_t)LDA)] : 0.0;
      s_U[threadIdx.x][threadIdx.y + 8] =
          (n + 8 < N) ? A[ipa + (n + 8) * ((size_t)LDA)] : 0.0;
      s_U[threadIdx.x][threadIdx.y + 16] =
          (n + 16 < N) ? A[ipa + (n + 16) * ((size_t)LDA)] : 0.0;
      s_U[threadIdx.x][threadIdx.y + 24] =
          (n + 24 < N) ? A[ipa + (n + 24) * ((size_t)LDA)] : 0.0;
    }
  }

  __syncthreads();

  const int um = threadIdx.y + TILE_DIM_01T * blockIdx.x;
  const int un = threadIdx.x + TILE_DIM_01T * blockIdx.y;

  if(un < N) {
    const int uipau0 = (um + 0 < M) ? LINDXAU[um + 0] : -1;
    const int uipau1 = (um + 8 < M) ? LINDXAU[um + 8] : -1;
    const int uipau2 = (um + 16 < M) ? LINDXAU[um + 16] : -1;
    const int uipau3 = (um + 24 < M) ? LINDXAU[um + 24] : -1;

    // write out chunks of U
    if(uipau0 >= 0)
      U[un + uipau0 * ((size_t)LDU)] = s_U[threadIdx.y + 0][threadIdx.x];
    if(uipau1 >= 0)
      U[un + uipau1 * ((size_t)LDU)] = s_U[threadIdx.y + 8][threadIdx.x];
    if(uipau2 >= 0)
      U[un + uipau2 * ((size_t)LDU)] = s_U[threadIdx.y + 16][threadIdx.x];
    if(uipau3 >= 0)
      U[un + uipau3 * ((size_t)LDU)] = s_U[threadIdx.y + 24][threadIdx.x];
  }
}

#define BLOCK_SIZE_01T 1024

/* Perform any local row swaps of A */
__global__ void dlaswp01T_2(const int M, const int N,
                            double* __restrict__ A, const int LDA,
                            const int* __restrict__ LINDXA, const int* __restrict__ LINDXAU) {

  __shared__ double s_A[BLOCK_SIZE_01T];

  const int n = blockIdx.x;
  const int m = threadIdx.x;

  int ipau, ipa;

  if(m < M) {
    ipau = LINDXAU[m];
    ipa  = LINDXA[m];

    // read in
    s_A[m] = (ipau < 0) ? A[ipa + n * ((size_t)LDA)] : 0.0;
  }
  __syncthreads();

  if(m < M) {
    if(ipau < 0) { // swap into A
      A[-ipau + n * ((size_t)LDA)] = s_A[m];
    }
  }
}

void HIP::dlaswp01T(const int M, const int N, double* A, const int LDA, double* U, const int LDU, const int* LINDXA, const int* LINDXAU) {

    if((M <= 0) || (N <= 0)) return;
        
    dim3 grid_size((M + TILE_DIM_01T - 1) / TILE_DIM_01T, (N + TILE_DIM_01T - 1) / TILE_DIM_01T);
    dim3 block_size(TILE_DIM_01T, BLOCK_ROWS_01T);
    hipLaunchKernelGGL((dlaswp01T_1), grid_size, block_size, 0, pdlaswpStream,
                        M, N, A, LDA, U, LDU, LINDXA, LINDXAU);
    assert(((void)"NB too large in HPL_dlaswp01T", M <= BLOCK_SIZE_01T));
    hipLaunchKernelGGL(
        (dlaswp01T_2), N, M, 0, pdlaswpStream, M, N, A, LDA, LINDXA, LINDXAU);
}

#define TILE_DIM_06T 32
#define BLOCK_ROWS_06T 8

__global__ void dlaswp06T_kernel(const int M, const int N,
                          double* __restrict__ A, const int LDA, 
                          double* __restrict__ U, const int LDU,
                          const int* __restrict__ LINDXA) {

  __shared__ double s_U[TILE_DIM_06T][TILE_DIM_06T + 1];
  __shared__ double s_A[TILE_DIM_06T][TILE_DIM_06T + 1];

  const int am = threadIdx.x + TILE_DIM_06T * blockIdx.x;
  const int an = threadIdx.y + TILE_DIM_06T * blockIdx.y;

  const int um = threadIdx.y + TILE_DIM_06T * blockIdx.x;
  const int un = threadIdx.x + TILE_DIM_06T * blockIdx.y;

  int aip;

  if(am < M) {
    aip = LINDXA[am];
    s_A[threadIdx.x][threadIdx.y + 0] =
        (an + 0 < N) ? A[aip + (an + 0) * ((size_t)LDA)] : 0.0;
    s_A[threadIdx.x][threadIdx.y + 8] =
        (an + 8 < N) ? A[aip + (an + 8) * ((size_t)LDA)] : 0.0;
    s_A[threadIdx.x][threadIdx.y + 16] =
        (an + 16 < N) ? A[aip + (an + 16) * ((size_t)LDA)] : 0.0;
    s_A[threadIdx.x][threadIdx.y + 24] =
        (an + 24 < N) ? A[aip + (an + 24) * ((size_t)LDA)] : 0.0;
  }

  if(un < N) {
    s_U[threadIdx.y + 0][threadIdx.x] =
        (um + 0 < M) ? U[un + (um + 0) * ((size_t)LDU)] : 0.0;
    s_U[threadIdx.y + 8][threadIdx.x] =
        (um + 8 < M) ? U[un + (um + 8) * ((size_t)LDU)] : 0.0;
    s_U[threadIdx.y + 16][threadIdx.x] =
        (um + 16 < M) ? U[un + (um + 16) * ((size_t)LDU)] : 0.0;
    s_U[threadIdx.y + 24][threadIdx.x] =
        (um + 24 < M) ? U[un + (um + 24) * ((size_t)LDU)] : 0.0;
  }

  __syncthreads();

  // swap
  if(am < M) {
    if((an + 0) < N)
      A[aip + (an + 0) * ((size_t)LDA)] = s_U[threadIdx.x][threadIdx.y + 0];
    if((an + 8) < N)
      A[aip + (an + 8) * ((size_t)LDA)] = s_U[threadIdx.x][threadIdx.y + 8];
    if((an + 16) < N)
      A[aip + (an + 16) * ((size_t)LDA)] = s_U[threadIdx.x][threadIdx.y + 16];
    if((an + 24) < N)
      A[aip + (an + 24) * ((size_t)LDA)] = s_U[threadIdx.x][threadIdx.y + 24];
  }

  if(un < N) {
    if((um + 0) < M)
      U[un + (um + 0) * ((size_t)LDU)] = s_A[threadIdx.y + 0][threadIdx.x];
    if((um + 8) < M)
      U[un + (um + 8) * ((size_t)LDU)] = s_A[threadIdx.y + 8][threadIdx.x];
    if((um + 16) < M)
      U[un + (um + 16) * ((size_t)LDU)] = s_A[threadIdx.y + 16][threadIdx.x];
    if((um + 24) < M)
      U[un + (um + 24) * ((size_t)LDU)] = s_A[threadIdx.y + 24][threadIdx.x];
  }
}

void HIP::dlaswp06T(const int M, const int N, double *A, const int LDA, double*    U, const int LDU, const int* LINDXA) {

  if((M <= 0) || (N <= 0)) return;

  dim3 grid_size((M + TILE_DIM_06T - 1) / TILE_DIM_06T, (N + TILE_DIM_06T - 1) / TILE_DIM_06T);
  dim3 block_size(TILE_DIM_06T, BLOCK_ROWS_06T);
  hipLaunchKernelGGL((dlaswp06T_kernel), grid_size, block_size, 0, pdlaswpStream,
                     M, N, A, LDA, U, LDU, LINDXA);
}

#define BLOCK_SIZE_10N 512

__global__ void dlaswp10N_kernel(const int M, const int N, double* __restrict__ A, const int LDA, const int* __restrict__ IPIV) {

  const int m = threadIdx.x + BLOCK_SIZE_10N * blockIdx.x;

  if(m < M) {
    for(int i = 0; i < N; i++) {
      const int ip = IPIV[i];

      if(ip != i) {
        // swap
        const double Ai           = A[m + i * ((size_t)LDA)];
        const double Aip          = A[m + ip * ((size_t)LDA)];
        A[m + i * ((size_t)LDA)]  = Aip;
        A[m + ip * ((size_t)LDA)] = Ai;
      }
    }
  }
}

void HIP::dlaswp10N(const int  M, const int  N, double* A, const int  LDA, const int* IPIV) {
 
  if((M <= 0) || (N <= 0)) return;

  dim3 grid_size((M + BLOCK_SIZE_10N - 1) / BLOCK_SIZE_10N);
  hipLaunchKernelGGL(
      (dlaswp10N_kernel), grid_size, dim3(BLOCK_SIZE_10N), 0, pdlaswpStream, M, N, A, LDA, IPIV);

}