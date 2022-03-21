
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
    rocblas_set_pointer_mode(_handle, rocblas_pointer_mode_host);
    HIP_CHECK_ERROR(hipStreamCreate(&computeStream));
    HIP_CHECK_ERROR(hipStreamCreate(&dataStream));
    
    ROCBLAS_CHECK_STATUS(rocblas_set_stream(_handle, computeStream));
    
    HIP_CHECK_ERROR(hipEventCreate(&panelUpdate));
    HIP_CHECK_ERROR(hipEventCreate(&panelCopy));
    HIP_CHECK_ERROR(hipEventCreate(&dlaswpStart));
    HIP_CHECK_ERROR(hipEventCreate(&dlaswpStop));
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
}

void HIP::malloc(void** ptr, size_t size)
{
    GPUInfo("%-25s %-12ld (B) \t%-5s", "[Allocate]", "Memory of size",  size, "HIP");
    HIP_CHECK_ERROR(hipMalloc(ptr, size));
}

void HIP::free(void** ptr)
{
    HIP_CHECK_ERROR(hipFree(*ptr));
    ROCBLAS_CHECK_STATUS(rocblas_destroy_handle(_handle));
    HIP_CHECK_ERROR(hipStreamDestroy(computeStream));
    HIP_CHECK_ERROR(hipStreamDestroy(dataStream));
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
    p->WORK = NULL;
    p->IWORK = NULL;
    p->IWORK2 = NULL;
    HIP::panel_init( GRID, ALGO, M, N, JB, A, IA, JA, TAG, p );
    *PANEL = p;
}

void HIP::panel_init(HPL_T_grid *GRID, HPL_T_palg *ALGO, const int M, const int N, const int JB, 
                        HPL_T_pmat *A, const int IA, const int JA, const int TAG, HPL_T_panel *PANEL)
{
    size_t                     dalign;
    int                        icurcol, icurrow, ii, itmp1, jj, lwork,
                                ml2, mp, mycol, myrow, nb, npcol, nprow,
                                nq, nu;
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

    if( npcol == 1 )                             /* P x 1 process grid */
    {                                     /* space for L1, DPIV, DINFO */
        lwork = ALGO->align + ( PANEL->len = JB * JB + JB + 1 );
        if( nprow > 1 )                                 /* space for U */
        { nu = nq - JB; lwork += JB * Mmax( 0, nu ); }

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
        PANEL->DPIV  = (double *)HPL_PTR( PANEL->WORK, dalign ) + JB * JB;
        PANEL->DINFO = PANEL->DPIV + JB;
        *(PANEL->DINFO) = 0.0;
        PANEL->U     = ( nprow > 1 ? PANEL->DINFO + 1: NULL );
        PANEL->dU    = (double *)HPL_PTR( PANEL->WORK, dalign ) + JB * JB;
    }
    else
    {                                        /* space for L2, L1, DPIV */
        ml2 = ( myrow == icurrow ? mp - JB : mp ); ml2 = Mmax( 0, ml2 );
        PANEL->len = ml2*JB + ( itmp1 = JB*JB + JB + 1 );
    #ifdef HPL_COPY_L
        lwork = ALGO->align + PANEL->len;
    #else
        lwork = ALGO->align + ( mycol == icurcol ? itmp1 : PANEL->len );
    #endif

        if( nprow > 1 )                                 /* space for U */
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
        PANEL->DPIV  = PANEL->L1   + JB * JB;
        PANEL->dDPIV  = PANEL->dL1   + JB * JB;
        PANEL->DINFO = PANEL->DPIV + JB;
        *(PANEL->DINFO) = 0.0;
        PANEL->U     = ( nprow > 1 ? PANEL->DINFO + 1 : NULL );
        PANEL->dU    = PANEL->dL1   + JB * JB;;
    }


    if( nprow == 1 ) { lwork = 3*JB; }
    else
    {
        itmp1 = (JB << 2); lwork = nprow + 1; itmp1 = Mmax( itmp1, lwork );
        lwork = 4 + (9 * JB) + (3 * nprow) + itmp1;
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
    // hipMemcpy2D(PANEL->A,  PANEL->lda*sizeof(double),
    //                 PANEL->dA, PANEL->lda*sizeof(double),
    //                 PANEL->mp*sizeof(double), jb,
    //                 hipMemcpyDeviceToHost);

}

void HIP::panel_send_to_device(HPL_T_panel *PANEL)
{
    double *A, *dA;
    int jb, i, ml2;
    /* ..
     * .. Executable Statements ..
     */
    jb = PANEL->jb;

    if (jb <= 0)
        return;

        // copy A and/or L2
#ifdef HPL_COPY_L
#else
    if (PANEL->grid->mycol == PANEL->pcol)
    { // L2 reuses A
        A = Mptr(PANEL->A, 0, -jb, PANEL->lda);
        dA = Mptr(PANEL->dA, 0, -jb, PANEL->lda);

        hipMemcpy2DAsync(dA, PANEL->lda * sizeof(double),
                         A, PANEL->lda * sizeof(double),
                         PANEL->mp * sizeof(double), jb,
                         hipMemcpyHostToDevice, dataStream);
        // hipMemcpy2D(dA, PANEL->lda * sizeof(double),
        //                  A, PANEL->lda * sizeof(double),
        //                  PANEL->mp * sizeof(double), jb,
        //                  hipMemcpyHostToDevice);
    }
    else
    {
        ml2 = (PANEL->grid->myrow == PANEL->prow ? PANEL->mp - jb : PANEL->mp);
        if (ml2 > 0)
            hipMemcpy2DAsync(PANEL->dL2, PANEL->ldl2 * sizeof(double),
                             PANEL->L2, PANEL->ldl2 * sizeof(double),
                             ml2 * sizeof(double), jb,
                             hipMemcpyHostToDevice, dataStream);
            // hipMemcpy2D(PANEL->dL2, PANEL->ldl2 * sizeof(double),
            //                  PANEL->L2, PANEL->ldl2 * sizeof(double),
            //                  ml2 * sizeof(double), jb,
            //                  hipMemcpyHostToDevice);
    }
#endif
    // copy L1
    hipMemcpy2DAsync(PANEL->dL1, jb * sizeof(double),
                     PANEL->L1, jb * sizeof(double),
                     jb * sizeof(double), jb,
                     hipMemcpyHostToDevice, dataStream);
    // hipMemcpy2D(PANEL->dL1, jb * sizeof(double),
    //                  PANEL->L1, jb * sizeof(double),
    //                  jb * sizeof(double), jb,
    //                  hipMemcpyHostToDevice);
    // unroll pivoting and send to device
    int *ipiv = PANEL->IWORK;
    int *dipiv = PANEL->dIWORK;
    int *ipiv_ex = PANEL->IWORK + jb;
    int *dipiv_ex = PANEL->dIWORK + jb;

    int *upiv = PANEL->IWORK2;

    for (i = 0; i < jb; i++)
    {
        ipiv[i] = (int)(PANEL->DPIV[i]) - PANEL->ii;
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

    // hipMemcpy2D(dipiv, jb * sizeof(int),
    //                  upiv, jb * sizeof(int),
    //                  jb * sizeof(int), 1,
    //                  hipMemcpyHostToDevice);
    // hipMemcpy2D(dipiv_ex, jb * sizeof(int),
    //                  ipiv_ex, jb * sizeof(int),
    //                  jb * sizeof(int), 1,
    //                  hipMemcpyHostToDevice);
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
    }
    return (MPI_SUCCESS);
}
// int HIP::panel_free(HPL_T_panel *ptr)
// {
//     GPUInfo("%-40s \t%-5s", "[Deallocate]", "Panel resources", "HIP");
//     if( ptr->WORK  ) HIP_CHECK_ERROR(hipFree( ptr->WORK  ));
//     if( ptr->IWORK ) HIP_CHECK_ERROR(hipFree( ptr->IWORK ));
//     return( MPI_SUCCESS );
// }

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
// int HIP::panel_disp(HPL_T_panel **ptr)
// {
//     GPUInfo("%-40s \t%-5s", "[Deallocate]", "Panel structure", "HIP");
//     int err = HIP::panel_free(*ptr);
//     if(*ptr) HIP_CHECK_ERROR(hipFree( ptr ));
//     *ptr = NULL;
//     return( err );
// }

void gPrintMat(const int M, const int N, const int LDA, const double *A)
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
    // rocrand_generate_normal_double(generator, A, mp*nq, 0.0, 0.25);
    hipDeviceSynchronize();

    rocrand_destroy_generator(generator);
}

void HIP::event_record(enum HIP::HPL_EVENT _event){
    switch (_event)
    {
    case PANEL_COPY:
    HIP_CHECK_ERROR(hipEventRecord(panelCopy, dataStream));
    HIP_CHECK_ERROR(hipEventSynchronize(panelCopy));        
    break;

    case PANEL_UPDATE:
    HIP_CHECK_ERROR(hipEventRecord(panelUpdate, dataStream));
    HIP_CHECK_ERROR(hipEventSynchronize(panelUpdate));
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
    GPUInfo("%-25s %-8d%-8d \t%-5s", "[TRSM]", "With B of (R:C)", M, N, "HIP");
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
    hipDeviceSynchronize();
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
    GPUInfo("%-25s %-8d%-8d \t%-5s", "[DGEMM]", "With C of (R:C)", LDC, N, "HIP");
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

    hipDeviceSynchronize();                         
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
__launch_bounds__(TILE_DIM *BLOCK_ROWS)  
_dlatcpy(const int M, const int N, const double* __restrict__ A, const int LDA,
         double* __restrict__ B, const int LDB) 
{

}

/*
* Copies the transpose of an array A into an array B.
*/
void HIP::atcpy(const int M, const int N, const double *A, const int LDA,
                double *B, const int LDB)
{
    GPUInfo("%-25s %-8d%-8d \t%-5s", "[LATCOPY]", "With A of (R:C)", M, N, "HIP");
    dim3 grid_size((M+TILE_DIM-1)/TILE_DIM, (N+TILE_DIM-1)/TILE_DIM);
    dim3 block_size(TILE_DIM, BLOCK_ROWS);
    _dlatcpy<<<grid_size, block_size, 0, 0>>>(M, N, A, LDA, B, LDB);
}

void HIP::move_data(double *DST, const double *SRC, const size_t SIZE, const int KIND)
{
    char title[25] = "[MOVE_"; strcat(title,_memcpyKind[KIND]); strcat(title,"]");
    GPUInfo("%-25s %-12ld (B) \t%-5s", title, "Memory of size",  SIZE, "HIP");
    HIP_CHECK_ERROR(hipMemcpy(DST, SRC, SIZE, (hipMemcpyKind)KIND));
}

#define BLOCK_SIZE 512

__global__ void _dlaswp00N(const int N, const int M,
                     double* __restrict__ A,
                     const int LDA,
                     const int* __restrict__ IPIV) {
                    //  const int* IPIV) {

   __shared__ double s_An_init[2048];
   __shared__ double s_An_ipiv[2048];

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

    const int block_size = 512, grid_size = N;
    hipLaunchKernelGGL(_dlaswp00N, dim3(grid_size), dim3(block_size), 0, stream,
                                      N, M, A, LDA, IPIV);
}