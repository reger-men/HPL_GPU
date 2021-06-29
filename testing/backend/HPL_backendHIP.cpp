#include "backend/hpl_backendHIP.h"


void HIP::init(size_t num_gpus)
{
    int rank, size, count;    
    MPI_Comm_rank( MPI_COMM_WORLD, &rank );
    MPI_Comm_size( MPI_COMM_WORLD, &size );

    hipDeviceProp_t hipDeviceProp;
    HIP_CHECK_ERROR(hipGetDeviceCount(&count));
    
    //TODO: set dynamic device id
    int device_id = 0; 

    HIP_CHECK_ERROR(hipSetDevice(device_id));

    // Get device properties
    HIP_CHECK_ERROR(hipGetDeviceProperties(&hipDeviceProp, device_id));

    GPUInfo("Using HIP Device %s with Properties:", hipDeviceProp.name);
    GPUInfo("\ttotalGlobalMem = %lld", (unsigned long long int)hipDeviceProp.totalGlobalMem);
    GPUInfo("\tsharedMemPerBlock = %lld", (unsigned long long int)hipDeviceProp.sharedMemPerBlock);
    GPUInfo("\tregsPerBlock = %d", hipDeviceProp.regsPerBlock);
    GPUInfo("\twarpSize = %d", hipDeviceProp.warpSize);
    GPUInfo("\tmaxThreadsPerBlock = %d", hipDeviceProp.maxThreadsPerBlock);
    GPUInfo("\tmaxThreadsDim = %d %d %d", hipDeviceProp.maxThreadsDim[0], hipDeviceProp.maxThreadsDim[1], hipDeviceProp.maxThreadsDim[2]);
    GPUInfo("\tmaxGridSize = %d %d %d", hipDeviceProp.maxGridSize[0], hipDeviceProp.maxGridSize[1], hipDeviceProp.maxGridSize[2]);
    GPUInfo("\ttotalConstMem = %lld", (unsigned long long int)hipDeviceProp.totalConstMem);
    GPUInfo("\tmajor = %d", hipDeviceProp.major);
    GPUInfo("\tminor = %d", hipDeviceProp.minor);
    GPUInfo("\tclockRate = %d", hipDeviceProp.clockRate);
    GPUInfo("\tmemoryClockRate = %d", hipDeviceProp.memoryClockRate);
    GPUInfo("\tmultiProcessorCount = %d", hipDeviceProp.multiProcessorCount);
    GPUInfo("\tPCIBusID = %d", hipDeviceProp.pciBusID);
    GPUInfo(" ");


    //Init ROCBlas
    rocblas_initialize();
    ROCBLAS_CHECK_STATUS(rocblas_create_handle(&_handle));
}

void HIP::release()
{
    ROCBLAS_CHECK_STATUS(rocblas_destroy_handle(_handle));
}

void HIP::malloc(void** ptr, size_t size)
{
    GPUInfo("allocate memory on HIP of size %ld", size);
    HIP_CHECK_ERROR(hipMalloc(ptr, size));
}

void HIP::free(void** ptr)
{
    HIP_CHECK_ERROR(hipFree(*ptr));
}

int HIP::panel_free(HPL_T_panel *ptr)
{
    GPUInfo("deallocate memory on HIP\n");
    if( ptr->WORK  ) HIP_CHECK_ERROR(hipFree( ptr->WORK  ));
    if( ptr->IWORK ) HIP_CHECK_ERROR(hipFree( ptr->IWORK ));
    return( MPI_SUCCESS );
}

int HIP::panel_disp(HPL_T_panel **ptr)
{
    GPUInfo("deallocate the panel structure on CPU\n");
    int err = HIP::panel_free(*ptr);
    if(*ptr) HIP_CHECK_ERROR(hipFree( ptr ));
    *ptr = NULL;
    return( err );
}

void HIP::matgen(const HPL_T_grid *GRID, const int M, const int N,
                 const int NB, double *A, const int LDA,
                 const int ISEED)
{
    GPUInfo("generate matrix on HIP");
    int mp, mycol, myrow, npcol, nprow, nq;
    (void) HPL_grid_info( GRID, &nprow, &npcol, &myrow, &mycol );
    
    Mnumroc( mp, M, NB, NB, myrow, 0, nprow );
    Mnumroc( nq, N, NB, NB, mycol, 0, npcol );

    if( ( mp <= 0 ) || ( nq <= 0 ) ) return;
    mp = (mp<LDA) ? LDA : mp;
    
    rocrand_generator generator;
    ROCRAND_CHECK_STATUS(rocrand_create_generator(&generator, ROCRAND_RNG_PSEUDO_DEFAULT)); // ROCRAND_RNG_PSEUDO_DEFAULT));
    ROCRAND_CHECK_STATUS(rocrand_set_seed(generator, ISEED));

    //TODO: generate numbers in this range (-0.5, 0.5]
    ROCRAND_CHECK_STATUS(rocrand_generate_normal_double(generator, A, mp*nq, 0, 0.1));
    ROCRAND_CHECK_STATUS(rocrand_destroy_generator(generator));
}

int HIP::idamax(const int N, const double *DX, const int INCX)
{
    GPUInfo("DMAX on HIP");
    rocblas_int result;
    ROCBLAS_CHECK_STATUS(rocblas_idamax(_handle, N, DX, INCX, &result));
    return result;
}

void HIP::daxpy(const int N, const double DA, const double *DX, const int INCX, double *DY, 
                const int INCY)
{
    GPUInfo("DAXPY on HIP");
    ROCBLAS_CHECK_STATUS(rocblas_daxpy(_handle, N, &DA, DX, INCX, DY, INCY));
}

void HIP::dscal(const int N, const double DA, double *DX, const int INCX)
{
    GPUInfo("DSCAL on HIP");
    ROCBLAS_CHECK_STATUS(rocblas_dscal(_handle, N, &DA, DX, INCX));
}

void HIP::dswap(const int N, double *DX, const int INCX, double *DY, const int INCY)
{    
    GPUInfo("DSWAP on HIP");
    ROCBLAS_CHECK_STATUS(rocblas_dswap(_handle, N, DX, INCX, DY, INCY));
}

void HIP::dger( const enum HPL_ORDER ORDER, const int M, const int N, const double ALPHA, const double *X,
               const int INCX, double *Y, const int INCY, double *A, const int LDA)
{
    GPUInfo("DGER on HIP");
    //rocBLAS uses column-major storage for 2D arrays
    ROCBLAS_CHECK_STATUS(rocblas_dger(_handle, M, N, &ALPHA, X, INCX, Y, INCY, A, LDA));
}

void HIP::trsm( const enum HPL_ORDER ORDER, const enum HPL_SIDE SIDE, 
                const enum HPL_UPLO UPLO, const enum HPL_TRANS TRANSA, 
                const enum HPL_DIAG DIAG, const int M, const int N, 
                const double ALPHA, const double *A, const int LDA, double *B, const int LDB)
{
    GPUInfo("TRSM on HIP");
    //rocBLAS uses column-major storage for 2D arrays
    ROCBLAS_CHECK_STATUS(rocblas_dtrsm(_handle, (rocblas_side)SIDE, (rocblas_fill)UPLO, (rocblas_operation)TRANSA, 
                  (rocblas_diagonal)DIAG, M, N, &ALPHA, A, LDA, B, LDB));
}

void HIP::trsv(const enum HPL_ORDER ORDER, const enum HPL_UPLO UPLO,
                const enum HPL_TRANS TRANSA, const enum HPL_DIAG DIAG,
                const int N, const double *A, const int LDA,
                double *X, const int INCX)
{  
    GPUInfo("TRSV on HIP");
}

void HIP::dgemm(const enum HPL_ORDER ORDER, const enum HPL_TRANS TRANSA, 
                const enum HPL_TRANS TRANSB, const int M, const int N, const int K, 
                const double ALPHA, const double *A, const int LDA, 
                const double *B, const int LDB, const double BETA, double *C, 
                const int LDC)
{
    GPUInfo("DGEMM on HIP");
    //rocBLAS uses column-major storage for 2D arrays
    ROCBLAS_CHECK_STATUS(rocblas_dgemm(_handle, (rocblas_operation)TRANSA, (rocblas_operation)TRANSB, 
                         M, N, K, &ALPHA, A, LDA, B, LDB, &BETA, C, LDC));
}

void HIP::dgemv(const enum HPL_ORDER ORDER, const enum HPL_TRANS TRANS, const int M, const int N,
                const double ALPHA, const double *A, const int LDA, const double *X, const int INCX,
                const double BETA, double *Y, const int INCY)
{
    GPUInfo("DGEMV on HIP");
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
    GPUInfo("COPY on HIP");
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
    GPUInfo("A copy on HIP");
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
    GPUInfo("A transpose copy on HIP");
    dim3 grid_size((M+TILE_DIM-1)/TILE_DIM, (N+TILE_DIM-1)/TILE_DIM);
    dim3 block_size(TILE_DIM, BLOCK_ROWS);
    _dlatcpy<<<grid_size, block_size, 0, 0>>>(M, N, A, LDA, B, LDB);
}