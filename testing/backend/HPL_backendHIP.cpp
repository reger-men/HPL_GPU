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
    ROCBLAS_CHECK_STATUS(rocblas_create_handle(&handle));
}

void HIP::release()
{
    ROCBLAS_CHECK_STATUS(rocblas_destroy_handle(handle));
}

void HIP::malloc(void** ptr, size_t size)
{
    GPUInfo("allocate memory on GPU of size %ld", size);
    HIP_CHECK_ERROR(hipMalloc(ptr, size));
}

void HIP::matgen(const HPL_T_grid *GRID, const int M, const int N,
                 const int NB, double *A, const int LDA,
                 const int ISEED)
{
    GPUInfo("generate matrix on GPU");
    int mp, mycol, myrow, npcol, nprow, nq;
    (void) HPL_grid_info( GRID, &nprow, &npcol, &myrow, &mycol );
    
    Mnumroc( mp, M, NB, NB, myrow, 0, nprow );
    Mnumroc( nq, N, NB, NB, mycol, 0, npcol );

    if( ( mp <= 0 ) || ( nq <= 0 ) ) return;
    mp = (mp<LDA) ? LDA : mp;
    
    rocrand_generator generator;
    ROCRAND_CHECK_STATUS(rocrand_create_generator(&generator, ROCRAND_RNG_PSEUDO_DEFAULT)); // ROCRAND_RNG_PSEUDO_DEFAULT));
    ROCRAND_CHECK_STATUS(rocrand_set_seed(generator, ISEED));

    //TODO: generate numbers between -0.5 & 0.5
    ROCRAND_CHECK_STATUS(rocrand_generate_normal_double(generator, A, mp*nq, 0, 0.1));
    ROCRAND_CHECK_STATUS(rocrand_destroy_generator(generator));

}