cmake_minimum_required( VERSION 3.0 FATAL_ERROR )

project( hplhip VERSION 0.1.0 LANGUAGES CXX )

# current Git branch
EXECUTE_PROCESS(
  COMMAND git rev-parse --abbrev-ref HEAD
  WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
  OUTPUT_VARIABLE GIT_BRANCH
  OUTPUT_STRIP_TRAILING_WHITESPACE
)
set(HPLHIP_BRANCH ${GIT_BRANCH})

# Git hash
EXECUTE_PROCESS(
  COMMAND git log -1 --format=%h
  WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
  OUTPUT_VARIABLE GIT_COMMIT_HASH
  OUTPUT_STRIP_TRAILING_WHITESPACE
)
set(HPLHIP_COMMIT ${GIT_COMMIT_HASH})

# Add rocM root dir to CMAKE_PREFIX_PATH, usually /opt/rocm
find_package( hip REQUIRED )

# switch compiler and linker on non-Windows
if(NOT WIN32)
  set(CMAKE_CXX_COMPILER ${HIP_HIPCC_EXECUTABLE})
  set(CMAKE_CXX_LINKER   ${HIP_HIPCC_EXECUTABLE})
endif()

set( CMAKE_CXX_STANDARD 14 )

# don't need MPI C++ bindings, now deprecated
set( MPI_CXX_SKIP_MPICXX ON )
find_package( MPI REQUIRED )

# search for rocRAND/hipRAND-config.cmake in CMAKE_PREFIX_PATH
find_package(rocrand)
if (rocrand_FOUND)
    message(STATUS "Found rocRAND version ${rocrand_VERSION}")
    set(HPLHIP_DEVRAND_INCLUDE_DIRS "${rocrand_INCLUDE_DIRS}")
    set(HPLHIP_USE_ROCRAND 1)
else ()
    find_package(hiprand)
    if (hiprand_FOUND)
        message(STATUS "Found hipRAND version ${hiprand_VERSION}")
        set(HPLHIP_DEVRAND_INCLUDE_DIRS "${hiprand_INCLUDE_DIRS}")
        set(HPLHIP_USE_HIPRAND 1)
    else ()
        message(FATAL_ERROR "Didn't find any device random generators")
    endif ()
endif ()

add_executable( xhplhip testing/ptest/HPL_pddriver.cpp
testing/ptest/HPL_pdinfo.cpp testing/ptest/HPL_pdtest.cpp
src/auxil/HPL_dlamch.cpp src/auxil/HPL_fprintf.cpp src/grid/HPL_grid_exit.cpp
src/grid/HPL_all_reduce.cpp src/grid/HPL_broadcast.cpp
src/grid/HPL_grid_info.cpp src/grid/HPL_grid_init.cpp src/grid/HPL_max.cpp
src/grid/HPL_min.cpp src/grid/HPL_sum.cpp src/grid/HPL_barrier.cpp
src/pauxil/HPL_pdlamch.cpp src/pauxil/HPL_numroc.cpp src/pauxil/HPL_numrocI.cpp
 src/pauxil/HPL_pdlange.cpp src/pauxil/HPL_indxg2p.cpp src/pauxil/HPL_infog2l.cpp
src/pfact/HPL_pdpancrN.cpp src/pfact/HPL_pdpancrT.cpp
src/pfact/HPL_pdpanllN.cpp src/pfact/HPL_pdpanllT.cpp
src/pfact/HPL_pdpanrlN.cpp src/pfact/HPL_pdpanrlT.cpp
src/pfact/HPL_pdrpancrN.cpp src/pfact/HPL_pdrpancrT.cpp
src/pfact/HPL_pdrpanllN.cpp src/pfact/HPL_pdrpanllT.cpp
src/pfact/HPL_pdrpanrlN.cpp src/pfact/HPL_pdrpanrlT.cpp
src/pgesv/HPL_pdupdateNN.cpp src/pgesv/HPL_pdupdateNT.cpp
src/pgesv/HPL_pdupdateTN.cpp src/pgesv/HPL_pdupdateTT.cpp
src/pauxil/HPL_pwarn.cpp src/comm/HPL_bcast.cpp src/comm/HPL_blong.cpp
src/comm/HPL_blonM.cpp src/comm/HPL_1ring.cpp src/comm/HPL_2ring.cpp
src/comm/HPL_1rinM.cpp src/comm/HPL_2rinM.cpp src/comm/HPL_packL.cpp
src/comm/HPL_sdrv.cpp src/comm/HPL_send.cpp src/pgesv/HPL_pdlaswp00N.cpp
src/comm/HPL_recv.cpp src/grid/HPL_reduce.cpp src/comm/HPL_binit.cpp
src/comm/HPL_bwait.cpp
src/pgesv/HPL_pdlaswp00T.cpp src/pgesv/HPL_pdlaswp01N.cpp
src/pgesv/HPL_pdlaswp01T.cpp src/pgesv/HPL_pdupdateNT.cpp
src/pgesv/HPL_pdupdateTN.cpp src/pgesv/HPL_pdupdateTT.cpp
src/pgesv/HPL_pdupdateNN.cpp
src/pfact/HPL_pdrpanllN.cpp src/pfact/HPL_pdrpanllT.cpp
src/pfact/HPL_pdrpanrlN.cpp src/pfact/HPL_pdrpanrlT.cpp
src/pauxil/HPL_dlaswp00N.cpp src/pauxil/HPL_dlaswp01N.cpp
src/pauxil/HPL_dlaswp01T.cpp src/pauxil/HPL_dlaswp02N.cpp
src/pauxil/HPL_dlaswp03N.cpp src/pauxil/HPL_dlaswp03T.cpp
src/pauxil/HPL_dlaswp04N.cpp src/pauxil/HPL_dlaswp04T.cpp
src/pauxil/HPL_dlaswp05N.cpp src/pauxil/HPL_dlaswp05T.cpp
src/pauxil/HPL_dlaswp06N.cpp src/pauxil/HPL_dlaswp06T.cpp
src/pauxil/HPL_dlaswp10N.cpp src/pfact/HPL_dlocmax.cpp
src/pfact/HPL_dlocswpN.cpp src/pfact/HPL_dlocswpT.cpp src/pgesv/HPL_equil.cpp
src/pfact/HPL_pdfact.cpp src/pauxil/HPL_pabort.cpp src/pfact/HPL_pdmxswp.cpp
src/pgesv/HPL_pipid.cpp src/pgesv/HPL_plindx0.cpp
src/pgesv/HPL_plindx1.cpp src/pgesv/HPL_plindx10.cpp
src/pgesv/HPL_rollN.cpp src/pgesv/HPL_rollT.cpp
src/pgesv/HPL_spreadN.cpp src/pgesv/HPL_spreadT.cpp
src/pgesv/HPL_logsort.cpp src/pgesv/HPL_perm.cpp
src/pgesv/HPL_pdgesv.cpp src/pgesv/HPL_pdgesv0.cpp src/pgesv/HPL_pdgesvK2.cpp
src/pgesv/HPL_pdtrsv.cpp src/blas/HPL_idamax.cpp
testing/ptimer/HPL_ptimer_walltime.cpp testing/ptimer/HPL_ptimer.cpp
testing/ptimer/HPL_ptimer_cputime.cpp
src/panel/HPL_pdpanel_new.cpp src/panel/HPL_pdpanel_init.cpp
src/panel/HPL_pdpanel_free.cpp src/panel/HPL_pdpanel_disp.cpp
testing/backend/HPL_backendWrapper.cpp testing/backend/HPL_backendHIP.cpp)

target_include_directories( xhplhip PUBLIC hip:device
    ${HIP_ROOT_DIR}/include
    ${HPLHIP_DEVRAND_INCLUDE_DIRS}
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)

target_link_libraries( xhplhip ${MPI_CXX_LIBRARIES})

configure_file( include/hplhip_config.hin ${CMAKE_CURRENT_SOURCE_DIR}/include/hplhip_config.h @ONLY NEWLINE_STYLE LF )

install(TARGETS xhplhip RUNTIME DESTINATION bin)
