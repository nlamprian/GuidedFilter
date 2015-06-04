cmake_minimum_required ( VERSION 2.8.12.2 )
project ( GuidedFilter )
set ( FNAME "guided_filter" )
set ( PROJECT_VERSION_MAJOR 1 )
set ( PROJECT_VERSION_MINOR 1 )
set ( PROJECT_VERSION_PATCH 2 )

option ( BUILD_TESTS "Build tests" OFF )

list ( APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake_modules )

find_package ( OpenCL REQUIRED )
find_package ( OpenGL REQUIRED )

add_definitions ( -std=c++0x )

set ( CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -static-libstdc++" )
# set ( WARNINGS "-Wall -Wextra" )
set ( CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${WARNINGS}" )

set ( EXECUTABLE_OUTPUT_PATH ${PROJECT_BINARY_DIR}/bin )
set ( LIBRARY_OUTPUT_PATH ${PROJECT_BINARY_DIR}/lib )

set ( COMMON_INCLUDES ${PROJECT_SOURCE_DIR}/include )

include_directories ( ${COMMON_INCLUDES} 
                      ${OPENCL_INCLUDE_DIR} )

# set ( CLUtils_ROOT <set-to-your-local-path-if-necessary> )
find_package ( CLUtils QUIET )
if ( NOT CLUtils_FOUND )
    add_subdirectory ( external/CLUtils )
endif ( NOT CLUtils_FOUND )

add_subdirectory ( src )
add_subdirectory ( kernels )
add_subdirectory ( examples )
add_subdirectory ( docs )

if ( BUILD_TESTS )
    enable_testing (  )
    
    find_package ( GTest QUIET )
    if ( NOT GTEST_FOUND )
        add_subdirectory ( external/gtest )
    endif ( NOT GTEST_FOUND )
    
    add_subdirectory ( tests )
endif ( BUILD_TESTS )
