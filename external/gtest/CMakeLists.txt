include ( ExternalProject )

set ( GTEST_TARGET googletest PARENT_SCOPE )

ExternalProject_Add ( 
    googletest 
    SVN_REPOSITORY http://googletest.googlecode.com/svn/trunk
    PREFIX ${CMAKE_CURRENT_BINARY_DIR} 
    CMAKE_ARGS -DCMAKE_ARCHIVE_OUTPUT_DIRECTORY_DEBUG:PATH=DebugLibs
               -DCMAKE_ARCHIVE_OUTPUT_DIRECTORY_RELEASE:PATH=ReleaseLibs
               -DCMAKE_CXX_FLAGS=${MSVC_COMPILER_DEFS}
               -Dgtest_force_shared_crt=ON
               # -Dgtest_build_samples=ON 
    INSTALL_COMMAND ""
)

ExternalProject_Get_Property ( googletest SOURCE_DIR )
set ( GTEST_INCLUDE_DIRS ${SOURCE_DIR}/include PARENT_SCOPE )

ExternalProject_Get_Property ( googletest BINARY_DIR )
set ( GTEST_LIBS_DIR ${BINARY_DIR} PARENT_SCOPE )

set ( GTEST_FOUND TRUE PARENT_SCOPE )

set ( GTEST_ROOT ${SOURCE_DIR} PARENT_SCOPE )

set ( 
    GTEST_BOTH_LIBRARIES 
    ${BINARY_DIR}/libgtest.a 
    ${BINARY_DIR}/libgtest_main.a 
    PARENT_SCOPE 
)
