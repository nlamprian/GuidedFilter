include ( ExternalProject )

set ( CLUtils_TARGET CLUtils PARENT_SCOPE )

ExternalProject_Add ( 
    CLUtils 
    GIT_REPOSITORY https://github.com/pAIgn10/CLUtils.git 
    PREFIX ${CMAKE_CURRENT_BINARY_DIR} 
    CMAKE_ARGS -DCMAKE_INSTALL_PREFIX=${CMAKE_INSTALL_PREFIX} # -DBUILD_TESTS=ON 
    INSTALL_COMMAND ""
)

ExternalProject_Get_Property ( CLUtils SOURCE_DIR )
set ( CLUtils_INCLUDE_DIR ${SOURCE_DIR}/include PARENT_SCOPE )

ExternalProject_Get_Property ( CLUtils BINARY_DIR )
set ( CLUtils_LIBRARIES ${BINARY_DIR}/lib/libCLUtils.a PARENT_SCOPE )

set ( CLUtils_FOUND TRUE PARENT_SCOPE )

set ( CLUtils_ROOT ${SOURCE_DIR} PARENT_SCOPE )

install ( SCRIPT ${BINARY_DIR}/cmake_install.cmake )
