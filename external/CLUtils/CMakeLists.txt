include ( ExternalProject )

ExternalProject_Add ( 
    CLUtils 
    GIT_REPOSITORY https://github.com/pAIgn10/CLUtils.git 
    PREFIX ${CMAKE_CURRENT_BINARY_DIR} 
    # CMAKE_ARGS -DBUILD_TESTS=ON 
    INSTALL_COMMAND ""
)

ExternalProject_Get_Property ( CLUtils SOURCE_DIR )
set ( CLUtils_INCLUDE_DIR ${SOURCE_DIR}/include PARENT_SCOPE )

ExternalProject_Get_Property ( CLUtils BINARY_DIR )
set ( CLUtils_LIBRARIES ${BINARY_DIR}/lib/libCLUtils.a PARENT_SCOPE )

install ( FILES ${SOURCE_DIR}/include/CLUtils.hpp DESTINATION include )
install ( FILES ${BINARY_DIR}/lib/libCLUtils.a DESTINATION lib )
