file ( GLOB_RECURSE KERNEL_SRCS ${CMAKE_CURRENT_SOURCE_DIR}/*.cl )

add_custom_target ( kernels ALL )

foreach ( SRC ${KERNEL_SRCS} )

    file ( RELATIVE_PATH R_SRC ${CMAKE_CURRENT_SOURCE_DIR} ${SRC} )

    add_custom_command ( 
        TARGET kernels POST_BUILD 
        COMMAND ${CMAKE_COMMAND} -E copy_if_different ${SRC} ${PROJECT_BINARY_DIR}/kernels/${R_SRC}
    )

endforeach ( SRC )
