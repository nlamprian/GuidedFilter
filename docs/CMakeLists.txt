find_package ( Doxygen QUIET )

if ( DOXYGEN_FOUND )
    
    set ( DOC_DIR ${PROJECT_BINARY_DIR}/docs )

    set ( DOXYGEN_SEARCH_DIRS_LIST 
          "${PROJECT_SOURCE_DIR}/include"
          "${PROJECT_SOURCE_DIR}/src"
          "${PROJECT_SOURCE_DIR}/kernels"
          "${PROJECT_SOURCE_DIR}/tests"
          "${PROJECT_SOURCE_DIR}/examples"
          "${PROJECT_SOURCE_DIR}/docs"
    )

    foreach ( DIR ${DOXYGEN_SEARCH_DIRS_LIST} )
        set ( DOXYGEN_SEARCH_DIRS "${DOXYGEN_SEARCH_DIRS} ${DIR}" )
    endforeach ( DIR )

    configure_file ( 
        ${PROJECT_SOURCE_DIR}/docs/Doxyfile.in 
        ${DOC_DIR}/Doxyfile 
    )
    
    add_custom_target ( 
        doxygen
        COMMAND ${DOXYGEN_EXECUTABLE} ${DOC_DIR}/Doxyfile
        COMMAND cp -r ${PROJECT_SOURCE_DIR}/docs/css ${DOC_DIR}/html
        COMMAND cp -r ${PROJECT_SOURCE_DIR}/docs/scss ${DOC_DIR}/html
        COMMAND cp -r ${PROJECT_SOURCE_DIR}/docs/fonts ${DOC_DIR}/html
        COMMAND cp -r ${PROJECT_SOURCE_DIR}/docs/less ${DOC_DIR}/html
    )

endif ( DOXYGEN_FOUND )
