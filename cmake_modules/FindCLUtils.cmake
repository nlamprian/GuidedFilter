# Try to find CLUtils
# Once done this will define:
#
#   CLUtils_ROOT         - if set, it will try to find in this folder
#   CLUtils_FOUND        - system has CLUtils
#   CLUtils_INCLUDE_DIR  - the CLUtils include directory
#   CLUtils_LIBRARIES    - link these to use CLUtils

find_path ( 
    CLUtils_INCLUDE_DIR
    NAMES CLUtils.hpp
    HINTS ${CLUtils_ROOT}/include 
          /usr/include
          /usr/local/include 
    DOC "The directory where CLUtils headers reside"
)

find_library ( 
    CLUtils_LIBRARY 
    NAMES CLUtils 
    PATHS ${CLUtils_ROOT}/build/lib 
          /usr/lib 
          /usr/local/lib 
    DOC "The directory where CLUtils library resides"
)

include ( FindPackageHandleStandardArgs )

find_package_handle_standard_args ( 
    CLUtils 
    FOUND_VAR CLUtils_FOUND
    REQUIRED_VARS CLUtils_INCLUDE_DIR CLUtils_LIBRARY 
)

if ( CLUtils_FOUND )
    set ( CLUtils_LIBRARIES ${CLUtils_LIBRARY} )
else ( CLUtils_FOUND )
    set ( CLUtils_LIBRARIES )
endif ( CLUtils_FOUND )

mark_as_advanced ( 
    CLUtils_INCLUDE_DIR
    CLUtils_LIBRARY
)
