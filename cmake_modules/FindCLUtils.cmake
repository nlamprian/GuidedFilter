# Try to find CLUtils
#
# The following variables are optionally searched for defaults:
#   CLUtils_ROOT         - Root directory of CLUtils source tree
# 
# Once done, this will define:
#   CLUtils_FOUND        - system has CLUtils
#   CLUtils_INCLUDE_DIR  - the CLUtils include directory
#   CLUtils_LIBRARIES    - link these to use CLUtils

find_path ( 
    CLUtils_INCLUDE_DIR
    NAMES CLUtils.hpp
    HINTS ${CLUtils_ROOT}/include 
          /usr/include
          /usr/local/include 
    DOC "The directory where the CLUtils header resides"
)

find_library ( 
    CLUtils_LIBRARY 
    NAMES CLUtils 
    PATHS ${CLUtils_ROOT}/build/lib 
          ${CLUtils_ROOT}/../CLUtils-build/lib 
          /usr/lib 
          /usr/local/lib 
    DOC "The CLUtils library"
)

include ( FindPackageHandleStandardArgs )

find_package_handle_standard_args ( 
    CLUtils 
    FOUND_VAR CLUtils_FOUND 
    REQUIRED_VARS CLUtils_INCLUDE_DIR CLUtils_LIBRARY 
)

if ( CLUtils_FOUND )
    set ( CLUtils_LIBRARIES ${CLUtils_LIBRARY} )
    message ( STATUS "Found CLUtils:" )
    message ( STATUS " - Includes: ${CLUtils_INCLUDE_DIR}" )
    message ( STATUS " - Libraries: ${CLUtils_LIBRARIES}" )
else ( CLUtils_FOUND )
    set ( CLUtils_LIBRARIES )
endif ( CLUtils_FOUND )

mark_as_advanced ( 
    CLUtils_INCLUDE_DIR
    CLUtils_LIBRARY
)
