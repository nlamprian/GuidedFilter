# Try to find GuidedFilter
# Once done this will define:
#
#   GuidedFilter_ROOT         - if set, it will try to find in this folder
#   GuidedFilter_FOUND        - system has GuidedFilter
#   GuidedFilter_INCLUDE_DIR  - the GuidedFilter include directory
#   GuidedFilter_LIBRARIES    - link these to use GuidedFilter

find_path ( 
    GuidedFilter_INCLUDE_DIR
    NAMES algorithms.hpp common.hpp math.hpp tests/helperFuncs.hpp 
    HINTS ${GuidedFilter_ROOT}/include 
          /usr/include
          /usr/local/include 
    DOC "The directory where GuidedFilter headers reside"
)

find_library ( 
    GuidedFilter_LIBRARY 
    NAMES GFAlgorithms GFMath GFHelperFuncs  
    PATHS ${GuidedFilter_ROOT}/build/lib 
          /usr/lib 
          /usr/local/lib 
    DOC "The directory where GuidedFilter libraries reside"
)

include ( FindPackageHandleStandardArgs )

find_package_handle_standard_args ( 
    GuidedFilter 
    FOUND_VAR GuidedFilter_FOUND
    REQUIRED_VARS GuidedFilter_INCLUDE_DIR GuidedFilter_LIBRARY 
)

if ( GuidedFilter_FOUND )
    set ( GuidedFilter_LIBRARIES ${GuidedFilter_LIBRARY} )
else ( GuidedFilter_FOUND )
    set ( GuidedFilter_LIBRARIES )
endif ( GuidedFilter_FOUND )

mark_as_advanced ( 
    GuidedFilter_INCLUDE_DIR
    GuidedFilter_LIBRARY
)
