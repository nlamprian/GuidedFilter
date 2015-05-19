# Try to find GuidedFilter
# Once done this will define:
#
#   GuidedFilter_ROOT         - if set, it will try to find in this folder
#   GuidedFilter_FOUND        - system has GuidedFilter
#   GuidedFilter_INCLUDE_DIR  - the GuidedFilter include directory
#   GuidedFilter_LIBRARIES    - link these to use GuidedFilter

find_path ( 
    GuidedFilter_INCLUDE_DIR
    NAMES GuidedFilter/common.hpp GuidedFilter/math.hpp 
          GuidedFilter/algorithms.hpp GuidedFilter/tests/helper_funcs.hpp 
    HINTS ${GuidedFilter_ROOT}/include 
          /usr/include
          /usr/local/include 
    DOC "The directory where GuidedFilter headers reside"
)

find_library ( 
    GuidedFilter_LIB_ALGORITHMS 
    NAMES GFAlgorithms 
    PATHS ${GuidedFilter_ROOT}/build/lib 
          /usr/lib/GuidedFilter 
          /usr/local/lib/GuidedFilter 
    DOC "The Guided Filter algorithms library"
)

find_library ( 
    GuidedFilter_LIB_MATH 
    NAMES GFMath 
    PATHS ${GuidedFilter_ROOT}/build/lib 
          /usr/lib/GuidedFilter 
          /usr/local/lib/GuidedFilter 
    DOC "The Guided Filter math library"
)

find_library ( 
    GuidedFilter_LIB_HELPERFUNCS 
    NAMES GFHelperFuncs 
    PATHS ${GuidedFilter_ROOT}/build/lib 
          /usr/lib/GuidedFilter 
          /usr/local/lib/GuidedFilter 
    DOC "The Guided Filter helper functions library"
)

include ( FindPackageHandleStandardArgs )

find_package_handle_standard_args ( 
    GuidedFilter 
    FOUND_VAR GuidedFilter_FOUND
    REQUIRED_VARS GuidedFilter_INCLUDE_DIR 
                  GuidedFilter_LIB_ALGORITHMS 
                  GuidedFilter_LIB_MATH 
                  GuidedFilter_LIB_HELPERFUNCS 
)

if ( GuidedFilter_FOUND )
    set ( 
        GuidedFilter_LIBRARIES 
        ${GuidedFilter_LIB_ALGORITHMS} 
        ${GuidedFilter_LIB_MATH} 
        ${GuidedFilter_LIB_HELPERFUNCS} 
    )
else ( GuidedFilter_FOUND )
    set ( GuidedFilter_LIBRARIES )
endif ( GuidedFilter_FOUND )

mark_as_advanced ( 
    GuidedFilter_INCLUDE_DIR
    GuidedFilter_LIB_ALGORITHMS 
    GuidedFilter_LIB_MATH 
    GuidedFilter_LIB_HELPERFUNCS 
)
