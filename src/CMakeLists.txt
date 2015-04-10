add_library ( HelperFuncs STATIC GuidedFilter/tests/helperFuncs.cpp )
add_library ( GFAlgorithms STATIC GuidedFilter/algorithms.cpp )
add_library ( GFMath STATIC GuidedFilter/math.cpp )

target_include_directories ( 
    HelperFuncs PUBLIC 
    ${COMMON_INCLUDES} 
)

target_include_directories ( 
    GFAlgorithms PUBLIC 
    ${COMMON_INCLUDES} 
)

target_include_directories ( 
    GFMath PUBLIC 
    ${COMMON_INCLUDES} 
)

target_link_libraries (
    GFAlgorithms LINK_PUBLIC 
    GFMath
)

install ( DIRECTORY ${PROJECT_SOURCE_DIR}/include/ DESTINATION include )
install ( DIRECTORY ${PROJECT_BINARY_DIR}/lib/ DESTINATION lib/GuidedFilter )
