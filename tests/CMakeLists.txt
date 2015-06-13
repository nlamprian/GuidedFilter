if ( BUILD_TESTS )

    find_package ( Threads REQUIRED )

    include_directories ( ${CLUtils_INCLUDE_DIR} 
                          ${GTEST_INCLUDE_DIRS}
                          ${OPENGL_INCLUDE_DIRS} )

    add_executable ( ${FNAME}_tests_img testsImageSupport.cpp )
    add_executable ( ${FNAME}_tests_math testsMath.cpp )
    add_executable ( ${FNAME}_tests_box testsBoxFilter.cpp )
    add_executable ( ${FNAME}_tests_gf testsGuidedFilter.cpp )

    add_dependencies ( ${FNAME}_tests_img CLUtils googletest )
    add_dependencies ( ${FNAME}_tests_math CLUtils googletest )
    add_dependencies ( ${FNAME}_tests_box CLUtils googletest )
    add_dependencies ( ${FNAME}_tests_gf CLUtils googletest )

    target_link_libraries ( ${FNAME}_tests_img LINK_PUBLIC ${CLUtils_LIBRARIES} 
                                                           GFHelperFuncs 
                                                           GFAlgorithms
                                                           ${OPENGL_LIBRARIES}
                                                           ${OPENCL_LIBRARIES}
                                                           ${GTEST_BOTH_LIBRARIES}
                                                           ${CMAKE_THREAD_LIBS_INIT} )

    target_link_libraries ( ${FNAME}_tests_math LINK_PUBLIC ${CLUtils_LIBRARIES} 
                                                            GFHelperFuncs 
                                                            GFMath
                                                            ${OPENGL_LIBRARIES}
                                                            ${OPENCL_LIBRARIES}
                                                            ${GTEST_BOTH_LIBRARIES}
                                                            ${CMAKE_THREAD_LIBS_INIT} )

    target_link_libraries ( ${FNAME}_tests_box LINK_PUBLIC ${CLUtils_LIBRARIES} 
                                                           GFHelperFuncs 
                                                           GFAlgorithms 
                                                           ${OPENGL_LIBRARIES}
                                                           ${OPENCL_LIBRARIES}
                                                           ${GTEST_BOTH_LIBRARIES}
                                                           ${CMAKE_THREAD_LIBS_INIT} )

    target_link_libraries ( ${FNAME}_tests_gf LINK_PUBLIC ${CLUtils_LIBRARIES} 
                                                          GFHelperFuncs 
                                                          GFAlgorithms GFMath
                                                          ${OPENGL_LIBRARIES}
                                                          ${OPENCL_LIBRARIES}
                                                          ${GTEST_BOTH_LIBRARIES}
                                                          ${CMAKE_THREAD_LIBS_INIT} )

    add_test ( NAME ${FNAME}_tests_img 
               COMMAND ${EXECUTABLE_OUTPUT_PATH}/${FNAME}_tests_img 
               WORKING_DIRECTORY ${CMAKE_BINARY_DIR} )

    add_test ( NAME ${FNAME}_tests_math 
               COMMAND ${EXECUTABLE_OUTPUT_PATH}/${FNAME}_tests_math 
               WORKING_DIRECTORY ${CMAKE_BINARY_DIR} )

    add_test ( NAME ${FNAME}_tests_box 
               COMMAND ${EXECUTABLE_OUTPUT_PATH}/${FNAME}_tests_box 
               WORKING_DIRECTORY ${CMAKE_BINARY_DIR} )

    add_test ( NAME ${FNAME}_tests_gf 
               COMMAND ${EXECUTABLE_OUTPUT_PATH}/${FNAME}_tests_gf 
               WORKING_DIRECTORY ${CMAKE_BINARY_DIR} )

    add_custom_target ( check COMMAND ${CMAKE_CTEST_COMMAND} --verbose )

endif ( BUILD_TESTS )
