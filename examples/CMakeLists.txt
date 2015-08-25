find_package ( OpenCV2 QUIET )
find_package ( Freenect QUIET )

include_directories ( ${CLUtils_INCLUDE_DIR} )

if ( OpenCV2_FOUND )

    include_directories ( ${OpenCV2_INCLUDE_DIRS} )

    add_executable ( ${FNAME}_image guidedFilter_image.cpp )

    add_dependencies ( ${FNAME}_image CLUtils )

    target_link_libraries ( ${FNAME}_image ${CLUtils_LIBRARIES}
                                           GFAlgorithms GFMath
                                           ${OPENGL_LIBRARIES} 
                                           ${OpenCV2_LIBRARIES} 
                                           ${OPENCL_LIBRARIES} )

endif ( OpenCV2_FOUND )

if ( FREENECT_FOUND )

    find_package ( GLUT REQUIRED )
    find_package ( libusb-1.0 REQUIRED )
    find_package ( Threads REQUIRED )

    include_directories ( ${OPENGL_INCLUDE_DIRS}
                          ${GLUT_INCLUDE_DIRS}
                          ${LIBUSB_1_INCLUDE_DIRS}
                          ${FREENECT_INCLUDE_DIR} )

    add_executable ( ${FNAME}_kinect_rgb guidedFilter_kinect_rgb.cpp )

    add_dependencies ( ${FNAME}_kinect_rgb CLUtils )

    target_link_libraries ( ${FNAME}_kinect_rgb ${CLUtils_LIBRARIES}
                                                GFAlgorithms GFMath 
                                                ${OPENGL_LIBRARIES}
                                                ${GLUT_LIBRARY}
                                                ${FREENECT_LIBRARIES} 
                                                ${OPENCL_LIBRARIES}
                                                ${CMAKE_THREAD_LIBS_INIT} )

    find_package ( GLEW REQUIRED )

    include_directories ( ${GLEW_INCLUDE_DIRS} )

    add_executable ( ${FNAME}_kinect_point_cloud guidedFilter_kinect_point_cloud.cpp )

    add_dependencies ( ${FNAME}_kinect_point_cloud CLUtils )

    target_link_libraries ( ${FNAME}_kinect_point_cloud ${CLUtils_LIBRARIES}
                                                        GFAlgorithms GFMath 
                                                        ${OPENGL_LIBRARIES} 
                                                        ${GLUT_LIBRARY} 
                                                        ${GLEW_LIBRARIES} 
                                                        ${FREENECT_LIBRARIES} 
                                                        ${OPENCL_LIBRARIES} 
                                                        ${CMAKE_THREAD_LIBS_INIT} )

endif ( FREENECT_FOUND )
