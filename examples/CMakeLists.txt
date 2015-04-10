find_package ( Freenect QUIET )
find_package ( OpenCV QUIET )

if ( OpenCV_FOUND )

    include_directories ( ${OpenCV_INCLUDE_DIR} )

    add_executable ( ${FNAME}_image guidedFilter_image.cpp )

    target_link_libraries ( ${FNAME}_image CLUtils
                                           GFAlgorithms GFMath
                                           ${OPENGL_LIBRARIES} 
                                           ${OpenCV_LIBRARIES} 
                                           ${OPENCL_LIBRARIES} )

endif (  )

if ( FREENECT_FOUND )

    find_package ( GLUT REQUIRED )
    find_package ( libusb-1.0 REQUIRED )
    find_package ( Threads REQUIRED )

    include_directories ( ${OPENGL_INCLUDE_DIRS}
                          ${GLUT_INCLUDE_DIRS}
                          ${LIBUSB_1_INCLUDE_DIRS}
                          ${FREENECT_INCLUDE_DIR} )

    add_executable ( ${FNAME}_kinect_rgb guidedFilter_kinect_rgb.cpp )

    target_link_libraries ( ${FNAME}_kinect_rgb CLUtils
                                                GFAlgorithms GFMath 
                                                ${OPENGL_LIBRARIES}
                                                ${GLUT_LIBRARY}
                                                ${FREENECT_LIBRARIES} 
                                                ${OPENCL_LIBRARIES}
                                                ${CMAKE_THREAD_LIBS_INIT} )

    find_package ( GLEW REQUIRED )

    include_directories ( ${GLEW_INCLUDE_DIRS} )

    add_executable ( ${FNAME}_kinect_point_cloud guidedFilter_kinect_point_cloud.cpp )

    target_link_libraries ( ${FNAME}_kinect_point_cloud CLUtils
                                                        GFAlgorithms GFMath 
                                                        ${OPENGL_LIBRARIES} 
                                                        ${GLUT_LIBRARY} 
                                                        ${GLEW_LIBRARIES} 
                                                        ${FREENECT_LIBRARIES} 
                                                        ${OPENCL_LIBRARIES} 
                                                        ${CMAKE_THREAD_LIBS_INIT} )

endif (  )
