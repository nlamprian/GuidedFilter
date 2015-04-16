# Try to find OpenCL
# Once done this will define
#  
#  OPENCL_FOUND        - system has OpenCL
#  OPENCL_INCLUDE_DIR  - the OpenCL include directory
#  OPENCL_LIBRARIES    - link these to use OpenCL

set(OPENCL_INC_SEARCH_PATH
	${OPENCL_INCLUDE_DIR}
	$ENV{OPENCL_INCLUDE_DIR}
	/usr/include				# linux (download headers from khronos)
	/usr/local/cuda/include		# for NVIDIA SDK in linux (default installation directory)
	$ENV{AMDAPPSDKROOT}/include # for ATI in windows
)
set(OPENCL_LIB_SEARCH_PATH
	${OPENCL_LIBRARY_DIR}
    $ENV{OPENCL_LIBRARY_DIR}
	/usr/lib						# linux
	$ENV{AMDAPPSDKROOT}/lib/x86_64	# for ATI in windows
)

find_path(OPENCL_INCLUDE_DIR
    NAMES CL/cl.h CL/cl.hpp
    PATHS ${OPENCL_INC_SEARCH_PATH}
	DOC "The directory where opencl headers reside"
)

find_library(OPENCL_LIBRARY
	NAMES OpenCL
	PATHS ${OPENCL_LIB_SEARCH_PATH}
	DOC "The directory where opencl library resides"
)

include(FindPackageHandleStandardArgs)

find_package_handle_standard_args(
	OPENCL
	DEFAULT_MSG
	OPENCL_LIBRARY OPENCL_INCLUDE_DIR
)

if(OPENCL_FOUND)
	set(OPENCL_LIBRARIES ${OPENCL_LIBRARY})
else(OPENCL_FOUND)
	set(OPENCL_LIBRARIES)
endif(OPENCL_FOUND)

mark_as_advanced(
	OPENCL_INCLUDE_DIR
	OPENCL_LIBRARY
)
