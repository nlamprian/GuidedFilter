#.rst:
# FindOPENCL
# ----------
#
# Try to find OpenCL
#
# Once done this will define::
#
#   OPENCL_FOUND          - True if OPENCL was found
#   OPENCL_INCLUDE_DIR    - include directories for OPENCL
#   OPENCL_LIBRARIES      - link against this library to use OPENCL
#   OPENCL_VERSION_STRING - Highest supported OPENCL version (eg. 1.2)
#   OPENCL_VERSION_MAJOR  - The major version of the OPENCL implementation
#   OPENCL_VERSION_MINOR  - The minor version of the OPENCL implementation
#
# The module will also define two cache variables::
#
#   OPENCL_INCLUDE_DIR    - the OPENCL include directory
#   OPENCL_LIBRARY        - the path to the OPENCL library
#

#=============================================================================
# Modified by Nick Lamprianidis

# ------------------------------------
# Copyright 2014, Matthaeus G. Chajdas
# All rights reserved.

# Redistribution and use in source and binary forms, with or without modification, 
# are permitted provided that the following conditions are met:

# 1. Redistributions of source code must retain the above copyright notice, this 
# list of conditions and the following disclaimer.

# 2. Redistributions in binary form must reproduce the above copyright notice, 
# this list of conditions and the following disclaimer in the documentation 
# and/or other materials provided with the distribution.

# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND 
# ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED 
# WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. 
# IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, 
# INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, 
# BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, 
# DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF 
# LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE 
# OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED 
# OF THE POSSIBILITY OF SUCH DAMAGE.

function(_FIND_OPENCL_VERSION)
  include(CheckSymbolExists)
  include(CMakePushCheckState)
  set(CMAKE_REQUIRED_QUIET ${OPENCL_FIND_QUIETLY})

  CMAKE_PUSH_CHECK_STATE()
  foreach(VERSION "2_0" "1_2" "1_1" "1_0")
    set(CMAKE_REQUIRED_INCLUDES "${OPENCL_INCLUDE_DIR}")

    if(APPLE)
      CHECK_SYMBOL_EXISTS(
        CL_VERSION_${VERSION}
        "${OPENCL_INCLUDE_DIR}/OPENCL/cl.h"
        OPENCL_VERSION_${VERSION})
    else()
      CHECK_SYMBOL_EXISTS(
        CL_VERSION_${VERSION}
        "${OPENCL_INCLUDE_DIR}/CL/cl.h"
        OPENCL_VERSION_${VERSION})
    endif()

    if(OPENCL_VERSION_${VERSION})
      string(REPLACE "_" "." VERSION "${VERSION}")
      set(OPENCL_VERSION_STRING ${VERSION} PARENT_SCOPE)
      string(REGEX MATCHALL "[0-9]+" version_components "${VERSION}")
      list(GET version_components 0 major_version)
      list(GET version_components 1 minor_version)
      set(OPENCL_VERSION_MAJOR ${major_version} PARENT_SCOPE)
      set(OPENCL_VERSION_MINOR ${minor_version} PARENT_SCOPE)
      break()
    endif()
  endforeach()
  CMAKE_POP_CHECK_STATE()
endfunction()

find_path(OPENCL_INCLUDE_DIR
  NAMES
    CL/cl.h OPENCL/cl.h
  PATHS
    ENV "PROGRAMFILES(X86)"
    ENV AMDAPPSDKROOT
    ENV INTELOCLSDKROOT
    ENV NVSDKCOMPUTE_ROOT
    ENV CUDA_PATH
    ENV ATISTREAMSDKROOT
  PATH_SUFFIXES
    include
    OPENCL/common/inc
    "AMD APP/include")

_FIND_OPENCL_VERSION()

if(WIN32)
  if(CMAKE_SIZEOF_VOID_P EQUAL 4)
    find_library(OPENCL_LIBRARY
      NAMES OpenCL
      PATHS
        ENV "PROGRAMFILES(X86)"
        ENV AMDAPPSDKROOT
        ENV INTELOCLSDKROOT
        ENV CUDA_PATH
        ENV NVSDKCOMPUTE_ROOT
        ENV ATISTREAMSDKROOT
      PATH_SUFFIXES
        "AMD APP/lib/x86"
        lib/x86
        lib/Win32
        OpenCL/common/lib/Win32)
  elseif(CMAKE_SIZEOF_VOID_P EQUAL 8)
    find_library(OPENCL_LIBRARY
      NAMES OpenCL
      PATHS
        ENV "PROGRAMFILES(X86)"
        ENV AMDAPPSDKROOT
        ENV INTELOCLSDKROOT
        ENV CUDA_PATH
        ENV NVSDKCOMPUTE_ROOT
        ENV ATISTREAMSDKROOT
      PATH_SUFFIXES
        "AMD APP/lib/x86_64"
        lib/x86_64
        lib/x64
        OpenCL/common/lib/x64)
  endif()
else()
  find_library(OPENCL_LIBRARY
    NAMES OpenCL
    PATHS
      ENV AMDAPPSDKROOT
      ENV INTELOCLSDKROOT
      ENV CUDA_PATH
      ENV NVSDKCOMPUTE_ROOT
      ENV ATISTREAMSDKROOT
      /usr
      /usr/local
    PATH_SUFFIXES
      lib/x86_64
      lib/x64
      lib)
endif()

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(
  OpenCL
  FOUND_VAR OPENCL_FOUND
  REQUIRED_VARS OPENCL_LIBRARY OPENCL_INCLUDE_DIR
  VERSION_VAR OPENCL_VERSION_STRING)

mark_as_advanced(
  OPENCL_INCLUDE_DIR
  OPENCL_LIBRARY)

if(OPENCL_FOUND)
  set(OPENCL_LIBRARIES ${OPENCL_LIBRARY})
  message(STATUS " - Includes: ${OPENCL_INCLUDE_DIR}")
endif(OPENCL_FOUND)
