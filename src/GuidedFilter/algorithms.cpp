/*! \file algorithms.cpp
 *  \brief Defines classes that organize the execution of OpenCL kernels.
 *  \details Each class hides the details of the execution of a kernel. They
 *           initialize the necessary buffers, set up the workspaces, and 
 *           run the kernels.
 *  \author Nick Lamprianidis
 *  \version 1.2.0
 *  \date 2015
 *  \copyright The MIT License (MIT)
 *  \par
 *  Copyright (c) 2015 Nick Lamprianidis
 *  \par
 *  Permission is hereby granted, free of charge, to any person obtaining a copy
 *  of this software and associated documentation files (the "Software"), to deal
 *  in the Software without restriction, including without limitation the rights
 *  to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 *  copies of the Software, and to permit persons to whom the Software is
 *  furnished to do so, subject to the following conditions:
 *  \par
 *  The above copyright notice and this permission notice shall be included in
 *  all copies or substantial portions of the Software.
 *  \par
 *  THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 *  IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 *  FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 *  AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 *  LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 *  OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 *  THE SOFTWARE.
 */

#include <iostream>
#include <sstream>
#include <cmath>
#include <CLUtils.hpp>
#include <GuidedFilter/algorithms.hpp>
#include <GuidedFilter/math.hpp>


/*! \note All the classes assume there is a fully configured `clutils::CLEnv` 
 *        environment. This means, there is a known context on which they will 
 *        operate, there is a known command queue which they will use, and all 
 *        the necessary kernel code has been compiled. For more info on **CLUtils**, 
 *        you can check the [online documentation](https://clutils.nlamprian.me/).
 */
namespace cl_algo
{
namespace GF
{

    /*! \param[in] _env opencl environment.
     *  \param[in] _info opencl configuration. Specifies the context, queue, etc, to be used.
     */
    SeparateRGB<SeparateRGBConfig::FLOAT_FLOAT>::SeparateRGB (
        clutils::CLEnv &_env, clutils::CLEnvInfo<1> _info) : 
        env (_env), info (_info), 
        context (env.getContext (info.pIdx)), 
        queue (env.getQueue (info.ctxIdx, info.qIdx[0])), 
        kernel (env.getProgram (info.pgIdx), "separateRGBChannels_Float2Float")
    {
        wgMultiple = kernel.getWorkGroupInfo
            <CL_KERNEL_PREFERRED_WORK_GROUP_SIZE_MULTIPLE> (env.devices[info.pIdx][info.dIdx]);
    }


    /*! \details This interface exists to allow CL memory sharing between different kernels.
     *
     *  \param[in] mem enumeration value specifying the requested memory object.
     *  \return A reference to the requested memory object.
     */
    cl::Memory& SeparateRGB<SeparateRGBConfig::FLOAT_FLOAT>::get (SeparateRGB::Memory mem)
    {
        switch (mem)
        {
            case SeparateRGB::Memory::H_IN:
                return hBufferIn;
            case SeparateRGB::Memory::H_OUT_R:
                return hBufferOutR;
            case SeparateRGB::Memory::H_OUT_G:
                return hBufferOutG;
            case SeparateRGB::Memory::H_OUT_B:
                return hBufferOutB;
            case SeparateRGB::Memory::D_IN:
                return dBufferIn;
            case SeparateRGB::Memory::D_OUT_R:
                return dBufferOutR;
            case SeparateRGB::Memory::D_OUT_G:
                return dBufferOutG;
            case SeparateRGB::Memory::D_OUT_B:
                return dBufferOutB;
        }
    }


    /*! \details Sets up memory objects as necessary, and defines the kernel workspaces.
     *  \note If you have assigned a memory object to one member variable of the class 
     *        before the call to `init`, then that memory will be maintained. Otherwise, 
     *        a new memory object will be created.
     *        
     *  \param[in] _width width (\#pixels/row) of the array to be processed.
     *  \param[in] _height height (\#pixels/column) of the array to be processed.
     *  \param[in] _staging flag to indicate whether or not to instantiate the staging buffers.
     */
    void SeparateRGB<SeparateRGBConfig::FLOAT_FLOAT>::init (
        unsigned int _width, unsigned int _height, Staging _staging)
    {
        // Logically rearrange data (pixels)
        width = 3; height = _width * _height;
        bufferInSize = width * height * sizeof (cl_float);
        bufferOutSize = width * height * sizeof (cl_float) / 3;
        staging = _staging;

        try
        {
            if (height == 0)
                throw "The image cannot have zeroed dimensions";
            if (height % 3 != 0)
                throw "The number of pixels in the image must be a multiple of 3";
        }
        catch (const char *error)
        {
            std::cerr << "Error[SeparateRGB<SeparateRGBConfig::FLOAT_FLOAT>]: " << error << std::endl;
            exit (EXIT_FAILURE);
        }

        size_t wgM = wgMultiple;
        while (height % (3 * wgM) != 0) wgM >>= 1;

        // Set workspaces
        global = cl::NDRange (height);
        local = cl::NDRange (3 * wgM);

        if (local[0] % wgMultiple)
            std::cout << "Warning[SeparateRGB<SeparateRGBConfig::FLOAT_FLOAT>]: "
                      << "The work-group size [" << local[0] 
                      << "] is not a multiple of the preferred size [" 
                      << wgMultiple << "] on this device" << std::endl;

        // Create staging buffers
        bool io = false;
        switch (staging)
        {
            case Staging::NONE:
                hPtrIn = nullptr;
                hPtrOutR = nullptr;
                hPtrOutG = nullptr;
                hPtrOutB = nullptr;
                break;

            case Staging::IO:
                io = true;

            case Staging::I:
                if (hBufferIn () == nullptr)
                    hBufferIn = cl::Buffer (context, CL_MEM_ALLOC_HOST_PTR, bufferInSize);

                hPtrIn = (cl_float *) queue.enqueueMapBuffer (
                    hBufferIn, CL_FALSE, CL_MAP_WRITE, 0, bufferInSize);
                queue.enqueueUnmapMemObject (hBufferIn, hPtrIn);

                if (!io)
                {
                    queue.finish ();
                    hPtrOutR = nullptr;
                    hPtrOutG = nullptr;
                    hPtrOutB = nullptr;
                    break;
                }

            case Staging::O:
                if (hBufferOutR () == nullptr)
                    hBufferOutR = cl::Buffer (context, CL_MEM_ALLOC_HOST_PTR, bufferOutSize);
                if (hBufferOutG () == nullptr)
                    hBufferOutG = cl::Buffer (context, CL_MEM_ALLOC_HOST_PTR, bufferOutSize);
                if (hBufferOutB () == nullptr)
                    hBufferOutB = cl::Buffer (context, CL_MEM_ALLOC_HOST_PTR, bufferOutSize);

                hPtrOutR = (cl_float *) queue.enqueueMapBuffer (
                    hBufferOutR, CL_FALSE, CL_MAP_READ, 0, bufferOutSize);
                hPtrOutG = (cl_float *) queue.enqueueMapBuffer (
                    hBufferOutG, CL_FALSE, CL_MAP_READ, 0, bufferOutSize);
                hPtrOutB = (cl_float *) queue.enqueueMapBuffer (
                    hBufferOutB, CL_FALSE, CL_MAP_READ, 0, bufferOutSize);
                queue.enqueueUnmapMemObject (hBufferOutR, hPtrOutR);
                queue.enqueueUnmapMemObject (hBufferOutG, hPtrOutG);
                queue.enqueueUnmapMemObject (hBufferOutB, hPtrOutB);
                queue.finish ();

                if (!io) hPtrIn = nullptr;
                break;
        }
        
        // Create device buffers
        if (dBufferIn () == nullptr)
            dBufferIn = cl::Buffer (context, CL_MEM_READ_ONLY, bufferInSize);
        if (dBufferOutR () == nullptr)
            dBufferOutR = cl::Buffer (context, CL_MEM_WRITE_ONLY, bufferOutSize);
        if (dBufferOutG () == nullptr)
            dBufferOutG = cl::Buffer (context, CL_MEM_WRITE_ONLY, bufferOutSize);
        if (dBufferOutB () == nullptr)
            dBufferOutB = cl::Buffer (context, CL_MEM_WRITE_ONLY, bufferOutSize);

        // Set kernel arguments
        kernel.setArg (0, dBufferIn);
        kernel.setArg (1, dBufferOutR);
        kernel.setArg (2, dBufferOutG);
        kernel.setArg (3, dBufferOutB);
        kernel.setArg (4, cl::Local (3 * local[0] * sizeof (cl_float)));
    }


    /*! \details The transfer happens from a staging buffer on the host to the 
     *           associated (specified) device buffer.
     *  
     *  \param[in] mem enumeration value specifying an input device buffer.
     *  \param[in] ptr a pointer to an array holding input data. If not NULL, the 
     *                 data from `ptr` will be copied to the associated staging buffer.
     *  \param[in] block a flag to indicate whether to perform a blocking 
     *                   or a non-blocking operation.
     *  \param[in] events a wait-list of events.
     *  \param[out] event event associated with the write operation to the device buffer.
     */
    void SeparateRGB<SeparateRGBConfig::FLOAT_FLOAT>::write (
        SeparateRGB::Memory mem, void *ptr, bool block, const std::vector<cl::Event> *events, cl::Event *event)
    {
        if (staging == Staging::I || staging == Staging::IO)
        {
            switch (mem)
            {
                case SeparateRGB::Memory::D_IN:
                    if (ptr != nullptr)
                        std::copy ((cl_float *) ptr, (cl_float *) ptr + width * height, hPtrIn);
                    queue.enqueueWriteBuffer (dBufferIn, block, 0, bufferInSize, hPtrIn, events, event);
                    break;
                default:
                    break;
            }
        }
    }


    /*! \details The transfer happens from a device buffer to the associated 
     *           (specified) staging buffer on the host.
     *  
     *  \param[in] mem enumeration value specifying an output staging buffer.
     *  \param[in] block a flag to indicate whether to perform a blocking 
     *                   or a non-blocking operation.
     *  \param[in] events a wait-list of events.
     *  \param[out] event event associated with the read operation to the staging buffer.
     */
    void* SeparateRGB<SeparateRGBConfig::FLOAT_FLOAT>::read (
        SeparateRGB::Memory mem, bool block, const std::vector<cl::Event> *events, cl::Event *event)
    {
        if (staging == Staging::O || staging == Staging::IO)
        {
            switch (mem)
            {
                case SeparateRGB::Memory::H_OUT_R:
                    queue.enqueueReadBuffer (dBufferOutR, block, 0, bufferOutSize, hPtrOutR, events, event);
                    return hPtrOutR;
                case SeparateRGB::Memory::H_OUT_G:
                    queue.enqueueReadBuffer (dBufferOutG, block, 0, bufferOutSize, hPtrOutG, events, event);
                    return hPtrOutG;
                case SeparateRGB::Memory::H_OUT_B:
                    queue.enqueueReadBuffer (dBufferOutB, block, 0, bufferOutSize, hPtrOutB, events, event);
                    return hPtrOutB;
                default:
                    return nullptr;
            }
        }
        return nullptr;
    }


    /*! \details The function call is non-blocking.
     *
     *  \param[in] events a wait-list of events.
     *  \param[out] event event associated with the last kernel execution.
     */
    void SeparateRGB<SeparateRGBConfig::FLOAT_FLOAT>::run (const std::vector<cl::Event> *events, cl::Event *event)
    {
        queue.enqueueNDRangeKernel (kernel, cl::NullRange, global, local, events, event);
    }


    /*! \param[in] _env opencl environment.
     *  \param[in] _info opencl configuration. Specifies the context, queue, etc, to be used.
     */
    SeparateRGB<SeparateRGBConfig::UCHAR_FLOAT>::SeparateRGB (
        clutils::CLEnv &_env, clutils::CLEnvInfo<1> _info) : 
        env (_env), info (_info), 
        context (env.getContext (info.pIdx)), 
        queue (env.getQueue (info.ctxIdx, info.qIdx[0])), 
        kernel (env.getProgram (info.pgIdx), "separateRGBChannels_Uchar2Float")
    {
        wgMultiple = kernel.getWorkGroupInfo
            <CL_KERNEL_PREFERRED_WORK_GROUP_SIZE_MULTIPLE> (env.devices[info.pIdx][info.dIdx]);
    }


    /*! \details This interface exists to allow CL memory sharing between different kernels.
     *
     *  \param[in] mem enumeration value specifying the requested memory object.
     *  \return A reference to the requested memory object.
     */
    cl::Memory& SeparateRGB<SeparateRGBConfig::UCHAR_FLOAT>::get (SeparateRGB::Memory mem)
    {
        switch (mem)
        {
            case SeparateRGB::Memory::H_IN:
                return hBufferIn;
            case SeparateRGB::Memory::H_OUT_R:
                return hBufferOutR;
            case SeparateRGB::Memory::H_OUT_G:
                return hBufferOutG;
            case SeparateRGB::Memory::H_OUT_B:
                return hBufferOutB;
            case SeparateRGB::Memory::D_IN:
                return dBufferIn;
            case SeparateRGB::Memory::D_OUT_R:
                return dBufferOutR;
            case SeparateRGB::Memory::D_OUT_G:
                return dBufferOutG;
            case SeparateRGB::Memory::D_OUT_B:
                return dBufferOutB;
        }
    }


    /*! \details Sets up memory objects as necessary, and defines the kernel workspaces.
     *  \note If you have assigned a memory object to one member variable of the class 
     *        before the call to `init`, then that memory will be maintained. Otherwise, 
     *        a new memory object will be created.
     *        
     *  \param[in] _width width (\#pixels/row) of the array to be processed.
     *  \param[in] _height height (\#pixels/column) of the array to be processed.
     *  \param[in] _staging flag to indicate whether or not to instantiate the staging buffers.
     */
    void SeparateRGB<SeparateRGBConfig::UCHAR_FLOAT>::init (
        unsigned int _width, unsigned int _height, Staging _staging)
    {
        // Logically rearrange data (pixels)
        width = 3; height = _width * _height;
        bufferInSize = width * height * sizeof (cl_uchar);
        bufferOutSize = width * height * sizeof (cl_float) / 3;
        staging = _staging;

        try
        {
            if (height == 0)
                throw "The image cannot have zeroed dimensions";
            if (height % 3 != 0)
                throw "The number of pixels in the image must be a multiple of 3";
        }
        catch (const char *error)
        {
            std::cerr << "Error[SeparateRGB<SeparateRGBConfig::UCHAR_FLOAT>]: " << error << std::endl;
            exit (EXIT_FAILURE);
        }

        size_t wgM = wgMultiple;
        while (height % (3 * wgM) != 0) wgM >>= 1;

        // Set workspaces
        global = cl::NDRange (height);
        local = cl::NDRange (3 * wgM);

        if (local[0] % wgMultiple)
            std::cout << "Warning[SeparateRGB<SeparateRGBConfig::UCHAR_FLOAT>]: "
                      << "The work-group size [" << local[0] 
                      << "] is not a multiple of the preferred size [" 
                      << wgMultiple << "] on this device" << std::endl;

        // Create staging buffers
        bool io = false;
        switch (staging)
        {
            case Staging::NONE:
                hPtrIn = nullptr;
                hPtrOutR = nullptr;
                hPtrOutG = nullptr;
                hPtrOutB = nullptr;
                break;

            case Staging::IO:
                io = true;

            case Staging::I:
                if (hBufferIn () == nullptr)
                    hBufferIn = cl::Buffer (context, CL_MEM_ALLOC_HOST_PTR, bufferInSize);

                hPtrIn = (cl_uchar *) queue.enqueueMapBuffer (
                    hBufferIn, CL_FALSE, CL_MAP_WRITE, 0, bufferInSize);
                queue.enqueueUnmapMemObject (hBufferIn, hPtrIn);

                if (!io)
                {
                    queue.finish ();
                    hPtrOutR = nullptr;
                    hPtrOutG = nullptr;
                    hPtrOutB = nullptr;
                    break;
                }

            case Staging::O:
                if (hBufferOutR () == nullptr)
                    hBufferOutR = cl::Buffer (context, CL_MEM_ALLOC_HOST_PTR, bufferOutSize);
                if (hBufferOutG () == nullptr)
                    hBufferOutG = cl::Buffer (context, CL_MEM_ALLOC_HOST_PTR, bufferOutSize);
                if (hBufferOutB () == nullptr)
                    hBufferOutB = cl::Buffer (context, CL_MEM_ALLOC_HOST_PTR, bufferOutSize);

                hPtrOutR = (cl_float *) queue.enqueueMapBuffer (
                    hBufferOutR, CL_FALSE, CL_MAP_READ, 0, bufferOutSize);
                hPtrOutG = (cl_float *) queue.enqueueMapBuffer (
                    hBufferOutG, CL_FALSE, CL_MAP_READ, 0, bufferOutSize);
                hPtrOutB = (cl_float *) queue.enqueueMapBuffer (
                    hBufferOutB, CL_FALSE, CL_MAP_READ, 0, bufferOutSize);
                queue.enqueueUnmapMemObject (hBufferOutR, hPtrOutR);
                queue.enqueueUnmapMemObject (hBufferOutG, hPtrOutG);
                queue.enqueueUnmapMemObject (hBufferOutB, hPtrOutB);
                queue.finish ();

                if (!io) hPtrIn = nullptr;
                break;
        }
        
        // Create device buffers
        if (dBufferIn () == nullptr)
            dBufferIn = cl::Buffer (context, CL_MEM_READ_ONLY, bufferInSize);
        if (dBufferOutR () == nullptr)
            dBufferOutR = cl::Buffer (context, CL_MEM_WRITE_ONLY, bufferOutSize);
        if (dBufferOutG () == nullptr)
            dBufferOutG = cl::Buffer (context, CL_MEM_WRITE_ONLY, bufferOutSize);
        if (dBufferOutB () == nullptr)
            dBufferOutB = cl::Buffer (context, CL_MEM_WRITE_ONLY, bufferOutSize);

        // Set kernel arguments
        kernel.setArg (0, dBufferIn);
        kernel.setArg (1, dBufferOutR);
        kernel.setArg (2, dBufferOutG);
        kernel.setArg (3, dBufferOutB);
        kernel.setArg (4, cl::Local (3 * local[0] * sizeof (cl_uchar)));
    }


    /*! \details The transfer happens from a staging buffer on the host to the 
     *           associated (specified) device buffer.
     *  
     *  \param[in] mem enumeration value specifying an input device buffer.
     *  \param[in] ptr a pointer to an array holding input data. If not NULL, the 
     *                 data from `ptr` will be copied to the associated staging buffer.
     *  \param[in] block a flag to indicate whether to perform a blocking 
     *                   or a non-blocking operation.
     *  \param[in] events a wait-list of events.
     *  \param[out] event event associated with the write operation to the device buffer.
     */
    void SeparateRGB<SeparateRGBConfig::UCHAR_FLOAT>::write (
        SeparateRGB::Memory mem, void *ptr, bool block, const std::vector<cl::Event> *events, cl::Event *event)
    {
        if (staging == Staging::I || staging == Staging::IO)
        {
            switch (mem)
            {
                case SeparateRGB::Memory::D_IN:
                    if (ptr != nullptr)
                        std::copy ((cl_uchar *) ptr, (cl_uchar *) ptr + width * height, hPtrIn);
                    queue.enqueueWriteBuffer (dBufferIn, block, 0, bufferInSize, hPtrIn, events, event);
                    break;
                default:
                    break;
            }
        }
    }


    /*! \details The transfer happens from a device buffer to the associated 
     *           (specified) staging buffer on the host.
     *  
     *  \param[in] mem enumeration value specifying an output staging buffer.
     *  \param[in] block a flag to indicate whether to perform a blocking 
     *                   or a non-blocking operation.
     *  \param[in] events a wait-list of events.
     *  \param[out] event event associated with the read operation to the staging buffer.
     */
    void* SeparateRGB<SeparateRGBConfig::UCHAR_FLOAT>::read (
        SeparateRGB::Memory mem, bool block, const std::vector<cl::Event> *events, cl::Event *event)
    {
        if (staging == Staging::O || staging == Staging::IO)
        {
            switch (mem)
            {
                case SeparateRGB::Memory::H_OUT_R:
                    queue.enqueueReadBuffer (dBufferOutR, block, 0, bufferOutSize, hPtrOutR, events, event);
                    return hPtrOutR;
                case SeparateRGB::Memory::H_OUT_G:
                    queue.enqueueReadBuffer (dBufferOutG, block, 0, bufferOutSize, hPtrOutG, events, event);
                    return hPtrOutG;
                case SeparateRGB::Memory::H_OUT_B:
                    queue.enqueueReadBuffer (dBufferOutB, block, 0, bufferOutSize, hPtrOutB, events, event);
                    return hPtrOutB;
                default:
                    return nullptr;
            }
        }
        return nullptr;
    }


    /*! \details The function call is non-blocking.
     *
     *  \param[in] events a wait-list of events.
     *  \param[out] event event associated with the last kernel execution.
     */
    void SeparateRGB<SeparateRGBConfig::UCHAR_FLOAT>::run (const std::vector<cl::Event> *events, cl::Event *event)
    {
        queue.enqueueNDRangeKernel (kernel, cl::NullRange, global, local, events, event);
    }


    /*! \param[in] _env opencl environment.
     *  \param[in] _info opencl configuration. Specifies the context, queue, etc, to be used.
     */
    CombineRGB<CombineRGBConfig::FLOAT_FLOAT>::CombineRGB (clutils::CLEnv &_env, clutils::CLEnvInfo<1> _info) : 
        env (_env), info (_info), 
        context (env.getContext (info.pIdx)), 
        queue (env.getQueue (info.ctxIdx, info.qIdx[0])), 
        kernel (env.getProgram (info.pgIdx), "combineRGBChannels_Float2Float")
    {
        wgMultiple = kernel.getWorkGroupInfo
            <CL_KERNEL_PREFERRED_WORK_GROUP_SIZE_MULTIPLE> (env.devices[info.pIdx][info.dIdx]);
    }


    /*! \details This interface exists to allow CL memory sharing between different kernels.
     *
     *  \param[in] mem enumeration value specifying the requested memory object.
     *  \return A reference to the requested memory object.
     */
    cl::Memory& CombineRGB<CombineRGBConfig::FLOAT_FLOAT>::get (CombineRGB::Memory mem)
    {
        switch (mem)
        {
            case CombineRGB::Memory::H_IN_R:
                return hBufferInR;
            case CombineRGB::Memory::H_IN_G:
                return hBufferInG;
            case CombineRGB::Memory::H_IN_B:
                return hBufferInB;
            case CombineRGB::Memory::H_OUT:
                return hBufferOut;
            case CombineRGB::Memory::D_IN_R:
                return dBufferInR;
            case CombineRGB::Memory::D_IN_G:
                return dBufferInG;
            case CombineRGB::Memory::D_IN_B:
                return dBufferInB;
            case CombineRGB::Memory::D_OUT:
                return dBufferOut;
        }
    }


    /*! \details Sets up memory objects as necessary, and defines the kernel workspaces.
     *  \note If you have assigned a memory object to one member variable of the class 
     *        before the call to `init`, then that memory will be maintained. Otherwise, 
     *        a new memory object will be created.
     *        
     *  \param[in] _width width (\#pixels/row) of the array to be processed.
     *  \param[in] _height height (\#pixels/column) of the array to be processed.
     *  \param[in] _staging flag to indicate whether or not to instantiate the staging buffers.
     */
    void CombineRGB<CombineRGBConfig::FLOAT_FLOAT>::init (
        unsigned int _width, unsigned int _height, Staging _staging)
    {
        // Logically rearrange data (pixels)
        width = _width * _height; height = 3;
        bufferInSize = width * height * sizeof (cl_float) / 3;
        bufferOutSize = width * height * sizeof (cl_float);
        staging = _staging;

        try
        {
            if (width == 0)
                throw "The image cannot have zeroed dimensions";
            if (width % 3 != 0)
                throw "The number of pixels in the image must be a multiple of 3";
        }
        catch (const char *error)
        {
            std::cerr << "Error[CombineRGB<CombineRGBConfig::FLOAT_FLOAT>]: " << error << std::endl;
            exit (EXIT_FAILURE);
        }

        size_t wgM = wgMultiple;
        while (width % (3 * wgM) != 0) wgM >>= 1;

        // Set workspaces
        global = cl::NDRange (width);
        local = cl::NDRange (3 * wgM);

        if (local[0] % wgMultiple)
            std::cout << "Warning[CombineRGB<CombineRGBConfig::FLOAT_FLOAT>]: "
                      << "The work-group size [" << local[0] 
                      << "] is not a multiple of the preferred size [" 
                      << wgMultiple << "] on this device" << std::endl;

        // Create staging buffers
        bool io = false;
        switch (staging)
        {
            case Staging::NONE:
                hPtrInR = nullptr;
                hPtrInG = nullptr;
                hPtrInB = nullptr;
                hPtrOut = nullptr;
                break;

            case Staging::IO:
                io = true;

            case Staging::I:
                if (hBufferInR () == nullptr)
                    hBufferInR = cl::Buffer (context, CL_MEM_ALLOC_HOST_PTR, bufferInSize);
                if (hBufferInG () == nullptr)
                    hBufferInG = cl::Buffer (context, CL_MEM_ALLOC_HOST_PTR, bufferInSize);
                if (hBufferInB () == nullptr)
                    hBufferInB = cl::Buffer (context, CL_MEM_ALLOC_HOST_PTR, bufferInSize);

                hPtrInR = (cl_float *) queue.enqueueMapBuffer (
                    hBufferInR, CL_FALSE, CL_MAP_WRITE, 0, bufferInSize);
                hPtrInG = (cl_float *) queue.enqueueMapBuffer (
                    hBufferInG, CL_FALSE, CL_MAP_WRITE, 0, bufferInSize);
                hPtrInB = (cl_float *) queue.enqueueMapBuffer (
                    hBufferInB, CL_FALSE, CL_MAP_WRITE, 0, bufferInSize);
                queue.enqueueUnmapMemObject (hBufferInR, hPtrInR);
                queue.enqueueUnmapMemObject (hBufferInG, hPtrInG);
                queue.enqueueUnmapMemObject (hBufferInB, hPtrInB);

                if (!io)
                {
                    queue.finish ();
                    hPtrOut = nullptr;
                    break;
                }

            case Staging::O:
                if (hBufferOut () == nullptr)
                    hBufferOut = cl::Buffer (context, CL_MEM_ALLOC_HOST_PTR, bufferOutSize);

                hPtrOut = (cl_float *) queue.enqueueMapBuffer (
                    hBufferOut, CL_FALSE, CL_MAP_READ, 0, bufferOutSize);
                queue.enqueueUnmapMemObject (hBufferOut, hPtrOut);
                queue.finish ();

                if (!io)
                {
                    hPtrInR = nullptr;
                    hPtrInG = nullptr;
                    hPtrInB = nullptr;
                }
                break;
        }
        
        // Create device buffers
        if (dBufferInR () == nullptr)
            dBufferInR = cl::Buffer (context, CL_MEM_READ_ONLY, bufferInSize);
        if (dBufferInG () == nullptr)
            dBufferInG = cl::Buffer (context, CL_MEM_READ_ONLY, bufferInSize);
        if (dBufferInB () == nullptr)
            dBufferInB = cl::Buffer (context, CL_MEM_READ_ONLY, bufferInSize);
        if (dBufferOut () == nullptr)
            dBufferOut = cl::Buffer (context, CL_MEM_WRITE_ONLY, bufferOutSize);

        // Set kernel arguments
        kernel.setArg (0, dBufferInR);
        kernel.setArg (1, dBufferInG);
        kernel.setArg (2, dBufferInB);
        kernel.setArg (3, dBufferOut);
        kernel.setArg (4, cl::Local (3 * local[0] * sizeof (cl_float)));
    }


    /*! \details The transfer happens from a staging buffer on the host to the 
     *           associated (specified) device buffer.
     *  
     *  \param[in] mem enumeration value specifying an input device buffer.
     *  \param[in] ptr a pointer to an array holding input data. If not NULL, the 
     *                 data from `ptr` will be copied to the associated staging buffer.
     *  \param[in] block a flag to indicate whether to perform a blocking 
     *                   or a non-blocking operation.
     *  \param[in] events a wait-list of events.
     *  \param[out] event event associated with the write operation to the device buffer.
     */
    void CombineRGB<CombineRGBConfig::FLOAT_FLOAT>::write (
        CombineRGB::Memory mem, void *ptr, bool block, const std::vector<cl::Event> *events, cl::Event *event)
    {
        if (staging == Staging::I || staging == Staging::IO)
        {
            switch (mem)
            {
                case CombineRGB::Memory::D_IN_R:
                    if (ptr != nullptr)
                        std::copy ((cl_float *) ptr, (cl_float *) ptr + bufferInSize / sizeof (cl_float), hPtrInR);
                    queue.enqueueWriteBuffer (dBufferInR, block, 0, bufferInSize, hPtrInR, events, event);
                    break;
                case CombineRGB::Memory::D_IN_G:
                    if (ptr != nullptr)
                        std::copy ((cl_float *) ptr, (cl_float *) ptr + bufferInSize / sizeof (cl_float), hPtrInG);
                    queue.enqueueWriteBuffer (dBufferInG, block, 0, bufferInSize, hPtrInG, events, event);
                    break;
                case CombineRGB::Memory::D_IN_B:
                    if (ptr != nullptr)
                        std::copy ((cl_float *) ptr, (cl_float *) ptr + bufferInSize / sizeof (cl_float), hPtrInB);
                    queue.enqueueWriteBuffer (dBufferInB, block, 0, bufferInSize, hPtrInB, events, event);
                    break;
                default:
                    break;
            }
        }
    }


    /*! \details The transfer happens from a device buffer to the associated 
     *           (specified) staging buffer on the host.
     *  
     *  \param[in] mem enumeration value specifying an output staging buffer.
     *  \param[in] block a flag to indicate whether to perform a blocking 
     *                   or a non-blocking operation.
     *  \param[in] events a wait-list of events.
     *  \param[out] event event associated with the read operation to the staging buffer.
     */
    void* CombineRGB<CombineRGBConfig::FLOAT_FLOAT>::read (
        CombineRGB::Memory mem, bool block, const std::vector<cl::Event> *events, cl::Event *event)
    {
        if (staging == Staging::O || staging == Staging::IO)
        {
            switch (mem)
            {
                case CombineRGB::Memory::H_OUT:
                    queue.enqueueReadBuffer (dBufferOut, block, 0, bufferOutSize, hPtrOut, events, event);
                    return hPtrOut;
                default:
                    return nullptr;
            }
        }
        return nullptr;
    }


    /*! \details The function call is non-blocking.
     *
     *  \param[in] events a wait-list of events.
     *  \param[out] event event associated with the last kernel execution.
     */
    void CombineRGB<CombineRGBConfig::FLOAT_FLOAT>::run (const std::vector<cl::Event> *events, cl::Event *event)
    {
        queue.enqueueNDRangeKernel (kernel, cl::NullRange, global, local, events, event);
    }


    /*! \param[in] _env opencl environment.
     *  \param[in] _info opencl configuration. Specifies the context, queue, etc, to be used.
     */
    CombineRGB<CombineRGBConfig::FLOAT_UCHAR>::CombineRGB (clutils::CLEnv &_env, clutils::CLEnvInfo<1> _info) : 
        env (_env), info (_info), 
        context (env.getContext (info.pIdx)), 
        queue (env.getQueue (info.ctxIdx, info.qIdx[0])), 
        kernel (env.getProgram (info.pgIdx), "combineRGBChannels_Float2Uchar")
    {
        wgMultiple = kernel.getWorkGroupInfo
            <CL_KERNEL_PREFERRED_WORK_GROUP_SIZE_MULTIPLE> (env.devices[info.pIdx][info.dIdx]);
    }


    /*! \details This interface exists to allow CL memory sharing between different kernels.
     *
     *  \param[in] mem enumeration value specifying the requested memory object.
     *  \return A reference to the requested memory object.
     */
    cl::Memory& CombineRGB<CombineRGBConfig::FLOAT_UCHAR>::get (CombineRGB::Memory mem)
    {
        switch (mem)
        {
            case CombineRGB::Memory::H_IN_R:
                return hBufferInR;
            case CombineRGB::Memory::H_IN_G:
                return hBufferInG;
            case CombineRGB::Memory::H_IN_B:
                return hBufferInB;
            case CombineRGB::Memory::H_OUT:
                return hBufferOut;
            case CombineRGB::Memory::D_IN_R:
                return dBufferInR;
            case CombineRGB::Memory::D_IN_G:
                return dBufferInG;
            case CombineRGB::Memory::D_IN_B:
                return dBufferInB;
            case CombineRGB::Memory::D_OUT:
                return dBufferOut;
        }
    }


    /*! \details Sets up memory objects as necessary, and defines the kernel workspaces.
     *  \note If you have assigned a memory object to one member variable of the class 
     *        before the call to `init`, then that memory will be maintained. Otherwise, 
     *        a new memory object will be created.
     *        
     *  \param[in] _width width (\#pixels/row) of the array to be processed.
     *  \param[in] _height height (\#pixels/column) of the array to be processed.
     *  \param[in] _staging flag to indicate whether or not to instantiate the staging buffers.
     */
    void CombineRGB<CombineRGBConfig::FLOAT_UCHAR>::init (
        unsigned int _width, unsigned int _height, Staging _staging)
    {
        // Logically rearrange data (pixels)
        width = _width * _height; height = 3;
        bufferInSize = width * height * sizeof (cl_float) / 3;
        bufferOutSize = width * height * sizeof (cl_uchar);
        staging = _staging;

        try
        {
            if (width == 0)
                throw "The image cannot have zeroed dimensions";
            if (width % 3 != 0)
                throw "The number of pixels in the image must be a multiple of 3";
        }
        catch (const char *error)
        {
            std::cerr << "Error[CombineRGB<CombineRGBConfig::FLOAT_UCHAR>]: " << error << std::endl;
            exit (EXIT_FAILURE);
        }

        size_t wgM = wgMultiple;
        while (width % (3 * wgM) != 0) wgM >>= 1;

        // Set workspaces
        global = cl::NDRange (width);
        local = cl::NDRange (3 * wgM);

        if (local[0] % wgMultiple)
            std::cout << "Warning[CombineRGB<CombineRGBConfig::FLOAT_UCHAR>]: "
                      << "The work-group size [" << local[0] 
                      << "] is not a multiple of the preferred size [" 
                      << wgMultiple << "] on this device" << std::endl;

        // Create staging buffers
        bool io = false;
        switch (staging)
        {
            case Staging::NONE:
                hPtrInR = nullptr;
                hPtrInG = nullptr;
                hPtrInB = nullptr;
                hPtrOut = nullptr;
                break;

            case Staging::IO:
                io = true;

            case Staging::I:
                if (hBufferInR () == nullptr)
                    hBufferInR = cl::Buffer (context, CL_MEM_ALLOC_HOST_PTR, bufferInSize);
                if (hBufferInG () == nullptr)
                    hBufferInG = cl::Buffer (context, CL_MEM_ALLOC_HOST_PTR, bufferInSize);
                if (hBufferInB () == nullptr)
                    hBufferInB = cl::Buffer (context, CL_MEM_ALLOC_HOST_PTR, bufferInSize);

                hPtrInR = (cl_float *) queue.enqueueMapBuffer (
                    hBufferInR, CL_FALSE, CL_MAP_WRITE, 0, bufferInSize);
                hPtrInG = (cl_float *) queue.enqueueMapBuffer (
                    hBufferInG, CL_FALSE, CL_MAP_WRITE, 0, bufferInSize);
                hPtrInB = (cl_float *) queue.enqueueMapBuffer (
                    hBufferInB, CL_FALSE, CL_MAP_WRITE, 0, bufferInSize);
                queue.enqueueUnmapMemObject (hBufferInR, hPtrInR);
                queue.enqueueUnmapMemObject (hBufferInG, hPtrInG);
                queue.enqueueUnmapMemObject (hBufferInB, hPtrInB);

                if (!io)
                {
                    queue.finish ();
                    hPtrOut = nullptr;
                    break;
                }

            case Staging::O:
                if (hBufferOut () == nullptr)
                    hBufferOut = cl::Buffer (context, CL_MEM_ALLOC_HOST_PTR, bufferOutSize);

                hPtrOut = (cl_uchar *) queue.enqueueMapBuffer (
                    hBufferOut, CL_FALSE, CL_MAP_READ, 0, bufferOutSize);
                queue.enqueueUnmapMemObject (hBufferOut, hPtrOut);
                queue.finish ();

                if (!io)
                {
                    hPtrInR = nullptr;
                    hPtrInG = nullptr;
                    hPtrInB = nullptr;
                }
                break;
        }
        
        // Create device buffers
        if (dBufferInR () == nullptr)
            dBufferInR = cl::Buffer (context, CL_MEM_READ_ONLY, bufferInSize);
        if (dBufferInG () == nullptr)
            dBufferInG = cl::Buffer (context, CL_MEM_READ_ONLY, bufferInSize);
        if (dBufferInB () == nullptr)
            dBufferInB = cl::Buffer (context, CL_MEM_READ_ONLY, bufferInSize);
        if (dBufferOut () == nullptr)
            dBufferOut = cl::Buffer (context, CL_MEM_WRITE_ONLY, bufferOutSize);

        // Set kernel arguments
        kernel.setArg (0, dBufferInR);
        kernel.setArg (1, dBufferInG);
        kernel.setArg (2, dBufferInB);
        kernel.setArg (3, dBufferOut);
        kernel.setArg (4, cl::Local (3 * local[0] * sizeof (cl_float)));
    }


    /*! \details The transfer happens from a staging buffer on the host to the 
     *           associated (specified) device buffer.
     *  
     *  \param[in] mem enumeration value specifying an input device buffer.
     *  \param[in] ptr a pointer to an array holding input data. If not NULL, the 
     *                 data from `ptr` will be copied to the associated staging buffer.
     *  \param[in] block a flag to indicate whether to perform a blocking 
     *                   or a non-blocking operation.
     *  \param[in] events a wait-list of events.
     *  \param[out] event event associated with the write operation to the device buffer.
     */
    void CombineRGB<CombineRGBConfig::FLOAT_UCHAR>::write (
        CombineRGB::Memory mem, void *ptr, bool block, const std::vector<cl::Event> *events, cl::Event *event)
    {
        if (staging == Staging::I || staging == Staging::IO)
        {
            switch (mem)
            {
                case CombineRGB::Memory::D_IN_R:
                    if (ptr != nullptr)
                        std::copy ((cl_float *) ptr, (cl_float *) ptr + bufferInSize / sizeof (cl_float), hPtrInR);
                    queue.enqueueWriteBuffer (dBufferInR, block, 0, bufferInSize, hPtrInR, events, event);
                    break;
                case CombineRGB::Memory::D_IN_G:
                    if (ptr != nullptr)
                        std::copy ((cl_float *) ptr, (cl_float *) ptr + bufferInSize / sizeof (cl_float), hPtrInG);
                    queue.enqueueWriteBuffer (dBufferInG, block, 0, bufferInSize, hPtrInG, events, event);
                    break;
                case CombineRGB::Memory::D_IN_B:
                    if (ptr != nullptr)
                        std::copy ((cl_float *) ptr, (cl_float *) ptr + bufferInSize / sizeof (cl_float), hPtrInB);
                    queue.enqueueWriteBuffer (dBufferInB, block, 0, bufferInSize, hPtrInB, events, event);
                    break;
                default:
                    break;
            }
        }
    }


    /*! \details The transfer happens from a device buffer to the associated 
     *           (specified) staging buffer on the host.
     *  
     *  \param[in] mem enumeration value specifying an output staging buffer.
     *  \param[in] block a flag to indicate whether to perform a blocking 
     *                   or a non-blocking operation.
     *  \param[in] events a wait-list of events.
     *  \param[out] event event associated with the read operation to the staging buffer.
     */
    void* CombineRGB<CombineRGBConfig::FLOAT_UCHAR>::read (
        CombineRGB::Memory mem, bool block, const std::vector<cl::Event> *events, cl::Event *event)
    {
        if (staging == Staging::O || staging == Staging::IO)
        {
            switch (mem)
            {
                case CombineRGB::Memory::H_OUT:
                    queue.enqueueReadBuffer (dBufferOut, block, 0, bufferOutSize, hPtrOut, events, event);
                    return hPtrOut;
                default:
                    return nullptr;
            }
        }
        return nullptr;
    }


    /*! \details The function call is non-blocking.
     *
     *  \param[in] events a wait-list of events.
     *  \param[out] event event associated with the last kernel execution.
     */
    void CombineRGB<CombineRGBConfig::FLOAT_UCHAR>::run (const std::vector<cl::Event> *events, cl::Event *event)
    {
        queue.enqueueNDRangeKernel (kernel, cl::NullRange, global, local, events, event);
    }


    /*! \param[in] _env opencl environment.
     *  \param[in] _info opencl configuration. Specifies the context, queue, etc, to be used.
     */
    Depth<DepthConfig::USHORT_FLOAT>::Depth (clutils::CLEnv &_env, clutils::CLEnvInfo<1> _info) : 
        env (_env), info (_info), 
        context (env.getContext (info.pIdx)), 
        queue (env.getQueue (info.ctxIdx, info.qIdx[0])), 
        kernel (env.getProgram (info.pgIdx), "depth_Ushort2Float")
    {
    }


    /*! \details This interface exists to allow CL memory sharing between different kernels.
     *
     *  \param[in] mem enumeration value specifying the requested memory object.
     *  \return A reference to the requested memory object.
     */
    cl::Memory& Depth<DepthConfig::USHORT_FLOAT>::get (Depth::Memory mem)
    {
        switch (mem)
        {
            case Depth::Memory::H_IN:
                return hBufferIn;
            case Depth::Memory::H_OUT:
                return hBufferOut;
            case Depth::Memory::D_IN:
                return dBufferIn;
            case Depth::Memory::D_OUT:
                return dBufferOut;
        }
    }


    /*! \details Sets up memory objects as necessary, and defines the kernel workspaces.
     *  \note If you have assigned a memory object to one member variable of the class 
     *        before the call to `init`, then that memory will be maintained. Otherwise, 
     *        a new memory object will be created.
     *        
     *  \param[in] _width width of the array to be processed.
     *  \param[in] _height height of the array to be processed.
     *  \param[in] _scaling factor by which to scale the depth values in the output array.
     *  \param[in] _staging flag to indicate whether or not to instantiate the staging buffers.
     */
    void Depth<DepthConfig::USHORT_FLOAT>::init (unsigned int _width, unsigned int _height, 
                                                 float _scaling, Staging _staging)
    {
        length = _width * _height;
        bufferInSize = length * sizeof (cl_ushort);
        bufferOutSize = length * sizeof (cl_float);
        scaling = _scaling;
        staging = _staging;

        try
        {
            if (length == 0)
                throw "The image cannot have zeroed dimensions";

            if (length % 4 != 0)
                throw "The number of elements in the array has to be a multiple of 4";
        }
        catch (const char *error)
        {
            std::cerr << "Error[Depth<DepthConfig::USHORT_FLOAT>]: " << error << std::endl;
            exit (EXIT_FAILURE);
        }

        // Set workspaces
        global = cl::NDRange (length / 4);

        // Create staging buffers
        bool io = false;
        switch (staging)
        {
            case Staging::NONE:
                hPtrIn = nullptr;
                hPtrOut = nullptr;
                break;

            case Staging::IO:
                io = true;

            case Staging::I:
                if (hBufferIn () == nullptr)
                    hBufferIn = cl::Buffer (context, CL_MEM_ALLOC_HOST_PTR, bufferInSize);

                hPtrIn = (cl_ushort *) queue.enqueueMapBuffer (
                    hBufferIn, CL_FALSE, CL_MAP_WRITE, 0, bufferInSize);
                queue.enqueueUnmapMemObject (hBufferIn, hPtrIn);

                if (!io)
                {
                    queue.finish ();
                    hPtrOut = nullptr;
                    break;
                }

            case Staging::O:
                if (hBufferOut () == nullptr)
                    hBufferOut = cl::Buffer (context, CL_MEM_ALLOC_HOST_PTR, bufferOutSize);

                hPtrOut = (cl_float *) queue.enqueueMapBuffer (
                    hBufferOut, CL_FALSE, CL_MAP_READ, 0, bufferOutSize);
                queue.enqueueUnmapMemObject (hBufferOut, hPtrOut);
                queue.finish ();

                if (!io) hPtrIn = nullptr;
                break;
        }
        
        // Create device buffers
        if (dBufferIn () == nullptr)
            dBufferIn = cl::Buffer (context, CL_MEM_READ_ONLY, bufferInSize);
        if (dBufferOut () == nullptr)
            dBufferOut = cl::Buffer (context, CL_MEM_WRITE_ONLY, bufferOutSize);

        // Set kernel arguments
        kernel.setArg (0, dBufferIn);
        kernel.setArg (1, dBufferOut);
        kernel.setArg (2, scaling);
    }


    /*! \details The transfer happens from a staging buffer on the host to the 
     *           associated (specified) device buffer.
     *  
     *  \param[in] mem enumeration value specifying an input device buffer.
     *  \param[in] ptr a pointer to an array holding input data. If not NULL, the 
     *                 data from `ptr` will be copied to the associated staging buffer.
     *  \param[in] block a flag to indicate whether to perform a blocking 
     *                   or a non-blocking operation.
     *  \param[in] events a wait-list of events.
     *  \param[out] event event associated with the write operation to the device buffer.
     */
    void Depth<DepthConfig::USHORT_FLOAT>::write (
        Depth::Memory mem, void *ptr, bool block, const std::vector<cl::Event> *events, cl::Event *event)
    {
        if (staging == Staging::I || staging == Staging::IO)
        {
            switch (mem)
            {
                case Depth::Memory::D_IN:
                    if (ptr != nullptr)
                        std::copy ((cl_ushort *) ptr, (cl_ushort *) ptr + length, hPtrIn);
                    queue.enqueueWriteBuffer (dBufferIn, block, 0, bufferInSize, hPtrIn, events, event);
                    break;
                default:
                    break;
            }
        }
    }


    /*! \details The transfer happens from a device buffer to the associated 
     *           (specified) staging buffer on the host.
     *  
     *  \param[in] mem enumeration value specifying an output staging buffer.
     *  \param[in] block a flag to indicate whether to perform a blocking 
     *                   or a non-blocking operation.
     *  \param[in] events a wait-list of events.
     *  \param[out] event event associated with the read operation to the staging buffer.
     */
    void* Depth<DepthConfig::USHORT_FLOAT>::read (
        Depth::Memory mem, bool block, const std::vector<cl::Event> *events, cl::Event *event)
    {
        if (staging == Staging::O || staging == Staging::IO)
        {
            switch (mem)
            {
                case Depth::Memory::H_OUT:
                    queue.enqueueReadBuffer (dBufferOut, block, 0, bufferOutSize, hPtrOut, events, event);
                    return hPtrOut;
                default:
                    return nullptr;
            }
        }
        return nullptr;
    }


    /*! \details The function call is non-blocking.
     *
     *  \param[in] events a wait-list of events.
     *  \param[out] event event associated with the last kernel execution.
     */
    void Depth<DepthConfig::USHORT_FLOAT>::run (const std::vector<cl::Event> *events, cl::Event *event)
    {
        queue.enqueueNDRangeKernel (kernel, cl::NullRange, global, cl::NullRange, events, event);
    }


    /*! \return The scaling factor.
     */
    float Depth<DepthConfig::USHORT_FLOAT>::getScaling ()
    {
        return scaling;
    }


    /*! \details Updates the kernel argument for the scaling factor.
     *
     *  \param[in] _scaling scaling factor.
     */
    void Depth<DepthConfig::USHORT_FLOAT>::setScaling (float _scaling)
    {
        scaling = _scaling;
        kernel.setArg (3, scaling);
    }


    /*! \param[in] _env opencl environment.
     *  \param[in] _info opencl configuration. Specifies the context, queue, etc, to be used.
     */
    DepthTo3D::DepthTo3D (clutils::CLEnv &_env, clutils::CLEnvInfo<1> _info) : 
        env (_env), info (_info), 
        context (env.getContext (info.pIdx)), 
        queue (env.getQueue (info.ctxIdx, info.qIdx[0])), 
        kernel (env.getProgram (info.pgIdx), "depthTo3D")
    {
    }


    /*! \details This interface exists to allow CL memory sharing between different kernels.
     *
     *  \param[in] mem enumeration value specifying the requested memory object.
     *  \return A reference to the requested memory object.
     */
    cl::Memory& DepthTo3D::get (DepthTo3D::Memory mem)
    {
        switch (mem)
        {
            case DepthTo3D::Memory::H_IN:
                return hBufferIn;
            case DepthTo3D::Memory::H_OUT:
                return hBufferOut;
            case DepthTo3D::Memory::D_IN:
                return dBufferIn;
            case DepthTo3D::Memory::D_OUT:
                return dBufferOut;
        }
    }


    /*! \details Sets up memory objects as necessary, and defines the kernel workspaces.
     *  \note If you have assigned a memory object to one member variable of the class 
     *        before the call to `init`, then that memory will be maintained. Otherwise, 
     *        a new memory object will be created.
     *        
     *  \param[in] _width width of the array to be processed.
     *  \param[in] _height height of the array to be processed.
     *  \param[in] _f focal length.
     *  \param[in] _scaling factor by which to scale the depth values before building the point cloud.
     *  \param[in] _staging flag to indicate whether or not to instantiate the staging buffers.
     */
    void DepthTo3D::init (unsigned int _width, unsigned int _height, float _f, float _scaling, Staging _staging)
    {
        width = _width; height = _height; f = _f;
        bufferInSize = width * height * sizeof (cl_float);
        bufferOutSize = width * height * sizeof (cl_float4);
        scaling = _scaling;
        staging = _staging;

        try
        {
            if ((width == 0) || (height == 0))
                throw "The image cannot have zeroed dimensions";
        }
        catch (const char *error)
        {
            std::cerr << "Error[DepthTo3D]: " << error << std::endl;
            exit (EXIT_FAILURE);
        }

        // Set workspaces
        global = cl::NDRange (width, height);

        // Create staging buffers
        bool io = false;
        switch (staging)
        {
            case Staging::NONE:
                hPtrIn = nullptr;
                hPtrOut = nullptr;
                break;

            case Staging::IO:
                io = true;

            case Staging::I:
                if (hBufferIn () == nullptr)
                    hBufferIn = cl::Buffer (context, CL_MEM_ALLOC_HOST_PTR, bufferInSize);

                hPtrIn = (cl_float *) queue.enqueueMapBuffer (
                    hBufferIn, CL_FALSE, CL_MAP_WRITE, 0, bufferInSize);
                queue.enqueueUnmapMemObject (hBufferIn, hPtrIn);

                if (!io)
                {
                    queue.finish ();
                    hPtrOut = nullptr;
                    break;
                }

            case Staging::O:
                if (hBufferOut () == nullptr)
                    hBufferOut = cl::Buffer (context, CL_MEM_ALLOC_HOST_PTR, bufferOutSize);

                hPtrOut = (cl_float4 *) queue.enqueueMapBuffer (
                    hBufferOut, CL_FALSE, CL_MAP_READ, 0, bufferOutSize);
                queue.enqueueUnmapMemObject (hBufferOut, hPtrOut);
                queue.finish ();

                if (!io) hPtrIn = nullptr;
                break;
        }
        
        // Create device buffers
        if (dBufferIn () == nullptr)
            dBufferIn = cl::Buffer (context, CL_MEM_READ_ONLY, bufferInSize);
        if (dBufferOut () == nullptr)
            dBufferOut = cl::Buffer (context, CL_MEM_WRITE_ONLY, bufferOutSize);

        // Set kernel arguments
        kernel.setArg (0, dBufferIn);
        kernel.setArg (1, dBufferOut);
        kernel.setArg (2, f);
        kernel.setArg (3, scaling);
    }


    /*! \details The transfer happens from a staging buffer on the host to the 
     *           associated (specified) device buffer.
     *  
     *  \param[in] mem enumeration value specifying an input device buffer.
     *  \param[in] ptr a pointer to an array holding input data. If not NULL, the 
     *                 data from `ptr` will be copied to the associated staging buffer.
     *  \param[in] block a flag to indicate whether to perform a blocking 
     *                   or a non-blocking operation.
     *  \param[in] events a wait-list of events.
     *  \param[out] event event associated with the write operation to the device buffer.
     */
    void DepthTo3D::write (DepthTo3D::Memory mem, void *ptr, bool block, 
                           const std::vector<cl::Event> *events, cl::Event *event)
    {
        if (staging == Staging::I || staging == Staging::IO)
        {
            switch (mem)
            {
                case DepthTo3D::Memory::D_IN:
                    if (ptr != nullptr)
                        std::copy ((cl_float *) ptr, (cl_float *) ptr + width * height, hPtrIn);
                    queue.enqueueWriteBuffer (dBufferIn, block, 0, bufferInSize, hPtrIn, events, event);
                    break;
                default:
                    break;
            }
        }
    }


    /*! \details The transfer happens from a device buffer to the associated 
     *           (specified) staging buffer on the host.
     *  
     *  \param[in] mem enumeration value specifying an output staging buffer.
     *  \param[in] block a flag to indicate whether to perform a blocking 
     *                   or a non-blocking operation.
     *  \param[in] events a wait-list of events.
     *  \param[out] event event associated with the read operation to the staging buffer.
     */
    void* DepthTo3D::read (DepthTo3D::Memory mem, bool block, 
                           const std::vector<cl::Event> *events, cl::Event *event)
    {
        if (staging == Staging::O || staging == Staging::IO)
        {
            switch (mem)
            {
                case DepthTo3D::Memory::H_OUT:
                    queue.enqueueReadBuffer (dBufferOut, block, 0, bufferOutSize, hPtrOut, events, event);
                    return hPtrOut;
                default:
                    return nullptr;
            }
        }
        return nullptr;
    }


    /*! \details The function call is non-blocking.
     *
     *  \param[in] events a wait-list of events.
     *  \param[out] event event associated with the last kernel execution.
     */
    void DepthTo3D::run (const std::vector<cl::Event> *events, cl::Event *event)
    {
        queue.enqueueNDRangeKernel (kernel, cl::NullRange, global, cl::NullRange, events, event);
    }


    /*! \return The focal length of the camera used.
     */
    float DepthTo3D::getFocalLength ()
    {
        return f;
    }


    /*! \details Updates the kernel argument for the focal length.
     *
     *  \param[in] _f the focal length.
     */
    void DepthTo3D::setFocalLength (float _f)
    {
        f = _f;
        kernel.setArg (2, f);
    }


    /*! \return The scaling factor.
     */
    float DepthTo3D::getScaling ()
    {
        return scaling;
    }


    /*! \details Updates the kernel argument for the scaling factor.
     *
     *  \param[in] _scaling scaling factor.
     */
    void DepthTo3D::setScaling (float _scaling)
    {
        scaling = _scaling;
        kernel.setArg (4, scaling);
    }


    /*! \param[in] _env opencl environment.
     *  \param[in] _info opencl configuration. Specifies the context, queue, etc, to be used.
     */
    RGBDTo8D::RGBDTo8D (clutils::CLEnv &_env, clutils::CLEnvInfo<1> _info) : 
        env (_env), info (_info), 
        context (env.getContext (info.pIdx)), 
        queue (env.getQueue (info.ctxIdx, info.qIdx[0])), 
        kernel (env.getProgram (info.pgIdx), "rgbdTo8D")
    {
        wgMultiple = kernel.getWorkGroupInfo
            <CL_KERNEL_PREFERRED_WORK_GROUP_SIZE_MULTIPLE> (env.devices[info.pIdx][info.dIdx]);
    }


    /*! \details This interface exists to allow CL memory sharing between different kernels.
     *
     *  \param[in] mem enumeration value specifying the requested memory object.
     *  \return A reference to the requested memory object.
     */
    cl::Memory& RGBDTo8D::get (RGBDTo8D::Memory mem)
    {
        switch (mem)
        {
            case RGBDTo8D::Memory::H_IN_D:
                return hBufferInD;
            case RGBDTo8D::Memory::H_IN_R:
                return hBufferInR;
            case RGBDTo8D::Memory::H_IN_G:
                return hBufferInG;
            case RGBDTo8D::Memory::H_IN_B:
                return hBufferInB;
            case RGBDTo8D::Memory::H_OUT:
                return hBufferOut;
            case RGBDTo8D::Memory::D_IN_D:
                return dBufferInD;
            case RGBDTo8D::Memory::D_IN_R:
                return dBufferInR;
            case RGBDTo8D::Memory::D_IN_G:
                return dBufferInG;
            case RGBDTo8D::Memory::D_IN_B:
                return dBufferInB;
            case RGBDTo8D::Memory::D_OUT:
                return dBufferOut;
        }
    }


    /*! \details Sets up memory objects as necessary, and defines the kernel workspaces.
     *  \note If you have assigned a memory object to one member variable of the class 
     *        before the call to `init`, then that memory will be maintained. Otherwise, 
     *        a new memory object will be created.
     *        
     *  \param[in] _width width of the array to be processed.
     *  \param[in] _height height of the array to be processed.
     *  \param[in] _f focal length.
     *  \param[in] _scaling factor by which to scale the depth values before building the point cloud.
     *  \param[in] _rgbNorm flag to indicate whether to perform RGB normalization.
     *  \param[in] _staging flag to indicate whether or not to instantiate the staging buffers.
     */
    void RGBDTo8D::init (unsigned int _width, unsigned int _height, float _f, float _scaling, int _rgbNorm, Staging _staging)
    {
        width = _width; height = _height; f = _f;
        points = width * height;
        bufferInSize = points * sizeof (cl_float);
        bufferOutSize = points * sizeof (cl_float8);
        scaling = _scaling;
        rgbNorm = _rgbNorm;
        staging = _staging;

        try
        {
            if (points == 0)
                throw "The image cannot have zeroed dimensions";
            if (points % 3 != 0)
                throw "The number of pixels in the images must be a multiple of 3";
        }
        catch (const char *error)
        {
            std::cerr << "Error[RGBDTo8D]: " << error << std::endl;
            exit (EXIT_FAILURE);
        }

        size_t wgM = wgMultiple;
        while (points % (3 * wgM) != 0) wgM >>= 1;

        // Set workspaces
        global = cl::NDRange (points);
        local = cl::NDRange (3 * wgM);

        if (local[0] % wgMultiple)
            std::cout << "Warning[RGBDTo8D]: "
                      << "The work-group size [" << local[0] 
                      << "] is not a multiple of the preferred size [" 
                      << wgMultiple << "] on this device" << std::endl;

        // Create staging buffers
        bool io = false;
        switch (staging)
        {
            case Staging::NONE:
                hPtrInD = nullptr;
                hPtrInR = nullptr;
                hPtrInG = nullptr;
                hPtrInB = nullptr;
                hPtrOut = nullptr;
                break;

            case Staging::IO:
                io = true;

            case Staging::I:
                if (hBufferInD () == nullptr)
                    hBufferInD = cl::Buffer (context, CL_MEM_ALLOC_HOST_PTR, bufferInSize);
                if (hBufferInR () == nullptr)
                    hBufferInR = cl::Buffer (context, CL_MEM_ALLOC_HOST_PTR, bufferInSize);
                if (hBufferInG () == nullptr)
                    hBufferInG = cl::Buffer (context, CL_MEM_ALLOC_HOST_PTR, bufferInSize);
                if (hBufferInB () == nullptr)
                    hBufferInB = cl::Buffer (context, CL_MEM_ALLOC_HOST_PTR, bufferInSize);

                hPtrInD = (cl_float *) queue.enqueueMapBuffer (
                    hBufferInD, CL_FALSE, CL_MAP_WRITE, 0, bufferInSize);
                hPtrInR = (cl_float *) queue.enqueueMapBuffer (
                    hBufferInR, CL_FALSE, CL_MAP_WRITE, 0, bufferInSize);
                hPtrInG = (cl_float *) queue.enqueueMapBuffer (
                    hBufferInG, CL_FALSE, CL_MAP_WRITE, 0, bufferInSize);
                hPtrInB = (cl_float *) queue.enqueueMapBuffer (
                    hBufferInB, CL_FALSE, CL_MAP_WRITE, 0, bufferInSize);
                queue.enqueueUnmapMemObject (hBufferInD, hPtrInD);
                queue.enqueueUnmapMemObject (hBufferInR, hPtrInR);
                queue.enqueueUnmapMemObject (hBufferInG, hPtrInG);
                queue.enqueueUnmapMemObject (hBufferInB, hPtrInB);

                if (!io)
                {
                    queue.finish ();
                    hPtrOut = nullptr;
                    break;
                }

            case Staging::O:
                if (hBufferOut () == nullptr)
                    hBufferOut = cl::Buffer (context, CL_MEM_ALLOC_HOST_PTR, bufferOutSize);

                hPtrOut = (cl_float8 *) queue.enqueueMapBuffer (
                    hBufferOut, CL_FALSE, CL_MAP_READ, 0, bufferOutSize);
                queue.enqueueUnmapMemObject (hBufferOut, hPtrOut);
                queue.finish ();

                if (!io)
                {
                    hPtrInD = nullptr;
                    hPtrInR = nullptr;
                    hPtrInG = nullptr;
                    hPtrInB = nullptr;
                }
                break;
        }
        
        // Create device buffers
        if (dBufferInD () == nullptr)
            dBufferInD = cl::Buffer (context, CL_MEM_READ_ONLY, bufferInSize);
        if (dBufferInR () == nullptr)
            dBufferInR = cl::Buffer (context, CL_MEM_READ_ONLY, bufferInSize);
        if (dBufferInG () == nullptr)
            dBufferInG = cl::Buffer (context, CL_MEM_READ_ONLY, bufferInSize);
        if (dBufferInB () == nullptr)
            dBufferInB = cl::Buffer (context, CL_MEM_READ_ONLY, bufferInSize);
        if (dBufferOut () == nullptr)
            dBufferOut = cl::Buffer (context, CL_MEM_WRITE_ONLY, bufferOutSize);

        // Set kernel arguments
        kernel.setArg (0, dBufferInD);
        kernel.setArg (1, dBufferInR);
        kernel.setArg (2, dBufferInG);
        kernel.setArg (3, dBufferInB);
        kernel.setArg (4, dBufferOut);
        kernel.setArg (5, cl::Local (3 * local[0] * sizeof (cl_float)));
        kernel.setArg (6, width);
        kernel.setArg (7, f);
        kernel.setArg (8, scaling);
        kernel.setArg (9, rgbNorm);
    }


    /*! \details The transfer happens from a staging buffer on the host to the 
     *           associated (specified) device buffer.
     *  
     *  \param[in] mem enumeration value specifying an input device buffer.
     *  \param[in] ptr a pointer to an array holding input data. If not NULL, the 
     *                 data from `ptr` will be copied to the associated staging buffer.
     *  \param[in] block a flag to indicate whether to perform a blocking 
     *                   or a non-blocking operation.
     *  \param[in] events a wait-list of events.
     *  \param[out] event event associated with the write operation to the device buffer.
     */
    void RGBDTo8D::write (RGBDTo8D::Memory mem, void *ptr, bool block, 
                           const std::vector<cl::Event> *events, cl::Event *event)
    {
        if (staging == Staging::I || staging == Staging::IO)
        {
            switch (mem)
            {
                case RGBDTo8D::Memory::D_IN_D:
                    if (ptr != nullptr)
                        std::copy ((cl_float *) ptr, (cl_float *) ptr + points, hPtrInD);
                    queue.enqueueWriteBuffer (dBufferInD, block, 0, bufferInSize, hPtrInD, events, event);
                    break;
                case RGBDTo8D::Memory::D_IN_R:
                    if (ptr != nullptr)
                        std::copy ((cl_float *) ptr, (cl_float *) ptr + points, hPtrInR);
                    queue.enqueueWriteBuffer (dBufferInR, block, 0, bufferInSize, hPtrInR, events, event);
                    break;
                case RGBDTo8D::Memory::D_IN_G:
                    if (ptr != nullptr)
                        std::copy ((cl_float *) ptr, (cl_float *) ptr + points, hPtrInG);
                    queue.enqueueWriteBuffer (dBufferInG, block, 0, bufferInSize, hPtrInG, events, event);
                    break;
                case RGBDTo8D::Memory::D_IN_B:
                    if (ptr != nullptr)
                        std::copy ((cl_float *) ptr, (cl_float *) ptr + points, hPtrInB);
                    queue.enqueueWriteBuffer (dBufferInB, block, 0, bufferInSize, hPtrInB, events, event);
                    break;
                default:
                    break;
            }
        }
    }


    /*! \details The transfer happens from a device buffer to the associated 
     *           (specified) staging buffer on the host.
     *  
     *  \param[in] mem enumeration value specifying an output staging buffer.
     *  \param[in] block a flag to indicate whether to perform a blocking 
     *                   or a non-blocking operation.
     *  \param[in] events a wait-list of events.
     *  \param[out] event event associated with the read operation to the staging buffer.
     */
    void* RGBDTo8D::read (RGBDTo8D::Memory mem, bool block, 
                           const std::vector<cl::Event> *events, cl::Event *event)
    {
        if (staging == Staging::O || staging == Staging::IO)
        {
            switch (mem)
            {
                case RGBDTo8D::Memory::H_OUT:
                    queue.enqueueReadBuffer (dBufferOut, block, 0, bufferOutSize, hPtrOut, events, event);
                    return hPtrOut;
                default:
                    return nullptr;
            }
        }
        return nullptr;
    }


    /*! \details The function call is non-blocking.
     *
     *  \param[in] events a wait-list of events.
     *  \param[out] event event associated with the last kernel execution.
     */
    void RGBDTo8D::run (const std::vector<cl::Event> *events, cl::Event *event)
    {
        queue.enqueueNDRangeKernel (kernel, cl::NullRange, global, local, events, event);
    }


    /*! \return The focal length of the camera used. */
    float RGBDTo8D::getFocalLength ()
    {
        return f;
    }


    /*! \details Updates the kernel argument for the focal length.
     *
     *  \param[in] _f the focal length.
     */
    void RGBDTo8D::setFocalLength (float _f)
    {
        f = _f;
        kernel.setArg (7, f);
    }


    /*! \return The scaling factor. */
    float RGBDTo8D::getScaling ()
    {
        return scaling;
    }


    /*! \details Updates the kernel argument for the scaling factor.
     *
     *  \param[in] _scaling scaling factor.
     */
    void RGBDTo8D::setScaling (float _scaling)
    {
        scaling = _scaling;
        kernel.setArg (8, scaling);
    }


    /*! \return The flag for RGB normalization. */
    int RGBDTo8D::getRGBNorm ()
    {
        return rgbNorm;
    }


    /*! \details Updates the flag for RGB normalization.
     *
     *  \param[in] _rgbNorm flag that indicates whether to perform RGB normalization.
     */
    void RGBDTo8D::setRGBNorm (int _rgbNorm)
    {
        rgbNorm = _rgbNorm;
        kernel.setArg (9, rgbNorm);
    }


    /*! \param[in] _env opencl environment.
     *  \param[in] _info opencl configuration. Specifies the context, queue, etc, to be used.
     */
    SplitPC8D::SplitPC8D (clutils::CLEnv &_env, clutils::CLEnvInfo<1> _info) : 
        env (_env), info (_info), 
        context (env.getContext (info.pIdx)), 
        queue (env.getQueue (info.ctxIdx, info.qIdx[0])), 
        kernel (env.getProgram (info.pgIdx), "splitPC8D")
    {
    }


    /*! \details This interface exists to allow CL memory sharing between different kernels.
     *
     *  \param[in] mem enumeration value specifying the requested memory object.
     *  \return A reference to the requested memory object.
     */
    cl::Memory& SplitPC8D::get (SplitPC8D::Memory mem)
    {
        switch (mem)
        {
            case SplitPC8D::Memory::H_IN:
                return hBufferIn;
            case SplitPC8D::Memory::H_OUT_PC4D:
                return hBufferOutPC4D;
            case SplitPC8D::Memory::H_OUT_RGBA:
                return hBufferOutRGBA;
            case SplitPC8D::Memory::D_IN:
                return dBufferIn;
            case SplitPC8D::Memory::D_OUT_PC4D:
                return dBufferOutPC4D;
            case SplitPC8D::Memory::D_OUT_RGBA:
                return dBufferOutRGBA;
        }
    }


    /*! \details Sets up memory objects as necessary, and defines the kernel workspaces.
     *  \note If you have assigned a memory object to one member variable of the class 
     *        before the call to `init`, then that memory will be maintained. Otherwise, 
     *        a new memory object will be created.
     *        
     *  \param[in] _n number of points in the point cloud.
     *  \param[in] _offset number of points to skip in the output arrays. The kernel will 
     *                     write in the output arrays starting at position `_offset` (`cl\_float4` 
     *                     types are considered). Use the maximum offset you expect to need, so 
     *                     the required memory can be allocated. You can change it afterwards 
     *                     dynamically to the one required at any moment.
     *  \param[in] _staging flag to indicate whether or not to instantiate the staging buffers.
     */
    void SplitPC8D::init (unsigned int _n, unsigned int _offset, Staging _staging)
    {
        n = _n;
        offset = _offset;
        bufferInSize = n * sizeof (cl_float8);
        bufferOutSize = (offset + n) * sizeof (cl_float4);
        staging = _staging;

        try
        {
            if (n == 0)
                throw "The point cloud cannot be empty";
        }
        catch (const char *error)
        {
            std::cerr << "Error[SplitPC8D]: " << error << std::endl;
            exit (EXIT_FAILURE);
        }

        // Set workspace
        global = cl::NDRange (n);

        // Create staging buffers
        bool io = false;
        switch (staging)
        {
            case Staging::NONE:
                hPtrIn = nullptr;
                hPtrOutPC4D = nullptr;
                hPtrOutRGBA = nullptr;
                break;

            case Staging::IO:
                io = true;

            case Staging::I:
                if (hBufferIn () == nullptr)
                    hBufferIn = cl::Buffer (context, CL_MEM_ALLOC_HOST_PTR, bufferInSize);

                hPtrIn = (cl_float *) queue.enqueueMapBuffer (
                    hBufferIn, CL_FALSE, CL_MAP_WRITE, 0, bufferInSize);
                queue.enqueueUnmapMemObject (hBufferIn, hPtrIn);

                if (!io)
                {
                    queue.finish ();
                    hPtrOutPC4D = nullptr;
                    hPtrOutRGBA = nullptr;
                    break;
                }

            case Staging::O:
                if (hBufferOutPC4D () == nullptr)
                    hBufferOutPC4D = cl::Buffer (context, CL_MEM_ALLOC_HOST_PTR, bufferOutSize);
                if (hBufferOutRGBA () == nullptr)
                    hBufferOutRGBA = cl::Buffer (context, CL_MEM_ALLOC_HOST_PTR, bufferOutSize);

                hPtrOutPC4D = (cl_float *) queue.enqueueMapBuffer (
                    hBufferOutPC4D, CL_FALSE, CL_MAP_READ, 0, bufferOutSize);
                hPtrOutRGBA = (cl_float *) queue.enqueueMapBuffer (
                    hBufferOutRGBA, CL_FALSE, CL_MAP_READ, 0, bufferOutSize);
                queue.enqueueUnmapMemObject (hBufferOutPC4D, hPtrOutPC4D);
                queue.enqueueUnmapMemObject (hBufferOutRGBA, hPtrOutRGBA);
                queue.finish ();

                if (!io) hPtrIn = nullptr;
                break;
        }
        
        // Create device buffers
        if (dBufferIn () == nullptr)
            dBufferIn = cl::Buffer (context, CL_MEM_READ_ONLY, bufferInSize);
        if (dBufferOutPC4D () == nullptr)
            dBufferOutPC4D = cl::Buffer (context, CL_MEM_WRITE_ONLY, bufferOutSize);
        if (dBufferOutRGBA () == nullptr)
            dBufferOutRGBA = cl::Buffer (context, CL_MEM_WRITE_ONLY, bufferOutSize);

        // Set kernel arguments
        kernel.setArg (0, dBufferIn);
        kernel.setArg (1, dBufferOutPC4D);
        kernel.setArg (2, dBufferOutRGBA);
        kernel.setArg (3, offset);
    }


    /*! \details The transfer happens from a staging buffer on the host to the 
     *           associated (specified) device buffer.
     *  
     *  \param[in] mem enumeration value specifying an input device buffer.
     *  \param[in] ptr a pointer to an array holding input data. If not NULL, the 
     *                 data from `ptr` will be copied to the associated staging buffer.
     *  \param[in] block a flag to indicate whether to perform a blocking 
     *                   or a non-blocking operation.
     *  \param[in] events a wait-list of events.
     *  \param[out] event event associated with the write operation to the device buffer.
     */
    void SplitPC8D::write (SplitPC8D::Memory mem, void *ptr, bool block, 
                           const std::vector<cl::Event> *events, cl::Event *event)
    {
        if (staging == Staging::I || staging == Staging::IO)
        {
            switch (mem)
            {
                case SplitPC8D::Memory::D_IN:
                    if (ptr != nullptr)
                        std::copy ((cl_float8 *) ptr, (cl_float8 *) ptr + n, (cl_float8 *) hPtrIn);
                    queue.enqueueWriteBuffer (dBufferIn, block, 0, bufferInSize, hPtrIn, events, event);
                    break;
                default:
                    break;
            }
        }
    }


    /*! \details The transfer happens from a device buffer to the associated 
     *           (specified) staging buffer on the host.
     *  
     *  \param[in] mem enumeration value specifying an output staging buffer.
     *  \param[in] block a flag to indicate whether to perform a blocking 
     *                   or a non-blocking operation.
     *  \param[in] events a wait-list of events.
     *  \param[out] event event associated with the read operation to the staging buffer.
     */
    void* SplitPC8D::read (SplitPC8D::Memory mem, bool block, 
                           const std::vector<cl::Event> *events, cl::Event *event)
    {
        if (staging == Staging::O || staging == Staging::IO)
        {
            switch (mem)
            {
                case SplitPC8D::Memory::H_OUT_PC4D:
                    queue.enqueueReadBuffer (dBufferOutPC4D, block, 0, bufferOutSize, hPtrOutPC4D, events, event);
                    return hPtrOutPC4D;
                case SplitPC8D::Memory::H_OUT_RGBA:
                    queue.enqueueReadBuffer (dBufferOutRGBA, block, 0, bufferOutSize, hPtrOutRGBA, events, event);
                    return hPtrOutRGBA;
                default:
                    return nullptr;
            }
        }
        return nullptr;
    }


    /*! \details The function call is non-blocking.
     *
     *  \param[in] events a wait-list of events.
     *  \param[out] event event associated with the last kernel execution.
     */
    void SplitPC8D::run (const std::vector<cl::Event> *events, cl::Event *event)
    {
        queue.enqueueNDRangeKernel (kernel, cl::NullRange, global, cl::NullRange, events, event);
    }


    /*! \return The offset.
     */
    unsigned int SplitPC8D::getOffset ()
    {
        return offset;
    }


    /*! \details Updates the kernel argument for the offset.
     *
     *  \param[in] _offset the offset to set.
     */
    void SplitPC8D::setOffset (unsigned int _offset)
    {
        offset = _offset;
        kernel.setArg (3, offset);
    }


    /*! \param[in] _env opencl environment.
     *  \param[in] _info opencl configuration. Specifies the context, queue, etc, to be used.
     */
    RGBNorm::RGBNorm (clutils::CLEnv &_env, clutils::CLEnvInfo<1> _info) : 
        env (_env), info (_info), 
        context (env.getContext (info.pIdx)), 
        queue (env.getQueue (info.ctxIdx, info.qIdx[0])), 
        kernel (env.getProgram (info.pgIdx), "rgbNorm")
    {
    }


    /*! \details This interface exists to allow CL memory sharing between different kernels.
     *
     *  \param[in] mem enumeration value specifying the requested memory object.
     *  \return A reference to the requested memory object.
     */
    cl::Memory& RGBNorm::get (RGBNorm::Memory mem)
    {
        switch (mem)
        {
            case RGBNorm::Memory::H_IN:
                return hBufferIn;
            case RGBNorm::Memory::H_OUT:
                return hBufferOut;
            case RGBNorm::Memory::D_IN:
                return dBufferIn;
            case RGBNorm::Memory::D_OUT:
                return dBufferOut;
        }
    }


    /*! \details Sets up memory objects as necessary, and defines the kernel workspaces.
     *  \note If you have assigned a memory object to one member variable of the class 
     *        before the call to `init`, then that memory will be maintained. Otherwise, 
     *        a new memory object will be created.
     *        
     *  \param[in] _width width (\#pixels/row) of the array to be processed.
     *  \param[in] _height height (\#pixels/column) of the array to be processed.
     *  \param[in] _staging flag to indicate whether or not to instantiate the staging buffers.
     */
    void RGBNorm::init (unsigned int _width, unsigned int _height, Staging _staging)
    {
        // Logically rearrange data (pixels)
        width = _width; height = _height;
        bufferSize = 3 * width * height * sizeof (cl_float);
        staging = _staging;

        try
        {
            if ((width == 0) || (height == 0))
                throw "The image cannot have zeroed dimensions";
        }
        catch (const char *error)
        {
            std::cerr << "Error[RGBNorm]: " << error << std::endl;
            exit (EXIT_FAILURE);
        }

        // Set workspaces
        global = cl::NDRange (width * height);

        // Create staging buffers
        bool io = false;
        switch (staging)
        {
            case Staging::NONE:
                hPtrIn = nullptr;
                hPtrOut = nullptr;
                break;

            case Staging::IO:
                io = true;

            case Staging::I:
                if (hBufferIn () == nullptr)
                    hBufferIn = cl::Buffer (context, CL_MEM_ALLOC_HOST_PTR, bufferSize);

                hPtrIn = (cl_float *) queue.enqueueMapBuffer (
                    hBufferIn, CL_FALSE, CL_MAP_WRITE, 0, bufferSize);
                queue.enqueueUnmapMemObject (hBufferIn, hPtrIn);

                if (!io)
                {
                    queue.finish ();
                    hPtrOut = nullptr;
                    break;
                }

            case Staging::O:
                if (hBufferOut () == nullptr)
                    hBufferOut = cl::Buffer (context, CL_MEM_ALLOC_HOST_PTR, bufferSize);

                hPtrOut = (cl_float *) queue.enqueueMapBuffer (
                    hBufferOut, CL_FALSE, CL_MAP_READ, 0, bufferSize);
                queue.enqueueUnmapMemObject (hBufferOut, hPtrOut);
                queue.finish ();

                if (!io) hPtrIn = nullptr;
                break;
        }
        
        // Create device buffers
        if (dBufferIn () == nullptr)
            dBufferIn = cl::Buffer (context, CL_MEM_READ_ONLY, bufferSize);
        if (dBufferOut () == nullptr)
            dBufferOut = cl::Buffer (context, CL_MEM_WRITE_ONLY, bufferSize);

        // Set kernel arguments
        kernel.setArg (0, dBufferIn);
        kernel.setArg (1, dBufferOut);
    }


    /*! \details The transfer happens from a staging buffer on the host to the 
     *           associated (specified) device buffer.
     *  
     *  \param[in] mem enumeration value specifying an input device buffer.
     *  \param[in] ptr a pointer to an array holding input data. If not NULL, the 
     *                 data from `ptr` will be copied to the associated staging buffer.
     *  \param[in] block a flag to indicate whether to perform a blocking 
     *                   or a non-blocking operation.
     *  \param[in] events a wait-list of events.
     *  \param[out] event event associated with the write operation to the device buffer.
     */
    void RGBNorm::write (RGBNorm::Memory mem, void *ptr, bool block, 
                         const std::vector<cl::Event> *events, cl::Event *event)
    {
        if (staging == Staging::I || staging == Staging::IO)
        {
            switch (mem)
            {
                case RGBNorm::Memory::D_IN:
                    if (ptr != nullptr)
                        std::copy ((cl_float *) ptr, (cl_float *) ptr + 3 * width * height, hPtrIn);
                    queue.enqueueWriteBuffer (dBufferIn, block, 0, bufferSize, hPtrIn, events, event);
                    break;
                default:
                    break;
            }
        }
    }


    /*! \details The transfer happens from a device buffer to the associated 
     *           (specified) staging buffer on the host.
     *  
     *  \param[in] mem enumeration value specifying an output staging buffer.
     *  \param[in] block a flag to indicate whether to perform a blocking 
     *                   or a non-blocking operation.
     *  \param[in] events a wait-list of events.
     *  \param[out] event event associated with the read operation to the staging buffer.
     */
    void* RGBNorm::read (RGBNorm::Memory mem, bool block, 
                         const std::vector<cl::Event> *events, cl::Event *event)
    {
        if (staging == Staging::O || staging == Staging::IO)
        {
            switch (mem)
            {
                case RGBNorm::Memory::H_OUT:
                    queue.enqueueReadBuffer (dBufferOut, block, 0, bufferSize, hPtrOut, events, event);
                    return hPtrOut;
                default:
                    return nullptr;
            }
        }
        return nullptr;
    }


    /*! \details The function call is non-blocking.
     *
     *  \param[in] events a wait-list of events.
     *  \param[out] event event associated with the last kernel execution.
     */
    void RGBNorm::run (const std::vector<cl::Event> *events, cl::Event *event)
    {
        queue.enqueueNDRangeKernel (kernel, cl::NullRange, global, cl::NullRange, events, event);
    }


    /*! \param[in] _env opencl environment.
     *  \param[in] _info opencl configuration. Specifies the context, queue, etc, to be used.
     */
    Scan::Scan (clutils::CLEnv &_env, clutils::CLEnvInfo<1> _info) : 
        env (_env), info (_info), 
        context (env.getContext (info.pIdx)), 
        queue (env.getQueue (info.ctxIdx, info.qIdx[0])), 
        kernelScan (env.getProgram (info.pgIdx), "inclusiveScan_f"), 
        kernelSumsScan (env.getProgram (info.pgIdx), "inclusiveScan_f"), 
        kernelAddSums (env.getProgram (info.pgIdx), "addGroupSums_f")
    {
        wgMultiple = kernelScan.getWorkGroupInfo
            <CL_KERNEL_PREFERRED_WORK_GROUP_SIZE_MULTIPLE> (env.devices[info.pIdx][info.dIdx]);
    }


    /*! \details This interface exists to allow CL memory sharing between different kernels.
     *
     *  \param[in] mem enumeration value specifying the requested memory object.
     *  \return A reference to the requested memory object.
     */
    cl::Memory& Scan::get (Scan::Memory mem)
    {
        switch (mem)
        {
            case Scan::Memory::H_IN:
                return hBufferIn;
            case Scan::Memory::H_OUT:
                return hBufferOut;
            case Scan::Memory::D_IN:
                return dBufferIn;
            case Scan::Memory::D_SUMS:
                return dBufferSums;
            case Scan::Memory::D_OUT:
                return dBufferOut;
        }
    }


    /*! \details Sets up memory objects as necessary, and defines the kernel workspaces.
     *  \note If you have assigned a memory object to one member variable of the class 
     *        before the call to `init`, then that memory will be maintained. Otherwise, 
     *        a new memory object will be created.
     *  \note Working with `float` elements and having large summations can be problematic.
     *        It is advised that a scaling is applied on the elements for better accuracy.
     *        
     *  \param[in] _width width of the input array.
     *  \param[in] _height height of the input array.
     *  \param[in] _scaling factor by which to scale the array elements before processing.
     *  \param[in] _staging flag to indicate whether or not to instantiate the staging buffers.
     */
    void Scan::init (unsigned int _width, unsigned int _height, float _scaling, Staging _staging)
    {
        width = _width; height = _height;
        bufferSize = width * height * sizeof (cl_float);
        scaling = _scaling;
        staging = _staging;

        // Establish the number of work-groups per row
        wgXdim = ceil (width / (float) (8 * wgMultiple));
        // Round up to a multiple of 4 (data are handled as int4)
        if ((wgXdim != 1) && (wgXdim % 4)) wgXdim += 4 - wgXdim % 4;

        bufferSumsSize = wgXdim * height * sizeof (cl_float);

        try
        {
            if (wgXdim == 0)
                throw "The array cannot have zero columns";

            if (width % 4 != 0)
                throw "The number of columns in the array must be a multiple of 4";

            // (8 * wgMultiple) elements per work-group
            if (width > std::pow (8 * wgMultiple, 2))
            {
                std::ostringstream ss;
                ss << "The current configuration of Scan supports arrays ";
                ss << "of up to " << std::pow (8 * wgMultiple, 2) << " columns";
                throw ss.str ().c_str ();
            }
        }
        catch (const char *error)
        {
            std::cerr << "Error[Scan]: " << error << std::endl;
            exit (EXIT_FAILURE);
        }

        // Set workspaces
        globalScan = cl::NDRange (wgXdim * wgMultiple, height);
        localScan = cl::NDRange (wgMultiple, 1);
        globalSumsScan = cl::NDRange (wgMultiple, height);
        globalAddSums = cl::NDRange (2 * (wgXdim - 1) * wgMultiple, height);
        localAddSums = cl::NDRange (2 * wgMultiple, 1);
        offsetAddSums = cl::NDRange (2 * wgMultiple, 0);

        // Create staging buffers
        bool io = false;
        switch (staging)
        {
            case Staging::NONE:
                hPtrIn = nullptr;
                hPtrOut = nullptr;
                break;

            case Staging::IO:
                io = true;

            case Staging::I:
                if (hBufferIn () == nullptr)
                    hBufferIn = cl::Buffer (context, CL_MEM_ALLOC_HOST_PTR, bufferSize);

                hPtrIn = (cl_float *) queue.enqueueMapBuffer (
                    hBufferIn, CL_FALSE, CL_MAP_WRITE, 0, bufferSize);
                queue.enqueueUnmapMemObject (hBufferIn, hPtrIn);

                if (!io)
                {
                    queue.finish ();
                    hPtrOut = nullptr;
                    break;
                }

            case Staging::O:
                if (hBufferOut () == nullptr)
                    hBufferOut = cl::Buffer (context, CL_MEM_ALLOC_HOST_PTR, bufferSize);

                hPtrOut = (cl_float *) queue.enqueueMapBuffer (
                    hBufferOut, CL_FALSE, CL_MAP_READ, 0, bufferSize);
                queue.enqueueUnmapMemObject (hBufferOut, hPtrOut);
                queue.finish ();

                if (!io) hPtrIn = nullptr;
                break;
        }
        
        // Create device buffers
        if (dBufferIn () == nullptr)
            dBufferIn = cl::Buffer (context, CL_MEM_READ_ONLY, bufferSize);
        if (dBufferSums () == nullptr)
            dBufferSums = cl::Buffer (context, CL_MEM_READ_WRITE, bufferSumsSize);
        if (dBufferOut () == nullptr)
            dBufferOut = cl::Buffer (context, CL_MEM_READ_WRITE, bufferSize);

        // Set kernel arguments
        if (wgXdim == 1)
        {
            kernelScan.setArg (0, dBufferIn);
            kernelScan.setArg (1, dBufferOut);
            kernelScan.setArg (2, cl::Local (2 * localScan[0] * sizeof (cl_float)));
            kernelScan.setArg (3, dBufferSums);  // Unused
            kernelScan.setArg (4, width / 4);
            kernelScan.setArg (5, scaling);
        }
        else
        {
            kernelScan.setArg (0, dBufferIn);
            kernelScan.setArg (1, dBufferOut);
            kernelScan.setArg (2, cl::Local (2 * localScan[0] * sizeof (cl_float)));
            kernelScan.setArg (3, dBufferSums);
            kernelScan.setArg (4, width / 4);
            kernelScan.setArg (5, scaling);

            kernelSumsScan.setArg (0, dBufferSums);
            kernelSumsScan.setArg (1, dBufferSums);
            kernelSumsScan.setArg (2, cl::Local (2 * localScan[0] * sizeof (cl_float)));
            kernelSumsScan.setArg (3, dBufferSums);  // Unused
            kernelSumsScan.setArg (4, (cl_uint) (wgXdim / 4));
            kernelSumsScan.setArg (5, 1.f);

            kernelAddSums.setArg (0, dBufferSums);
            kernelAddSums.setArg (1, dBufferOut);
            kernelAddSums.setArg (2, width / 4);
        }
    }


    /*! \details The transfer happens from a staging buffer on the host to the 
     *           associated (specified) device buffer.
     *  
     *  \param[in] mem enumeration value specifying an input device buffer.
     *  \param[in] ptr a pointer to an array holding input data. If not NULL, the 
     *                 data from `ptr` will be copied to the associated staging buffer.
     *  \param[in] block a flag to indicate whether to perform a blocking 
     *                   or a non-blocking operation.
     *  \param[in] events a wait-list of events.
     *  \param[out] event event associated with the write operation to the device buffer.
     */
    void Scan::write (Scan::Memory mem, void *ptr, bool block, 
                      const std::vector<cl::Event> *events, cl::Event *event)
    {
        if (staging == Staging::I || staging == Staging::IO)
        {
            switch (mem)
            {
                case Scan::Memory::D_IN:
                    if (ptr != nullptr)
                        std::copy ((cl_float *) ptr, (cl_float *) ptr + width * height, hPtrIn);
                    queue.enqueueWriteBuffer (dBufferIn, block, 0, bufferSize, hPtrIn, events, event);
                    break;
                default:
                    break;
            }
        }
    }


    /*! \details The transfer happens from a device buffer to the associated 
     *           (specified) staging buffer on the host.
     *  
     *  \param[in] mem enumeration value specifying an output staging buffer.
     *  \param[in] block a flag to indicate whether to perform a blocking 
     *                   or a non-blocking operation.
     *  \param[in] events a wait-list of events.
     *  \param[out] event event associated with the read operation to the staging buffer.
     */
    void* Scan::read (Scan::Memory mem, bool block, 
                      const std::vector<cl::Event> *events, cl::Event *event)
    {
        if (staging == Staging::O || staging == Staging::IO)
        {
            switch (mem)
            {
                case Scan::Memory::H_OUT:
                    queue.enqueueReadBuffer (dBufferOut, block, 0, bufferSize, hPtrOut, events, event);
                    return hPtrOut;
                default:
                    return nullptr;
            }
        }
        return nullptr;
    }


    /*! \details The function call is non-blocking.
     *
     *  \param[in] events a wait-list of events.
     *  \param[out] event event associated with the last kernel execution.
     */
    void Scan::run (const std::vector<cl::Event> *events, cl::Event *event)
    {
        if (wgXdim == 1)
        {
            queue.enqueueNDRangeKernel (
                kernelScan, cl::NullRange, globalScan, localScan, events, event);
        }
        else
        {
            queue.enqueueNDRangeKernel (
                kernelScan, cl::NullRange, globalScan, localScan, events);

            queue.enqueueNDRangeKernel (
                kernelSumsScan, cl::NullRange, globalSumsScan, localScan);

            queue.enqueueNDRangeKernel (
                kernelAddSums, offsetAddSums, globalAddSums, localAddSums, nullptr, event);
        }
    }


    /*! \return The scaling factor.
     */
    float Scan::getScaling ()
    {
        return scaling;
    }


    /*! \details Updates the kernel argument for the scaling factor.
     *
     *  \param[in] _scaling scaling factor.
     */
    void Scan::setScaling (float _scaling)
    {
        scaling = _scaling;
        kernelScan.setArg (5, scaling);
    }


    /*! \param[in] _env opencl environment.
     *  \param[in] _info opencl configuration. Specifies the context, queue, etc, to be used.
     */
    Transpose::Transpose (clutils::CLEnv &_env, clutils::CLEnvInfo<1> _info) : 
        env (_env), info (_info), 
        context (env.getContext (info.pIdx)), 
        queue (env.getQueue (info.ctxIdx, info.qIdx[0])), 
        kernel (env.getProgram (info.pgIdx), "transpose")
    {
    }


    /*! \details This interface exists to allow CL memory sharing between different kernels.
     *
     *  \param[in] mem enumeration value specifying the requested memory object.
     *  \return A reference to the requested memory object.
     */
    cl::Memory& Transpose::get (Transpose::Memory mem)
    {
        switch (mem)
        {
            case Transpose::Memory::H_IN:
                return hBufferIn;
            case Transpose::Memory::H_OUT:
                return hBufferOut;
            case Transpose::Memory::D_IN:
                return dBufferIn;
            case Transpose::Memory::D_OUT:
                return dBufferOut;
        }
    }


    /*! \details Sets up memory objects as necessary, and defines the kernel workspaces.
     *  \note If you have assigned a memory object to one member variable of the class 
     *        before the call to `init`, then that memory will be maintained. Otherwise, 
     *        a new memory object will be created.
     *        
     *  \param[in] _width width of the input array to be processed.
     *  \param[in] _height height of the input array to be processed.
     *  \param[in] _staging flag to indicate whether or not to instantiate the staging buffers.
     */
    void Transpose::init (unsigned int _width, unsigned int _height, Staging _staging)
    {
        width = _width; height = _height;
        bufferSize = width * height * sizeof (cl_float);
        unsigned int lSide;
        staging = _staging;

        try
        {
            //* Each work-item processes a 4x4 block of data
            unsigned int xN = width / 4;
            unsigned int yN = height / 4;
            unsigned int xR = width % 4;
            unsigned int yR = height % 4;

            if (xR || yR || (xN == 0) || (yN == 0))
                throw "The number of rows, and columns, in the array must be "
                      "a multiple of 4 and a strictly positive number";

            // Choose a side for the local (square) workspace
            //* At the very least, 1 will always satisfy the conditions (1)
            // (1) Both input dimensions, after divided by 4, must fit 
            //     an integer number of work-groups
            for (lSide = 16; lSide > 0; lSide >>= 1)
                if ((xN % lSide == 0) && (yN % lSide == 0))
                    break;
        }
        catch (const char *error)
        {
            std::cerr << "Error[Transpose]: " << error << std::endl;
            exit (EXIT_FAILURE);
        }

        size_t localSize = lSide * lSide;
        size_t wgMultiple = kernel.getWorkGroupInfo
                <CL_KERNEL_PREFERRED_WORK_GROUP_SIZE_MULTIPLE> (env.devices[info.pIdx][info.dIdx]);
        if (localSize % wgMultiple)
            std::cout << "Warning[Transpose]: The work-group size [" << localSize 
                      << "] is not a multiple of the preferred size [" 
                      << wgMultiple << "] on this device" << std::endl;

        // Set workspaces
        global = cl::NDRange (width / 4, height / 4);
        local = cl::NDRange (lSide, lSide);

        // Create staging buffers
        bool io = false;
        switch (staging)
        {
            case Staging::NONE:
                hPtrIn = nullptr;
                hPtrOut = nullptr;
                break;

            case Staging::IO:
                io = true;

            case Staging::I:
                if (hBufferIn () == nullptr)
                    hBufferIn = cl::Buffer (context, CL_MEM_ALLOC_HOST_PTR, bufferSize);

                hPtrIn = (cl_float *) queue.enqueueMapBuffer (
                    hBufferIn, CL_FALSE, CL_MAP_WRITE, 0, bufferSize);
                queue.enqueueUnmapMemObject (hBufferIn, hPtrIn);

                if (!io)
                {
                    queue.finish ();
                    hPtrOut = nullptr;
                    break;
                }

            case Staging::O:
                if (hBufferOut () == nullptr)
                    hBufferOut = cl::Buffer (context, CL_MEM_ALLOC_HOST_PTR, bufferSize);

                hPtrOut = (cl_float *) queue.enqueueMapBuffer (
                    hBufferOut, CL_FALSE, CL_MAP_READ, 0, bufferSize);
                queue.enqueueUnmapMemObject (hBufferOut, hPtrOut);
                queue.finish ();

                if (!io) hPtrIn = nullptr;
                break;
        }

        // Create device buffers
        if (dBufferIn () == nullptr)
            dBufferIn = cl::Buffer (context, CL_MEM_READ_ONLY, bufferSize);
        if (dBufferOut () == nullptr)
            dBufferOut = cl::Buffer (context, CL_MEM_WRITE_ONLY, bufferSize);

        // Set kernel arguments
        kernel.setArg (0, dBufferIn);
        kernel.setArg (1, dBufferOut);
        kernel.setArg (2, cl::Local (4 * 4 * local[0] * local[1] * sizeof (cl_float)));
    }


    /*! \details The transfer happens from a staging buffer on the host to the 
     *           associated (specified) device buffer.
     *  
     *  \param[in] mem enumeration value specifying an input device buffer.
     *  \param[in] ptr a pointer to an array holding input data. If not NULL, the 
     *                 data from `ptr` will be copied to the associated staging buffer.
     *  \param[in] block a flag to indicate whether to perform a blocking 
     *                   or a non-blocking operation.
     *  \param[in] events a wait-list of events.
     *  \param[out] event event associated with the write operation to the device buffer.
     */
    void Transpose::write (Transpose::Memory mem, void *ptr, bool block, 
                           const std::vector<cl::Event> *events, cl::Event *event)
    {
        if (staging == Staging::I || staging == Staging::IO)
        {
            switch (mem)
            {
                case Transpose::Memory::D_IN:
                    if (ptr != nullptr)
                        std::copy ((cl_float *) ptr, (cl_float *) ptr + width * height, hPtrIn);
                    queue.enqueueWriteBuffer (dBufferIn, block, 0, bufferSize, hPtrIn, events, event);
                    break;
                default:
                    break;
            }
        }
    }


    /*! \details The transfer happens from a device buffer to the associated 
     *           (specified) staging buffer on the host.
     *  
     *  \param[in] mem enumeration value specifying an output staging buffer.
     *  \param[in] block a flag to indicate whether to perform a blocking 
     *                   or a non-blocking operation.
     *  \param[in] events a wait-list of events.
     *  \param[out] event event associated with the read operation to the staging buffer.
     */
    void* Transpose::read (Transpose::Memory mem, bool block, 
                           const std::vector<cl::Event> *events, cl::Event *event)
    {
        if (staging == Staging::O || staging == Staging::IO)
        {
            switch (mem)
            {
                case Transpose::Memory::H_OUT:
                    queue.enqueueReadBuffer (dBufferOut, block, 0, bufferSize, hPtrOut, events, event);
                    return hPtrOut;
                default:
                    return nullptr;
            }
        }
        return nullptr;
    }


    /*! \details The function call is non-blocking.
     *
     *  \param[in] events a wait-list of events.
     *  \param[out] event event associated with the kernel execution.
     */
    void Transpose::run (const std::vector<cl::Event> *events, cl::Event *event)
    {
        queue.enqueueNDRangeKernel (kernel, cl::NullRange, global, local, events, event);
    }


    /*! \param[in] _env opencl environment.
     *  \param[in] _info opencl configuration. Specifies the context, queue, etc, to be used.
     *  \param[in] _transposed a flag to indicate whether to leave the output 
     *                         array in a transposed configuration, or not. If 
     *                         false, a (second) transposition is performed on 
     *                         the output array to bring the array at the 
     *                         initial configuration.
     */
    SAT::SAT (clutils::CLEnv &_env, clutils::CLEnvInfo<1> _info, bool _transposed) : 
        env (_env), info (_info), 
        context (env.getContext (info.pIdx)), 
        queue (env.getQueue (info.ctxIdx, info.qIdx[0])), 
        scanRows    (Scan (env, info)), transpose1 (Transpose (env, info)),
        scanColumns (Scan (env, info)), transpose2 (Transpose (env, info)), 
        transposed (_transposed)
    {
    }


    /*! \details This interface exists to allow CL memory sharing between different kernels.
     *
     *  \param[in] mem enumeration value specifying the requested memory object.
     *  \return A reference to the requested memory object.
     */
    cl::Memory& SAT::get (SAT::Memory mem)
    {
        switch (mem)
        {
            case SAT::Memory::H_IN:
                return hBufferIn;
            case SAT::Memory::H_OUT:
                return hBufferOut;
            case SAT::Memory::D_IN:
                return scanRows.get (Scan::Memory::D_IN);
            case SAT::Memory::D_OUT:
                if (transposed) return scanColumns.get (Scan::Memory::D_OUT);
                else return transpose2.get (Transpose::Memory::D_OUT);
        }
    }


    /*! \details Sets up memory objects as necessary, and defines the kernel workspaces.
     *  \note If you have assigned a memory object to one member variable of the class 
     *        before the call to `init`, then that memory will be maintained. Otherwise, 
     *        a new memory object will be created.
     *  \note Working with `float` elements and having large summations can be problematic.
     *        It is advised that a scaling is applied on the elements for better accuracy.
     *        
     *  \param[in] _width width of the input array to be processed.
     *  \param[in] _height height of the input array to be processed.
     *  \param[in] _scaling factor by which to scale the array elements before processing.
     *  \param[in] _staging flag to indicate whether or not to instantiate the staging buffers.
     */
    void SAT::init (unsigned int _width, unsigned int _height, float _scaling, Staging _staging)
    {
        width = _width; height = _height;
        bufferSize = width * height * sizeof (cl_float);
        scaling = _scaling;
        staging = _staging;

        try
        {
            if ((width == 0) || (height == 0))
                throw "The image cannot have zeroed dimensions";
        }
        catch (const char *error)
        {
            std::cerr << "Error[SAT]: " << error << std::endl;
            exit (EXIT_FAILURE);
        }

        // Create staging buffers
        bool io = false;
        switch (staging)
        {
            case Staging::NONE:
                hPtrIn = nullptr;
                hPtrOut = nullptr;
                break;

            case Staging::IO:
                io = true;

            case Staging::I:
                if (hBufferIn () == nullptr)
                    hBufferIn = cl::Buffer (context, CL_MEM_ALLOC_HOST_PTR, bufferSize);

                hPtrIn = (cl_float *) queue.enqueueMapBuffer (
                    hBufferIn, CL_FALSE, CL_MAP_WRITE, 0, bufferSize);
                queue.enqueueUnmapMemObject (hBufferIn, hPtrIn);

                if (!io)
                {
                    queue.finish ();
                    hPtrOut = nullptr;
                    break;
                }

            case Staging::O:
                if (hBufferOut () == nullptr)
                    hBufferOut = cl::Buffer (context, CL_MEM_ALLOC_HOST_PTR, bufferSize);

                hPtrOut = (cl_float *) queue.enqueueMapBuffer (
                    hBufferOut, CL_FALSE, CL_MAP_READ, 0, bufferSize);
                queue.enqueueUnmapMemObject (hBufferOut, hPtrOut);
                queue.finish ();

                if (!io) hPtrIn = nullptr;
                break;
        }

        scanRows.init (width, height, scaling, Staging::NONE);

        transpose1.get (Transpose::Memory::D_IN) = scanRows.get (Scan::Memory::D_OUT);
        transpose1.get (Transpose::Memory::D_OUT) = cl::Buffer (context, CL_MEM_READ_WRITE, bufferSize);
        transpose1.init (width, height, Staging::NONE);

        scanColumns.get (Scan::Memory::D_IN) = transpose1.get (Transpose::Memory::D_OUT);
        scanColumns.init (height, width, 1.f, Staging::NONE);

        if (!transposed)
        {
            transpose2.get (Transpose::Memory::D_IN) = scanColumns.get (Scan::Memory::D_OUT);
            transpose2.init (height, width, Staging::NONE);
        }
    }


    /*! \details The transfer happens from a staging buffer on the host to the 
     *           associated (specified) device buffer.
     *  
     *  \param[in] mem enumeration value specifying an input device buffer.
     *  \param[in] ptr a pointer to an array holding input data. If not NULL, the 
     *                 data from `ptr` will be copied to the associated staging buffer.
     *  \param[in] block a flag to indicate whether to perform a blocking 
     *                   or a non-blocking operation.
     *  \param[in] events a wait-list of events.
     *  \param[out] event event associated with the write operation to the device buffer.
     */
    void SAT::write (SAT::Memory mem, void *ptr, bool block, 
                     const std::vector<cl::Event> *events, cl::Event *event)
    {
        if (staging == Staging::I || staging == Staging::IO)
        {
            switch (mem)
            {
                case SAT::Memory::D_IN:
                    if (ptr != nullptr)
                        std::copy ((cl_float *) ptr, (cl_float *) ptr + width * height, hPtrIn);
                    queue.enqueueWriteBuffer ((cl::Buffer&) scanRows.get (Scan::Memory::D_IN), 
                        block, 0, bufferSize, hPtrIn, events, event);
                    break;
                default:
                    break;
            }
        }
    }


    /*! \details The transfer happens from a device buffer to the associated 
     *           (specified) staging buffer on the host.
     *  
     *  \param[in] mem enumeration value specifying an output staging buffer.
     *  \param[in] block a flag to indicate whether to perform a blocking 
     *                   or a non-blocking operation.
     *  \param[in] events a wait-list of events.
     *  \param[out] event event associated with the read operation to the staging buffer.
     */
    void* SAT::read (SAT::Memory mem, bool block, 
                     const std::vector<cl::Event> *events, cl::Event *event)
    {
        if (staging == Staging::O || staging == Staging::IO)
        {
            switch (mem)
            {
                case SAT::Memory::H_OUT:
                    if (transposed)
                        queue.enqueueReadBuffer ((cl::Buffer&) scanColumns.get (Scan::Memory::D_OUT), 
                            block, 0, bufferSize, hPtrOut, events, event);
                    else
                        queue.enqueueReadBuffer ((cl::Buffer&) transpose2.get (Transpose::Memory::D_OUT), 
                            block, 0, bufferSize, hPtrOut, events, event);
                    return hPtrOut;
                default:
                    return nullptr;
            }
        }
        return nullptr;
    }


    /*! \details The function call is non-blocking.
     *
     *  \param[in] events a wait-list of events.
     *  \param[out] event event associated with the kernel execution.
     */
    void SAT::run (const std::vector<cl::Event> *events, cl::Event *event)
    {
        scanRows.run (events);
        transpose1.run ();
        scanColumns.run ();

        if (!transposed)
            transpose2.run (nullptr, event);
    }


    /*! \return The scaling factor.
     */
    float SAT::getScaling ()
    {
        return scaling;
    }


    /*! \details Updates the kernel argument for the scaling factor of the rows scan.
     *
     *  \param[in] _scaling scaling factor.
     */
    void SAT::setScaling (float _scaling)
    {
        scaling = _scaling;
        scanRows.setScaling (scaling);
    }


    /*! \param[in] _env opencl environment.
     *  \param[in] _info opencl configuration. Specifies the context, queue, etc, to be used.
     */
    BoxFilterSAT::BoxFilterSAT (clutils::CLEnv &_env, clutils::CLEnvInfo<1> _info) : 
        env (_env), info (_info), 
        context (env.getContext (info.pIdx)), 
        queue (env.getQueue (info.ctxIdx, info.qIdx[0])), 
        kernel (env.getProgram (info.pgIdx), "boxFilterSAT_Tr"),
        sat (_env, _info)
    {
        // The class requires 16x16 work-groups (256 work-items per work-group)
        // The following code checks that this specification is possible

        cl::Device &device = env.devices[info.pIdx][info.dIdx];

        size_t maxLocalSize = device.getInfo<CL_DEVICE_MAX_WORK_GROUP_SIZE> ();

        std::vector<size_t> maxLocalDim = device.getInfo<CL_DEVICE_MAX_WORK_ITEM_SIZES> ();

        size_t wgMultiple = kernel.getWorkGroupInfo
            <CL_KERNEL_PREFERRED_WORK_GROUP_SIZE_MULTIPLE> (device);

        size_t localSize = lXdim * lYdim;

        try
        {
            if ((lXdim > maxLocalDim[0]) || (lYdim > maxLocalDim[1]))
            {
                std::ostringstream ss;
                ss << "The maximum work-group dimensions ";
                ss << "[" << maxLocalDim[0] << "][" << maxLocalDim[1] << "] ";
                ss << "are not enough (16x16 work-groups are required) on this device";
                throw ss.str ();
            }

            if (localSize > maxLocalSize)
            {
                std::ostringstream ss;
                ss << "The maximum work-group size ";
                ss << "[" << maxLocalSize << "] " << "is not enough ";
                ss << "(256 work-items per work-group are required) on this device";
                throw ss.str ();
            }

        }
        catch (const std::string &error)
        {
            std::cerr << "Error[BoxFilterSAT]: " << error << std::endl;
            exit (EXIT_FAILURE);
        }

        if (localSize % wgMultiple)
            std::cout << "Warning[BoxFilterSAT]: The work-group size [" << localSize 
                      << "] is not a multiple of the preferred size [" 
                      << wgMultiple << "] on this device" << std::endl;
    }


    /*! \details This interface exists to allow CL memory sharing between different kernels.
     *
     *  \param[in] mem enumeration value specifying the requested memory object.
     *  \return A reference to the requested memory object.
     */
    cl::Memory& BoxFilterSAT::get (BoxFilterSAT::Memory mem)
    {
        switch (mem)
        {
            case BoxFilterSAT::Memory::H_IN:
                return hBufferIn;
            case BoxFilterSAT::Memory::H_OUT:
                return hBufferOut;
            case BoxFilterSAT::Memory::D_IN:
                return sat.get (SAT::Memory::D_IN);
            case BoxFilterSAT::Memory::D_OUT:
                return dBufferOut;
        }
    }


    /*! \details Sets up memory objects as necessary, and defines the kernel workspaces.
     *  \note If you have assigned a memory object to one member variable of the class 
     *        before the call to `init`, then that memory will be maintained. Otherwise, 
     *        a new memory object will be created.
     *  \note Working with `float` elements and having large summations can be problematic.
     *        It is advised that a scaling is applied on the elements for better accuracy.
     *        A default value of \f$ 0.0001\ (1e-4) \f$ is normally applied. This scaling
     *        is only internal to the algorithm. No further processing is necessary on the output buffer.
     *        
     *  \param[in] _width width of the input array to be processed.
     *  \param[in] _height height of the input array to be processed.
     *  \param[in] _radius radius of the square filter window, i.e. \f$\ radius=filter\_width/2-1\f$.
     *  \param[in] _scaling factor by which to scale the array elements before processing.
     *  \param[in] _staging flag to indicate whether or not to instantiate the staging buffers.
     */
    void BoxFilterSAT::init (unsigned int _width, unsigned int _height, int _radius, float _scaling, Staging _staging)
    {
        width = _width; height = _height; radius = _radius;
        bufferSize = width * height * sizeof (cl_float);
        scaling = _scaling;
        staging = _staging;

        try
        {
            if ((width == 0) || (height == 0))
                throw "The image cannot have zeroed dimensions";

            if (height % lXdim != 0)
            {
                std::ostringstream ss;
                ss << "The image width has to be a multiple of " << lXdim;
                throw ss.str ().c_str ();
            }

            if (width % lYdim != 0)
            {
                std::ostringstream ss;
                ss << "The image height has to be a multiple of " << lYdim;
                throw ss.str ().c_str ();
            }
        }
        catch (const char *error)
        {
            std::cerr << "Error[BoxFilterSAT]: " << error << std::endl;
            exit (EXIT_FAILURE);
        }
        
        // Create staging buffers
        bool io = false;
        switch (staging)
        {
            case Staging::NONE:
                hPtrIn = nullptr;
                hPtrOut = nullptr;
                break;

            case Staging::IO:
                io = true;

            case Staging::I:
                if (hBufferIn () == nullptr)
                    hBufferIn = cl::Buffer (context, CL_MEM_ALLOC_HOST_PTR, bufferSize);

                hPtrIn = (cl_float *) queue.enqueueMapBuffer (
                    hBufferIn, CL_FALSE, CL_MAP_WRITE, 0, bufferSize);
                queue.enqueueUnmapMemObject (hBufferIn, hPtrIn);

                if (!io)
                {
                    queue.finish ();
                    hPtrOut = nullptr;
                    break;
                }

            case Staging::O:
                if (hBufferOut () == nullptr)
                    hBufferOut = cl::Buffer (context, CL_MEM_ALLOC_HOST_PTR, bufferSize);

                hPtrOut = (cl_float *) queue.enqueueMapBuffer (
                    hBufferOut, CL_FALSE, CL_MAP_READ, 0, bufferSize);
                queue.enqueueUnmapMemObject (hBufferOut, hPtrOut);
                queue.finish ();

                if (!io) hPtrIn = nullptr;
                break;
        }

        sat.init (width, height, scaling, Staging::NONE);

        // Create device buffers
        if (dBufferOut () == nullptr)
            dBufferOut = cl::Buffer (context, CL_MEM_WRITE_ONLY, bufferSize);

        // Set workspaces
        global = cl::NDRange (height, width);
        local = cl::NDRange (lXdim, lYdim);

        // Set kernel arguments
        kernel.setArg (0, sat.get (SAT::Memory::D_OUT));
        kernel.setArg (1, dBufferOut);
        kernel.setArg (2, cl::Local (lXdim * lYdim * sizeof (float)));
        kernel.setArg (3, radius);
        kernel.setArg (4, 1.f / scaling);
    }


    /*! \details The transfer happens from a staging buffer on the host to the 
     *           associated (specified) device buffer.
     *  
     *  \param[in] mem enumeration value specifying an input device buffer.
     *  \param[in] ptr a pointer to an array holding input data. If not NULL, the 
     *                 data from `ptr` will be copied to the associated staging buffer.
     *  \param[in] block a flag to indicate whether to perform a blocking 
     *                   or a non-blocking operation.
     *  \param[in] events a wait-list of events.
     *  \param[out] event event associated with the write operation to the device buffer.
     */
    void BoxFilterSAT::write (BoxFilterSAT::Memory mem, void *ptr, bool block, 
                              const std::vector<cl::Event> *events, cl::Event *event)
    {
        if (staging == Staging::I || staging == Staging::IO)
        {
            switch (mem)
            {
                case BoxFilterSAT::Memory::D_IN:
                    if (ptr != nullptr)
                        std::copy ((cl_float *) ptr, (cl_float *) ptr + width * height, hPtrIn);
                    queue.enqueueWriteBuffer ((cl::Buffer&) sat.get (SAT::Memory::D_IN), 
                        block, 0, bufferSize, hPtrIn, events, event);
                    break;
                default:
                    break;
            }
        }
    }


    /*! \details The transfer happens from a device buffer to the associated 
     *           (specified) staging buffer on the host.
     *  
     *  \param[in] mem enumeration value specifying an output staging buffer.
     *  \param[in] block a flag to indicate whether to perform a blocking 
     *                   or a non-blocking operation.
     *  \param[in] events a wait-list of events.
     *  \param[out] event event associated with the read operation to the staging buffer.
     */
    void* BoxFilterSAT::read (BoxFilterSAT::Memory mem, bool block, 
                              const std::vector<cl::Event> *events, cl::Event *event)
    {
        if (staging == Staging::O || staging == Staging::IO)
        {
            switch (mem)
            {
                case BoxFilterSAT::Memory::H_OUT:
                    queue.enqueueReadBuffer (dBufferOut, block, 0, bufferSize, hPtrOut, events, event);
                    return hPtrOut;
                default:
                    return nullptr;
            }
        }
        return nullptr;
    }


    /*! \details The function call is non-blocking.
     *
     *  \param[in] events a wait-list of events.
     *  \param[out] event event associated with the kernel execution.
     */
    void BoxFilterSAT::run (const std::vector<cl::Event> *events, cl::Event *event)
    {
        sat.run (events);
        queue.enqueueNDRangeKernel (kernel, cl::NullRange, global, local, nullptr, event);
    }


    /*! \return The radius of the square filter window.
     */
    int BoxFilterSAT::getRadius ()
    {
        return radius;
    }


    /*! \details Updates the kernel argument for the filter window radius.
     *
     *  \param[in] _radius the radius of the square filter window.
     */
    void BoxFilterSAT::setRadius (int _radius)
    {
        radius = _radius;
        kernel.setArg (3, radius);
    }

    /*! \return The scaling factor.
     */
    float BoxFilterSAT::getScaling ()
    {
        return scaling;
    }


    /*! \details Updates the kernel argument for the scaling factor.
     *
     *  \param[in] _scaling scaling factor.
     */
    void BoxFilterSAT::setScaling (float _scaling)
    {
        scaling = _scaling;
        sat.setScaling (scaling);
        kernel.setArg (4, 1.f / scaling);
    }


    /*! \param[in] _env opencl environment.
     *  \param[in] _info opencl configuration. Specifies the context, queue, etc, to be used.
     */
    BoxFilter::BoxFilter (clutils::CLEnv &_env, clutils::CLEnvInfo<1> _info) : 
        env (_env), info (_info), 
        context (env.getContext (info.pIdx)), 
        queue (env.getQueue (info.ctxIdx, info.qIdx[0])), 
        kernel (env.getProgram (info.pgIdx), "boxFilter")
    {
        // The class requires 16x16 work-groups (256 work-items per work-group)
        // The following code checks that this specification is possible

        cl::Device &device = env.devices[info.pIdx][info.dIdx];

        size_t maxLocalSize = device.getInfo<CL_DEVICE_MAX_WORK_GROUP_SIZE> ();

        std::vector<size_t> maxLocalDim = device.getInfo<CL_DEVICE_MAX_WORK_ITEM_SIZES> ();

        size_t wgMultiple = kernel.getWorkGroupInfo
            <CL_KERNEL_PREFERRED_WORK_GROUP_SIZE_MULTIPLE> (device);

        size_t localSize = lXdim * lYdim;

        try
        {
            if ((lXdim > maxLocalDim[0]) || (lYdim > maxLocalDim[1]))
            {
                std::ostringstream ss;
                ss << "The maximum work-group dimensions ";
                ss << "[" << maxLocalDim[0] << "][" << maxLocalDim[1] << "] ";
                ss << "are not enough (16x16 work-groups are required) on this device";
                throw ss.str ();
            }

            if (localSize > maxLocalSize)
            {
                std::ostringstream ss;
                ss << "The maximum work-group size ";
                ss << "[" << maxLocalSize << "] " << "is not enough ";
                ss << "(256 work-items per work-group are required) on this device";
                throw ss.str ();
            }

        }
        catch (const std::string &error)
        {
            std::cerr << "Error[BoxFilter]: " << error << std::endl;
            exit (EXIT_FAILURE);
        }

        if (localSize % wgMultiple)
            std::cout << "Warning[BoxFilter]: The work-group size [" << localSize 
                      << "] is not a multiple of the preferred size [" 
                      << wgMultiple << "] on this device" << std::endl;
    }


    /*! \details This interface exists to allow CL memory sharing between different kernels.
     *
     *  \param[in] mem enumeration value specifying the requested memory object.
     *  \return A reference to the requested memory object.
     */
    cl::Memory& BoxFilter::get (BoxFilter::Memory mem)
    {
        switch (mem)
        {
            case BoxFilter::Memory::H_IN:
                return hBufferIn;
            case BoxFilter::Memory::H_OUT:
                return hBufferOut;
            case BoxFilter::Memory::D_IN:
                return dBufferIn;
            case BoxFilter::Memory::D_OUT:
                return dBufferOut;
        }
    }


    /*! \details Sets up memory objects as necessary, and defines the kernel workspaces.
     *  \note If you have assigned a memory object to one member variable of the class 
     *        before the call to `init`, then that memory will be maintained. Otherwise, 
     *        a new memory object will be created.
     *        
     *  \param[in] _width width of the input array to be processed.
     *  \param[in] _height height of the input array to be processed.
     *  \param[in] _radius radius of the square filter window, i.e. \f$\ radius=filter\_width/2-1\f$.
     *  \param[in] _staging flag to indicate whether or not to instantiate the staging buffers.
     */
    void BoxFilter::init (unsigned int _width, unsigned int _height, int _radius, Staging _staging)
    {
        width = _width; height = _height; radius = _radius;
        bufferSize = width * height * sizeof (cl_float);
        staging = _staging;

        try
        {
            if ((width == 0) || (height == 0))
                throw "The image cannot have zeroed dimensions";

            if (height % lXdim != 0)
            {
                std::ostringstream ss;
                ss << "The image width has to be a multiple of " << lXdim;
                throw ss.str ().c_str ();
            }

            if (width % lYdim != 0)
            {
                std::ostringstream ss;
                ss << "The image height has to be a multiple of " << lYdim;
                throw ss.str ().c_str ();
            }
        }
        catch (const char *error)
        {
            std::cerr << "Error[BoxFilter]: " << error << std::endl;
            exit (EXIT_FAILURE);
        }
        
        // Set workspaces
        global = cl::NDRange (width, height);
        local = cl::NDRange (lXdim, lYdim);

        // Create staging buffers
        bool io = false;
        switch (staging)
        {
            case Staging::NONE:
                hPtrIn = nullptr;
                hPtrOut = nullptr;
                break;

            case Staging::IO:
                io = true;

            case Staging::I:
                if (hBufferIn () == nullptr)
                    hBufferIn = cl::Buffer (context, CL_MEM_ALLOC_HOST_PTR, bufferSize);

                hPtrIn = (cl_float *) queue.enqueueMapBuffer (
                    hBufferIn, CL_FALSE, CL_MAP_WRITE, 0, bufferSize);
                queue.enqueueUnmapMemObject (hBufferIn, hPtrIn);

                if (!io)
                {
                    queue.finish ();
                    hPtrOut = nullptr;
                    break;
                }

            case Staging::O:
                if (hBufferOut () == nullptr)
                    hBufferOut = cl::Buffer (context, CL_MEM_ALLOC_HOST_PTR, bufferSize);

                hPtrOut = (cl_float *) queue.enqueueMapBuffer (
                    hBufferOut, CL_FALSE, CL_MAP_READ, 0, bufferSize);
                queue.enqueueUnmapMemObject (hBufferOut, hPtrOut);
                queue.finish ();

                if (!io) hPtrIn = nullptr;
                break;
        }
        
        // Create device buffers
        if (dBufferIn () == nullptr)
            dBufferIn = cl::Buffer (context, CL_MEM_READ_ONLY, bufferSize);
        if (dBufferOut () == nullptr)
            dBufferOut = cl::Buffer (context, CL_MEM_WRITE_ONLY, bufferSize);

        // Set kernel arguments
        kernel.setArg (0, dBufferIn);
        kernel.setArg (1, dBufferOut);
        kernel.setArg (2, cl::Local ((lXdim + 2 * radius) * (lYdim + 2 * radius) * sizeof (cl_float)));
        kernel.setArg (3, radius);
    }


    /*! \details The transfer happens from a staging buffer on the host to the 
     *           associated (specified) device buffer.
     *  
     *  \param[in] mem enumeration value specifying an input device buffer.
     *  \param[in] ptr a pointer to an array holding input data. If not NULL, the 
     *                 data from `ptr` will be copied to the associated staging buffer.
     *  \param[in] block a flag to indicate whether to perform a blocking 
     *                   or a non-blocking operation.
     *  \param[in] events a wait-list of events.
     *  \param[out] event event associated with the write operation to the device buffer.
     */
    void BoxFilter::write (BoxFilter::Memory mem, void *ptr, bool block, 
                           const std::vector<cl::Event> *events, cl::Event *event)
    {
        if (staging == Staging::I || staging == Staging::IO)
        {
            switch (mem)
            {
                case BoxFilter::Memory::D_IN:
                    if (ptr != nullptr)
                        std::copy ((cl_float *) ptr, (cl_float *) ptr + width * height, hPtrIn);
                    queue.enqueueWriteBuffer (dBufferIn, block, 0, bufferSize, hPtrIn, events, event);
                    break;
                default:
                    break;
            }
        }
    }


    /*! \details The transfer happens from a device buffer to the associated 
     *           (specified) staging buffer on the host.
     *  
     *  \param[in] mem enumeration value specifying an output staging buffer.
     *  \param[in] block a flag to indicate whether to perform a blocking 
     *                   or a non-blocking operation.
     *  \param[in] events a wait-list of events.
     *  \param[out] event event associated with the read operation to the staging buffer.
     */
    void* BoxFilter::read (BoxFilter::Memory mem, bool block, 
                           const std::vector<cl::Event> *events, cl::Event *event)
    {
        if (staging == Staging::O || staging == Staging::IO)
        {
            switch (mem)
            {
                case BoxFilter::Memory::H_OUT:
                    queue.enqueueReadBuffer (dBufferOut, block, 0, bufferSize, hPtrOut, events, event);
                    return hPtrOut;
                default:
                    return nullptr;
            }
        }
        return nullptr;
    }


    /*! \details The function call is non-blocking.
     *
     *  \param[in] events a wait-list of events.
     *  \param[out] event event associated with the kernel execution.
     */
    void BoxFilter::run (const std::vector<cl::Event> *events, cl::Event *event)
    {
        queue.enqueueNDRangeKernel (kernel, cl::NullRange, global, local, events, event);
    }


    /*! \return The radius of the square filter window.
     */
    int BoxFilter::getRadius ()
    {
        return radius;
    }


    /*! \details Updates the kernel argument for the filter radius.
     *
     *  \param[in] _radius radius of the square filter window.
     */
    void BoxFilter::setRadius (int _radius)
    {
        radius = _radius;
        kernel.setArg (3, radius);
    }


    /*! \param[in] _env opencl environment.
     *  \param[in] _info opencl configuration. Specifies the context, queue, etc, to be used.
     *                   The class requires **two** `(2)` **command queues** (on the same device).
     */
    GuidedFilter<GuidedFilterConfig::I_EQ_P>::GuidedFilter (clutils::CLEnv &_env, clutils::CLEnvInfo<2> _info) : 
        env (_env), info (_info), 
        context (env.getContext (info.pIdx)), 
        queue0 (env.getQueue (info.ctxIdx, info.qIdx[0])), 
        mean_p  (env, info.getCLEnvInfo (0)), mean_p2 (env, info.getCLEnvInfo (1)), 
        mean_a  (env, info.getCLEnvInfo (0)), mean_b  (env, info.getCLEnvInfo (1)), 
        squared (env, info.getCLEnvInfo (1)), 
        ab (env.getProgram (info.pgIdx), "gf_ab"), 
        q (env.getProgram (info.pgIdx), "gf_q"), 
        waitListAB (1), waitListMB(1), waitListQ (1)
    {
    }


    /*! \details This interface exists to allow CL memory sharing between different kernels.
     *
     *  \param[in] mem enumeration value specifying the requested memory object.
     *  \return A reference to the requested memory object.
     */
    cl::Memory& GuidedFilter<GuidedFilterConfig::I_EQ_P>::get (GuidedFilter::Memory mem)
    {
        switch (mem)
        {
            case GuidedFilter::Memory::H_IN:
                return hBufferIn;
            case GuidedFilter::Memory::H_OUT:
                return hBufferOut;
            case GuidedFilter::Memory::D_IN:
                return dBufferIn;
            case GuidedFilter::Memory::D_OUT:
                return dBufferOut;
            case GuidedFilter::Memory::D_A:
                return dBufferOutA;
            case GuidedFilter::Memory::D_B:
                return dBufferOutB;
        }
    }


    /*! \details Sets up memory objects as necessary, and defines the kernel workspaces.
     *  \note If you have assigned a memory object to one member variable of the class 
     *        before the call to `init`, then that memory will be maintained. Otherwise, 
     *        a new memory object will be created.
     *  \note For better accuracy control a scaling is applied on the array elements 
     *        internally in `BoxFilterSAT`. A default value of \f$ 0.0001\ (1e-4) \f$ 
     *        is normally used. The input data are assumed to be of `uchar` type promoted 
     *        to `float` and normalized to `1.0`. Configure the scaling for your own data.
     *        
     *  \param[in] _width width of the input array to be processed.
     *  \param[in] _height height of the input array to be processed.
     *  \param[in] _radius radius of the square filter window, i.e. \f$\ radius=filter\_width/2-1\f$.
     *  \param[in] _eps regularization parameter \f$ \epsilon \f$.
     *  \param[in] _zero_out flag to indicate whether or not to zero out invalid pixels. 
     *                       For more information, look at `gf_q`'s documentation 
     *                       in `kernels/guidedFilter_kernels.cl`.
     *  \param[in] _boxScaling scaling factor applied internally to `BoxFilterSAT`.
     *  \param[in] _outputScaling scaling factor applied to the output array. Set this to `1/s`, if 
     *                            you had to apply an `s` scaling to the input array before processing.
     *  \param[in] _staging flag to indicate whether or not to instantiate the staging buffers.
     */
    void GuidedFilter<GuidedFilterConfig::I_EQ_P>::init (
        unsigned int _width, unsigned int _height, int _radius, float _eps, 
        int _zero_out, float _boxScaling, float _outputScaling, Staging _staging)
    {
        width = _width; height = _height; radius = _radius; eps = _eps;
        bufferSize = width * height * sizeof (cl_float);
        zero_out = _zero_out;
        boxScaling = _boxScaling;
        outputScaling = _outputScaling;
        staging = _staging;

        try
        {
            if ((width == 0) || (height == 0))
                throw "The image cannot have zeroed dimensions";

            if ((width * height) % 4 != 0)
                throw "The number of elements in the array has to be a multiple of 4";
        }
        catch (const char *error)
        {
            std::cerr << "Error[GuidedFilter<GuidedFilterConfig::I_EQ_P>]: " << error << std::endl;
            exit (EXIT_FAILURE);
        }

        // Create staging buffers
        bool io = false;
        switch (staging)
        {
            case Staging::NONE:
                hPtrIn = nullptr;
                hPtrOut = nullptr;
                break;

            case Staging::IO:
                io = true;

            case Staging::I:
                if (hBufferIn () == nullptr)
                    hBufferIn = cl::Buffer (context, CL_MEM_ALLOC_HOST_PTR, bufferSize);

                hPtrIn = (cl_float *) queue0.enqueueMapBuffer (
                    hBufferIn, CL_FALSE, CL_MAP_WRITE, 0, bufferSize);
                queue0.enqueueUnmapMemObject (hBufferIn, hPtrIn);

                if (!io)
                {
                    queue0.finish ();
                    hPtrOut = nullptr;
                    break;
                }

            case Staging::O:
                if (hBufferOut () == nullptr)
                    hBufferOut = cl::Buffer (context, CL_MEM_ALLOC_HOST_PTR, bufferSize);

                hPtrOut = (cl_float *) queue0.enqueueMapBuffer (
                    hBufferOut, CL_FALSE, CL_MAP_READ, 0, bufferSize);
                queue0.enqueueUnmapMemObject (hBufferOut, hPtrOut);
                queue0.finish ();

                if (!io) hPtrIn = nullptr;
                break;
        }
        
        // Create device buffers
        if (dBufferIn () == nullptr)
            dBufferIn = cl::Buffer (context, CL_MEM_READ_ONLY, bufferSize);
        if (dBufferOut () == nullptr)
            dBufferOut = cl::Buffer (context, CL_MEM_WRITE_ONLY, bufferSize);

        mean_p.get (BoxFilterSAT::Memory::D_IN) = dBufferIn;
        mean_p.get (BoxFilterSAT::Memory::D_OUT) = cl::Buffer (context, CL_MEM_READ_WRITE, bufferSize);
        mean_p.init (width, height, radius, boxScaling, Staging::NONE);

        squared.get (Math::Pown::Memory::D_IN) = dBufferIn;
        squared.get (Math::Pown::Memory::D_OUT) = cl::Buffer (context, CL_MEM_READ_WRITE, bufferSize);
        squared.init (width, height, 2, Staging::NONE);

        mean_p2.get (BoxFilterSAT::Memory::D_IN) = squared.get (Math::Pown::Memory::D_OUT);
        mean_p2.get (BoxFilterSAT::Memory::D_OUT) = cl::Buffer (context, CL_MEM_READ_WRITE, bufferSize);
        mean_p2.init (width, height, radius, boxScaling, Staging::NONE);

        dBufferOutA = cl::Buffer (context, CL_MEM_READ_WRITE, bufferSize);
        dBufferOutB = cl::Buffer (context, CL_MEM_READ_WRITE, bufferSize);
        ab.setArg (0, mean_p.get (BoxFilterSAT::Memory::D_OUT));
        ab.setArg (1, mean_p2.get (BoxFilterSAT::Memory::D_OUT));
        ab.setArg (2, dBufferOutA);
        ab.setArg (3, dBufferOutB);
        ab.setArg (4, eps);

        mean_a.get (BoxFilterSAT::Memory::D_IN) = dBufferOutA;
        mean_a.get (BoxFilterSAT::Memory::D_OUT) = cl::Buffer (context, CL_MEM_READ_WRITE, bufferSize);
        mean_a.init (width, height, radius, boxScaling, Staging::NONE);

        mean_b.get (BoxFilterSAT::Memory::D_IN) = dBufferOutB;
        mean_b.get (BoxFilterSAT::Memory::D_OUT) = cl::Buffer (context, CL_MEM_READ_WRITE, bufferSize);
        mean_b.init (width, height, radius, boxScaling, Staging::NONE);

        q.setArg (0, dBufferIn);
        q.setArg (1, mean_a.get (BoxFilterSAT::Memory::D_OUT));
        q.setArg (2, mean_b.get (BoxFilterSAT::Memory::D_OUT));
        q.setArg (3, dBufferOut);
        q.setArg (4, zero_out);
        q.setArg (5, outputScaling);
        
        // Set workspaces (common to both own kernels: ab, q)
        global = cl::NDRange (width * height / 4);
    }


    /*! \details The transfer happens from a staging buffer on the host to the 
     *           associated (specified) device buffer.
     *  \note The transfer is handled by the first command queue.
     *  
     *  \param[in] mem enumeration value specifying an input device buffer.
     *  \param[in] ptr a pointer to an array holding input data. If not NULL, the 
     *                 data from `ptr` will be copied to the associated staging buffer.
     *  \param[in] block a flag to indicate whether to perform a blocking 
     *                   or a non-blocking operation.
     *  \param[in] events a wait-list of events.
     *  \param[out] event event associated with the write operation to the device buffer.
     */
    void GuidedFilter<GuidedFilterConfig::I_EQ_P>::write (
        GuidedFilter::Memory mem, void *ptr, bool block, const std::vector<cl::Event> *events, cl::Event *event)
    {
        if (staging == Staging::I || staging == Staging::IO)
        {
            switch (mem)
            {
                case GuidedFilter::Memory::D_IN:
                    if (ptr != nullptr)
                        std::copy ((cl_float *) ptr, (cl_float *) ptr + width * height, hPtrIn);
                    queue0.enqueueWriteBuffer (dBufferIn, block, 0, bufferSize, hPtrIn, events, event);
                    break;
                default:
                    break;
            }
        }
    }


    /*! \details The transfer happens from a device buffer to the associated 
     *           (specified) staging buffer on the host.
     *  \note The transfer is handled by the first command queue.
     *  
     *  \param[in] mem enumeration value specifying an output staging buffer.
     *  \param[in] block a flag to indicate whether to perform a blocking 
     *                   or a non-blocking operation.
     *  \param[in] events a wait-list of events.
     *  \param[out] event event associated with the read operation to the staging buffer.
     */
    void* GuidedFilter<GuidedFilterConfig::I_EQ_P>::read (
        GuidedFilter::Memory mem, bool block, const std::vector<cl::Event> *events, cl::Event *event)
    {
        if (staging == Staging::O || staging == Staging::IO)
        {
            switch (mem)
            {
                case GuidedFilter::Memory::H_OUT:
                    queue0.enqueueReadBuffer (dBufferOut, block, 0, bufferSize, hPtrOut, events, event);
                    return hPtrOut;
                default:
                    return nullptr;
            }
        }
        return nullptr;
    }


    /*! \details The function call is non-blocking.
     *
     *  \param[in] events a wait-list of events.
     *  \param[out] event event associated with the kernel execution.
     */
    void GuidedFilter<GuidedFilterConfig::I_EQ_P>::run (const std::vector<cl::Event> *events, cl::Event *event)
    {
        mean_p.run (events);
        squared.run (events);
        mean_p2.run (nullptr, &p2Event); waitListAB[0] = p2Event;
        queue0.enqueueNDRangeKernel (ab, cl::NullRange, global, cl::NullRange, &waitListAB, &abEvent);
        mean_a.run ();
        waitListMB[0] = abEvent;
        mean_b.run (&waitListMB, &mbEvent); waitListQ[0] = mbEvent;
        queue0.enqueueNDRangeKernel (q, cl::NullRange, global, cl::NullRange, &waitListQ, event);
    }


    /*! \return The radius of the square filter window.
     */
    int GuidedFilter<GuidedFilterConfig::I_EQ_P>::getRadius ()
    {
        return radius;
    }


    /*! \details Updates the kernel argument for the filter window radius.
     *
     *  \param[in] _radius radius of the square filter window.
     */
    void GuidedFilter<GuidedFilterConfig::I_EQ_P>::setRadius (int _radius)
    {
        radius = _radius;
        mean_p.setRadius (radius);
        mean_p2.setRadius (radius);
        mean_a.setRadius (radius);
        mean_b.setRadius (radius);
    }


    /*! \return The regularization parameter \f$\epsilon\f$.
     */
    float GuidedFilter<GuidedFilterConfig::I_EQ_P>::getEps ()
    {
        return eps;
    }


    /*! \details Updates the kernel argument for the regularization parameter \f$\epsilon\f$.
     *
     *  \param[in] _eps regularization parameter \f$\epsilon\f$.
     */
    void GuidedFilter<GuidedFilterConfig::I_EQ_P>::setEps (float _eps)
    {
        eps = _eps;
        ab.setArg (4, eps);
    }


    /*! \return The scaling factor applied internally to `BoxFilterSAT`.
     */
    float GuidedFilter<GuidedFilterConfig::I_EQ_P>::getBoxScaling ()
    {
        return boxScaling;
    }


    /*! \details Updates the kernel argument for internal scaling 
     *           of the array elements in `BoxFilterSAT`.
     *
     *  \param[in] _boxScaling scaling factor for `BoxFilterSAT`.
     */
    void GuidedFilter<GuidedFilterConfig::I_EQ_P>::setBoxScaling (float _boxScaling)
    {
        boxScaling = _boxScaling;
        mean_p.setScaling (boxScaling);
        mean_p2.setScaling (boxScaling);
        mean_a.setScaling (boxScaling);
        mean_b.setScaling (boxScaling);
    }


    /*! \return The scaling factor for the output array.
     */
    float GuidedFilter<GuidedFilterConfig::I_EQ_P>::getOutputScaling ()
    {
        return outputScaling;
    }


    /*! \details Updates the kernel argument for scaling of the output array.
     *
     *  \param[in] _outputScaling scaling factor for the output array.
     */
    void GuidedFilter<GuidedFilterConfig::I_EQ_P>::setOutputScaling (float _outputScaling)
    {
        outputScaling = _outputScaling;
        mean_p.setScaling (outputScaling);
        mean_p2.setScaling (outputScaling);
        mean_a.setScaling (outputScaling);
        mean_b.setScaling (outputScaling);
    }


    /*! \return The `zero_out` flag.
     */
    int GuidedFilter<GuidedFilterConfig::I_EQ_P>::getZeroing ()
    {
        return zero_out;
    }


    /*! \details Updates the kernel argument for the `zero_out` flag.
     *
     *  \param[in] _zero_out `zero_out` flag.
     */
    void GuidedFilter<GuidedFilterConfig::I_EQ_P>::setZeroing (int _zero_out)
    {
        zero_out = _zero_out;
        q.setArg (4, zero_out);
    }


    /*! \param[in] _env opencl environment.
     *  \param[in] _info opencl configuration. Specifies the context, queue, etc, to be used.
     *                   The class requires **two** `(2)` **command queues** (on the same device).
     */
    GuidedFilter<GuidedFilterConfig::I_NEQ_P>::GuidedFilter (clutils::CLEnv &_env, clutils::CLEnvInfo<2> _info) : 
        env (_env), info (_info), 
        context (env.getContext (info.pIdx)), 
        queue0 (env.getQueue (info.ctxIdx, info.qIdx[0])), 
        mean_I  (env, info.getCLEnvInfo (0)), mean_p  (env, info.getCLEnvInfo (1)), 
        corr_I  (env, info.getCLEnvInfo (0)), corr_Ip (env, info.getCLEnvInfo (1)), 
        mean_a  (env, info.getCLEnvInfo (0)), mean_b  (env, info.getCLEnvInfo (1)), 
        mult_II (env, info.getCLEnvInfo (0)), mult_Ip (env, info.getCLEnvInfo (1)), 
        var (env.getProgram (info.pgIdx), "gf_var_Ip"), 
        ab (env.getProgram (info.pgIdx), "gf_ab_Ip"), 
        q (env.getProgram (info.pgIdx), "gf_q"), 
        waitListVar (1), waitListMB(1), waitListQ (1)
    {
    }


    /*! \details This interface exists to allow CL memory sharing between different kernels.
     *
     *  \param[in] mem enumeration value specifying the requested memory object.
     *  \return A reference to the requested memory object.
     */
    cl::Memory& GuidedFilter<GuidedFilterConfig::I_NEQ_P>::get (GuidedFilter::Memory mem)
    {
        switch (mem)
        {
            case GuidedFilter::Memory::H_IN_I:
                return hBufferInI;
            case GuidedFilter::Memory::H_IN_P:
                return hBufferInP;
            case GuidedFilter::Memory::H_OUT:
                return hBufferOut;
            case GuidedFilter::Memory::D_IN_I:
                return dBufferInI;
            case GuidedFilter::Memory::D_IN_P:
                return dBufferInP;
            case GuidedFilter::Memory::D_OUT:
                return dBufferOut;
            case GuidedFilter::Memory::D_A:
                return dBufferOutA;
            case GuidedFilter::Memory::D_B:
                return dBufferOutB;
            case GuidedFilter::Memory::D_VAR_I:
                return dBufferOutVarI;
            case GuidedFilter::Memory::D_COV_IP:
                return dBufferOutCovIp;
        }
    }


    /*! \details Sets up memory objects as necessary, and defines the kernel workspaces.
     *  \note If you have assigned a memory object to one member variable of the class 
     *        before the call to `init`, then that memory will be maintained. Otherwise, 
     *        a new memory object will be created.
     *  \note For better accuracy control a scaling is applied on the array elements 
     *        internally in `BoxFilterSAT`. A default value of \f$ 0.0001\ (1e-4) \f$ 
     *        is normally used. The input data are assumed to be of `uchar` type promoted 
     *        to `float` and normalized to `1.0`. Configure the scaling for your own data.
     *        
     *  \param[in] _width width of the input array to be processed.
     *  \param[in] _height height of the input array to be processed.
     *  \param[in] _radius radius of the square filter window, i.e. \f$\ radius=filter\_width/2-1\f$.
     *  \param[in] _eps regularization parameter \f$ \epsilon \f$.
     *  \param[in] _zero_out flag to indicate whether or not to zero out invalid pixels. 
     *                       For more information, look at `gf_q`'s documentation 
     *                       in `kernels/guidedFilter_kernels.cl`.
     *  \param[in] _boxScaling scaling factor applied internally to `BoxFilterSAT`.
     *  \param[in] _staging flag to indicate whether or not to instantiate the staging buffers.
     */
    void GuidedFilter<GuidedFilterConfig::I_NEQ_P>::init (
        unsigned int _width, unsigned int _height, int _radius, float _eps, 
        int _zero_out, float _boxScaling, Staging _staging)
    {
        width = _width; height = _height; radius = _radius; eps = _eps;
        bufferSize = width * height * sizeof (cl_float);
        zero_out = _zero_out;
        boxScaling = _boxScaling;
        staging = _staging;

        try
        {
            if ((width == 0) || (height == 0))
                throw "The image cannot have zeroed dimensions";

            if ((width * height) % 4 != 0)
                throw "The number of elements in the array has to be a multiple of 4";
        }
        catch (const char *error)
        {
            std::cerr << "Error[GuidedFilter<GuidedFilterConfig::I_NEQ_P>]: " << error << std::endl;
            exit (EXIT_FAILURE);
        }

        // Create staging buffers
        bool io = false;
        switch (staging)
        {
            case Staging::NONE:
                hPtrInI = nullptr;
                hPtrInP = nullptr;
                hPtrOut = nullptr;
                break;

            case Staging::IO:
                io = true;

            case Staging::I:
                if (hBufferInI () == nullptr)
                    hBufferInI = cl::Buffer (context, CL_MEM_ALLOC_HOST_PTR, bufferSize);
                if (hBufferInP () == nullptr)
                    hBufferInP = cl::Buffer (context, CL_MEM_ALLOC_HOST_PTR, bufferSize);

                hPtrInI = (cl_float *) queue0.enqueueMapBuffer (
                    hBufferInI, CL_FALSE, CL_MAP_WRITE, 0, bufferSize);
                hPtrInP = (cl_float *) queue0.enqueueMapBuffer (
                    hBufferInP, CL_FALSE, CL_MAP_WRITE, 0, bufferSize);
                queue0.enqueueUnmapMemObject (hBufferInI, hPtrInI);
                queue0.enqueueUnmapMemObject (hBufferInP, hPtrInP);

                if (!io)
                {
                    queue0.finish ();
                    hPtrOut = nullptr;
                    break;
                }

            case Staging::O:
                if (hBufferOut () == nullptr)
                    hBufferOut = cl::Buffer (context, CL_MEM_ALLOC_HOST_PTR, bufferSize);

                hPtrOut = (cl_float *) queue0.enqueueMapBuffer (
                    hBufferOut, CL_FALSE, CL_MAP_READ, 0, bufferSize);
                queue0.enqueueUnmapMemObject (hBufferOut, hPtrOut);
                queue0.finish ();

                if (!io) { hPtrInI = nullptr; hPtrInP = nullptr; }
                break;
        }
        
        // Create device buffers
        if (dBufferInI () == nullptr)
            dBufferInI = cl::Buffer (context, CL_MEM_READ_ONLY, bufferSize);
        if (dBufferInP () == nullptr)
            dBufferInP = cl::Buffer (context, CL_MEM_READ_ONLY, bufferSize);
        if (dBufferOut () == nullptr)
            dBufferOut = cl::Buffer (context, CL_MEM_WRITE_ONLY, bufferSize);

        mean_I.get (BoxFilterSAT::Memory::D_IN) = dBufferInI;
        mean_I.get (BoxFilterSAT::Memory::D_OUT) = cl::Buffer (context, CL_MEM_READ_WRITE, bufferSize);
        mean_I.init (width, height, radius, boxScaling, Staging::NONE);

        mean_p.get (BoxFilterSAT::Memory::D_IN) = dBufferInP;
        mean_p.get (BoxFilterSAT::Memory::D_OUT) = cl::Buffer (context, CL_MEM_READ_WRITE, bufferSize);
        mean_p.init (width, height, radius, boxScaling, Staging::NONE);

        mult_II.get (Math::Mult::Memory::D_IN_A) = dBufferInI;
        mult_II.get (Math::Mult::Memory::D_IN_B) = dBufferInI;
        mult_II.get (Math::Mult::Memory::D_OUT) = cl::Buffer (context, CL_MEM_READ_WRITE, bufferSize);
        mult_II.init (width, height, Staging::NONE);

        mult_Ip.get (Math::Mult::Memory::D_IN_A) = dBufferInI;
        mult_Ip.get (Math::Mult::Memory::D_IN_B) = dBufferInP;
        mult_Ip.get (Math::Mult::Memory::D_OUT) = cl::Buffer (context, CL_MEM_READ_WRITE, bufferSize);
        mult_Ip.init (width, height, Staging::NONE);

        corr_I.get (BoxFilterSAT::Memory::D_IN) = mult_II.get (Math::Mult::Memory::D_OUT);
        corr_I.get (BoxFilterSAT::Memory::D_OUT) = cl::Buffer (context, CL_MEM_READ_WRITE, bufferSize);
        corr_I.init (width, height, radius, boxScaling, Staging::NONE);

        corr_Ip.get (BoxFilterSAT::Memory::D_IN) = mult_Ip.get (Math::Mult::Memory::D_OUT);
        corr_Ip.get (BoxFilterSAT::Memory::D_OUT) = cl::Buffer (context, CL_MEM_READ_WRITE, bufferSize);
        corr_Ip.init (width, height, radius, boxScaling, Staging::NONE);

        dBufferOutVarI = cl::Buffer (context, CL_MEM_READ_WRITE, bufferSize);
        dBufferOutCovIp = cl::Buffer (context, CL_MEM_READ_WRITE, bufferSize);
        var.setArg (0, corr_I.get (BoxFilterSAT::Memory::D_OUT));
        var.setArg (1, corr_Ip.get (BoxFilterSAT::Memory::D_OUT));
        var.setArg (2, mean_I.get (BoxFilterSAT::Memory::D_OUT));
        var.setArg (3, mean_p.get (BoxFilterSAT::Memory::D_OUT));
        var.setArg (4, dBufferOutVarI);
        var.setArg (5, dBufferOutCovIp);

        dBufferOutA = cl::Buffer (context, CL_MEM_READ_WRITE, bufferSize);
        dBufferOutB = cl::Buffer (context, CL_MEM_READ_WRITE, bufferSize);
        ab.setArg (0, dBufferOutVarI);
        ab.setArg (1, dBufferOutCovIp);
        ab.setArg (2, mean_I.get (BoxFilterSAT::Memory::D_OUT));
        ab.setArg (3, mean_p.get (BoxFilterSAT::Memory::D_OUT));
        ab.setArg (4, dBufferOutA);
        ab.setArg (5, dBufferOutB);
        ab.setArg (6, eps);

        mean_a.get (BoxFilterSAT::Memory::D_IN) = dBufferOutA;
        mean_a.get (BoxFilterSAT::Memory::D_OUT) = cl::Buffer (context, CL_MEM_READ_WRITE, bufferSize);
        mean_a.init (width, height, radius, boxScaling, Staging::NONE);

        mean_b.get (BoxFilterSAT::Memory::D_IN) = dBufferOutB;
        mean_b.get (BoxFilterSAT::Memory::D_OUT) = cl::Buffer (context, CL_MEM_READ_WRITE, bufferSize);
        mean_b.init (width, height, radius, boxScaling, Staging::NONE);

        q.setArg (0, dBufferInI);
        q.setArg (1, mean_a.get (BoxFilterSAT::Memory::D_OUT));
        q.setArg (2, mean_b.get (BoxFilterSAT::Memory::D_OUT));
        q.setArg (3, dBufferOut);
        q.setArg (4, zero_out);
        q.setArg (5, 1.f);
        
        // Set workspaces (common to both own kernels: ab, q)
        global = cl::NDRange (width * height / 4);
    }


    /*! \details The transfer happens from a staging buffer on the host to the 
     *           associated (specified) device buffer.
     *  \note The transfer is handled by the first command queue.
     *  
     *  \param[in] mem enumeration value specifying an input device buffer.
     *  \param[in] ptr a pointer to an array holding input data. If not NULL, the 
     *                 data from `ptr` will be copied to the associated staging buffer.
     *  \param[in] block a flag to indicate whether to perform a blocking 
     *                   or a non-blocking operation.
     *  \param[in] events a wait-list of events.
     *  \param[out] event event associated with the write operation to the device buffer.
     */
    void GuidedFilter<GuidedFilterConfig::I_NEQ_P>::write (
        GuidedFilter::Memory mem, void *ptr, bool block, const std::vector<cl::Event> *events, cl::Event *event)
    {
        if (staging == Staging::I || staging == Staging::IO)
        {
            switch (mem)
            {
                case GuidedFilter::Memory::D_IN_I:
                    if (ptr != nullptr)
                        std::copy ((cl_float *) ptr, (cl_float *) ptr + width * height, hPtrInI);
                    queue0.enqueueWriteBuffer (dBufferInI, block, 0, bufferSize, hPtrInI, events, event);
                    break;
                case GuidedFilter::Memory::D_IN_P:
                    if (ptr != nullptr)
                        std::copy ((cl_float *) ptr, (cl_float *) ptr + width * height, hPtrInP);
                    queue0.enqueueWriteBuffer (dBufferInP, block, 0, bufferSize, hPtrInP, events, event);
                    break;
                default:
                    break;
            }
        }
    }


    /*! \details The transfer happens from a device buffer to the associated 
     *           (specified) staging buffer on the host.
     *  \note The transfer is handled by the first command queue.
     *  
     *  \param[in] mem enumeration value specifying an output staging buffer.
     *  \param[in] block a flag to indicate whether to perform a blocking 
     *                   or a non-blocking operation.
     *  \param[in] events a wait-list of events.
     *  \param[out] event event associated with the read operation to the staging buffer.
     */
    void* GuidedFilter<GuidedFilterConfig::I_NEQ_P>::read (
        GuidedFilter::Memory mem, bool block, const std::vector<cl::Event> *events, cl::Event *event)
    {
        if (staging == Staging::O || staging == Staging::IO)
        {
            switch (mem)
            {
                case GuidedFilter::Memory::H_OUT:
                    queue0.enqueueReadBuffer (dBufferOut, block, 0, bufferSize, hPtrOut, events, event);
                    return hPtrOut;
                default:
                    return nullptr;
            }
        }
        return nullptr;
    }


    /*! \details The function call is non-blocking.
     *
     *  \param[in] events a wait-list of events.
     *  \param[out] event event associated with the kernel execution.
     */
    void GuidedFilter<GuidedFilterConfig::I_NEQ_P>::run (const std::vector<cl::Event> *events, cl::Event *event)
    {
        mean_I.run (events);
        mean_p.run (events);
        
        mult_II.run (events);
        corr_I.run ();
        mult_Ip.run (events);
        corr_Ip.run (nullptr, &corrIpEvent); waitListVar[0] = corrIpEvent;
        
        queue0.enqueueNDRangeKernel (var, cl::NullRange, global, cl::NullRange, &waitListVar);
        queue0.enqueueNDRangeKernel (ab, cl::NullRange, global, cl::NullRange, nullptr, &abEvent);
        mean_a.run ();
        waitListMB[0] = abEvent;
        mean_b.run (&waitListMB, &mbEvent); waitListQ[0] = mbEvent;
        queue0.enqueueNDRangeKernel (q, cl::NullRange, global, cl::NullRange, &waitListQ, event);
    }


    /*! \return The radius of the square filter window.
     */
    int GuidedFilter<GuidedFilterConfig::I_NEQ_P>::getRadius ()
    {
        return radius;
    }


    /*! \details Updates the kernel argument for the filter window radius.
     *
     *  \param[in] _radius radius of the square filter window.
     */
    void GuidedFilter<GuidedFilterConfig::I_NEQ_P>::setRadius (int _radius)
    {
        radius = _radius;
        mean_I.setRadius (radius);
        mean_p.setRadius (radius);
        corr_I.setRadius (radius);
        corr_Ip.setRadius (radius);
        mean_a.setRadius (radius);
        mean_b.setRadius (radius);
    }


    /*! \return The regularization parameter \f$\epsilon\f$.
     */
    float GuidedFilter<GuidedFilterConfig::I_NEQ_P>::getEps ()
    {
        return eps;
    }


    /*! \details Updates the kernel argument for the regularization parameter \f$\epsilon\f$.
     *
     *  \param[in] _eps regularization parameter \f$\epsilon\f$.
     */
    void GuidedFilter<GuidedFilterConfig::I_NEQ_P>::setEps (float _eps)
    {
        eps = _eps;
        ab.setArg (6, eps);
    }


    /*! \return The scaling factor applied internally to `BoxFilterSAT`.
     */
    float GuidedFilter<GuidedFilterConfig::I_NEQ_P>::getBoxScaling ()
    {
        return boxScaling;
    }


    /*! \details Updates the kernel argument for internal scaling 
     *           of the array elements in `BoxFilterSAT`.
     *
     *  \param[in] _scaling scaling factor for `BoxFilterSAT`.
     */
    void GuidedFilter<GuidedFilterConfig::I_NEQ_P>::setBoxScaling (float _boxScaling)
    {
        boxScaling = _boxScaling;
        mean_I.setScaling (boxScaling);
        mean_p.setScaling (boxScaling);
        corr_I.setScaling (boxScaling);
        corr_Ip.setScaling (boxScaling);
        mean_a.setScaling (boxScaling);
        mean_b.setScaling (boxScaling);
    }


    /*! \return The `zero_out` flag.
     */
    int GuidedFilter<GuidedFilterConfig::I_NEQ_P>::getZeroing ()
    {
        return zero_out;
    }


    /*! \details Updates the kernel argument for the `zero_out` flag.
     *
     *  \param[in] _zero_out `zero_out` flag.
     */
    void GuidedFilter<GuidedFilterConfig::I_NEQ_P>::setZeroing (int _zero_out)
    {
        zero_out = _zero_out;
        q.setArg (4, zero_out);
    }


    namespace Kinect
    {

        /*! \param[in] _env opencl environment.
         *  \param[in] _info opencl configuration. Specifies the context, queue, etc, to be used.
         *                   The class requires **two** `(2)` **command queues** (on the same device).
         */
        GuidedFilterRGB<GuidedFilterRGBConfig::SEPARATED>::GuidedFilterRGB (
            clutils::CLEnv &_env, clutils::CLEnvInfo<2> _info) : 
            env (_env), info (_info), 
            context (env.getContext (info.pIdx)), 
            queue0 (env.getQueue (info.ctxIdx, info.qIdx[0])), 
            sRGB  (env, info.getCLEnvInfo (0)), 
            gfR (env, info), gfG (env, info), gfB (env, info), 
            waitList (1)
        {
        }


        /*! \details This interface exists to allow CL memory sharing between different kernels.
         *
         *  \param[in] mem enumeration value specifying the requested memory object.
         *  \return A reference to the requested memory object.
         */
        cl::Memory& GuidedFilterRGB<GuidedFilterRGBConfig::SEPARATED>::get (GuidedFilterRGB::Memory mem)
        {
            switch (mem)
            {
                case GuidedFilterRGB::Memory::H_IN:
                    return hBufferIn;
                case GuidedFilterRGB::Memory::H_OUT_R:
                    return hBufferOutR;
                case GuidedFilterRGB::Memory::H_OUT_G:
                    return hBufferOutG;
                case GuidedFilterRGB::Memory::H_OUT_B:
                    return hBufferOutB;
                case GuidedFilterRGB::Memory::D_INTR_R:
                    return sRGB.get (SeparateRGB<SeparateRGBConfig::UCHAR_FLOAT>::Memory::D_OUT_R);
                case GuidedFilterRGB::Memory::D_INTR_G:
                    return sRGB.get (SeparateRGB<SeparateRGBConfig::UCHAR_FLOAT>::Memory::D_OUT_G);
                case GuidedFilterRGB::Memory::D_INTR_B:
                    return sRGB.get (SeparateRGB<SeparateRGBConfig::UCHAR_FLOAT>::Memory::D_OUT_B);
                case GuidedFilterRGB::Memory::D_IN:
                    return dBufferIn;
                case GuidedFilterRGB::Memory::D_OUT_R:
                    return dBufferOutR;
                case GuidedFilterRGB::Memory::D_OUT_G:
                    return dBufferOutG;
                case GuidedFilterRGB::Memory::D_OUT_B:
                    return dBufferOutB;
            }
        }


        /*! \details Sets up memory objects as necessary, and defines the kernel workspaces.
         *  \note If you have assigned a memory object to one member variable of the class 
         *        before the call to `init`, then that memory will be maintained. Otherwise, 
         *        a new memory object will be created.
         *        
         *  \param[in] _width width of the input array to be processed.
         *  \param[in] _height height of the input array to be processed.
         *  \param[in] _radius radius of the square filter window, i.e. \f$\ radius=filter\_width/2-1\f$.
         *  \param[in] _eps regularization parameter \f$ \epsilon \f$.
         *  \param[in] _staging flag to indicate whether or not to instantiate the staging buffers.
         */
        void GuidedFilterRGB<GuidedFilterRGBConfig::SEPARATED>::init (
            unsigned int _width, unsigned int _height, int _radius, float _eps, Staging _staging)
        {
            width = _width; height = _height; radius = _radius; eps = _eps;
            bufferInSize = 3 * width * height * sizeof (cl_uchar);
            bufferOutSize = width * height * sizeof (cl_float);
            staging = _staging;

            // Create staging buffers
            bool io = false;
            switch (staging)
            {
                case Staging::NONE:
                    hPtrIn = nullptr;
                    hPtrOutR = nullptr;
                    hPtrOutG = nullptr;
                    hPtrOutB = nullptr;
                    break;

                case Staging::IO:
                    io = true;

                case Staging::I:
                    if (hBufferIn () == nullptr)
                        hBufferIn = cl::Buffer (context, CL_MEM_ALLOC_HOST_PTR, bufferInSize);

                    hPtrIn = (cl_uchar *) queue0.enqueueMapBuffer (
                        hBufferIn, CL_FALSE, CL_MAP_WRITE, 0, bufferInSize);
                    queue0.enqueueUnmapMemObject (hBufferIn, hPtrIn);

                    if (!io)
                    {
                        queue0.finish ();
                        hPtrOutR = nullptr;
                        hPtrOutG = nullptr;
                        hPtrOutB = nullptr;
                        break;
                    }

                case Staging::O:
                    if (hBufferOutR () == nullptr)
                        hBufferOutR = cl::Buffer (context, CL_MEM_ALLOC_HOST_PTR, bufferOutSize);
                    if (hBufferOutG () == nullptr)
                        hBufferOutG = cl::Buffer (context, CL_MEM_ALLOC_HOST_PTR, bufferOutSize);
                    if (hBufferOutB () == nullptr)
                        hBufferOutB = cl::Buffer (context, CL_MEM_ALLOC_HOST_PTR, bufferOutSize);

                    hPtrOutR = (cl_float *) queue0.enqueueMapBuffer (
                        hBufferOutR, CL_FALSE, CL_MAP_READ, 0, bufferOutSize);
                    hPtrOutG = (cl_float *) queue0.enqueueMapBuffer (
                        hBufferOutG, CL_FALSE, CL_MAP_READ, 0, bufferOutSize);
                    hPtrOutB = (cl_float *) queue0.enqueueMapBuffer (
                        hBufferOutB, CL_FALSE, CL_MAP_READ, 0, bufferOutSize);
                    queue0.enqueueUnmapMemObject (hBufferOutR, hPtrOutR);
                    queue0.enqueueUnmapMemObject (hBufferOutG, hPtrOutG);
                    queue0.enqueueUnmapMemObject (hBufferOutB, hPtrOutB);
                    queue0.finish ();

                    if (!io) hPtrIn = nullptr;
                    break;
            }
            
            // Create device buffers
            if (dBufferIn () == nullptr)
                dBufferIn = cl::Buffer (context, CL_MEM_READ_ONLY, bufferInSize);
            if (dBufferOutR () == nullptr)
                dBufferOutR = cl::Buffer (context, CL_MEM_WRITE_ONLY, bufferOutSize);
            if (dBufferOutG () == nullptr)
                dBufferOutG = cl::Buffer (context, CL_MEM_WRITE_ONLY, bufferOutSize);
            if (dBufferOutB () == nullptr)
                dBufferOutB = cl::Buffer (context, CL_MEM_WRITE_ONLY, bufferOutSize);

            sRGB.get (SeparateRGB<SeparateRGBConfig::UCHAR_FLOAT>::Memory::D_IN) = dBufferIn;
            sRGB.get (SeparateRGB<SeparateRGBConfig::UCHAR_FLOAT>::Memory::D_OUT_R) = 
                cl::Buffer (context, CL_MEM_READ_WRITE, bufferOutSize);
            sRGB.get (SeparateRGB<SeparateRGBConfig::UCHAR_FLOAT>::Memory::D_OUT_G) = 
                cl::Buffer (context, CL_MEM_READ_WRITE, bufferOutSize);
            sRGB.get (SeparateRGB<SeparateRGBConfig::UCHAR_FLOAT>::Memory::D_OUT_B) = 
                cl::Buffer (context, CL_MEM_READ_WRITE, bufferOutSize);
            sRGB.init (width, height, Staging::NONE);

            gfR.get (GuidedFilter<GuidedFilterConfig::I_EQ_P>::Memory::D_IN) = 
                sRGB.get (SeparateRGB<SeparateRGBConfig::UCHAR_FLOAT>::Memory::D_OUT_R);
            gfR.get (GuidedFilter<GuidedFilterConfig::I_EQ_P>::Memory::D_OUT) = dBufferOutR;
            gfR.init (width, height, radius, eps, 0, 1e-4f, 1.f, Staging::NONE);

            gfG.get (GuidedFilter<GuidedFilterConfig::I_EQ_P>::Memory::D_IN) = 
                sRGB.get (SeparateRGB<SeparateRGBConfig::UCHAR_FLOAT>::Memory::D_OUT_G);
            gfG.get (GuidedFilter<GuidedFilterConfig::I_EQ_P>::Memory::D_OUT) = dBufferOutG;
            gfG.init (width, height, radius, eps, 0, 1e-4f, 1.f, Staging::NONE);

            gfB.get (GuidedFilter<GuidedFilterConfig::I_EQ_P>::Memory::D_IN) = 
                sRGB.get (SeparateRGB<SeparateRGBConfig::UCHAR_FLOAT>::Memory::D_OUT_B);
            gfB.get (GuidedFilter<GuidedFilterConfig::I_EQ_P>::Memory::D_OUT) = dBufferOutB;
            gfB.init (width, height, radius, eps, 0, 1e-4f, 1.f, Staging::NONE);
        }


        /*! \details The transfer happens from a staging buffer on the host to the 
         *           associated (specified) device buffer.
         *  \note The transfer is handled by the first command queue.
         *  
         *  \param[in] mem enumeration value specifying an input device buffer.
         *  \param[in] ptr a pointer to an array holding input data. If not NULL, the 
         *                 data from `ptr` will be copied to the associated staging buffer.
         *  \param[in] block a flag to indicate whether to perform a blocking 
         *                   or a non-blocking operation.
         *  \param[in] events a wait-list of events.
         *  \param[out] event event associated with the write operation to the device buffer.
         */
        void GuidedFilterRGB<GuidedFilterRGBConfig::SEPARATED>::write (
            GuidedFilterRGB::Memory mem, void *ptr, bool block, const std::vector<cl::Event> *events, cl::Event *event)
        {
            if (staging == Staging::I || staging == Staging::IO)
            {
                switch (mem)
                {
                    case GuidedFilterRGB::Memory::D_IN:
                        if (ptr != nullptr)
                            std::copy ((cl_uchar *) ptr, (cl_uchar *) ptr + 3 * width * height, hPtrIn);
                        queue0.enqueueWriteBuffer (dBufferIn, block, 0, bufferInSize, hPtrIn, events, event);
                        break;
                    default:
                        break;
                }
            }
        }


        /*! \details The transfer happens from a device buffer to the associated 
         *           (specified) staging buffer on the host.
         *  \note The transfer is handled by the first command queue.
         *  
         *  \param[in] mem enumeration value specifying an output staging buffer.
         *  \param[in] block a flag to indicate whether to perform a blocking 
         *                   or a non-blocking operation.
         *  \param[in] events a wait-list of events.
         *  \param[out] event event associated with the read operation to the staging buffer.
         */
        void* GuidedFilterRGB<GuidedFilterRGBConfig::SEPARATED>::read (
            GuidedFilterRGB::Memory mem, bool block, const std::vector<cl::Event> *events, cl::Event *event)
        {
            if (staging == Staging::O || staging == Staging::IO)
            {
                switch (mem)
                {
                    case GuidedFilterRGB::Memory::H_OUT_R:
                        queue0.enqueueReadBuffer (dBufferOutR, block, 0, bufferOutSize, hPtrOutR, events, event);
                        return hPtrOutR;
                    case GuidedFilterRGB::Memory::H_OUT_G:
                        queue0.enqueueReadBuffer (dBufferOutG, block, 0, bufferOutSize, hPtrOutG, events, event);
                        return hPtrOutG;
                    case GuidedFilterRGB::Memory::H_OUT_B:
                        queue0.enqueueReadBuffer (dBufferOutB, block, 0, bufferOutSize, hPtrOutB, events, event);
                        return hPtrOutB;
                    default:
                        return nullptr;
                }
            }
            return nullptr;
        }


        /*! \details The function call is non-blocking.
         *
         *  \param[in] events a wait-list of events.
         *  \param[out] event event associated with the kernel execution.
         */
        void GuidedFilterRGB<GuidedFilterRGBConfig::SEPARATED>::run (
            const std::vector<cl::Event> *events, cl::Event *event)
        {
            sRGB.run (events, &sEvent); waitList[0] = sEvent;
            gfR.run (&waitList);
            gfG.run ();
            gfB.run (nullptr, event);
        }


        /*! \return The radius of the square filter window.
         */
        int GuidedFilterRGB<GuidedFilterRGBConfig::SEPARATED>::getRadius ()
        {
            return radius;
        }


        /*! \details Updates the kernel argument for the filter window radius.
         *
         *  \param[in] _radius radius of the square filter window.
         */
        void GuidedFilterRGB<GuidedFilterRGBConfig::SEPARATED>::setRadius (int _radius)
        {
            radius = _radius;
            gfR.setRadius (radius);
            gfG.setRadius (radius);
            gfB.setRadius (radius);
        }


        /*! \return The regularization parameter \f$\epsilon\f$.
         */
        float GuidedFilterRGB<GuidedFilterRGBConfig::SEPARATED>::getEps ()
        {
            return eps;
        }


        /*! \details Updates the kernel argument for the regularization parameter \f$\epsilon\f$.
         *
         *  \param[in] _eps regularization parameter \f$\epsilon\f$.
         */
        void GuidedFilterRGB<GuidedFilterRGBConfig::SEPARATED>::setEps (float _eps)
        {
            eps = _eps;
            gfR.setEps (eps);
            gfG.setEps (eps);
            gfB.setEps (eps);
        }


        /*! \param[in] _env opencl environment.
         *  \param[in] _info opencl configuration. Specifies the context, queue, etc, to be used.
         *                   The class requires **two** `(2)` **command queues** (on the same device).
         */
        GuidedFilterRGB<GuidedFilterRGBConfig::INTERLEAVED_FLOAT>::GuidedFilterRGB (
            clutils::CLEnv &_env, clutils::CLEnvInfo<2> _info) : 
            env (_env), info (_info), 
            context (env.getContext (info.pIdx)), 
            queue0 (env.getQueue (info.ctxIdx, info.qIdx[0])), 
            sRGB  (env, info.getCLEnvInfo (0)), 
            gfR (env, info), gfG (env, info), gfB (env, info), 
            cRGB  (env, info.getCLEnvInfo (0)), 
            waitList (1)
        {
        }


        /*! \details This interface exists to allow CL memory sharing between different kernels.
         *
         *  \param[in] mem enumeration value specifying the requested memory object.
         *  \return A reference to the requested memory object.
         */
        cl::Memory& GuidedFilterRGB<GuidedFilterRGBConfig::INTERLEAVED_FLOAT>::get (GuidedFilterRGB::Memory mem)
        {
            switch (mem)
            {
                case GuidedFilterRGB::Memory::H_IN:
                    return hBufferIn;
                case GuidedFilterRGB::Memory::H_OUT:
                    return hBufferOut;
                case GuidedFilterRGB::Memory::D_INTR_R:
                    return sRGB.get (SeparateRGB<SeparateRGBConfig::UCHAR_FLOAT>::Memory::D_OUT_R);
                case GuidedFilterRGB::Memory::D_INTR_G:
                    return sRGB.get (SeparateRGB<SeparateRGBConfig::UCHAR_FLOAT>::Memory::D_OUT_G);
                case GuidedFilterRGB::Memory::D_INTR_B:
                    return sRGB.get (SeparateRGB<SeparateRGBConfig::UCHAR_FLOAT>::Memory::D_OUT_B);
                case GuidedFilterRGB::Memory::D_IN:
                    return dBufferIn;
                case GuidedFilterRGB::Memory::D_OUT:
                    return dBufferOut;
            }
        }


        /*! \details Sets up memory objects as necessary, and defines the kernel workspaces.
         *  \note If you have assigned a memory object to one member variable of the class 
         *        before the call to `init`, then that memory will be maintained. Otherwise, 
         *        a new memory object will be created.
         *        
         *  \param[in] _width width of the input array to be processed.
         *  \param[in] _height height of the input array to be processed.
         *  \param[in] _radius radius of the square filter window, i.e. \f$\ radius=filter\_width/2-1\f$.
         *  \param[in] _eps regularization parameter \f$ \epsilon \f$.
         *  \param[in] _staging flag to indicate whether or not to instantiate the staging buffers.
         */
        void GuidedFilterRGB<GuidedFilterRGBConfig::INTERLEAVED_FLOAT>::init (
            unsigned int _width, unsigned int _height, int _radius, float _eps, Staging _staging)
        {
            width = _width; height = _height; radius = _radius; eps = _eps;
            bufferInSize = 3 * width * height * sizeof (cl_uchar);
            bufferOutSize = 3 * width * height * sizeof (cl_float);
            staging = _staging;

            // Create staging buffers
            bool io = false;
            switch (staging)
            {
                case Staging::NONE:
                    hPtrIn = nullptr;
                    hPtrOut = nullptr;
                    break;

                case Staging::IO:
                    io = true;

                case Staging::I:
                    if (hBufferIn () == nullptr)
                        hBufferIn = cl::Buffer (context, CL_MEM_ALLOC_HOST_PTR, bufferInSize);

                    hPtrIn = (cl_uchar *) queue0.enqueueMapBuffer (
                        hBufferIn, CL_FALSE, CL_MAP_WRITE, 0, bufferInSize);
                    queue0.enqueueUnmapMemObject (hBufferIn, hPtrIn);

                    if (!io)
                    {
                        queue0.finish ();
                        hPtrOut = nullptr;
                        break;
                    }

                case Staging::O:
                    if (hBufferOut () == nullptr)
                        hBufferOut = cl::Buffer (context, CL_MEM_ALLOC_HOST_PTR, bufferOutSize);

                    hPtrOut = (cl_float *) queue0.enqueueMapBuffer (
                        hBufferOut, CL_FALSE, CL_MAP_READ, 0, bufferOutSize);
                    queue0.enqueueUnmapMemObject (hBufferOut, hPtrOut);
                    queue0.finish ();

                    if (!io) hPtrIn = nullptr;
                    break;
            }
            
            // Create device buffers
            if (dBufferIn () == nullptr)
                dBufferIn = cl::Buffer (context, CL_MEM_READ_ONLY, bufferInSize);
            if (dBufferOut () == nullptr)
                dBufferOut = cl::Buffer (context, CL_MEM_WRITE_ONLY, bufferOutSize);

            sRGB.get (SeparateRGB<SeparateRGBConfig::UCHAR_FLOAT>::Memory::D_IN) = dBufferIn;
            sRGB.get (SeparateRGB<SeparateRGBConfig::UCHAR_FLOAT>::Memory::D_OUT_R) = 
                cl::Buffer (context, CL_MEM_READ_WRITE, bufferOutSize / 3);
            sRGB.get (SeparateRGB<SeparateRGBConfig::UCHAR_FLOAT>::Memory::D_OUT_G) = 
                cl::Buffer (context, CL_MEM_READ_WRITE, bufferOutSize / 3);
            sRGB.get (SeparateRGB<SeparateRGBConfig::UCHAR_FLOAT>::Memory::D_OUT_B) = 
                cl::Buffer (context, CL_MEM_READ_WRITE, bufferOutSize / 3);
            sRGB.init (width, height, Staging::NONE);

            gfR.get (GuidedFilter<GuidedFilterConfig::I_EQ_P>::Memory::D_IN) = 
                sRGB.get (SeparateRGB<SeparateRGBConfig::UCHAR_FLOAT>::Memory::D_OUT_R);
            gfR.get (GuidedFilter<GuidedFilterConfig::I_EQ_P>::Memory::D_OUT) = 
                cl::Buffer (context, CL_MEM_READ_WRITE, bufferOutSize / 3);
            gfR.init (width, height, radius, eps, 0, 1e-4f, 1.f, Staging::NONE);

            gfG.get (GuidedFilter<GuidedFilterConfig::I_EQ_P>::Memory::D_IN) = 
                sRGB.get (SeparateRGB<SeparateRGBConfig::UCHAR_FLOAT>::Memory::D_OUT_G);
            gfG.get (GuidedFilter<GuidedFilterConfig::I_EQ_P>::Memory::D_OUT) = 
                cl::Buffer (context, CL_MEM_READ_WRITE, bufferOutSize / 3);
            gfG.init (width, height, radius, eps, 0, 1e-4f, 1.f, Staging::NONE);

            gfB.get (GuidedFilter<GuidedFilterConfig::I_EQ_P>::Memory::D_IN) = 
                sRGB.get (SeparateRGB<SeparateRGBConfig::UCHAR_FLOAT>::Memory::D_OUT_B);
            gfB.get (GuidedFilter<GuidedFilterConfig::I_EQ_P>::Memory::D_OUT) = 
                cl::Buffer (context, CL_MEM_READ_WRITE, bufferOutSize / 3);
            gfB.init (width, height, radius, eps, 0, 1e-4f, 1.f, Staging::NONE);

            cRGB.get (CombineRGB<CombineRGBConfig::FLOAT_FLOAT>::Memory::D_IN_R) = 
                gfR.get (GuidedFilter<GuidedFilterConfig::I_EQ_P>::Memory::D_OUT);
            cRGB.get (CombineRGB<CombineRGBConfig::FLOAT_FLOAT>::Memory::D_IN_G) = 
                gfG.get (GuidedFilter<GuidedFilterConfig::I_EQ_P>::Memory::D_OUT);
            cRGB.get (CombineRGB<CombineRGBConfig::FLOAT_FLOAT>::Memory::D_IN_B) = 
                gfB.get (GuidedFilter<GuidedFilterConfig::I_EQ_P>::Memory::D_OUT);
            cRGB.get (CombineRGB<CombineRGBConfig::FLOAT_FLOAT>::Memory::D_OUT) = dBufferOut;
            cRGB.init (width, height, Staging::NONE);
        }


        /*! \details The transfer happens from a staging buffer on the host to the 
         *           associated (specified) device buffer.
         *  \note The transfer is handled by the first command queue.
         *  
         *  \param[in] mem enumeration value specifying an input device buffer.
         *  \param[in] ptr a pointer to an array holding input data. If not NULL, the 
         *                 data from `ptr` will be copied to the associated staging buffer.
         *  \param[in] block a flag to indicate whether to perform a blocking 
         *                   or a non-blocking operation.
         *  \param[in] events a wait-list of events.
         *  \param[out] event event associated with the write operation to the device buffer.
         */
        void GuidedFilterRGB<GuidedFilterRGBConfig::INTERLEAVED_FLOAT>::write (
            GuidedFilterRGB::Memory mem, void *ptr, bool block, const std::vector<cl::Event> *events, cl::Event *event)
        {
            if (staging == Staging::I || staging == Staging::IO)
            {
                switch (mem)
                {
                    case GuidedFilterRGB::Memory::D_IN:
                        if (ptr != nullptr)
                            std::copy ((cl_uchar *) ptr, (cl_uchar *) ptr + 3 * width * height, hPtrIn);
                        queue0.enqueueWriteBuffer (dBufferIn, block, 0, bufferInSize, hPtrIn, events, event);
                        break;
                    default:
                        break;
                }
            }
        }


        /*! \details The transfer happens from a device buffer to the associated 
         *           (specified) staging buffer on the host.
         *  \note The transfer is handled by the first command queue.
         *  
         *  \param[in] mem enumeration value specifying an output staging buffer.
         *  \param[in] block a flag to indicate whether to perform a blocking 
         *                   or a non-blocking operation.
         *  \param[in] events a wait-list of events.
         *  \param[out] event event associated with the read operation to the staging buffer.
         */
        void* GuidedFilterRGB<GuidedFilterRGBConfig::INTERLEAVED_FLOAT>::read (
            GuidedFilterRGB::Memory mem, bool block, const std::vector<cl::Event> *events, cl::Event *event)
        {
            if (staging == Staging::O || staging == Staging::IO)
            {
                switch (mem)
                {
                    case GuidedFilterRGB::Memory::H_OUT:
                        queue0.enqueueReadBuffer (dBufferOut, block, 0, bufferOutSize, hPtrOut, events, event);
                        return hPtrOut;
                    default:
                        return nullptr;
                }
            }
            return nullptr;
        }


        /*! \details The function call is non-blocking.
         *
         *  \param[in] events a wait-list of events.
         *  \param[out] event event associated with the kernel execution.
         */
        void GuidedFilterRGB<GuidedFilterRGBConfig::INTERLEAVED_FLOAT>::run (
            const std::vector<cl::Event> *events, cl::Event *event)
        {
            sRGB.run (events, &sEvent); waitList[0] = sEvent;
            gfR.run (&waitList);
            gfG.run ();
            gfB.run ();
            cRGB.run (nullptr, event);
        }


        /*! \return The radius of the square filter window.
         */
        int GuidedFilterRGB<GuidedFilterRGBConfig::INTERLEAVED_FLOAT>::getRadius ()
        {
            return radius;
        }


        /*! \details Updates the kernel argument for the filter window radius.
         *
         *  \param[in] _radius radius of the square filter window.
         */
        void GuidedFilterRGB<GuidedFilterRGBConfig::INTERLEAVED_FLOAT>::setRadius (int _radius)
        {
            radius = _radius;
            gfR.setRadius (radius);
            gfG.setRadius (radius);
            gfB.setRadius (radius);
        }


        /*! \return The regularization parameter \f$\epsilon\f$.
         */
        float GuidedFilterRGB<GuidedFilterRGBConfig::INTERLEAVED_FLOAT>::getEps ()
        {
            return eps;
        }


        /*! \details Updates the kernel argument for the regularization parameter \f$\epsilon\f$.
         *
         *  \param[in] _eps regularization parameter \f$\epsilon\f$.
         */
        void GuidedFilterRGB<GuidedFilterRGBConfig::INTERLEAVED_FLOAT>::setEps (float _eps)
        {
            eps = _eps;
            gfR.setEps (eps);
            gfG.setEps (eps);
            gfB.setEps (eps);
        }


        /*! \param[in] _env opencl environment.
         *  \param[in] _info opencl configuration. Specifies the context, queue, etc, to be used.
         *                   The class requires **two** `(2)` **command queues** (on the same device).
         */
        GuidedFilterDepth::GuidedFilterDepth (
            clutils::CLEnv &_env, clutils::CLEnvInfo<2> _info) : 
            env (_env), info (_info), 
            context (env.getContext (info.pIdx)), 
            queue0 (env.getQueue (info.ctxIdx, info.qIdx[0])), 
            depth  (env, info.getCLEnvInfo (0)), gf (env, info), 
            waitList (1)
        {
        }


        /*! \details This interface exists to allow CL memory sharing between different kernels.
         *
         *  \param[in] mem enumeration value specifying the requested memory object.
         *  \return A reference to the requested memory object.
         */
        cl::Memory& GuidedFilterDepth::get (GuidedFilterDepth::Memory mem)
        {
            switch (mem)
            {
                case GuidedFilterDepth::Memory::H_IN:
                    return hBufferIn;
                case GuidedFilterDepth::Memory::H_OUT:
                    return hBufferOut;
                case GuidedFilterDepth::Memory::D_IN:
                    return dBufferIn;
                case GuidedFilterDepth::Memory::D_OUT:
                    return dBufferOut;
            }
        }


        /*! \details Sets up memory objects as necessary, and defines the kernel workspaces.
         *  \note If you have assigned a memory object to one member variable of the class 
         *        before the call to `init`, then that memory will be maintained. Otherwise, 
         *        a new memory object will be created.
         *        
         *  \param[in] _width width of the input array to be processed.
         *  \param[in] _height height of the input array to be processed.
         *  \param[in] _radius radius of the square filter window, i.e. \f$\ radius=filter\_width/2-1\f$.
         *  \param[in] _eps regularization parameter \f$ \epsilon \f$.
         *  \param[in] _dScaling factor by which to scale the depth values. It's independent from the 
         *                       scaling applied in `BoxFilterSAT`. This is another level of scaling 
         *                       (internal to the algorithm) applied to the input array before the 
         *                       processing by `GuidedFilter`.
         *  \param[in] _staging flag to indicate whether or not to instantiate the staging buffers.
         */
        void GuidedFilterDepth::init (
            unsigned int _width, unsigned int _height, int _radius, float _eps, float _dScaling, Staging _staging)
        {
            width = _width; height = _height; radius = _radius; eps = _eps;
            bufferInSize = width * height * sizeof (cl_ushort);
            bufferOutSize = width * height * sizeof (cl_float);
            dScaling = _dScaling;
            staging = _staging;

            // Create staging buffers
            bool io = false;
            switch (staging)
            {
                case Staging::NONE:
                    hPtrIn = nullptr;
                    hPtrOut = nullptr;
                    break;

                case Staging::IO:
                    io = true;

                case Staging::I:
                    if (hBufferIn () == nullptr)
                        hBufferIn = cl::Buffer (context, CL_MEM_ALLOC_HOST_PTR, bufferInSize);

                    hPtrIn = (cl_ushort *) queue0.enqueueMapBuffer (
                        hBufferIn, CL_FALSE, CL_MAP_WRITE, 0, bufferInSize);
                    queue0.enqueueUnmapMemObject (hBufferIn, hPtrIn);

                    if (!io)
                    {
                        queue0.finish ();
                        hPtrOut = nullptr;
                        break;
                    }

                case Staging::O:
                    if (hBufferOut () == nullptr)
                        hBufferOut = cl::Buffer (context, CL_MEM_ALLOC_HOST_PTR, bufferOutSize);

                    hPtrOut = (cl_float *) queue0.enqueueMapBuffer (
                        hBufferOut, CL_FALSE, CL_MAP_READ, 0, bufferOutSize);
                    queue0.enqueueUnmapMemObject (hBufferOut, hPtrOut);
                    queue0.finish ();

                    if (!io) hPtrIn = nullptr;
                    break;
            }
            
            // Create device buffers
            if (dBufferIn () == nullptr)
                dBufferIn = cl::Buffer (context, CL_MEM_READ_ONLY, bufferInSize);
            if (dBufferOut () == nullptr)
                dBufferOut = cl::Buffer (context, CL_MEM_WRITE_ONLY, bufferOutSize);

            depth.get (Depth<DepthConfig::USHORT_FLOAT>::Memory::D_IN) = dBufferIn;
            depth.get (Depth<DepthConfig::USHORT_FLOAT>::Memory::D_OUT) = 
                cl::Buffer (context, CL_MEM_READ_WRITE, bufferOutSize);
            depth.init (width, height, dScaling, Staging::NONE);

            gf.get (GuidedFilter<GuidedFilterConfig::I_EQ_P>::Memory::D_IN) = 
                depth.get (Depth<DepthConfig::USHORT_FLOAT>::Memory::D_OUT);
            gf.get (GuidedFilter<GuidedFilterConfig::I_EQ_P>::Memory::D_OUT) = dBufferOut;
            gf.init (width, height, radius, eps, 1, 1e-6f, 1.f / dScaling, Staging::NONE);
        }


        /*! \details The transfer happens from a staging buffer on the host to the 
         *           associated (specified) device buffer.
         *  \note The transfer is handled by the first command queue.
         *  
         *  \param[in] mem enumeration value specifying an input device buffer.
         *  \param[in] ptr a pointer to an array holding input data. If not NULL, the 
         *                 data from `ptr` will be copied to the associated staging buffer.
         *  \param[in] block a flag to indicate whether to perform a blocking 
         *                   or a non-blocking operation.
         *  \param[in] events a wait-list of events.
         *  \param[out] event event associated with the write operation to the device buffer.
         */
        void GuidedFilterDepth::write (GuidedFilterDepth::Memory mem, void *ptr, bool block, 
            const std::vector<cl::Event> *events, cl::Event *event)
        {
            if (staging == Staging::I || staging == Staging::IO)
            {
                switch (mem)
                {
                    case GuidedFilterDepth::Memory::D_IN:
                        if (ptr != nullptr)
                            std::copy ((cl_ushort *) ptr, (cl_ushort *) ptr + width * height, hPtrIn);
                        queue0.enqueueWriteBuffer (dBufferIn, block, 0, bufferInSize, hPtrIn, events, event);
                        break;
                    default:
                        break;
                }
            }
        }


        /*! \details The transfer happens from a device buffer to the associated 
         *           (specified) staging buffer on the host.
         *  \note The transfer is handled by the first command queue.
         *  
         *  \param[in] mem enumeration value specifying an output staging buffer.
         *  \param[in] block a flag to indicate whether to perform a blocking 
         *                   or a non-blocking operation.
         *  \param[in] events a wait-list of events.
         *  \param[out] event event associated with the read operation to the staging buffer.
         */
        void* GuidedFilterDepth::read (GuidedFilterDepth::Memory mem, bool block, 
            const std::vector<cl::Event> *events, cl::Event *event)
        {
            if (staging == Staging::O || staging == Staging::IO)
            {
                switch (mem)
                {
                    case GuidedFilterDepth::Memory::H_OUT:
                        queue0.enqueueReadBuffer (dBufferOut, block, 0, bufferOutSize, hPtrOut, events, event);
                        return hPtrOut;
                    default:
                        return nullptr;
                }
            }
            return nullptr;
        }


        /*! \details The function call is non-blocking.
         *
         *  \param[in] events a wait-list of events.
         *  \param[out] event event associated with the kernel execution.
         */
        void GuidedFilterDepth::run (const std::vector<cl::Event> *events, cl::Event *event)
        {
            depth.run (events, &dEvent); waitList[0] = dEvent;
            gf.run (&waitList, event);
        }


        /*! \return The radius of the square filter window.
         */
        int GuidedFilterDepth::getRadius ()
        {
            return radius;
        }


        /*! \details Updates the kernel argument for the filter window radius.
         *
         *  \param[in] _radius radius of the square filter window.
         */
        void GuidedFilterDepth::setRadius (int _radius)
        {
            radius = _radius;
            gf.setRadius (radius);
        }


        /*! \return The regularization parameter \f$\epsilon\f$.
         */
        float GuidedFilterDepth::getEps ()
        {
            return eps;
        }


        /*! \details Updates the kernel argument for the regularization parameter \f$\epsilon\f$.
         *
         *  \param[in] _eps regularization parameter \f$\epsilon\f$.
         */
        void GuidedFilterDepth::setEps (float _eps)
        {
            eps = _eps;
            gf.setEps (eps);
        }


        /*! \return The depth scaling factor.
         */
        float GuidedFilterDepth::getDScaling ()
        {
            return dScaling;
        }


        /*! \details Updates the kernel argument for the depth scaling factor.
         *
         *  \param[in] _dScaling  depth scaling factor.
         */
        void GuidedFilterDepth::setDScaling (float _dScaling)
        {
            dScaling = _dScaling;
            depth.setScaling (dScaling);
            gf.setOutputScaling (1.f / dScaling);
        }

    }

}
}
