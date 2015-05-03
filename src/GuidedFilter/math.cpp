/*! \file math.cpp
 *  \brief Defines classes that organize the execution of OpenCL math kernels.
 *  \details Each class hides the details of the execution of a kernel. They
 *           initialize the necessary buffers, set up the workspaces, and 
 *           run the kernels.
 *  \author Nick Lamprianidis
 *  \version 1.1.1
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
#include <GuidedFilter/math.hpp>
#include <GuidedFilter/common.hpp>


/*! \note All the classes assume there is a fully configured `clutils::CLEnv` 
 *        environment. This means, there is a known context on which they will 
 *        operate, there is a known command queue which they will use, and all 
 *        the necessary kernel code has been compiled. For more info on **CLUtils**, 
 *        you can check the [online documentation](http://clutils.paign10.me/).
 */
namespace cl_algo
{
namespace GF
{
namespace Math
{

    /*! \param[in] _env opencl environment.
     *  \param[in] _info opencl configuration. Specifies the context, queue, etc, to be used.
     */
    Mult::Mult (clutils::CLEnv &_env, clutils::CLEnvInfo<1> _info) : 
        env (_env), info (_info), 
        context (env.getContext (info.pIdx)), 
        queue (env.getQueue (info.ctxIdx, info.qIdx[0])), 
        kernel (env.getProgram (info.pgIdx), "mult")
    {
    }


    /*! \details This interface exists to allow CL memory sharing between different kernels.
     *
     *  \param[in] mem enumeration value specifying the requested memory object.
     *  \return A reference to the requested memory object.
     */
    cl::Memory& Mult::get (Mult::Memory mem)
    {
        switch (mem)
        {
            case Mult::Memory::H_IN_A:
                return hBufferInA;
            case Mult::Memory::H_IN_B:
                return hBufferInB;
            case Mult::Memory::H_OUT:
                return hBufferOut;
            case Mult::Memory::D_IN_A:
                return dBufferInA;
            case Mult::Memory::D_IN_B:
                return dBufferInB;
            case Mult::Memory::D_OUT:
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
    void Mult::init (unsigned int _width, unsigned int _height, Staging _staging)
    {
        length = _width * _height;
        bufferSize = length * sizeof (cl_float);
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
            std::cerr << "Error[Mult]: " << error << std::endl;
            exit (EXIT_FAILURE);
        }

        // Set workspaces
        global = cl::NDRange (length / 4);

        bool io = false;
        switch (staging)
        {
            case Staging::NONE:
                hPtrInA = nullptr;
                hPtrInB = nullptr;
                hPtrOut = nullptr;
                break;

            case Staging::IO:
                io = true;

            case Staging::I:
                if (hBufferInA () == nullptr)
                    hBufferInA = cl::Buffer (context, CL_MEM_ALLOC_HOST_PTR, bufferSize);
                if (hBufferInB () == nullptr)
                    hBufferInB = cl::Buffer (context, CL_MEM_ALLOC_HOST_PTR, bufferSize);

                hPtrInA = (cl_float *) queue.enqueueMapBuffer (
                    hBufferInA, CL_FALSE, CL_MAP_WRITE, 0, bufferSize);
                hPtrInB = (cl_float *) queue.enqueueMapBuffer (
                    hBufferInB, CL_FALSE, CL_MAP_WRITE, 0, bufferSize);
                queue.enqueueUnmapMemObject (hBufferInA, hPtrInA);
                queue.enqueueUnmapMemObject (hBufferInB, hPtrInB);

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
                
                if (!io)
                {
                    hPtrInA = nullptr;
                    hPtrInB = nullptr;
                }
                break;
        }
        
        // Create device buffers
        if (dBufferInA () == nullptr)
            dBufferInA = cl::Buffer (context, CL_MEM_READ_ONLY, bufferSize);
        if (dBufferInB () == nullptr)
            dBufferInB = cl::Buffer (context, CL_MEM_READ_ONLY, bufferSize);
        if (dBufferOut () == nullptr)
            dBufferOut = cl::Buffer (context, CL_MEM_WRITE_ONLY, bufferSize);

        // Set kernel arguments
        kernel.setArg (0, dBufferInA);
        kernel.setArg (1, dBufferInB);
        kernel.setArg (2, dBufferOut);
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
    void Mult::write (Mult::Memory mem, void *ptr, bool block, 
                      const std::vector<cl::Event> *events, cl::Event *event)
    {
        if (staging == Staging::IO)
        {
            switch (mem)
            {
                case Mult::Memory::D_IN_A:
                    if (ptr != nullptr)
                        std::copy ((cl_float *) ptr, (cl_float *) ptr + length, hPtrInA);
                    queue.enqueueWriteBuffer (dBufferInA, block, 0, bufferSize, hPtrInA, events, event);
                    break;
                case Mult::Memory::D_IN_B:
                    if (ptr != nullptr)
                        std::copy ((cl_float *) ptr, (cl_float *) ptr + length, hPtrInB);
                    queue.enqueueWriteBuffer (dBufferInB, block, 0, bufferSize, hPtrInB, events, event);
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
    void* Mult::read (Mult::Memory mem, bool block, 
                      const std::vector<cl::Event> *events, cl::Event *event)
    {
        if (staging == Staging::IO)
        {
            switch (mem)
            {
                case Mult::Memory::H_OUT:
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
    void Mult::run (const std::vector<cl::Event> *events, cl::Event *event)
    {
        queue.enqueueNDRangeKernel (kernel, cl::NullRange, global, cl::NullRange, events, event);
    }


    /*! \param[in] _env opencl environment.
     *  \param[in] _info opencl configuration. Specifies the context, queue, etc, to be used.
     */
    Pown::Pown (clutils::CLEnv &_env, clutils::CLEnvInfo<1> _info) : 
        env (_env), info (_info), 
        context (env.getContext (info.pIdx)), 
        queue (env.getQueue (info.ctxIdx, info.qIdx[0])), 
        kernel (env.getProgram (info.pgIdx), "pown_")
    {
    }


    /*! \details This interface exists to allow CL memory sharing between different kernels.
     *
     *  \param[in] mem enumeration value specifying the requested memory object.
     *  \return A reference to the requested memory object.
     */
    cl::Memory& Pown::get (Pown::Memory mem)
    {
        switch (mem)
        {
            case Pown::Memory::H_IN:
                return hBufferIn;
            case Pown::Memory::H_OUT:
                return hBufferOut;
            case Pown::Memory::D_IN:
                return dBufferIn;
            case Pown::Memory::D_OUT:
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
     *  \param[in] _n power to which to raise the input array.
     *  \param[in] _staging flag to indicate whether or not to instantiate the staging buffers.
     */
    void Pown::init (unsigned int _width, unsigned int _height, int _n, Staging _staging)
    {
        length = _width * _height; n = _n;
        bufferSize = length * sizeof (cl_float);
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
            std::cerr << "Error[Pown]: " << error << std::endl;
            exit (EXIT_FAILURE);
        }

        // Set workspaces
        global = cl::NDRange (length / 4);

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
                
                if (!io)
                {
                    hPtrIn = nullptr;
                }
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
        kernel.setArg (2, n);
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
    void Pown::write (Pown::Memory mem, void *ptr, bool block, 
                      const std::vector<cl::Event> *events, cl::Event *event)
    {
        if (staging == Staging::IO)
        {
            switch (mem)
            {
                case Pown::Memory::D_IN:
                    if (ptr != nullptr)
                        std::copy ((cl_float *) ptr, (cl_float *) ptr + length, hPtrIn);
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
    void* Pown::read (Pown::Memory mem, bool block, 
                      const std::vector<cl::Event> *events, cl::Event *event)
    {
        if (staging == Staging::IO)
        {
            switch (mem)
            {
                case Pown::Memory::H_OUT:
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
    void Pown::run (const std::vector<cl::Event> *events, cl::Event *event)
    {
        queue.enqueueNDRangeKernel (kernel, cl::NullRange, global, cl::NullRange, events, event);
    }


    /*! \return The power.
     */
    int Pown::getPower ()
    {
        return n;
    }


    /*! \details Updates the kernel argument for the power.
     *
     *  \param[in] _n power.
     */
    void Pown::setPower (int _n)
    {
        n = _n;
        kernel.setArg (3, n);
    }

}
}
}
