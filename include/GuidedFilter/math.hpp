/*! \file math.hpp
 *  \brief Declares classes that organize the execution of OpenCL math kernels.
 *  \details Each class hides the details of kernel execution. They
 *           initialize the necessary buffers, set up the workspaces, 
 *           and run the kernels.
 *  \author Nick Lamprianidis
 *  \version 1.1.2
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

#ifndef GF_MATH_HPP
#define GF_MATH_HPP

#include <CLUtils.hpp>
#include <GuidedFilter/common.hpp>


/*! \brief Offers classes which set up kernel execution parameters and 
 *         provide interfaces for the handling of memory objects.
 */
namespace cl_algo
{
namespace GF
{
/*! \brief Offers classes associated with mathematics operations.
 */
namespace Math
{

    /*! \brief Interface class for the `mult` kernel.
     *  \details `mult` multiplies two input arrays together, element-wise. 
     *           For more details, look at the kernel's documentation.
     *  \note The `mult` kernel is available in `kernels/math_kernels.cl`.
     *  \note The class creates its own buffers. If you would like to provide 
     *        your own buffers, call `get` to get references to the placeholders 
     *        within the class and assign them to your buffers. You will have to 
     *        do this strictly before the call to `init`. You can also call `get` 
     *        (after the call to `init`) to get a reference to a buffer within 
     *        the class and assign it to another kernel class instance further 
     *        down in your task pipeline.
     *  
     *        The following input/output `OpenCL` memory objects are created by a `Mult` instance:<br>
     *        |  Name  | Type | Placement | I/O | Use | Properties | Size |
     *        |  ---   |:---: |   :---:   |:---:|:---:|   :---:    |:---: |
     *        | H_IN_A | Buffer | Host   | I | Staging     | CL_MEM_READ_WRITE | \f$3*width*height*sizeof\ (cl\_float)\f$ |
     *        | H_IN_B | Buffer | Host   | I | Staging     | CL_MEM_READ_WRITE | \f$3*width*height*sizeof\ (cl\_float)\f$ |
     *        | H_OUT  | Buffer | Host   | O | Staging     | CL_MEM_READ_WRITE | \f$3*width*height*sizeof\ (cl\_float)\f$ |
     *        | D_IN_A | Buffer | Device | I | Processing  | CL_MEM_READ_ONLY  | \f$3*width*height*sizeof\ (cl\_float)\f$ |
     *        | D_IN_B | Buffer | Device | I | Processing  | CL_MEM_READ_ONLY  | \f$3*width*height*sizeof\ (cl\_float)\f$ |
     *        | D_OUT  | Buffer | Device | O | Processing  | CL_MEM_WRITE_ONLY | \f$3*width*height*sizeof\ (cl\_float)\f$ |
     */
    class Mult
    {
    public:
        /*! \brief Enumerates the memory objects handled by the class.
         *  \note `H_*` names refer to staging buffers on the host.
         *  \note `D_*` names refer to buffers on the device.
         */
        enum class Memory : uint8_t
        {
            H_IN_A,  /*!< Input staging buffer for first operand.*/
            H_IN_B,  /*!< Input staging buffer for second operand.*/
            H_OUT,   /*!< Output staging buffer.*/
            D_IN_A,  /*!< Input buffer for first operand.*/
            D_IN_B,  /*!< Input buffer for second operand.*/
            D_OUT    /*!< Output buffer.*/
        };

        /*! \brief Configures an OpenCL environment as specified by `_info`. */
        Mult (clutils::CLEnv &_env, clutils::CLEnvInfo<1> _info);
        /*! \brief Returns a reference to an internal memory object used by `mult`. */
        cl::Memory& get (Mult::Memory mem);
        /*! \brief Configures kernel execution parameters. */
        void init (unsigned int _width, unsigned int _height, Staging _staging = Staging::IO);
        /*! \brief Performs a data transfer to a device buffer. */
        void write (Mult::Memory mem = Mult::Memory::D_IN_A, void *ptr = nullptr, bool block = CL_FALSE, 
                    const std::vector<cl::Event> *events = nullptr, cl::Event *event = nullptr);
        /*! \brief Performs a data transfer to a staging buffer. */
        void* read (Mult::Memory mem = Mult::Memory::H_OUT, bool block = CL_TRUE, 
                    const std::vector<cl::Event> *events = nullptr, cl::Event *event = nullptr);
        /*! \brief Executes the necessary kernels. */
        void run (const std::vector<cl::Event> *events = nullptr, cl::Event *event = nullptr);

        cl_float *hPtrInA;  /*!< Mapping of the input staging buffer. */
        cl_float *hPtrInB;  /*!< Mapping of the input staging buffer. */
        cl_float *hPtrOut;  /*!< Mapping of the output staging buffer. */

    private:
        clutils::CLEnv &env;
        clutils::CLEnvInfo<1> &info;
        cl::Context context;
        cl::CommandQueue queue;
        cl::Kernel kernel;
        cl::NDRange global;
        Staging staging;
        unsigned int length, bufferSize;
        cl::Buffer hBufferInA, hBufferInB, hBufferOut;
        cl::Buffer dBufferInA, dBufferInB, dBufferOut;

    public:
        /*! \brief Executes the necessary kernels.
         *  \details This `run` instance is used for profiling.
         *  
         *  \param[in] timer `GPUTimer` that does the profiling of the kernel executions.
         *  \param[in] events a wait-list of events.
         *  \return Τhe total execution time measured by the timer.
         */
        template <typename period>
        double run (clutils::GPUTimer<period> &timer, const std::vector<cl::Event> *events = nullptr)
        {
            queue.enqueueNDRangeKernel (kernel, cl::NullRange, global, cl::NullRange, events, &timer.event ());
            queue.flush (); timer.wait ();
            return timer.duration ();
        }

    };


    /*! \brief Interface class for the `pown_` kernel.
     *  \details `pown_` raises an array to an integer power, element-wise. 
     *           For more details, look at the kernel's documentation.
     *  \note The `pown_` kernel is available in `kernels/math_kernels.cl`.
     *  \note The class creates its own buffers. If you would like to provide 
     *        your own buffers, call `get` to get references to the placeholders 
     *        within the class and assign them to your buffers. You will have to 
     *        do this strictly before the call to `init`. You can also call `get` 
     *        (after the call to `init`) to get a reference to a buffer within 
     *        the class and assign it to another kernel class instance further 
     *        down in your task pipeline.
     *  
     *        The following input/output `OpenCL` memory objects are created by a `Pown` instance:<br>
     *        | Name | Type | Placement | I/O | Use | Properties | Size |
     *        | ---  |:---: |   :---:   |:---:|:---:|   :---:    |:---: |
     *        | H_IN | Buffer | Host   | I | Staging     | CL_MEM_READ_WRITE | \f$3*width*height*sizeof\ (cl\_float)\f$ |
     *        | H_OUT| Buffer | Host   | O | Staging     | CL_MEM_READ_WRITE | \f$3*width*height*sizeof\ (cl\_float)\f$ |
     *        | D_IN | Buffer | Device | I | Processing  | CL_MEM_READ_ONLY  | \f$3*width*height*sizeof\ (cl\_float)\f$ |
     *        | D_OUT| Buffer | Device | O | Processing  | CL_MEM_WRITE_ONLY | \f$3*width*height*sizeof\ (cl\_float)\f$ |
     */
    class Pown
    {
    public:
        /*! \brief Enumerates the memory objects handled by the class.
         *  \note `H_*` names refer to staging buffers on the host.
         *  \note `D_*` names refer to buffers on the device.
         */
        enum class Memory : uint8_t
        {
            H_IN,   /*!< Input staging buffer. */
            H_OUT,  /*!< Output staging buffer. */
            D_IN,   /*!< Input buffer. */
            D_OUT   /*!< Output buffer. */
        };

        /*! \brief Configures an OpenCL environment as specified by `_info`. */
        Pown (clutils::CLEnv &_env, clutils::CLEnvInfo<1> _info);
        /*! \brief Returns a reference to an internal memory object used by `pown_`. */
        cl::Memory& get (Pown::Memory mem);
        /*! \brief Configures kernel execution parameters. */
        void init (unsigned int _width, unsigned int _height, int _n, Staging _staging = Staging::IO);
        /*! \brief Performs a data transfer to a device buffer. */
        void write (Pown::Memory mem = Pown::Memory::D_IN, void *ptr = nullptr, bool block = CL_FALSE, 
                    const std::vector<cl::Event> *events = nullptr, cl::Event *event = nullptr);
        /*! \brief Performs a data transfer to a staging buffer. */
        void* read (Pown::Memory mem = Pown::Memory::H_OUT, bool block = CL_TRUE, 
                    const std::vector<cl::Event> *events = nullptr, cl::Event *event = nullptr);
        /*! \brief Executes the necessary kernels. */
        void run (const std::vector<cl::Event> *events = nullptr, cl::Event *event = nullptr);
        /*! \brief Gets the power. */
        int getPower ();
        /*! \brief Sets the power. */
        void setPower (int _n);

        cl_float *hPtrIn;  /*!< Mapping of the input staging buffer. */
        cl_float *hPtrOut;  /*!< Mapping of the output staging buffer. */

    private:
        clutils::CLEnv &env;
        clutils::CLEnvInfo<1> &info;
        cl::Context context;
        cl::CommandQueue queue;
        cl::Kernel kernel;
        cl::NDRange global;
        Staging staging;
        unsigned int length, bufferSize;
        int n;
        cl::Buffer hBufferIn, hBufferOut;
        cl::Buffer dBufferIn, dBufferOut;

    public:
        /*! \brief Executes the necessary kernels.
         *  \details This `run` instance is used for profiling.
         *  
         *  \param[in] timer `GPUTimer` that does the profiling of the kernel executions.
         *  \param[in] events a wait-list of events.
         *  \return Τhe total execution time measured by the timer.
         */
        template <typename period>
        double run (clutils::GPUTimer<period> &timer, const std::vector<cl::Event> *events = nullptr)
        {
            queue.enqueueNDRangeKernel (kernel, cl::NullRange, global, cl::NullRange, events, &timer.event ());
            queue.flush (); timer.wait ();
            return timer.duration ();
        }

    };

}
}
}

#endif  // GF_MATH_HPP
