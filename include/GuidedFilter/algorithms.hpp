/*! \file algorithms.hpp
 *  \brief Declares classes that organize the execution of OpenCL kernels.
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

#ifndef GF_ALGORITHMS_HPP
#define GF_ALGORITHMS_HPP

#include <CLUtils.hpp>
#include <GuidedFilter/common.hpp>
#include <GuidedFilter/math.hpp>


/*! \brief Offers classes which set up kernel execution parameters and 
 *         provide interfaces for the handling of memory objects.
 */
namespace cl_algo
{
/*! \brief Offers classes associated with the Guided Filter algorithm.
 */
namespace GF
{

    /*! \brief Enumerates configurations for the `SeparateRGB` class. */
    enum class SeparateRGBConfig : uint8_t
    {
        FLOAT_FLOAT,  /*!< Identifies the case of `float` input data and `float` output data. */
        UCHAR_FLOAT   /*!< Identifies the case of `uchar` input data and `float` output data. */
    };


    /*! \brief Interface class for the `separateRGBChannels` kernels.
     *  \details The `separateRGBChannels` kernels perform array transposition on an RGB image. 
     *           For more details, look at the kernels' documentation.
     *  \note The `separateRGBChannels` kernels are available in `kernels/imageSupport_kernels.cl`.
     *  \note This is just a declaration. Look at the explicit template specializations
     *        for specific instantiations of the class.
     *        
     *  \tparam C configures the class to work with different types of data.
     */
    template <SeparateRGBConfig C>
    class SeparateRGB;


    /*! \brief Interface class for the `separateRGBChannels_Float2Float` kernel.
     *  \details `separateRGBChannels_Float2Float` performs array transposition on an RGB image. 
     *           For more details, look at the kernel's documentation.
     *  \note The `separateRGBChannels_Float2Float` kernel is available 
     *        in `kernels/imageSupport_kernels.cl`.
     *  \note This is a specialization for the case of `float` input and output data.
     *  \note The class creates its own buffers. If you would like to provide 
     *        your own buffers, call `get` to get references to the placeholders 
     *        within the class and assign them to your buffers. You will have to 
     *        do this strictly before the call to `init`. You can also call `get` 
     *        (after the call to `init`) to get a reference to a buffer within 
     *        the class and assign it to another kernel class instance further 
     *        down in your task pipeline.
     *  
     *        The following input/output `OpenCL` memory objects are created by 
     *        a `SeparateRGB<SeparateRGBConfig::FLOAT_FLOAT>` instance:<br>
     *        |  Name  | Type | Placement | I/O | Use | Properties | Size |
     *        |  ---   |:---: |   :---:   |:---:|:---:|   :---:    |:---: |
     *        | H_IN   | Buffer | Host   | I | Staging     | CL_MEM_READ_WRITE | \f$3*width*height*sizeof\ (cl\_float)\f$ |
     *        | H_OUT_R| Buffer | Host   | O | Staging     | CL_MEM_READ_WRITE | \f$  width*height*sizeof\ (cl\_float)\f$ |
     *        | H_OUT_G| Buffer | Host   | O | Staging     | CL_MEM_READ_WRITE | \f$  width*height*sizeof\ (cl\_float)\f$ |
     *        | H_OUT_B| Buffer | Host   | O | Staging     | CL_MEM_READ_WRITE | \f$  width*height*sizeof\ (cl\_float)\f$ |
     *        | D_IN   | Buffer | Device | I | Processing  | CL_MEM_READ_ONLY  | \f$3*width*height*sizeof\ (cl\_float)\f$ |
     *        | D_OUT_R| Buffer | Device | O | Processing  | CL_MEM_WRITE_ONLY | \f$  width*height*sizeof\ (cl\_float)\f$ |
     *        | D_OUT_G| Buffer | Device | O | Processing  | CL_MEM_WRITE_ONLY | \f$  width*height*sizeof\ (cl\_float)\f$ |
     *        | D_OUT_B| Buffer | Device | O | Processing  | CL_MEM_WRITE_ONLY | \f$  width*height*sizeof\ (cl\_float)\f$ |
     */
    template <>
    class SeparateRGB<SeparateRGBConfig::FLOAT_FLOAT>
    {
    public:
        /*! \brief Enumerates the memory objects handled by the class.
         *  \note `H_*` names refer to staging buffers on the host.
         *  \note `D_*` names refer to buffers on the device.
         */
        enum class Memory : uint8_t
        {
            H_IN,     /*!< Input staging buffer. */
            H_OUT_R,  /*!< Output staging buffer for channel R. */
            H_OUT_G,  /*!< Output staging buffer for channel G. */
            H_OUT_B,  /*!< Output staging buffer for channel B. */
            D_IN,     /*!< Input buffer. */
            D_OUT_R,  /*!< Output buffer for channel R. */
            D_OUT_G,  /*!< Output buffer for channel G. */
            D_OUT_B   /*!< Output buffer for channel B. */
        };

        /*! \brief Configures an OpenCL environment as specified by `_info`. */
        SeparateRGB (clutils::CLEnv &_env, clutils::CLEnvInfo<1> _info);
        /*! \brief Returns a reference to an internal memory object. */
        cl::Memory& get (SeparateRGB::Memory mem);
        /*! \brief Configures kernel execution parameters. */
        void init (unsigned int _width, unsigned int _height, Staging _staging = Staging::IO);
        /*! \brief Performs a data transfer to a device buffer. */
        void write (SeparateRGB::Memory mem = SeparateRGB::Memory::D_IN, void *ptr = nullptr, bool block = CL_FALSE, 
                    const std::vector<cl::Event> *events = nullptr, cl::Event *event = nullptr);
        /*! \brief Performs a data transfer to a staging buffer. */
        void* read (SeparateRGB::Memory mem = SeparateRGB::Memory::H_OUT_R, bool block = CL_TRUE, 
                    const std::vector<cl::Event> *events = nullptr, cl::Event *event = nullptr);
        /*! \brief Executes the necessary kernels. */
        void run (const std::vector<cl::Event> *events = nullptr, cl::Event *event = nullptr);

        cl_float *hPtrIn;  /*!< Mapping of the input staging buffer. */
        cl_float *hPtrOutR;  /*!< Mapping of the output staging buffer for channel R. */
        cl_float *hPtrOutG;  /*!< Mapping of the output staging buffer for channel G. */
        cl_float *hPtrOutB;  /*!< Mapping of the output staging buffer for channel B. */

    private:
        clutils::CLEnv &env;
        clutils::CLEnvInfo<1> info;
        cl::Context context;
        cl::CommandQueue queue;
        cl::Kernel kernel;
        cl::NDRange global, local;
        Staging staging;
        size_t wgMultiple;
        unsigned int width, height;
        unsigned int bufferInSize, bufferOutSize;
        cl::Buffer hBufferIn, hBufferOutR, hBufferOutG, hBufferOutB;
        cl::Buffer dBufferIn, dBufferOutR, dBufferOutG, dBufferOutB;

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
            queue.enqueueNDRangeKernel (kernel, cl::NullRange, global, local, events, &timer.event ());
            queue.flush (); timer.wait ();

            return timer.duration ();
        }

    };


    /*! \brief Interface class for the `separateRGBChannels_Uchar2Float` kernel.
     *  \details `separateRGBChannels_Uchar2Float` performs array transposition on an RGB image,
     *           while promoting its `uchar` type to `float` and normalizing its values to one.
     *           For more details, look at the kernel's documentation.
     *  \note The `separateRGBChannels_Uchar2Float` kernel is available in `kernels/imageSupport_kernels.cl`.
     *  \note This is a specialization for the case of `uchar` input data and
     *        `float` output data. The data are also normalized to one (1).
     *  \note The class creates its own buffers. If you would like to provide 
     *        your own buffers, call `get` to get references to the placeholders 
     *        within the class and assign them to your buffers. You will have to 
     *        do this strictly before the call to `init`. You can also call `get` 
     *        (after the call to `init`) to get a reference to a buffer within 
     *        the class and assign it to another kernel class instance further 
     *        down in your task pipeline.
     *  
     *        The following input/output `OpenCL` memory objects are created by 
     *        a `SeparateRGB<SeparateRGBConfig::UCHAR_FLOAT>` instance:<br>
     *        |  Name  | Type | Placement | I/O | Use | Properties | Size |
     *        |  ---   |:---: |   :---:   |:---:|:---:|   :---:    |:---: |
     *        | H_IN   | Buffer | Host   | I | Staging     | CL_MEM_READ_WRITE | \f$3*width*height*sizeof\ (cl\_uchar)\f$ |
     *        | H_OUT_R| Buffer | Host   | O | Staging     | CL_MEM_READ_WRITE | \f$  width*height*sizeof\ (cl\_float)\f$ |
     *        | H_OUT_G| Buffer | Host   | O | Staging     | CL_MEM_READ_WRITE | \f$  width*height*sizeof\ (cl\_float)\f$ |
     *        | H_OUT_B| Buffer | Host   | O | Staging     | CL_MEM_READ_WRITE | \f$  width*height*sizeof\ (cl\_float)\f$ |
     *        | D_IN   | Buffer | Device | I | Processing  | CL_MEM_READ_ONLY  | \f$3*width*height*sizeof\ (cl\_uchar)\f$ |
     *        | D_OUT_R| Buffer | Device | O | Processing  | CL_MEM_WRITE_ONLY | \f$  width*height*sizeof\ (cl\_float)\f$ |
     *        | D_OUT_G| Buffer | Device | O | Processing  | CL_MEM_WRITE_ONLY | \f$  width*height*sizeof\ (cl\_float)\f$ |
     *        | D_OUT_B| Buffer | Device | O | Processing  | CL_MEM_WRITE_ONLY | \f$  width*height*sizeof\ (cl\_float)\f$ |
     */
    template <>
    class SeparateRGB<SeparateRGBConfig::UCHAR_FLOAT>
    {
    public:
        /*! \brief Enumerates the memory objects handled by the class.
         *  \note `H_*` names refer to staging buffers on the host.
         *  \note `D_*` names refer to buffers on the device.
         */
        enum class Memory : uint8_t
        {
            H_IN,     /*!< Input staging buffer. */
            H_OUT_R,  /*!< Output staging buffer for channel R. */
            H_OUT_G,  /*!< Output staging buffer for channel G. */
            H_OUT_B,  /*!< Output staging buffer for channel B. */
            D_IN,     /*!< Input buffer. */
            D_OUT_R,  /*!< Output buffer for channel R. */
            D_OUT_G,  /*!< Output buffer for channel G. */
            D_OUT_B   /*!< Output buffer for channel B. */
        };

        /*! \brief Configures an OpenCL environment as specified by `_info`. */
        SeparateRGB (clutils::CLEnv &_env, clutils::CLEnvInfo<1> _info);
        /*! \brief Returns a reference to an internal memory object. */
        cl::Memory& get (SeparateRGB::Memory mem);
        /*! \brief Configures kernel execution parameters. */
        void init (unsigned int _width, unsigned int _height, Staging _staging = Staging::IO);
        /*! \brief Performs a data transfer to a device buffer. */
        void write (SeparateRGB::Memory mem = SeparateRGB::Memory::D_IN, void *ptr = nullptr, bool block = CL_FALSE, 
                    const std::vector<cl::Event> *events = nullptr, cl::Event *event = nullptr);
        /*! \brief Performs a data transfer to a staging buffer. */
        void* read (SeparateRGB::Memory mem = SeparateRGB::Memory::H_OUT_R, bool block = CL_TRUE, 
                    const std::vector<cl::Event> *events = nullptr, cl::Event *event = nullptr);
        /*! \brief Executes the necessary kernels. */
        void run (const std::vector<cl::Event> *events = nullptr, cl::Event *event = nullptr);

        cl_uchar *hPtrIn;  /*!< Mapping of the input staging buffer. */
        cl_float *hPtrOutR;  /*!< Mapping of the output staging buffer for channel R. */
        cl_float *hPtrOutG;  /*!< Mapping of the output staging buffer for channel G. */
        cl_float *hPtrOutB;  /*!< Mapping of the output staging buffer for channel B. */

    private:
        clutils::CLEnv &env;
        clutils::CLEnvInfo<1> info;
        cl::Context context;
        cl::CommandQueue queue;
        cl::Kernel kernel;
        cl::NDRange global, local;
        Staging staging;
        size_t wgMultiple;
        unsigned int width, height;
        unsigned int bufferInSize, bufferOutSize;
        cl::Buffer hBufferIn, hBufferOutR, hBufferOutG, hBufferOutB;
        cl::Buffer dBufferIn, dBufferOutR, dBufferOutG, dBufferOutB;

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
            queue.enqueueNDRangeKernel (kernel, cl::NullRange, global, local, events, &timer.event ());
            queue.flush (); timer.wait ();

            return timer.duration ();
        }

    };


    /*! \brief Enumerates configurations for the `CombineRGB` class. */
    enum class CombineRGBConfig : uint8_t
    {
        FLOAT_FLOAT,  /*!< Identifies the case of `float` input data and `float` output data. */
        FLOAT_UCHAR   /*!< Identifies the case of `float` input data and `uchar` output data. */
    };


    /*! \brief Interface class for the `combineRGBChannels` kernels.
     *  \details The `combineRGBChannels` kernels perform array transposition on an RGB image. 
     *           For more details, look at the kernels' documentation.
     *  \note The `combineRGBChannels` kernels are available in `kernels/imageSupport_kernels.cl`.
     *  \note This is just a declaration. Look at the explicit template specializations
     *        for specific instantiations of the class.
     *        
     *  \tparam C configures the class to work with different types of data.
     */
    template <CombineRGBConfig C>
    class CombineRGB;


    /*! \brief Interface class for the `combineRGBChannels_Float2Float` kernel.
     *  \details `combineRGBChannels_Float2Float` performs array transposition on three, R, G, and B, images. 
     *           For more details, look at the kernel's documentation.
     *  \note The `combineRGBChannels_Float2Float` kernel is available in `kernels/imageSupport_kernels.cl`.
     *  \note This is a specialization for the case of `float` input and output data.
     *  \note The class creates its own buffers. If you would like to provide 
     *        your own buffers, call `get` to get references to the placeholders 
     *        within the class and assign them to your buffers. You will have to 
     *        do this strictly before the call to `init`. You can also call `get` 
     *        (after the call to `init`) to get a reference to a buffer within 
     *        the class and assign it to another kernel class instance further 
     *        down in your task pipeline.
     *  
     *        The following input/output `OpenCL` memory objects are created by 
     *        a `CombineRGB<CombineRGBConfig::FLOAT_FLOAT>` instance:<br>
     *        |  Name  | Type | Placement | I/O | Use | Properties | Size |
     *        |  ---   |:---: |   :---:   |:---:|:---:|   :---:    |:---: |
     *        | H_IN_R | Buffer | Host   | I | Staging     | CL_MEM_READ_WRITE | \f$  width*height*sizeof\ (cl\_float)\f$ |
     *        | H_IN_G | Buffer | Host   | I | Staging     | CL_MEM_READ_WRITE | \f$  width*height*sizeof\ (cl\_float)\f$ |
     *        | H_IN_B | Buffer | Host   | I | Staging     | CL_MEM_READ_WRITE | \f$  width*height*sizeof\ (cl\_float)\f$ |
     *        | H_OUT  | Buffer | Host   | O | Staging     | CL_MEM_READ_WRITE | \f$3*width*height*sizeof\ (cl\_float)\f$ |
     *        | D_IN_R | Buffer | Device | I | Processing  | CL_MEM_READ_ONLY  | \f$  width*height*sizeof\ (cl\_float)\f$ |
     *        | D_IN_G | Buffer | Device | I | Processing  | CL_MEM_READ_ONLY  | \f$  width*height*sizeof\ (cl\_float)\f$ |
     *        | D_IN_B | Buffer | Device | I | Processing  | CL_MEM_READ_ONLY  | \f$  width*height*sizeof\ (cl\_float)\f$ |
     *        | D_OUT  | Buffer | Device | O | Processing  | CL_MEM_WRITE_ONLY | \f$3*width*height*sizeof\ (cl\_float)\f$ |
     */
    template <>
    class CombineRGB<CombineRGBConfig::FLOAT_FLOAT>
    {
    public:
        /*! \brief Enumerates the memory objects handled by the class.
         *  \note `H_*` names refer to staging buffers on the host.
         *  \note `D_*` names refer to buffers on the device.
         */
        enum class Memory : uint8_t
        {
            H_IN_R,  /*!< Input staging buffer for channel R. */
            H_IN_G,  /*!< Input staging buffer for channel G. */
            H_IN_B,  /*!< Input staging buffer for channel B. */
            H_OUT,   /*!< Output staging buffer. */
            D_IN_R,  /*!< Input buffer for channel R. */
            D_IN_G,  /*!< Input buffer for channel G. */
            D_IN_B,  /*!< Input buffer for channel B. */
            D_OUT    /*!< Output buffer. */
        };

        /*! \brief Configures an OpenCL environment as specified by `_info`. */
        CombineRGB (clutils::CLEnv &_env, clutils::CLEnvInfo<1> _info);
        /*! \brief Returns a reference to an internal memory object. */
        cl::Memory& get (CombineRGB::Memory mem);
        /*! \brief Configures kernel execution parameters. */
        void init (unsigned int _width, unsigned int _height, Staging _staging = Staging::IO);
        /*! \brief Performs a data transfer to a device buffer. */
        void write (CombineRGB::Memory mem = CombineRGB::Memory::D_IN_R, void *ptr = nullptr, bool block = CL_FALSE, 
                    const std::vector<cl::Event> *events = nullptr, cl::Event *event = nullptr);
        /*! \brief Performs a data transfer to a staging buffer. */
        void* read (CombineRGB::Memory mem = CombineRGB::Memory::H_OUT, bool block = CL_TRUE, 
                    const std::vector<cl::Event> *events = nullptr, cl::Event *event = nullptr);
        /*! \brief Executes the necessary kernels. */
        void run (const std::vector<cl::Event> *events = nullptr, cl::Event *event = nullptr);

        cl_float *hPtrInR;  /*!< Mapping of the input staging buffer for channel R. */
        cl_float *hPtrInG;  /*!< Mapping of the input staging buffer for channel G. */
        cl_float *hPtrInB;  /*!< Mapping of the input staging buffer for channel B. */
        cl_float *hPtrOut;  /*!< Mapping of the output staging buffer. */

    private:
        clutils::CLEnv &env;
        clutils::CLEnvInfo<1> info;
        cl::Context context;
        cl::CommandQueue queue;
        cl::Kernel kernel;
        cl::NDRange global, local;
        Staging staging;
        size_t wgMultiple;
        unsigned int width, height;
        unsigned int bufferInSize, bufferOutSize;
        cl::Buffer hBufferInR, hBufferInG, hBufferInB, hBufferOut;
        cl::Buffer dBufferInR, dBufferInG, dBufferInB, dBufferOut;

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
            queue.enqueueNDRangeKernel (kernel, cl::NullRange, global, local, events, &timer.event ());
            queue.flush (); timer.wait ();

            return timer.duration ();
        }

    };


    /*! \brief Interface class for the `combineRGBChannels_FloatUchar` kernel.
     *  \details `combineRGBChannels_FloatUchar` performs array transposition on three, R, G, and B, images,
     *           while demoting their `float` type to `uchar`, and scaling the data to `255`.
     *           For more details, look at the kernel's documentation.
     *  \note The `combineRGBChannels_FloatUchar` kernel is available in `kernels/imageSupport_kernels.cl`.
     *  \note This is a specialization for the case of `float` input data and
     *        `uchar` output data. The data are also scaled to `255`.
     *  \note The class creates its own buffers. If you would like to provide 
     *        your own buffers, call `get` to get references to the placeholders 
     *        within the class and assign them to your buffers. You will have to 
     *        do this strictly before the call to `init`. You can also call `get` 
     *        (after the call to `init`) to get a reference to a buffer within 
     *        the class and assign it to another kernel class instance further 
     *        down in your task pipeline.
     *  
     *        The following input/output `OpenCL` memory objects are created by 
     *        a `CombineRGB<CombineRGBConfig::FLOAT_UCHAR>` instance:<br>
     *        |  Name  | Type | Placement | I/O | Use | Properties | Size |
     *        |  ---   |:---: |   :---:   |:---:|:---:|   :---:    |:---: |
     *        | H_IN_R | Buffer | Host   | I | Staging     | CL_MEM_READ_WRITE | \f$  width*height*sizeof\ (cl\_float)\f$ |
     *        | H_IN_G | Buffer | Host   | I | Staging     | CL_MEM_READ_WRITE | \f$  width*height*sizeof\ (cl\_float)\f$ |
     *        | H_IN_B | Buffer | Host   | I | Staging     | CL_MEM_READ_WRITE | \f$  width*height*sizeof\ (cl\_float)\f$ |
     *        | H_OUT  | Buffer | Host   | O | Staging     | CL_MEM_READ_WRITE | \f$3*width*height*sizeof\ (cl\_uchar)\f$ |
     *        | D_IN_R | Buffer | Device | I | Processing  | CL_MEM_READ_ONLY  | \f$  width*height*sizeof\ (cl\_float)\f$ |
     *        | D_IN_G | Buffer | Device | I | Processing  | CL_MEM_READ_ONLY  | \f$  width*height*sizeof\ (cl\_float)\f$ |
     *        | D_IN_B | Buffer | Device | I | Processing  | CL_MEM_READ_ONLY  | \f$  width*height*sizeof\ (cl\_float)\f$ |
     *        | D_OUT  | Buffer | Device | O | Processing  | CL_MEM_WRITE_ONLY | \f$3*width*height*sizeof\ (cl\_uchar)\f$ |
     */
    template <>
    class CombineRGB<CombineRGBConfig::FLOAT_UCHAR>
    {
    public:
        /*! \brief Enumerates the memory objects handled by the class.
         *  \note `H_*` names refer to staging buffers on the host.
         *  \note `D_*` names refer to buffers on the device.
         */
        enum class Memory : uint8_t
        {
            H_IN_R,  /*!< Input staging buffer for channel R. */
            H_IN_G,  /*!< Input staging buffer for channel G. */
            H_IN_B,  /*!< Input staging buffer for channel B. */
            H_OUT,   /*!< Output staging buffer. */
            D_IN_R,  /*!< Input buffer for channel R. */
            D_IN_G,  /*!< Input buffer for channel G. */
            D_IN_B,  /*!< Input buffer for channel B. */
            D_OUT    /*!< Output buffer. */
        };

        /*! \brief Configures an OpenCL environment as specified by `_info`. */
        CombineRGB (clutils::CLEnv &_env, clutils::CLEnvInfo<1> _info);
        /*! \brief Returns a reference to an internal memory object. */
        cl::Memory& get (CombineRGB::Memory mem);
        /*! \brief Configures kernel execution parameters. */
        void init (unsigned int _width, unsigned int _height, Staging _staging = Staging::IO);
        /*! \brief Performs a data transfer to a device buffer. */
        void write (CombineRGB::Memory mem = CombineRGB::Memory::D_IN_R, 
                    void *ptr = nullptr, bool block = CL_FALSE, 
                    const std::vector<cl::Event> *events = nullptr, cl::Event *event = nullptr);
        /*! \brief Performs a data transfer to a staging buffer. */
        void* read (CombineRGB::Memory mem = CombineRGB::Memory::H_OUT, bool block = CL_TRUE, 
                    const std::vector<cl::Event> *events = nullptr, cl::Event *event = nullptr);
        /*! \brief Executes the necessary kernels. */
        void run (const std::vector<cl::Event> *events = nullptr, cl::Event *event = nullptr);

        cl_float *hPtrInR;  /*!< Mapping of the input staging buffer for channel R. */
        cl_float *hPtrInG;  /*!< Mapping of the input staging buffer for channel G. */
        cl_float *hPtrInB;  /*!< Mapping of the input staging buffer for channel B. */
        cl_uchar *hPtrOut;  /*!< Mapping of the output staging buffer. */

    private:
        clutils::CLEnv &env;
        clutils::CLEnvInfo<1> info;
        cl::Context context;
        cl::CommandQueue queue;
        cl::Kernel kernel;
        cl::NDRange global, local;
        Staging staging;
        size_t wgMultiple;
        unsigned int width, height;
        unsigned int bufferInSize, bufferOutSize;
        cl::Buffer hBufferInR, hBufferInG, hBufferInB, hBufferOut;
        cl::Buffer dBufferInR, dBufferInG, dBufferInB, dBufferOut;

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
            queue.enqueueNDRangeKernel (kernel, cl::NullRange, global, local, events, &timer.event ());
            queue.flush (); timer.wait ();

            return timer.duration ();
        }

    };


    /*! \brief Enumerates configurations for the `Depth` class. */
    enum class DepthConfig : uint8_t
    {
        USHORT_FLOAT  /*!< Identifies the case of `ushort` input data and `float` output data. */
    };


    /*! \brief Interface class for the `depth` kernels.
     *  \details The `depth` kernels transform the data of a `Depth` image. 
     *           For more details, look at the kernels' documentation.
     *  \note The `depth` kernels are available in `kernels/imageSupport_kernels.cl`.
     *  \note This is just a declaration. Look at the explicit template specializations
     *        for specific instantiations of the class.
     *        
     *  \tparam C configures the class to work with different types of data.
     */
    template <DepthConfig C>
    class Depth;


    /*! \brief Interface class for the `depth_Ushort2Float` kernel.
     *  \details `depth_Ushort2Float` promotes the `unsigned short` 
     *           type of a (depth) image to `float`.
     *           For more details, look at the kernel's documentation.
     *  \note The `depth_Ushort2Float` kernel is available in `kernels/imageSupport_kernels.cl`.
     *  \note This is a specialization for the case of `ushort` input data and `float` output data.
     *  \note The class creates its own buffers. If you would like to provide 
     *        your own buffers, call `get` to get references to the placeholders 
     *        within the class and assign them to your buffers. You will have to 
     *        do this strictly before the call to `init`. You can also call `get` 
     *        (after the call to `init`) to get a reference to a buffer within 
     *        the class and assign it to another kernel class instance further 
     *        down in your task pipeline.
     *  
     *        The following input/output `OpenCL` memory objects are created by 
     *        a `Depth<DepthConfig::USHORT_FLOAT>` instance:<br>
     *        | Name | Type | Placement | I/O | Use | Properties | Size |
     *        | ---  |:---: |   :---:   |:---:|:---:|   :---:    |:---: |
     *        | H_IN | Buffer | Host   | I | Staging     | CL_MEM_READ_WRITE | \f$width*height*sizeof\ (cl\_ushort)\f$ |
     *        | H_OUT| Buffer | Host   | O | Staging     | CL_MEM_READ_WRITE | \f$width*height*sizeof\ (cl\_float) \f$ |
     *        | D_IN | Buffer | Device | I | Processing  | CL_MEM_READ_ONLY  | \f$width*height*sizeof\ (cl\_ushort)\f$ |
     *        | D_OUT| Buffer | Device | O | Processing  | CL_MEM_WRITE_ONLY | \f$width*height*sizeof\ (cl\_float) \f$ |
     */
    template <>
    class Depth<DepthConfig::USHORT_FLOAT>
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
        Depth (clutils::CLEnv &_env, clutils::CLEnvInfo<1> _info);
        /*! \brief Returns a reference to an internal memory object. */
        cl::Memory& get (Depth::Memory mem);
        /*! \brief Configures kernel execution parameters. */
        void init (unsigned int _width, unsigned int _height, float _scaling = 1.f, Staging _staging = Staging::IO);
        /*! \brief Performs a data transfer to a device buffer. */
        void write (Depth::Memory mem = Depth::Memory::D_IN, void *ptr = nullptr, bool block = CL_FALSE, 
                    const std::vector<cl::Event> *events = nullptr, cl::Event *event = nullptr);
        /*! \brief Performs a data transfer to a staging buffer. */
        void* read (Depth::Memory mem = Depth::Memory::H_OUT, bool block = CL_TRUE, 
                    const std::vector<cl::Event> *events = nullptr, cl::Event *event = nullptr);
        /*! \brief Executes the necessary kernels. */
        void run (const std::vector<cl::Event> *events = nullptr, cl::Event *event = nullptr);
        /*! \brief Gets the scaling factor. */
        float getScaling ();
        /*! \brief Sets the scaling factor. */
        void setScaling (float _scaling);

        cl_ushort *hPtrIn;  /*!< Mapping of the input staging buffer. */
        cl_float *hPtrOut;  /*!< Mapping of the output staging buffer. */

    private:
        clutils::CLEnv &env;
        clutils::CLEnvInfo<1> info;
        cl::Context context;
        cl::CommandQueue queue;
        cl::Kernel kernel;
        cl::NDRange global;
        Staging staging;
        unsigned int length;
        unsigned int bufferInSize, bufferOutSize;
        float scaling;
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


    /*! \brief Interface class for the `depthTo3D` kernel.
     *  \details `depthTo3D` performs a transformation from 2D image coordinates 
     *           to 3D world coordinates (it forms a point cloud).
     *           For more details, look at the kernel's documentation.
     *  \note The `depthTo3D` kernel is available in `kernels/imageSupport_kernels.cl`.
     *  \note The class creates its own buffers. If you would like to provide 
     *        your own buffers, call `get` to get references to the placeholders 
     *        within the class and assign them to your buffers. You will have to 
     *        do this strictly before the call to `init`. You can also call `get` 
     *        (after the call to `init`) to get a reference to a buffer within 
     *        the class and assign it to another kernel class instance further 
     *        down in your task pipeline.
     *  
     *        The following input/output `OpenCL` memory objects are created by a `DepthTo3D` instance:<br>
     *        | Name | Type | Placement | I/O | Use | Properties | Size |
     *        | ---  |:---: |   :---:   |:---:|:---:|   :---:    |:---: |
     *        | H_IN | Buffer | Host   | I | Staging     | CL_MEM_READ_WRITE | \f$width*height*sizeof\ (cl\_float) \f$ |
     *        | H_OUT| Buffer | Host   | O | Staging     | CL_MEM_READ_WRITE | \f$width*height*sizeof\ (cl\_float4)\f$ |
     *        | D_IN | Buffer | Device | I | Processing  | CL_MEM_READ_ONLY  | \f$width*height*sizeof\ (cl\_float) \f$ |
     *        | D_OUT| Buffer | Device | O | Processing  | CL_MEM_WRITE_ONLY | \f$width*height*sizeof\ (cl\_float4)\f$ |
     */
    class DepthTo3D
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
        DepthTo3D (clutils::CLEnv &_env, clutils::CLEnvInfo<1> _info);
        /*! \brief Returns a reference to an internal memory object. */
        cl::Memory& get (DepthTo3D::Memory mem);
        /*! \brief Configures kernel execution parameters. */
        void init (unsigned int _width, unsigned int _height, float _f, float _scaling = 1.f, Staging _staging = Staging::IO);
        /*! \brief Performs a data transfer to a device buffer. */
        void write (DepthTo3D::Memory mem = DepthTo3D::Memory::D_IN, void *ptr = nullptr, bool block = CL_FALSE, 
                    const std::vector<cl::Event> *events = nullptr, cl::Event *event = nullptr);
        /*! \brief Performs a data transfer to a staging buffer. */
        void* read (DepthTo3D::Memory mem = DepthTo3D::Memory::H_OUT, bool block = CL_TRUE, 
                    const std::vector<cl::Event> *events = nullptr, cl::Event *event = nullptr);
        /*! \brief Executes the necessary kernels. */
        void run (const std::vector<cl::Event> *events = nullptr, cl::Event *event = nullptr);
        /*! \brief Gets the focal length. */
        float getFocalLength ();
        /*! \brief Sets the focal length. */
        void setFocalLength (float _f);
        /*! \brief Gets the scaling factor. */
        float getScaling ();
        /*! \brief Sets the scaling factor. */
        void setScaling (float _scaling);

        cl_float *hPtrIn;  /*!< Mapping of the input staging buffer. */
        cl_float4 *hPtrOut;  /*!< Mapping of the output staging buffer. */

    private:
        clutils::CLEnv &env;
        clutils::CLEnvInfo<1> info;
        cl::Context context;
        cl::CommandQueue queue;
        cl::Kernel kernel;
        cl::NDRange global;
        Staging staging;
        unsigned int width, height;
        unsigned int bufferInSize, bufferOutSize;
        float f, scaling;
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


    /*! \brief Interface class for the `rgbdTo8D` kernel.
     *  \details `rgbdTo8D` transforms and fuses together geometry and color 
     *           information into 8D feature points.
     *           For more details, look at the kernel's documentation.
     *  \note The `rgbdTo8D` kernel is available in `kernels/imageSupport_kernels.cl`.
     *  \note The class creates its own buffers. If you would like to provide 
     *        your own buffers, call `get` to get references to the placeholders 
     *        within the class and assign them to your buffers. You will have to 
     *        do this strictly before the call to `init`. You can also call `get` 
     *        (after the call to `init`) to get a reference to a buffer within 
     *        the class and assign it to another kernel class instance further 
     *        down in your task pipeline.
     *  
     *        The following input/output `OpenCL` memory objects are created by a `RGBDTo8D` instance:<br>
     *        | Name | Type | Placement | I/O | Use | Properties | Size |
     *        | ---  |:---: |   :---:   |:---:|:---:|   :---:    |:---: |
     *        | H_IN_D| Buffer | Host   | I | Staging     | CL_MEM_READ_WRITE | \f$width*height*sizeof\ (cl\_float) \f$ |
     *        | H_IN_R| Buffer | Host   | I | Staging     | CL_MEM_READ_WRITE | \f$width*height*sizeof\ (cl\_float) \f$ |
     *        | H_IN_G| Buffer | Host   | I | Staging     | CL_MEM_READ_WRITE | \f$width*height*sizeof\ (cl\_float) \f$ |
     *        | H_IN_B| Buffer | Host   | I | Staging     | CL_MEM_READ_WRITE | \f$width*height*sizeof\ (cl\_float) \f$ |
     *        | H_OUT | Buffer | Host   | O | Staging     | CL_MEM_READ_WRITE | \f$width*height*sizeof\ (cl\_float8)\f$ |
     *        | D_IN_D| Buffer | Device | I | Processing  | CL_MEM_READ_ONLY  | \f$width*height*sizeof\ (cl\_float) \f$ |
     *        | D_IN_R| Buffer | Device | I | Processing  | CL_MEM_READ_ONLY  | \f$width*height*sizeof\ (cl\_float) \f$ |
     *        | D_IN_G| Buffer | Device | I | Processing  | CL_MEM_READ_ONLY  | \f$width*height*sizeof\ (cl\_float) \f$ |
     *        | D_IN_B| Buffer | Device | I | Processing  | CL_MEM_READ_ONLY  | \f$width*height*sizeof\ (cl\_float) \f$ |
     *        | D_OUT | Buffer | Device | O | Processing  | CL_MEM_WRITE_ONLY | \f$width*height*sizeof\ (cl\_float8)\f$ |
     */
    class RGBDTo8D
    {
    public:
        /*! \brief Enumerates the memory objects handled by the class.
         *  \note `H_*` names refer to staging buffers on the host.
         *  \note `D_*` names refer to buffers on the device.
         */
        enum class Memory : uint8_t
        {
            H_IN_D,  /*!< Input staging buffer for the %Depth image. */
            H_IN_R,  /*!< Input staging buffer for channel R of the RGB image. */
            H_IN_G,  /*!< Input staging buffer for channel G of the RGB image. */
            H_IN_B,  /*!< Input staging buffer for channel B of the RGB image. */
            H_OUT,   /*!< Output staging buffer. */
            D_IN_D,  /*!< Input buffer for the %Depth image. */
            D_IN_R,  /*!< Input buffer for channel R of the RGB image. */
            D_IN_G,  /*!< Input buffer for channel G of the RGB image. */
            D_IN_B,  /*!< Input buffer for channel B of the RGB image. */
            D_OUT    /*!< Output buffer. */
        };

        /*! \brief Configures an OpenCL environment as specified by `_info`. */
        RGBDTo8D (clutils::CLEnv &_env, clutils::CLEnvInfo<1> _info);
        /*! \brief Returns a reference to an internal memory object. */
        cl::Memory& get (RGBDTo8D::Memory mem);
        /*! \brief Configures kernel execution parameters. */
        void init (unsigned int _width, unsigned int _height, float _f, float _scaling = 1.f, Staging _staging = Staging::IO);
        /*! \brief Performs a data transfer to a device buffer. */
        void write (RGBDTo8D::Memory mem = RGBDTo8D::Memory::D_IN_D, void *ptr = nullptr, bool block = CL_FALSE, 
                    const std::vector<cl::Event> *events = nullptr, cl::Event *event = nullptr);
        /*! \brief Performs a data transfer to a staging buffer. */
        void* read (RGBDTo8D::Memory mem = RGBDTo8D::Memory::H_OUT, bool block = CL_TRUE, 
                    const std::vector<cl::Event> *events = nullptr, cl::Event *event = nullptr);
        /*! \brief Executes the necessary kernels. */
        void run (const std::vector<cl::Event> *events = nullptr, cl::Event *event = nullptr);
        /*! \brief Gets the focal length. */
        float getFocalLength ();
        /*! \brief Sets the focal length. */
        void setFocalLength (float _f);
        /*! \brief Gets the scaling factor. */
        float getScaling ();
        /*! \brief Sets the scaling factor. */
        void setScaling (float _scaling);

        cl_float *hPtrInD;  /*!< Mapping of the input staging buffer for the %Depth image. */
        cl_float *hPtrInR;  /*!< Mapping of the input staging buffer for channel R of the RGB image. */
        cl_float *hPtrInG;  /*!< Mapping of the input staging buffer for channel G of the RGB image. */
        cl_float *hPtrInB;  /*!< Mapping of the input staging buffer for channel B of the RGB image. */
        cl_float8 *hPtrOut;  /*!< Mapping of the output staging buffer. */

    private:
        clutils::CLEnv &env;
        clutils::CLEnvInfo<1> info;
        cl::Context context;
        cl::CommandQueue queue;
        cl::Kernel kernel;
        cl::NDRange global, local;
        Staging staging;
        size_t wgMultiple;
        unsigned int width, height, points;
        unsigned int bufferInSize, bufferOutSize;
        float f, scaling;
        cl::Buffer hBufferInD, hBufferInR, hBufferInG, hBufferInB, hBufferOut;
        cl::Buffer dBufferInD, dBufferInR, dBufferInG, dBufferInB, dBufferOut;

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
            queue.enqueueNDRangeKernel (kernel, cl::NullRange, global, local, events, &timer.event ());
            queue.flush (); timer.wait ();

            return timer.duration ();
        }

    };


    /*! \brief Interface class for the `rgbNorm` kernel.
     *  \details `rgbNorm` performs RGB color normalization.
     *           For more details, look at the kernel's documentation.
     *  \note The `rgbNorm` kernel is available in `kernels/imageSupport_kernels.cl`.
     *  \note The class creates its own buffers. If you would like to provide 
     *        your own buffers, call `get` to get references to the placeholders 
     *        within the class and assign them to your buffers. You will have to 
     *        do this strictly before the call to `init`. You can also call `get` 
     *        (after the call to `init`) to get a reference to a buffer within 
     *        the class and assign it to another kernel class instance further 
     *        down in your task pipeline.
     *  
     *        The following input/output `OpenCL` memory objects are created by a `RGBNorm` instance:<br>
     *        | Name | Type | Placement | I/O | Use | Properties | Size |
     *        | ---  |:---: |   :---:   |:---:|:---:|   :---:    |:---: |
     *        | H_IN | Buffer | Host   | I | Staging     | CL_MEM_READ_WRITE | \f$3*width*height*sizeof\ (cl\_float)\f$ |
     *        | H_OUT| Buffer | Host   | O | Staging     | CL_MEM_READ_WRITE | \f$3*width*height*sizeof\ (cl\_float)\f$ |
     *        | D_IN | Buffer | Device | I | Processing  | CL_MEM_READ_ONLY  | \f$3*width*height*sizeof\ (cl\_float)\f$ |
     *        | D_OUT| Buffer | Device | O | Processing  | CL_MEM_WRITE_ONLY | \f$3*width*height*sizeof\ (cl\_float)\f$ |
     */
    class RGBNorm
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
        RGBNorm (clutils::CLEnv &_env, clutils::CLEnvInfo<1> _info);
        /*! \brief Returns a reference to an internal memory object. */
        cl::Memory& get (RGBNorm::Memory mem);
        /*! \brief Configures kernel execution parameters. */
        void init (unsigned int _width, unsigned int _height, Staging _staging = Staging::IO);
        /*! \brief Performs a data transfer to a device buffer. */
        void write (RGBNorm::Memory mem = RGBNorm::Memory::D_IN, void *ptr = nullptr, bool block = CL_FALSE, 
                    const std::vector<cl::Event> *events = nullptr, cl::Event *event = nullptr);
        /*! \brief Performs a data transfer to a staging buffer. */
        void* read (RGBNorm::Memory mem = RGBNorm::Memory::H_OUT, bool block = CL_TRUE, 
                    const std::vector<cl::Event> *events = nullptr, cl::Event *event = nullptr);
        /*! \brief Executes the necessary kernels. */
        void run (const std::vector<cl::Event> *events = nullptr, cl::Event *event = nullptr);

        cl_float *hPtrIn;  /*!< Mapping of the input staging buffer. */
        cl_float *hPtrOut;  /*!< Mapping of the output staging buffer. */

    private:
        clutils::CLEnv &env;
        clutils::CLEnvInfo<1> info;
        cl::Context context;
        cl::CommandQueue queue;
        cl::Kernel kernel;
        cl::NDRange global;
        Staging staging;
        unsigned int width, height, bufferSize;
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


    /*! \brief Interface class for the `scan` kernel.
     *  \details `scan` performs a scan operation on each row in an array. 
     *           For more details, look at the kernel's documentation.
     *  \note The `scan` kernel is available in `kernels/scan_kernels.cl`.
     *  \note The class creates its own buffers. If you would like to provide 
     *        your own buffers, call `get` to get references to the placeholders 
     *        within the class and assign them to your buffers. You will have to 
     *        do this strictly before the call to `init`. You can also call `get` 
     *        (after the call to `init`) to get a reference to a buffer within 
     *        the class and assign it to another kernel class instance further 
     *        down in your task pipeline.
     *  
     *        The following input/output `OpenCL` memory objects are created by a `Scan` instance:<br>
     *        | Name | Type | Placement | I/O | Use | Properties | Size |
     *        | ---  |:---: |   :---:   |:---:|:---:|   :---:    |:---: |
     *        | H_IN | Buffer | Host   | I | Staging     | CL_MEM_READ_WRITE | \f$width*height*sizeof\ (cl\_float)\f$ |
     *        | H_OUT| Buffer | Host   | O | Staging     | CL_MEM_READ_WRITE | \f$width*height*sizeof\ (cl\_float)\f$ |
     *        | D_IN | Buffer | Device | I | Processing  | CL_MEM_READ_ONLY  | \f$width*height*sizeof\ (cl\_float)\f$ |
     *        | D_OUT| Buffer | Device | O | Processing  | CL_MEM_READ_WRITE | \f$width*height*sizeof\ (cl\_float)\f$ |
     */
    class Scan
    {
    public:
        /*! \brief Enumerates the memory objects handled by the class.
         *  \note `H_*` names refer to staging buffers on the host.
         *  \note `D_*` names refer to buffers on the device.
         */
        enum class Memory : uint8_t
        {
            H_IN,    /*!< Input staging buffer. */
            H_OUT,   /*!< Output staging buffer. */
            D_IN,    /*!< Input buffer. */
            D_SUMS,  /*!< Buffer of partial group sums. */
            D_OUT    /*!< Output buffer. */
        };

        /*! \brief Configures an OpenCL environment as specified by `_info`. */
        Scan (clutils::CLEnv &_env, clutils::CLEnvInfo<1> _info);
        /*! \brief Returns a reference to an internal memory object. */
        cl::Memory& get (Scan::Memory mem);
        /*! \brief Configures kernel execution parameters. */
        void init (unsigned int _width, unsigned int _height, float _scaling = 1.f, Staging _staging = Staging::IO);
        /*! \brief Performs a data transfer to a device buffer. */
        void write (Scan::Memory mem = Scan::Memory::D_IN, void *ptr = nullptr, bool block = CL_FALSE, 
                    const std::vector<cl::Event> *events = nullptr, cl::Event *event = nullptr);
        /*! \brief Performs a data transfer to a staging buffer. */
        void* read (Scan::Memory mem = Scan::Memory::H_OUT, bool block = CL_TRUE, 
                    const std::vector<cl::Event> *events = nullptr, cl::Event *event = nullptr);
        /*! \brief Executes the necessary kernels. */
        void run (const std::vector<cl::Event> *events = nullptr, cl::Event *event = nullptr);
        /*! \brief Gets the scaling factor. */
        float getScaling ();
        /*! \brief Sets the scaling factor. */
        void setScaling (float _scaling);

        cl_float *hPtrIn;  /*!< Mapping of the input staging buffer. */
        cl_float *hPtrOut;  /*!< Mapping of the output staging buffer. */

    private:
        clutils::CLEnv &env;
        clutils::CLEnvInfo<1> info;
        cl::Context context;
        cl::CommandQueue queue;
        cl::Kernel kernelScan, kernelSumsScan, kernelAddSums;
        cl::NDRange globalScan, globalSumsScan, localScan;
        cl::NDRange globalAddSums, localAddSums, offsetAddSums;
        Staging staging;
        size_t wgMultiple, wgXdim;
        unsigned int width, height, bufferSize, bufferSumsSize;
        float scaling;
        cl::Buffer hBufferIn, hBufferOut;
        cl::Buffer dBufferIn, dBufferOut, dBufferSums;

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
            double pTime;

            if (wgXdim == 1)
            {
                queue.enqueueNDRangeKernel (
                    kernelScan, cl::NullRange, globalScan, localScan, events, &timer.event ());
                queue.flush (); timer.wait ();
                pTime = timer.duration ();
            }
            else
            {
                queue.enqueueNDRangeKernel (
                    kernelScan, cl::NullRange, globalScan, localScan, events, &timer.event ());
                queue.flush (); timer.wait ();
                pTime = timer.duration ();

                queue.enqueueNDRangeKernel (
                    kernelSumsScan, cl::NullRange, globalSumsScan, localScan, nullptr, &timer.event ());
                queue.flush (); timer.wait ();
                pTime += timer.duration ();

                queue.enqueueNDRangeKernel (
                    kernelAddSums, offsetAddSums, globalAddSums, localAddSums, nullptr, &timer.event ());
                queue.flush (); timer.wait ();
                pTime += timer.duration ();
            }

            return pTime;
        }

    };


    /*! \brief Interface class for the `transpose` kernel.
     *  \details `transpose` performs a matrix transposition. 
     *           For more details, look at the kernel's documentation.
     *  \note The `transpose` kernel is available in `kernels/transpose_kernels.cl`.
     *  \note The class creates its own buffers. If you would like to provide 
     *        your own buffers, call `get` to get references to the placeholders 
     *        within the class and assign them to your buffers. You will have to 
     *        do this strictly before the call to `init`. You can also call `get` 
     *        (after the call to `init`) to get a reference to a buffer within 
     *        the class and assign it to another kernel class instance further 
     *        down in your task pipeline.
     *  
     *        The following input/output `OpenCL` memory objects are created by a `Transpose` instance:<br>
     *        | Name | Type | Placement | I/O | Use | Properties | Size |
     *        | ---  |:---: |   :---:   |:---:|:---:|   :---:    |:---: |
     *        | H_IN | Buffer | Host   | I | Staging     | CL_MEM_READ_WRITE | \f$width*height*sizeof\ (cl\_float)\f$ |
     *        | H_OUT| Buffer | Host   | O | Staging     | CL_MEM_READ_WRITE | \f$width*height*sizeof\ (cl\_float)\f$ |
     *        | D_IN | Buffer | Device | I | Processing  | CL_MEM_READ_ONLY  | \f$width*height*sizeof\ (cl\_float)\f$ |
     *        | D_OUT| Buffer | Device | O | Processing  | CL_MEM_WRITE_ONLY | \f$width*height*sizeof\ (cl\_float)\f$ |
     */
    class Transpose
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
        Transpose (clutils::CLEnv &_env, clutils::CLEnvInfo<1> _info);
        /*! \brief Returns a reference to an internal memory object. */
        cl::Memory& get (Transpose::Memory mem);
        /*! \brief Configures kernel execution parameters. */
        void init (unsigned int _width, unsigned int _height, Staging _staging = Staging::IO);
        /*! \brief Performs a data transfer to a device buffer. */
        void write (Transpose::Memory mem = Transpose::Memory::D_IN, void *ptr = nullptr, bool block = CL_FALSE, 
                    const std::vector<cl::Event> *events = nullptr, cl::Event *event = nullptr);
        /*! \brief Performs a data transfer to a staging buffer. */
        void* read (Transpose::Memory mem = Transpose::Memory::H_OUT, bool block = CL_TRUE, 
                    const std::vector<cl::Event> *events = nullptr, cl::Event *event = nullptr);
        /*! \brief Executes the necessary kernels. */
        void run (const std::vector<cl::Event> *events = nullptr, cl::Event *event = nullptr);

        cl_float *hPtrIn;  /*!< Mapping of the input staging buffer. */
        cl_float *hPtrOut;  /*!< Mapping of the output staging buffer. */

    private:
        clutils::CLEnv &env;
        clutils::CLEnvInfo<1> info;
        cl::Context context;
        cl::CommandQueue queue;
        cl::Kernel kernel;
        cl::NDRange global, local;
        Staging staging;
        unsigned int width, height, bufferSize;
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
            queue.enqueueNDRangeKernel (kernel, cl::NullRange, global, local, events, &timer.event ());
            queue.flush (); timer.wait ();

            return timer.duration ();
        }

    };


    /*! \brief Interface class for the `Summed Area Table` operation.
     *  \note The class makes use of the `scan` and `transpose` kernels.
     *  \note It first scans the rows, then transposes the array, and then 
     *        scans the columns. Lastly, there is the option to leave the array
     *        in the transposed configuration, or transpose it again.
     *  \note The class creates its own buffers. If you would like to provide 
     *        your own buffers, call `get` to get references to the placeholders 
     *        within the class and assign them to your buffers. You will have to 
     *        do this strictly before the call to `init`. You can also call `get` 
     *        (after the call to `init`) to get a reference to a buffer within 
     *        the class and assign it to another kernel class instance further 
     *        down in your task pipeline.
     *  
     *        The following input/output `OpenCL` memory objects are created by a `SAT` instance:<br>
     *        | Name | Type | Placement | I/O | Use | Properties | Size |
     *        | ---  |:---: |   :---:   |:---:|:---:|   :---:    |:---: |
     *        | H_IN | Buffer | Host   | I | Staging     | CL_MEM_READ_WRITE | \f$width*height*sizeof\ (cl\_float)\f$ |
     *        | H_OUT| Buffer | Host   | O | Staging     | CL_MEM_READ_WRITE | \f$width*height*sizeof\ (cl\_float)\f$ |
     *        | D_IN | Buffer | Device | I | Processing  | CL_MEM_READ_ONLY  | \f$width*height*sizeof\ (cl\_float)\f$ |
     *        | D_OUT| Buffer | Device | O | Processing  | CL_MEM_READ_WRITE `(transposed)`<br>CL_MEM_WRITE_ONLY `(!transposed)` | \f$width*height*sizeof\ (cl\_float)\f$ |
     */
    class SAT
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
        SAT (clutils::CLEnv &_env, clutils::CLEnvInfo<1> _info, bool _transposed = true);
        /*! \brief Returns a reference to an internal memory object. */
        cl::Memory& get (SAT::Memory mem);
        /*! \brief Configures kernel execution parameters. */
        void init (unsigned int _width, unsigned int _height, float _scaling = 1.f, Staging _staging = Staging::IO);
        /*! \brief Performs a data transfer to a device buffer. */
        void write (SAT::Memory mem = SAT::Memory::D_IN, void *ptr = nullptr, bool block = CL_FALSE, 
                    const std::vector<cl::Event> *events = nullptr, cl::Event *event = nullptr);
        /*! \brief Performs a data transfer to a staging buffer. */
        void* read (SAT::Memory mem = SAT::Memory::H_OUT, bool block = CL_TRUE, 
                    const std::vector<cl::Event> *events = nullptr, cl::Event *event = nullptr);
        /*! \brief Executes the necessary kernels. */
        void run (const std::vector<cl::Event> *events = nullptr, cl::Event *event = nullptr);
        /*! \brief Gets the scaling factor. */
        float getScaling ();
        /*! \brief Sets the scaling factor. */
        void setScaling (float _scaling);

        cl_float *hPtrIn;  /*!< Mapping of the input staging buffer. */
        cl_float *hPtrOut;  /*!< Mapping of the output staging buffer. */

    private:
        clutils::CLEnv &env;
        clutils::CLEnvInfo<1> info;
        cl::Context context;
        cl::CommandQueue queue;
        Scan scanRows, scanColumns;
        Transpose transpose1, transpose2;
        Staging staging;
        unsigned int width, height, bufferSize;
        float scaling;
        bool transposed;
        cl::Buffer hBufferIn, hBufferOut;

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
            double pTime;

            pTime = scanRows.run (timer, events);
            pTime += transpose1.run (timer);
            pTime += scanColumns.run (timer);
            
            if (!transposed)
                pTime += transpose2.run (timer);

            return pTime;
        }

    };


    /*! \brief Interface class for the `boxFilterSAT{_Tr}` kernel.
     *  \details `boxFilterSAT{_Tr}` performs a mean filtering operation. 
     *           For more details, look at the kernel's documentation.
     *  \note The `boxFilterSAT{_Tr}` kernel is available in `kernels/boxFilter_kernels.cl`.
     *  \note The class creates its own buffers. If you would like to provide 
     *        your own buffers, call `get` to get references to the placeholders 
     *        within the class and assign them to your buffers. You will have to 
     *        do this strictly before the call to `init`. You can also call `get` 
     *        (after the call to `init`) to get a reference to a buffer within 
     *        the class and assign it to another kernel class instance further 
     *        down in your task pipeline.
     *  
     *        The following input/output `OpenCL` memory objects are created by a `BoxFilterSAT` instance:<br>
     *        | Name | Type | Placement | I/O | Use | Properties | Size |
     *        | ---  |:---: |   :---:   |:---:|:---:|   :---:    |:---: |
     *        | H_IN | Buffer | Host   | I | Staging     | CL_MEM_READ_WRITE | \f$width*height*sizeof\ (cl\_float)\f$ |
     *        | H_OUT| Buffer | Host   | O | Staging     | CL_MEM_READ_WRITE | \f$width*height*sizeof\ (cl\_float)\f$ |
     *        | D_IN | Buffer | Device | I | Processing  | CL_MEM_READ_ONLY  | \f$width*height*sizeof\ (cl\_float)\f$ |
     *        | D_OUT| Buffer | Device | O | Processing  | CL_MEM_WRITE_ONLY | \f$width*height*sizeof\ (cl\_float)\f$ |
     */
    class BoxFilterSAT
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
        BoxFilterSAT (clutils::CLEnv &_env, clutils::CLEnvInfo<1> _info);
        /*! \brief Returns a reference to an internal memory object. */
        cl::Memory& get (BoxFilterSAT::Memory mem);
        /*! \brief Configures kernel execution parameters. */
        void init (unsigned int _width, unsigned int _height, int _radius, float _scaling = 1e-4f, Staging _staging = Staging::IO);
        /*! \brief Performs a data transfer to a device buffer. */
        void write (BoxFilterSAT::Memory mem = BoxFilterSAT::Memory::D_IN, void *ptr = nullptr, bool block = CL_FALSE, 
                    const std::vector<cl::Event> *events = nullptr, cl::Event *event = nullptr);
        /*! \brief Performs a data transfer to a staging buffer. */
        void* read (BoxFilterSAT::Memory mem = BoxFilterSAT::Memory::H_OUT, bool block = CL_TRUE, 
                    const std::vector<cl::Event> *events = nullptr, cl::Event *event = nullptr);
        /*! \brief Executes the necessary kernels. */
        void run (const std::vector<cl::Event> *events = nullptr, cl::Event *event = nullptr);
        /*! \brief Gets the filter window radius. */
        int getRadius ();
        /*! \brief Sets the filter window radius. */
        void setRadius (int _radius);
        /*! \brief Gets the scaling factor. */
        float getScaling ();
        /*! \brief Sets the scaling factor. */
        void setScaling (float _scaling);

        cl_float *hPtrIn;  /*!< Mapping of the input staging buffer. */
        cl_float *hPtrOut;  /*!< Mapping of the output staging buffer. */

    private:
        static const unsigned int lXdim = 16;
        static const unsigned int lYdim = 16;
        clutils::CLEnv &env;
        clutils::CLEnvInfo<1> info;
        cl::Context context;
        cl::CommandQueue queue;
        cl::Kernel kernel;
        cl::NDRange global, local;
        Staging staging;
        unsigned int width, height, bufferSize;
        int radius;
        float scaling;
        SAT sat;
        cl::Buffer hBufferIn, hBufferOut;
        cl::Buffer dBufferOut;

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
            double pTime;

            pTime = sat.run (timer, events);
            
            queue.enqueueNDRangeKernel (
                kernel, cl::NullRange, global, local, nullptr, &timer.event ());
            queue.flush (); timer.wait ();
            pTime += timer.duration ();

            return pTime;
        }

    };


    /*! \brief Interface class for the `boxFilter` kernel.
     *  \details `boxFilter` performs a mean filtering operation. 
     *           For more details, look at the kernel's documentation.
     *  \note The `boxFilter` kernel is available in `kernels/boxFilter_kernels.cl`.
     *  \note The class creates its own buffers. If you would like to provide 
     *        your own buffers, call `get` to get references to the placeholders 
     *        within the class and assign them to your buffers. You will have to 
     *        do this strictly before the call to `init`. You can also call `get` 
     *        (after the call to `init`) to get a reference to a buffer within 
     *        the class and assign it to another kernel class instance further 
     *        down in your task pipeline.
     *  
     *        The following input/output `OpenCL` memory objects are created by a `BoxFilter` instance:<br>
     *        | Name | Type | Placement | I/O | Use | Properties | Size |
     *        | ---  |:---: |   :---:   |:---:|:---:|   :---:    |:---: |
     *        | H_IN | Buffer | Host   | I | Staging     | CL_MEM_READ_WRITE | \f$width*height*sizeof\ (cl\_float)\f$ |
     *        | H_OUT| Buffer | Host   | O | Staging     | CL_MEM_READ_WRITE | \f$width*height*sizeof\ (cl\_float)\f$ |
     *        | D_IN | Buffer | Device | I | Processing  | CL_MEM_READ_ONLY  | \f$width*height*sizeof\ (cl\_float)\f$ |
     *        | D_OUT| Buffer | Device | O | Processing  | CL_MEM_WRITE_ONLY | \f$width*height*sizeof\ (cl\_float)\f$ |
     */
    class BoxFilter
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
        BoxFilter (clutils::CLEnv &_env, clutils::CLEnvInfo<1> _info);
        /*! \brief Returns a reference to an internal memory object. */
        cl::Memory& get (BoxFilter::Memory mem);
        /*! \brief Configures kernel execution parameters. */
        void init (unsigned int _width, unsigned int _height, int _radius, Staging _staging = Staging::IO);
        /*! \brief Performs a data transfer to a device buffer. */
        void write (BoxFilter::Memory mem = BoxFilter::Memory::D_IN, void *ptr = nullptr, bool block = CL_FALSE, 
                    const std::vector<cl::Event> *events = nullptr, cl::Event *event = nullptr);
        /*! \brief Performs a data transfer to a staging buffer. */
        void* read (BoxFilter::Memory mem = BoxFilter::Memory::H_OUT, bool block = CL_TRUE, 
                    const std::vector<cl::Event> *events = nullptr, cl::Event *event = nullptr);
        /*! \brief Executes the necessary kernels. */
        void run (const std::vector<cl::Event> *events = nullptr, cl::Event *event = nullptr);
        /*! \brief Gets the filter window radius. */
        int getRadius ();
        /*! \brief Sets the filter window radius. */
        void setRadius (int _radius);

        cl_float *hPtrIn;  /*!< Mapping of the input staging buffer. */
        cl_float *hPtrOut;  /*!< Mapping of the output staging buffer. */

    private:
        static const unsigned int lXdim = 16;
        static const unsigned int lYdim = 16;
        clutils::CLEnv &env;
        clutils::CLEnvInfo<1> info;
        cl::Context context;
        cl::CommandQueue queue;
        cl::Kernel kernel;
        cl::NDRange global, local;
        Staging staging;
        unsigned int width, height, bufferSize;
        int radius;
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
            queue.enqueueNDRangeKernel (
                kernel, cl::NullRange, global, local, nullptr, &timer.event ());
            queue.flush (); timer.wait ();

            return timer.duration ();
        }

    };


    /*! \brief Enumerates configurations for the `Guided Filter` algorithm. */
    enum class GuidedFilterConfig : uint8_t
    {
        I_NEQ_P,  /*!< Identifies the general case where \f$ I \neq p \f$. */
        I_EQ_P    /*!< Identifies the special case where \f$ I == p \f$. */
    };


    /*! \brief Interface class for the `Guided Filter` algorithm.
     *  \details The `Guided Filter` algorithm performs a number of operations,
     *           one of which is edge preserving smoothing.
     *  \note Two cases are covered. One where the guidance image `I` 
     *        is different from the input image `p`, and another where 
     *        these two are the same.
     *  \note This is just a declaration. Look at the explicit template specializations
     *        for specific instantiations of the class.
     *        
     *  \tparam Ip enables one of the two cases of `Guided Filter` algorithm, 
     *             \f$ I \neq p\ \f$ or \f$\ I == p \f$.
     */
    template <GuidedFilterConfig Ip>
    class GuidedFilter;


    /*! \brief Interface class for the `Guided Filter` pipeline.
     *  \details This instantiation covers the case where \f$ I == p \f$.
     *  \note The kernels, specific to the `Guided Filter` algorithm, 
     *        are available in `kernels/guidedFilter_kernels.cl`.
     *  \note The class creates its own buffers. If you would like to provide 
     *        your own buffers, call `get` to get references to the placeholders 
     *        within the class and assign them to your buffers. You will have to 
     *        do this strictly before the call to `init`. You can also call `get` 
     *        (after the call to `init`) to get a reference to a buffer within 
     *        the class and assign it to another kernel class instance further 
     *        down in your task pipeline.
     *  
     *        The following input/output `OpenCL` memory objects are created by 
     *        a `GuidedFilter<GuidedFilterConfig::I_EQ_P>` instance:<br>
     *        | Name | Type | Placement | I/O | Use | Properties | Size |
     *        | ---  |:---: |   :---:   |:---:|:---:|   :---:    |:---: |
     *        | H_IN | Buffer | Host   | I | Staging     | CL_MEM_READ_WRITE | \f$width*height*sizeof\ (cl\_float)\f$ |
     *        | H_OUT| Buffer | Host   | O | Staging     | CL_MEM_READ_WRITE | \f$width*height*sizeof\ (cl\_float)\f$ |
     *        | D_IN | Buffer | Device | I | Processing  | CL_MEM_READ_ONLY  | \f$width*height*sizeof\ (cl\_float)\f$ |
     *        | D_OUT| Buffer | Device | O | Processing  | CL_MEM_WRITE_ONLY | \f$width*height*sizeof\ (cl\_float)\f$ |
     */
    template <>
    class GuidedFilter<GuidedFilterConfig::I_EQ_P>
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
            D_OUT,  /*!< Output buffer. */
            D_A,    /*!< Buffer of \f$ a \f$ coefficients. */
            D_B     /*!< Buffer of \f$ b \f$ coefficients. */
        };

        /*! \brief Configures an OpenCL environment as specified by `_info`. */
        GuidedFilter (clutils::CLEnv &_env, clutils::CLEnvInfo<2> _info);
        /*! \brief Returns a reference to an internal memory object. */
        cl::Memory& get (GuidedFilter::Memory mem);
        /*! \brief Configures kernel execution parameters. */
        void init (unsigned int _width, unsigned int _height, int _radius, float _eps, 
                   int _zero_out = 0, float _scaling = 1e-4f, Staging _staging = Staging::IO);
        /*! \brief Performs a data transfer to a device buffer. */
        void write (GuidedFilter::Memory mem = GuidedFilter::Memory::D_IN, void *ptr = nullptr, bool block = CL_FALSE, 
                    const std::vector<cl::Event> *events = nullptr, cl::Event *event = nullptr);
        /*! \brief Performs a data transfer to a staging buffer. */
        void* read (GuidedFilter::Memory mem = GuidedFilter::Memory::H_OUT, bool block = CL_TRUE, 
                    const std::vector<cl::Event> *events = nullptr, cl::Event *event = nullptr);
        /*! \brief Executes the necessary kernels. */
        void run (const std::vector<cl::Event> *events = nullptr, cl::Event *event = nullptr);
        /*! \brief Gets the filter window radius. */
        int getRadius ();
        /*! \brief Sets the filter window radius. */
        void setRadius (int _radius);
        /*! \brief Gets the regularization parameter \f$\epsilon\f$. */
        float getEps ();
        /*! \brief Sets the regularization parameter \f$\epsilon\f$. */
        void setEps (float _eps);
        /*! \brief Gets the scaling factor. */
        float getScaling ();
        /*! \brief Sets the scaling factor. */
        void setScaling (float _scaling);
        /*! \brief Gets the `zero_out` flag. */
        int getZeroing ();
        /*! \brief Sets the `zero_out` flag. */
        void setZeroing (int _zero_out);

        cl_float *hPtrIn;  /*!< Mapping of the input staging buffer. */
        cl_float *hPtrOut;  /*!< Mapping of the output staging buffer. */

    private:
        clutils::CLEnv &env;
        clutils::CLEnvInfo<2> info;
        cl::Context context;
        cl::CommandQueue queue0;
        BoxFilterSAT mean_p, mean_p2, mean_a, mean_b;
        Math::Pown squared;
        cl::Kernel ab, q;
        cl::NDRange global;
        Staging staging;
        unsigned int width, height, bufferSize;
        int radius; float eps;
        int zero_out;
        float scaling;
        cl::Buffer hBufferIn, hBufferOut;
        cl::Buffer dBufferIn, dBufferOut;
        cl::Buffer dBufferOutA, dBufferOutB;
        cl::Event p2Event, abEvent, mbEvent;
        std::vector<cl::Event> waitListAB, waitListMB, waitListQ;

    public:
        /*! \brief Executes the necessary kernels.
         *  \details This `run` instance is used for profiling.
         *  \note The execution is handled by two separate command queues. The 
         *        time measured is the flat **execution** time of all the kernels. 
         *        There is overlap in the execution of the kernels, but there are 
         *        also gaps between them. As a compromise, and in order to simplify 
         *        things, the execution times of all the kernels are simply added 
         *        together. Look at the `Timeline Trace` of `CodeXL` 
         *        for a more detailed view.
         *  
         *  \param[in] timer `GPUTimer` that does the profiling of the kernel executions.
         *  \param[in] events a wait-list of events.
         *  \return Τhe total execution time measured by the timer.
         */
        template <typename period>
        double run (clutils::GPUTimer<period> &timer, const std::vector<cl::Event> *events = nullptr)
        {
            double pTime;

            pTime = mean_p.run (timer, events);
            pTime += squared.run (timer);
            pTime += mean_p2.run (timer);
            
            queue0.enqueueNDRangeKernel (ab, cl::NullRange, global, cl::NullRange, nullptr, &timer.event ());
            queue0.flush (); timer.wait ();
            pTime += timer.duration ();

            pTime += mean_a.run (timer);
            pTime += mean_b.run (timer);
            
            queue0.enqueueNDRangeKernel (q, cl::NullRange, global, cl::NullRange, nullptr, &timer.event ());
            queue0.flush (); timer.wait ();
            pTime += timer.duration ();

            return pTime;
        }

    };


    /*! \brief Interface class for the `Guided Filter` pipeline.
     *  \details This instantiation covers the case where \f$ I \neq p \f$.
     *  \note The kernels, specific to the `Guided Filter` algorithm, 
     *        are available in `kernels/guidedFilter_kernels.cl`.
     *  \note The class creates its own buffers. If you would like to provide 
     *        your own buffers, call `get` to get references to the placeholders 
     *        within the class and assign them to your buffers. You will have to 
     *        do this strictly before the call to `init`. You can also call `get` 
     *        (after the call to `init`) to get a reference to a buffer within 
     *        the class and assign it to another kernel class instance further 
     *        down in your task pipeline.

     *        The following input/output `OpenCL` memory objects are created by 
     *        a `GuidedFilter<GuidedFilterConfig::I_NEQ_P>` instance:<br>
     *        | Name | Type | Placement | I/O | Use | Properties | Size |
     *        | ---  |:---: |   :---:   |:---:|:---:|   :---:    |:---: |
     *        | H_IN_I | Buffer | Host   | I | Staging     | CL_MEM_READ_WRITE | \f$width*height*sizeof\ (cl\_float)\f$ |
     *        | H_IN_P | Buffer | Host   | I | Staging     | CL_MEM_READ_WRITE | \f$width*height*sizeof\ (cl\_float)\f$ |
     *        | H_OUT  | Buffer | Host   | O | Staging     | CL_MEM_READ_WRITE | \f$width*height*sizeof\ (cl\_float)\f$ |
     *        | D_IN_I | Buffer | Device | I | Processing  | CL_MEM_READ_ONLY  | \f$width*height*sizeof\ (cl\_float)\f$ |
     *        | D_IN_P | Buffer | Device | I | Processing  | CL_MEM_READ_ONLY  | \f$width*height*sizeof\ (cl\_float)\f$ |
     *        | D_OUT  | Buffer | Device | O | Processing  | CL_MEM_WRITE_ONLY | \f$width*height*sizeof\ (cl\_float)\f$ |
     */
    template <>
    class GuidedFilter<GuidedFilterConfig::I_NEQ_P>
    {
    public:
        /*! \brief Enumerates the memory objects handled by the class.
         *  \note `H_*` names refer to staging buffers on the host.
         *  \note `D_*` names refer to buffers on the device.
         */
        enum class Memory : uint8_t
        {
            H_IN_I,    /*!< Input staging buffer for the guidance image. */
            H_IN_P,    /*!< Input staging buffer for the input image. */
            H_OUT,     /*!< Output staging buffer. */
            D_IN_I,    /*!< Input buffer for the guidance image. */
            D_IN_P,    /*!< Input buffer for the input image. */
            D_OUT,     /*!< Output buffer. */
            D_VAR_I,   /*!< Buffer of variance values for the guidance image. */
            D_COV_IP,  /*!< Buffer of covariance values between the guidance and input images. */
            D_A,       /*!< Buffer of \f$ a \f$ coefficients. */
            D_B        /*!< Buffer of \f$ b \f$ coefficients. */
        };

        /*! \brief Configures an OpenCL environment as specified by `_info`. */
        GuidedFilter (clutils::CLEnv &_env, clutils::CLEnvInfo<2> _info);
        /*! \brief Returns a reference to an internal memory object. */
        cl::Memory& get (GuidedFilter::Memory mem);
        /*! \brief Configures kernel execution parameters. */
        void init (unsigned int _width, unsigned int _height, int _radius, float _eps, 
                   int _zero_out = 0, float _scaling = 1e-4f, Staging _staging = Staging::IO);
        /*! \brief Performs a data transfer to a device buffer. */
        void write (GuidedFilter::Memory mem = GuidedFilter::Memory::D_IN_I, void *ptr = nullptr, bool block = CL_FALSE, 
                    const std::vector<cl::Event> *events = nullptr, cl::Event *event = nullptr);
        /*! \brief Performs a data transfer to a staging buffer. */
        void* read (GuidedFilter::Memory mem = GuidedFilter::Memory::H_OUT, bool block = CL_TRUE, 
                    const std::vector<cl::Event> *events = nullptr, cl::Event *event = nullptr);
        /*! \brief Executes the necessary kernels. */
        void run (const std::vector<cl::Event> *events = nullptr, cl::Event *event = nullptr);
        /*! \brief Gets the filter window radius. */
        int getRadius ();
        /*! \brief Sets the filter window radius. */
        void setRadius (int _radius);
        /*! \brief Gets the regularization parameter \f$\epsilon\f$. */
        float getEps ();
        /*! \brief Sets the regularization parameter \f$\epsilon\f$. */
        void setEps (float _eps);
        /*! \brief Gets the scaling factor. */
        float getScaling ();
        /*! \brief Sets the scaling factor. */
        void setScaling (float _scaling);
        /*! \brief Gets the `zero_out` flag. */
        int getZeroing ();
        /*! \brief Sets the `zero_out` flag. */
        void setZeroing (int _zero_out);

        cl_float *hPtrInI;  /*!< Mapping of the input staging buffer for the guidance image. */
        cl_float *hPtrInP;  /*!< Mapping of the input staging buffer for the input image. */
        cl_float *hPtrOut;  /*!< Mapping of the output staging buffer. */

    private:
        clutils::CLEnv &env;
        clutils::CLEnvInfo<2> info;
        cl::Context context;
        cl::CommandQueue queue0;
        BoxFilterSAT mean_I, mean_p, corr_I, corr_Ip, mean_a, mean_b;
        Math::Mult mult_II, mult_Ip;
        cl::Kernel var, ab, q;
        cl::NDRange global;
        Staging staging;
        unsigned int width, height, bufferSize;
        int radius; float eps;
        int zero_out;
        float scaling;
        cl::Buffer hBufferInI, hBufferInP, hBufferOut;
        cl::Buffer dBufferInI, dBufferInP, dBufferOut;
        cl::Buffer dBufferOutVarI, dBufferOutCovIp;
        cl::Buffer dBufferOutA, dBufferOutB;
        cl::Event corrIpEvent, abEvent, mbEvent;
        std::vector<cl::Event> waitListVar, waitListMB, waitListQ;

    public:
        /*! \brief Executes the necessary kernels.
         *  \details This `run` instance is used for profiling.
         *  \note The execution is handled by two separate command queues. The 
         *        time measured is the flat **execution** time of all the kernels. 
         *        There is overlap in the execution of the kernels, but there are 
         *        also gaps between them. As a compromise, and in order to simplify 
         *        things, the execution times of all the kernels are simply added 
         *        together. Look at the `Timeline Trace` of `CodeXL` 
         *        for a more detailed view.
         *  
         *  \param[in] timer `GPUTimer` that does the profiling of the kernel executions.
         *  \param[in] events a wait-list of events.
         *  \return Τhe total execution time measured by the timer.
         */
        template <typename period>
        double run (clutils::GPUTimer<period> &timer, const std::vector<cl::Event> *events = nullptr)
        {
            double pTime;

            pTime = mean_I.run (timer, events);
            pTime += mean_p.run (timer, events);
            pTime += mult_II.run (timer, events);
            pTime += mult_Ip.run (timer, events);
            pTime += corr_I.run (timer);
            pTime += corr_Ip.run (timer);
            
            queue0.enqueueNDRangeKernel (var, cl::NullRange, global, cl::NullRange, nullptr, &timer.event ());
            queue0.flush (); timer.wait ();
            pTime += timer.duration ();

            queue0.enqueueNDRangeKernel (ab, cl::NullRange, global, cl::NullRange, nullptr, &timer.event ());
            queue0.flush (); timer.wait ();
            pTime += timer.duration ();

            pTime += mean_a.run (timer);
            pTime += mean_b.run (timer);
            
            queue0.enqueueNDRangeKernel (q, cl::NullRange, global, cl::NullRange, nullptr, &timer.event ());
            queue0.flush (); timer.wait ();
            pTime += timer.duration ();

            return pTime;
        }

    };


    /*! \brief Offers classes that relate to some kind of processing 
     *         of the `%Kinect` `RGB` and `%Depth` streams.
     */
    namespace Kinect
    {

        /*! \brief Enumerates configurations for the `GuidedFilterRGB` class. */
        enum class GuidedFilterRGBConfig : uint8_t
        {
            INTERLEAVED_FLOAT,  /*!< Identifies the case where the output channels are mixed together in an RGB image of type `float`. */
            SEPARATED           /*!< Identifies the case where the output channels are left separated in independent images. */
        };


        /*! \brief Interface class for performing `Guided Image Filtering` on an RGB image from `%Kinect`.
         *  \details Edge preserving smoothing is done on each of the three channels separately.
         *  \note This is just a declaration. Look at the explicit template specializations
         *        for specific instantiations of the class.
         *        
         *  \tparam C specifies whether to leave the channels separated or interleave them after processing.
         */
        template <GuidedFilterRGBConfig C>
        class GuidedFilterRGB;


        /*! \brief Interface class for performing `Guided Image Filtering` 
         *         on an `RGB` image from `%Kinect`.
         *  \details This instantiation covers the case where the processed channels
         *           are left separated in independent buffers.
         *  \note The value \f$ 0.0001\ (10^{-4}) \f$ is used for scaling in `BoxFilterSAT`.
         *  \note The class creates its own buffers. If you would like to provide 
         *        your own buffers, call `get` to get references to the placeholders 
         *        within the class and assign them to your buffers. You will have to 
         *        do this strictly before the call to `init`. You can also call `get` 
         *        (after the call to `init`) to get a reference to a buffer within 
         *        the class and assign it to another kernel class instance further 
         *        down in your task pipeline.
         *  
         *        The following input/output `OpenCL` memory objects are created by 
         *        a `GuidedFilterRGB<GuidedFilterRGBConfig::SEPARATED>` instance:<br>
         *        |  Name  | Type | Placement | I/O | Use | Properties | Size |
         *        |  ---   |:---: |   :---:   |:---:|:---:|   :---:    |:---: |
         *        | H_IN   | Buffer | Host   | I | Staging     | CL_MEM_READ_WRITE | \f$3*width*height*sizeof\ (cl\_uchar)\f$ |
         *        | H_OUT_R| Buffer | Host   | O | Staging     | CL_MEM_READ_WRITE | \f$  width*height*sizeof\ (cl\_float)\f$ |
         *        | H_OUT_G| Buffer | Host   | O | Staging     | CL_MEM_READ_WRITE | \f$  width*height*sizeof\ (cl\_float)\f$ |
         *        | H_OUT_B| Buffer | Host   | O | Staging     | CL_MEM_READ_WRITE | \f$  width*height*sizeof\ (cl\_float)\f$ |
         *        | D_IN   | Buffer | Device | I | Processing  | CL_MEM_READ_ONLY  | \f$3*width*height*sizeof\ (cl\_uchar)\f$ |
         *        | D_OUT_R| Buffer | Device | O | Processing  | CL_MEM_WRITE_ONLY | \f$  width*height*sizeof\ (cl\_float)\f$ |
         *        | D_OUT_G| Buffer | Device | O | Processing  | CL_MEM_WRITE_ONLY | \f$  width*height*sizeof\ (cl\_float)\f$ |
         *        | D_OUT_B| Buffer | Device | O | Processing  | CL_MEM_WRITE_ONLY | \f$  width*height*sizeof\ (cl\_float)\f$ |
         */
        template <>
        class GuidedFilterRGB<GuidedFilterRGBConfig::SEPARATED>
        {
        public:
            /*! \brief Enumerates the memory objects handled by the class.
             *  \note `H_*` names refer to staging buffers on the host.
             *  \note `D_*` names refer to buffers on the device.
             */
            enum class Memory : uint8_t
            {
                H_IN,      /*!< Input staging buffer. */
                H_OUT_R,   /*!< Output staging buffer for channel R. */
                H_OUT_G,   /*!< Output staging buffer for channel G. */
                H_OUT_B,   /*!< Output staging buffer for channel B. */
                D_IN,      /*!< Input buffer. */
                D_INTR_R,  /*!< Intermediate buffer for input channel R. */
                D_INTR_G,  /*!< Intermediate buffer for input channel G. */
                D_INTR_B,  /*!< Intermediate buffer for input channel B. */
                D_OUT_R,   /*!< Output buffer for channel R. */
                D_OUT_G,   /*!< Output buffer for channel G. */
                D_OUT_B    /*!< Output buffer for channel B. */
            };

            /*! \brief Configures an OpenCL environment as specified by `_info`. */
            GuidedFilterRGB (clutils::CLEnv &_env, clutils::CLEnvInfo<2> _info);
            /*! \brief Returns a reference to an internal memory object. */
            cl::Memory& get (GuidedFilterRGB::Memory mem);
            /*! \brief Configures kernel execution parameters. */
            void init (unsigned int _width, unsigned int _height, 
                       int _radius, float _eps, Staging _staging = Staging::IO);
            /*! \brief Performs a data transfer to a device buffer. */
            void write (GuidedFilterRGB::Memory mem = GuidedFilterRGB::Memory::D_IN, void *ptr = nullptr, 
                        bool block = CL_FALSE, const std::vector<cl::Event> *events = nullptr, cl::Event *event = nullptr);
            /*! \brief Performs a data transfer to a staging buffer. */
            void* read (GuidedFilterRGB::Memory mem = GuidedFilterRGB::Memory::H_OUT_R, bool block = CL_TRUE, 
                        const std::vector<cl::Event> *events = nullptr, cl::Event *event = nullptr);
            /*! \brief Executes the necessary kernels. */
            void run (const std::vector<cl::Event> *events = nullptr, cl::Event *event = nullptr);
            /*! \brief Gets the filter window radius. */
            int getRadius ();
            /*! \brief Sets the filter window radius. */
            void setRadius (int _radius);
            /*! \brief Gets the regularization parameter \f$\epsilon\f$. */
            float getEps ();
            /*! \brief Sets the regularization parameter \f$\epsilon\f$. */
            void setEps (float _eps);

            cl_uchar *hPtrIn;  /*!< Mapping of the input staging buffer. */
            cl_float *hPtrOutR;  /*!< Mapping of the output staging buffer for the R channel. */
            cl_float *hPtrOutG;  /*!< Mapping of the output staging buffer for the G channel. */
            cl_float *hPtrOutB;  /*!< Mapping of the output staging buffer for the B channel. */

        private:
            clutils::CLEnv &env;
            clutils::CLEnvInfo<2> info;
            cl::Context context;
            cl::CommandQueue queue0;
            SeparateRGB<SeparateRGBConfig::UCHAR_FLOAT> sRGB;
            GuidedFilter<GuidedFilterConfig::I_EQ_P> gfR, gfG, gfB;
            Staging staging;
            unsigned int width, height;
            unsigned int bufferInSize, bufferOutSize;
            int radius; float eps;
            cl::Buffer hBufferIn, hBufferOutR, hBufferOutG, hBufferOutB;
            cl::Buffer dBufferIn, dBufferOutR, dBufferOutG, dBufferOutB;
            cl::Event sEvent; std::vector<cl::Event> waitList;

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
                double pTime;

                pTime = sRGB.run (timer, events);
                pTime += gfR.run (timer);
                pTime += gfG.run (timer);
                pTime += gfB.run (timer);

                return pTime;
            }

        };


        /*! \brief Interface class for performing `Guided Image Filtering` 
         *         on an `RGB` image from `%Kinect`.
         *  \details This instantiation covers the case where the processed channels
         *           are finally mixed together in an RGB image of type `float`.
         *  \note The value \f$ 0.0001\ (10^{-4}) \f$ is used for scaling in `BoxFilterSAT`.
         *  \note The class creates its own buffers. If you would like to provide 
         *        your own buffers, call `get` to get references to the placeholders 
         *        within the class and assign them to your buffers. You will have to 
         *        do this strictly before the call to `init`. You can also call `get` 
         *        (after the call to `init`) to get a reference to a buffer within 
         *        the class and assign it to another kernel class instance further 
         *        down in your task pipeline.
         *  
         *        The following input/output `OpenCL` memory objects are created by 
         *        a `GuidedFilterRGB<GuidedFilterRGBConfig::INTERLEAVED_FLOAT>` instance:<br>
         *        | Name  | Type | Placement | I/O | Use | Properties | Size |
         *        |  ---  |:---: |   :---:   |:---:|:---:|   :---:    |:---: |
         *        | H_IN  | Buffer | Host   | I | Staging     | CL_MEM_READ_WRITE | \f$3*width*height*sizeof\ (cl\_uchar)\f$ |
         *        | H_OUT | Buffer | Host   | O | Staging     | CL_MEM_READ_WRITE | \f$3*width*height*sizeof\ (cl\_float)\f$ |
         *        | D_IN  | Buffer | Device | I | Processing  | CL_MEM_READ_ONLY  | \f$3*width*height*sizeof\ (cl\_uchar)\f$ |
         *        | D_OUT | Buffer | Device | O | Processing  | CL_MEM_WRITE_ONLY | \f$3*width*height*sizeof\ (cl\_float)\f$ |
         */
        template <>
        class GuidedFilterRGB<GuidedFilterRGBConfig::INTERLEAVED_FLOAT>
        {
        public:
            /*! \brief Enumerates the memory objects handled by the class.
             *  \note `H_*` names refer to staging buffers on the host.
             *  \note `D_*` names refer to buffers on the device.
             */
            enum class Memory : uint8_t
            {
                H_IN,      /*!< Input staging buffer. */
                H_OUT,     /*!< Output staging buffer. */
                D_IN,      /*!< Input buffer. */
                D_INTR_R,  /*!< Intermediate buffer for input channel R. */
                D_INTR_G,  /*!< Intermediate buffer for input channel G. */
                D_INTR_B,  /*!< Intermediate buffer for input channel B. */
                D_OUT      /*!< Output buffer. */
            };

            /*! \brief Configures an OpenCL environment as specified by `_info`. */
            GuidedFilterRGB (clutils::CLEnv &_env, clutils::CLEnvInfo<2> _info);
            /*! \brief Returns a reference to an internal memory object. */
            cl::Memory& get (GuidedFilterRGB::Memory mem);
            /*! \brief Configures kernel execution parameters. */
            void init (unsigned int _width, unsigned int _height, 
                       int _radius, float _eps, Staging _staging = Staging::IO);
            /*! \brief Performs a data transfer to a device buffer. */
            void write (GuidedFilterRGB::Memory mem = GuidedFilterRGB::Memory::D_IN, void *ptr = nullptr, 
                        bool block = CL_FALSE, const std::vector<cl::Event> *events = nullptr, cl::Event *event = nullptr);
            /*! \brief Performs a data transfer to a staging buffer. */
            void* read (GuidedFilterRGB::Memory mem = GuidedFilterRGB::Memory::H_OUT, bool block = CL_TRUE, 
                        const std::vector<cl::Event> *events = nullptr, cl::Event *event = nullptr);
            /*! \brief Executes the necessary kernels. */
            void run (const std::vector<cl::Event> *events = nullptr, cl::Event *event = nullptr);
            /*! \brief Gets the filter window radius. */
            int getRadius ();
            /*! \brief Sets the filter window radius. */
            void setRadius (int _radius);
            /*! \brief Gets the regularization parameter \f$\epsilon\f$. */
            float getEps ();
            /*! \brief Sets the regularization parameter \f$\epsilon\f$. */
            void setEps (float _eps);

            cl_uchar *hPtrIn;  /*!< Mapping of the input staging buffer. */
            cl_float *hPtrOut;  /*!< Mapping of the output staging buffer. */

        private:
            clutils::CLEnv &env;
            clutils::CLEnvInfo<2> info;
            cl::Context context;
            cl::CommandQueue queue0;
            SeparateRGB<SeparateRGBConfig::UCHAR_FLOAT> sRGB;
            GuidedFilter<GuidedFilterConfig::I_EQ_P> gfR, gfG, gfB;
            CombineRGB<CombineRGBConfig::FLOAT_FLOAT> cRGB;
            Staging staging;
            unsigned int width, height;
            unsigned int bufferInSize, bufferOutSize;
            int radius; float eps;
            cl::Buffer hBufferIn, hBufferOut;
            cl::Buffer dBufferIn, dBufferOut;
            cl::Event sEvent; std::vector<cl::Event> waitList;

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
                double pTime;

                pTime = sRGB.run (timer, events);
                pTime += gfR.run (timer);
                pTime += gfG.run (timer);
                pTime += gfB.run (timer);
                pTime = cRGB.run (timer);

                return pTime;
            }

        };


        /*! \brief Interface class for performing `Guided Image Filtering` 
         *         on a `%Depth` image from `%Kinect`.
         *  \note The value \f$ 0.000001\ (10^{-6}) \f$ is used for scaling in `BoxFilterSAT`.
         *  \note The class creates its own buffers. If you would like to provide 
         *        your own buffers, call `get` to get references to the placeholders 
         *        within the class and assign them to your buffers. You will have to 
         *        do this strictly before the call to `init`. You can also call `get` 
         *        (after the call to `init`) to get a reference to a buffer within 
         *        the class and assign it to another kernel class instance further 
         *        down in your task pipeline.
         *  
         *        The following input/output `OpenCL` memory objects are created by a `GuidedFilterDepth` instance:<br>
         *        | Name  | Type | Placement | I/O | Use | Properties | Size |
         *        |  ---  |:---: |   :---:   |:---:|:---:|   :---:    |:---: |
         *        | H_IN  | Buffer | Host   | I | Staging     | CL_MEM_READ_WRITE | \f$width*height*sizeof\ (cl\_ushort)\f$ |
         *        | H_OUT | Buffer | Host   | O | Staging     | CL_MEM_READ_WRITE | \f$width*height*sizeof\ (cl\_float)\f$  |
         *        | D_IN  | Buffer | Device | I | Processing  | CL_MEM_READ_ONLY  | \f$width*height*sizeof\ (cl\_ushort)\f$ |
         *        | D_OUT | Buffer | Device | O | Processing  | CL_MEM_WRITE_ONLY | \f$width*height*sizeof\ (cl\_float)\f$  |
         */
        class GuidedFilterDepth
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
            GuidedFilterDepth (clutils::CLEnv &_env, clutils::CLEnvInfo<2> _info);
            /*! \brief Returns a reference to an internal memory object. */
            cl::Memory& get (GuidedFilterDepth::Memory mem);
            /*! \brief Configures kernel execution parameters. */
            void init (unsigned int _width, unsigned int _height, 
                       int _radius, float _eps, float _dScaling = 1e-3f, Staging _staging = Staging::IO);
            /*! \brief Performs a data transfer to a device buffer. */
            void write (GuidedFilterDepth::Memory mem = GuidedFilterDepth::Memory::D_IN, void *ptr = nullptr, 
                        bool block = CL_FALSE, const std::vector<cl::Event> *events = nullptr, cl::Event *event = nullptr);
            /*! \brief Performs a data transfer to a staging buffer. */
            void* read (GuidedFilterDepth::Memory mem = GuidedFilterDepth::Memory::H_OUT, bool block = CL_TRUE, 
                        const std::vector<cl::Event> *events = nullptr, cl::Event *event = nullptr);
            /*! \brief Executes the necessary kernels. */
            void run (const std::vector<cl::Event> *events = nullptr, cl::Event *event = nullptr);
            /*! \brief Gets the filter window radius. */
            int getRadius ();
            /*! \brief Sets the filter window radius. */
            void setRadius (int _radius);
            /*! \brief Gets the regularization parameter \f$\epsilon\f$. */
            float getEps ();
            /*! \brief Sets the regularization parameter \f$\epsilon\f$. */
            void setEps (float _eps);
            /*! \brief Gets the depth scaling factor. */
            float getDScaling ();
            /*! \brief Sets the depth scaling factor. */
            void setDScaling (float _dScaling);

            cl_ushort *hPtrIn;  /*!< Mapping of the input staging buffer. */
            cl_float *hPtrOut;  /*!< Mapping of the output staging buffer. */

        private:
            clutils::CLEnv &env;
            clutils::CLEnvInfo<2> info;
            cl::Context context;
            cl::CommandQueue queue0;
            Depth<DepthConfig::USHORT_FLOAT> depth;
            GuidedFilter<GuidedFilterConfig::I_EQ_P> gf;
            Staging staging;
            unsigned int width, height;
            unsigned int bufferInSize, bufferOutSize;
            int radius; float eps; float dScaling;
            cl::Buffer hBufferIn, hBufferOut;
            cl::Buffer dBufferIn, dBufferOut;
            cl::Event dEvent; std::vector<cl::Event> waitList;

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
                double pTime;

                pTime = depth.run (timer, events);
                pTime += gf.run (timer);

                return pTime;
            }

        };

    }

}
}

#endif  // GF_ALGORITHMS_HPP