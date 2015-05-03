/*! \file testsImageSupport.cpp
 *  \brief Google Test Unit Tests for the Image Support kernels.
 *  \note Use the `--profiling` flag to enable profiling of the kernels.
 *  \note The benchmarks in these tests are against naive CPU implementations 
 *        of the associated algorithms. They are used only for testing purposes, 
 *        and not for examining the performance of their GPU alternatives.
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
#include <vector>
#include <string>
#include <algorithm>
#include <numeric>
#include <chrono>
#include <random>
#include <limits>
#include <cmath>
#include <gtest/gtest.h>
#include <CLUtils.hpp>
#include <GuidedFilter/algorithms.hpp>
#include <GuidedFilter/tests/helperFuncs.hpp>


// Kernel filenames
const std::string kernel_filename_img { "kernels/imageSupport_kernels.cl" };

// Uniform random number generators
extern std::function<unsigned char ()> rNum_0_255;
extern std::function<unsigned short ()> rNum_0_10000;
extern std::function<float ()> rNum_R_0_1;

bool profiling;  // Flag to enable profiling of the kernels


/*! \brief Tests the **separateRGBChannels_Float2Float** kernel.
 *  \details The operation is a matrix transposition.
 */
TEST (ImageSupport, separateRGBChannels_Float2Float)
{
    try
    {
        const unsigned int width = 640, height = 480;
        const unsigned int pixels = width * height;
        const unsigned int bufferInSize = 3 * width * height * sizeof (cl_float);
        const unsigned int bufferOutSize = width * height * sizeof (cl_float);

        // Setup the OpenCL environment
        clutils::CLEnv clEnv;
        clEnv.addContext (0);
        clEnv.addQueue (0, 0, CL_QUEUE_PROFILING_ENABLE);
        clEnv.addProgram (0, kernel_filename_img);
        
        // Configure kernel execution parameters
        clutils::CLEnvInfo<1> info (0, 0, 0, { 0 }, 0);
        const cl_algo::GF::SeparateRGBConfig C = cl_algo::GF::SeparateRGBConfig::FLOAT_FLOAT;
        cl_algo::GF::SeparateRGB<C> sRGB (clEnv, info);
        sRGB.init (width, height);

        // Initialize data (writes on staging buffer directly)
        std::generate (sRGB.hPtrIn, sRGB.hPtrIn + bufferInSize / sizeof (cl_float), rNum_0_255);
        // printBuffer ("Original:", sRGB.hPtrIn, 3, width * height);
        
        sRGB.write ();  // Copy data to device

        sRGB.run ();  // Execute kernels (74 us)
        
        // Copy results to host
        cl_float *R = (cl_float *) sRGB.read (cl_algo::GF::SeparateRGB<C>::Memory::H_OUT_R, CL_FALSE);
        cl_float *G = (cl_float *) sRGB.read (cl_algo::GF::SeparateRGB<C>::Memory::H_OUT_G, CL_FALSE);
        cl_float *B = (cl_float *) sRGB.read (cl_algo::GF::SeparateRGB<C>::Memory::H_OUT_B);
        // printBufferF ("Received R:", R, width, height, 1);
        // printBufferF ("Received G:", G, width, height, 1);
        // printBufferF ("Received B:", B, width, height, 1);

        // Produce reference transposed image
        cl_float *refR = new cl_float[pixels];
        cl_float *refG = new cl_float[pixels];
        cl_float *refB = new cl_float[pixels];
        cpuSeparateRGB (sRGB.hPtrIn, refR, refG, refB, pixels);
        // printBuffer ("Expected R:", refR, width, height);
        // printBuffer ("Expected G:", refG, width, height);
        // printBuffer ("Expected B:", refB, width, height);

        // Verify the **transposed** array
        for (uint i = 0; i < pixels; ++i)
        {
            ASSERT_EQ (refR[i], R[i]);
            ASSERT_EQ (refG[i], G[i]);
            ASSERT_EQ (refB[i], B[i]);
        }

        // Profiling ===========================================================
        if (profiling)
        {
            const int nRepeat = 1;  /* Number of times to perform the tests. */

            // CPU
            clutils::CPUTimer<double, std::milli> cTimer;
            clutils::ProfilingInfo<nRepeat> pCPU ("CPU");
            for (int i = 0; i < nRepeat; ++i)
            {
                cTimer.start ();
                cpuSeparateRGB (sRGB.hPtrIn, refR, refG, refB, pixels);
                pCPU[i] = cTimer.stop ();
            }
            
            // GPU
            clutils::GPUTimer<std::milli> gTimer (clEnv.devices[0][0]);
            clutils::ProfilingInfo<nRepeat> pGPU ("GPU");
            for (int i = 0; i < nRepeat; ++i)
                pGPU[i] = sRGB.run (gTimer);

            // Benchmark
            pGPU.print (pCPU, "SeparateRGB<SeparateRGBConfig::FLOAT_FLOAT>");
        }

    }
    catch (const cl::Error &error)
    {
        std::cerr << error.what ()
                  << " (" << clutils::getOpenCLErrorCodeString (error.err ()) 
                  << ")"  << std::endl;
        exit (EXIT_FAILURE);
    }
}


/*! \brief Tests the **separateRGBChannels_Uchar2Float** kernel.
 *  \details The operation is a matrix transposition.
 */
TEST (ImageSupport, separateRGBChannels_Uchar2Float)
{
    try
    {
        const unsigned int width = 640, height = 480;
        const unsigned int pixels = width * height;
        const unsigned int bufferInSize = 3 * pixels * sizeof (cl_uchar);
        const unsigned int bufferOutSize = pixels * sizeof (cl_float);

        // Setup the OpenCL environment
        clutils::CLEnv clEnv;
        clEnv.addContext (0);
        clEnv.addQueue (0, 0, CL_QUEUE_PROFILING_ENABLE);
        clEnv.addProgram (0, kernel_filename_img);
        
        // Configure kernel execution parameters
        clutils::CLEnvInfo<1> info (0, 0, 0, { 0 }, 0);
        const cl_algo::GF::SeparateRGBConfig C = cl_algo::GF::SeparateRGBConfig::UCHAR_FLOAT;
        cl_algo::GF::SeparateRGB<C> sRGB (clEnv, info);
        sRGB.init (width, height);

        // Initialize data (writes on staging buffer directly)
        std::generate (sRGB.hPtrIn, sRGB.hPtrIn + bufferInSize / sizeof (cl_uchar), rNum_0_255);
        // printBuffer ("Original:", sRGB.hPtrIn, 3, width * height);
        
        sRGB.write ();  // Copy data to device

        sRGB.run ();  // Execute kernels (81 us)
        
        // Copy results to host
        cl_float *R = (cl_float *) sRGB.read (cl_algo::GF::SeparateRGB<C>::Memory::H_OUT_R, CL_FALSE);
        cl_float *G = (cl_float *) sRGB.read (cl_algo::GF::SeparateRGB<C>::Memory::H_OUT_G, CL_FALSE);
        cl_float *B = (cl_float *) sRGB.read (cl_algo::GF::SeparateRGB<C>::Memory::H_OUT_B);
        // printBufferF ("Received R:", R, width, height, 1);
        // printBufferF ("Received G:", G, width, height, 1);
        // printBufferF ("Received B:", B, width, height, 1);

        // Produce reference transposed image
        cl_float *refR = new float[pixels];
        cl_float *refG = new float[pixels];
        cl_float *refB = new float[pixels];
        cpuSeparateRGB_N_Norm (sRGB.hPtrIn, refR, refG, refB, pixels);
        // printBufferF ("Expected R:", refR, width, height, 1);
        // printBufferF ("Expected G:", refG, width, height, 1);
        // printBufferF ("Expected B:", refB, width, height, 1);

        // Verify the **transposed** array
        float eps = 42 * std::numeric_limits<float>::epsilon ();  // 5.00679e-06
        for (uint i = 0; i < pixels; ++i)
        {
            ASSERT_LT (std::abs (refR[i] - R[i]), eps);
            ASSERT_LT (std::abs (refG[i] - G[i]), eps);
            ASSERT_LT (std::abs (refB[i] - B[i]), eps);
        }

        // Profiling ===========================================================
        if (profiling)
        {
            const int nRepeat = 1;  /* Number of times to perform the tests. */

            // CPU
            clutils::CPUTimer<double, std::milli> cTimer;
            clutils::ProfilingInfo<nRepeat> pCPU ("CPU");
            for (int i = 0; i < nRepeat; ++i)
            {
                cTimer.start ();
                cpuSeparateRGB_N_Norm (sRGB.hPtrIn, refR, refG, refB, pixels);
                pCPU[i] = cTimer.stop ();
            }
            
            // GPU
            clutils::GPUTimer<std::milli> gTimer (clEnv.devices[0][0]);
            clutils::ProfilingInfo<nRepeat> pGPU ("GPU");
            for (int i = 0; i < nRepeat; ++i)
                pGPU[i] = sRGB.run (gTimer);

            // Benchmark
            pGPU.print (pCPU, "SeparateRGB<SeparateRGBConfig::UCHAR_FLOAT>");
        }

    }
    catch (const cl::Error &error)
    {
        std::cerr << error.what ()
                  << " (" << clutils::getOpenCLErrorCodeString (error.err ()) 
                  << ")"  << std::endl;
        exit (EXIT_FAILURE);
    }
}


/*! \brief Tests the **combineRGBChannels_Float2Float** kernel.
 *  \details The operation is a matrix transposition.
 */
TEST (ImageSupport, combineRGBChannels_Float2Float)
{
    try
    {
        const unsigned int width = 640, height = 480;
        const unsigned int pixels = width * height;
        const unsigned int bufferInSize = pixels * sizeof (cl_float);
        const unsigned int bufferOutSize = 3 * pixels * sizeof (cl_float);

        // Setup the OpenCL environment
        clutils::CLEnv clEnv;
        clEnv.addContext (0);
        clEnv.addQueue (0, 0, CL_QUEUE_PROFILING_ENABLE);
        clEnv.addProgram (0, kernel_filename_img);

        // Configure kernel execution parameters
        clutils::CLEnvInfo<1> info (0, 0, 0, { 0 }, 0);
        const cl_algo::GF::CombineRGBConfig C = cl_algo::GF::CombineRGBConfig::FLOAT_FLOAT;
        cl_algo::GF::CombineRGB<C> cRGB (clEnv, info);
        cRGB.init (width, height);

        // Initialize data (writes on staging buffer directly)
        std::generate (cRGB.hPtrInR, cRGB.hPtrInR + pixels, rNum_0_255);
        std::generate (cRGB.hPtrInG, cRGB.hPtrInG + pixels, rNum_0_255);
        std::generate (cRGB.hPtrInB, cRGB.hPtrInB + pixels, rNum_0_255);
        // printBuffer ("Original R:", cRGB.hPtrInR, width, height);
        // printBuffer ("Original G:", cRGB.hPtrInG, width, height);
        // printBuffer ("Original B:", cRGB.hPtrInB, width, height);
        
        // Copy data to device
        cRGB.write (cl_algo::GF::CombineRGB<C>::Memory::D_IN_R);
        cRGB.write (cl_algo::GF::CombineRGB<C>::Memory::D_IN_G);
        cRGB.write (cl_algo::GF::CombineRGB<C>::Memory::D_IN_B);

        cRGB.run ();  // Execute kernels (82 us)
        
        cl_float *results = (cl_float *) cRGB.read ();  // Copy results to host
        // printBuffer ("Received:", results, 3, width * height);

        // Produce reference transposed image
        cl_float *refCombRGB = new cl_float[3 * pixels];
        cpuCombineRGB (cRGB.hPtrInR, cRGB.hPtrInG, cRGB.hPtrInB, refCombRGB, pixels);
        // printBuffer ("Expected:", refCombRGB, 3, width * height);

        // Verify the **transposed** array
        for (uint i = 0; i < pixels; ++i)
        {
            ASSERT_EQ (refCombRGB[i * 3],     results[i * 3]);
            ASSERT_EQ (refCombRGB[i * 3 + 1], results[i * 3 + 1]);
            ASSERT_EQ (refCombRGB[i * 3 + 2], results[i * 3 + 2]);
        }

        // Profiling ===========================================================
        if (profiling)
        {
            const int nRepeat = 1;  /* Number of times to perform the tests. */

            // CPU
            clutils::CPUTimer<double, std::milli> cTimer;
            clutils::ProfilingInfo<nRepeat> pCPU ("CPU");
            for (int i = 0; i < nRepeat; ++i)
            {
                cTimer.start ();
                cpuCombineRGB (cRGB.hPtrInR, cRGB.hPtrInG, cRGB.hPtrInB, refCombRGB, pixels);
                pCPU[i] = cTimer.stop ();
            }
            
            // GPU
            clutils::GPUTimer<std::milli> gTimer (clEnv.devices[0][0]);
            clutils::ProfilingInfo<nRepeat> pGPU ("GPU");
            for (int i = 0; i < nRepeat; ++i)
                pGPU[i] = cRGB.run (gTimer);

            // Benchmark
            pGPU.print (pCPU, "CombineRGB<CombineRGBConfig::FLOAT_FLOAT>");
        }

    }
    catch (const cl::Error &error)
    {
        std::cerr << error.what ()
                  << " (" << clutils::getOpenCLErrorCodeString (error.err ()) 
                  << ")"  << std::endl;
        exit (EXIT_FAILURE);
    }
}


/*! \brief Tests the **combineRGBChannels_Float2Uchar** kernel.
 *  \details The operation is a matrix transposition.
 */
TEST (ImageSupport, combineRGBChannels_Float2Uchar)
{
    try
    {
        const unsigned int width = 640, height = 480;
        const unsigned int pixels = width * height;
        const unsigned int bufferInSize = width * height * sizeof (cl_float);
        const unsigned int bufferOutSize = 3 * width * height * sizeof (cl_uchar);

        // Setup the OpenCL environment
        clutils::CLEnv clEnv;
        clEnv.addContext (0);
        clEnv.addQueue (0, 0, CL_QUEUE_PROFILING_ENABLE);
        clEnv.addProgram (0, kernel_filename_img);

        // Configure kernel execution parameters
        clutils::CLEnvInfo<1> info (0, 0, 0, { 0 }, 0);
        const cl_algo::GF::CombineRGBConfig C = cl_algo::GF::CombineRGBConfig::FLOAT_UCHAR;
        cl_algo::GF::CombineRGB<C> cRGB (clEnv, info);
        cRGB.init (width, height);

        // Initialize data (writes on staging buffer directly)
        std::generate (cRGB.hPtrInR, cRGB.hPtrInR + pixels, rNum_R_0_1);
        std::generate (cRGB.hPtrInG, cRGB.hPtrInG + pixels, rNum_R_0_1);
        std::generate (cRGB.hPtrInB, cRGB.hPtrInB + pixels, rNum_R_0_1);
        // printBufferF ("Original R:", cRGB.hPtrInR, width, height, 0);
        // printBufferF ("Original G:", cRGB.hPtrInG, width, height, 0);
        // printBufferF ("Original B:", cRGB.hPtrInB, width, height, 0);
        
        // Copy data to device
        cRGB.write (cl_algo::GF::CombineRGB<C>::Memory::D_IN_R);
        cRGB.write (cl_algo::GF::CombineRGB<C>::Memory::D_IN_G);
        cRGB.write (cl_algo::GF::CombineRGB<C>::Memory::D_IN_B);

        cRGB.run ();  // Execute kernels (82 us)
        
        cl_uchar *results = (cl_uchar *) cRGB.read ();  // Copy results to host
        // printBuffer ("Received:", results, 3, width * height);

        // Produce reference transposed image
        cl_uchar *refCombRGB = new cl_uchar[3 * pixels];
        cpuTranspose_N_Scale (cRGB.hPtrInR, cRGB.hPtrInG, cRGB.hPtrInB, refCombRGB, pixels);
        // printBuffer ("Expected:", refCombRGB, 3, width * height);

        // Verify the **transposed** array
        for (uint i = 0; i < pixels; ++i)
        {
            ASSERT_LE (std::abs (refCombRGB[i * 3]     - results[i * 3]), 0);
            ASSERT_LE (std::abs (refCombRGB[i * 3 + 1] - results[i * 3 + 1]), 0);
            ASSERT_LE (std::abs (refCombRGB[i * 3 + 2] - results[i * 3 + 2]), 0);
        }

        // Profiling ===========================================================
        if (profiling)
        {
            const int nRepeat = 1;  /* Number of times to perform the tests. */

            // CPU
            clutils::CPUTimer<double, std::milli> cTimer;
            clutils::ProfilingInfo<nRepeat> pCPU ("CPU");
            for (int i = 0; i < nRepeat; ++i)
            {
                cTimer.start ();
                cpuTranspose_N_Scale (cRGB.hPtrInR, cRGB.hPtrInG, cRGB.hPtrInB, refCombRGB, pixels);
                pCPU[i] = cTimer.stop ();
            }
            
            // GPU
            clutils::GPUTimer<std::milli> gTimer (clEnv.devices[0][0]);
            clutils::ProfilingInfo<nRepeat> pGPU ("GPU");
            for (int i = 0; i < nRepeat; ++i)
                pGPU[i] = cRGB.run (gTimer);

            // Benchmark
            pGPU.print (pCPU, "CombineRGB<CombineRGBConfig::FLOAT_UCHAR>");
        }

    }
    catch (const cl::Error &error)
    {
        std::cerr << error.what ()
                  << " (" << clutils::getOpenCLErrorCodeString (error.err ()) 
                  << ")"  << std::endl;
        exit (EXIT_FAILURE);
    }
}


/*! \brief Tests the **depth_Ushort2Float** kernel.
 *  \details The operation is a type promotion from `ushort` to `float`.
 */
TEST (ImageSupport, depth_Ushort2Float)
{
    try
    {
        const unsigned int width = 640, height = 480;
        const unsigned int bufferInSize = width * height * sizeof (cl_ushort);
        const unsigned int bufferOutSize = width * height * sizeof (cl_float);

        // Setup the OpenCL environment
        clutils::CLEnv clEnv;
        clEnv.addContext (0);
        clEnv.addQueue (0, 0, CL_QUEUE_PROFILING_ENABLE);
        clEnv.addProgram (0, kernel_filename_img);

        // Configure kernel execution parameters
        clutils::CLEnvInfo<1> info (0, 0, 0, { 0 }, 0);
        cl_algo::GF::Depth<cl_algo::GF::DepthConfig::USHORT_FLOAT> df (clEnv, info);
        df.init (width, height);
        
        // Initialize data (writes on staging buffer directly)
        std::generate (df.hPtrIn, df.hPtrIn + bufferInSize / sizeof (cl_ushort), rNum_0_10000);
        // printBuffer ("Original:", df.hPtrIn, width, height);
        
        df.write ();  // Copy data to device

        df.run ();  // Execute kernels (14 us)
        
        // Copy results to host
        cl_float *results = (cl_float *) df.read ();

        // Produce reference point cloud
        cl_float refDepth[width * height];
        for (uint i = 0; i < width * height; ++i) refDepth[i] = df.hPtrIn[i];

        // Verify point cloud
        float eps = 1 * std::numeric_limits<float>::epsilon ();  // 1.19209e-07
        for (uint row = 0; row < height; ++row)
            for (uint col = 0; col < width; ++col)
                ASSERT_LT (std::abs (refDepth[row * width + col] - results[row * width + col]), eps);

        // Profiling ===========================================================
        if (profiling)
        {
            const int nRepeat = 1;  /* Number of times to perform the tests. */

            // CPU
            clutils::CPUTimer<double, std::milli> cTimer;
            clutils::ProfilingInfo<nRepeat> pCPU ("CPU");
            for (int i = 0; i < nRepeat; ++i)
            {
                cTimer.start ();
                for (uint i = 0; i < width * height; ++i) refDepth[i] = df.hPtrIn[i];
                pCPU[i] = cTimer.stop ();
            }

            // GPU
            clutils::GPUTimer<std::milli> gTimer (clEnv.devices[0][0]);
            clutils::ProfilingInfo<nRepeat> pGPU ("GPU");
            for (int i = 0; i < nRepeat; ++i)
                pGPU[i] = df.run (gTimer);

            // Benchmark
            pGPU.print (pCPU, "Depth<DepthConfig::USHORT_FLOAT>");
        }

    }
    catch (const cl::Error &error)
    {
        std::cerr << error.what ()
                  << " (" << clutils::getOpenCLErrorCodeString (error.err ()) 
                  << ")"  << std::endl;
        exit (EXIT_FAILURE);
    }
}


/*! \brief Tests the **depthTo3D** kernel.
 *  \details The operation is a transformation from the image plane 
 *           to the world coordinates (w.r.t. camera frame).
 */
TEST (ImageSupport, depthTo3D)
{
    try
    {
        const float f = 595.f;  // focal length (for Kinect)
        const unsigned int width = 640, height = 480;
        const unsigned int nPoints = width * height;
        const unsigned int bufferInSize = width * height * sizeof (cl_float);
        const unsigned int bufferOutSize = width * height * sizeof (cl_float4);

        // Setup the OpenCL environment
        clutils::CLEnv clEnv;
        clEnv.addContext (0);
        clEnv.addQueue (0, 0, CL_QUEUE_PROFILING_ENABLE);
        clEnv.addProgram (0, kernel_filename_img);

        // Configure kernel execution parameters
        clutils::CLEnvInfo<1> info (0, 0, 0, { 0 }, 0);
        cl_algo::GF::DepthTo3D d3 (clEnv, info);
        d3.init (width, height, f);
        
        // Initialize data (writes on staging buffer directly)
        std::generate (d3.hPtrIn, d3.hPtrIn + nPoints, rNum_0_10000);
        // printBuffer ("Original:", d3.hPtrIn, width, height);
        
        d3.write ();  // Copy data to device

        d3.run ();  // Execute kernels (45 us)
        
        // Copy results to host
        cl_float4 *results = (cl_float4 *) d3.read ();

        // Produce reference point cloud
        cl_float4 *refPCloud = new cl_float4[width * height];
        cpuDepthTo3D (d3.hPtrIn, refPCloud, width, height, f);

        // Verify point cloud
        float eps = 4200 * std::numeric_limits<float>::epsilon ();  // 0.000500679
        for (uint row = 0; row < height; ++row)
        {
            for (uint col = 0; col < width; ++col)
            {   
                cl_float *dValue = (cl_float *) &results[row * width + col];
                cl_float *hValue = (cl_float *) &refPCloud[row * width + col];
                
                ASSERT_LT (std::abs (hValue[0] - dValue[0]), eps);
                ASSERT_LT (std::abs (hValue[1] - dValue[1]), eps);
                ASSERT_LT (std::abs (hValue[2] - dValue[2]), eps);
            }
        }

        // Profiling ===========================================================
        if (profiling)
        {
            const int nRepeat = 1;  /* Number of times to perform the tests. */

            // CPU
            clutils::CPUTimer<double, std::milli> cTimer;
            clutils::ProfilingInfo<nRepeat> pCPU ("CPU");
            for (int i = 0; i < nRepeat; ++i)
            {
                cTimer.start ();
                cpuDepthTo3D (d3.hPtrIn, refPCloud, width, height, f);
                pCPU[i] = cTimer.stop ();
            }

            // GPU
            clutils::GPUTimer<std::milli> gTimer (clEnv.devices[0][0]);
            clutils::ProfilingInfo<nRepeat> pGPU ("GPU");
            for (int i = 0; i < nRepeat; ++i)
                pGPU[i] = d3.run (gTimer);

            // Benchmark
            pGPU.print (pCPU, "DepthTo3D");
        }

    }
    catch (const cl::Error &error)
    {
        std::cerr << error.what ()
                  << " (" << clutils::getOpenCLErrorCodeString (error.err ()) 
                  << ")"  << std::endl;
        exit (EXIT_FAILURE);
    }
}


/*! \brief Tests the **rgbNorm** kernel.
 *  \details The operation is an RGB vector approximate normalization.
 */
TEST (ImageSupport, rgbNorm)
{
    try
    {
        const unsigned int width = 640, height = 480;
        const unsigned int bufferSize = 3 * width * height * sizeof (cl_float);

        // Setup the OpenCL environment
        clutils::CLEnv clEnv;
        clEnv.addContext (0);
        clEnv.addQueue (0, 0, CL_QUEUE_PROFILING_ENABLE);
        clEnv.addProgram (0, kernel_filename_img);

        // Configure kernel execution parameters
        clutils::CLEnvInfo<1> info (0, 0, 0, { 0 }, 0);
        cl_algo::GF::RGBNorm norm (clEnv, info);
        norm.init (width, height);

        // Initialize data (writes on staging buffer directly)
        std::generate (norm.hPtrIn, norm.hPtrIn + bufferSize / sizeof (cl_float), rNum_0_255);
        // printBuffer ("Original:", norm.hPtrIn, 3, width * height);
        
        norm.write ();  // Copy data to device

        norm.run ();  // Execute kernels (50 us)
        
        cl_float *results = (cl_float *) norm.read ();  // Copy results to host
        // printBuffer ("Received:", results, 3, width * height);

        // Produce reference RGB normalized image
        cl_float *refRGBNorm = new cl_float[bufferSize];
        cpuRGBNorm (norm.hPtrIn, refRGBNorm, width, height);
        // printBuffer ("Expected:", refRGBNorm, 3, width * height);

        // Verify the normalized image
        float eps = 420 * std::numeric_limits<float>::epsilon ();  // 5.00679e-05
        for (uint row = 0; row < height; ++row)
        {
            for (uint col = 0; col < width; ++col)
            {   
                uint rank = (row * width + col) * 3;

                ASSERT_LE (std::abs (refRGBNorm[rank]   - results[rank]), eps);
                ASSERT_LE (std::abs (refRGBNorm[rank+1] - results[rank+1]), eps);
                ASSERT_LE (std::abs (refRGBNorm[rank+2] - results[rank+2]), eps);
            }
        }

        // Profiling ===========================================================
        if (profiling)
        {
            const int nRepeat = 1;  /* Number of times to perform the tests. */

            // CPU
            clutils::CPUTimer<double, std::milli> cTimer;
            clutils::ProfilingInfo<nRepeat> pCPU ("CPU");
            for (int i = 0; i < nRepeat; ++i)
            {
                cTimer.start ();
                cpuRGBNorm (norm.hPtrIn, refRGBNorm, width, height);
                pCPU[i] = cTimer.stop ();
            }
            
            // GPU
            clutils::GPUTimer<std::milli> gTimer (clEnv.devices[0][0]);
            clutils::ProfilingInfo<nRepeat> pGPU ("GPU");
            for (int i = 0; i < nRepeat; ++i)
                pGPU[i] = norm.run (gTimer);

            // Benchmark
            pGPU.print (pCPU, "RGBNorm");
        }

    }
    catch (const cl::Error &error)
    {
        std::cerr << error.what ()
                  << " (" << clutils::getOpenCLErrorCodeString (error.err ()) 
                  << ")"  << std::endl;
        exit (EXIT_FAILURE);
    }
}


/*! \brief Tests the **rgbdTo8D** kernel.
 *  \details The kernel fuses geometry and color values into 8D feature points.
 */
TEST (ImageSupport, rgbdTo8D)
{
    try
    {
        const float f = 595.f;  // focal length (for Kinect)
        const unsigned int width = 640, height = 480;
        const unsigned int points = width * height;
        const unsigned int bufferInSize = points * sizeof (cl_float);
        const unsigned int bufferOutSize = points * sizeof (cl_float8);

        // Setup the OpenCL environment
        clutils::CLEnv clEnv;
        clEnv.addContext (0);
        clEnv.addQueue (0, 0, CL_QUEUE_PROFILING_ENABLE);
        clEnv.addProgram (0, kernel_filename_img);

        // Configure kernel execution parameters
        clutils::CLEnvInfo<1> info (0, 0, 0, { 0 }, 0);
        cl_algo::GF::RGBDTo8D to8D (clEnv, info);
        to8D.init (width, height, f);

        // Initialize data (writes on staging buffer directly)
        std::generate (to8D.hPtrInD, to8D.hPtrInD + points, rNum_0_10000);
        std::generate (to8D.hPtrInR, to8D.hPtrInR + points, rNum_R_0_1);
        std::generate (to8D.hPtrInG, to8D.hPtrInG + points, rNum_R_0_1);
        std::generate (to8D.hPtrInB, to8D.hPtrInB + points, rNum_R_0_1);
        // printBufferF ("Original D:", to8D.hPtrInD, width, height, 1);
        // printBufferF ("Original R:", to8D.hPtrInR, width, height, 1);
        // printBufferF ("Original G:", to8D.hPtrInG, width, height, 1);
        // printBufferF ("Original B:", to8D.hPtrInB, width, height, 1);
        
        // Copy data to device
        to8D.write (cl_algo::GF::RGBDTo8D::Memory::D_IN_D);
        to8D.write (cl_algo::GF::RGBDTo8D::Memory::D_IN_R);
        to8D.write (cl_algo::GF::RGBDTo8D::Memory::D_IN_G);
        to8D.write (cl_algo::GF::RGBDTo8D::Memory::D_IN_B);

        to8D.run ();  // Execute kernels (121 us)
        
        cl_float8 *results = (cl_float8 *) to8D.read ();  // Copy results to host
        // printBufferF ("Received:", (cl_float *) results, 8, points, 1);

        // Produce reference 8D feature points
        cl_float8 *ref8D = new cl_float8[points];
        cpuRGBDTo8D (to8D.hPtrInD, to8D.hPtrInR, to8D.hPtrInG, to8D.hPtrInB, ref8D, width, height, f);
        // printBufferF ("Expected:", (cl_float *) ref8D, 8, points, 1);

        // Verify the array of 8D feature points
        float eps = 4200 * std::numeric_limits<float>::epsilon ();  // 0.000500679
        for (uint i = 0; i < points; ++i)
        {
            cl_float *refPoint = (cl_float *) &ref8D[i];
            cl_float *gpuPoint = (cl_float *) &results[i];

            for (uint j = 0; j < 8; ++j)
                ASSERT_LT (std::abs (refPoint[j] - gpuPoint[j]), eps);
        }

        // Profiling ===========================================================
        if (profiling)
        {
            const int nRepeat = 1;  /* Number of times to perform the tests. */

            // CPU
            clutils::CPUTimer<double, std::milli> cTimer;
            clutils::ProfilingInfo<nRepeat> pCPU ("CPU");
            for (int i = 0; i < nRepeat; ++i)
            {
                cTimer.start ();
                cpuRGBDTo8D (to8D.hPtrInD, to8D.hPtrInR, to8D.hPtrInG, to8D.hPtrInB, ref8D, width, height, f);
                pCPU[i] = cTimer.stop ();
            }
            
            // GPU
            clutils::GPUTimer<std::milli> gTimer (clEnv.devices[0][0]);
            clutils::ProfilingInfo<nRepeat> pGPU ("GPU");
            for (int i = 0; i < nRepeat; ++i)
                pGPU[i] = to8D.run (gTimer);

            // Benchmark
            pGPU.print (pCPU, "rgbdTo8D");
        }

    }
    catch (const cl::Error &error)
    {
        std::cerr << error.what ()
                  << " (" << clutils::getOpenCLErrorCodeString (error.err ()) 
                  << ")"  << std::endl;
        exit (EXIT_FAILURE);
    }
}


int main (int argc, char **argv)
{
    profiling = setProfilingFlag (argc, argv);

    ::testing::InitGoogleTest (&argc, argv);

    return RUN_ALL_TESTS ();
}
