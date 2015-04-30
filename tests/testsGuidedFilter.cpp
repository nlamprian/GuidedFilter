/*! \file testsGuidedFilter.cpp
 *  \brief Google Test Unit Tests for the Guided Filtering pipelines.
 *  \note Use the `--profiling` flag to enable profiling of the kernels.
 *  \note The benchmarks in these tests are against naive CPU implementations 
 *        of the associated algorithms. They are used only for testing purposes, 
 *        and not for examining the performance of their GPU alternatives.
 *  \author Nick Lamprianidis
 *  \version 1.1
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
#include <chrono>
#include <random>
#include <limits>
#include <cmath>
#include <gtest/gtest.h>
#include <CLUtils.hpp>
#include <GuidedFilter/algorithms.hpp>
#include <GuidedFilter/tests/helperFuncs.hpp>


// Kernel filenames
const std::string kernel_filename_img  { "kernels/imageSupport_kernels.cl" };
const std::string kernel_filename_scan { "kernels/prefixSum_kernels.cl" };
const std::string kernel_filename_tr   { "kernels/transpose_kernels.cl" };
const std::string kernel_filename_box  { "kernels/boxFilter_kernels.cl" };
const std::string kernel_filename_math { "kernels/math_kernels.cl" };
const std::string kernel_filename_gf   { "kernels/guidedFilter_kernels.cl" };

// Uniform random number generators
extern std::function<unsigned char ()> rNum_0_255;
extern std::function<unsigned short ()> rNum_0_10000;
extern std::function<float ()> rNum_R_0_1;

bool profiling;  // Flag to enable profiling of the kernels


/*! \brief Tests the **Guided Filter** algorithm for the special case \f$\ I = p \f$.
 *  \details The operation is an edge preserving smoothing effect on an image.
 */
TEST (GuidedFilter, guidedFilter)
{
    try
    {
        const std::vector<std::string> kernel_files = { kernel_filename_scan, 
                                                        kernel_filename_tr, 
                                                        kernel_filename_box,
                                                        kernel_filename_math, 
                                                        kernel_filename_gf };
        const unsigned int width = 640, height = 480;
        const unsigned int bufferSize = width * height * sizeof (cl_float);
        const unsigned int gfRadius = 4;
        const float gfEps = std::pow (0.1, 2);

        // Setup the OpenCL environment
        clutils::CLEnv clEnv;
        clEnv.addContext (0);
        clEnv.addQueue (0, 0, CL_QUEUE_PROFILING_ENABLE);
        clEnv.addQueue (0, 0, CL_QUEUE_PROFILING_ENABLE);
        clEnv.addProgram (0, kernel_files);

        // Configure kernel execution parameters
        clutils::CLEnvInfo<2> info (0, 0, 0, { 0, 1 }, 0);
        cl_algo::GuidedFilter<cl_algo::GuidedFilterConfig::I_EQ_P> gf (clEnv, info);
        gf.init (width, height, gfRadius, gfEps);

        // Initialize data (writes on staging buffer directly)
        std::generate (gf.hPtrIn, gf.hPtrIn + bufferSize / sizeof (cl_float), rNum_R_0_1);
        // printBufferF ("Original:", gf.hPtrIn, width, height, 3);

        gf.write ();  // Copy data to device

        gf.run ();  // Execute kernels (~ 0.790 ms)
        
        cl_float *results = (cl_float *) gf.read ();  // Copy results to host
        // printBufferF ("Received:", results, width, height, 3);

        // Produce reference filtered array
        cl_float *refGF = new cl_float[width * height];
        cpuGuidedFilter (gf.hPtrIn, refGF, width, height, gfRadius, gfEps);
        // printBufferF ("Expected:", refGF, width, height, 3);

        // Verify filtered output (~ 0.0004 error)
        float eps = 42000 * std::numeric_limits<float>::epsilon ();  // 0.00500679
        for (uint row = 0; row < height; ++row)
            for (uint col = 0; col < width; ++col)
                ASSERT_LT (std::abs (refGF[row * width + col] - results[row * width + col]), eps);

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
                cpuGuidedFilter (gf.hPtrIn, refGF, width, height, gfRadius, gfEps);
                pCPU[i] = cTimer.stop ();
            }
            
            // GPU
            clutils::GPUTimer<std::milli> gTimer (clEnv.devices[0][0]);
            clutils::ProfilingInfo<nRepeat> pGPU ("GPU");
            for (int i = 0; i < nRepeat; ++i)
                pGPU[i] = gf.run (gTimer);

            // Benchmark
            pGPU.print (pCPU, "GuidedFilter<GuidedFilterConfig::I_EQ_P>");
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


/*! \brief Tests the **Guided Filter** algorithm for the general case \f$\ I \neq p \f$.
 *  \details There are many applications for this algorithm, one which 
 *           is an edge preserving smoothing effect on an image.
 */
TEST (GuidedFilter, guidedFilterIp)
{
    try
    {
        const std::vector<std::string> kernel_files = { kernel_filename_img, 
                                                        kernel_filename_scan, 
                                                        kernel_filename_tr, 
                                                        kernel_filename_box,
                                                        kernel_filename_math, 
                                                        kernel_filename_gf };
        const unsigned int width = 640, height = 480;
        const unsigned int bufferSize = width * height * sizeof (cl_float);
        const unsigned int gfRadius = 7;
        const float gfEps = std::pow (0.1, 2);

        // Setup the OpenCL environment
        clutils::CLEnv clEnv;
        clEnv.addContext (0);
        clEnv.addQueue (0, 0, CL_QUEUE_PROFILING_ENABLE);
        clEnv.addQueue (0, 0, CL_QUEUE_PROFILING_ENABLE);
        clEnv.addProgram (0, kernel_files);

        // Configure kernel execution parameters
        clutils::CLEnvInfo<2> info (0, 0, 0, { 0, 1 }, 0);
        const cl_algo::GuidedFilterConfig Ip = cl_algo::GuidedFilterConfig::I_NEQ_P;
        cl_algo::GuidedFilter<Ip> gf (clEnv, info);
        gf.init (width, height, gfRadius, gfEps);

        // Initialize data (writes on staging buffer directly)
        std::generate (gf.hPtrInI, gf.hPtrInI + bufferSize / sizeof (cl_float), rNum_R_0_1);
        std::generate (gf.hPtrInP, gf.hPtrInP + bufferSize / sizeof (cl_float), rNum_R_0_1);
        // printBufferF ("Original I:", gf.hPtrInI, width, height, 3);
        // printBufferF ("Original p:", gf.hPtrInP, width, height, 3);

        // Copy data to device
        gf.write (cl_algo::GuidedFilter<Ip>::Memory::D_IN_I);
        gf.write (cl_algo::GuidedFilter<Ip>::Memory::D_IN_P);

        gf.run ();  // Execute kernels (~ 1.120 ms)
        
        cl_float *results = (cl_float *) gf.read ();  // Copy results to host
        // printBufferF ("Received:", results, width, height, 3);

        // Produce reference filtered array
        cl_float *refGF = new cl_float[width * height];
        cpuGuidedFilter (gf.hPtrInI, refGF, width, height, gfRadius, gfEps);
        // printBufferF ("Expected:", refGF, width, height, 3);

        // Verify filtered output (~ 0.0006 error)
        float eps = 42000 * std::numeric_limits<float>::epsilon ();  // 0.00500679
        for (uint row = 0; row < height; ++row)
            for (uint col = 0; col < width; ++col)
                ASSERT_LT (std::abs (refGF[row * width + col] - results[row * width + col]), eps);

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
                cpuGuidedFilter (gf.hPtrInI, refGF, width, height, gfRadius, gfEps);
                pCPU[i] = cTimer.stop ();
            }
            
            // GPU
            clutils::GPUTimer<std::milli> gTimer (clEnv.devices[0][0]);
            clutils::ProfilingInfo<nRepeat> pGPU ("GPU");
            for (int i = 0; i < nRepeat; ++i)
                pGPU[i] = gf.run (gTimer);

            // Benchmark
            pGPU.print (pCPU, "GuidedFilter<GuidedFilterConfig::I_EQ_P>");
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
