/*! \file testsMath.cpp
 *  \brief Google Test Unit Tests for math kernels.
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
#include <chrono>
#include <random>
#include <limits>
#include <cmath>
#include <gtest/gtest.h>
#include <CLUtils.hpp>
#include <GuidedFilter/math.hpp>
#include <GuidedFilter/tests/helperFuncs.hpp>


// Kernel filenames
const std::string kernel_filename_math { "kernels/math_kernels.cl" };

// Uniform random number generators
extern std::function<unsigned char ()> rNum_0_255;
extern std::function<unsigned short ()> rNum_0_10000;

bool profiling;  // Flag to enable profiling of the kernels


/*! \brief Tests the **mult** kernel.
 *  \details The operation is an element-wise array multiplication.
 */
TEST (Math, mult)
{
    try
    {
        const unsigned int width = 640, height = 480;
        const unsigned int bufferSize = width * height * sizeof (cl_float);

        // Setup the OpenCL environment
        clutils::CLEnv clEnv;
        clEnv.addContext (0);
        clEnv.addQueue (0, 0, CL_QUEUE_PROFILING_ENABLE);
        clEnv.addProgram (0, kernel_filename_math);

        // Configure kernel execution parameters
        clutils::CLEnvInfo<1> info (0, 0, 0, { 0 }, 0);
        cl_algo::GF::Math::Mult mult (clEnv, info);
        mult.init (width, height);

        // Initialize data (writes on staging buffer directly)
        std::generate (mult.hPtrInA, mult.hPtrInA + bufferSize / sizeof (cl_float), rNum_0_255);
        std::generate (mult.hPtrInB, mult.hPtrInB + bufferSize / sizeof (cl_float), rNum_0_255);
        // printBufferF ("Original A:", mult.hPtrInA, width, height, 0);
        // printBufferF ("Original B:", mult.hPtrInB, width, height, 0);

        // Copy data to device
        mult.write (cl_algo::GF::Math::Mult::Memory::D_IN_A);
        mult.write (cl_algo::GF::Math::Mult::Memory::D_IN_B);

        mult.run ();  // Execute kernels (26 us)
        
        cl_float *results = (cl_float *) mult.read ();  // Copy results to host
        // printBufferF ("Received:", results, width, height, 1);

        // Produce reference blurred array
        cl_float refMult[width * height];
        cpuMult (mult.hPtrInA, mult.hPtrInB, refMult, width, height);
        // printBufferF ("Expected:", refMult, width, height, 1);

        // Verify blurred output
        float eps = std::numeric_limits<float>::epsilon ();  // 1.19209e-07
        for (uint row = 0; row < height; ++row)
            for (uint col = 0; col < width; ++col)
                ASSERT_LT (std::abs (refMult[row * width + col] - results[row * width + col]), eps);

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
                cpuMult (mult.hPtrInA, mult.hPtrInB, refMult, width, height);
                pCPU[i] = cTimer.stop ();
            }
            
            // GPU
            clutils::GPUTimer<std::milli> gTimer (clEnv.devices[0][0]);
            clutils::ProfilingInfo<nRepeat> pGPU ("GPU");
            for (int i = 0; i < nRepeat; ++i)
                pGPU[i] = mult.run (gTimer);

            // Benchmark
            pGPU.print (pCPU, "Mult");
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


/*! \brief Tests the **pown_** kernel.
 *  \details The operation is a raise to an integer power.
 */
TEST (Math, pown)
{
    try
    {
        const unsigned int width = 640, height = 480;
        const unsigned int bufferSize = width * height * sizeof (cl_float);
        const int power = 2;

        // Setup the OpenCL environment
        clutils::CLEnv clEnv;
        clEnv.addContext (0);
        clEnv.addQueue (0, 0, CL_QUEUE_PROFILING_ENABLE);
        clEnv.addProgram (0, kernel_filename_math);

        // Configure kernel execution parameters
        clutils::CLEnvInfo<1> info (0, 0, 0, { 0 }, 0);
        cl_algo::GF::Math::Pown pown (clEnv, info);
        pown.init (width, height, power);

        // Initialize data (writes on staging buffer directly)
        std::generate (pown.hPtrIn, pown.hPtrIn + bufferSize / sizeof (cl_float), rNum_0_255);
        // printBufferF ("Original:", pown.hPtrIn, width, height, 0);

        pown.write ();  // Copy data to device

        pown.run ();  // Execute kernels (59 us)
        
        cl_float *results = (cl_float *) pown.read ();  // Copy results to host
        // printBufferF ("Received:", results, width, height, 1);

        // Produce reference blurred array
        cl_float refPow[width * height];
        cpuPown (pown.hPtrIn, refPow, width, height, power);
        // printBufferF ("Expected:", refPow, width, height, 1);

        // Verify blurred output
        float eps = std::numeric_limits<float>::epsilon ();  // 1.19209e-07
        for (uint row = 0; row < height; ++row)
            for (uint col = 0; col < width; ++col)
                ASSERT_LT (std::abs (refPow[row * width + col] - results[row * width + col]), eps);

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
                cpuPown (pown.hPtrIn, refPow, width, height, power);
                pCPU[i] = cTimer.stop ();
            }
            
            // GPU
            clutils::GPUTimer<std::milli> gTimer (clEnv.devices[0][0]);
            clutils::ProfilingInfo<nRepeat> pGPU ("GPU");
            for (int i = 0; i < nRepeat; ++i)
                pGPU[i] = pown.run (gTimer);

            // Benchmark
            pGPU.print (pCPU, "Pown");
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
