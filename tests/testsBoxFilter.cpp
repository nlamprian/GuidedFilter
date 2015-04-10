/*! \file testsBoxFilter.cpp
 *  \brief Google Test Unit Tests for the Box Filtering kernels.
 *  \note Use the `--profiling` flag to enable profiling of the kernels.
 *  \note The benchmarks in these tests are against naive CPU implementations 
 *        of the associated algorithms. They are used only for testing purposes, 
 *        and not for examining the performance of their GPU alternatives.
 *  \author Nick Lamprianidis
 *  \version 1.0
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
const std::string kernel_filename_scan { "kernels/prefixSum_kernels.cl" };
const std::string kernel_filename_tr   { "kernels/transpose_kernels.cl" };
const std::string kernel_filename_box  { "kernels/boxFilter_kernels.cl" };

// Uniform random number generators
extern std::function<unsigned char ()> rNum_0_255;
extern std::function<unsigned short ()> rNum_0_10000;
extern std::function<float ()> rNum_R_0_1;
extern std::function<float ()> rNum_R_1_255_E__6;

bool profiling;  // Flag to enable profiling of the kernels


/*! \brief Tests the **prefixSum** kernel.
 *  \details The operation is a prefix sum scan on the rows in an array.
 */
TEST (BoxFilter, prefixSum)
{
    try
    {
        const unsigned int width = 640, height = 480;
        const unsigned int bufferSize = width * height * sizeof (cl_float);

        // Set up OpenCL environment
        clutils::CLEnv clEnv;
        clEnv.addContext (0);
        cl::CommandQueue &queue (clEnv.addQueue (0, 0, CL_QUEUE_PROFILING_ENABLE));
        clEnv.addProgram (0, kernel_filename_scan);

        // Configure kernel execution parameters
        clutils::CLEnvInfo<1> info (0, 0, 0, { 0 }, 0);
        cl_algo::PrefixSum scan (clEnv, info);
        scan.init (width, height);

        // Initialize data (writes on staging buffer directly)
        std::generate (scan.hPtrIn, scan.hPtrIn + bufferSize / sizeof (cl_float), rNum_R_1_255_E__6);
        // printBufferF ("Original:", scan.hPtrIn, width, height, 3);

        scan.write ();  // Copy data to device

        scan.run ();  // Execute kernels
        
        // Check partial (group) sums
        // cl_float groupSums[bufferSize / width];
        // queue.enqueueReadBuffer ((cl::Buffer &) scan.get (scan.Memory::D_SUMS), 
        //                          CL_TRUE, 0, bufferSize / width, groupSums);
        // printBufferF ("\nPartial Sums:", groupSums, height, 1, 3);

        cl_float *results = (cl_float *) scan.read ();  // Copy results to host
        // printBufferF ("Received:", results, width, height, 3);

        // Produce reference prefix sum array
        cl_float refScan[width * height];
        cpuScan (scan.hPtrIn, refScan, width, height);
        // printBufferF ("Expected:", refScan, width, height, 3);

        // Verify prefix sum output
        float eps = 42 * std::numeric_limits<float>::epsilon ();  // 5.00679e-06
        for (uint row = 0; row < height; ++row)
            for (uint col = 0; col < width; ++col)
                ASSERT_LT (std::abs (refScan[row * width + col] - results[row * width + col]), eps);

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
                cpuScan (scan.hPtrIn, refScan, width, height);
                pCPU[i] = cTimer.stop ();
            }
            
            // GPU
            clutils::GPUTimer<std::milli> gTimer (clEnv.devices[0][0]);
            clutils::ProfilingInfo<nRepeat> pGPU ("GPU");
            for (int i = 0; i < nRepeat; ++i)
                pGPU[i] = scan.run (gTimer);

            // Benchmark
            pGPU.print (pCPU, "PrefixSum");
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


/*! \brief Tests the **transpose** kernel.
 *  \details The operation is a matrix transposition.
 */
TEST (BoxFilter, transpose)
{
    try
    {
        const unsigned int width = 640, height = 480;
        const unsigned int bufferSize = width * height * sizeof (cl_float);

        // Setup the OpenCL environment
        clutils::CLEnv clEnv;
        clEnv.addContext (0);
        clEnv.addQueue (0, 0, CL_QUEUE_PROFILING_ENABLE);
        clEnv.addProgram (0, kernel_filename_tr);

        // Configure kernel execution parameters
        clutils::CLEnvInfo<1> info (0, 0, 0, { 0 }, 0);
        cl_algo::Transpose transpose (clEnv, info);
        transpose.init (width, height);

        // Initialize data (writes on staging buffer directly)
        std::generate (transpose.hPtrIn, transpose.hPtrIn + bufferSize / sizeof (cl_float), rNum_R_0_1);
        // printBufferF ("Original:", transpose.hPtrIn, width, height, 3);
        
        transpose.write ();  // Copy data to device

        transpose.run ();  // Execute kernels
        
        cl_float *results = (cl_float *) transpose.read ();  // Copy results to host
        // printBufferF ("Received:", results, width, height, 3);

        // Produce reference transposed array
        cl_float refTr[width * height];
        cpuTranspose (transpose.hPtrIn, refTr, width, height);
        // printBufferF ("Expected:", refTr, width, height, 3);

        // Verify transposed output
        for (uint row = 0; row < height; ++row)
            for (uint col = 0; col < width; ++col)
                ASSERT_EQ (refTr[row * width + col], results[row * width + col]);

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
                cpuScan (transpose.hPtrIn, refTr, width, height);
                pCPU[i] = cTimer.stop ();
            }
            
            // GPU
            clutils::GPUTimer<std::milli> gTimer (clEnv.devices[0][0]);
            clutils::ProfilingInfo<nRepeat> pGPU ("GPU");
            for (int i = 0; i < nRepeat; ++i)
                pGPU[i] = transpose.run (gTimer);

            // Benchmark
            pGPU.print (pCPU, "Transpose");
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


/*! \brief Tests the construction of a Summed Area Table (**SAT**).
 *  \details The operations performed are a prefix sum scan on the rows 
 *           in an array, an array transposition, and then a prefix sum 
 *           scan on the columns of the array.
 */
TEST (BoxFilter, sat)
{
    try
    {
        const std::vector<std::string> kernel_files = { kernel_filename_scan, 
                                                        kernel_filename_tr };
        const unsigned int width = 640, height = 480;
        const unsigned int bufferSize = width * height * sizeof (cl_float);

        // Setup the OpenCL environment
        clutils::CLEnv clEnv;
        clEnv.addContext (0);
        clEnv.addQueue (0, 0, CL_QUEUE_PROFILING_ENABLE);
        clEnv.addProgram (0, kernel_files);

        // Configure kernel execution parameters
        clutils::CLEnvInfo<1> info (0, 0, 0, { 0 }, 0);
        cl_algo::SAT sat (clEnv, info);
        sat.init (width, height);

        // Initialize data (writes on staging buffer directly)
        std::generate (sat.hPtrIn, sat.hPtrIn + bufferSize / sizeof (cl_float), rNum_R_1_255_E__6);
        // printBufferF ("Original:", sat.hPtrIn, width, height, 3);

        sat.write ();  // Copy data to device

        sat.run ();  // Execute kernels
        
        cl_float *results = (cl_float *) sat.read ();  // Copy results to host
        // printBufferF ("Received:", results, height, width, 5);

        // Produce reference SAT array
        cl_float refTmp[width * height], refSAT[width * height];
        cpuSAT (sat.hPtrIn, refTmp, width, height);
        cpuTranspose (refTmp, refSAT, width, height);
        // printBufferF ("Expected:", refSAT, height, width, 5);

        // Verify SAT output
        float eps = 420 * std::numeric_limits<float>::epsilon ();  // 5.00679e-05
        for (uint row = 0; row < width; ++row)
            for (uint col = 0; col < height; ++col)
                ASSERT_LT (std::abs (refSAT[row * height + col] - results[row * height + col]), eps);

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
                cpuSAT (sat.hPtrIn, refTmp, width, height);
                cpuTranspose(refTmp, refSAT, width, height);
                pCPU[i] = cTimer.stop ();
            }
            
            // GPU
            clutils::GPUTimer<std::milli> gTimer (clEnv.devices[0][0]);
            clutils::ProfilingInfo<nRepeat> pGPU ("GPU");
            for (int i = 0; i < nRepeat; ++i)
                pGPU[i] = sat.run (gTimer);

            // Benchmark
            pGPU.print (pCPU, "SAT");
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


/*! \brief Tests the **boxFilterSAT** kernel.
 *  \details The operation is a blurring effect (mean filtering) on an image,
 *           using SAT arrays.
 */
TEST (BoxFilter, boxFilterSAT)
{
    try
    {
        const std::vector<std::string> kernel_files = { kernel_filename_scan, 
                                                        kernel_filename_tr, 
                                                        kernel_filename_box };
        const unsigned int width = 640, height = 480;
        const unsigned int bufferSize = width * height * sizeof (cl_float);
        const unsigned int filterRadius = 3;

        // Setup the OpenCL environment
        clutils::CLEnv clEnv;
        clEnv.addContext (0);
        clEnv.addQueue (0, 0, CL_QUEUE_PROFILING_ENABLE);
        clEnv.addProgram (0, kernel_files);

        // Configure kernel execution parameters
        clutils::CLEnvInfo<1> info (0, 0, 0, { 0 }, 0);
        cl_algo::BoxFilterSAT box (clEnv, info);
        box.init (width, height, filterRadius);

        // Initialize data (writes on staging buffer directly)
        std::generate (box.hPtrIn, box.hPtrIn + bufferSize / sizeof (cl_float), rNum_R_0_1);
        // printBufferF ("Original:", box.hPtrIn, width, height, 5);

        box.write ();  // Copy data to device

        box.run ();  // Execute kernels
        
        cl_float *results = (cl_float *) box.read ();  // Copy results to host
        // printBufferF ("Received:", results, width, height, 5);

        // Produce reference blurred array
        cl_float refBox[width * height];
        cpuBoxFilter (box.hPtrIn, refBox, width, height, filterRadius);
        // printBufferF ("Expected:", refBox, width, height, 5);

        // Verify blurred output (~ 0.0009 error)
        float eps = 42000 * std::numeric_limits<float>::epsilon ();  // 0.00500679
        for (uint row = 0; row < height; ++row)
            for (uint col = 0; col < width; ++col)
                ASSERT_LT (std::abs (refBox[row * width + col] - results[row * width + col]), eps);

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
                cpuBoxFilter (box.hPtrIn, refBox, width, height, filterRadius);
                pCPU[i] = cTimer.stop ();
            }
            
            // GPU
            clutils::GPUTimer<std::milli> gTimer (clEnv.devices[0][0]);
            clutils::ProfilingInfo<nRepeat> pGPU ("GPU");
            for (int i = 0; i < nRepeat; ++i)
                pGPU[i] = box.run (gTimer);

            // Benchmark
            pGPU.print (pCPU, "BoxFilterSAT");
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


/*! \brief Tests the **boxFilter** kernel.
 *  \details The operation is a blurring effect (mean filtering) on an image.
 */
TEST (BoxFilter, boxFilter)
{
    try
    {
        const unsigned int width = 640, height = 480;
        const unsigned int bufferSize = width * height * sizeof (cl_float);
        const unsigned int filterRadius = 3;

        // Setup the OpenCL environment
        clutils::CLEnv clEnv;
        clEnv.addContext (0);
        clEnv.addQueue (0, 0, CL_QUEUE_PROFILING_ENABLE);
        clEnv.addProgram (0, kernel_filename_box);

        // Configure kernel execution parameters
        clutils::CLEnvInfo<1> info (0, 0, 0, { 0 }, 0);
        cl_algo::BoxFilter box (clEnv, info);
        box.init (width, height, filterRadius);

        // Initialize data (writes on staging buffer directly)
        std::generate (box.hPtrIn, box.hPtrIn + bufferSize / sizeof (cl_float), rNum_R_0_1);
        // printBufferF ("Original:", box.hPtrIn, width, height, 3);

        box.write ();  // Copy data to device

        box.run ();  // Execute kernels
        
        cl_float *results = (cl_float *) box.read ();  // Copy results to host
        // printBufferF ("Received:", results, width, height, 5);

        // Produce reference blurred array
        cl_float refBox[width * height];
        cpuBoxFilter (box.hPtrIn, refBox, width, height, filterRadius);
        // printBufferF ("Expected:", refBox, width, height, 5);

        // Verify blurred output
        float eps = 420 * std::numeric_limits<float>::epsilon ();  // 5.00679e-05
        for (uint row = 0; row < height; ++row)
            for (uint col = 0; col < width; ++col)
                ASSERT_LT (std::abs (refBox[row * width + col] - results[row * width + col]), eps);

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
                cpuBoxFilter (box.hPtrIn, refBox, width, height, filterRadius);
                pCPU[i] = cTimer.stop ();
            }
            
            // GPU
            clutils::GPUTimer<std::milli> gTimer (clEnv.devices[0][0]);
            clutils::ProfilingInfo<nRepeat> pGPU ("GPU");
            for (int i = 0; i < nRepeat; ++i)
                pGPU[i] = box.run (gTimer);

            // Benchmark
            pGPU.print (pCPU, "BoxFilter");
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
