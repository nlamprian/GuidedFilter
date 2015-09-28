/*! \file helper_funcs.hpp
 *  \brief Declarations of helper functions for testing.
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

#ifndef GF_HELPERFUNCS_HPP
#define GF_HELPERFUNCS_HPP

#include <cassert>

#if defined(__APPLE__) || defined(__MACOSX)
#include <OpenCL/cl.hpp>
#else
#include <CL/cl.hpp>
#endif


namespace GF
{

    /*! \brief Checks the command line arguments for the profiling flag, `--profiling`. */
    bool setProfilingFlag (int argc, char **argv);


    /*! \brief Returns the first power of 2 greater than or equal to the input.
     *
     *  \param[in] num input data.
     *  \return The first power of 2 >= num.
     */
    template <typename T>
    uint64_t nextPow2 (T num)
    {
        assert (num >= 0);

        uint64_t pow;
        for (pow = 1; pow < (uint64_t) num; pow <<= 1) ;

        return pow;
    }


    /*! \brief Prints an array of an integer type to standard output.
     *
     *  \tparam T type of the data to be printed.
     *  \param[in] title legend for the output.
     *  \param[in] ptr array that is to be displayed.
     *  \param[in] width width of the array.
     *  \param[in] height height of the array.
     */
    template <typename T>
    void printBuffer (const char *title, T *ptr, uint32_t width, uint32_t height)
    {
        std::cout << title << std::endl;

        for (int row = 0; row < height; ++row)
        {
            for (int col = 0; col < width; ++col)
            {
                std::cout << std::setw (3 * sizeof (T)) << +ptr[row * width + col] << " ";
            }
            std::cout << std::endl;
        }

        std::cout << std::endl;
    }


    /*! \brief Prints an array of floating-point type to standard output.
     *
     *  \tparam T type of the data to be printed.
     *  \param[in] title legend for the output.
     *  \param[in] ptr array that is to be displayed.
     *  \param[in] width width of the array.
     *  \param[in] height height of the array.
     *  \param[in] prec the number of decimal places to print.
     */
    template <typename T>
    void printBufferF (const char *title, T *ptr, uint32_t width, uint32_t height, uint32_t prec)
    {
        std::ios::fmtflags f (std::cout.flags ());
        std::cout << title << std::endl;
        std::cout << std::fixed << std::setprecision (prec);

        for (int row = 0; row < height; ++row)
        {
            for (int col = 0; col < width; ++col)
            {
                std::cout << std::setw (5 + prec) << ptr[row * width + col] << " ";
            }
            std::cout << std::endl;
        }

        std::cout << std::endl;
        std::cout.flags (f);
    }


    /*! \brief Performs a matrix transposition.
     *  \details It is just a naive serial implementation.
     *
     *  \param[in] in input image.
     *  \param[out] out output (transpose) image.
     *  \param[in] width width of the input array.
     *  \param[in] height height of the input array.
     */
    template <typename T1, typename T2>
    void cpuTranspose (T1 *in, T2 *out, uint32_t width, uint32_t height)
    {
        for (uint row = 0; row < height; ++row)
            for (uint col = 0; col < width; ++col)
                out[col * height + row] = in[row * width + col];
    }


    /*! \brief Performs a matrix transposition.
     *  \details It is just a naive serial implementation.
     *
     *  \param[in] in input image.
     *  \param[out] r output image with channel r.
     *  \param[out] g output image with channel g.
     *  \param[out] b output image with channel b.
     *  \param[in] pixels number of pixels in the input array.
     */
    template <typename T>
    void cpuSeparateRGB (T *in, T *r, T *g, T *b, uint32_t pixels)
    {
        for (uint i = 0; i < pixels; ++i)
        {
            r[i] = in[i * 3];
            g[i] = in[i * 3 + 1];
            b[i] = in[i * 3 + 2];
        }
    }


    /*! \brief Performs a matrix transposition.
     *  \details It is just a naive serial implementation.
     *
     *  \param[in] in input image.
     *  \param[out] r output image with channel r.
     *  \param[out] g output image with channel g.
     *  \param[out] b output image with channel b.
     *  \param[in] pixels number of pixels in the input array.
     */
    template <typename T2>
    void cpuSeparateRGB_N_Norm (unsigned char *in, T2 *r, T2 *g, T2 *b, uint32_t pixels)
    {
        for (uint i = 0; i < pixels; ++i)
        {
            r[i] = (T2) in[i * 3] / 255.f;
            g[i] = (T2) in[i * 3 + 1] / 255.f;
            b[i] = (T2) in[i * 3 + 2] / 255.f;
        }
    }


    /*! \brief Performs a matrix transposition.
     *  \details It is just a naive serial implementation.
     *
     *  \param[in] r input image with channel r.
     *  \param[in] g input image with channel g.
     *  \param[in] b input image with channel b.
     *  \param[out] out input image.
     *  \param[in] pixels number of pixels in the input array.
     */
    template <typename T>
    void cpuCombineRGB (T *r, T *g, T *b, T *out, uint32_t pixels)
    {
        for (uint i = 0; i < pixels; ++i)
        {
            out[i * 3]     = r[i];
            out[i * 3 + 1] = g[i];
            out[i * 3 + 2] = b[i];
        }
    }


    /*! \brief Performs a matrix transpose.
     *  \details It is just a naive serial implementation.
     *
     *  \param[in] r input image with channel r.
     *  \param[in] g input image with channel g.
     *  \param[in] b input image with channel b.
     *  \param[out] out output (transpose) image.
     *  \param[in] pixels number of pixels in the input array.
     */
    template <typename T1>
    void cpuTranspose_N_Scale (T1 *r, T1 *g, T1 *b, unsigned char *out, uint32_t pixels)
    {
        for (uint i = 0; i < pixels; ++i)
        {
            out[i * 3]     = (unsigned char) (r[i] * 255.f);
            out[i * 3 + 1] = (unsigned char) (g[i] * 255.f);
            out[i * 3 + 2] = (unsigned char) (b[i] * 255.f);
        }
    }


    /*! \brief Transforms a depth image to a point cloud.
     *  \details It is just a naive serial implementation.
     *
     *  \param[in] depth depth image (for Kinect, type: uint16, unit: mm).
     *  \param[out] pCloud point cloud.
     *  \param[in] width width of the array.
     *  \param[in] height height of the array.
     *  \param[in] f focal length (for Kinect: 595.f).
     */
    template <typename T>
    void cpuDepthTo3D (T *depth, cl_float4 *pCloud, uint32_t width, uint32_t height, float f)
    {
        for (uint row = 0; row < height; ++row)
        {
            for (uint col = 0; col < width; ++col)
            {   
                T d = depth[row * width + col];
                pCloud[row * width + col] = { (col - (width - 1) / 2.f) * (float) d / f,
                                              (row - (height - 1) / 2.f) * (float) d / f,
                                              (float) d, 1.f };
            }
        }
    }


    /*! \brief Fuses geometry and color values into 8D feature points.
     *  \details It is just a naive serial implementation.
     *
     *  \param[in] D depth image.
     *  \param[in] R channel R of RGB image.
     *  \param[in] G channel G of RGB image.
     *  \param[in] B channel B of RGB image.
     *  \param[out] points array of 8D feature points.
     *  \param[in] width width of the input arrays.
     *  \param[in] height height of the input arrays.
     *  \param[in] f focal length (for Kinect: 595.f).
     *  \param[in] rgbNorm flag to indicate whether to perform RGB normalization.
     */
    template <typename T>
    void cpuRGBDTo8D (T *D, T *R, T *G, T *B, cl_float8 *points, uint32_t width, uint32_t height, float f, bool rgbNorm)
    {
        for (uint row = 0; row < height; ++row)
        {
            for (uint col = 0; col < width; ++col)
            {
                int i = row * width + col;

                T d = D[i];
                T r = R[i];
                T g = G[i];
                T b = B[i];

                if (rgbNorm)
                {
                    T sum_ = r + g + b;
                    T factor = (sum_ == 0.0) ? 0.0 : 1.0 / sum_;
                    r *= factor;
                    g *= factor;
                    b *= factor;
                }

                points[i] = { (col - (width - 1) / 2.f) * (float) d / f,
                              (row - (height - 1) / 2.f) * (float) d / f,
                              (float) d, 1.f, r, g, b, 1.f };
            }
        }
    }


    /*! \brief Splits an 8-D point cloud into 4-D geometry points and RGBA color points.
     *  \details It is just a naive serial implementation.
     *
     *  \param[in] pc8d array with 8-D points (homogeneous coordiates + RGBA values).
     *  \param[out] pc4d array with 4-D geometry points.
     *  \param[out] rgba array with 4-D color points.
     *  \param[in] n number of points in the 8-D point cloud.
     */
    template <typename T>
    void cpuSplitPC8D (T *pc8d, T *pc4d, T *rgba, uint32_t n)
    {
        for (uint k = 0; k < n; ++k)
        {
            cl_float *point = pc8d + (k << 3);

            for (uint j = 0; j < 4; ++j)
            {
                pc4d[(k << 2) + j] = point[j];
                rgba[(k << 2) + j] = point[4 + j];
            }
        }
    }


    /*! \brief Performs RGB color normalization.
     *  \details That is $$ \\hat{p}.i = \\frac{p.i}{p.r + p.g + p.b},\\ \\ i=\\{r,g,b\\} $$
     *
     *  \tparam T type of the data to be handled.
     *  \param[in] in input data.
     *  \param[out] out output (RGB normalized) data.
     *  \param[in] width width of the array.
     *  \param[in] height height of the array.
     */
    template <typename T>
    void cpuRGBNorm (T *in, T *out, uint32_t width, uint32_t height)
    {
        for (uint row = 0; row < height; ++row)
        {
            for (uint col = 0; col < width; ++col)
            {   
                uint rank = (row * width + col) * 3;
                T sum_ = in[rank] + in[rank+1] + in[rank+2];
                T factor = (sum_ == 0) ? 0.0 : 1.0 / sum_;
                out[rank] = (T) (factor * in[rank]);
                out[rank+1] = (T) (factor * in[rank+1]); 
                out[rank+2] = (T) (factor * in[rank+2]);
            }
        }
    }


    /*! \brief Performs a scan operation.
     *  \details It is just a naive serial implementation.
     *
     *  \tparam T type of the data to be handled.
     *  \param[in] in input data.
     *  \param[out] out output (scan) data.
     *  \param[in] width width of the array.
     *  \param[in] height height of the array.
     */
    template <typename T>
    void cpuScan (T *in, T *out, uint32_t width, uint32_t height)
    {
        // Initialize the first element of each row
        for (uint32_t row = 0; row < height; ++row)
            out[row * width] = in[row * width];
        // Perform the scan
        for (uint32_t row = 0; row < height; ++row)
            for (uint32_t col = 1; col < width; ++col)
                out[row * width + col] = out[row * width + col - 1] + in[row * width + col];
    }


    /*! \brief Constructs a Summed Area Table (SAT).
     *  \details It is just a naive serial implementation.
     *
     *  \tparam T type of the data to be handled.
     *  \param[in] in input data.
     *  \param[out] out output (SAT) data.
     *  \param[in] width width of the array.
     *  \param[in] height height of the array.
     */
    template <typename T>
    void cpuSAT (T *in, T *out, uint32_t width, uint32_t height)
    {
        // Initialize the first element of each row
        for (uint32_t row = 0; row < height; ++row)
            out[row * width] = in[row * width];
        // Perform a scan on the rows
        for (uint32_t row = 0; row < height; ++row)
            for (uint32_t col = 1; col < width; ++col)
                out[row * width + col] = out[row * width + col - 1] + in[row * width + col];
        // Perform a scan on the columns
        for (uint32_t col = 0; col < width; ++col)
            for (uint32_t row = 1; row < height; ++row)
                out[row * width + col] = out[(row - 1) * width + col] + out[row * width + col];
    }


    /*! \brief Performs a blurring effect (mean filtering) on an array.
     *  \details It is just a naive serial implementation.
     *
     *  \param[in] in input array.
     *  \param[out] out output (blurred) array.
     *  \param[in] width width of the array.
     *  \param[in] height height of the array.
     *  \param[in] radius radius of the square filter window.
     */
    template <typename T>
    void cpuBoxFilter (T *in, T *out, int width, int height, int radius)
    {
        for (int row = 0; row < height; ++row)
        {
            for (int col = 0; col < width; ++col)
            {
                T sum = 0.f;
                int n = 0;
                int ix, iy;

                for (int fRow = -radius; fRow <= radius; ++fRow)
                {
                    for (int fCol = -radius; fCol <= radius; ++fCol)
                    {
                        ix = col + fCol;
                        iy = row + fRow;

                        if ((ix >= 0) && (iy >= 0) && (ix < width) && (iy < height))
                        {
                            sum += in[iy * width + ix];
                            n++;
                        }
                    }
                }

                out[row * width + col] = sum / (T) n;
            }
        }
    }


    /*! \brief Performs an element-wise array multiplication.
     *  \details It is just a naive serial implementation.
     *
     *  \param[in] a input array with the first operand.
     *  \param[in] b input array with the second operand.
     *  \param[out] out output array.
     *  \param[in] width width of the array.
     *  \param[in] height height of the array.
     */
    template <typename T>
    void cpuMult (T *a, T *b, T *out, int width, int height)
    {
        for (int row = 0; row < height; ++row)
        {
            for (int col = 0; col < width; ++col)
            {
                int idx = row * width + col;
                out[idx] = a[idx] * b[idx];
            }
        }
    }


    /*! \brief Performs a raise to an integer power.
     *  \details It is just a naive serial implementation.
     *
     *  \param[in] in input array.
     *  \param[out] out output array.
     *  \param[in] width width of the array.
     *  \param[in] height height of the array.
     *  \param[in] n power.
     */
    template <typename T>
    void cpuPown (T *in, T *out, int width, int height, int n)
    {
        for (int row = 0; row < height; ++row)
        {
            for (int col = 0; col < width; ++col)
            {
                int idx = row * width + col;
                out[idx] = pow (in[idx], n);
            }
        }
    }


    /*! \brief Performs edge preserving smoothing on an array.
     *  \details It is just a naive serial implementation.
     *
     *  \param[in] p input array.
     *  \param[out] q output (smoothed) array.
     *  \param[in] width width of the array.
     *  \param[in] height height of the array.
     *  \param[in] radius radius of the square filter window.
     *  \param[in] eps regularization parameter \f$ \epsilon \f$.
     */
    template <typename T>
    void cpuGuidedFilter (T *p, T *q, int width, int height, int radius, float eps)
    {
        int pixels = width * height;
        T *mean_p = new T[pixels];
        T *p2 = new T[pixels];
        T *mean_p2 = new T[pixels];
        T *a = new T[pixels];
        T *b = new T[pixels];
        T *mean_a = new T[pixels];
        T *mean_b = new T[pixels];

        cpuBoxFilter (p, mean_p, width, height, radius);
        cpuPown (p, p2, width, height, 2);
        cpuBoxFilter (p2, mean_p2, width, height, radius);

        for (int row = 0; row < height; ++row)
        {
            for (int col = 0; col < width; ++col)
            {
                int idx = row * width + col;
                T var = mean_p2[idx] - mean_p[idx] * mean_p[idx];
                a[idx] = var / (var + eps);
                b[idx] = (1 - a[idx]) * mean_p[idx];
            }
        }

        cpuBoxFilter (a, mean_a, width, height, radius);
        cpuBoxFilter (b, mean_b, width, height, radius);

        for (int row = 0; row < height; ++row)
        {
            for (int col = 0; col < width; ++col)
            {
                int idx = row * width + col;
                q[idx] = mean_a[idx] * p[idx] + mean_b[idx];
            }
        }
    }

}

#endif  // GF_HELPERFUNCS_HPP
