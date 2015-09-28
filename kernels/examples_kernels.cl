/*! \file examples_kernels.cl
 *  \brief Kernels used by the provided example applications.
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


/*! \brief Combines the 3 channels of an RGB Image.
 *  \details Assembles the pixels, performs RGB normalization, 
 *           and stores the resulting pixel to an image object.
 *  \note It's meant to be used for storing the results on an OpenGL texture.
 *  \note Performs a matrix transposition on an RGB image `(SoA -> AoS)`.
 *        For avoiding alignment restrictions, the `SoA` structure 
 *        is broken out to the individual channels, R, G, B.
 *  \note The global workspace should be one-dimensional `(= # pixels 
 *        in the input buffer)`. The global and local workspaces 
 *        should be a **multiple of 3**.
 *
 *  \param[in] r input buffer with all the pixel values in channel R.
 *  \param[in] g input buffer with all the pixel values in channel G.
 *  \param[in] b input buffer with all the pixel values in channel B.
 *  \param[out] out output RGBA image.
 *  \param[in] data local buffer with size `3 x (# work-items in work-group) x sizeof (float)` bytes.
 *  \param[in] width width the images.
 *  \param[in] norm flag to indicate whether to perform RGB normalization before storing the pixels.
 */
kernel
void combineRGBGL (global float *r, global float *g, global float *b, 
                   write_only image2d_t out, local float *data, int width, int norm)
{
    global float *addr[] = { r, g, b };

    // Workspace dimensions
    uint pixels = get_global_size (0);
    uint lXdim = get_local_size (0);

    // Workspace indices
    uint gX = get_global_id (0);
    uint lX = get_local_id (0);
    uint wgX = get_group_id (0);

    // Each 1/3 work-items in the work-group reads in 
    // a triplet of values on channel, R, G, B, respectively
    uchar channel = (3 * lX) / lXdim;
    uint rank = lX % (lXdim / 3);
    global float *img = addr[channel];
    vstore3 (vload3 (rank, &img[wgX * lXdim]), rank, &data[channel * lXdim]);
    barrier (CLK_LOCAL_MEM_FENCE);

    // Each work-item in the work-group assembles and stores a pixel
    float3 color = { data[lX], data[lXdim + lX], data[2 * lXdim + lX] };

    if (norm)
    {
        float sum_ = dot (color, 1.f);
        float factor = select (native_recip (sum_), 0.f, isequal (sum_, 0.f));
        color *= factor;
    }

    float4 pixel = (float4) (color, 1.f);

    // Store the pixel
    int2 coords = { gX % width, gX / width };
    write_imagef (out, coords, pixel);
}


/*! \brief Combines the 3 channels of an RGB Image.
 *  \details Assembles the pixels, performs RGB normalization, 
 *           and stores the resulting pixel to an image object.
 *  \note It's meant to be used for storing the results on an OpenGL vertex buffer.
 *  \note Performs a matrix transposition on an RGB image `(SoA -> AoS)`.
 *        For avoiding alignment restrictions, the `SoA` structure 
 *        is broken out to the individual channels, R, G, B.
 *  \note The global workspace should be one-dimensional `(= # pixels 
 *        in the input buffer)`. The global and local workspaces 
 *        should be a **multiple of 3**.
 *
 *  \param[in] r input buffer with all the pixel values in channel R.
 *  \param[in] g input buffer with all the pixel values in channel G.
 *  \param[in] b input buffer with all the pixel values in channel B.
 *  \param[out] out output RGBA image.
 *  \param[in] data local buffer with size `3 x (# work-items in work-group) x sizeof (float)` bytes.
 *  \param[in] width width the images.
 *  \param[in] norm flag to indicate whether to perform RGB normalization before storing the pixels.
 */
kernel
void combineRGBGL_PC (global float *r, global float *g, global float *b, 
                      global float4 *out, local float *data, int width, int norm)
{
    global float *addr[] = { r, g, b };

    // Workspace dimensions
    uint pixels = get_global_size (0);
    uint lXdim = get_local_size (0);

    // Workspace indices
    uint gX = get_global_id (0);
    uint lX = get_local_id (0);
    uint wgX = get_group_id (0);

    // Each 1/3 work-items in the work-group reads in 
    // a triplet of values on channel, R, G, B, respectively
    uchar channel = (3 * lX) / lXdim;
    uint rank = lX % (lXdim / 3);
    global float *img = addr[channel];
    vstore3 (vload3 (rank, &img[wgX * lXdim]), rank, &data[channel * lXdim]);
    barrier (CLK_LOCAL_MEM_FENCE);

    // Each work-item in the work-group assembles and stores a pixel
    float3 color = { data[lX], data[lXdim + lX], data[2 * lXdim + lX] };

    if (norm)
    {
        float sum_ = dot (color, 1.f);
        float factor = select (native_recip (sum_), 0.f, isequal (sum_, 0.f));
        color *= factor;
    }

    out[gX] = (float4) (color, 1.f);
}
