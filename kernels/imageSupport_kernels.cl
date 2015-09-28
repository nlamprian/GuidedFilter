/*! \file imageSupport_kernels.cl
 *  \brief Kernels for manipulating images.
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


/*! \brief Separates the 3 channels of an RGB image.
 *  \details Performs a matrix transposition on an RGB image `(AoS -> SoA)`.
 *           For avoiding alignment restrictions, the `SoA` structure 
 *           is broken out to the individual channels, R, G, B.
 *  \note The global workspace should be one-dimensional `(= # pixels 
 *        in the input buffer)`. The global and local workspaces 
 *        should be **multiples of 3**.
 *
 *  \param[in] AoS input buffer with the following (logical) arrangement: float[total-pixels][3].
 *                 Each row contains the RGB values of a pixel.
 *  \param[out] r output buffer with all the pixel values in the first channel, R.
 *  \param[out] g output buffer with all the pixel values in the second channel, G.
 *  \param[out] b output buffer with all the pixel values in the third channel, B.
 *  \param[in] data local buffer with size `3 x (# work-items in work-group) x sizeof (float)` bytes.
 */
kernel
void separateRGBChannels_Float2Float (global float *AoS, 
                                      global float *r, global float *g, global float *b, 
                                      local float *data)
{
    global float *addr[] = { r, g, b };

    // Workspace dimensions
    uint lXdim = get_local_size (0);

    // Workspace indices
    uint gX = get_global_id (0);
    uint lX = get_local_id (0);
    uint wgX = get_group_id (0);

    // Each work-item in the work-group reads in a pixel's values
    vstore3 (vload3 (gX, AoS), lX, data);
    barrier (CLK_LOCAL_MEM_FENCE);

    // With each 1/3 work-items in the work-group, indices will offset by one,
    // handling this way first the R, then the G, and then the B values
    uint lastIdx = 3 * lXdim - 1;
    uint baseIdx = (9 * lX) % lastIdx;
    
    // A triplet of values on the same channel
    float3 triplet = { data[baseIdx], data[baseIdx + 3], data[baseIdx + 6] };
    
    // Each 1/3 work-items in the work-group 
    // stores the values of one channel
    uchar channel = (3 * lX) / lXdim;
    global float *img = addr[channel];

    vstore3 (triplet, lX % (lXdim / 3), &img[wgX * lXdim]);
}


/*! \brief Separates the 3 channels of an RGB image.
 *  \details Performs a matrix transposition on an RGB image `(AoS -> SoA)`, 
 *           and promotes the `uchar` type to `float` while normalizing
 *           the values to one. For avoiding alignment restrictions, the `SoA`
 *           structure is broken out to the individual channels, R, G, B.
 *  \note The global workspace should be one-dimensional `(= # pixels 
 *        in the input buffer)`. The global and local workspaces 
 *        should be **multiples of 3**.
 *
 *  \param[in] AoS input buffer with the following (logical) arrangement: uchar[total-pixels][3].
 *                 Each row contains the RGB values of a pixel.
 *  \param[out] r output buffer with all the pixel values in the first channel, R.
 *  \param[out] g output buffer with all the pixel values in the second channel, G.
 *  \param[out] b output buffer with all the pixel values in the third channel, B.
 *  \param[in] data local buffer with size `3 x (# work-items in work-group) x sizeof (uchar)` bytes.
 */
kernel
void separateRGBChannels_Uchar2Float (global uchar *AoS, 
                                      global float *r, global float *g, global float *b, 
                                      local uchar *data)
{
    global float *addr[] = { r, g, b };

    // Workspace dimensions
    uint lXdim = get_local_size (0);

    // Workspace indices
    uint gX = get_global_id (0);
    uint lX = get_local_id (0);
    uint wgX = get_group_id (0);

    // Each work-item in the work-group reads in a pixel's values
    vstore3 (vload3 (gX, AoS), lX, data);
    barrier (CLK_LOCAL_MEM_FENCE);

    // With each 1/3 work-items in the work-group, indices will offset by one,
    // handling this way first the R, then the G, and then the B values
    uint lastIdx = 3 * lXdim - 1;
    uint baseIdx = (9 * lX) % lastIdx;
    
    // A triplet of values on the same channel
    float3 triplet = { data[baseIdx], data[baseIdx + 3], data[baseIdx + 6] };

    // Normalize the values
    triplet /= 255.f;
    
    // Each 1/3 work-items in the work-group 
    // stores the values of one channel
    uchar channel = (3 * lX) / lXdim;
    global float *img = addr[channel];

    vstore3 (triplet, lX % (lXdim / 3), &img[wgX * lXdim]);
}


/*! \brief Combines the 3 channels of an RGB Image.
 *  \details Performs a matrix transposition on an RGB image `(SoA -> AoS)`.
 *           For avoiding alignment restrictions, the `SoA` structure 
 *           is broken out to the individual channels, R, G, B.
 *  \note The global workspace should be one-dimensional `(= # pixels 
 *        in the input buffer)`. The global and local workspaces 
 *        should be **multiples of 3**.
 *
 *  \param[in] r input buffer with all the pixel values in channel R.
 *  \param[in] g input buffer with all the pixel values in channel G.
 *  \param[in] b input buffer with all the pixel values in channel B.
 *  \param[out] AoS output buffer with the following (logical) arrangement: float[total-pixels][3].
 *                  Each row contains the RGB values of a pixel.
 *  \param[in] data local buffer with size `3 x (# work-items in work-group) x sizeof (float)` bytes.
 */
kernel
void combineRGBChannels_Float2Float (global float *r, global float *g, global float *b, 
                                     global float *AoS, local float *data)
{
    global float *addr[] = { r, g, b };

    // Workspace dimensions
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
    float3 pixel = { data[lX], data[lXdim + lX], data[2 * lXdim + lX] };

    vstore3 (pixel, gX, AoS);
}


/*! \brief Combines the 3 channels of an RGB Image.
 *  \details Performs a matrix transposition on an RGB image `(SoA -> AoS)`,
 *           demotes the `float` type to `uchar`, and scales the data to `255`.
 *           For avoiding alignment restrictions, the `SoA` structure is broken 
 *           out to the individual channels, R, G, B.
 *  \note The global workspace should be one-dimensional `(= # pixels 
 *        in the input buffer)`. The global and local workspaces 
 *        should be **multiples of 3**.
 *
 *  \param[in] r input buffer with all the pixel values in channel R.
 *  \param[in] g input buffer with all the pixel values in channel G.
 *  \param[in] b input buffer with all the pixel values in channel B.
 *  \param[out] AoS output buffer with the following (logical) arrangement: uchar[total-pixels][3].
 *                  Each row contains the RGB values of a pixel.
 *  \param[in] data local buffer with size `3 x (# work-items in work-group) x sizeof (float)` bytes.
 */
kernel
void combineRGBChannels_Float2Uchar (global float *r, global float *g, global float *b, 
                                     global uchar *AoS, local float *data)
{
    global float *addr[] = { r, g, b };

    // Workspace dimensions
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
    float3 triplet = { data[lX], data[lXdim + lX], data[2 * lXdim + lX] };

    // Scale the values
    triplet *= 255.f;

    // Demote the type
    uchar3 pixel = convert_uchar3 (triplet);

    vstore3 (pixel, gX, AoS);
}


/*! \brief Converts a buffer from type `uchort` to `float`.
 *  \note The global workspace should be one dimensional and equal to 
 *        the number of elements in the image divided by 4.
 *
 *  \param[in] depth depth image (for Kinect, type: uint16, unit: mm).
 *  \param[out] fDepth depth image with type `float`.
 *  \param[in] scaling factor by which to scale the depth values in the output array.
 */
kernel
void depth_Ushort2Float (global ushort4 *depth, global float4 *fDepth, float scaling)
{
    uint gX = get_global_id (0);

    fDepth[gX] = convert_float4 (depth[gX]) * scaling;
}


/*! \brief Transforms a depth image to a point cloud.
 *  \note The global workspace should be equal to the dimensions of the image.
 *
 *  \param[in] depth depth image.
 *  \param[out] pCloud point cloud (in 4D homogeneous coordinates with \f$ w=1 \f$).
 *  \param[in] f focal length (for Kinect: 595.f).
 *  \param[in] scaling factor by which to scale the depth values before building the point cloud.
 */
kernel
void depthTo3D (global float *depth, global float4 *pCloud, float f, float scaling)
{
    // Workspace dimensions
    uint cols = get_global_size (0);
    uint rows = get_global_size (1);

    // Workspace indices
    uint gX = get_global_id (0);
    uint gY = get_global_id (1);

    // Flatten indices
    uint idx = gY * cols + gX;

    float d = depth[idx] * scaling;
    float4 point = { (gX - (cols - 1) / 2.f) * d / f,  // X = (x - cx) * d / fx
                     (gY - (rows - 1) / 2.f) * d / f,  // Y = (y - cy) * d / fy
                      d, 1.f };                        // Z = d

    pCloud[idx] = point;
}


/*! \brief Performs RGB color normalization.
 *  \details That is $$ \\hat{p}.i = \\frac{p.i}{p.r + p.g + p.b},\\ \\ i=\\{r,g,b\\} $$
 *  \note The global workspace should be one-dimensional `(= # pixels in the input buffer)`.
 *
 *  \param[in] in original frame.
 *  \param[out] out processed frame.
 */
kernel
void rgbNorm (global float *in, global float *out)
{
    uint gX = get_global_id (0);

    // Calculate normalizing factor
    float3 pixel = vload3 (gX, in);
    float sum_ = dot (pixel, 1.f);
    float factor = select (native_recip (sum_), 0.f, isequal (sum_, 0.f));
    
    // Normalize and store
    pixel *= factor;
    vstore3 (pixel, gX, out);
}


/*! \brief Fuses 3-D space coordinates and RGB color values into 
 *         8-D feature points (homogeneous coordinates + RGBA values).
 *  \details Gathers the R, G, B channel values and assembles them into a pixel 
 *           with opacity set to 1.0. It can optionally perform RGB normalization. 
 *           Transforms the depth values into 4-D homogeneous coordinates, setting \f$ w=1 \f$.
 *  \note The global workspace should be one-dimensional `(= # elements 
 *        in the input buffers)`. Both global and local workspaces 
 *        should be **multiples of 3**.
 *
 *  \param[in] depth depth array.
 *  \param[in] r channel R array.
 *  \param[in] g channel G array.
 *  \param[in] b channel B array.
 *  \param[out] p8D array of 8D elements (homogeneous coordinates + RGBA values).
 *  \param[in] data local buffer with size `3 x (# work-items in work-group) x sizeof (float)` bytes.
 *  \param[in] cols number of columns (width) in the input images.
 *  \param[in] f focal length (for Kinect: 595.f).
 *  \param[in] scaling factor by which to scale the depth values before building the point cloud.
 *  \param[in] rgb_norm flag to indicate whether to perform RGB normalization.
 */
kernel
void rgbdTo8D (global float *depth, global float *r, global float *g, global float *b, 
               global float8 *p8D, local float *data, uint cols, float f, float scaling, int rgb_norm)
{
    global float *addr[] = { r, g, b };

    // Workspace dimensions
    uint rows = get_global_size (0) / cols;
    uint lXdim = get_local_size (0);

    // Workspace indices
    uint gX = get_global_id (0);
    uint lX = get_local_id (0);
    uint wgX = get_group_id (0);

    // Collect RGB values ======================================================

    // Each 1/3 work-items in the work-group reads in 
    // a triplet of values on channel, R, G, B, respectively
    uchar channel = (3 * lX) / lXdim;
    uint rank = lX % (lXdim / 3);
    global float *img = addr[channel];
    vstore3 (vload3 (rank, &img[wgX * lXdim]), rank, &data[channel * lXdim]);
    barrier (CLK_LOCAL_MEM_FENCE);

    // Each work-item in the work-group assembles a pixel    
    float3 pixel = (float3) (data[lX], data[lXdim + lX], data[2 * lXdim + lX]);

    // Perform RGB normalization
    if (rgb_norm)
    {
        float sum_ = dot (pixel, 1.f);
        float factor = select (native_recip (sum_), 0.f, isequal (sum_, 0.f));
        pixel *= factor;
    }
    
    float4 color = (float4) (pixel, 1.f);

    // Build 3D coordinates ====================================================

    uint x = get_global_id (0) % cols;
    uint y = get_global_id (0) / cols;

    float d = depth[gX] * scaling;
    float4 geometry = { (x - (cols - 1) / 2.f) * d / f,  // X = (x - cx) * d / fx
                        (y - (rows - 1) / 2.f) * d / f,  // Y = (y - cy) * d / fy
                         d, 1.f };                       // Z = d

    // Store feature point =====================================================

    float8 point = (float8) (geometry, color);
    p8D[gX] = point;
}


/*! \brief Splits an 8-D point cloud into 4-D geometry points and RGBA color points.
 *  \note The global workspace should be one-dimensional. The **x** dimension 
 *        of the global workspace, \f$ gXdim \f$, should be equal to the number 
 *        of points in the point cloud. The local workspace is irrelevant.
 *
 *  \param[in] pc8d array with 8-D points (homogeneous coordinates + RGBA values).
 *  \param[out] pc4d array with 4-D geometry points.
 *  \param[out] rgba array with 4-D color points.
 *  \param[in] offset number of points to skip in the output arrays. The kernel will 
 *                    write in the output arrays starting at position `offset`.
 */
kernel
void splitPC8D (global float8 *pc8d, global float4 *pc4d, global float4 *rgba, unsigned int offset)
{
    uint gX = get_global_id (0);

    float8 point = pc8d[gX];
    size_t pos = offset + gX;
    pc4d[pos] = point.lo;
    rgba[pos] = point.hi;
}
