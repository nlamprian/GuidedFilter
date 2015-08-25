/*! \file boxFilter_kernels.cl
 *  \brief Kernels for performing `Box Filtering`.
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


/*! \brief Performs box (mean) filtering.
 *  \details Accepts a SAT array, \f$ sat_{M \times N} \f$, performs the 
 *           filtering, and outputs the result, \f$ out_{M \times N} \f$.
 *           The work complexity is `O(1)` in the window size.
 *  \note The **x** dimension of the global workspace, \f$ gXdim \f$, should be 
 *        equal to the number of columns, `N`, in the image. That is, 
 *        \f$ \ gXdim = N \f$. The **y** dimension of the global workspace, 
 *        \f$ gYdim \f$, should be equal to the number of rows, `M`, in the 
 *        image. That is, \f$ \ gYdim = M \f$. The local workspace is irrelevant.
 *
 *  \param[in] sat input array of `float` elements.
 *  \param[out] out output (blurred) array of `float` elements.
 *  \param[in] radius radius of the square filter window.
 */
kernel
void boxFilterSAT (global float *sat, global float *out, int radius)
{
    // Workspace dimensions
    int gXdim = get_global_size (0);
    int gYdim = get_global_size (1);

    // Workspace indices
    int gX = get_global_id (0);
    int gY = get_global_id (1);

    // Filter window coordinates
    int2 c0 = { gX - radius - 1, gY - radius - 1 };                            // Top left corner indices
    int2 c1 = { min (gX + radius, gXdim - 1), min (gY + radius, gYdim - 1) };  // Bottom right corner indices
    int2 outOfBounds = isless (convert_float2 (c0), 0.f);

    float sum = 0.f;
    sum += select (sat[c0.y * gXdim + c0.x], 0.f, outOfBounds.x || outOfBounds.y);  // Top left corner
    sum -= select (sat[c0.y * gXdim + c1.x], 0.f, outOfBounds.y);                   // Top right corner
    sum -= select (sat[c1.y * gXdim + c0.x], 0.f, outOfBounds.x);                   // Bottom left corner
    sum +=         sat[c1.y * gXdim + c1.x];                                        // Bottom right corner

    // Number of elements in the filter window
    int2 d = c1 - select (c0, -1, outOfBounds);
    float n = d.x * d.y;

    // Store mean value
    out[gY * gXdim + gX] = sum / n;
}


#define NUM_BF_STORING_WORK_ITEMS 16 * 16 / 4  // 64

/*! \brief Performs box (mean) filtering.
 *  \details Accepts a transposed SAT array, \f$ sat_{N \times M} \f$, performs 
 *           the filtering, and outputs the result, \f$ out_{M \times N} \f$.
 *           The work complexity is `O(1)` in the window size.
 *  \note Both dimensions of the image have to be **multiples of the work-group 
 *        dimensions**, respectively. This specification could be overcome by 
 *        extending the buffers [clEnqueue(Read/Write)BufferRect] and including 
 *        bounds checking within the kernel. These cases won't be handled.
 *  \note The **x** dimension of the global workspace, \f$ gXdim \f$, should be 
 *        equal to the number of columns, `M`, in the SAT array. That is, 
 *        \f$ \ gXdim = M \f$. The **y** dimension of the global workspace, 
 *        \f$ gYdim \f$, should be equal to the number of rows, `N`, in the SAT
 *        array. That is, \f$ \ gYdim = N \f$. The local workspace should be 
 *        `16x16`. That is, \f$ \ lXdim = lYdim = 16 \f$.
 *  \note Each work-item filters one pixel, and then the first 64 work-items in 
 *        each work-group store a transposed 4 pixel block in global memory.

 *  \param[in] sat input array of `float` elements.
 *  \param[out] out output (blurred) array of `float` elements.
 *  \param[in] data local buffer. Its size should be `1 float` element for 
 *                  each work-item in a work-group. That is, \f$ lXdim*lYdim*sizeof\ (float) \f$.
 *  \param[in] radius radius of the square filter window.
 *  \param[in] scaling factor by which to scale the array elements after processing.
 */
kernel
void boxFilterSAT_Tr (global float *sat, global float4 *out, local float *data, int radius, float scaling)
{
    // Workspace dimensions
    int gXdim = get_global_size (0);
    int gYdim = get_global_size (1);
    int lXdim = get_local_size (0);
    int lYdim = get_local_size (1);

    // Workspace indices
    int gX = get_global_id (0);
    int gY = get_global_id (1);
    int lX = get_local_id (0);
    int lY = get_local_id (1);
    int wgX = get_group_id (0);
    int wgY = get_group_id (1);

    // Filter window coordinates
    int2 c0 = { gX - radius - 1, gY - radius - 1 };                            // Top left corner indices
    int2 c1 = { min (gX + radius, gXdim - 1), min (gY + radius, gYdim - 1) };  // Bottom right corner indices
    int2 outOfBounds = isless (convert_float2 (c0), 0.f);

    float sum = 0.f;
    sum += select (sat[c0.y * gXdim + c0.x], 0.f, outOfBounds.x || outOfBounds.y);  // Top left corner
    sum -= select (sat[c0.y * gXdim + c1.x], 0.f, outOfBounds.y);                   // Top right corner
    sum -= select (sat[c1.y * gXdim + c0.x], 0.f, outOfBounds.x);                   // Bottom left corner
    sum +=         sat[c1.y * gXdim + c1.x];                                        // Bottom right corner

    // Number of elements in the filter window
    int2 d = c1 - select (c0, -1, outOfBounds);
    float n = d.x * d.y;

    // Flatten indices
    int idx = lY * lXdim + lX;

    data[idx] = scaling * sum / n;
    barrier (CLK_LOCAL_MEM_FENCE);

    if (idx < NUM_BF_STORING_WORK_ITEMS)
    {
        // Read a transposed float4 element
        //* Elements are processed in column order
        int iy = idx % 4;
        int ix = idx / 4;
        int base = 4 * iy * lXdim + ix;
        float4 pixels = { data[base], 
                          data[base + lXdim], 
                          data[base + 2 * lXdim], 
                          data[base + 3 * lXdim] };

        // Store the float4 element witin the work-group block in the transposed position
        //* Elements are stored in row order. The block has already been transposed
        out[(wgX * lXdim + ix) * gYdim / 4 + (wgY * lYdim / 4 + iy)] = pixels;
    }
}


/*! \brief Performs box (mean) filtering.
 *  \details The work complexity is `O(n)` in the window size.
 *  \note Both dimensions of the image have to be **multiples of the work-group 
 *        dimensions**, respectively. This specification could be overcome by 
 *        extending the buffers [clEnqueue(Read/Write)BufferRect] and including 
 *        bounds checking within the kernel. These cases won't be handled.
 *  \note The **x** dimension of the global workspace, \f$ gXdim \f$, should be 
 *        equal to the number of columns, `N`, in the image. That is, 
 *        \f$ \ gXdim = N \f$. The **y** dimension of the global workspace, 
 *        \f$ gYdim \f$, should be equal to the number of rows, `M`, in the 
 *        image. That is, \f$ \ gYdim = M \f$. The local workspace should be 
 *        `16x16`. That is, \f$ \ lXdim = lYdim = 16 \f$.
 *  \note Vector reads are avoided since the required alignments complicate 
 *        memory address calculations.

 *  \param[in] in input array of `float` elements.
 *  \param[out] out output (blurred) array of `float` elements.
 *  \param[in] data local buffer. Its size should be `1 float` element for 
 *                  each work-item in a work-group and each halo pixel. 
 *                  That is, \f$ (lXdim+2*radius)*(lYdim+2*radius)*sizeof\ (float) \f$.
 *  \param[in] radius radius of the square filter window.
 */
kernel
void boxFilter (global float *in, global float *out, local float *data, int radius)
{
    // Workspace dimensions
    int gXdim = get_global_size (0);
    int gYdim = get_global_size (1);
    int lXdim = get_local_size (0);
    int lYdim = get_local_size (1);
    int lWidth = lXdim + 2 * radius;

    // Workspace indices
    int gX = get_global_id (0);
    int gY = get_global_id (1);
    int lX = get_local_id (0);
    int lY = get_local_id (1);

    // Load data in local memory
    for (int y = lY, iy = gY-radius; y < lYdim + 2 * radius; y += lYdim, iy += lYdim)
    {
        for (int x = lX, ix = gX-radius; x < lXdim + 2 * radius; x += lXdim, ix += lXdim)
        {
            uint flag = (ix >= 0 && iy >= 0 && ix < gXdim && iy < gYdim);
            data[y * lWidth + x] = select (0.f, in[iy * gXdim + ix], flag);
        }
    }
    barrier (CLK_LOCAL_MEM_FENCE);

    // Compute the sum of the filter window elements
    float sum = 0.f;
    for (int fRow = lY; fRow <= lY + 2 * radius; ++fRow)
        for (int fCol = lX; fCol <= lX + 2 * radius; ++fCol)
            sum += data[fRow * lWidth + fCol];

    // Filter window coordinates
    int2 c0 = { gX - radius - 1, gY - radius - 1 };                            // Top left corner indices
    int2 c1 = { min (gX + radius, gXdim - 1), min (gY + radius, gYdim - 1) };  // Bottom right corner indices
    int2 outOfBounds = c0 < 0;

    // Number of elements in the filter window
    int2 d = c1 - select (c0, -1, outOfBounds);
    float n = d.x * d.y;

    // Store mean value
    out[gY * gXdim + gX] = sum / n;
}
