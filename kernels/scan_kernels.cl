/*! \file scan_kernels.cl
 *  \brief Kernels for performing the `Scan` operation.
 *  \author Nick Lamprianidis
 *  \version 2.0
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


/*! \brief Performs an inclusive scan operation on the columns of an array.
 *  \details The parallel scan algorithm by [Blelloch][1] is implemented.
 *           [1]: http://http.developer.nvidia.com/GPUGems3/gpugems3_ch39.html
 *  \note When there are multiple rows in the array, a scan operation is 
 *        performed per row, in parallel.
 *  \note The number of elements, `N`, in a row of the array should be a **multiple 
 *        of 4** (the data are handled as `float4`). The **x** dimension of the 
 *        global workspace, \f$ gXdim \f$, should be greater than or equal to the number of 
 *        elements in a row of the array divided by 8. That is, \f$ \ gXdim \geq N/8 \f$. 
 *        Each work-item handles `8 float` (= `2 float4`) elements in a row of the array. 
 *        The **y** dimension of the global workspace, \f$ gYdim \f$, should be equal 
 *        to the number of rows, `M`, in the array. That is, \f$ \ gYdim = M \f$. 
 *        The local workspace should be `1` in the **y** dimension, and a 
 *        **power of 2** in the **x** dimension. It is recommended 
 *        to use one `wavefront/warp` per work-group.
 *  \note When the number of elements per row of the array is small enough to be 
 *        handled by a single work-group, the output array will contain the true 
 *        scan result. When the elements are more than that, they are partitioned 
 *        into blocks and scanned independently. In this case, the kernel outputs 
 *        the results from each block scan operation. A scan should then be made on 
 *        the sums of the elements of each block per row. Finally, the results from 
 *        the last block-sums scan should be added in the corresponding block. The 
 *        number of work-groups in the **x** dimension, \f$ wgXdim \f$, **for the 
 *        case of multiple work-groups**, should be made a **multiple of 4**. The 
 *        potential extra work-groups are used for enforcing correctness. They write 
 *        the necessary identity operands, `0`, in the sums array, since in the 
 *        next phase the sums array is going to be handled as `float4`.
 *
 *  \param[in] in input array of `float` elements.
 *  \param[out] out (scan per work-group) output array of `float` elements.
 *  \param[in] data local buffer. Its size should be `2 float` elements for each 
 *                  work-item in a work-group. That is \f$ 2*lXdim*sizeof\ (float) \f$.
 *  \param[out] sums array of block sums. Each work-group outputs the sum of its elements. 
 *                   It's size should be \f$ M \times wgXdim \f$.
 *  \param[in] n the number of elements in a row of the array divided by 4.
 *  \param[in] scaling factor by which to scale the array elements before processing.
 */
kernel
void inclusiveScan_f (global float4 *in, global float4 *out, local float *data, 
                      global float *sums, uint n, float scaling)
{
    // Workspace dimensions
    uint lXdim = get_local_size (0);
    uint wgXdim = get_num_groups (0);

    // Workspace indices
    uint gX = get_global_id (0);
    uint gY = get_global_id (1);
    uint lX = get_local_id (0);
    uint wgX = get_group_id (0);

    uint offset = 1;

    // Load 8 float elements per work-item
    int4 flag = (int4) (2 * gX) < (int4) (n);
    float4 a = select ((float4) (0.f), in[gY * n + 2 * gX] * scaling, flag);
    flag = (int4) (2 * gX + 1) < (int4) (n);
    float4 b = select ((float4) (0.f), in[gY * n + 2 * gX + 1] * scaling, flag);

    // Perform a serial scan on the 2 float4 elements
    a.y += a.x; a.z += a.y; a.w += a.z;
    b.y += b.x; b.z += b.y; b.w += b.z;

    // Store the sum of each float4 element
    data[2 * lX] = a.w;
    data[2 * lX + 1] = b.w;

    // Perform a scan on the float4 sums

    // Up-Sweep phase
    for (uint d = lXdim; d > 0; d >>= 1)
    {
        barrier (CLK_LOCAL_MEM_FENCE);
        if (lX < d)
        {
            uint ai = offset * (2 * lX + 1) - 1;
            uint bi = offset * (2 * lX + 2) - 1;
            data[bi] += data[ai];
        }
        offset <<= 1;
    }

    // Store the work-group sum
    if ((wgXdim != 1) && (lX == lXdim - 1))
        sums[gY * wgXdim + wgX] = data[2 * lX + 1];

    // Clear the last register
    if (lX == (lXdim - 1))
        data[2 * lX + 1] = 0.f;

    // Down-Sweep phase
    for (uint d = 1; d < (2 * lXdim); d <<= 1)
    {
        offset >>= 1;
        barrier (CLK_LOCAL_MEM_FENCE);
        if (lX < d)
        {
            uint ai = offset * (2 * lX + 1) - 1;
            uint bi = offset * (2 * lX + 2) - 1;
            float tmp = data[ai];
            data[ai] = data[bi];
            data[bi] += tmp;
        }
    }
    barrier (CLK_LOCAL_MEM_FENCE);

    // Update the sums on the float4 elements
    // and store the results
    if ((2 * gX) < n)
    {
        a += data[2 * lX];
        out[gY * n + 2 * gX] = a;
    }

    if ((2 * gX + 1) < n)
    {
        b += data[2 * lX + 1];
        out[gY * n + 2 * gX + 1] = b;
    }
}


/*! \brief Adds the group sums in the associated blocks.
 *  \details It's the second part of the [Blelloch][1] scan algorithm.
 *           [1]: http://http.developer.nvidia.com/GPUGems3/gpugems3_ch39.html
 *  \note `scan` handled `2 float4` elements per work-item. `addGroupSums`
 *        handles `1 float4` element per work-item. The global workspace should 
 *        be \f$ 2*(wgXdim-1)*lXdim_{scan} \f$ in the **x** dimension, and \f$ M \f$ 
 *        in the **y** dimension. The global workspace should also have an offset 
 *        \f$ 2*lXdim_{scan} \f$ in the **x** dimension. The local workspace
 *        should be \f$ 2*lXdim_{scan} \f$ in the **x** dimension, and `1`
 *        in the **y** dimension.
 *  \note This part should follow after a scan has been performed on the group sums.
 *
 *  \param[in] sums (scan) array of work-group sums. Its size is \f$M \times wgXdim\f$.
 *  \param[out] out (scan) output array of `float` elements (before processing, it 
 *                  contains the block scans performed in a previous step.
 *  \param[in] n the number of elements in a row of the array divided by 4.
 */
kernel
void addGroupSums_f (global float *sums, global float4 *out, uint n)
{
    // Workspace dimensions
    uint wgXdim = get_num_groups (0);

    // Workspace indices
    uint gX = get_global_id (0);
    uint gY = get_global_id (1);
    uint wgX = get_group_id (0);

    float sum = sums[gY * (wgXdim + 1) + wgX];

    if (gX < n)
        out[gY * n + gX] += sum;
}
