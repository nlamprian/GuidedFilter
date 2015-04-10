/*! \file prefixSum_kernels.cl
 *  \brief Kernels for performing the `Prefix Sum` operation.
 *  \note The kernels present in this file are developed with the goal of 
 *        maximum performance under the project's main application which is the 
 *        processing of 640x480 images on the Guided Image Filtering pipeline. 
 *        The algorithms themselves and their performance might not generalize 
 *        well under different environments. When possible, comments will be 
 *        made in the documentation of each kernel to try to highlight 
 *        the specific issues.
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


/*! \brief Performs an inclusive prefix sum scan on the columns of an array.
 *  \details The parallel scan algorithm by [Blelloch][1] is implemented. If 
 *           multiple work-groups per row are dispatched, `prefixSum` should be 
 *           followed by `addPartialSums`.
 *           [1]: http://http.developer.nvidia.com/GPUGems3/gpugems3_ch39.html
 *  \note When there are multiple rows in the array, a scan operation is 
 *        performed per row, in parallel.
 *  \note The number of elements, `N`, in a row of the array should be a multiple 
 *        of `4` (the data are handled as `float4`). The **x** dimension of the 
 *        global workspace, \f$ gXdim \f$, should be a **power of 2** and greater 
 *        than the number of elements in a row of the array divided by 8. 
 *        That is, \f$ \ gXdim \geq N/8 \f$. Each work-item handles `8 float` 
 *        (= `2 float4`) elements in a row of the array. The **y** dimension of
 *        the global workspace, \f$ gYdim \f$, should be equal to the number 
 *        of rows, `M`, in the array. That is, \f$ \ gYdim = M \f$. The local 
 *        workspace should be `1` in the **y** dimension, but you are free to 
 *        set the **x** dimension. It is recommended though to use one 
 *        `wavefront/warp` per work-group.
 *
 *  \param[in] in input array of `float` elements.
 *  \param[out] out (prefix sum per group) output array of `float` elements.
 *  \param[in] data local buffer. Its size should be `2 float` elements for each 
 *                  work-item in a work-group. That is \f$ 2*lXdim*sizeof\ (float) \f$.
 *  \param[out] sums array of (partial) sums. Each work-group but the last 
 *                   in each row of the array outputs the sum of its elements. 
 *                   It's size should be \f$ M \times (wgXdim-1) \f$.
 *  \param[in] n the number of elements in a row of the array divided by 4.
 *  \param[in] scaling factor by which to scale the array elements before processing.
 */
kernel
void prefixSum (global float4 *in, global float4 *out, local float *data, 
                global float *sums, uint n, float scaling)
{
    // Workspace dimensions
    uint gXdim = get_global_size (0);
    uint lXdim = get_local_size (0);
    uint wgXdim = get_num_groups (0);

    // Workspace indices
    uint gX = get_global_id (0);
    uint gY = get_global_id (1);
    uint lX = get_local_id (0);
    uint wgX = get_group_id (0);
    uint wgY = get_group_id (1);

    uint offset = 1;
    float4 a, b;

    if (gX < (n / 2))
    {
        a = in[gY * n + 2 * gX] * scaling;
        b = in[gY * n + 2 * gX + 1] * scaling;

        // Perform a serial scan 
        // on the 2 float4 elements
        a.y += a.x; a.z += a.y; a.w += a.z;
        b.y += b.x; b.z += b.y; b.w += b.z;

        // Store the sum of each float4
        data[2 * lX] = a.w;
        data[2 * lX + 1] = b.w;
    }

    // Perform a parallel sum scan 
    // on the float4 sums

    // Up-Sweep phase
    for (uint d = gXdim; d > 0; d >>= 1)
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

    // Store the partial group sum
    if (gX == (lXdim - 1) && wgX < (wgXdim - 1))
    {
        sums[wgY * (wgXdim - 1) + wgX] = data[2 * lX + 1];
    }

    if (lX == (lXdim - 1))
    {
        data[2 * lX + 1] = 0.f;
    }

    // Down-Sweep phase
    for (uint d = 1; d < (gXdim << 1); d <<= 1)
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

    if (gX < (n / 2))
    {
        // Update the partial sums
        // on the int4 elements
        a += data[2 * lX];
        b += data[2 * lX + 1];
        
        out[gY * n + 2 * gX] = a;
        out[gY * n + 2 * gX + 1] = b;
    }
}


/*! \brief Adds the partial sums of the first half of each row to the 
           second half of each row in the array.
 *  \details It's the second part of the [Blelloch][1] scan algorithm. 
 *           It should follow the execution of `prefixSum`.
 *           [1]: http://http.developer.nvidia.com/GPUGems3/gpugems3_ch39.html
 *  \note This part of the algorithm is not generalized. It assumes that 
 *        `prefixSum` was called with 2 work-groups in the **x** dimension 
 *        of the global workspace.
 *  \note `prefixSum` handled `2 float4` elements per work-item. `addPartialSums`
 *        handles `1 float4` element per work-item. The global workspace should 
 *        be \f$ 2*lXdim_{prefixSum} \f$ in the **x** dimension, and \f$ M \f$ 
 *        in the **y** dimension. The global workspace should also have an offset 
 *        \f$ 2*lXdim_{prefixSum} \f$ in the **x** dimension. The local workspace
 *        can be a null range.
 *  \note To generalize this part of the algorithm, perform a scan on each row 
 *        in the `pSum` array, and add the results to the corresponding parts 
 *        of the `pScan` array.
 *
 *  \param[in] pSum array of (partial) sums of the first half of each row 
 *                  in the input array to `prefixSum`. Its size is \f$M \times 1\f$.
 *  \param[out] pScan array of int elements with the partial sums from `prefixSum`.
 *  \param[in] n the number of elements in a row of the array divided by 4.
 */
kernel
void addPartialSums (constant float *pSum, global float4 *pScan, uint n)
{
    // Workspace indices
    uint gX = get_global_id (0);
    uint gY = get_global_id (1);

    float sum = pSum[gY];

    if (gX < n)
        pScan[gY * n + gX] += sum;
}


// / *! \brief Performs an inclusive prefix sum scan on the columns of an array.
//  *  \details Plain implementation with ints and bank conflicts. Needs one
//  *           additional int in shared memory to save the sum of the scan.
//  *  \note Out of date.
//  */
// kernel
// void prefixSum (global int *in, global int *out, local int *data, uint n)
// {
//     // Workspace dimensions
//     uint lXdim = get_local_size (0);

//     // Workspace indices
//     uint gX = get_global_id (0);
//     uint gY = get_global_id (1);
//     uint lX = get_local_id (0);

//     uint offset = 1;

//     data[2 * lX] = in[gY * n + 2 * gX];
//     data[2 * lX + 1] = in[gY * n + 2 * gX + 1];

//     for (uint d = n >> 1; d > 0; d >>= 1)
//     {
//         barrier (CLK_LOCAL_MEM_FENCE);
//         if (lX < d)
//         {
//             uint ai = offset * (2 * lX + 1) - 1;
//             uint bi = offset * (2 * lX + 2) - 1;
//             data[bi] += data[ai];
//         }
//         offset *= 2;
//     }

//     if (lX == (lXdim - 1))
//     {
//         data[2 * lX + 2] = data[2 * lX + 1];
//         data[2 * lX + 1] = 0;
//     }

//     for (uint d = 1; d < n; d *= 2)
//     {
//         offset >>= 1;
//         barrier (CLK_LOCAL_MEM_FENCE);
//         if (lX < d)
//         {
//             uint ai = offset * (2 * lX + 1) - 1;
//             uint bi = offset * (2 * lX + 2) - 1;
//             int tmp = data[ai];
//             data[ai] = data[bi];
//             data[bi] += tmp;
//         }
//     }

//     barrier (CLK_LOCAL_MEM_FENCE);
//     out[gY * n + 2 * gX] = data[2 * lX + 1];
//     out[gY * n + 2 * gX + 1] = data[2 * lX + 2];
// }


// / *! \brief Performs an inclusive prefix sum scan on the columns of an array.
//  *  \details Bank conflict free implementation. Adds one int padding in shared
//  *           memory per NUM_BANKS (= 32).
//  *  \note Out of date.
//  */
// kernel
// void prefixSum (global int4 *in, global int4 *out, local int *data, uint n)
// {
//     // Workspace dimensions
//     uint lXdim = get_local_size (0);

//     // Workspace indices
//     uint gX = get_global_id (0);
//     uint gY = get_global_id (1);
//     uint lX = get_local_id (0);
    
//     uint base1 = 2 * lX;
//     uint base2 = 2 * lX + 1;
//     uint off1 = (base1) >> 5;
//     uint off2 = (base2) >> 5;
//     uint sh1 = base1 + off1;
//     uint sh2 = base2 + off2;

//     uint offset = 1;

//     int4 a = in[gY * n + 2 * gX];
//     int4 b = in[gY * n + 2 * gX + 1];

//     a.y += a.x; a.z += a.y; a.w += a.z;
//     b.y += b.x; b.z += b.y; b.w += b.z;

//     data[sh1] = a.w;
//     data[sh2] = b.w;

//     for (uint d = n >> 1; d > 0; d >>= 1)
//     {
//         barrier (CLK_LOCAL_MEM_FENCE);
//         if (lX < d)
//         {
//             uint ai = offset * (base1 + 1) - 1;
//             uint bi = offset * (base1 + 2) - 1;
//             data[(bi >> 5) + bi] += data[(ai >> 5) + ai];
//         }
//         offset *= 2;
//     }

//     if (lX == (lXdim - 1))
//     {
//         data[(base2 >> 5) + base2] = 0;
//     }

//     for (uint d = 1; d < n; d *= 2)
//     {
//         offset >>= 1;
//         barrier (CLK_LOCAL_MEM_FENCE);
//         if (lX < d)
//         {
//             uint ai = offset * (base1 + 1) - 1;
//             uint bi = offset * (base1 + 2) - 1;
//             int tmp = data[(ai >> 5) + ai];
//             data[(ai >> 5) + ai] = data[(bi >> 5) + bi];
//             data[(bi >> 5) + bi] += tmp;
//         }
//     }

//     barrier (CLK_LOCAL_MEM_FENCE);
//     out[gY * n + 2 * gX] = a + data[sh1];
//     out[gY * n + 2 * gX + 1] = b + data[sh2];
// }
