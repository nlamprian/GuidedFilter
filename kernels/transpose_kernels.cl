/*! \file transpose_kernels.cl
 *  \brief Kernels for performing the `Transpose` operation.
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


/*! \brief Performs a matrix transposition.
 *  \note Both dimensions of the matrix have to be **multiples of 4**. Other 
 *        than that, the matrix can have any dimensions ratio. It doesn't 
 *        have to be square.
 *  \note The **x** dimension of the global workspace, \f$ gXdim \f$, should be 
 *        equal to the number of columns, `N`, in the matrix divided by 4. That 
 *        is, \f$ \ gXdim = N/4 \f$. The **y** dimension of the global workspace, 
 *        \f$ gYdim \f$, should be equal to the number of rows, `M`, in the 
 *        matrix divided by 4. That is, \f$ \ gYdim = M/4 \f$. The local 
 *        workspace should be **square**. That is, \f$ \ lXdim = lYdim \f$.
 *  \note Each work-item transposes a square `4x4` block.
 *
 *  \param[in] in input matrix of `float` elements.
 *  \param[out] out output (transposed) matrix of `float` elements.
 *  \param[in] data local buffer. Its size should be `16 float` elements for 
 *                  each work-item in a work-group. That is, \f$ 4*4*lXdim*lYdim*sizeof\ (float) \f$.
 */
kernel
void transpose (global float4 *in, global float4 *out, local float4 *data)
{
    // Workspace dimensions
    uint gXdim = get_global_size (0);
    uint gYdim = get_global_size (1);
    uint lXdim = get_local_size (0);

    // Workspace indices
    uint lX = get_local_id (0);
    uint lY = get_local_id (1);
    uint wgX = get_group_id (0);
    uint wgY = get_group_id (1);

    uint baseIn = 4 * (wgY * lXdim + lY) * gXdim + (wgX * lXdim + lX);
    uint baseOut = 4 * (wgX * lXdim + lY) * gYdim + (wgY * lXdim + lX);

    // Load a 4x4 block of data within
    // a larger work-group block.
    uint idx = 4 * lY * lXdim + lX;
    data[idx]             = in[baseIn];
    data[idx + lXdim]     = in[baseIn + gXdim];
    data[idx + 2 * lXdim] = in[baseIn + 2 * gXdim];
    data[idx + 3 * lXdim] = in[baseIn + 3 * gXdim];

    barrier (CLK_LOCAL_MEM_FENCE);

    // Read a 4x4 block of data from 
    // the transposed position within 
    // the larger work-group block.
    uint idxTr = 4 * lX * lXdim + lY;
    float4 a = data[idxTr];
    float4 b = data[idxTr + lXdim];
    float4 c = data[idxTr + 2 * lXdim];
    float4 d = data[idxTr + 3 * lXdim];

    // Transpose the 4x4 block of data, and store it 
    // at the same position within the work-group block, 
    // but at the transposed position of the work-group 
    // block within the matrix.
    out[baseOut]             = (float4) (a.x, b.x, c.x, d.x);
    out[baseOut + gYdim]     = (float4) (a.y, b.y, c.y, d.y);
    out[baseOut + 2 * gYdim] = (float4) (a.z, b.z, c.z, d.z);
    out[baseOut + 3 * gYdim] = (float4) (a.w, b.w, c.w, d.w);
}
