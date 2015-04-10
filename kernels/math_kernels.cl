/*! \file math_kernels.cl
 *  \brief Kernels for performing `math` operations.
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


/*! \brief Multiplies two input arrays together, element-wise.
 *  \note The global workspace should be one-dimensional. The **x** dimension 
 *        of the global workspace, \f$ gXdim \f$, should be equal to the 
 *        number of elements in the array, `M x N`, divided by 4. That is, 
 *        \f$ \ gXdim = M*N/4 \f$. The local workspace is irrelevant.
 *
 *  \param[in] a first operand. Input array of `float` elements.
 *  \param[in] b second operand. Input array of `float` elements.
 *  \param[out] out product. Output array of `float` elements.
 */
kernel
void mult (global float4 *a, global float4 *b, global float4 *out)
{
    int gX = get_global_id (0);

	out[gX] = a[gX] * b[gX];
}


/*! \brief Raises an array to an integer power, element-wise.
 *  \note The global workspace should be one-dimensional. The **x** dimension 
 *        of the global workspace, \f$ gXdim \f$, should be equal to the 
 *        number of elements in the array, `M x N`, divided by 4. That is, 
 *        \f$ \ gXdim = M*N/4 \f$. The local workspace is irrelevant.
 *
 *  \param[in] in input array of `float` elements.
 *  \param[out] out output array of `float` elements.
 *  \param[in] n power to which to raise the array.
 */
kernel
void pown_ (global float4 *in, global float4 *out, int n)
{
    int gX = get_global_id (0);

	out[gX] = pown(in[gX], n);
}
