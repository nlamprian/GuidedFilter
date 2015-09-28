/*! \file guidedFilter_kernels.cl
 *  \brief Kernels for performing `Guided Image Filtering`.
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


/*! \brief Computes the `a` and `b` coefficients in the Guided Filter algorithm.
 *  \note The global workspace should be one-dimensional. The **x** dimension 
 *        of the global workspace, \f$ gXdim \f$, should be equal to the 
 *        number of elements in the arrays, `M x N`, divided by 4. That is, 
 *        \f$ \ gXdim = M*N/4 \f$. The local workspace is irrelevant.
 *
 *  \param[in] mean_p array of average \f$ p \f$ values in the local windows.
 *  \param[in] mean_p2 array of average \f$ p^2 \f$ values in the local windows.
 *  \param[out] a array of \f$ a \f$ coefficients for the local models.
 *  \param[out] b array of \f$ b \f$ coefficients for the local models.
 *  \param[in] eps regularization parameter \f$ \epsilon \f$.
 */
kernel
void gf_ab (global float4 *mean_p, global float4 *mean_p2, 
            global float4 *a, global float4 *b, float eps)
{
    int gX = get_global_id (0);
    
    float4 m_p = mean_p[gX];
    float4 var_p = mean_p2[gX] - pown (m_p, 2);
    float4 a_ = var_p / (var_p + eps);
    
    a[gX] = a_;
    b[gX] = (1.f - a_) * m_p;
}


/*! \brief Computes the filtered output `q` in the Guided Filter algorithm.
 *  \details x.
 *  \note The global workspace should be one-dimensional. The **x** dimension 
 *        of the global workspace, \f$ gXdim \f$, should be equal to the 
 *        number of elements in the arrays, `M x N`, divided by 4. That is, 
 *        \f$ \ gXdim = M*N/4 \f$. The local workspace is irrelevant.
 *  \note When dealing with depth images, there might be invalid pixels
 *        (for Kinect, those pixels have value \f$ 0 \f$). Those invalid pixels 
 *        define surfaces that the Guided Filter algorithm tries to smooth out / 
 *        bring closer to nearby surfaces. The result is that those pixels 
 *        end up all over the place. By setting the `zero_out` flag, a procedure 
 *        is enabled for zeroing out in the \f$ q \f$ array those pixels
 *        that are zero in the \f$ p \f$ array.
 *
 *  \param[in] p input array \f$ p \f$.
 *  \param[in] mean_a array of average \f$ a \f$ values in the local windows.
 *  \param[in] mean_b array of average \f$ b \f$ values in the local windows.
 *  \param[in] q output array \f$ q \f$.
 *  \param[in] zero_out flag to indicate whether to zero out invalid pixels.
 *  \param[in] scaling factor by which to scale the pixel values in the output array.
 */
kernel
void gf_q (global float4 *p, global float4 *mean_a, global float4 *mean_b, 
           global float4 *q, int zero_out, float scaling)
{
    int gX = get_global_id (0);

    float4 p_ = p[gX];
    float4 q_ = mean_a[gX] * p_ + mean_b[gX];

    // Find the zero pixels in p, and if zeroing is enabled,
    // zero out the corresponding pixels in q
    int4 p_select = isequal (p_, 0.f) * zero_out;
    float4 q_z = select(q_, 0.f, p_select);

    q[gX] = scaling * q_z;
}


/*! \brief Computes the `a` and `b` coefficients in the Guided Filter algorithm.
 *  \note The global workspace should be one-dimensional. The **x** dimension 
 *        of the global workspace, \f$ gXdim \f$, should be equal to the 
 *        number of elements in the arrays, `M x N`, divided by 4. That is, 
 *        \f$ \ gXdim = M*N/4 \f$. The local workspace is irrelevant.
 *
 *  \param[in] corr_I array of average \f$ I*I \f$ values in the local windows.
 *  \param[in] corr_Ip array of average \f$ I*p \f$ values in the local windows.
 *  \param[in] mean_I array of average \f$ I \f$ values in the local windows.
 *  \param[in] mean_p array of average \f$ p \f$ values in the local windows.
 *  \param[out] var_I array of variance values for \f$ p \f$ in the local windows.
 *  \param[out] cov_Ip array of covariance values for \f$ I,p \f$ in the local windows.
 */
kernel
void gf_var_Ip (global float4 *corr_I, global float4 *corr_Ip, 
                global float4 *mean_I, global float4 *mean_p, 
                global float4 *var_I, global float4 *cov_Ip)
{
    int gX = get_global_id (0);
    
    float4 m_I = mean_I[gX];

    var_I[gX] = corr_I[gX] - m_I * m_I;
    cov_Ip[gX] = corr_Ip[gX] - m_I * mean_p[gX];
}


/*! \brief Computes the `a` and `b` coefficients in the Guided Filter algorithm.
 *  \note The global workspace should be one-dimensional. The **x** dimension 
 *        of the global workspace, \f$ gXdim \f$, should be equal to the 
 *        number of elements in the arrays, `M x N`, divided by 4. That is, 
 *        \f$ \ gXdim = M*N/4 \f$. The local workspace is irrelevant.
 *
 *  \param[in] var_I array of variance values for \f$ p \f$ in the local windows.
 *  \param[in] cov_Ip array of covariance values for \f$ I,p \f$ in the local windows.
 *  \param[in] mean_I array of average \f$ I \f$ values in the local windows.
 *  \param[in] mean_p array of average \f$ p \f$ values in the local windows.
 *  \param[out] a array of \f$ a \f$ coefficients for the local models.
 *  \param[out] b array of \f$ b \f$ coefficients for the local models.
 *  \param[in] eps regularization parameter \f$ \epsilon \f$.
 */
kernel
void gf_ab_Ip (global float4 *var_I, global float4 *cov_Ip, 
               global float4 *mean_I, global float4 *mean_p, 
               global float4 *a, global float4 *b, float eps)
{
    int gX = get_global_id (0);
    
    float4 a_ = cov_Ip[gX] / (var_I[gX] + eps);
    
    a[gX] = a_;
    b[gX] = mean_p[gX] - a_ * mean_I[gX];
}
