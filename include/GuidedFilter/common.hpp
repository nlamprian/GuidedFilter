/*! \file common.hpp
 *  \brief Declares classes used by the OpenCL interface classes in `cl_algo`.
 *  \author Nick Lamprianidis
 *  \version 1.1.1
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

#ifndef COMMON_HPP
#define COMMON_HPP


/*! \brief Offers classes which set up kernel execution parameters and 
 *         provide interfaces for the handling of memory objects.
 */
namespace cl_algo
{
namespace GF
{

    /*! \brief Enumerates staging buffer configurations.
     *  \details It's meant to be used when making a call to the `init` 
     *           method of one of the `cl_algo` classes. 
     *           It specifies which staging buffers to be instantiated.
     *  
     *  \param NONE do not instantiate any staging buffers.
     *  \param I instantiate the input staging buffers.
     *  \param O instantiate the output staging buffers.
     *  \param IO instantiate both input and output staging buffers.
     */
    enum class Staging : uint8_t { NONE, I, O, IO };

}
}

#endif  // COMMON_HPP