/*! \file helperFuncs.cpp
 *  \brief Helper functions for testing.
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

#include <iostream>
#include <iomanip>
#include <functional>
#include <chrono>
#include <random>
#include <GuidedFilter/tests/helperFuncs.hpp>


namespace
{
    // Random number generators setup
    auto seed = std::chrono::system_clock::now ().time_since_epoch ().count ();
    std::default_random_engine generator { seed };
    std::uniform_int_distribution<unsigned char> distribution1 { 0, 255 };
    std::uniform_int_distribution<unsigned short> distribution2 { 0, 10000 };
    std::uniform_real_distribution<float> distributionR1 { 0.f, 1.f };
    std::uniform_real_distribution<float> distributionR2 { 1e-6f, 255 * 1e-6f };
}

/*! \brief Uniform number generator in the range \f$[0, 255]\f$. */
std::function<unsigned char ()> rNum_0_255 = std::bind (distribution1, generator);
/*! \brief Uniform number generator in the range \f$[0, 10000]\f$. */
std::function<unsigned short ()> rNum_0_10000 = std::bind (distribution2, generator);
/*! \brief Uniform number generator in the range \f$[0.0, 1.0)\f$. */
std::function<float ()> rNum_R_0_1 = std::bind (distributionR1, generator);
/*! \brief Uniform number generator in the range \f$[1e-6, 255*1e-6)\f$. */
std::function<float ()> rNum_R_1_255_E__6 = std::bind (distributionR2, generator);


/*! \param[in] argc command line argument count
 *  \param[in] argv command line arguments
 *  \return A flag to indicate whether a command line argument 
 *          for profiling was provided.
 */
bool setProfilingFlag (int argc, char **argv)
{
    std::string prof = "--profiling";

    for (int8_t i = 0; i < argc; ++i)
        if (prof.compare (argv[i]) == 0)
            return true;

    return false;
}
