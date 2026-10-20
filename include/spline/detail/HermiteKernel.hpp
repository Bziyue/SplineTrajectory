/*
    MIT License

    Copyright (c) 2025 Deping Zhang (beiyuena@foxmail.com)

    Permission is hereby granted, free of charge, to any person obtaining a copy
    of this software and associated documentation files (the "Software"), to deal
    in the Software without restriction, including without limitation the rights
    to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
    copies of the Software, and to permit persons to whom the Software is
    furnished to do so, subject to the following conditions:

    The above copyright notice and this permission notice shall be included in all
    copies or substantial portions of the Software.

    THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
    IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
    FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
    AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
    LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
    OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
    SOFTWARE.
*/

/** @file HermiteKernel.hpp
 * @brief Exact unit-interval Hermite and energy constants for orders 2, 3 and 4.
 * Endpoint columns are [p, v, ...] at the start followed by those at the end.
 * H maps endpoint derivatives to ascending powers. Q = H^T G H, where
 * G integrates the squared S-th derivative on [0,1]. E converts that derivative
 * from powers to shifted Legendre polynomials, whose squared norms are 1/(2j+1).
 * These small compile-time tables have no per-instance storage or initialization. */
#ifndef SPLINE_HERMITE_KERNEL_HPP
#define SPLINE_HERMITE_KERNEL_HPP
#include <type_traits>
#include <Eigen/Core>
namespace SplineTrajectory { namespace detail {
/** @brief Unit-interval constants; only the supported orders have definitions. */
template<int S> struct HermiteKernel;
/** @brief Visit fixed indices so zeros and matrix sizes disappear at compile time.
 * @param function Callable receiving an integral_constant index. */
template<int Begin, int End, class Function>
EIGEN_ALWAYS_INLINE void staticFor(Function &&function)
{
    if constexpr (Begin < End)
    {
        function(std::integral_constant<int, Begin>{});
        staticFor<Begin + 1, End>(function);
    }
}
/** @brief Small nonnegative factorial, evaluated at compile time for kernel indices.
 * @param n Nonnegative integer no greater than seven. @return n!. */
constexpr double factorial(int n) { return n < 2 ? 1.0 : n * factorial(n - 1); }
/** @brief Minimum derivative order 2: degree 3. */
template<> struct HermiteKernel<2>
{
    inline static constexpr double H[4][4] = {
        {1.0, 0.0, 0.0, 0.0},
        {0.0, 1.0, 0.0, 0.0},
        {-3.0, -2.0, 3.0, -1.0},
        {2.0, 1.0, -2.0, 1.0},
    };
    inline static constexpr double Q[4][4] = {
        {12.0, 6.0, -12.0, 6.0},
        {6.0, 4.0, -6.0, 2.0},
        {-12.0, -6.0, 12.0, -6.0},
        {6.0, 2.0, -6.0, 4.0},
    };
    inline static constexpr double E[2][2] = {
        {2.0, 3.0},
        {0.0, 3.0},
    };
};
/** @brief Minimum derivative order 3: degree 5. */
template<> struct HermiteKernel<3>
{
    inline static constexpr double H[6][6] = {
        {1.0, 0.0, 0.0, 0.0, 0.0, 0.0},
        {0.0, 1.0, 0.0, 0.0, 0.0, 0.0},
        {0.0, 0.0, (1.0 / 2.0), 0.0, 0.0, 0.0},
        {-10.0, -6.0, (-3.0 / 2.0), 10.0, -4.0, (1.0 / 2.0)},
        {15.0, 8.0, (3.0 / 2.0), -15.0, 7.0, -1.0},
        {-6.0, -3.0, (-1.0 / 2.0), 6.0, -3.0, (1.0 / 2.0)},
    };
    inline static constexpr double Q[6][6] = {
        {720.0, 360.0, 60.0, -720.0, 360.0, -60.0},
        {360.0, 192.0, 36.0, -360.0, 168.0, -24.0},
        {60.0, 36.0, 9.0, -60.0, 24.0, -3.0},
        {-720.0, -360.0, -60.0, 720.0, -360.0, 60.0},
        {360.0, 168.0, 24.0, -360.0, 192.0, -36.0},
        {-60.0, -24.0, -3.0, 60.0, -36.0, 9.0},
    };
    inline static constexpr double E[3][3] = {
        {6.0, 12.0, 20.0},
        {0.0, 12.0, 30.0},
        {0.0, 0.0, 10.0},
    };
};
/** @brief Minimum derivative order 4: degree 7. */
template<> struct HermiteKernel<4>
{
    inline static constexpr double H[8][8] = {
        {1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0},
        {0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0},
        {0.0, 0.0, (1.0 / 2.0), 0.0, 0.0, 0.0, 0.0, 0.0},
        {0.0, 0.0, 0.0, (1.0 / 6.0), 0.0, 0.0, 0.0, 0.0},
        {-35.0, -20.0, -5.0, (-2.0 / 3.0), 35.0, -15.0, (5.0 / 2.0), (-1.0 / 6.0)},
        {84.0, 45.0, 10.0, 1.0, -84.0, 39.0, -7.0, (1.0 / 2.0)},
        {-70.0, -36.0, (-15.0 / 2.0), (-2.0 / 3.0), 70.0, -34.0, (13.0 / 2.0), (-1.0 / 2.0)},
        {20.0, 10.0, 2.0, (1.0 / 6.0), -20.0, 10.0, -2.0, (1.0 / 6.0)},
    };
    inline static constexpr double Q[8][8] = {
        {100800.0, 50400.0, 10080.0, 840.0, -100800.0, 50400.0, -10080.0, 840.0},
        {50400.0, 25920.0, 5400.0, 480.0, -50400.0, 24480.0, -4680.0, 360.0},
        {10080.0, 5400.0, 1200.0, 120.0, -10080.0, 4680.0, -840.0, 60.0},
        {840.0, 480.0, 120.0, 16.0, -840.0, 360.0, -60.0, 4.0},
        {-100800.0, -50400.0, -10080.0, -840.0, 100800.0, -50400.0, 10080.0, -840.0},
        {50400.0, 24480.0, 4680.0, 360.0, -50400.0, 25920.0, -5400.0, 480.0},
        {-10080.0, -4680.0, -840.0, -60.0, 10080.0, -5400.0, 1200.0, -120.0},
        {840.0, 360.0, 60.0, 4.0, -840.0, 480.0, -120.0, 16.0},
    };
    inline static constexpr double E[4][4] = {
        {24.0, 60.0, 120.0, 210.0},
        {0.0, 60.0, 180.0, 378.0},
        {0.0, 0.0, 60.0, 210.0},
        {0.0, 0.0, 0.0, 42.0},
    };
};
}} // namespace SplineTrajectory::detail
#endif
