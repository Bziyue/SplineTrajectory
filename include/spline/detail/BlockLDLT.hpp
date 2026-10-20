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

/** @file BlockLDLT.hpp
 * @brief Inverse-free symmetric block tridiagonal LDLT for fixed small blocks. */
#ifndef SPLINE_BLOCK_LDLT_HPP
#define SPLINE_BLOCK_LDLT_HPP
#include <Eigen/Dense>
#include <vector>
#include <array>
#include <cmath>

namespace SplineTrajectory { namespace detail {
/** @brief Reusable factorization of an SPD block tridiagonal matrix.
 * @tparam M Block size (one, two or three). Blocks and coordinate rows are contiguous.
 * @note One owner must serialize factorization and solves. Storage is reused for a fixed topology. */
template<int M>
class BlockLDLT
{
    using Block = Eigen::Matrix<double, M, M, (M == 1 ? Eigen::ColMajor : Eigen::RowMajor)>;
    struct Factor
    {
        // Pack the strict-lower triangle, previous-knot coupling and reciprocal pivots.
        // No unused upper triangle or duplicate transpose cache crosses the forward/reverse sweeps.
        static constexpr int Triangle = M*(M-1)/2;
        std::array<double,Triangle+M*M+M> data;
        /** @brief Access strict-lower entry. @param r Row greater than c. @param c Column. @return Stored factor. */
        double &lower(int r,int c) { return data[r*(r-1)/2+c]; }
        /** @brief Read strict-lower entry. @param r Row greater than c. @param c Column. @return Stored factor. */
        double lower(int r,int c) const { return data[r*(r-1)/2+c]; }
        /** @brief Borrow coupling block. @return Mutable unaligned map owned by this factor. */
        Eigen::Map<Block> previous() { return Eigen::Map<Block>(data.data()+Triangle); }
        /** @brief Borrow coupling block. @return Const unaligned map owned by this factor. */
        Eigen::Map<const Block> previous() const { return Eigen::Map<const Block>(data.data()+Triangle); }
        /** @brief Access reciprocal pivot. @param r Row. @return Stored reciprocal. */
        double &inverse_pivot(int r) { return data[Triangle+M*M+r]; }
        /** @brief Read reciprocal pivot. @param r Row. @return Stored reciprocal. */
        double inverse_pivot(int r) const { return data[Triangle+M*M+r]; }
    };
    std::vector<Factor, Eigen::aligned_allocator<Factor>> factors_;
public:
    /** @brief Prepare storage without initializing numeric factors. @param count Interior knot count. */
    void resize(int count) { factors_.resize(count); }

    /** @brief Append one symmetric Schur block in increasing index order.
     * @param index Interior block index; all previous blocks must be factored.
     * @param diagonal Physical diagonal block.
     * @param previous Physical coupling to the preceding block; ignored for index zero.
     * @return False for a nonfinite or nonpositive pivot; no regularization is applied. */
    bool factor(int index, const Block &diagonal, const Block &previous)
    {
        auto &f = factors_[index];
        Block schur;
        for (int r = 0; r < M; ++r)
        {
            if (!(diagonal(r,r) > 0.0) || !std::isfinite(diagonal(r,r))) return false;
        }
        for (int r = 0; r < M; ++r)
            for (int c = 0; c <= r; ++c)
                schur(r,c) = diagonal(r,c);

        // Eliminate the preceding knot using its triangular factors, never a block inverse.
        if (index > 0)
        {
            const auto &p = factors_[index - 1];
            Block transformed;
            for (int r = 0; r < M; ++r)
                for (int c = 0; c < M; ++c)
                {
                    double value = previous(r,c);
                    for (int k = 0; k < c; ++k) value -= transformed(r,k) * p.lower(c,k);
                    transformed(r,c) = value;
                    f.previous()(r,c) = value * p.inverse_pivot(c);
                }
            for (int r = 0; r < M; ++r)
                for (int c = 0; c <= r; ++c)
                    for (int k = 0; k < M; ++k)
                        schur(r,c) -= f.previous()(r,k) * transformed(c,k);
        }

        // Unit-lower LDLT uses reciprocals only; reject breakdown before publishing coefficients.
        Eigen::Matrix<double, M, 1> pivot;
        for (int c = 0; c < M; ++c)
        {
            double value = schur(c,c);
            for (int k = 0; k < c; ++k) value -= f.lower(c,k) * f.lower(c,k) * pivot(k);
            if (!(value > 0.0) || !std::isfinite(value)) return false;
            pivot(c) = value;
            f.inverse_pivot(c) = 1.0 / value;
            for (int r = c + 1; r < M; ++r)
            {
                double lower = schur(r,c);
                for (int k = 0; k < c; ++k) lower -= f.lower(r,k) * f.lower(c,k) * pivot(k);
                f.lower(r,c) = lower * f.inverse_pivot(c);
            }
        }
        return true;
    }

    /** @brief Solve A X = B in place using the same factors for primal and adjoint systems.
     * @param[in,out] values Physical right-hand sides, then physical solutions; rows are knot-major.
     * @param offset First row of the interior knots; endpoint rows are untouched. */
    template<class Matrix>
    void solve(Matrix &values, int offset) const
    {
        const int count = static_cast<int>(factors_.size());
        for (int i = 0; i < count; ++i)
        {
            const auto &f = factors_[i];
            auto block = values.template middleRows<M>(offset + M * i);
            for (int r = 0; r < M; ++r)
            {
                if (i > 0)
                    for (int k = 0; k < M; ++k)
                        block.row(r) -= f.previous()(r,k) * values.row(offset + M * (i-1) + k);
                for (int k = 0; k < r; ++k) block.row(r) -= f.lower(r,k) * block.row(k);
            }
        }
        for (int i = count - 1; i >= 0; --i)
        {
            const auto &f = factors_[i];
            auto block = values.template middleRows<M>(offset + M * i);
            for (int r = M - 1; r >= 0; --r)
            {
                block.row(r) *= f.inverse_pivot(r);
                if (i + 1 < count)
                    for (int k = 0; k < M; ++k)
                        block.row(r) -= factors_[i+1].previous()(k,r) * values.row(offset + M * (i+1) + k);
                for (int k = r + 1; k < M; ++k) block.row(r) -= f.lower(k,r) * block.row(k);
            }
        }
    }
};
}} // namespace SplineTrajectory::detail
#endif
