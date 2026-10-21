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

/** @file SplineTrajectory.hpp
 * @brief Owning piecewise polynomials and independent cubic, quintic and septic splines. */

#ifndef SPLINE_TRAJECTORY_HPP
#define SPLINE_TRAJECTORY_HPP

#include <Eigen/Dense>
#include <array>
#include <vector>
#include <algorithm>
#include <cmath>
#include <cstddef>
#include <iterator>
#include <limits>
#include <cassert>
#include <optional>
#include <utility>

namespace SplineTrajectory
{
    /** @brief Recoverable input, state, resource and numerical failures across the library. */
    enum class ErrorCode
    {
        Ok, InvalidInput, DimensionMismatch, InvalidState, OutOfRange,
        ResourceLimit, NumericalFailure, NotPrepared, SamplingNotPrepared, GradientMismatch
    };

    /** @brief Allocation-free operation status; diagnostic text has static storage duration. */
    struct [[nodiscard]] Status
    {
        ErrorCode code = ErrorCode::Ok;
        int index = -1;
        const char *message = "";

        /** @brief Construct success without allocating. */
        constexpr Status() noexcept = default;
        /** @brief Describe a failure or success without owning dynamic text.
         * @param value Error category. @param text Non-null immutable static text.
         * @param location Relevant segment, sample or input index; -1 when not applicable. */
        constexpr Status(ErrorCode value, const char *text, int location = -1) noexcept
            : code(value), index(location), message(text) {}
        /** @brief Check success. @return True exactly when the error code is Ok. */
        constexpr bool ok() const noexcept { return code == ErrorCode::Ok; }
        /** @brief Check success in a condition. @return Same as ok(). */
        constexpr explicit operator bool() const noexcept { return ok(); }
        /** @brief Construct success. @return An allocation-free successful status. */
        static constexpr Status success() noexcept { return {}; }
    };

    /** @brief An owned successful value or a failure; only T itself may allocate.
     * @tparam T Owned value type. Check ok() before accessing value(); no exceptions are
     *       used for access-contract violations. Independent objects are reentrant. */
    template <class T>
    class [[nodiscard]] StatusOr
    {
        Status status_;
        std::optional<T> value_;
    public:
        /** @brief Copy a successful value. @param value Value copied into this result. */
        StatusOr(const T &value) : value_(value) {}
        /** @brief Move a successful value into this result. @param value Owned source value. */
        StatusOr(T &&value) : value_(std::move(value)) {}
        /** @brief Own a failure without constructing T.
         * @param status Non-success status; passing success violates the constructor contract. */
        StatusOr(Status status) : status_(status)
        {
            assert(!status.ok());
        }
        /** @brief Inspect success. @return True when an owned value is present. */
        bool ok() const noexcept { return value_.has_value(); }
        /** @brief Inspect success in a condition. @return Same as ok(). */
        explicit operator bool() const noexcept { return ok(); }
        /** @brief Inspect status. @return Status valid independently of the result's lifetime. */
        Status status() const noexcept { return status_; }
        /** @brief Borrow the successful value. @return Reference valid until this result changes.
         * @pre ok() is true; concurrent mutation is not allowed. */
        const T &value() const & { assert(ok()); return *value_; }
        /** @brief Borrow the successful value for exclusive mutation.
         * @return Reference valid until this result changes. @pre ok() is true. */
        T &value() & { assert(ok()); return *value_; }
        /** @brief Move the successful value out. @return Rvalue reference to owned storage.
         * @pre ok() is true; the caller must finish moving before this result is destroyed. */
        T &&value() && { assert(ok()); return std::move(*value_); }
    };

    namespace detail
    {
        /** @brief Check dense numeric buffers without an early-exit branch per coefficient.
         * @param values Borrowed dense Eigen object; strided references are supported.
         * @return True exactly when every entry is finite, including empty buffers.
         * @note Integer flags allow the compiler to vectorize the scan with ordinary CPU
         *       instructions; no fast-math assumptions, allocation or shared writes are used. */
        template<class Derived>
        inline bool allFinite(const Eigen::DenseBase<Derived> &values)
        {
            unsigned int nonfinite = 0;
            if constexpr (Derived::Flags & Eigen::LinearAccessBit)
            {
                // Owned buffers expose one contiguous scan, including row-major 3D matrices.
                for (Eigen::Index i = 0; i < values.size(); ++i)
                    nonfinite |= !std::isfinite(values.derived().coeff(i));
            }
            else
            {
                for (Eigen::Index outer = 0; outer < values.outerSize(); ++outer)
                    for (Eigen::Index inner = 0; inner < values.innerSize(); ++inner)
                    {
                        const auto value = Derived::IsRowMajor ? values.derived().coeff(outer, inner)
                                                              : values.derived().coeff(inner, outer);
                        nonfinite |= !std::isfinite(value);
                    }
            }
            return nonfinite == 0;
        }
    }

    // Piecewise polynomials and shared input/output types.

    template <typename T>
    using SplineVector = std::vector<T, Eigen::aligned_allocator<T>>;

    /** @brief Physical boundary derivatives in caller-defined coordinates; positions are waypoints. */
    template <int DIM>
    struct BoundaryConditions
    {
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW

        using Vector = Eigen::Matrix<double, DIM, 1>;

        Vector start_velocity = Vector::Zero();
        Vector start_acceleration = Vector::Zero();
        Vector start_jerk = Vector::Zero();

        Vector end_velocity = Vector::Zero();
        Vector end_acceleration = Vector::Zero();
        Vector end_jerk = Vector::Zero();

        BoundaryConditions() = default;

        BoundaryConditions(const Vector &start_vel, const Vector &end_vel)
            : start_velocity(start_vel), end_velocity(end_vel) {}

        BoundaryConditions(const Vector &start_vel, const Vector &start_acc,
                           const Vector &end_vel, const Vector &end_acc)
            : start_velocity(start_vel), start_acceleration(start_acc),
              end_velocity(end_vel), end_acceleration(end_acc) {}

        BoundaryConditions(const Vector &start_vel, const Vector &start_acc, const Vector &start_jerk_value,
                           const Vector &end_vel, const Vector &end_acc, const Vector &end_jerk_value)
            : start_velocity(start_vel), start_acceleration(start_acc), start_jerk(start_jerk_value),
              end_velocity(end_vel), end_acceleration(end_acc), end_jerk(end_jerk_value) {}
    };

    enum class Deriv : int
    {
        Pos = 0,
        Vel = 1,
        Acc = 2,
        Jerk = 3,
        Snap = 4,
        Crackle = 5,
        Pop = 6
    };

    /** @brief Owning piecewise polynomial in caller-defined coordinates and seconds.
     * @tparam DIM Positive spatial dimension.
     * @tparam MaxCoefficients Compile-time coefficient capacity, or Eigen::Dynamic.
     * @note Rows are piece-major, then ascending powers of local physical time. Actual
     *       coefficient count may be smaller than capacity. Const queries never write
     *       shared state. Updates invalidate every borrowed segment, iterator and sampler.
     *       Callers guarantee finite inputs, valid orders/indices and matching input/output shapes.
     *       Knots must be strictly increasing, with at least one piece; coefficient count and total
     *       rows must fit capacity and int indices. These input preconditions are not checked.
     *       Owning queries return values; output-reference queries return void. Querying requires an
     *       initialized polynomial, asserted at entry. Query and coefficient-transform arithmetic is not scanned for overflow;
     *       callers guarantee a representable working range, including any extrapolation.
     *       Borrowed samplers/segments require an initialized immutable source and return values directly.
     *       Source mutation invalidates these views. Allocation
     *       failures and external callback exceptions are not translated into library statuses. */
    template <int DIM, int MaxCoefficients = Eigen::Dynamic>
    class PPolyND
    {
    public:
        static_assert(DIM > 0, "Spatial dimension must be positive");
        static_assert(MaxCoefficients == Eigen::Dynamic || MaxCoefficients > 0,
                      "Coefficient capacity must be positive or dynamic");
        static constexpr int kDimension = DIM;
        static constexpr int kMaxCoefficients = MaxCoefficients;
        using Vector = Eigen::Matrix<double, DIM, 1>;
        using RowVector = Eigen::Matrix<double, 1, DIM>;
        static constexpr int kMatrixOptions = (DIM == 1) ? Eigen::ColMajor : Eigen::RowMajor;
        using CoefficientMatrix = Eigen::Matrix<double, Eigen::Dynamic, DIM, kMatrixOptions>;
        /** @brief Owned single-time values indexed by physical derivative order, including position. */
        template <int MAX_DERIVATIVE>
        using Derivatives = std::array<Vector, MAX_DERIVATIVE + 1>;
        /** @brief Owned batch matrices indexed by derivative order, with one row per query time. */
        template <int MAX_DERIVATIVE>
        using DerivativeBatch = std::array<CoefficientMatrix, MAX_DERIVATIVE + 1>;
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    private:
        static constexpr bool kHasFixedCapacity = (MaxCoefficients != Eigen::Dynamic);
        static constexpr int kCapacity = MaxCoefficients;

        std::vector<double> breakpoints_;
        CoefficientMatrix coefficients_;
        int num_segments_{0};
        int num_coeffs_{0};
        bool is_initialized_{false};

        static constexpr int kStaticFactorMaxOrder = 8;
        using DerivativeFactorTable = std::array<std::array<double, kStaticFactorMaxOrder>, kStaticFactorMaxOrder>;

        static constexpr DerivativeFactorTable makeStaticDerivativeFactorTable()
        {
            DerivativeFactorTable table{};
            for (int n = 0; n < kStaticFactorMaxOrder; ++n)
            {
                table[n][0] = 1.0;
                double acc = 1.0;
                for (int k = 1; k <= n; ++k)
                {
                    acc *= static_cast<double>(n - k + 1);
                    table[n][k] = acc;
                }
            }
            return table;
        }
        static constexpr DerivativeFactorTable kStaticDerivativeFactorTable_ = makeStaticDerivativeFactorTable();

        /** @brief Compute a falling factorial for an in-range derivative.
         * @param n Nonnegative monomial power. @param k Derivative order in [0, n].
         * @return n!/(n-k)!; callers establish the index bounds before entering this helper. */
        inline double derivativeFactor(int n, int k) const
        {
            if constexpr (kHasFixedCapacity && kCapacity <= kStaticFactorMaxOrder)
                return kStaticDerivativeFactorTable_[n][k];
            if (n < kStaticFactorMaxOrder)
                return kStaticDerivativeFactorTable_[n][k];
            double factor = 1.0;
            for (int i = 0; i < k; ++i) factor *= n - i;
            return factor;
        }

        /** @brief Evaluate one piece with Horner's rule, without allocating or changing storage.
         * @param segment_idx Valid piece index in this polynomial.
         * @param t Local physical time in seconds, measured from the piece's left endpoint.
         * @param derivative_order Physical derivative order; out-of-degree orders produce zero.
         * @return Derivative in the caller's coordinates; the source must remain unmodified. */
        template<int Count>
        inline Vector evaluateSegmentHorner(int segment_idx, double t, int derivative_order) const
        {
            const int count = Count == Eigen::Dynamic ? num_coeffs_ : Count;
            if (derivative_order >= count)
                return Vector::Zero();

            const int base_row = segment_idx * count;
            Vector result = derivativeFactor(count - 1, derivative_order) *
                                coefficients_.row(base_row + count - 1).transpose();
            for (int k = count - 2; k >= derivative_order; --k)
                result = result * t + derivativeFactor(k, derivative_order) *
                                     coefficients_.row(base_row + k).transpose();
            return result;
        }

        /** @brief Select a fixed Horner bound only when actual degree fills the static capacity.
         * @param segment_idx Valid piece index. @param t Local physical seconds.
         * @param derivative_order Nonnegative physical derivative order.
         * @return Owned derivative; reduced-degree and dynamic polynomials use their actual count. */
        inline Vector evaluateSegmentHorner(int segment_idx, double t, int derivative_order) const
        {
            if constexpr (kHasFixedCapacity)
                if (num_coeffs_ == kCapacity)
                    return evaluateSegmentHorner<kCapacity>(segment_idx, t, derivative_order);
            return evaluateSegmentHorner<Eigen::Dynamic>(segment_idx, t, derivative_order);
        }

        /** @brief Evaluate all requested derivatives of an already located piece.
         * @tparam MAX_DERIVATIVE Nonnegative highest physical derivative order.
         * @param segment Valid piece index in this initialized polynomial.
         * @param local_time Physical time in seconds from the piece's left endpoint.
         * @return Owned derivative array in source coordinates, with zeros above degree.
         * @note No allocation or source mutation occurs; concurrent reads are independent. */
        template <int MAX_DERIVATIVE>
        Derivatives<MAX_DERIVATIVE> evaluateSegmentDerivatives(int segment, double local_time) const
        {
            static_assert(MAX_DERIVATIVE >= 0, "Derivative order must be nonnegative");
            Derivatives<MAX_DERIVATIVE> values;
            for (int d = 0; d <= MAX_DERIVATIVE; ++d)
                values[d] = evaluateSegmentHorner(segment, local_time, d);
            return values;
        }

        /** @brief Allocate one result matrix per requested derivative.
         * @tparam MAX_DERIVATIVE Nonnegative highest physical derivative order.
         * @param times Query times on the source axis; only their count determines output shape.
         * @return Owned, uninitialized matrices with times.size() rows and DIM columns.
         * @note Uses no polynomial or sampler state; independent calls are reentrant. */
        template <int MAX_DERIVATIVE>
        static DerivativeBatch<MAX_DERIVATIVE> makeDerivativeBatch(const std::vector<double> &times)
        {
            static_assert(MAX_DERIVATIVE >= 0, "Derivative order must be nonnegative");
            DerivativeBatch<MAX_DERIVATIVE> output;
            for (auto &matrix : output)
                matrix.resize(static_cast<Eigen::Index>(times.size()), DIM);
            return output;
        }

        /** @brief Evaluate a batch element using the previous element's piece as a lookup hint.
         * @param t Time on the polynomial's axis, in seconds; outside times extrapolate.
         * @param derivative_order Nonnegative physical derivative order, checked by the caller.
         * @param[in,out] segment Previous piece index, initially -1; updated for finite in-degree queries.
         * @return Owned derivative; derivatives above degree are zero.
         * @pre The source is initialized and the caller guarantees representable arithmetic.
         * @note Only the caller's hint changes; independent batches may read the same source. */
        Vector evaluateWithHint(double t, int derivative_order, int &segment) const
        {
            if (derivative_order >= num_coeffs_) return Vector::Zero().eval();
            segment = findSegment(t, segment);
            return evaluateSegmentHorner(segment, t - breakpoints_[segment], derivative_order);
        }

    public:
        /**
         * @brief Caller-owned evaluator with a segment cursor and current-piece derivative cache.
         * @tparam MAX_DERIVATIVE Highest supported derivative; zero supports position only.
         * @note Borrows its source, which must remain alive and unmodified. Source updates,
         *       assignment or moves invalidate this sampler. Each independent traversal owns
         *       a separate sampler; no query state is stored in the polynomial. Query order is
         *       unrestricted; nearby forward queries benefit most from cursor reuse.
         */
        template <int MAX_DERIVATIVE>
        class Sampler
        {
            static_assert(MAX_DERIVATIVE >= 0, "Derivative order must be nonnegative");
            using PieceMatrix = Eigen::Matrix<double, MaxCoefficients, DIM, kMatrixOptions>;
            const PPolyND *source_;
            std::array<PieceMatrix, MAX_DERIVATIVE + 1> coefficients_;
            int segment_ = -1;

            /** @brief Cache physical derivative coefficients for one source piece.
             * @param segment Valid source piece index; the source must remain unmodified.
             * @note Only this sampler's cursor and already allocated coefficient storage change. */
            void preparePiece(int segment)
            {
                segment_ = segment;
                const int count = source_->num_coeffs_;
                for (int d = 0; d <= MAX_DERIVATIVE && d < count; ++d)
                    for (int k = d; k < count; ++k)
                        coefficients_[d].row(k - d) = source_->derivativeFactor(k, d) *
                            source_->coefficients_.row(segment * count + k);
            }

            /** @brief Locate a finite query and refresh the cache only when its piece changes.
             * @param t Time on the valid source's axis, in seconds; outside times extrapolate.
             * @return Physical time in seconds relative to the selected piece's start. */
            double prepareQuery(double t)
            {
                const int segment = source_->findSegment(t, segment_);
                if (segment != segment_) preparePiece(segment);
                return t - source_->breakpoints_[segment];
            }

            /** @brief Evaluate one cached derivative without locating or allocating.
             * @param local_time Physical time in seconds from the cached piece's start.
             * @param derivative_order Supported order in [0, MAX_DERIVATIVE].
             * @return Owned vector in the source coordinates, or zero above the source degree.
             * @note Requires a prepared piece. The sampler and source remain unmodified. */
            template<int Count>
            Vector evaluateCached(double local_time, int derivative_order) const
            {
                const int count = (Count == Eigen::Dynamic ? source_->num_coeffs_ : Count) - derivative_order;
                if (count <= 0) return Vector::Zero();
                Vector value = coefficients_[derivative_order].row(count - 1).transpose();
                for (int k = count - 2; k >= 0; --k)
                    value = value * local_time + coefficients_[derivative_order].row(k).transpose();
                return value;
            }

            /** @brief Use a static loop bound for full-capacity pieces and actual count otherwise.
             * @param local_time Seconds relative to the cached piece.
             * @param derivative_order Order in [0, MAX_DERIVATIVE].
             * @return Owned derivative; requires a prepared immutable source piece. */
            Vector evaluateCached(double local_time, int derivative_order) const
            {
                if constexpr (kHasFixedCapacity)
                    if (source_->num_coeffs_ == kCapacity)
                        return evaluateCached<kCapacity>(local_time, derivative_order);
                return evaluateCached<Eigen::Dynamic>(local_time, derivative_order);
            }

            public:
            using Sample = Derivatives<MAX_DERIVATIVE>;
            using Batch = DerivativeBatch<MAX_DERIVATIVE>;

            /** @brief Bind an independent sampler without changing or warming the source.
             * @param source Initialized immutable polynomial; must outlive this sampler.
             * @pre Source validity is asserted here; subsequent queries rely on this lifetime contract.
             * @note Dynamic-capacity polynomials allocate piece storage here; fixed-capacity
             *       ones do not. Evaluation reuses this storage without allocating. */
            explicit Sampler(const PPolyND &source) : source_(&source)
            {
                assert(source.isValid() && "Sampler requires an initialized, immutable source");
                if constexpr (!kHasFixedCapacity)
                    for (int d = 0; d <= MAX_DERIVATIVE && d < source.num_coeffs_; ++d)
                        coefficients_[d].resize(source.num_coeffs_ - d, DIM);
            }
            Sampler(PPolyND &&) = delete;
            Sampler(const PPolyND &&) = delete;

            /** @brief Evaluate one physical derivative using this sampler's cursor and cache.
             * @param t Time on the source axis, in seconds; outside times extrapolate.
             * @param derivative_order Requested order in [0, MAX_DERIVATIVE].
             * @return Owned vector; derivatives above the source degree are zero.
             * @note Interior knots use the right piece. Requires exclusive access to this
             *       sampler and an unmodified source; no allocation occurs. */
            Vector evaluate(double t, int derivative_order)
            {
                const double local_time = prepareQuery(t);
                return evaluateCached(local_time, derivative_order);
            }

            /** @brief Evaluate a named derivative, defaulting to position.
             * @param t Time on the source axis, in seconds.
             * @param type Supported physical derivative; defaults to position.
             * @return Owned vector with evaluate(t, int)'s value and lifetime contract. */
            Vector evaluate(double t, Deriv type = Deriv::Pos)
            {
                return evaluate(t, static_cast<int>(type));
            }

            /** @brief Evaluate physical derivatives 0..MAX_DERIVATIVE with one piece lookup.
             * @param t Time on the source axis, in seconds; outside times extrapolate.
             * @return Owned derivative array, with zero entries above degree.
             * @note Updates only this sampler's cursor/cache. Above-degree derivatives are zero;
             *       source lifetime, exclusive sampler access and knot rules match evaluate(). */
            Sample evaluateDerivatives(double t)
            {
                Sample values;
                const double local_time = prepareQuery(t);
                for (int d = 0; d <= MAX_DERIVATIVE; ++d)
                    values[d] = evaluateCached(local_time, d);
                return values;
            }

            /** @brief Write one derivative for a batch while retaining the sampler's cursor.
             * @param times Query times on the source axis, in seconds, in any order.
             * @param[out] output Caller-owned matrix with times.size() rows and DIM columns.
             * @param derivative_order Supported physical derivative order; defaults to position.
             * @note No allocation or source mutation occurs. Per-query values follow evaluate();
             *       requires exclusive sampler/output access. Cursor state survives this call. */
            void evaluateInto(const std::vector<double> &times, Eigen::Ref<CoefficientMatrix> output,
                                int derivative_order = 0)
            {
                if (derivative_order >= source_->num_coeffs_)
                {
                    output.setZero();
                    return;
                }
                for (std::size_t i = 0; i < times.size(); ++i)
                {
                    const double local_time = prepareQuery(times[i]);
                    output.row(static_cast<Eigen::Index>(i)) = evaluateCached(local_time, derivative_order).transpose();
                }
            }

            /** @brief Write a named derivative using the same batch contract as the integer overload.
             * @param times Query times on the source axis, in seconds.
             * @param[out] output Pre-sized caller-owned result matrix; storage is reused.
             * @param type Supported physical derivative; the caller guarantees the order is supported. */
            void evaluateInto(const std::vector<double> &times, Eigen::Ref<CoefficientMatrix> output, Deriv type)
            {
                return evaluateInto(times, output, static_cast<int>(type));
            }

            /** @brief Allocate a batch result and evaluate one supported derivative into it.
             * @param times Query times on the source axis, in seconds, in any order.
             * @param derivative_order Requested order in [0, MAX_DERIVATIVE].
             * @return Owning matrix with one row per time; values follow evaluate().
             * @note Updates this sampler's cursor/cache while leaving its source unmodified. */
            CoefficientMatrix evaluate(const std::vector<double> &times, int derivative_order)
            {
                CoefficientMatrix output(static_cast<Eigen::Index>(times.size()), DIM);
                evaluateInto(times, output, derivative_order);
                return output;
            }

            /** @brief Allocate a batch for a named derivative, defaulting to position.
             * @param times Query times on the source axis, in seconds.
             * @param type Supported physical derivative; defaults to position.
             * @return Owning result matrix with the integer overload's value and lifetime contract. */
            CoefficientMatrix evaluate(const std::vector<double> &times, Deriv type = Deriv::Pos)
            {
                return evaluate(times, static_cast<int>(type));
            }

            /** @brief Write every supported derivative for a batch, locating each time once.
             * @param times Query times on the source axis, in seconds, in any order.
             * @param[out] output Matrices indexed by derivative order; each must have times.size()
             *        rows and DIM columns. Matrices must not alias source coefficient storage.
             * @note No allocation occurs. Per-query values follow evaluateDerivatives(); only
             *       this sampler's cursor/cache and the caller's output change. */
            void evaluateDerivativesInto(const std::vector<double> &times, Batch &output)
            {
                for (std::size_t i = 0; i < times.size(); ++i)
                {
                    const double local_time = prepareQuery(times[i]);
                    for (int d = 0; d <= MAX_DERIVATIVE; ++d)
                        output[d].row(static_cast<Eigen::Index>(i)) = evaluateCached(local_time, d).transpose();
                }
            }

            /** @brief Allocate a batch of every supported physical derivative.
             * @param times Query times on the source axis, in seconds, in any order.
             * @return Owning matrices indexed by derivative order, with one row per time.
             * @note Delegates to evaluateDerivativesInto(); the sampler's cursor is retained. */
            Batch evaluateDerivatives(const std::vector<double> &times)
            {
                Batch output = makeDerivativeBatch<MAX_DERIVATIVE>(times);
                evaluateDerivativesInto(times, output);
                return output;
            }
        };

        /** @brief Create an independent sequential evaluator borrowing this polynomial.
         * @tparam MAX_DERIVATIVE Highest requested derivative; zero means position only.
         * @return Sampler valid while this polynomial remains alive and unmodified.
         * @note Construction never changes the polynomial and does not require prewarming. */
        template <int MAX_DERIVATIVE>
        Sampler<MAX_DERIVATIVE> makeSampler() const & { return Sampler<MAX_DERIVATIVE>(*this); }

        template <int MAX_DERIVATIVE>
        Sampler<MAX_DERIVATIVE> makeSampler() const && = delete;

        class Segment
        {
            friend class PPolyND;
            const PPolyND *parent_;
            int idx_;

            /** @brief Bind a piece of an initialized immutable source.
             * @param parent Non-null borrowed source, alive and unchanged for this view's lifetime.
             * @param idx Valid piece index; construction checks source initialization only. */
            Segment(const PPolyND *parent, int idx) : parent_(parent), idx_(idx)
            { assert(parent->isValid()); }

        public:
            EIGEN_MAKE_ALIGNED_OPERATOR_NEW

            double duration() const
            {
                return parent_->breakpoints_[idx_ + 1] - parent_->breakpoints_[idx_];
            }

            double startTime() const { return parent_->breakpoints_[idx_]; }

            double endTime() const { return parent_->breakpoints_[idx_ + 1]; }

            int index() const { return idx_; }

            /** @brief Evaluate in seconds relative to this piece's start; extrapolation is allowed.
             * @param t Local physical seconds. @param type Requested physical derivative.
             * @return Derivative; derivatives above degree are zero. */
            Vector evaluateLocal(double t, Deriv type = Deriv::Pos) const
            {
                return evaluateLocal(t, static_cast<int>(type));
            }

            /** @brief Evaluate a derivative on this borrowed piece without locating a segment.
             * @param t Local physical seconds from this piece's start; extrapolation is allowed.
             * @param derivative_order Nonnegative physical derivative order.
             * @return Owned vector; the view requires a valid, unchanged source.
             * @pre The source remains alive and unchanged; arithmetic is representable.
             * @note Const queries are reentrant and do not allocate. */
            Vector evaluateLocal(double t, int derivative_order) const
            {
                return parent_->evaluateSegmentHorner(idx_, t, derivative_order);
            }

            auto coefficients() const -> decltype(parent_->coefficients_.block(0, 0, 0, 0))
            {
                return parent_->coefficients_.block(idx_ * parent_->num_coeffs_, 0, parent_->num_coeffs_, DIM);
            }
        };

        /** @brief Random-access iterator over borrowed segment proxies; compare only within one source. */
        class ConstIterator
        {
            const PPolyND *ptr_ = nullptr;
            int idx_ = 0;

        public:
            using iterator_category = std::random_access_iterator_tag;
            using value_type = Segment;
            using difference_type = std::ptrdiff_t;
            using pointer = Segment *;
            using reference = Segment;

            ConstIterator() = default;
            ConstIterator(const PPolyND *ptr, int idx) : ptr_(ptr), idx_(idx) {}

            reference operator*() const { return Segment(ptr_, idx_); }

            class Proxy
            {
                Segment seg;

            public:
                Proxy(Segment s) : seg(s) {}
                const Segment *operator->() const { return &seg; }
            };
            Proxy operator->() const { return Proxy(Segment(ptr_, idx_)); }

            ConstIterator &operator++()
            {
                ++idx_;
                return *this;
            }
            ConstIterator operator++(int)
            {
                ConstIterator tmp = *this;
                ++idx_;
                return tmp;
            }
            ConstIterator &operator--()
            {
                --idx_;
                return *this;
            }
            ConstIterator operator--(int)
            {
                ConstIterator tmp = *this;
                --idx_;
                return tmp;
            }

            bool operator==(const ConstIterator &other) const { return idx_ == other.idx_ && ptr_ == other.ptr_; }
            bool operator!=(const ConstIterator &other) const { return !(*this == other); }

            difference_type operator-(const ConstIterator &other) const { return idx_ - other.idx_; }
            ConstIterator operator+(difference_type n) const { return ConstIterator(ptr_, idx_ + n); }
            ConstIterator operator-(difference_type n) const { return *this + (-n); }
            ConstIterator &operator+=(difference_type n) { idx_ += n; return *this; }
            ConstIterator &operator-=(difference_type n) { return *this += -n; }
            reference operator[](difference_type n) const { return *(*this + n); }
            bool operator<(const ConstIterator &other) const { return idx_ < other.idx_; }
            bool operator>(const ConstIterator &other) const { return other < *this; }
            bool operator<=(const ConstIterator &other) const { return !(other < *this); }
            bool operator>=(const ConstIterator &other) const { return !(*this < other); }
            friend ConstIterator operator+(difference_type n, ConstIterator it) { return it + n; }
        };

        PPolyND() = default;

        /** @brief Copy N+1 knots and N*coefficient_count ascending-power coefficient rows.
         * @param breakpoints Strictly increasing physical times in seconds.
         * @param coefficients Piece-major ascending-power rows in the caller's coordinate frame.
         * @param coefficient_count Positive coefficient count within the declared capacity.
         * @pre Inputs satisfy update()'s finite-value, shape and capacity contract; not checked.
         * @return Independent owning polynomial; allocation failures propagate normally. */
        static PPolyND create(const std::vector<double> &breakpoints,
                                      const CoefficientMatrix &coefficients, int coefficient_count)
        {
            PPolyND result;
            result.update(breakpoints, coefficients, coefficient_count);
            return result;
        }

        PPolyND(const PPolyND &) = default;
        PPolyND &operator=(const PPolyND &) = default;
        PPolyND(PPolyND &&other) { *this = std::move(other); }
        PPolyND &operator=(PPolyND &&other)
        {
            if (this != &other)
            {
                breakpoints_ = std::move(other.breakpoints_);
                coefficients_ = std::move(other.coefficients_);
                num_segments_ = other.num_segments_;
                num_coeffs_ = other.num_coeffs_;
                is_initialized_ = other.is_initialized_;
                other.clear();
            }
            return *this;
        }

        /** @brief Invalidate the polynomial and all its borrowed views. */
        void clear()
        {
            is_initialized_ = false;
            num_segments_ = num_coeffs_ = 0;
            breakpoints_.clear();
            coefficients_.resize(0, DIM);
        }

        ConstIterator begin() const { return ConstIterator(this, 0); }
        ConstIterator end() const { return ConstIterator(this, num_segments_); }

        Segment operator[](int idx) const &
        {
            return Segment(this, idx);
        }

        Segment operator[](int) const && = delete;
        Segment at(int) const && = delete;
        /** @brief Borrow a segment; source mutation invalidates the returned view.
         * @param idx Valid index in [0, numSegments()); not checked.
         * @return Successful borrowed segment. The source must outlive the view. */
        Segment at(int idx) const &
        {
            return Segment(this, idx);
        }

        /** @brief Query the trajectory time axis in seconds, extrapolating outside its domain.
         * @param t Physical query time in seconds on the trajectory axis.
         * @param derivative_order Nonnegative physical derivative order.
         * @return Owned physical derivative; derivatives above degree are zero.
         * @pre The polynomial is initialized and query inputs satisfy the caller contract. */
        Vector evaluate(double t, int derivative_order) const
        {
            assert(is_initialized_);
            int segment = -1;
            return evaluateWithHint(t, derivative_order, segment);
        }

        /** @brief Evaluate a named physical derivative, defaulting to position.
         * @param t Time on the trajectory axis, in seconds; outside times extrapolate.
         * @param type Physical derivative; defaults to position.
         * @return Owned vector with the integer overload's value and lifetime contract; no shared writes. */
        Vector evaluate(double t, Deriv type = Deriv::Pos) const
        {
            return evaluate(t, static_cast<int>(type));
        }

        /** @brief Evaluate derivatives 0..MAX_DERIVATIVE with one segment lookup and no shared writes.
         * @param t Time on this spline's axis; endpoints use the adjacent end piece, interior knots use the right piece.
         * @return Owned derivative array. Outside times extrapolate the end piece.
         * @pre The polynomial is initialized and query times are finite.
         * @note Const calls are reentrant; update requires exclusive ownership. No heap allocation occurs during evaluation. */
        template <int MAX_DERIVATIVE>
        Derivatives<MAX_DERIVATIVE> evaluateDerivatives(double t) const
        {
            static_assert(MAX_DERIVATIVE >= 0, "Derivative order must be nonnegative");
            assert(is_initialized_);
            const int segment = findSegment(t);
            auto values = evaluateSegmentDerivatives<MAX_DERIVATIVE>(segment, t - breakpoints_[segment]);
            return values;
        }

        /** @brief Evaluate one derivative into caller-owned rows without allocating.
         * @param times Query times on the trajectory axis.
         * @param[out] output Matrix with times.size() rows and DIM columns.
         * @param derivative_order Nonnegative physical derivative order.
         * @pre The polynomial is initialized; inputs and pre-sized outputs satisfy the caller contract.
         * @note Query order is unrestricted; no order scan or input reordering occurs. Each
         *       query checks the current and next piece before falling back to a regular lookup.
         *       The cursor is local to this call and never stored in the polynomial.
         *       Times must be finite; values follow the single-query evaluate() contract. */
        void evaluateInto(const std::vector<double> &times, Eigen::Ref<CoefficientMatrix> output,
                            int derivative_order = 0) const
        {
            assert(is_initialized_);
            if (derivative_order >= num_coeffs_)
            {
                output.setZero();
                return;
            }
            int segment = -1;
            for (std::size_t i = 0; i < times.size(); ++i)
            {
                output.row(static_cast<Eigen::Index>(i)) =
                    evaluateWithHint(times[i], derivative_order, segment).transpose();
            }
        }

        /** @brief Write a named derivative using the same batch contract as the integer overload.
         * @param times Query times on the trajectory axis, in seconds.
         * @param[out] output Pre-sized caller-owned result matrix; storage is reused.
         * @param type Physical derivative; order and output shape are caller preconditions.
         * @pre The polynomial is initialized; inputs and pre-sized outputs satisfy the caller contract. */
        void evaluateInto(const std::vector<double> &times, Eigen::Ref<CoefficientMatrix> output, Deriv type) const
        {
            return evaluateInto(times, output, static_cast<int>(type));
        }

        /** @brief Allocate and evaluate one physical derivative for a batch.
         * @param times Query times on the trajectory axis, in seconds, in any order.
         * @param derivative_order Nonnegative physical derivative order.
         * @return Owning matrix with one row per query time and DIM columns.
         * @note Delegates to evaluateInto(); no query state is retained in the polynomial. */
        CoefficientMatrix evaluate(const std::vector<double> &times, int derivative_order) const
        {
            CoefficientMatrix output(static_cast<Eigen::Index>(times.size()), DIM);
            evaluateInto(times, output, derivative_order);
            return output;
        }

        /** @brief Allocate a batch for a named derivative, defaulting to position.
         * @param times Query times on the trajectory axis, in seconds.
         * @param type Physical derivative; defaults to position.
         * @return Owning result matrix with the integer overload's value and lifetime contract. */
        CoefficientMatrix evaluate(const std::vector<double> &times, Deriv type = Deriv::Pos) const
        {
            return evaluate(times, static_cast<int>(type));
        }

        /** @brief Write physical derivatives 0..MAX_DERIVATIVE for each time with one lookup.
         * @tparam MAX_DERIVATIVE Nonnegative highest requested physical derivative order.
         * @param times Query times on the trajectory axis, in seconds, in any order.
         * @param[out] output Matrices indexed by derivative order; each must have times.size()
         *        rows and DIM columns. Output must not alias source coefficient storage.
         * @pre The polynomial is initialized; inputs and pre-sized outputs satisfy the caller contract.
         * @note No allocation, order scan or source mutation occurs. A call-local cursor checks
         *       the current and next piece before a regular lookup, for queries in any order.
         *       Requires an initialized source; above-degree derivatives return zero, and outside
         *       queries extrapolate. Independent calls with separate outputs are reentrant. */
        template <int MAX_DERIVATIVE>
        void evaluateDerivativesInto(const std::vector<double> &times,
                                       DerivativeBatch<MAX_DERIVATIVE> &output) const
        {
            static_assert(MAX_DERIVATIVE >= 0, "Derivative order must be nonnegative");
            assert(is_initialized_);
            int segment = -1;
            for (std::size_t i = 0; i < times.size(); ++i)
            {
                segment = findSegment(times[i], segment);
                const double local_time = times[i] - breakpoints_[segment];
                for (int d = 0; d <= MAX_DERIVATIVE; ++d)
                    output[d].row(static_cast<Eigen::Index>(i)) =
                        evaluateSegmentHorner(segment, local_time, d).transpose();
            }
        }

        /** @brief Allocate a batch of physical derivatives 0..MAX_DERIVATIVE.
         * @tparam MAX_DERIVATIVE Nonnegative highest requested physical derivative order.
         * @param times Query times on the trajectory axis, in seconds, in any order.
         * @return Owning matrices indexed by derivative order, with one row per query time.
         * @note Delegates to evaluateDerivativesInto(); source data remains unmodified. */
        template <int MAX_DERIVATIVE>
        DerivativeBatch<MAX_DERIVATIVE> evaluateDerivatives(const std::vector<double> &times) const
        {
            static_assert(MAX_DERIVATIVE >= 0, "Derivative order must be nonnegative");
            auto output = makeDerivativeBatch<MAX_DERIVATIVE>(times);
            evaluateDerivativesInto<MAX_DERIVATIVE>(times, output);
            return output;
        }

        /** @brief Approximate arc length over the full domain using left-endpoint velocities.
         * @param dt Positive sampling step in seconds.
         * @return Length in the caller's spatial units, or a state/time-sequence failure.
         * @note Const and reentrant; allocates sample times. Arithmetic range is a precondition. */
        StatusOr<double> length(double dt = 0.01) const
        {
            return length(startTime(), endTime(), dt);
        }

        /** @brief Integrate speed over a caller-selected interval using left-endpoint samples.
         * @param start_t Finite interval start on the trajectory axis, in seconds.
         * @param end_t Finite interval end at or after start_t, in seconds.
         * @param dt Positive sampling step in seconds.
         * @return Length in the caller's spatial units, or a state/time-sequence failure.
         * @note Const and reentrant; allocates sample times and reuses a local segment cursor. */
        StatusOr<double> length(double start_t, double end_t, double dt = 0.01) const
        {
            if (!is_initialized_) return Status{ErrorCode::InvalidState, "Polynomial is not initialized"};
            auto sequence = generateTimeSequence(start_t, end_t, dt);
            if (!sequence) return sequence.status();
            const auto &time_sequence = sequence.value();

            double total_length = 0.0;
            int segment = -1;
            for (size_t i = 0; i < time_sequence.size() - 1; ++i)
            {
                double t_current = time_sequence[i];
                double t_next = time_sequence[i + 1];
                double dt_actual = t_next - t_current;

                const auto velocity = evaluateWithHint(t_current, static_cast<int>(Deriv::Vel), segment);
                total_length += velocity.norm() * dt_actual;
            }

            return total_length;
        }

        /** @brief Copy caller-validated polynomial data, invalidating all borrowed views.
         * @param breakpoints Finite strictly increasing absolute seconds, at least two entries.
         * @param coefficients Finite piece-major ascending-power coefficients, (knots - 1) * count rows.
         * @param num_coefficients Positive count within static capacity; total rows fit int indices.
         * @pre Inputs satisfy the class contract; no input scan is performed. Requires exclusive access.
         * @note Allocation failures propagate normally. */
        void update(const std::vector<double> &breakpoints, const CoefficientMatrix &coefficients, int num_coefficients)
        {
            is_initialized_ = false;
            breakpoints_ = breakpoints;
            coefficients_ = coefficients;
            num_coeffs_ = num_coefficients;
            num_segments_ = static_cast<int>(breakpoints.size()) - 1;
            is_initialized_ = true;
        }

        bool isValid() const { return is_initialized_; }
        int dimension() const { return DIM; }
        int coefficientCount() const { return num_coeffs_; }
        int degree() const { return num_coeffs_ > 0 ? num_coeffs_ - 1 : 0; }
        int numSegments() const { return num_segments_; }
        double startTime() const { return breakpoints_.empty() ? 0.0 : breakpoints_.front(); }
        double endTime() const { return breakpoints_.empty() ? 0.0 : breakpoints_.back(); }
        double duration() const { return endTime() - startTime(); }
        const std::vector<double> &breakpoints() const { return breakpoints_; }
        const CoefficientMatrix &coefficients() const { return coefficients_; }

        /** @brief Generate increasing query times on the caller's time axis, including both endpoints.
         * @param start_t Finite interval start in seconds. @param end_t Finite end, at or after start.
         * @param dt Positive finite step in seconds.
         * @pre Finite start/end/step, end_t >= start_t, and dt > 0; inputs are not checked.
         * @return Owned times (one for a zero-length interval), or ResourceLimit/NumericalFailure
         *         when the computed count or progression is not representable. */
        StatusOr<std::vector<double>> generateTimeSequence(double start_t, double end_t, double dt) const
        {
            const double steps = std::floor((end_t - start_t) / dt);
            if (!std::isfinite(steps) || steps > static_cast<double>(std::numeric_limits<int>::max() - 2) ||
                (start_t < end_t && start_t + dt <= start_t))
                return Status{ErrorCode::ResourceLimit, "Time sequence cannot be represented"};
            std::vector<double> result;
            result.reserve(static_cast<std::size_t>(steps) + 2);
            result.push_back(start_t);
            for (int i = 1; i <= static_cast<int>(steps); ++i)
            {
                const double t = start_t + i * dt;
                if (t >= end_t) break;
                if (t <= result.back()) return Status{ErrorCode::NumericalFailure, "Time sequence cannot advance", i};
                result.push_back(t);
            }
            if (end_t != start_t) result.push_back(end_t);
            return result;
        }

        StatusOr<std::vector<double>> generateTimeSequence(double dt) const
        {
            if (!is_initialized_) return Status{ErrorCode::InvalidState, "Polynomial is not initialized"};
            return generateTimeSequence(startTime(), endTime(), dt);
        }

        /** @brief Differentiate the local-power coefficients into an owning polynomial.
         * @param derivative_order Nonnegative physical derivative order.
         * @return Owned polynomial or invalid-state failure; above-degree derivatives are zero.
         * @pre The coefficient arithmetic is representable. Const calls are reentrant. */
        StatusOr<PPolyND> derivative(int derivative_order = 1) const
        {
            if (num_segments_ == 0 || derivative_order >= num_coeffs_)
            {
                if (num_segments_ == 0)
                    return Status{ErrorCode::InvalidState, "Polynomial is not initialized"};
                CoefficientMatrix zero_coeffs = CoefficientMatrix::Zero(num_segments_, DIM);
                return create(breakpoints_, zero_coeffs, 1);
            }

            int new_order = num_coeffs_ - derivative_order;
            CoefficientMatrix new_coeffs(num_segments_ * new_order, DIM);

            for (int seg = 0; seg < num_segments_; ++seg)
            {
                for (int k = 0; k < new_order; ++k)
                {
                    int orig_k = k + derivative_order;
                    double coeff_factor = derivativeFactor(orig_k, derivative_order);
                    RowVector orig_coeff = coefficients_.row(seg * num_coeffs_ + orig_k);
                    new_coeffs.row(seg * new_order + k) = coeff_factor * orig_coeff;
                }
            }

            return create(breakpoints_, new_coeffs, new_order);
        }

        /** @brief Extract and rebase an interval using exact coefficient translation.
         * @param start Start on the original time axis, within the source domain.
         * @param end End on the original axis, strictly after start and within the source domain.
         * @return Owned zero-origin spline, or a failure for invalid state or unrepresentable time knots.
         * @note Preserves piece boundaries and coefficients outside the first clipped piece; const and reentrant. */
        StatusOr<PPolyND> extractInterval(double start, double end) const
        {
            if (!is_initialized_) return Status{ErrorCode::InvalidState, "Polynomial is not initialized"};
            const int first = findSegment(start);
            const int last = static_cast<int>(std::lower_bound(breakpoints_.begin() + 1,
                breakpoints_.end(), end) - breakpoints_.begin());
            CoefficientMatrix coefficients = coefficients_.middleRows(first * num_coeffs_, (last - first) * num_coeffs_);
            const double delta = start - breakpoints_[first];
            if (delta != 0.0)
            {
                coefficients.topRows(num_coeffs_).setZero();
                for (int target = 0; target < num_coeffs_; ++target)
                    for (int source = target; source < num_coeffs_; ++source)
                    {
                        double binomial = 1.0;
                        for (int k = 1; k <= target; ++k)
                            binomial *= static_cast<double>(source - k + 1) / k;
                        coefficients.row(target) += binomial * std::pow(delta, source - target) *
                            coefficients_.row(first * num_coeffs_ + source);
                    }
            }
            std::vector<double> times{0.0};
            for (int i = first; i < last; ++i)
            {
                const double next = times.back() + std::min(end, breakpoints_[i + 1]) - std::max(start, breakpoints_[i]);
                if (!std::isfinite(next) || next <= times.back())
                    return Status{ErrorCode::NumericalFailure, "Unrepresentable extracted time knot", i};
                times.push_back(next);
            }

            return create(times, coefficients, num_coeffs_);
        }

        /** @brief Append two zero-origin splines at coefficient level without a continuity policy.
         * @param suffix Same-order zero-origin spline appended after this value.
         * @return Owned concatenated spline, or a failure for invalid state or unrepresentable time knots.
         * @note Callers own seam acceptance policy; inputs are read-only, no fitting or resampling occurs. */
        StatusOr<PPolyND> appendRebased(const PPolyND &suffix) const
        {
            if (!is_initialized_)
                return Status{ErrorCode::InvalidState, "Polynomial is not initialized"};
            std::vector<double> times = breakpoints_;
            for (auto it = suffix.breakpoints_.begin() + 1; it != suffix.breakpoints_.end(); ++it)
            {
                const double next = endTime() + *it;
                if (!std::isfinite(next) || next <= times.back())
                    return Status{ErrorCode::NumericalFailure, "Unrepresentable appended time knot"};
                times.push_back(next);
            }
            CoefficientMatrix coefficients(coefficients_.rows() + suffix.coefficients_.rows(), DIM);
            coefficients.topRows(coefficients_.rows()) = coefficients_;
            coefficients.bottomRows(suffix.coefficients_.rows()) = suffix.coefficients_;
            return create(times, coefficients, num_coeffs_);
        }

        /** @brief Construct an owning zero polynomial; coefficient count must fit the capacity.
         * @param breakpoints Caller-validated physical knots in seconds.
         * @param num_coefficients Positive number of ascending-power coefficients per piece.
         * @return Independent owning polynomial; allocation failures propagate normally. */
        static PPolyND zero(const std::vector<double> &breakpoints, int num_coefficients = 1)
        {
            int num_segments = static_cast<int>(breakpoints.size()) - 1;
            CoefficientMatrix zero_coeffs = CoefficientMatrix::Zero(num_segments * num_coefficients, DIM);
            return create(breakpoints, zero_coeffs, num_coefficients);
        }

        /** @brief Create an owning constant polynomial from caller-validated knots and value.
         * @param breakpoints Strictly increasing physical knots in seconds.
         * @param constant_value Finite value in the caller's coordinate frame.
         * @return Independent owning polynomial; allocation failures propagate normally. */
        static PPolyND constant(const std::vector<double> &breakpoints, const Vector &constant_value)
        {
            int num_segments = static_cast<int>(breakpoints.size()) - 1;
            CoefficientMatrix coeffs = CoefficientMatrix::Zero(num_segments, DIM);
            for (int i = 0; i < num_segments; ++i)
            {
                coeffs.row(i) = constant_value.transpose();
            }
            return create(breakpoints, coeffs, 1);
        }

    private:
        template <int> friend class CubicSplineND;
        template <int> friend class QuinticSplineND;
        template <int> friend class SepticSplineND;

        /** @brief Prepare owning polynomial storage for exclusive coefficient writes.
         * @param durations Positive physical piece durations in seconds.
         * @param start Finite start time on the caller's time axis, in seconds.
         * @param count Positive coefficient count per piece, within the static capacity.
         * @return Success with writable storage, or failure with an empty invalid polynomial.
         * @note Requires validated input shape, start time and coefficient capacity. Reuses
         *       same-size storage and stays invalid until publishCoefficients(); the spline
         *       must finish writing every coefficient before publishing. */
        Status prepareCoefficients(const std::vector<double> &durations, double start, int count)
        {
            // Invalidate before writing; the caller has already validated the input shape.
            is_initialized_ = false;

            // Build a strictly increasing physical time axis before exposing coefficient storage.
            breakpoints_.resize(durations.size() + 1);
            breakpoints_[0] = start;
            for (std::size_t i = 0; i < durations.size(); ++i)
            {
                const double next = breakpoints_[i] + durations[i];
                if (!std::isfinite(next) || next <= breakpoints_[i])
                {
                    clear();
                    return {ErrorCode::NumericalFailure, "Unrepresentable time knot", static_cast<int>(i)};
                }
                breakpoints_[i + 1] = next;
            }
            num_segments_ = static_cast<int>(durations.size());
            num_coeffs_ = count;
            coefficients_.resize(num_segments_ * count, DIM);
            return Status::success();
        }

        /** @brief Publish the coefficients after a successful spline solve.
         * @pre Exclusive access after prepareCoefficients() and a complete coefficient write.
         *      The caller guarantees a representable numerical range.
         * @note Knots and coefficients remain in owning storage without a copy or output scan. */
        void publishCoefficients()
        {
            is_initialized_ = true;
        }

        /** @brief Locate a piece with right-sided interior knots and endpoint extrapolation.
         * @param t Finite time on the polynomial's axis, in seconds.
         * @return Piece index; no shared state is changed.
         * @pre The polynomial is initialized; public query entries check this once. */
        inline int findSegment(double t) const
        {
            if (t <= breakpoints_.front())
                return 0;
            if (t >= breakpoints_.back())
                return num_segments_ - 1;
            constexpr int kLinearSearchThreshold = 32;
            if (num_segments_ < kLinearSearchThreshold)
            {
                for (int i = 0; i < num_segments_; ++i)
                    if (t < breakpoints_[i + 1])
                        return i;
                return num_segments_ - 1;
            }
            auto it = std::upper_bound(breakpoints_.begin(), breakpoints_.end(), t);
            return static_cast<int>(std::distance(breakpoints_.begin(), it)) - 1;
        }

        /** @brief Reuse the current or next piece before searching for a distant query.
         * @param t Finite time on this valid polynomial's axis, in seconds.
         * @param hint Previous valid piece index, or -1 for the first query.
         * @return Piece index with the same knot/extrapolation contract as findSegment(t).
         * @note The hint belongs to the caller; the polynomial remains immutable. */
        inline int findSegment(double t, int hint) const
        {
            if (hint >= 0 && t >= breakpoints_[hint])
            {
                if (hint + 1 == num_segments_ || t < breakpoints_[hint + 1])
                    return hint;
                if (hint + 2 == num_segments_ || t < breakpoints_[hint + 2])
                    return hint + 1;
            }
            return findSegment(t);
        }

    };

    /** @brief Boundary adjoints; unused higher derivatives remain zero. */
    template <int DIM>
    struct BoundaryGradient
    {
        using Vector = Eigen::Matrix<double, DIM, 1>;
        Vector p = Vector::Zero(), v = Vector::Zero(), a = Vector::Zero(), j = Vector::Zero();
    };

    /** @brief Start and end boundary adjoints in the same coordinate frame. */
    template <int DIM>
    struct BoundaryGradientPair
    {
        BoundaryGradient<DIM> start, end;
    };

    /** @brief Total derivatives with respect to interpolation data, in physical coordinates. */
    template <int DIM>
    struct SplineGradients
    {
        using CoefficientMatrix = typename PPolyND<DIM>::CoefficientMatrix;
        using RowVector = typename PPolyND<DIM>::RowVector;
        CoefficientMatrix inner_points;
        Eigen::VectorXd durations;
        BoundaryGradient<DIM> start, end;
        /** @brief Allocate output storage for a topology without initializing values.
         * @param segments Positive segment count; output storage is exclusively owned. */
        void resetTopology(int segments)
        {
            inner_points.resize(std::max(0, segments - 1), DIM);
            durations.resize(segments);
        }
        void setZero()
        {
            inner_points.setZero(); durations.setZero();
            start = {}; end = {};
        }

        /** @brief Accumulate a position partial at an endpoint or interior waypoint.
         * @param knot Waypoint index in [0, N], with N given by the prepared durations size.
         * @param partial Physical position partials in the caller's coordinates; borrowed for this call.
         * @note Requires a prepared N-segment topology and exclusive access to this output.
         *       The row overload preserves vectorized accumulation without allocating. */
        void addWaypoint(int knot, const RowVector &partial)
        {
            if (knot == 0)
                start.p += partial.transpose();
            else if (knot == durations.size())
                end.p += partial.transpose();
            else
                inner_points.row(knot - 1) += partial;
        }


    };

    namespace detail
    {
        /** @brief Convert caller-validated absolute knots to physical durations.
         * @param knots At least two finite, strictly increasing times in seconds.
         * @param[out] durations Consecutive time differences; storage is reused.
         * @param[out] start First knot on the original time axis. */
        inline void durationsFromKnots(const std::vector<double> &knots,
                                       std::vector<double> &durations, double &start)
        {
            start = knots.front();
            durations.resize(knots.size() - 1);
            for (std::size_t i = 0; i < durations.size(); ++i)
                durations[i] = knots[i + 1] - knots[i];
        }

        /** @brief Store one gradient expression using the caller's overwrite or additive contract.
         * @tparam Add True to accumulate, false to overwrite; resolved at compile time.
         * @param[in,out] destination Scalar or Eigen output with the required shape.
         * @param value Evaluated in the caller's physical gradient units.
         * @note Requires non-aliasing inputs and exclusive output access; never resizes storage. */
        template<bool Add, class Destination, class Value>
        inline void storeGradient(Destination &&destination, const Value &value)
        {
            if constexpr (Add) destination += value;
            else destination = value;
        }

        /** @brief Compute interior position gradients from the highest-derivative jump at each knot.
         * @tparam Add Select accumulation or overwrite at compile time.
         * @tparam Scale Signed multiplier of the right-minus-left highest power coefficient.
         * @param polynomial Valid minimum-derivative spline in physical time; borrowed for this call.
         * @param[in,out] gradient Pre-sized interior waypoint gradients in the caller's coordinates.
         * @param weight Multiplier of the energy contribution.
         * @pre The spline is valid.
         * @note Requires the per-order energy multiplier and exclusive access to gradient.
         *       A single segment has no interior waypoints and produces an empty matrix. */
        template <int Scale, bool Add = false, int DIM, int CoefficientCount>
        inline void computeEnergyWaypointGradient(const PPolyND<DIM, CoefficientCount> &polynomial,
                                           typename PPolyND<DIM>::CoefficientMatrix &gradient, double weight)
        {
            const int segments = polynomial.numSegments();

            // Integration by parts leaves only the highest-derivative jump at each interior knot.
            const auto &coefficients = polynomial.coefficients();
            for (int knot = 1; knot < segments; ++knot)
            {
                const auto left = coefficients.row((knot - 1) * CoefficientCount + CoefficientCount - 1);
                const auto right = coefficients.row(knot * CoefficientCount + CoefficientCount - 1);
                // Keep each order's original subtraction direction and positive multiplication.
                if constexpr (Scale > 0)
                    storeGradient<Add>(gradient.row(knot - 1), (weight * static_cast<double>(Scale)) * (right - left));
                else
                    storeGradient<Add>(gradient.row(knot - 1), (weight * static_cast<double>(-Scale)) * (left - right));
            }
        }

        /** @brief Initialize physical gradients with independent duration partials.
         * @param duration_partials Direct physical-duration partials; borrowed for this call.
         *        May refer to gradient.durations itself.
         * @param[out] gradient Receives duration partials and zero position/boundary gradients.
         * @note Requires matching, already prepared topology and exclusive access to gradient.
         *       Reuses storage and avoids zeroing durations immediately before overwriting them. */
        template <typename Gradients>
        inline void initializeGradients(const Eigen::VectorXd &duration_partials, Gradients &gradient)
        {
            gradient.inner_points.setZero();
            gradient.durations = duration_partials;
            gradient.start = {};
            gradient.end = {};
        }

    }

    // Compact knot factors shared by quintic and septic splines.

    namespace detail
    {
        /** @brief One 2-by-2 diagonal factor and its coupling to the preceding knot.
         * @note A = L D L^T. Only the strict lower triangle, reciprocal pivots and coupling
         *       are stored (7 doubles). The unaligned matrix avoids padding between records. */
        struct BlockLDLT2
        {
            using Block = Eigen::Matrix<double, 2, 2, Eigen::RowMajor | Eigen::DontAlign>;
            double l10 = 0.0;
            double inverse_d0 = 0.0;
            double inverse_d1 = 0.0;
            Block coupling = Block::Zero();

            /** @brief Eliminate the preceding knot and factor the symmetric Schur block.
             * @param diagonal Symmetric physical diagonal block; only its lower triangle is read.
             * @param previous Coupling to the preceding knot in physical derivative coordinates.
             * @param prior Borrowed preceding factor, or nullptr at the first interior knot;
             *        it must remain valid for this call and is never retained.
             * @return True for finite positive pivots; false on numerical breakdown, without regularization. */
            bool factorize(const Block &diagonal, const Block &previous, const BlockLDLT2 *prior)
            {
                double s00 = diagonal(0, 0);
                double s10 = diagonal(1, 0);
                double s11 = diagonal(1, 1);

                // Transform the off-diagonal block through the preceding unit-lower factor.
                if (prior != nullptr)
                {
                    const double t00 = previous(0, 0);
                    const double t01 = previous(0, 1) - t00 * prior->l10;
                    coupling(0, 0) = t00 * prior->inverse_d0;
                    coupling(0, 1) = t01 * prior->inverse_d1;
                    const double t10 = previous(1, 0);
                    const double t11 = previous(1, 1) - t10 * prior->l10;
                    coupling(1, 0) = t10 * prior->inverse_d0;
                    coupling(1, 1) = t11 * prior->inverse_d1;

                    // Form only the lower triangle of the Schur complement.
                    s00 -= coupling(0, 0) * t00 + coupling(0, 1) * t01;
                    s10 -= coupling(1, 0) * t00 + coupling(1, 1) * t01;
                    s11 -= coupling(1, 0) * t10 + coupling(1, 1) * t11;
                }

                // Factor the small Schur block without an explicit matrix inverse.
                const double d0 = s00;
                if (!(d0 > 0.0) || !std::isfinite(d0))
                    return false;
                inverse_d0 = 1.0 / d0;
                l10 = s10 * inverse_d0;
                const double d1 = s11 - l10 * s10;
                if (!(d1 > 0.0) || !std::isfinite(d1))
                    return false;
                inverse_d1 = 1.0 / d1;
                return true;
            }

            /** @brief Apply one forward substitution block in place.
             * @param[in,out] values Knot-major right-hand sides; preceding rows already hold L^-1 b.
             * @param row First row of this interior knot; the first interior row is 2.
             * @note The matrix is exclusively borrowed for this call. Endpoint rows are untouched. */
            template <typename Matrix>
            void forward(Matrix &values, int row) const
            {
                for (int coordinate = 0; coordinate < values.cols(); ++coordinate)
                {
                    double y0 = values(row + 0, coordinate);
                    double y1 = values(row + 1, coordinate);
                    if (row > 2)
                    {
                        y0 -= coupling(0, 0) * values(row - 2, coordinate) + coupling(0, 1) * values(row - 1, coordinate);
                        y1 -= coupling(1, 0) * values(row - 2, coordinate) + coupling(1, 1) * values(row - 1, coordinate);
                    }
                    y1 -= l10 * y0;
                    values(row + 0, coordinate) = y0;
                    values(row + 1, coordinate) = y1;
                }
            }

            /** @brief Apply one backward substitution block in place.
             * @param[in,out] values Forward-substituted rows; following rows already hold the solution.
             * @param row First row of this interior knot.
             * @param next Borrowed following factor, or nullptr at the last interior knot;
             *        valid for this call only, never retained. Endpoint rows are untouched. */
            template <typename Matrix>
            void backward(Matrix &values, int row, const BlockLDLT2 *next) const
            {
                for (int coordinate = 0; coordinate < values.cols(); ++coordinate)
                {
                    double x0 = inverse_d0 * values(row + 0, coordinate);
                    double x1 = inverse_d1 * values(row + 1, coordinate);
                    if (next != nullptr)
                    {
                        x0 -= next->coupling(0, 0) * values(row + 2, coordinate) + next->coupling(1, 0) * values(row + 3, coordinate);
                        x1 -= next->coupling(0, 1) * values(row + 2, coordinate) + next->coupling(1, 1) * values(row + 3, coordinate);
                    }
                    x0 -= l10 * x1;
                    values(row + 0, coordinate) = x0;
                    values(row + 1, coordinate) = x1;
                }
            }
        };

        /** @brief One 3-by-3 diagonal factor and its coupling to the preceding knot.
         * @note A = L D L^T. Only the strict lower triangle, reciprocal pivots and coupling
         *       are stored (15 doubles). The unaligned matrix avoids padding between records. */
        struct BlockLDLT3
        {
            using Block = Eigen::Matrix<double, 3, 3, Eigen::RowMajor | Eigen::DontAlign>;
            double l10 = 0.0;
            double l20 = 0.0;
            double l21 = 0.0;
            double inverse_d0 = 0.0;
            double inverse_d1 = 0.0;
            double inverse_d2 = 0.0;
            Block coupling = Block::Zero();

            /** @brief Eliminate the preceding knot and factor the symmetric Schur block.
             * @param diagonal Symmetric physical diagonal block; only its lower triangle is read.
             * @param previous Coupling to the preceding knot in physical derivative coordinates.
             * @param prior Borrowed preceding factor, or nullptr at the first interior knot;
             *        it must remain valid for this call and is never retained.
             * @return True for finite positive pivots; false on numerical breakdown, without regularization. */
            bool factorize(const Block &diagonal, const Block &previous, const BlockLDLT3 *prior)
            {
                double s00 = diagonal(0, 0);
                double s10 = diagonal(1, 0);
                double s11 = diagonal(1, 1);
                double s20 = diagonal(2, 0);
                double s21 = diagonal(2, 1);
                double s22 = diagonal(2, 2);

                // Transform the off-diagonal block through the preceding unit-lower factor.
                if (prior != nullptr)
                {
                    const double t00 = previous(0, 0);
                    const double t01 = previous(0, 1) - t00 * prior->l10;
                    const double t02 = previous(0, 2) - t00 * prior->l20 - t01 * prior->l21;
                    coupling(0, 0) = t00 * prior->inverse_d0;
                    coupling(0, 1) = t01 * prior->inverse_d1;
                    coupling(0, 2) = t02 * prior->inverse_d2;
                    const double t10 = previous(1, 0);
                    const double t11 = previous(1, 1) - t10 * prior->l10;
                    const double t12 = previous(1, 2) - t10 * prior->l20 - t11 * prior->l21;
                    coupling(1, 0) = t10 * prior->inverse_d0;
                    coupling(1, 1) = t11 * prior->inverse_d1;
                    coupling(1, 2) = t12 * prior->inverse_d2;
                    const double t20 = previous(2, 0);
                    const double t21 = previous(2, 1) - t20 * prior->l10;
                    const double t22 = previous(2, 2) - t20 * prior->l20 - t21 * prior->l21;
                    coupling(2, 0) = t20 * prior->inverse_d0;
                    coupling(2, 1) = t21 * prior->inverse_d1;
                    coupling(2, 2) = t22 * prior->inverse_d2;

                    // Form only the lower triangle of the Schur complement.
                    s00 -= coupling(0, 0) * t00 + coupling(0, 1) * t01 + coupling(0, 2) * t02;
                    s10 -= coupling(1, 0) * t00 + coupling(1, 1) * t01 + coupling(1, 2) * t02;
                    s11 -= coupling(1, 0) * t10 + coupling(1, 1) * t11 + coupling(1, 2) * t12;
                    s20 -= coupling(2, 0) * t00 + coupling(2, 1) * t01 + coupling(2, 2) * t02;
                    s21 -= coupling(2, 0) * t10 + coupling(2, 1) * t11 + coupling(2, 2) * t12;
                    s22 -= coupling(2, 0) * t20 + coupling(2, 1) * t21 + coupling(2, 2) * t22;
                }

                // Factor the small Schur block without an explicit matrix inverse.
                const double d0 = s00;
                if (!(d0 > 0.0) || !std::isfinite(d0))
                    return false;
                inverse_d0 = 1.0 / d0;
                l10 = s10 * inverse_d0;
                l20 = s20 * inverse_d0;
                const double d1 = s11 - l10 * s10;
                if (!(d1 > 0.0) || !std::isfinite(d1))
                    return false;
                inverse_d1 = 1.0 / d1;
                // Reuse the eliminated off-diagonal entry in both remaining factors.
                const double reduced_s21 = s21 - l20 * s10;
                l21 = reduced_s21 * inverse_d1;
                const double d2 = s22 - l20 * s20 - l21 * reduced_s21;
                if (!(d2 > 0.0) || !std::isfinite(d2))
                    return false;
                inverse_d2 = 1.0 / d2;
                return true;
            }

            /** @brief Apply one forward substitution block in place.
             * @param[in,out] values Knot-major right-hand sides; preceding rows already hold L^-1 b.
             * @param row First row of this interior knot; the first interior row is 3.
             * @note The matrix is exclusively borrowed for this call. Endpoint rows are untouched. */
            template <typename Matrix>
            void forward(Matrix &values, int row) const
            {
                for (int coordinate = 0; coordinate < values.cols(); ++coordinate)
                {
                    double y0 = values(row + 0, coordinate);
                    double y1 = values(row + 1, coordinate);
                    double y2 = values(row + 2, coordinate);
                    if (row > 3)
                    {
                        y0 -= coupling(0, 0) * values(row - 3, coordinate)
                            + coupling(0, 1) * values(row - 2, coordinate)
                            + coupling(0, 2) * values(row - 1, coordinate);
                        y1 -= coupling(1, 0) * values(row - 3, coordinate)
                            + coupling(1, 1) * values(row - 2, coordinate)
                            + coupling(1, 2) * values(row - 1, coordinate);
                        y2 -= coupling(2, 0) * values(row - 3, coordinate)
                            + coupling(2, 1) * values(row - 2, coordinate)
                            + coupling(2, 2) * values(row - 1, coordinate);
                    }
                    y1 -= l10 * y0;
                    y2 -= l20 * y0 + l21 * y1;
                    values(row + 0, coordinate) = y0;
                    values(row + 1, coordinate) = y1;
                    values(row + 2, coordinate) = y2;
                }
            }

            /** @brief Apply one backward substitution block in place.
             * @param[in,out] values Forward-substituted rows; following rows already hold the solution.
             * @param row First row of this interior knot.
             * @param next Borrowed following factor, or nullptr at the last interior knot;
             *        valid for this call only, never retained. Endpoint rows are untouched. */
            template <typename Matrix>
            void backward(Matrix &values, int row, const BlockLDLT3 *next) const
            {
                for (int coordinate = 0; coordinate < values.cols(); ++coordinate)
                {
                    double x0 = inverse_d0 * values(row + 0, coordinate);
                    double x1 = inverse_d1 * values(row + 1, coordinate);
                    double x2 = inverse_d2 * values(row + 2, coordinate);
                    if (next != nullptr)
                    {
                        x0 -= next->coupling(0, 0) * values(row + 3, coordinate)
                            + next->coupling(1, 0) * values(row + 4, coordinate)
                            + next->coupling(2, 0) * values(row + 5, coordinate);
                        x1 -= next->coupling(0, 1) * values(row + 3, coordinate)
                            + next->coupling(1, 1) * values(row + 4, coordinate)
                            + next->coupling(2, 1) * values(row + 5, coordinate);
                        x2 -= next->coupling(0, 2) * values(row + 3, coordinate)
                            + next->coupling(1, 2) * values(row + 4, coordinate)
                            + next->coupling(2, 2) * values(row + 5, coordinate);
                    }
                    x1 -= l21 * x2;
                    x0 -= l10 * x1 + l20 * x2;
                    values(row + 0, coordinate) = x0;
                    values(row + 1, coordinate) = x1;
                    values(row + 2, coordinate) = x2;
                }
            }
        };

        static_assert(sizeof(BlockLDLT2) == 7 * sizeof(double),
                      "The quintic factor must not contain alignment padding");
        static_assert(sizeof(BlockLDLT3) == 15 * sizeof(double),
                      "The septic factor must not contain alignment padding");

        /** @brief Complete a block solve using cached factors, leaving clamped endpoints untouched.
         * @tparam DerivativesPerKnot Compile-time block size of the supplied factors.
         * @param factors The N-1 interior-knot factors from construction; borrowed for this call.
         * @param segments Positive segment count N associated with factors and values.
         * @param[in,out] values Forward-substituted physical derivative rows; receives the solution.
         * @note Requires matching factor and workspace topology and exclusive access to values.
         *       Descending knot order makes each following solution available without another buffer. */
        template <int DerivativesPerKnot, typename Factor, typename Matrix>
        inline void backSubstitute(const std::vector<Factor> &factors, int segments, Matrix &values)
        {
            for (int knot = segments - 1; knot > 0; --knot)
            {
                const Factor *next = knot + 1 < segments ? &factors[knot] : nullptr;
                factors[knot - 1].backward(values, knot * DerivativesPerKnot, next);
            }
        }

        /** @brief Solve an interior-knot system using the cached primal factors.
         * @tparam DerivativesPerKnot Compile-time block size of the supplied factors.
         * @param factors The N-1 interior-knot factors from construction; borrowed for this call.
         * @param segments Positive segment count N associated with factors and values.
         * @param[in,out] values Knot-major RHS with prepared endpoint rows; receives the interior solution.
         * @note Requires matching factor and workspace topology and exclusive access to values.
         *       Endpoint rows remain unchanged; adjoint callers set them to zero before solving. */
        template <int DerivativesPerKnot, typename Factor, typename Matrix>
        inline void solveWithFactors(const std::vector<Factor> &factors, int segments, Matrix &values)
        {
            // Forward substitution consumes the preceding knot without another RHS buffer.
            for (int knot = 1; knot < segments; ++knot)
                factors[knot - 1].forward(values, knot * DerivativesPerKnot);

            // Reuse the same descending sweep as the construction solve.
            backSubstitute<DerivativesPerKnot>(factors, segments, values);
        }
    } // namespace detail

    // Cubic splines: minimum acceleration.

    /** @brief Minimum-acceleration interpolation with a specialized O(N) scalar tridiagonal solver.
     * @note Minimize the squared second physical derivative integral with fixed waypoint
     *       positions and endpoint derivatives of orders 1 through 1. Coordinates
     *       are caller-defined; durations use seconds. Polynomial coefficients and knots
     *       have one owner. Updates and backward() require exclusive access; const energy
     *       and polynomial queries may run concurrently on an unchanged valid object.
     *       Copies own independent data. Moving invalidates borrowed views and the source polynomial.
     *       Updates return status and invalidate the polynomial on numerical failure; factories return StatusOr.
     *       Inputs are trusted: finite waypoints/boundaries/start, positive finite durations or strictly
     *       increasing finite knots, N >= 1, N+1 waypoint rows, and N*kCoefficientCount within int range.
     *       Energy and adjoint queries require successful construction and return plain values or void.
     *       Reference gradient outputs must be pre-sized; resetTopology(N) prepares full gradients.
     *       Adjoint inputs must be finite with N*kCoefficientCount coefficient rows and N time partials.
     *       All public overloads use this single contract; callers perform input validation and
     *       guarantee a representable working range. Coefficients, energy and gradients are not
     *       rescanned for overflow; construction still reports knot or factorization failure. */
    template <int DIM>
    class CubicSplineND
    {
    public:
        using Vector = Eigen::Matrix<double, DIM, 1>;
        using RowVector = Eigen::Matrix<double, 1, DIM>;
        static constexpr int kMatrixOptions = (DIM == 1) ? Eigen::ColMajor : Eigen::RowMajor;
        using CoefficientMatrix = Eigen::Matrix<double, Eigen::Dynamic, DIM, kMatrixOptions>;

        static constexpr int kDimension = DIM;
        static constexpr int kDegree = 3;
        static constexpr int kCoefficientCount = 4;
        static constexpr int kMinDerivativeOrder = 2;
        using Polynomial = PPolyND<DIM, kCoefficientCount>;

        using BoundaryGradient = SplineTrajectory::BoundaryGradient<DIM>;
        using BoundaryGradients = BoundaryGradientPair<DIM>;
        using Gradients = SplineGradients<DIM>;

    private:
        std::vector<double> time_segments_;
        CoefficientMatrix spatial_points_;
        BoundaryConditions<DIM> boundary_;
        int num_segments_;
        double start_time_;
        Polynomial trajectory_;

        // Physical velocities at all knots; only the interior rows are solved.
        CoefficientMatrix derivatives_;
        CoefficientMatrix segment_slopes_;
        // Normalized upper entries and physical inverse pivots are reused by the adjoint.
        Eigen::VectorXd upper_multipliers_;
        Eigen::VectorXd inverse_pivots_;
        CoefficientMatrix adjoint_;

        struct TimePowers
        {
            double h;
            double h_inv;  // h^-1
            double h2_inv; // h^-2
            double h3_inv; // h^-3
        };
        std::vector<TimePowers> time_powers_;

    public:
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW

        /** @brief Construct an empty, invalid spline with independent storage. */
        CubicSplineND() : num_segments_(0), start_time_(0.0) {}

        /** @brief Construct from N+1 strictly increasing finite knots and N+1 waypoint rows.
         * @param t_points Absolute times in seconds. @param spatial_points Physical waypoint rows.
         * @param boundary Clamped physical endpoint derivatives in the caller's frame.
         * @note Inputs are copied; failure is returned without publishing an invalid value.
         * @return Success, or a state/resource/numerical failure; input validity is a precondition. */
        static StatusOr<CubicSplineND> create(const std::vector<double> &t_points, const CoefficientMatrix &spatial_points,
                                    const BoundaryConditions<DIM> &boundary = {})
        {
            CubicSplineND result;
            if (const auto status = result.update(t_points, spatial_points, boundary); !status) return status;
            return result;
        }

        /** @brief Construct from N positive durations, N+1 waypoint rows and a common time origin.
         * @param time_segments Durations in seconds. @param spatial_points Physical waypoint rows.
         * @param start_time Absolute time origin in seconds.
         * @param boundary Clamped physical endpoint derivatives in the caller's frame.
         * @note Inputs are copied; failure is returned without publishing an invalid value.
         * @return Success, or a state/resource/numerical failure; input validity is a precondition. */
        static StatusOr<CubicSplineND> create(const std::vector<double> &time_segments, const CoefficientMatrix &spatial_points,
                                    double start_time, const BoundaryConditions<DIM> &boundary = {})
        {
            CubicSplineND result;
            if (const auto status = result.update(time_segments, spatial_points, start_time, boundary); !status) return status;
            return result;
        }

        /** @brief Replace interpolation data using caller-validated absolute knots.
         * @param t_points Absolute times in seconds. @param spatial_points Physical waypoint rows.
         * @param boundary Clamped physical endpoint derivatives in the caller's frame.
         * @pre Inputs satisfy the class contract; no input validation is performed.
         * @note Copies inputs, reuses fixed-topology storage and invalidates all borrowed views.
         * @return Success, or a state/resource/numerical failure; input validity is a precondition. */
        Status update(const std::vector<double> &t_points,
                    const CoefficientMatrix &spatial_points,
                    const BoundaryConditions<DIM> &boundary = BoundaryConditions<DIM>())
        {
            detail::durationsFromKnots(t_points, time_segments_, start_time_);
            return update(time_segments_, spatial_points, start_time_, boundary);
        }

        /** @brief Replace interpolation data using positive durations and a common time origin.
         * @param time_segments Durations in seconds. @param spatial_points Physical waypoint rows.
         * @param start_time Absolute time origin in seconds.
         * @param boundary Clamped physical endpoint derivatives in the caller's frame.
         * @pre Inputs satisfy the class contract; no input validation is performed.
         * @note Copies inputs, reuses fixed-topology storage and invalidates all borrowed views.
         *       Numeric construction failures clear the polynomial; inputs are not validated here.
         * @return Success, or a state/resource/numerical failure; input validity is a precondition. */
        Status update(const std::vector<double> &time_segments,
                    const CoefficientMatrix &spatial_points,
                    double start_time,
                    const BoundaryConditions<DIM> &boundary = BoundaryConditions<DIM>())
        {
            time_segments_ = time_segments;
            spatial_points_ = spatial_points;
            start_time_ = start_time;
            boundary_ = boundary;
            return updateSplineInternal();
        }

        /** @brief Inspect the published polynomial. @return True after successful construction; numerical range is a caller precondition. */
        bool isValid() const { return trajectory_.isValid(); }
        /** @brief Inspect spatial dimension. @return Compile-time DIM. */
        int dimension() const { return DIM; }

        /** @brief Inspect the absolute start time. @return Seconds, or zero for an invalid spline. */
        double startTime() const
        {
            return trajectory_.startTime();
        }

        /** @brief Inspect the absolute end time. @return Seconds, or zero for an invalid spline. */
        double endTime() const
        {
            return trajectory_.endTime();
        }

        /** @brief Inspect total duration. @return Seconds, or zero for an invalid spline. */
        double duration() const
        {
            return trajectory_.duration();
        }

        /** @brief Inspect input topology. @return Number of stored waypoints. */
        size_t numPoints() const
        {
            return static_cast<size_t>(spatial_points_.rows());
        }

        /** @brief Inspect the polynomial topology. @return Segment count, or zero when invalid. */
        int numSegments() const
        {
            return trajectory_.numSegments();
        }

        /** @brief Borrow physical waypoints. @return Storage valid until update, assignment or move. */
        const CoefficientMatrix &waypoints() const { return spatial_points_; }
        /** @brief Borrow segment durations in seconds. @return Storage valid until the next mutation. */
        const std::vector<double> &durations() const { return time_segments_; }
        /** @brief Borrow absolute knots in seconds. @return Storage valid until the next mutation. */
        const std::vector<double> &breakpoints() const { return trajectory_.breakpoints_; }
        /** @brief Borrow physical boundary data. @return Storage valid until the next mutation. */
        const BoundaryConditions<DIM> &boundary() const { return boundary_; }

        /** @brief Borrow the polynomial. @return View valid until the next update, assignment or move. */
        const Polynomial &polynomial() const & { return trajectory_; }
        /** @brief Prevent polynomial borrowing from a temporary spline. @return No callable rvalue overload. */
        const Polynomial &polynomial() const && = delete;
        /** @brief Make an independent owning snapshot. @return Polynomial with its own storage and lifetime. */
        Polynomial copyPolynomial() const { return trajectory_; }

        /** @brief Evaluate monomial bases in local physical time, through crackle.
         * @param t Local time in seconds.
         * @param[out] b_pos Position basis. @param[out] b_vel Velocity basis.
         * @param[out] b_acc Acceleration basis. @param[out] b_jerk Jerk basis.
         * @param[out] b_snap Snap basis. @param[out] b_crackle Crackle basis.
         * @note Derivatives above this polynomial's degree are zero. No shared state is touched. */
        static inline void computeBasisFunctions(double t,
                                                 Eigen::Matrix<double, 1, kCoefficientCount> &b_pos,
                                                 Eigen::Matrix<double, 1, kCoefficientCount> &b_vel,
                                                 Eigen::Matrix<double, 1, kCoefficientCount> &b_acc,
                                                 Eigen::Matrix<double, 1, kCoefficientCount> &b_jerk,
                                                 Eigen::Matrix<double, 1, kCoefficientCount> &b_snap,
                                                 Eigen::Matrix<double, 1, kCoefficientCount> &b_crackle)
        {
            double t1 = t, t2 = t1 * t1, t3 = t2 * t1;

            b_pos << 1, t1, t2, t3;
            b_vel << 0, 1, 2 * t1, 3 * t2;
            b_acc << 0, 0, 2, 6 * t1;
            b_jerk << 0, 0, 0, 6;
            b_snap.setZero();
            b_crackle.setZero();
        }

        /** @brief Integrate the squared minimum physical derivative.
         * @return Energy in physical units; requires a successfully constructed spline. */
        double energy() const
        {
            assert(trajectory_.isValid());

            double total_energy = 0.0;
            for (int i = 0; i < num_segments_; ++i)
            {
                const double T = time_segments_[i];

                const double T2 = T * T;
                const double T3 = T2 * T;

                RowVector c = trajectory_.coefficients_.row(i * 4 + 2);
                RowVector d = trajectory_.coefficients_.row(i * 4 + 3);

                total_energy += 12.0 * d.squaredNorm() * T3 +
                                12.0 * c.dot(d) * T2 +
                                4.0 * c.squaredNorm() * T;
            }

            return total_energy;
        }

        /**
         * @brief Compute partial gradient of energy w.r.t. polynomial coefficients.
         *
         * @param[out] gdC Overwritten partial gradient ∂E/∂C, size (num_segments * 4) × DIM.
         *         Rows follow the same piece-major ascending-power order as coefficients(). Durations are held fixed.
         * @note This computes only the direct partial derivative, not the full gradient.
         * @pre The spline is valid and reference outputs already have the required shapes. */
        void energyCoefficientPartials(CoefficientMatrix &gdC) const
        {
            assert(trajectory_.isValid());
            gdC.setZero();

            for (int i = 0; i < num_segments_; ++i)
            {
                const double T = time_segments_[i];
                double T2 = T * T;
                double T3 = T2 * T;

                const RowVector c2 = trajectory_.coefficients_.row(i * 4 + 2);
                const RowVector c3 = trajectory_.coefficients_.row(i * 4 + 3);

                gdC.row(i * 4 + 2) = 8.0 * c2 * T + 12.0 * c3 * T2;

                gdC.row(i * 4 + 3) = 12.0 * c2 * T2 + 24.0 * c3 * T3;
            }
        }

        /** @brief Allocate direct energy coefficient partials. @return Piece-major physical coefficient gradients. */
        CoefficientMatrix energyCoefficientPartials() const
        {
            CoefficientMatrix gdC(num_segments_ * kCoefficientCount, DIM);
            energyCoefficientPartials(gdC);
            return gdC;
        }

        /**
         * @brief Compute partial gradient of energy w.r.t. segment durations.
         *
         * @param[out] gdT Overwritten partial gradient ∂E/∂T, size num_segments.
         *         Element i is the direct partial derivative w.r.t. T[i].
         * @note Local physical-time coefficients are held fixed; spline interpolation constraints are not differentiated.
         * @pre The spline is valid and reference outputs already have the required shapes. */
        void energyDurationPartials(Eigen::VectorXd &gdT) const
        {
            assert(trajectory_.isValid());

            for (int i = 0; i < num_segments_; ++i)
            {
                const double T = time_segments_[i];

                const RowVector c2 = trajectory_.coefficients_.row(i * 4 + 2);
                const RowVector c3 = trajectory_.coefficients_.row(i * 4 + 3);

                RowVector acc_end = 2.0 * c2 + (6.0 * T) * c3;

                gdT(i) = acc_end.squaredNorm();
            }
        }

        /** @brief Allocate direct energy duration partials. @return Physical duration gradients at fixed coefficients. */
        Eigen::VectorXd energyDurationPartials() const
        {
            Eigen::VectorXd gdT(num_segments_);
            energyDurationPartials(gdT);
            return gdT;
        }

        /**
         * @brief Compute full gradient of energy w.r.t. segment durations.
         *
         * @param[out] grad Full gradient dE/dT, size num_segments.
         *         Includes both direct and indirect dependencies via chain rule.
         * @pre The spline is valid and reference outputs already have the required shapes. */
        void energyDurationGradient(Eigen::VectorXd &grad) const
        {
            assert(trajectory_.isValid());
            computeEnergyDurationGradient<false>(grad, 1.0);
        }

        /** @brief Allocate total energy duration gradients. @return Gradients including interpolation dependence. */
        Eigen::VectorXd energyDurationGradient() const
        {
            Eigen::VectorXd grad(num_segments_);
            energyDurationGradient(grad);
            return grad;
        }

        /**
         * @brief Compute full gradient of energy w.r.t. inner waypoints only.
         * @param[out] grad Full gradient dE/dP for inner points only.
         *         Size (N-1) × DIM (excludes boundary points P0 and PN).
         * @pre The spline is valid and reference outputs already have the required shapes. */
        void energyWaypointGradient(CoefficientMatrix &grad) const
        {
            assert(trajectory_.isValid());
            detail::computeEnergyWaypointGradient<12>(trajectory_, grad, 1.0);
        }

        /** @brief Allocate total energy waypoint gradients. @return Interior physical position gradients. */
        CoefficientMatrix energyWaypointGradient() const
        {
            CoefficientMatrix grad(num_segments_ - 1, DIM);
            energyWaypointGradient(grad);
            return grad;
        }

        /**
         * @brief Compute energy gradients w.r.t. boundary position and velocity.
         * @param[out] grad Overwritten start/end gradients in the caller's coordinates and
         *        physical-time derivative convention (seconds); acceleration and jerk are zero.
         * @pre The spline is valid and reference outputs already have the required shapes.
         * @note Valid calls do not allocate. Concurrent calls require separate outputs and no spline updates.
         */
        void energyBoundaryGradient(BoundaryGradients &grad) const
        {
            assert(trajectory_.isValid());
            computeEnergyBoundaryGradient<false>(grad.start, grad.end, 1.0);
        }

        /** @brief Return energy gradients w.r.t. boundary position and velocity.
         * @return Owned start/end gradients using physical seconds and the caller's coordinates;
         *         acceleration and jerk gradients are zero.
         * @pre The spline is valid.
         * @note Valid calls do not allocate. Concurrent calls require no spline updates. */
        BoundaryGradients energyBoundaryGradient() const
        {
            BoundaryGradients res;
            energyBoundaryGradient(res);
            return res;
        }

        /**
         * @brief Compute full energy gradient combining time, inner point, and boundary gradients.
         *
         * @return Gradients Structure containing:
         *       - inner_points: gradient w.r.t. inner waypoints, size (N-1) × DIM
         *       - durations: gradient w.r.t. physical segment durations, size N
         *       - start/end: gradients w.r.t. boundary states (position, velocity)
         */
        Gradients energyGradient() const
        {
            Gradients res;
            res.resetTopology(num_segments_);
            energyGradient(res);
            return res;
        }

        /**
         * @brief Compute full energy gradient combining time, inner point, and boundary gradients (reference overload).
         *
         * @param[out] grads Overwritten physical gradients; buffers must already match the spline topology.
         * @note Computes waypoint jumps, endpoint terms and duration contributions using the current spline.
         * @pre The spline is valid and reference outputs already have the required shapes. */
        void energyGradient(Gradients &grads) const
        {
            assert(trajectory_.isValid());
            detail::computeEnergyWaypointGradient<12>(trajectory_, grads.inner_points, 1.0);
            computeEnergyDurationGradient<false>(grads.durations, 1.0);
            computeEnergyBoundaryGradient<false>(grads.start, grads.end, 1.0);
        }

        /** @brief Add weighted minimum-derivative energy gradients to existing physical partials.
         * @param[in,out] grads Prepared, initialized gradients matching this spline's topology.
         * @param weight Energy multiplier; uses the same physical units as energy().
         * @pre The latest construction succeeded and arithmetic is representable.
         * @note No allocation or resizing; requires exclusive output access and no spline mutation. */
        void energyGradientAdd(Gradients &grads, double weight = 1.0) const
        {
            assert(trajectory_.isValid());
            detail::computeEnergyWaypointGradient<12, true>(trajectory_, grads.inner_points, weight);
            computeEnergyDurationGradient<true>(grads.durations, weight);
            computeEnergyBoundaryGradient<true>(grads.start, grads.end, weight);
        }

        /**
         * @brief Propagate gradients from coefficients/times to waypoints and boundaries.
         *
         * @param partialGradByCoeffs Input partial gradient ∂L/∂C, size (4N) × DIM.
         * @param partialGradByTimes  Input partial gradient ∂L/∂T, size N.
         * @return Gradients Structure containing:
         *       - inner_points: gradient w.r.t. inner waypoints, size (N-1) × DIM
         *       - durations: gradient w.r.t. physical segment durations, size N
         *       - start/end: gradients w.r.t. boundary states (position, velocity)
         */
        Gradients backward(const CoefficientMatrix &partialGradByCoeffs,
                           const Eigen::VectorXd &partialGradByTimes)
        {
            Gradients res;
            res.resetTopology(num_segments_);

            backward(partialGradByCoeffs, partialGradByTimes, res);

            return res;
        }

        /**
         * @brief Propagate gradients from coefficients/times to waypoints and boundaries (reference overload).
         *
         * @param partialGradByCoeffs Input partial gradient ∂L/∂C, size (4N) × DIM.
         * @param partialGradByTimes  Input partial gradient ∂L/∂T, size N.
         * @param[out] grads Overwritten physical gradients; buffers must already match the spline topology.
         * @pre Input partials are finite and match the constructed spline; no input scan is performed.
         * @note Pulls back the Hermite map, reuses the cached knot solve, then adds endpoint corrections.
         * @pre The spline is valid and reference outputs already have the required shapes. */
        void backward(const CoefficientMatrix &partialGradByCoeffs,
                      const Eigen::VectorXd &partialGradByTimes,
                      Gradients &grads)
        {
            assert(trajectory_.isValid());
            detail::initializeGradients(partialGradByTimes, grads);
            adjoint_.setZero();

            // Local Hermite map, cached velocity solve, then endpoint corrections.
            pullbackCoefficients(partialGradByCoeffs, grads);
            solveAdjoint(grads);
            // A fully clamped segment has no free knot derivatives to correct.
            if (num_segments_ > 1)
                accumulateAdjoint(grads);
        }

    private:
        /** @brief Write duration terms from the endpoint energy identity.
         * @tparam Add Select accumulation or overwrite at compile time.
         * @param[in,out] grad Pre-sized duration partials in physical seconds.
         * @param weight Energy multiplier. @pre The spline is valid; arithmetic is representable. */
        template<bool Add>
        void computeEnergyDurationGradient(Eigen::VectorXd &grad, double weight) const
        {
            for (int i = 0; i < num_segments_; ++i)
            {
                const RowVector c1 = trajectory_.coefficients_.row(i * 4 + 1);
                const RowVector c2 = trajectory_.coefficients_.row(i * 4 + 2);
                const RowVector c3 = trajectory_.coefficients_.row(i * 4 + 3);

                double term_acc = 4.0 * c2.squaredNorm();
                double term_jv = 12.0 * c1.dot(c3);

                detail::storeGradient<Add>(grad(i), weight * (-term_acc + term_jv));
            }
        }

        /** @brief Write integration-by-parts terms at the two clamped endpoints.
         * @tparam Add Select accumulation or overwrite at compile time.
         * @param[in,out] start Start endpoint partials in physical coordinates.
         * @param[in,out] end End endpoint partials in physical coordinates.
         * @param weight Energy multiplier. @pre The spline is valid; outputs are exclusive. */
        template<bool Add>
        void computeEnergyBoundaryGradient(BoundaryGradient &start, BoundaryGradient &end, double weight) const
        {
            if constexpr (!Add)
            {
                start.a.setZero(); end.a.setZero();
                start.j.setZero(); end.j.setZero();
            }

            // Evaluate the start-boundary terms at local time zero.
            const RowVector c2_first = trajectory_.coefficients_.row(2);
            const RowVector c3_first = trajectory_.coefficients_.row(3);

            detail::storeGradient<Add>(start.p, weight * (12.0 * c3_first.transpose()));
            detail::storeGradient<Add>(start.v, weight * (-4.0 * c2_first.transpose()));

            // Evaluate the end-boundary terms at the final segment's physical duration.
            int last_idx = num_segments_ - 1;
            double T = time_segments_[last_idx];

            const RowVector c2_last = trajectory_.coefficients_.row(last_idx * 4 + 2);
            const RowVector c3_last = trajectory_.coefficients_.row(last_idx * 4 + 3);

            RowVector acc_end = 2.0 * c2_last + 6.0 * c3_last * T;
            RowVector jerk_end = 6.0 * c3_last;

            detail::storeGradient<Add>(end.p, weight * (-2.0 * jerk_end.transpose()));
            detail::storeGradient<Add>(end.v, weight * (2.0 * acc_end.transpose()));
        }

        // Construction: prepare caches, solve knot derivatives, publish coefficients.
        /** @brief Rebuild from physical data; failed construction clears the published polynomial.
         * @note Requires exclusive access. Storage is reused when the topology is unchanged.
         * @return Success, or a state/resource/numerical failure; input validity is a precondition. */
        Status updateSplineInternal()
        {
            trajectory_.is_initialized_ = false;
            const auto times = trajectory_.prepareCoefficients(time_segments_, start_time_, kCoefficientCount);
            if (!times) { num_segments_ = 0; return times; }
            num_segments_ = static_cast<int>(time_segments_.size());

            prepareWorkspace();
            precomputeTimePowers();
            precomputeSlopes();
            solveInternalDerivatives();
            writeCoefficients();
            trajectory_.publishCoefficients();
            return Status::success();
        }

        /** @brief Reuse topology-dependent storage and refresh the clamped endpoint derivatives.
         * @note Requires validated input. Same-topology updates skip all cache resizing;
         *       the first adjoint reuses storage prepared during construction. */
        void prepareWorkspace()
        {
            if (time_powers_.size() != time_segments_.size())
                resizeWorkspace();

            // Boundary velocities change independently of the segment count.
            derivatives_.row(0) = boundary_.start_velocity.transpose();
            derivatives_.row(num_segments_) = boundary_.end_velocity.transpose();
        }

        /** @brief Resize primal, adjoint and factor caches after a segment-count change.
         * @note Duration-cache size records the prepared topology and is committed last.
         *       Allocation order is preserved; same-topology updates skip this function. */
        void resizeWorkspace()
        {
            adjoint_.resize(num_segments_ + 1, DIM);
            // Reserve in the original allocation order; publish the size after all caches are ready.
            time_powers_.reserve(num_segments_);
            segment_slopes_.resize(num_segments_, DIM);
            derivatives_.resize(num_segments_ + 1, DIM);
            upper_multipliers_.resize(num_segments_ - 1);
            inverse_pivots_.resize(num_segments_ - 1);
            time_powers_.resize(num_segments_);
        }

        /** @brief Cache physical duration reciprocals for interpolation and adjoints. */
        void precomputeTimePowers()
        {
            for (int i = 0; i < num_segments_; ++i)
            {
                double h = time_segments_[i];
                double iv = 1.0 / h;
                double iv2 = iv * iv;
                double iv3 = iv2 * iv;

                time_powers_[i].h = h;
                time_powers_[i].h_inv = iv;
                time_powers_[i].h2_inv = iv2;
                time_powers_[i].h3_inv = iv3;
            }
        }

        /** @brief Cache segment slopes in physical coordinate units per second. */
        void precomputeSlopes()
        {
            for (int i = 0; i < num_segments_; ++i)
                segment_slopes_.row(i) =
                    (spatial_points_.row(i + 1) - spatial_points_.row(i)) * time_powers_[i].h_inv;
        }

        /** @brief Solve the symmetric minimum-acceleration system for free interior velocities.
         * @note Endpoint velocities are clamped. A single segment needs no linear solve.
         *       Dividing each row by its upper coefficient removes a multiplication from
         *       the scalar elimination dependency chain; cached pivots retain physical units. */
        void solveInternalDerivatives()
        {
            // Half the energy has Hessian diagonal 4/T_left + 4/T_right and upper 2/T_right.
            for (int knot = 1; knot < num_segments_; ++knot)
            {
                const double duration_ratio = time_powers_[knot].h * time_powers_[knot - 1].h_inv;
                const double diagonal = 2.0 * (duration_ratio + 1.0);
                const double previous_upper = knot > 1 ? upper_multipliers_(knot - 2) : 0.0;
                const double inverse = 1.0 / (diagonal - duration_ratio * previous_upper);
                upper_multipliers_(knot - 1) = inverse;
                inverse_pivots_(knot - 1) = (0.5 * time_powers_[knot].h) * inverse;

                RowVector rhs = 3.0 * (duration_ratio * segment_slopes_.row(knot - 1) +
                                      segment_slopes_.row(knot));
                // The preceding row is either clamped or already forward-substituted.
                rhs -= duration_ratio * derivatives_.row(knot - 1);
                if (knot + 1 == num_segments_)
                    rhs -= derivatives_.row(num_segments_);
                derivatives_.row(knot) = inverse * rhs;
            }
            backSubstitute(derivatives_);
        }

        /** @brief Complete a primal or adjoint solve without changing the clamped endpoints.
         * @param[in,out] values Forward-substituted knot rows; receives the physical solution. */
        void backSubstitute(CoefficientMatrix &values) const
        {
            // The final boundary contribution was already moved to the RHS.
            for (int knot = num_segments_ - 2; knot > 0; --knot)
                values.row(knot) -= upper_multipliers_(knot - 1) * values.row(knot + 1);
        }

        /** @brief Generate physical cubic coefficients from endpoint positions and velocities. */
        void writeCoefficients()
        {
            for (int i = 0; i < num_segments_; ++i)
            {
                const auto &tp = time_powers_[i];
                trajectory_.coefficients_.row(i * 4) = spatial_points_.row(i);
                trajectory_.coefficients_.row(i * 4 + 1) = derivatives_.row(i);
                trajectory_.coefficients_.row(i * 4 + 2) = tp.h_inv *
                    (3.0 * segment_slopes_.row(i) - 2.0 * derivatives_.row(i) - derivatives_.row(i + 1));
                trajectory_.coefficients_.row(i * 4 + 3) = tp.h2_inv *
                    (-2.0 * segment_slopes_.row(i) + derivatives_.row(i) + derivatives_.row(i + 1));
            }
        }

        /** @brief Pull back the cubic Hermite map at fixed physical knot velocities.
         * @param coefficient_partials Piece-major physical coefficient partials, borrowed for this call.
         * @param[in,out] gradient Receives direct point/time partials; adjoint_ receives velocity seeds. */
        void pullbackCoefficients(const CoefficientMatrix &coefficient_partials, Gradients &gradient)
        {
            for (int i = 0; i < num_segments_; ++i)
            {
                const auto &tp = time_powers_[i];
                const RowVector grad_c2 = coefficient_partials.row(i * 4 + 2);
                const RowVector grad_c3 = coefficient_partials.row(i * 4 + 3);

                const RowVector point_partial = (3.0 * tp.h2_inv) * grad_c2 - (2.0 * tp.h3_inv) * grad_c3;
                gradient.addWaypoint(i, coefficient_partials.row(i * 4) - point_partial);
                gradient.addWaypoint(i + 1, point_partial);

                const RowVector left_partial = (-2.0 * tp.h_inv) * grad_c2 + tp.h2_inv * grad_c3;
                const RowVector right_partial = -tp.h_inv * grad_c2 + tp.h2_inv * grad_c3;
                adjoint_.row(i) += coefficient_partials.row(i * 4 + 1) + left_partial;
                adjoint_.row(i + 1) += right_partial;

                // Homogeneity accounts for coefficient powers and the T-scaled endpoint velocities.
                const double coefficient_scaling =
                    2.0 * grad_c2.dot(trajectory_.coefficients_.row(i * 4 + 2)) +
                    3.0 * grad_c3.dot(trajectory_.coefficients_.row(i * 4 + 3));
                const double velocity_scaling = left_partial.dot(derivatives_.row(i)) +
                                                right_partial.dot(derivatives_.row(i + 1));
                gradient.durations(i) += tp.h_inv * (velocity_scaling - coefficient_scaling);
            }
        }

        /** @brief Preserve clamped velocity seeds and reuse the symmetric primal factorization.
         * @param[in,out] gradient Receives direct endpoint velocity gradients.
         * @note Only interior velocities are free; zero endpoint adjoints enforce those constraints. */
        void solveAdjoint(Gradients &gradient)
        {
            gradient.start.v = adjoint_.row(0).transpose();
            gradient.end.v = adjoint_.row(num_segments_).transpose();
            adjoint_.row(0).setZero();
            adjoint_.row(num_segments_).setZero();

            for (int knot = 1; knot < num_segments_; ++knot)
            {
                adjoint_.row(knot) -= (2.0 * time_powers_[knot - 1].h_inv) * adjoint_.row(knot - 1);
                adjoint_.row(knot) *= inverse_pivots_(knot - 1);
            }
            backSubstitute(adjoint_);
        }

        /** @brief Add cubic adjoint endpoint forces and the polarized Hamiltonian time correction.
         * @param[in,out] gradient Total physical point, time and clamped velocity gradients. */
        void accumulateAdjoint(Gradients &gradient) const
        {
            for (int i = 0; i < num_segments_; ++i)
            {
                const auto &tp = time_powers_[i];
                // The adjoint has zero positions and the solved knot velocity adjoints.
                const RowVector lambda_c2 = -tp.h_inv * (2.0 * adjoint_.row(i) + adjoint_.row(i + 1));
                const RowVector lambda_c3 = tp.h2_inv * (adjoint_.row(i) + adjoint_.row(i + 1));
                const RowVector point_partial = -6.0 * lambda_c3;
                gradient.addWaypoint(i, point_partial);
                gradient.addWaypoint(i + 1, -point_partial);

                gradient.durations(i) +=
                    4.0 * trajectory_.coefficients_.row(i * 4 + 2).dot(lambda_c2) -
                    6.0 * adjoint_.row(i).dot(trajectory_.coefficients_.row(i * 4 + 3)) -
                    6.0 * derivatives_.row(i).dot(lambda_c3);

                if (i == 0)
                    gradient.start.v += 2.0 * lambda_c2.transpose();
                if (i + 1 == num_segments_)
                    gradient.end.v -= (2.0 * lambda_c2 + (6.0 * tp.h) * lambda_c3).transpose();
            }
        }
    };

    // Quintic splines: minimum jerk.

    /** @brief Minimum-jerk interpolation with a specialized O(N) 2-by-2 block tridiagonal solver.
     * @note Minimize the squared third physical derivative integral with fixed waypoint
     *       positions and endpoint derivatives of orders 1 through 2. Coordinates
     *       are caller-defined; durations use seconds. Polynomial coefficients and knots
     *       have one owner. Updates and backward() require exclusive access; const energy
     *       and polynomial queries may run concurrently on an unchanged valid object.
     *       Copies own independent data. Moving invalidates borrowed views and the source polynomial.
     *       Updates return status and invalidate the polynomial on numerical failure; factories return StatusOr.
     *       Inputs are trusted: finite waypoints/boundaries/start, positive finite durations or strictly
     *       increasing finite knots, N >= 1, N+1 waypoint rows, and N*kCoefficientCount within int range.
     *       Energy and adjoint queries require successful construction and return plain values or void.
     *       Reference gradient outputs must be pre-sized; resetTopology(N) prepares full gradients.
     *       Adjoint inputs must be finite with N*kCoefficientCount coefficient rows and N time partials.
     *       All public overloads use this single contract; callers perform input validation and
     *       guarantee a representable working range. Coefficients, energy and gradients are not
     *       rescanned for overflow; construction still reports knot or factorization failure. */
    template <int DIM>
    class QuinticSplineND
    {
    public:
        using Vector = Eigen::Matrix<double, DIM, 1>;
        using RowVector = Eigen::Matrix<double, 1, DIM>;
        static constexpr int kMatrixOptions = (DIM == 1) ? Eigen::ColMajor : Eigen::RowMajor;
        using CoefficientMatrix = Eigen::Matrix<double, Eigen::Dynamic, DIM, kMatrixOptions>;

        static constexpr int kDimension = DIM;
        static constexpr int kDegree = 5;
        static constexpr int kCoefficientCount = 6;
        static constexpr int kMinDerivativeOrder = 3;
        using Polynomial = PPolyND<DIM, kCoefficientCount>;

        using BoundaryGradient = SplineTrajectory::BoundaryGradient<DIM>;
        using BoundaryGradients = BoundaryGradientPair<DIM>;
        using Gradients = SplineGradients<DIM>;

    private:
        std::vector<double> time_segments_;
        double start_time_{0.0};

        CoefficientMatrix spatial_points_;

        BoundaryConditions<DIM> boundary_;

        int num_segments_{0};

        Polynomial trajectory_;

        // Knot-major rows: each knot's physical derivatives are adjacent in memory.
        static constexpr int kDerivativesPerKnot = 2;
        using Factor = detail::BlockLDLT2;
        using Block = typename Factor::Block;
        std::vector<Factor> factors_;
        CoefficientMatrix derivatives_;
        CoefficientMatrix adjoint_;

        struct TimePowers
        {
            double h;
            double h_inv;
            double h2_inv;
            double h3_inv;
            double h4_inv;
            double h5_inv;
        };
        std::vector<TimePowers> time_powers_;

    public:
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW

        /** @brief Construct an empty, invalid spline with independent storage. */
        QuinticSplineND() = default;

        /** @brief Construct from N+1 strictly increasing finite knots and N+1 waypoint rows.
         * @param t_points Absolute times in seconds. @param spatial_points Physical waypoint rows.
         * @param boundary Clamped physical endpoint derivatives in the caller's frame.
         * @note Inputs are copied; failure is returned without publishing an invalid value.
         * @return Success, or a state/resource/numerical failure; input validity is a precondition. */
        static StatusOr<QuinticSplineND> create(const std::vector<double> &t_points, const CoefficientMatrix &spatial_points,
                                    const BoundaryConditions<DIM> &boundary = {})
        {
            QuinticSplineND result;
            if (const auto status = result.update(t_points, spatial_points, boundary); !status) return status;
            return result;
        }

        /** @brief Construct from N positive durations, N+1 waypoint rows and a common time origin.
         * @param time_segments Durations in seconds. @param spatial_points Physical waypoint rows.
         * @param start_time Absolute time origin in seconds.
         * @param boundary Clamped physical endpoint derivatives in the caller's frame.
         * @note Inputs are copied; failure is returned without publishing an invalid value.
         * @return Success, or a state/resource/numerical failure; input validity is a precondition. */
        static StatusOr<QuinticSplineND> create(const std::vector<double> &time_segments, const CoefficientMatrix &spatial_points,
                                    double start_time, const BoundaryConditions<DIM> &boundary = {})
        {
            QuinticSplineND result;
            if (const auto status = result.update(time_segments, spatial_points, start_time, boundary); !status) return status;
            return result;
        }

        /** @brief Replace interpolation data using caller-validated absolute knots.
         * @param t_points Absolute times in seconds. @param spatial_points Physical waypoint rows.
         * @param boundary Clamped physical endpoint derivatives in the caller's frame.
         * @pre Inputs satisfy the class contract; no input validation is performed.
         * @note Copies inputs, reuses fixed-topology storage and invalidates all borrowed views.
         * @return Success, or a state/resource/numerical failure; input validity is a precondition. */
        Status update(const std::vector<double> &t_points,
                    const CoefficientMatrix &spatial_points,
                    const BoundaryConditions<DIM> &boundary = BoundaryConditions<DIM>())
        {
            detail::durationsFromKnots(t_points, time_segments_, start_time_);
            return update(time_segments_, spatial_points, start_time_, boundary);
        }

        /** @brief Replace interpolation data using positive durations and a common time origin.
         * @param time_segments Durations in seconds. @param spatial_points Physical waypoint rows.
         * @param start_time Absolute time origin in seconds.
         * @param boundary Clamped physical endpoint derivatives in the caller's frame.
         * @pre Inputs satisfy the class contract; no input validation is performed.
         * @note Copies inputs, reuses fixed-topology storage and invalidates all borrowed views.
         *       Numeric construction failures clear the polynomial; inputs are not validated here.
         * @return Success, or a state/resource/numerical failure; input validity is a precondition. */
        Status update(const std::vector<double> &time_segments,
                    const CoefficientMatrix &spatial_points,
                    double start_time,
                    const BoundaryConditions<DIM> &boundary = BoundaryConditions<DIM>())
        {
            time_segments_ = time_segments;
            spatial_points_ = spatial_points;
            start_time_ = start_time;
            boundary_ = boundary;
            return updateSplineInternal();
        }

        /** @brief Inspect the published polynomial. @return True after successful construction; numerical range is a caller precondition. */
        bool isValid() const { return trajectory_.isValid(); }
        /** @brief Inspect spatial dimension. @return Compile-time DIM. */
        int dimension() const { return DIM; }

        /** @brief Inspect the absolute start time. @return Seconds, or zero for an invalid spline. */
        double startTime() const
        {
            return trajectory_.startTime();
        }

        /** @brief Inspect the absolute end time. @return Seconds, or zero for an invalid spline. */
        double endTime() const
        {
            return trajectory_.endTime();
        }

        /** @brief Inspect total duration. @return Seconds, or zero for an invalid spline. */
        double duration() const
        {
            return trajectory_.duration();
        }

        /** @brief Inspect input topology. @return Number of stored waypoints. */
        size_t numPoints() const
        {
            return static_cast<size_t>(spatial_points_.rows());
        }

        /** @brief Inspect the polynomial topology. @return Segment count, or zero when invalid. */
        int numSegments() const
        {
            return trajectory_.numSegments();
        }

        /** @brief Borrow physical waypoints. @return Storage valid until update, assignment or move. */
        const CoefficientMatrix &waypoints() const { return spatial_points_; }
        /** @brief Borrow segment durations in seconds. @return Storage valid until the next mutation. */
        const std::vector<double> &durations() const { return time_segments_; }
        /** @brief Borrow absolute knots in seconds. @return Storage valid until the next mutation. */
        const std::vector<double> &breakpoints() const { return trajectory_.breakpoints_; }
        /** @brief Borrow physical boundary data. @return Storage valid until the next mutation. */
        const BoundaryConditions<DIM> &boundary() const { return boundary_; }

        /** @brief Borrow the polynomial. @return View valid until the next update, assignment or move. */
        const Polynomial &polynomial() const & { return trajectory_; }
        /** @brief Prevent polynomial borrowing from a temporary spline. @return No callable rvalue overload. */
        const Polynomial &polynomial() const && = delete;
        /** @brief Make an independent owning snapshot. @return Polynomial with its own storage and lifetime. */
        Polynomial copyPolynomial() const { return trajectory_; }

        /** @brief Evaluate monomial bases in local physical time, through crackle.
         * @param t Local time in seconds.
         * @param[out] b_pos Position basis. @param[out] b_vel Velocity basis.
         * @param[out] b_acc Acceleration basis. @param[out] b_jerk Jerk basis.
         * @param[out] b_snap Snap basis. @param[out] b_crackle Crackle basis.
         * @note Derivatives above this polynomial's degree are zero. No shared state is touched. */
        static inline void computeBasisFunctions(double t,
                                                 Eigen::Matrix<double, 1, kCoefficientCount> &b_pos,
                                                 Eigen::Matrix<double, 1, kCoefficientCount> &b_vel,
                                                 Eigen::Matrix<double, 1, kCoefficientCount> &b_acc,
                                                 Eigen::Matrix<double, 1, kCoefficientCount> &b_jerk,
                                                 Eigen::Matrix<double, 1, kCoefficientCount> &b_snap,
                                                 Eigen::Matrix<double, 1, kCoefficientCount> &b_crackle)
        {
            double t1 = t, t2 = t1 * t1, t3 = t2 * t1, t4 = t3 * t1, t5 = t4 * t1;

            b_pos << 1, t1, t2, t3, t4, t5;
            b_vel << 0, 1, 2 * t1, 3 * t2, 4 * t3, 5 * t4;
            b_acc << 0, 0, 2, 6 * t1, 12 * t2, 20 * t3;
            b_jerk << 0, 0, 0, 6, 24 * t1, 60 * t2;
            b_snap << 0, 0, 0, 0, 24, 120 * t1;
            b_crackle << 0, 0, 0, 0, 0, 120;
        }

        /** @brief Integrate the squared minimum physical derivative.
         * @return Energy in physical units; requires a successfully constructed spline. */
        double energy() const
        {
            assert(trajectory_.isValid());

            double total_energy = 0.0;
            for (int i = 0; i < num_segments_; ++i)
            {
                const double T = time_segments_[i];

                const double T2 = T * T;
                const double T3 = T2 * T;
                const double T4 = T3 * T;
                const double T5 = T4 * T;

                RowVector c3 = trajectory_.coefficients_.row(i * 6 + 3);
                RowVector c4 = trajectory_.coefficients_.row(i * 6 + 4);
                RowVector c5 = trajectory_.coefficients_.row(i * 6 + 5);

                total_energy += 36.0 * c3.squaredNorm() * T +
                                144.0 * c4.dot(c3) * T2 +
                                192.0 * c4.squaredNorm() * T3 +
                                240.0 * c5.dot(c3) * T3 +
                                720.0 * c5.dot(c4) * T4 +
                                720.0 * c5.squaredNorm() * T5;
            }

            return total_energy;
        }

        /**
         * @brief Compute partial gradient of energy w.r.t. polynomial coefficients.
         *
         * @param[out] gdC Overwritten partial gradient ∂E/∂C, size (num_segments * 6) × DIM.
         *         Rows follow the same piece-major ascending-power order as coefficients(). Durations are held fixed.
         * @note This computes only the direct partial derivative, not the full gradient.
         * @pre The spline is valid and reference outputs already have the required shapes. */
        void energyCoefficientPartials(CoefficientMatrix &gdC) const
        {
            assert(trajectory_.isValid());
            gdC.setZero();

            for (int i = 0; i < num_segments_; ++i)
            {
                double T = time_segments_[i];
                double T2 = T * T;
                double T3 = T2 * T;
                double T4 = T3 * T;
                double T5 = T4 * T;

                const RowVector c3 = trajectory_.coefficients_.row(i * 6 + 3);
                const RowVector c4 = trajectory_.coefficients_.row(i * 6 + 4);
                const RowVector c5 = trajectory_.coefficients_.row(i * 6 + 5);

                gdC.row(i * 6 + 3) = 72.0 * c3 * T +
                                     144.0 * c4 * T2 +
                                     240.0 * c5 * T3;

                gdC.row(i * 6 + 4) = 144.0 * c3 * T2 +
                                     384.0 * c4 * T3 +
                                     720.0 * c5 * T4;

                gdC.row(i * 6 + 5) = 240.0 * c3 * T3 +
                                     720.0 * c4 * T4 +
                                     1440.0 * c5 * T5;
            }
        }

        /** @brief Allocate direct energy coefficient partials. @return Piece-major physical coefficient gradients. */
        CoefficientMatrix energyCoefficientPartials() const
        {
            CoefficientMatrix gdC(num_segments_ * kCoefficientCount, DIM);
            energyCoefficientPartials(gdC);
            return gdC;
        }

        /**
         * @brief Compute partial gradient of energy w.r.t. segment durations.
         *
         * @param[out] gdT Overwritten partial gradient ∂E/∂T, size num_segments.
         *         Element i is the direct partial derivative w.r.t. T[i].
         * @note Local physical-time coefficients are held fixed; spline interpolation constraints are not differentiated.
         * @pre The spline is valid and reference outputs already have the required shapes. */
        void energyDurationPartials(Eigen::VectorXd &gdT) const
        {
            assert(trajectory_.isValid());

            for (int i = 0; i < num_segments_; ++i)
            {
                const double T = time_segments_[i];

                const RowVector c3 = trajectory_.coefficients_.row(i * 6 + 3);
                const RowVector c4 = trajectory_.coefficients_.row(i * 6 + 4);
                const RowVector c5 = trajectory_.coefficients_.row(i * 6 + 5);

                RowVector jerk_end = 6.0 * c3 + T * (24.0 * c4 + (60.0 * T) * c5);

                gdT(i) = jerk_end.squaredNorm();
            }
        }

        /** @brief Allocate direct energy duration partials. @return Physical duration gradients at fixed coefficients. */
        Eigen::VectorXd energyDurationPartials() const
        {
            Eigen::VectorXd gdT(num_segments_);
            energyDurationPartials(gdT);
            return gdT;
        }

        /**
         * @brief Compute full gradient of energy w.r.t. segment durations.
         *
         * @param[out] grad Full gradient dE/dT, size num_segments.
         *         Includes both direct and indirect dependencies via chain rule.
         * @pre The spline is valid and reference outputs already have the required shapes. */
        void energyDurationGradient(Eigen::VectorXd &grad) const
        {
            assert(trajectory_.isValid());
            computeEnergyDurationGradient<false>(grad, 1.0);
        }

        /** @brief Allocate total energy duration gradients. @return Gradients including interpolation dependence. */
        Eigen::VectorXd energyDurationGradient() const
        {
            Eigen::VectorXd grad(num_segments_);
            energyDurationGradient(grad);
            return grad;
        }

        /**
         * @brief Compute full gradient of energy w.r.t. inner waypoints only.
         * @param[out] grad Full gradient dE/dP for inner points only.
         *         Size (N-1) × DIM (excludes boundary points P0 and PN).
         * @pre The spline is valid and reference outputs already have the required shapes. */
        void energyWaypointGradient(CoefficientMatrix &grad) const
        {
            assert(trajectory_.isValid());
            detail::computeEnergyWaypointGradient<-240>(trajectory_, grad, 1.0);
        }

        /** @brief Allocate total energy waypoint gradients. @return Interior physical position gradients. */
        CoefficientMatrix energyWaypointGradient() const
        {
            CoefficientMatrix grad(num_segments_ - 1, DIM);
            energyWaypointGradient(grad);
            return grad;
        }

        /**
         * @brief Compute energy gradients w.r.t. boundary position, velocity and acceleration.
         * @param[out] grad Overwritten start/end gradients in the caller's coordinates and
         *        physical-time derivative convention (seconds); jerk gradients are zero.
         * @pre The spline is valid and reference outputs already have the required shapes.
         * @note Valid calls do not allocate. Concurrent calls require separate outputs and no spline updates.
         */
        void energyBoundaryGradient(BoundaryGradients &grad) const
        {
            assert(trajectory_.isValid());
            computeEnergyBoundaryGradient<false>(grad.start, grad.end, 1.0);
        }

        /** @brief Return energy gradients w.r.t. boundary position, velocity and acceleration.
         * @return Owned start/end gradients using physical seconds and the caller's coordinates;
         *         jerk gradients are zero.
         * @pre The spline is valid.
         * @note Valid calls do not allocate. Concurrent calls require no spline updates. */
        BoundaryGradients energyBoundaryGradient() const
        {
            BoundaryGradients res;
            energyBoundaryGradient(res);
            return res;
        }

        /**
         * @brief Compute full energy gradient combining time, inner point, and boundary gradients.
         *
         * @return Gradients Structure containing:
         *       - inner_points: gradient w.r.t. inner waypoints, size (N-1) × DIM
         *       - durations: gradient w.r.t. physical segment durations, size N
         *       - start/end: gradients w.r.t. boundary states (position, velocity, acceleration)
         */
        Gradients energyGradient() const
        {
            Gradients res;
            res.resetTopology(num_segments_);
            energyGradient(res);
            return res;
        }

        /**
         * @brief Compute full energy gradient combining time, inner point, and boundary gradients (reference overload).
         *
         * @param[out] grads Overwritten physical gradients; buffers must already match the spline topology.
         * @note Computes waypoint jumps, endpoint terms and duration contributions using the current spline.
         * @pre The spline is valid and reference outputs already have the required shapes. */
        void energyGradient(Gradients &grads) const
        {
            assert(trajectory_.isValid());
            detail::computeEnergyWaypointGradient<-240>(trajectory_, grads.inner_points, 1.0);
            computeEnergyDurationGradient<false>(grads.durations, 1.0);
            computeEnergyBoundaryGradient<false>(grads.start, grads.end, 1.0);
        }

        /** @brief Add weighted minimum-derivative energy gradients to existing physical partials.
         * @param[in,out] grads Prepared, initialized gradients matching this spline's topology.
         * @param weight Energy multiplier; uses the same physical units as energy().
         * @pre The latest construction succeeded and arithmetic is representable.
         * @note No allocation or resizing; requires exclusive output access and no spline mutation. */
        void energyGradientAdd(Gradients &grads, double weight = 1.0) const
        {
            assert(trajectory_.isValid());
            detail::computeEnergyWaypointGradient<-240, true>(trajectory_, grads.inner_points, weight);
            computeEnergyDurationGradient<true>(grads.durations, weight);
            computeEnergyBoundaryGradient<true>(grads.start, grads.end, weight);
        }

        /**
         * @brief Propagate gradients from coefficients/times to waypoints and boundaries.
         *
         * @param partialGradByCoeffs Input partial gradient ∂L/∂C, size (6N) × DIM.
         * @param partialGradByTimes  Input partial gradient ∂L/∂T, size N.
         * @return Gradients Structure containing:
         *       - inner_points: gradient w.r.t. inner waypoints, size (N-1) × DIM
         *       - durations: gradient w.r.t. physical segment durations, size N
         *       - start/end: gradients w.r.t. boundary states (position, velocity, acceleration)
         */
        Gradients backward(const CoefficientMatrix &partialGradByCoeffs,
                           const Eigen::VectorXd &partialGradByTimes)
        {
            Gradients res;
            res.resetTopology(num_segments_);

            backward(partialGradByCoeffs, partialGradByTimes, res);

            return res;
        }

        /**
         * @brief Propagate gradients from coefficients/times to waypoints and boundaries (reference overload).
         *
         * @param partialGradByCoeffs Input partial gradient ∂L/∂C, size (6N) × DIM.
         * @param partialGradByTimes  Input partial gradient ∂L/∂T, size N.
         * @param[out] grads Overwritten physical gradients; buffers must already match the spline topology.
         * @pre Input partials are finite and match the constructed spline; no input scan is performed.
         * @note Pulls back the Hermite map, reuses the cached knot solve, then adds endpoint corrections.
         * @pre The spline is valid and reference outputs already have the required shapes. */
        void backward(const CoefficientMatrix &partialGradByCoeffs,
                      const Eigen::VectorXd &partialGradByTimes,
                      Gradients &grads)
        {
            assert(trajectory_.isValid());
            detail::initializeGradients(partialGradByTimes, grads);
            adjoint_.setZero();

            // Local map, cached linear solve, then endpoint corrections.
            pullbackCoefficients(partialGradByCoeffs, grads);
            solveAdjoint(grads);
            // A fully clamped segment has no free knot derivatives to correct.
            if (num_segments_ > 1)
                accumulateAdjoint(grads);
        }

    private:
        /** @brief Write duration terms from the endpoint energy identity.
         * @tparam Add Select accumulation or overwrite at compile time.
         * @param[in,out] grad Pre-sized duration partials in physical seconds.
         * @param weight Energy multiplier. @pre The spline is valid; arithmetic is representable. */
        template<bool Add>
        void computeEnergyDurationGradient(Eigen::VectorXd &grad, double weight) const
        {
            for (int i = 0; i < num_segments_; ++i)
            {
                const RowVector c1 = trajectory_.coefficients_.row(i * 6 + 1);
                const RowVector c2 = trajectory_.coefficients_.row(i * 6 + 2);
                const RowVector c3 = trajectory_.coefficients_.row(i * 6 + 3);
                const RowVector c4 = trajectory_.coefficients_.row(i * 6 + 4);
                const RowVector c5 = trajectory_.coefficients_.row(i * 6 + 5);

                double term_jerk = 36.0 * c3.squaredNorm();
                double term_sa = 96.0 * c2.dot(c4);
                double term_cv = 240.0 * c1.dot(c5);

                detail::storeGradient<Add>(grad(i), weight * (-term_jerk + term_sa - term_cv));
            }
        }

        /** @brief Write integration-by-parts terms at the two clamped endpoints.
         * @tparam Add Select accumulation or overwrite at compile time.
         * @param[in,out] start Start endpoint partials in physical coordinates.
         * @param[in,out] end End endpoint partials in physical coordinates.
         * @param weight Energy multiplier. @pre The spline is valid; outputs are exclusive. */
        template<bool Add>
        void computeEnergyBoundaryGradient(BoundaryGradient &start, BoundaryGradient &end, double weight) const
        {
            if constexpr (!Add)
            {
                start.j.setZero(); end.j.setZero();
            }

            // Evaluate the start-boundary terms at local time zero.
            const RowVector c3_first = trajectory_.coefficients_.row(3);
            const RowVector c4_first = trajectory_.coefficients_.row(4);
            const RowVector c5_first = trajectory_.coefficients_.row(5);

            detail::storeGradient<Add>(start.p, weight * (-240.0 * c5_first.transpose()));
            detail::storeGradient<Add>(start.v, weight * (48.0 * c4_first.transpose()));
            detail::storeGradient<Add>(start.a, weight * (-12.0 * c3_first.transpose()));

            // Evaluate the end-boundary terms at the final segment's physical duration.
            int last_idx = num_segments_ - 1;
            double T = time_segments_[last_idx];
            double T2 = T * T;

            const RowVector c3_last = trajectory_.coefficients_.row(last_idx * 6 + 3);
            const RowVector c4_last = trajectory_.coefficients_.row(last_idx * 6 + 4);
            const RowVector c5_last = trajectory_.coefficients_.row(last_idx * 6 + 5);

            RowVector jerk_end = 6.0 * c3_last + 24.0 * c4_last * T + 60.0 * c5_last * T2;
            RowVector snap_end = 24.0 * c4_last + 120.0 * c5_last * T;
            RowVector crackle_end = 120.0 * c5_last;

            detail::storeGradient<Add>(end.p, weight * (2.0 * crackle_end.transpose()));
            detail::storeGradient<Add>(end.v, weight * (-2.0 * snap_end.transpose()));
            detail::storeGradient<Add>(end.a, weight * (2.0 * jerk_end.transpose()));
        }

        // Construction: prepare caches, solve knot derivatives, publish coefficients.
        /** @brief Rebuild from physical data; failed construction clears the published polynomial.
         * @note Requires exclusive access. Storage is reused when the topology is unchanged.
         * @return Success, or a state/resource/numerical failure; input validity is a precondition. */
        Status updateSplineInternal()
        {
            trajectory_.is_initialized_ = false;
            const auto times = trajectory_.prepareCoefficients(time_segments_, start_time_, kCoefficientCount);
            if (!times) { num_segments_ = 0; return times; }
            num_segments_ = static_cast<int>(time_segments_.size());

            prepareWorkspace();
            precomputeTimePowers();
            if (!solveInternalDerivatives())
            {
                trajectory_.clear();
                return {ErrorCode::NumericalFailure, "Spline factorization failed"};
            }
            writeCoefficients();
            trajectory_.publishCoefficients();
            return Status::success();
        }

        /** @brief Reuse topology-dependent storage and refresh the clamped endpoint derivatives.
         * @note Requires validated input. Same-topology updates skip all cache resizing;
         *       the first adjoint reuses storage prepared during construction. */
        void prepareWorkspace()
        {
            if (time_powers_.size() != time_segments_.size())
                resizeWorkspace();

            // Boundary values change independently of the segment count.
            derivatives_.row(0) = boundary_.start_velocity.transpose();
            derivatives_.row(num_segments_ * 2 + 0) = boundary_.end_velocity.transpose();
            derivatives_.row(1) = boundary_.start_acceleration.transpose();
            derivatives_.row(num_segments_ * 2 + 1) = boundary_.end_acceleration.transpose();
        }

        /** @brief Resize primal, adjoint and factor caches after a segment-count change.
         * @note Duration-cache size records the prepared topology and is committed last.
         *       Allocation order is preserved; same-topology updates skip this function. */
        void resizeWorkspace()
        {
            adjoint_.resize((num_segments_ + 1) * kDerivativesPerKnot, DIM);
            derivatives_.resize((num_segments_ + 1) * kDerivativesPerKnot, DIM);
            factors_.resize(num_segments_ - 1);
            time_powers_.resize(num_segments_);
        }

        /** @brief Cache reciprocal duration powers used by assembly, Hermite maps and adjoints. */
        void precomputeTimePowers()
        {
            for (int i = 0; i < num_segments_; ++i)
            {
                auto &tp = time_powers_[i];
                tp.h = time_segments_[i];
                tp.h_inv = 1.0 / tp.h;
                tp.h2_inv = tp.h_inv * tp.h_inv;
                tp.h3_inv = tp.h2_inv * tp.h_inv;
                tp.h4_inv = tp.h3_inv * tp.h_inv;
                tp.h5_inv = tp.h4_inv * tp.h_inv;
            }
        }

        /** @brief Assemble one knot's stationarity equation for half the integrated squared derivative.
         * @param knot Interior knot index in [1, num_segments_).
         * @param[out] diagonal Symmetric diagonal block of the physical energy Hessian.
         * @param[out] previous Coupling to the preceding knot.
         * @note Writes this knot's RHS to derivatives_; fixed endpoint terms move to that RHS. */
        void assembleKnotSystem(int knot, Block &diagonal, Block &previous)
        {
            const auto &left = time_powers_[knot - 1];
            const auto &right = time_powers_[knot];

            // These constants are the exact unit-interval Hermite energy Gram matrix.
            diagonal(0, 0) = 192.0 * left.h3_inv + 192.0 * right.h3_inv;
            diagonal(1, 0) = -36.0 * left.h2_inv + 36.0 * right.h2_inv;
            diagonal(1, 1) = 9.0 * left.h_inv + 9.0 * right.h_inv;
            previous(0, 0) = 168.0 * left.h3_inv;
            previous(0, 1) = 24.0 * left.h2_inv;
            previous(1, 0) = -24.0 * left.h2_inv;
            previous(1, 1) = -3.0 * left.h_inv;

            const RowVector delta_left = spatial_points_.row(knot) - spatial_points_.row(knot - 1);
            const RowVector delta_right = spatial_points_.row(knot + 1) - spatial_points_.row(knot);
            RowVector rhs0 = 360.0 * left.h4_inv * delta_left + 360.0 * right.h4_inv * delta_right;
            RowVector rhs1 = -60.0 * left.h3_inv * delta_left + 60.0 * right.h3_inv * delta_right;
            if (knot == 1)
            {
                rhs0 -= previous(0, 0) * derivatives_.row(0) + previous(0, 1) * derivatives_.row(1);
                rhs1 -= previous(1, 0) * derivatives_.row(0) + previous(1, 1) * derivatives_.row(1);
            }
            if (knot + 1 == num_segments_)
            {
                rhs0 -= 168.0 * right.h3_inv * derivatives_.row(num_segments_ * 2)
                    - 24.0 * right.h2_inv * derivatives_.row(num_segments_ * 2 + 1);
                rhs1 -= 24.0 * right.h2_inv * derivatives_.row(num_segments_ * 2)
                    - 3.0 * right.h_inv * derivatives_.row(num_segments_ * 2 + 1);
            }
            derivatives_.row(knot * 2) = rhs0;
            derivatives_.row(knot * 2 + 1) = rhs1;
        }

        /** @brief Factor and solve the block system in linear time.
         * @return False on a nonpositive/nonfinite pivot; true after solving all free derivatives.
         * @note Forward substitution is fused with assembly to avoid a separate RHS sweep. */
        bool solveInternalDerivatives()
        {
            for (int knot = 1; knot < num_segments_; ++knot)
            {
                Block diagonal, previous;
                assembleKnotSystem(knot, diagonal, previous);
                auto &factor = factors_[knot - 1];
                const Factor *prior = knot > 1 ? &factors_[knot - 2] : nullptr;
                if (!factor.factorize(diagonal, previous, prior))
                    return false;
                factor.forward(derivatives_, knot * kDerivativesPerKnot);
            }
            backSubstitute(derivatives_);
            return true;
        }

        /** @brief Finish a primal or adjoint solve using the stored diagonal/coupling factors.
         * @param[in,out] values Knot-major forward-substituted RHS; receives the solution.
         * @note Endpoint rows are preserved; the caller owns and exclusively supplies the matrix. */
        void backSubstitute(CoefficientMatrix &values) const
        {
            detail::backSubstitute<kDerivativesPerKnot>(factors_, num_segments_, values);
        }

        /** @brief Generate coefficients from endpoint residuals in physical coordinates.
         * @note Each row contains contiguous spatial coordinates, so the short expressions
         *       can use Eigen packets without changing the per-order mathematical formula. */
        void writeCoefficients()
        {
            for (int i = 0; i < num_segments_; ++i)
            {
                const auto &tp = time_powers_[i];
                const int coefficient_row = i * kCoefficientCount;
                const int left_row = i * kDerivativesPerKnot;
                const int right_row = left_row + kDerivativesPerKnot;
                const RowVector position = spatial_points_.row(i);
                const RowVector velocity = derivatives_.row(left_row);
                const RowVector acceleration = derivatives_.row(left_row + 1);
                const RowVector end_velocity = derivatives_.row(right_row);
                const RowVector end_acceleration = derivatives_.row(right_row + 1);
                const RowVector quadratic = 0.5 * acceleration;

                // Match the right endpoint after subtracting the left-end Taylor polynomial.
                const RowVector displacement = spatial_points_.row(i + 1) - position;
                const RowVector position_residual = displacement - tp.h * velocity - (tp.h * tp.h) * quadratic;
                const RowVector velocity_residual = end_velocity - velocity - tp.h * acceleration;
                const RowVector acceleration_residual = end_acceleration - acceleration;
                const RowVector cubic =
                    (10.0 * tp.h3_inv) * position_residual - (4.0 * tp.h2_inv) * velocity_residual +
                    (0.5 * tp.h_inv) * acceleration_residual;
                const RowVector quartic =
                    (-15.0 * tp.h4_inv) * position_residual + (7.0 * tp.h3_inv) * velocity_residual -
                    tp.h2_inv * acceleration_residual;
                const RowVector quintic =
                    (6.0 * tp.h5_inv) * position_residual - (3.0 * tp.h4_inv) * velocity_residual +
                    (0.5 * tp.h3_inv) * acceleration_residual;

                // Publish together, after all source rows and duration powers have been read.
                trajectory_.coefficients_.row(coefficient_row) = position;
                trajectory_.coefficients_.row(coefficient_row + 1) = velocity;
                trajectory_.coefficients_.row(coefficient_row + 2) = quadratic;
                trajectory_.coefficients_.row(coefficient_row + 3) = cubic;
                trajectory_.coefficients_.row(coefficient_row + 4) = quartic;
                trajectory_.coefficients_.row(coefficient_row + 5) = quintic;
            }
        }

        /** @brief Differentiate each segment's explicit Hermite map at fixed knot derivatives.
         * @param coefficient_partials Piece-major physical coefficient partials, valid for this call.
         * @param[in,out] gradient Receives direct point/time partials; adjoint_ receives derivative seeds.
         * @note Homogeneity gives dC/dT without materializing derivative matrices. */
        void pullbackCoefficients(const CoefficientMatrix &coefficient_partials, Gradients &gradient)
        {
            for (int i = 0; i < num_segments_; ++i)
            {
                const auto &tp = time_powers_[i];
                const int coefficient_row = i * kCoefficientCount;
                const int left_row = i * kDerivativesPerKnot;
                const int right_row = left_row + kDerivativesPerKnot;

                // Reuse solved states; a fully clamped segment reads its inputs directly.
                const RowVector left_velocity = derivatives_.row(left_row);
                const RowVector left_acceleration = derivatives_.row(left_row + 1);
                const RowVector right_velocity = derivatives_.row(right_row);
                const RowVector right_acceleration = derivatives_.row(right_row + 1);

                // Apply the same Hermite map across coordinates, sharing the time multipliers.
                const RowVector g3 = coefficient_partials.row(coefficient_row + 3);
                const RowVector g4 = coefficient_partials.row(coefficient_row + 4);
                const RowVector g5 = coefficient_partials.row(coefficient_row + 5);
                const RowVector grad_v_left = (-6.0 * tp.h2_inv) * g3
                    + (8.0 * tp.h3_inv) * g4 - (3.0 * tp.h4_inv) * g5;
                const RowVector grad_a_left = (-1.5 * tp.h_inv) * g3
                    + (1.5 * tp.h2_inv) * g4 - (0.5 * tp.h3_inv) * g5;
                const RowVector point_partial = (10.0 * tp.h3_inv) * g3
                    - (15.0 * tp.h4_inv) * g4 + (6.0 * tp.h5_inv) * g5;
                const RowVector grad_v_right = (-4.0 * tp.h2_inv) * g3
                    + (7.0 * tp.h3_inv) * g4 - (3.0 * tp.h4_inv) * g5;
                const RowVector grad_a_right = (0.5 * tp.h_inv) * g3
                    - tp.h2_inv * g4 + (0.5 * tp.h3_inv) * g5;

                gradient.addWaypoint(i, coefficient_partials.row(coefficient_row) - point_partial);
                gradient.addWaypoint(i + 1, point_partial);
                adjoint_.row(left_row) += grad_v_left + coefficient_partials.row(coefficient_row + 1);
                adjoint_.row(right_row) += grad_v_right;
                adjoint_.row(left_row + 1) += grad_a_left + 0.5 * coefficient_partials.row(coefficient_row + 2);
                adjoint_.row(right_row + 1) += grad_a_right;

                // c_k scales as T^-k; endpoint derivative order r scales as T^r.
                RowVector time_partials = -3.0 * g3.cwiseProduct(trajectory_.coefficients_.row(coefficient_row + 3))
                    - 4.0 * g4.cwiseProduct(trajectory_.coefficients_.row(coefficient_row + 4))
                    - 5.0 * g5.cwiseProduct(trajectory_.coefficients_.row(coefficient_row + 5));
                time_partials += grad_v_left.cwiseProduct(left_velocity)
                    + grad_v_right.cwiseProduct(right_velocity)
                    + 2.0 * grad_a_left.cwiseProduct(left_acceleration)
                    + 2.0 * grad_a_right.cwiseProduct(right_acceleration);
                gradient.durations(i) += tp.h_inv * time_partials.sum();
            }
        }

        /** @brief Preserve clamped boundary seeds and solve for free knot adjoints.
         * @param[in,out] gradient Receives direct boundary derivative gradients.
         * @note A is symmetric: the primal factors also solve A^T lambda = seed. */
        void solveAdjoint(Gradients &gradient)
        {
            gradient.start.v = adjoint_.row(0).transpose();
            gradient.end.v = adjoint_.row(num_segments_ * 2 + 0).transpose();
            gradient.start.a = adjoint_.row(1).transpose();
            gradient.end.a = adjoint_.row(num_segments_ * 2 + 1).transpose();
            adjoint_.template topRows<2>().setZero();
            adjoint_.template bottomRows<2>().setZero();

            detail::solveWithFactors<kDerivativesPerKnot>(factors_, num_segments_, adjoint_);
        }

        /** @brief Add the interpolation constraint's contribution using endpoint identities.
         * @param[in,out] gradient Physical point, time and clamped-boundary gradients.
         * @note The adjoint polynomial has zero positions and knot derivatives from adjoint_.
         *       Integration by parts gives point/boundary forces; the polarized Hamiltonian
         *       gives time corrections without constructing dA/dT or differentiating the solver. */
        void accumulateAdjoint(Gradients &gradient) const
        {
            for (int i = 0; i < num_segments_; ++i)
            {
                const auto &tp = time_powers_[i];
                const int coefficient_row = i * kCoefficientCount;
                const int left_row = i * kDerivativesPerKnot;
                const int right_row = left_row + kDerivativesPerKnot;

                // Map knot adjoints to a polynomial with zero endpoint positions.
                const RowVector adjoint_v_left = adjoint_.row(left_row);
                const RowVector adjoint_v_right = adjoint_.row(right_row);
                const RowVector adjoint_a_left = adjoint_.row(left_row + 1);
                const RowVector adjoint_a_right = adjoint_.row(right_row + 1);
                const RowVector lambda_c3 = (-6.0 * tp.h2_inv) * adjoint_v_left
                    - (4.0 * tp.h2_inv) * adjoint_v_right
                    - (1.5 * tp.h_inv) * adjoint_a_left
                    + (0.5 * tp.h_inv) * adjoint_a_right;
                const RowVector lambda_c4 = (8.0 * tp.h3_inv) * adjoint_v_left
                    + (7.0 * tp.h3_inv) * adjoint_v_right
                    + (1.5 * tp.h2_inv) * adjoint_a_left
                    - tp.h2_inv * adjoint_a_right;
                const RowVector lambda_c5 = (-3.0 * tp.h4_inv) * adjoint_v_left
                    - (3.0 * tp.h4_inv) * adjoint_v_right
                    - (0.5 * tp.h3_inv) * adjoint_a_left
                    + (0.5 * tp.h3_inv) * adjoint_a_right;

                // Endpoint forces use physical derivatives of the adjoint polynomial.
                const RowVector adjoint_jerk = 6.0 * lambda_c3;
                const RowVector adjoint_snap = 24.0 * lambda_c4;
                const RowVector adjoint_crackle = 120.0 * lambda_c5;
                gradient.addWaypoint(i, adjoint_crackle);
                gradient.addWaypoint(i + 1, -adjoint_crackle);

                // The polarized Hamiltonian contracts matching physical derivatives.
                const RowVector velocity = trajectory_.coefficients_.row(coefficient_row + 1);
                const RowVector acceleration = 2.0 * trajectory_.coefficients_.row(coefficient_row + 2);
                const RowVector jerk = 6.0 * trajectory_.coefficients_.row(coefficient_row + 3);
                const RowVector snap = 24.0 * trajectory_.coefficients_.row(coefficient_row + 4);
                const RowVector crackle = 120.0 * trajectory_.coefficients_.row(coefficient_row + 5);
                const RowVector time_partials = jerk.cwiseProduct(adjoint_jerk)
                    + adjoint_v_left.cwiseProduct(crackle) + velocity.cwiseProduct(adjoint_crackle)
                    - adjoint_a_left.cwiseProduct(snap) - acceleration.cwiseProduct(adjoint_snap);
                gradient.durations(i) += time_partials.sum();

                // Only clamped endpoints contribute boundary derivative corrections.
                if (i == 0)
                {
                    gradient.start.v -= adjoint_snap.transpose();
                    gradient.start.a += adjoint_jerk.transpose();
                }
                if (i + 1 == num_segments_)
                {
                    const RowVector end_v = tp.h * adjoint_crackle + adjoint_snap;
                    const RowVector end_a = ((60.0 * tp.h) * lambda_c5 + adjoint_snap) * tp.h + adjoint_jerk;
                    gradient.end.v += end_v.transpose();
                    gradient.end.a -= end_a.transpose();
                }
            }
        }
    };

    // Septic splines: minimum snap.

    /** @brief Minimum-snap interpolation with a specialized O(N) 3-by-3 block tridiagonal solver.
     * @note Minimize the squared fourth physical derivative integral with fixed waypoint
     *       positions and endpoint derivatives of orders 1 through 3. Coordinates
     *       are caller-defined; durations use seconds. Polynomial coefficients and knots
     *       have one owner. Updates and backward() require exclusive access; const energy
     *       and polynomial queries may run concurrently on an unchanged valid object.
     *       Copies own independent data. Moving invalidates borrowed views and the source polynomial.
     *       Updates return status and invalidate the polynomial on numerical failure; factories return StatusOr.
     *       Inputs are trusted: finite waypoints/boundaries/start, positive finite durations or strictly
     *       increasing finite knots, N >= 1, N+1 waypoint rows, and N*kCoefficientCount within int range.
     *       Energy and adjoint queries require successful construction and return plain values or void.
     *       Reference gradient outputs must be pre-sized; resetTopology(N) prepares full gradients.
     *       Adjoint inputs must be finite with N*kCoefficientCount coefficient rows and N time partials.
     *       All public overloads use this single contract; callers perform input validation and
     *       guarantee a representable working range. Coefficients, energy and gradients are not
     *       rescanned for overflow; construction still reports knot or factorization failure. */
    template <int DIM>
    class SepticSplineND
    {
    public:
        using Vector = Eigen::Matrix<double, DIM, 1>;
        using RowVector = Eigen::Matrix<double, 1, DIM>;
        static constexpr int kMatrixOptions = (DIM == 1) ? Eigen::ColMajor : Eigen::RowMajor;
        using CoefficientMatrix = Eigen::Matrix<double, Eigen::Dynamic, DIM, kMatrixOptions>;

        static constexpr int kDimension = DIM;
        static constexpr int kDegree = 7;
        static constexpr int kCoefficientCount = 8;
        static constexpr int kMinDerivativeOrder = 4;
        using Polynomial = PPolyND<DIM, kCoefficientCount>;

        using BoundaryGradient = SplineTrajectory::BoundaryGradient<DIM>;
        using BoundaryGradients = BoundaryGradientPair<DIM>;
        using Gradients = SplineGradients<DIM>;

    private:
        std::vector<double> time_segments_;
        double start_time_{0.0};

        CoefficientMatrix spatial_points_;

        BoundaryConditions<DIM> boundary_;

        int num_segments_{0};

        Polynomial trajectory_;

        // Knot-major rows: each knot's physical derivatives are adjacent in memory.
        static constexpr int kDerivativesPerKnot = 3;
        using Factor = detail::BlockLDLT3;
        using Block = typename Factor::Block;
        std::vector<Factor> factors_;
        CoefficientMatrix derivatives_;
        CoefficientMatrix adjoint_;

        struct TimePowers
        {
            double h;
            double h_inv;
            double h2_inv;
            double h3_inv;
            double h4_inv;
            double h5_inv;
            double h6_inv;
            double h7_inv;
        };
        std::vector<TimePowers> time_powers_;

    public:
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW

        /** @brief Construct an empty, invalid spline with independent storage. */
        SepticSplineND() = default;

        /** @brief Construct from N+1 strictly increasing finite knots and N+1 waypoint rows.
         * @param t_points Absolute times in seconds. @param spatial_points Physical waypoint rows.
         * @param boundary Clamped physical endpoint derivatives in the caller's frame.
         * @note Inputs are copied; failure is returned without publishing an invalid value.
         * @return Success, or a state/resource/numerical failure; input validity is a precondition. */
        static StatusOr<SepticSplineND> create(const std::vector<double> &t_points, const CoefficientMatrix &spatial_points,
                                    const BoundaryConditions<DIM> &boundary = {})
        {
            SepticSplineND result;
            if (const auto status = result.update(t_points, spatial_points, boundary); !status) return status;
            return result;
        }

        /** @brief Construct from N positive durations, N+1 waypoint rows and a common time origin.
         * @param time_segments Durations in seconds. @param spatial_points Physical waypoint rows.
         * @param start_time Absolute time origin in seconds.
         * @param boundary Clamped physical endpoint derivatives in the caller's frame.
         * @note Inputs are copied; failure is returned without publishing an invalid value.
         * @return Success, or a state/resource/numerical failure; input validity is a precondition. */
        static StatusOr<SepticSplineND> create(const std::vector<double> &time_segments, const CoefficientMatrix &spatial_points,
                                    double start_time, const BoundaryConditions<DIM> &boundary = {})
        {
            SepticSplineND result;
            if (const auto status = result.update(time_segments, spatial_points, start_time, boundary); !status) return status;
            return result;
        }

        /** @brief Replace interpolation data using caller-validated absolute knots.
         * @param t_points Absolute times in seconds. @param spatial_points Physical waypoint rows.
         * @param boundary Clamped physical endpoint derivatives in the caller's frame.
         * @pre Inputs satisfy the class contract; no input validation is performed.
         * @note Copies inputs, reuses fixed-topology storage and invalidates all borrowed views.
         * @return Success, or a state/resource/numerical failure; input validity is a precondition. */
        Status update(const std::vector<double> &t_points,
                    const CoefficientMatrix &spatial_points,
                    const BoundaryConditions<DIM> &boundary = BoundaryConditions<DIM>())
        {
            detail::durationsFromKnots(t_points, time_segments_, start_time_);
            return update(time_segments_, spatial_points, start_time_, boundary);
        }

        /** @brief Replace interpolation data using positive durations and a common time origin.
         * @param time_segments Durations in seconds. @param spatial_points Physical waypoint rows.
         * @param start_time Absolute time origin in seconds.
         * @param boundary Clamped physical endpoint derivatives in the caller's frame.
         * @pre Inputs satisfy the class contract; no input validation is performed.
         * @note Copies inputs, reuses fixed-topology storage and invalidates all borrowed views.
         *       Numeric construction failures clear the polynomial; inputs are not validated here.
         * @return Success, or a state/resource/numerical failure; input validity is a precondition. */
        Status update(const std::vector<double> &time_segments,
                    const CoefficientMatrix &spatial_points,
                    double start_time,
                    const BoundaryConditions<DIM> &boundary = BoundaryConditions<DIM>())
        {
            time_segments_ = time_segments;
            spatial_points_ = spatial_points;
            start_time_ = start_time;
            boundary_ = boundary;
            return updateSplineInternal();
        }

        /** @brief Inspect the published polynomial. @return True after successful construction; numerical range is a caller precondition. */
        bool isValid() const { return trajectory_.isValid(); }
        /** @brief Inspect spatial dimension. @return Compile-time DIM. */
        int dimension() const { return DIM; }

        /** @brief Inspect the absolute start time. @return Seconds, or zero for an invalid spline. */
        double startTime() const
        {
            return trajectory_.startTime();
        }

        /** @brief Inspect the absolute end time. @return Seconds, or zero for an invalid spline. */
        double endTime() const
        {
            return trajectory_.endTime();
        }

        /** @brief Inspect total duration. @return Seconds, or zero for an invalid spline. */
        double duration() const
        {
            return trajectory_.duration();
        }

        /** @brief Inspect input topology. @return Number of stored waypoints. */
        size_t numPoints() const
        {
            return static_cast<size_t>(spatial_points_.rows());
        }

        /** @brief Inspect the polynomial topology. @return Segment count, or zero when invalid. */
        int numSegments() const
        {
            return trajectory_.numSegments();
        }

        /** @brief Borrow physical waypoints. @return Storage valid until update, assignment or move. */
        const CoefficientMatrix &waypoints() const { return spatial_points_; }
        /** @brief Borrow segment durations in seconds. @return Storage valid until the next mutation. */
        const std::vector<double> &durations() const { return time_segments_; }
        /** @brief Borrow absolute knots in seconds. @return Storage valid until the next mutation. */
        const std::vector<double> &breakpoints() const { return trajectory_.breakpoints_; }
        /** @brief Borrow physical boundary data. @return Storage valid until the next mutation. */
        const BoundaryConditions<DIM> &boundary() const { return boundary_; }

        /** @brief Borrow the polynomial. @return View valid until the next update, assignment or move. */
        const Polynomial &polynomial() const & { return trajectory_; }
        /** @brief Prevent polynomial borrowing from a temporary spline. @return No callable rvalue overload. */
        const Polynomial &polynomial() const && = delete;
        /** @brief Make an independent owning snapshot. @return Polynomial with its own storage and lifetime. */
        Polynomial copyPolynomial() const { return trajectory_; }

        /** @brief Evaluate monomial bases in local physical time, through crackle.
         * @param t Local time in seconds.
         * @param[out] b_pos Position basis. @param[out] b_vel Velocity basis.
         * @param[out] b_acc Acceleration basis. @param[out] b_jerk Jerk basis.
         * @param[out] b_snap Snap basis. @param[out] b_crackle Crackle basis.
         * @note Derivatives above this polynomial's degree are zero. No shared state is touched. */
        static inline void computeBasisFunctions(double t,
                                                 Eigen::Matrix<double, 1, kCoefficientCount> &b_pos,
                                                 Eigen::Matrix<double, 1, kCoefficientCount> &b_vel,
                                                 Eigen::Matrix<double, 1, kCoefficientCount> &b_acc,
                                                 Eigen::Matrix<double, 1, kCoefficientCount> &b_jerk,
                                                 Eigen::Matrix<double, 1, kCoefficientCount> &b_snap,
                                                 Eigen::Matrix<double, 1, kCoefficientCount> &b_crackle)
        {
            double t1 = t, t2 = t1 * t1, t3 = t2 * t1, t4 = t3 * t1, t5 = t4 * t1, t6 = t5 * t1, t7 = t6 * t1;

            b_pos << 1, t1, t2, t3, t4, t5, t6, t7;
            b_vel << 0, 1, 2 * t1, 3 * t2, 4 * t3, 5 * t4, 6 * t5, 7 * t6;
            b_acc << 0, 0, 2, 6 * t1, 12 * t2, 20 * t3, 30 * t4, 42 * t5;
            b_jerk << 0, 0, 0, 6, 24 * t1, 60 * t2, 120 * t3, 210 * t4;
            b_snap << 0, 0, 0, 0, 24, 120 * t1, 360 * t2, 840 * t3;
            b_crackle << 0, 0, 0, 0, 0, 120, 720 * t1, 2520 * t2;
        }

        /** @brief Integrate the squared minimum physical derivative.
         * @return Energy in physical units; requires a successfully constructed spline. */
        double energy() const
        {
            assert(trajectory_.isValid());

            double total_energy = 0.0;
            for (int i = 0; i < num_segments_; ++i)
            {
                const double T = time_segments_[i];

                const double T2 = T * T;
                const double T3 = T2 * T;
                const double T4 = T3 * T;
                const double T5 = T4 * T;
                const double T6 = T4 * T2;
                const double T7 = T4 * T3;

                RowVector c4 = trajectory_.coefficients_.row(i * 8 + 4);
                RowVector c5 = trajectory_.coefficients_.row(i * 8 + 5);
                RowVector c6 = trajectory_.coefficients_.row(i * 8 + 6);
                RowVector c7 = trajectory_.coefficients_.row(i * 8 + 7);

                total_energy += 576.0 * c4.squaredNorm() * T +
                                2880.0 * c4.dot(c5) * T2 +
                                4800.0 * c5.squaredNorm() * T3 +
                                5760.0 * c4.dot(c6) * T3 +
                                21600.0 * c5.dot(c6) * T4 +
                                10080.0 * c4.dot(c7) * T4 +
                                25920.0 * c6.squaredNorm() * T5 +
                                40320.0 * c5.dot(c7) * T5 +
                                100800.0 * c6.dot(c7) * T6 +
                                100800.0 * c7.squaredNorm() * T7;
            }

            return total_energy;
        }

        /**
         * @brief Compute partial gradient of energy w.r.t. polynomial coefficients.
         *
         * @param[out] gdC Overwritten partial gradient ∂E/∂C, size (num_segments * 8) × DIM.
         *         Rows follow the same piece-major ascending-power order as coefficients(). Durations are held fixed.
         * @note This computes only the direct partial derivative, not the full gradient.
         * @pre The spline is valid and reference outputs already have the required shapes. */
        void energyCoefficientPartials(CoefficientMatrix &gdC) const
        {
            assert(trajectory_.isValid());
            gdC.setZero();

            for (int i = 0; i < num_segments_; ++i)
            {
                double T = time_segments_[i];
                double T2 = T * T;
                double T3 = T2 * T;
                double T4 = T3 * T;
                double T5 = T4 * T;
                double T6 = T5 * T;
                double T7 = T6 * T;

                const RowVector c4 = trajectory_.coefficients_.row(i * 8 + 4);
                const RowVector c5 = trajectory_.coefficients_.row(i * 8 + 5);
                const RowVector c6 = trajectory_.coefficients_.row(i * 8 + 6);
                const RowVector c7 = trajectory_.coefficients_.row(i * 8 + 7);

                gdC.row(i * 8 + 4) = 1152.0 * c4 * T +
                                     2880.0 * c5 * T2 +
                                     5760.0 * c6 * T3 +
                                     10080.0 * c7 * T4;

                gdC.row(i * 8 + 5) = 2880.0 * c4 * T2 +
                                     9600.0 * c5 * T3 +
                                     21600.0 * c6 * T4 +
                                     40320.0 * c7 * T5;

                gdC.row(i * 8 + 6) = 5760.0 * c4 * T3 +
                                     21600.0 * c5 * T4 +
                                     51840.0 * c6 * T5 +
                                     100800.0 * c7 * T6;

                gdC.row(i * 8 + 7) = 10080.0 * c4 * T4 +
                                     40320.0 * c5 * T5 +
                                     100800.0 * c6 * T6 +
                                     201600.0 * c7 * T7;
            }
        }

        /** @brief Allocate direct energy coefficient partials. @return Piece-major physical coefficient gradients. */
        CoefficientMatrix energyCoefficientPartials() const
        {
            CoefficientMatrix gdC(num_segments_ * kCoefficientCount, DIM);
            energyCoefficientPartials(gdC);
            return gdC;
        }

        /**
         * @brief Compute partial gradient of energy w.r.t. segment durations.
         *
         * @param[out] gdT Overwritten partial gradient ∂E/∂T, size num_segments.
         *         Element i is the direct partial derivative w.r.t. T[i].
         * @note Local physical-time coefficients are held fixed; spline interpolation constraints are not differentiated.
         * @pre The spline is valid and reference outputs already have the required shapes. */
        void energyDurationPartials(Eigen::VectorXd &gdT) const
        {
            assert(trajectory_.isValid());

            for (int i = 0; i < num_segments_; ++i)
            {
                const double T = time_segments_[i];

                const RowVector c4 = trajectory_.coefficients_.row(i * 8 + 4);
                const RowVector c5 = trajectory_.coefficients_.row(i * 8 + 5);
                const RowVector c6 = trajectory_.coefficients_.row(i * 8 + 6);
                const RowVector c7 = trajectory_.coefficients_.row(i * 8 + 7);

                RowVector snap_end = 24.0 * c4 + T * (120.0 * c5 + T * (360.0 * c6 + (840.0 * T) * c7));

                gdT(i) = snap_end.squaredNorm();
            }
        }

        /** @brief Allocate direct energy duration partials. @return Physical duration gradients at fixed coefficients. */
        Eigen::VectorXd energyDurationPartials() const
        {
            Eigen::VectorXd gdT(num_segments_);
            energyDurationPartials(gdT);
            return gdT;
        }

        /**
         * @brief Compute full gradient of energy w.r.t. segment durations.
         *
         * @param[out] grad Full gradient dE/dT, size num_segments.
         *         Includes both direct and indirect dependencies via chain rule.
         * @pre The spline is valid and reference outputs already have the required shapes. */
        void energyDurationGradient(Eigen::VectorXd &grad) const
        {
            assert(trajectory_.isValid());
            computeEnergyDurationGradient<false>(grad, 1.0);
        }

        /** @brief Allocate total energy duration gradients. @return Gradients including interpolation dependence. */
        Eigen::VectorXd energyDurationGradient() const
        {
            Eigen::VectorXd grad(num_segments_);
            energyDurationGradient(grad);
            return grad;
        }

        /**
         * @brief Compute full gradient of energy w.r.t. inner waypoints only.
         * @param[out] grad Full gradient dE/dP for inner points only.
         *         Size (N-1) × DIM (excludes boundary points P0 and PN).
         * @pre The spline is valid and reference outputs already have the required shapes. */
        void energyWaypointGradient(CoefficientMatrix &grad) const
        {
            assert(trajectory_.isValid());
            detail::computeEnergyWaypointGradient<10080>(trajectory_, grad, 1.0);
        }

        /** @brief Allocate total energy waypoint gradients. @return Interior physical position gradients. */
        CoefficientMatrix energyWaypointGradient() const
        {
            CoefficientMatrix grad(num_segments_ - 1, DIM);
            energyWaypointGradient(grad);
            return grad;
        }

        /**
         * @brief Compute energy gradients w.r.t. boundary position, velocity, acceleration and jerk.
         * @param[out] grad Overwritten start/end gradients in the caller's coordinates and
         *        physical-time derivative convention (seconds).
         * @pre The spline is valid and reference outputs already have the required shapes.
         * @note Valid calls do not allocate. Concurrent calls require separate outputs and no spline updates.
         */
        void energyBoundaryGradient(BoundaryGradients &grad) const
        {
            assert(trajectory_.isValid());
            computeEnergyBoundaryGradient<false>(grad.start, grad.end, 1.0);
        }

        /** @brief Return energy gradients w.r.t. boundary position, velocity, acceleration and jerk.
         * @return Owned start/end gradients using physical seconds and the caller's coordinates.
         * @pre The spline is valid.
         * @note Valid calls do not allocate. Concurrent calls require no spline updates. */
        BoundaryGradients energyBoundaryGradient() const
        {
            BoundaryGradients res;
            energyBoundaryGradient(res);
            return res;
        }

        /**
         * @brief Compute full energy gradient combining time, inner point, and boundary gradients.
         *
         * @return Gradients Structure containing:
         *       - inner_points: gradient w.r.t. inner waypoints, size (N-1) × DIM
         *       - durations: gradient w.r.t. physical segment durations, size N
         *       - start/end: gradients w.r.t. boundary states (position, velocity, acceleration, jerk)
         */
        Gradients energyGradient() const
        {
            Gradients res;
            res.resetTopology(num_segments_);
            energyGradient(res);
            return res;
        }

        /**
         * @brief Compute full energy gradient combining time, inner point, and boundary gradients (reference overload).
         *
         * @param[out] grads Overwritten physical gradients; buffers must already match the spline topology.
         * @note Computes waypoint jumps, endpoint terms and duration contributions using the current spline.
         * @pre The spline is valid and reference outputs already have the required shapes. */
        void energyGradient(Gradients &grads) const
        {
            assert(trajectory_.isValid());
            detail::computeEnergyWaypointGradient<10080>(trajectory_, grads.inner_points, 1.0);
            computeEnergyDurationGradient<false>(grads.durations, 1.0);
            computeEnergyBoundaryGradient<false>(grads.start, grads.end, 1.0);
        }

        /** @brief Add weighted minimum-derivative energy gradients to existing physical partials.
         * @param[in,out] grads Prepared, initialized gradients matching this spline's topology.
         * @param weight Energy multiplier; uses the same physical units as energy().
         * @pre The latest construction succeeded and arithmetic is representable.
         * @note No allocation or resizing; requires exclusive output access and no spline mutation. */
        void energyGradientAdd(Gradients &grads, double weight = 1.0) const
        {
            assert(trajectory_.isValid());
            detail::computeEnergyWaypointGradient<10080, true>(trajectory_, grads.inner_points, weight);
            computeEnergyDurationGradient<true>(grads.durations, weight);
            computeEnergyBoundaryGradient<true>(grads.start, grads.end, weight);
        }

        /**
         * @brief Propagate gradients from coefficients/times to waypoints and boundaries.
         *
         * @param partialGradByCoeffs Input partial gradient ∂L/∂C, size (8N) × DIM.
         * @param partialGradByTimes  Input partial gradient ∂L/∂T, size N.
         * @return Gradients Structure containing:
         *       - inner_points: gradient w.r.t. inner waypoints, size (N-1) × DIM
         *       - durations: gradient w.r.t. physical segment durations, size N
         *       - start/end: gradients w.r.t. boundary states (position, velocity, acceleration, jerk)
         */
        Gradients backward(const CoefficientMatrix &partialGradByCoeffs,
                           const Eigen::VectorXd &partialGradByTimes)
        {
            Gradients res;
            res.resetTopology(num_segments_);

            backward(partialGradByCoeffs, partialGradByTimes, res);

            return res;
        }

        /**
         * @brief Propagate gradients from coefficients/times to waypoints and boundaries (reference overload).
         *
         * @param partialGradByCoeffs Input partial gradient ∂L/∂C, size (8N) × DIM.
         * @param partialGradByTimes  Input partial gradient ∂L/∂T, size N.
         * @param[out] grads Overwritten physical gradients; buffers must already match the spline topology.
         * @pre Input partials are finite and match the constructed spline; no input scan is performed.
         * @note Pulls back the Hermite map, reuses the cached knot solve, then adds endpoint corrections.
         * @pre The spline is valid and reference outputs already have the required shapes. */
        void backward(const CoefficientMatrix &partialGradByCoeffs,
                      const Eigen::VectorXd &partialGradByTimes,
                      Gradients &grads)
        {
            assert(trajectory_.isValid());
            detail::initializeGradients(partialGradByTimes, grads);
            adjoint_.setZero();

            // Local map, cached linear solve, then endpoint corrections.
            pullbackCoefficients(partialGradByCoeffs, grads);
            solveAdjoint(grads);
            // A fully clamped segment has no free knot derivatives to correct.
            if (num_segments_ > 1)
                accumulateAdjoint(grads);
        }

    private:
        /** @brief Write duration terms from the endpoint energy identity.
         * @tparam Add Select accumulation or overwrite at compile time.
         * @param[in,out] grad Pre-sized duration partials in physical seconds.
         * @param weight Energy multiplier. @pre The spline is valid; arithmetic is representable. */
        template<bool Add>
        void computeEnergyDurationGradient(Eigen::VectorXd &grad, double weight) const
        {
            for (int i = 0; i < num_segments_; ++i)
            {

                const int offset = i * 8;
                const RowVector c1 = trajectory_.coefficients_.row(offset + 1);
                const RowVector c2 = trajectory_.coefficients_.row(offset + 2);
                const RowVector c3 = trajectory_.coefficients_.row(offset + 3);
                const RowVector c4 = trajectory_.coefficients_.row(offset + 4);
                const RowVector c5 = trajectory_.coefficients_.row(offset + 5);
                const RowVector c6 = trajectory_.coefficients_.row(offset + 6);
                const RowVector c7 = trajectory_.coefficients_.row(offset + 7);

                double term_snap = 576.0 * c4.squaredNorm();
                double term_cj = 1440.0 * c3.dot(c5);
                double term_pa = 2880.0 * c2.dot(c6);
                double term_dv = 10080.0 * c1.dot(c7);

                detail::storeGradient<Add>(grad(i), weight * (-term_snap + term_cj - term_pa + term_dv));
            }
        }

        /** @brief Write integration-by-parts terms at the two clamped endpoints.
         * @tparam Add Select accumulation or overwrite at compile time.
         * @param[in,out] start Start endpoint partials in physical coordinates.
         * @param[in,out] end End endpoint partials in physical coordinates.
         * @param weight Energy multiplier. @pre The spline is valid; outputs are exclusive. */
        template<bool Add>
        void computeEnergyBoundaryGradient(BoundaryGradient &start, BoundaryGradient &end, double weight) const
        {
            // Evaluate the start-boundary terms at local time zero.
            const RowVector c4_first = trajectory_.coefficients_.row(4);
            const RowVector c5_first = trajectory_.coefficients_.row(5);
            const RowVector c6_first = trajectory_.coefficients_.row(6);
            const RowVector c7_first = trajectory_.coefficients_.row(7);

            detail::storeGradient<Add>(start.p, weight * (10080.0 * c7_first.transpose()));
            detail::storeGradient<Add>(start.v, weight * (-1440.0 * c6_first.transpose()));
            detail::storeGradient<Add>(start.a, weight * (240.0 * c5_first.transpose()));
            detail::storeGradient<Add>(start.j, weight * (-48.0 * c4_first.transpose()));

            // Evaluate the end-boundary terms at the final segment's physical duration.
            int last_idx = num_segments_ - 1;
            double T = time_segments_[last_idx];
            double T2 = T * T;
            double T3 = T2 * T;

            const RowVector c4_last = trajectory_.coefficients_.row(last_idx * 8 + 4);
            const RowVector c5_last = trajectory_.coefficients_.row(last_idx * 8 + 5);
            const RowVector c6_last = trajectory_.coefficients_.row(last_idx * 8 + 6);
            const RowVector c7_last = trajectory_.coefficients_.row(last_idx * 8 + 7);

            RowVector snap_end = 24.0 * c4_last + 120.0 * c5_last * T +
                                     360.0 * c6_last * T2 + 840.0 * c7_last * T3;
            RowVector crackle_end = 120.0 * c5_last + 720.0 * c6_last * T + 2520.0 * c7_last * T2;
            RowVector pop_end = 720.0 * c6_last + 5040.0 * c7_last * T;
            RowVector d7_end = 5040.0 * c7_last;

            detail::storeGradient<Add>(end.p, weight * (-2.0 * d7_end.transpose()));
            detail::storeGradient<Add>(end.v, weight * (2.0 * pop_end.transpose()));
            detail::storeGradient<Add>(end.a, weight * (-2.0 * crackle_end.transpose()));
            detail::storeGradient<Add>(end.j, weight * (2.0 * snap_end.transpose()));
        }

        // Construction: prepare caches, solve knot derivatives, publish coefficients.
        /** @brief Rebuild from physical data; failed construction clears the published polynomial.
         * @note Requires exclusive access. Storage is reused when the topology is unchanged.
         * @return Success, or a state/resource/numerical failure; input validity is a precondition. */
        Status updateSplineInternal()
        {
            trajectory_.is_initialized_ = false;
            const auto times = trajectory_.prepareCoefficients(time_segments_, start_time_, kCoefficientCount);
            if (!times) { num_segments_ = 0; return times; }
            num_segments_ = static_cast<int>(time_segments_.size());

            prepareWorkspace();
            precomputeTimePowers();
            if (!solveInternalDerivatives())
            {
                trajectory_.clear();
                return {ErrorCode::NumericalFailure, "Spline factorization failed"};
            }
            writeCoefficients();
            trajectory_.publishCoefficients();
            return Status::success();
        }

        /** @brief Reuse topology-dependent storage and refresh the clamped endpoint derivatives.
         * @note Requires validated input. Same-topology updates skip all cache resizing;
         *       the first adjoint reuses storage prepared during construction. */
        void prepareWorkspace()
        {
            if (time_powers_.size() != time_segments_.size())
                resizeWorkspace();

            // Boundary values change independently of the segment count.
            derivatives_.row(0) = boundary_.start_velocity.transpose();
            derivatives_.row(num_segments_ * 3 + 0) = boundary_.end_velocity.transpose();
            derivatives_.row(1) = boundary_.start_acceleration.transpose();
            derivatives_.row(num_segments_ * 3 + 1) = boundary_.end_acceleration.transpose();
            derivatives_.row(2) = boundary_.start_jerk.transpose();
            derivatives_.row(num_segments_ * 3 + 2) = boundary_.end_jerk.transpose();
        }

        /** @brief Resize primal, adjoint and factor caches after a segment-count change.
         * @note Duration-cache size records the prepared topology and is committed last.
         *       Allocation order is preserved; same-topology updates skip this function. */
        void resizeWorkspace()
        {
            adjoint_.resize((num_segments_ + 1) * kDerivativesPerKnot, DIM);
            derivatives_.resize((num_segments_ + 1) * kDerivativesPerKnot, DIM);
            factors_.resize(num_segments_ - 1);
            time_powers_.resize(num_segments_);
        }

        /** @brief Cache reciprocal duration powers used by assembly, Hermite maps and adjoints. */
        void precomputeTimePowers()
        {
            for (int i = 0; i < num_segments_; ++i)
            {
                auto &tp = time_powers_[i];
                tp.h = time_segments_[i];
                tp.h_inv = 1.0 / tp.h;
                tp.h2_inv = tp.h_inv * tp.h_inv;
                tp.h3_inv = tp.h2_inv * tp.h_inv;
                tp.h4_inv = tp.h3_inv * tp.h_inv;
                tp.h5_inv = tp.h4_inv * tp.h_inv;
                tp.h6_inv = tp.h5_inv * tp.h_inv;
                tp.h7_inv = tp.h6_inv * tp.h_inv;
            }
        }

        /** @brief Assemble one knot's stationarity equation for half the integrated squared derivative.
         * @param knot Interior knot index in [1, num_segments_).
         * @param[out] diagonal Symmetric diagonal block of the physical energy Hessian.
         * @param[out] previous Coupling to the preceding knot.
         * @note Writes this knot's RHS to derivatives_; fixed endpoint terms move to that RHS. */
        void assembleKnotSystem(int knot, Block &diagonal, Block &previous)
        {
            const auto &left = time_powers_[knot - 1];
            const auto &right = time_powers_[knot];

            // These constants are the exact unit-interval Hermite energy Gram matrix.
            diagonal(0, 0) = 25920.0 * left.h5_inv + 25920.0 * right.h5_inv;
            diagonal(1, 0) = -5400.0 * left.h4_inv + 5400.0 * right.h4_inv;
            diagonal(1, 1) = 1200.0 * left.h3_inv + 1200.0 * right.h3_inv;
            diagonal(2, 0) = 480.0 * left.h3_inv + 480.0 * right.h3_inv;
            diagonal(2, 1) = -120.0 * left.h2_inv + 120.0 * right.h2_inv;
            diagonal(2, 2) = 16.0 * left.h_inv + 16.0 * right.h_inv;
            previous(0, 0) = 24480.0 * left.h5_inv;
            previous(0, 1) = 4680.0 * left.h4_inv;
            previous(0, 2) = 360.0 * left.h3_inv;
            previous(1, 0) = -4680.0 * left.h4_inv;
            previous(1, 1) = -840.0 * left.h3_inv;
            previous(1, 2) = -60.0 * left.h2_inv;
            previous(2, 0) = 360.0 * left.h3_inv;
            previous(2, 1) = 60.0 * left.h2_inv;
            previous(2, 2) = 4.0 * left.h_inv;

            const RowVector delta_left = spatial_points_.row(knot) - spatial_points_.row(knot - 1);
            const RowVector delta_right = spatial_points_.row(knot + 1) - spatial_points_.row(knot);
            RowVector rhs0 = 50400.0 * left.h6_inv * delta_left + 50400.0 * right.h6_inv * delta_right;
            RowVector rhs1 = -10080.0 * left.h5_inv * delta_left + 10080.0 * right.h5_inv * delta_right;
            RowVector rhs2 = 840.0 * left.h4_inv * delta_left + 840.0 * right.h4_inv * delta_right;
            if (knot == 1)
            {
                rhs0 -= previous(0, 0) * derivatives_.row(0)
                    + previous(0, 1) * derivatives_.row(1)
                    + previous(0, 2) * derivatives_.row(2);
                rhs1 -= previous(1, 0) * derivatives_.row(0)
                    + previous(1, 1) * derivatives_.row(1)
                    + previous(1, 2) * derivatives_.row(2);
                rhs2 -= previous(2, 0) * derivatives_.row(0)
                    + previous(2, 1) * derivatives_.row(1)
                    + previous(2, 2) * derivatives_.row(2);
            }
            if (knot + 1 == num_segments_)
            {
                rhs0 -= 24480.0 * right.h5_inv * derivatives_.row(num_segments_ * 3)
                    - 4680.0 * right.h4_inv * derivatives_.row(num_segments_ * 3 + 1)
                    + 360.0 * right.h3_inv * derivatives_.row(num_segments_ * 3 + 2);
                rhs1 -= 4680.0 * right.h4_inv * derivatives_.row(num_segments_ * 3)
                    - 840.0 * right.h3_inv * derivatives_.row(num_segments_ * 3 + 1)
                    + 60.0 * right.h2_inv * derivatives_.row(num_segments_ * 3 + 2);
                rhs2 -= 360.0 * right.h3_inv * derivatives_.row(num_segments_ * 3)
                    - 60.0 * right.h2_inv * derivatives_.row(num_segments_ * 3 + 1)
                    + 4.0 * right.h_inv * derivatives_.row(num_segments_ * 3 + 2);
            }
            derivatives_.row(knot * 3) = rhs0;
            derivatives_.row(knot * 3 + 1) = rhs1;
            derivatives_.row(knot * 3 + 2) = rhs2;
        }

        /** @brief Factor and solve the block system in linear time.
         * @return False on a nonpositive/nonfinite pivot; true after solving all free derivatives.
         * @note Forward substitution is fused with assembly to avoid a separate RHS sweep. */
        bool solveInternalDerivatives()
        {
            for (int knot = 1; knot < num_segments_; ++knot)
            {
                Block diagonal, previous;
                assembleKnotSystem(knot, diagonal, previous);
                auto &factor = factors_[knot - 1];
                const Factor *prior = knot > 1 ? &factors_[knot - 2] : nullptr;
                if (!factor.factorize(diagonal, previous, prior))
                    return false;
                factor.forward(derivatives_, knot * kDerivativesPerKnot);
            }
            backSubstitute(derivatives_);
            return true;
        }

        /** @brief Finish a primal or adjoint solve using the stored diagonal/coupling factors.
         * @param[in,out] values Knot-major forward-substituted RHS; receives the solution.
         * @note Endpoint rows are preserved; the caller owns and exclusively supplies the matrix. */
        void backSubstitute(CoefficientMatrix &values) const
        {
            detail::backSubstitute<kDerivativesPerKnot>(factors_, num_segments_, values);
        }

        /** @brief Generate physical-time coefficients from normalized endpoint residuals.
         * @note Each row contains contiguous spatial coordinates. The four residuals
         *       remove the left-end Taylor polynomial on the unit interval before the
         *       coefficients are converted back to physical time. */
        void writeCoefficients()
        {
            for (int i = 0; i < num_segments_; ++i)
            {
                const auto &tp = time_powers_[i];
                const int coefficient_row = i * kCoefficientCount;
                const int left_row = i * kDerivativesPerKnot;
                const int right_row = left_row + kDerivativesPerKnot;
                const RowVector position = spatial_points_.row(i);

                const RowVector velocity = derivatives_.row(left_row);
                const RowVector acceleration = derivatives_.row(left_row + 1);
                const RowVector jerk = derivatives_.row(left_row + 2);
                const RowVector end_velocity = derivatives_.row(right_row);
                const RowVector end_acceleration = derivatives_.row(right_row + 1);
                const RowVector end_jerk = derivatives_.row(right_row + 2);
                const RowVector quadratic = 0.5 * acceleration;
                const RowVector cubic = (1.0 / 6.0) * jerk;
                const double duration2 = tp.h * tp.h;
                const double duration3 = duration2 * tp.h;

                // Scale endpoint residuals to the unit interval; shared powers stay outside the map.
                const RowVector displacement = spatial_points_.row(i + 1) - position;
                const RowVector scaled_acceleration = duration2 * acceleration;
                const RowVector scaled_jerk = duration3 * jerk;
                const RowVector position_residual = displacement - tp.h * velocity -
                    0.5 * scaled_acceleration - (1.0 / 6.0) * scaled_jerk;
                const RowVector velocity_residual = tp.h * (end_velocity - velocity) -
                    scaled_acceleration - 0.5 * scaled_jerk;
                const RowVector acceleration_residual = duration2 * (end_acceleration - acceleration) - scaled_jerk;
                const RowVector jerk_residual = duration3 * (end_jerk - jerk);

                // Four explicit Hermite rows determine the high coefficients in physical time.
                const RowVector quartic = (35.0 * position_residual - 15.0 * velocity_residual +
                    2.5 * acceleration_residual - (1.0 / 6.0) * jerk_residual) * tp.h4_inv;
                const RowVector quintic = (-84.0 * position_residual + 39.0 * velocity_residual -
                    7.0 * acceleration_residual + 0.5 * jerk_residual) * tp.h5_inv;
                const RowVector sextic = (70.0 * position_residual - 34.0 * velocity_residual +
                    6.5 * acceleration_residual - 0.5 * jerk_residual) * tp.h6_inv;
                const RowVector septic = (-20.0 * position_residual + 10.0 * velocity_residual -
                    2.0 * acceleration_residual + (1.0 / 6.0) * jerk_residual) * tp.h7_inv;

                // Publish together, after all source rows and duration powers have been read.
                trajectory_.coefficients_.row(coefficient_row) = position;
                trajectory_.coefficients_.row(coefficient_row + 1) = velocity;
                trajectory_.coefficients_.row(coefficient_row + 2) = quadratic;
                trajectory_.coefficients_.row(coefficient_row + 3) = cubic;
                trajectory_.coefficients_.row(coefficient_row + 4) = quartic;
                trajectory_.coefficients_.row(coefficient_row + 5) = quintic;
                trajectory_.coefficients_.row(coefficient_row + 6) = sextic;
                trajectory_.coefficients_.row(coefficient_row + 7) = septic;
            }
        }

        /** @brief Differentiate each segment's explicit Hermite map at fixed knot derivatives.
         * @param coefficient_partials Piece-major physical coefficient partials, valid for this call.
         * @param[in,out] gradient Receives direct point/time partials; adjoint_ receives derivative seeds.
         * @note Homogeneity gives dC/dT without materializing derivative matrices. */
        void pullbackCoefficients(const CoefficientMatrix &coefficient_partials, Gradients &gradient)
        {
            for (int i = 0; i < num_segments_; ++i)
            {
                const auto &tp = time_powers_[i];
                const int coefficient_row = i * kCoefficientCount;
                const int left_row = i * kDerivativesPerKnot;
                const int right_row = left_row + kDerivativesPerKnot;

                // Borrow the same contiguous endpoint rows for every segment count.
                const auto left_velocity = derivatives_.row(left_row);
                const auto left_acceleration = derivatives_.row(left_row + 1);
                const auto left_jerk = derivatives_.row(left_row + 2);
                const auto right_velocity = derivatives_.row(right_row);
                const auto right_acceleration = derivatives_.row(right_row + 1);
                const auto right_jerk = derivatives_.row(right_row + 2);

                // Apply the same Hermite map across coordinates, sharing the time multipliers.
                const RowVector g4 = coefficient_partials.row(coefficient_row + 4);
                const RowVector g5 = coefficient_partials.row(coefficient_row + 5);
                const RowVector g6 = coefficient_partials.row(coefficient_row + 6);
                const RowVector g7 = coefficient_partials.row(coefficient_row + 7);
                const RowVector grad_v_left = (-20.0 * tp.h3_inv) * g4
                    + (45.0 * tp.h4_inv) * g5 - (36.0 * tp.h5_inv) * g6 + (10.0 * tp.h6_inv) * g7;
                const RowVector grad_a_left = (-5.0 * tp.h2_inv) * g4
                    + (10.0 * tp.h3_inv) * g5 - (7.5 * tp.h4_inv) * g6 + (2.0 * tp.h5_inv) * g7;
                const RowVector grad_j_left = (-(2.0 / 3.0) * tp.h_inv) * g4 + tp.h2_inv * g5
                    - ((2.0 / 3.0) * tp.h3_inv) * g6 + ((1.0 / 6.0) * tp.h4_inv) * g7;
                const RowVector point_partial = (35.0 * tp.h4_inv) * g4
                    - (84.0 * tp.h5_inv) * g5 + (70.0 * tp.h6_inv) * g6 - (20.0 * tp.h7_inv) * g7;
                const RowVector grad_v_right = (-15.0 * tp.h3_inv) * g4
                    + (39.0 * tp.h4_inv) * g5 - (34.0 * tp.h5_inv) * g6 + (10.0 * tp.h6_inv) * g7;
                const RowVector grad_a_right = (2.5 * tp.h2_inv) * g4
                    - (7.0 * tp.h3_inv) * g5 + (6.5 * tp.h4_inv) * g6 - (2.0 * tp.h5_inv) * g7;
                const RowVector grad_j_right = (-(1.0 / 6.0) * tp.h_inv) * g4
                    + (0.5 * tp.h2_inv) * g5 - (0.5 * tp.h3_inv) * g6 + ((1.0 / 6.0) * tp.h4_inv) * g7;

                gradient.addWaypoint(i, coefficient_partials.row(coefficient_row) - point_partial);
                gradient.addWaypoint(i + 1, point_partial);
                adjoint_.row(left_row) += grad_v_left + coefficient_partials.row(coefficient_row + 1);
                adjoint_.row(right_row) += grad_v_right;
                adjoint_.row(left_row + 1) += grad_a_left + 0.5 * coefficient_partials.row(coefficient_row + 2);
                adjoint_.row(right_row + 1) += grad_a_right;
                adjoint_.row(left_row + 2) += grad_j_left + (1.0 / 6.0) * coefficient_partials.row(coefficient_row + 3);
                adjoint_.row(right_row + 2) += grad_j_right;

                // c_k scales as T^-k; endpoint derivative order r scales as T^r.
                RowVector time_partials = -4.0 * g4.cwiseProduct(trajectory_.coefficients_.row(coefficient_row + 4))
                    - 5.0 * g5.cwiseProduct(trajectory_.coefficients_.row(coefficient_row + 5))
                    - 6.0 * g6.cwiseProduct(trajectory_.coefficients_.row(coefficient_row + 6))
                    - 7.0 * g7.cwiseProduct(trajectory_.coefficients_.row(coefficient_row + 7));
                time_partials += grad_v_left.cwiseProduct(left_velocity) + grad_v_right.cwiseProduct(right_velocity)
                    + 2.0 * grad_a_left.cwiseProduct(left_acceleration) + 2.0 * grad_a_right.cwiseProduct(right_acceleration)
                    + 3.0 * grad_j_left.cwiseProduct(left_jerk) + 3.0 * grad_j_right.cwiseProduct(right_jerk);
                gradient.durations(i) += tp.h_inv * time_partials.sum();
            }
        }

        /** @brief Preserve clamped boundary seeds and solve for free knot adjoints.
         * @param[in,out] gradient Receives direct boundary derivative gradients.
         * @note A is symmetric: the primal factors also solve A^T lambda = seed. */
        void solveAdjoint(Gradients &gradient)
        {
            gradient.start.v = adjoint_.row(0).transpose();
            gradient.end.v = adjoint_.row(num_segments_ * 3 + 0).transpose();
            gradient.start.a = adjoint_.row(1).transpose();
            gradient.end.a = adjoint_.row(num_segments_ * 3 + 1).transpose();
            gradient.start.j = adjoint_.row(2).transpose();
            gradient.end.j = adjoint_.row(num_segments_ * 3 + 2).transpose();
            adjoint_.template topRows<3>().setZero();
            adjoint_.template bottomRows<3>().setZero();

            detail::solveWithFactors<kDerivativesPerKnot>(factors_, num_segments_, adjoint_);
        }

        /** @brief Add the interpolation constraint's contribution using endpoint identities.
         * @param[in,out] gradient Physical point, time and clamped-boundary gradients.
         * @note The adjoint polynomial has zero positions and knot derivatives from adjoint_.
         *       Integration by parts gives point/boundary forces; the polarized Hamiltonian
         *       gives time corrections without constructing dA/dT or differentiating the solver. */
        void accumulateAdjoint(Gradients &gradient) const
        {
            for (int i = 0; i < num_segments_; ++i)
            {
                const auto &tp = time_powers_[i];
                const int coefficient_row = i * kCoefficientCount;
                const int left_row = i * kDerivativesPerKnot;
                const int right_row = left_row + kDerivativesPerKnot;

                // Map knot adjoints to a polynomial with zero endpoint positions.
                const RowVector adjoint_v_left = adjoint_.row(left_row);
                const RowVector adjoint_v_right = adjoint_.row(right_row);
                const RowVector adjoint_a_left = adjoint_.row(left_row + 1);
                const RowVector adjoint_a_right = adjoint_.row(right_row + 1);
                const RowVector adjoint_j_left = adjoint_.row(left_row + 2);
                const RowVector adjoint_j_right = adjoint_.row(right_row + 2);
                const RowVector lambda_c4 = (-20.0 * tp.h3_inv) * adjoint_v_left
                    - (15.0 * tp.h3_inv) * adjoint_v_right
                    - (5.0 * tp.h2_inv) * adjoint_a_left + (2.5 * tp.h2_inv) * adjoint_a_right
                    - ((2.0 / 3.0) * tp.h_inv) * adjoint_j_left - ((1.0 / 6.0) * tp.h_inv) * adjoint_j_right;
                const RowVector lambda_c5 = (45.0 * tp.h4_inv) * adjoint_v_left
                    + (39.0 * tp.h4_inv) * adjoint_v_right
                    + (10.0 * tp.h3_inv) * adjoint_a_left - (7.0 * tp.h3_inv) * adjoint_a_right
                    + tp.h2_inv * adjoint_j_left + (0.5 * tp.h2_inv) * adjoint_j_right;
                const RowVector lambda_c6 = (-36.0 * tp.h5_inv) * adjoint_v_left
                    - (34.0 * tp.h5_inv) * adjoint_v_right
                    - (7.5 * tp.h4_inv) * adjoint_a_left + (6.5 * tp.h4_inv) * adjoint_a_right
                    - ((2.0 / 3.0) * tp.h3_inv) * adjoint_j_left - (0.5 * tp.h3_inv) * adjoint_j_right;
                const RowVector lambda_c7 = (10.0 * tp.h6_inv) * adjoint_v_left
                    + (10.0 * tp.h6_inv) * adjoint_v_right
                    + (2.0 * tp.h5_inv) * adjoint_a_left - (2.0 * tp.h5_inv) * adjoint_a_right
                    + ((1.0 / 6.0) * tp.h4_inv) * adjoint_j_left + ((1.0 / 6.0) * tp.h4_inv) * adjoint_j_right;

                // Endpoint forces use physical derivatives of the adjoint polynomial.
                const RowVector adjoint_snap = 24.0 * lambda_c4;
                const RowVector adjoint_crackle = 120.0 * lambda_c5;
                const RowVector adjoint_pop = 720.0 * lambda_c6;
                const RowVector adjoint_seventh = 5040.0 * lambda_c7;
                gradient.addWaypoint(i, -adjoint_seventh);
                gradient.addWaypoint(i + 1, adjoint_seventh);

                // The polarized Hamiltonian contracts matching physical derivatives.
                const RowVector velocity = trajectory_.coefficients_.row(coefficient_row + 1);
                const RowVector acceleration = 2.0 * trajectory_.coefficients_.row(coefficient_row + 2);
                const RowVector jerk = 6.0 * trajectory_.coefficients_.row(coefficient_row + 3);
                const RowVector snap = 24.0 * trajectory_.coefficients_.row(coefficient_row + 4);
                const RowVector crackle = 120.0 * trajectory_.coefficients_.row(coefficient_row + 5);
                const RowVector pop = 720.0 * trajectory_.coefficients_.row(coefficient_row + 6);
                const RowVector seventh = 5040.0 * trajectory_.coefficients_.row(coefficient_row + 7);
                const RowVector time_partials = snap.cwiseProduct(adjoint_snap)
                    - adjoint_v_left.cwiseProduct(seventh) - velocity.cwiseProduct(adjoint_seventh)
                    + adjoint_a_left.cwiseProduct(pop) + acceleration.cwiseProduct(adjoint_pop)
                    - adjoint_j_left.cwiseProduct(crackle) - jerk.cwiseProduct(adjoint_crackle);
                gradient.durations(i) += time_partials.sum();

                // Only clamped endpoints contribute boundary derivative corrections.
                if (i == 0)
                {
                    gradient.start.v += adjoint_pop.transpose();
                    gradient.start.a -= adjoint_crackle.transpose();
                    gradient.start.j += adjoint_snap.transpose();
                }
                if (i + 1 == num_segments_)
                {
                    const RowVector end_v = tp.h * adjoint_seventh + adjoint_pop;
                    const RowVector end_a = ((2520.0 * tp.h) * lambda_c7 + adjoint_pop) * tp.h + adjoint_crackle;
                    const RowVector end_j = (((840.0 * tp.h) * lambda_c7 + 360.0 * lambda_c6) * tp.h
                        + adjoint_crackle) * tp.h + adjoint_snap;
                    gradient.end.v -= end_v.transpose();
                    gradient.end.a += end_a.transpose();
                    gradient.end.j -= end_j.transpose();
                }
            }
        }
    };

    // Public dimension aliases and compile-time order selection.

    using SplinePoint1d = Eigen::Matrix<double, 1, 1>;
    using SplinePoint2d = Eigen::Matrix<double, 2, 1>;
    using SplinePoint3d = Eigen::Matrix<double, 3, 1>;
    using SplinePoint4d = Eigen::Matrix<double, 4, 1>;
    using SplinePoint5d = Eigen::Matrix<double, 5, 1>;
    using SplinePoint6d = Eigen::Matrix<double, 6, 1>;
    using SplinePoint7d = Eigen::Matrix<double, 7, 1>;
    using SplinePoint8d = Eigen::Matrix<double, 8, 1>;
    using SplinePoint9d = Eigen::Matrix<double, 9, 1>;
    using SplinePoint10d = Eigen::Matrix<double, 10, 1>;

    using SplineVector1D = SplineVector<SplinePoint1d>;
    using SplineVector2D = SplineVector<SplinePoint2d>;
    using SplineVector3D = SplineVector<SplinePoint3d>;
    using SplineVector4D = SplineVector<SplinePoint4d>;
    using SplineVector5D = SplineVector<SplinePoint5d>;
    using SplineVector6D = SplineVector<SplinePoint6d>;
    using SplineVector7D = SplineVector<SplinePoint7d>;
    using SplineVector8D = SplineVector<SplinePoint8d>;
    using SplineVector9D = SplineVector<SplinePoint9d>;
    using SplineVector10D = SplineVector<SplinePoint10d>;

    using PPoly1D = PPolyND<1>;
    using PPoly2D = PPolyND<2>;
    using PPoly3D = PPolyND<3>;
    using PPoly4D = PPolyND<4>;
    using PPoly5D = PPolyND<5>;
    using PPoly6D = PPolyND<6>;
    using PPoly7D = PPolyND<7>;
    using PPoly8D = PPolyND<8>;
    using PPoly9D = PPolyND<9>;
    using PPoly10D = PPolyND<10>;
    using PPoly = PPoly3D;

    /**
     * @brief Unified minimum-derivative spline name.
     *
     * The three practically relevant orders remain compile-time-specialized
     * implementations, so selecting S adds no runtime branch or virtual call:
     * S=2 is cubic/minimum acceleration, S=3 quintic/minimum jerk, and S=4
     * septic/minimum snap.
     */
    template <int DIM, int S>
    struct MinDerivativeSplineSelector
    {
        static_assert(S >= 2 && S <= 4,
                      "MinDerivativeSplineND currently specializes S=2, 3, and 4.");
    };

    template <int DIM>
    struct MinDerivativeSplineSelector<DIM, 2>
    {
        using type = CubicSplineND<DIM>;
    };

    template <int DIM>
    struct MinDerivativeSplineSelector<DIM, 3>
    {
        using type = QuinticSplineND<DIM>;
    };

    template <int DIM>
    struct MinDerivativeSplineSelector<DIM, 4>
    {
        using type = SepticSplineND<DIM>;
    };

    template <int DIM, int S>
    using MinDerivativeSplineND =
        typename MinDerivativeSplineSelector<DIM, S>::type;

    using CubicSpline1D = CubicSplineND<1>;
    using CubicSpline2D = CubicSplineND<2>;
    using CubicSpline3D = CubicSplineND<3>;
    using CubicSpline4D = CubicSplineND<4>;
    using CubicSpline5D = CubicSplineND<5>;
    using CubicSpline6D = CubicSplineND<6>;
    using CubicSpline7D = CubicSplineND<7>;
    using CubicSpline8D = CubicSplineND<8>;
    using CubicSpline9D = CubicSplineND<9>;
    using CubicSpline10D = CubicSplineND<10>;
    using CubicSpline = CubicSpline3D;

    using QuinticSpline1D = QuinticSplineND<1>;
    using QuinticSpline2D = QuinticSplineND<2>;
    using QuinticSpline3D = QuinticSplineND<3>;
    using QuinticSpline4D = QuinticSplineND<4>;
    using QuinticSpline5D = QuinticSplineND<5>;
    using QuinticSpline6D = QuinticSplineND<6>;
    using QuinticSpline7D = QuinticSplineND<7>;
    using QuinticSpline8D = QuinticSplineND<8>;
    using QuinticSpline9D = QuinticSplineND<9>;
    using QuinticSpline10D = QuinticSplineND<10>;
    using QuinticSpline = QuinticSpline3D;

    using SepticSpline1D = SepticSplineND<1>;
    using SepticSpline2D = SepticSplineND<2>;
    using SepticSpline3D = SepticSplineND<3>;
    using SepticSpline4D = SepticSplineND<4>;
    using SepticSpline5D = SepticSplineND<5>;
    using SepticSpline6D = SepticSplineND<6>;
    using SepticSpline7D = SepticSplineND<7>;
    using SepticSpline8D = SepticSplineND<8>;
    using SepticSpline9D = SepticSplineND<9>;
    using SepticSpline10D = SepticSplineND<10>;
    using SepticSpline = SepticSpline3D;

} // namespace SplineTrajectory

#endif // SPLINE_TRAJECTORY_HPP
