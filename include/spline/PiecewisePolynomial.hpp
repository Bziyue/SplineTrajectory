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

/** @file PiecewisePolynomial.hpp
 * @brief Owning piecewise polynomials, samplers and physical boundary/gradient types. */

#ifndef SPLINE_PIECEWISE_POLYNOMIAL_HPP
#define SPLINE_PIECEWISE_POLYNOMIAL_HPP

#include <Eigen/Dense>
#include <array>
#include <vector>
#include <algorithm>
#include <cmath>
#include <cstddef>
#include <iterator>
#include <limits>
#include <stdexcept>
#include <utility>

namespace SplineTrajectory
{
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
     *       Invalid numeric construction produces an empty, invalid value. */
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

        inline double derivativeFactor(int n, int k) const
        {
            if (k < 0 || k > n)
                return 0.0;
            if constexpr (kHasFixedCapacity && kCapacity <= kStaticFactorMaxOrder)
                return kStaticDerivativeFactorTable_[n][k];
            if (n < kStaticFactorMaxOrder && num_coeffs_ <= kStaticFactorMaxOrder)
                return kStaticDerivativeFactorTable_[n][k];
            double factor = 1.0;
            for (int i = 0; i < k; ++i) factor *= n - i;
            return factor;
        }

        inline Vector evaluateSegmentHorner(int segment_idx, double t, int derivative_order) const
        {
            if (derivative_order >= num_coeffs_ || derivative_order < 0)
                return Vector::Zero();

            const int base_row = segment_idx * num_coeffs_;
            Vector result = derivativeFactor(num_coeffs_ - 1, derivative_order) *
                                coefficients_.row(base_row + num_coeffs_ - 1).transpose();
            for (int k = num_coeffs_ - 2; k >= derivative_order; --k)
                result = result * t + derivativeFactor(k, derivative_order) *
                                     coefficients_.row(base_row + k).transpose();
            return result;
        }

    public:
        /**
         * @brief Sequential derivative evaluator with private, current-piece coefficient storage.
         * @tparam MAX_DERIVATIVE Highest physical-time derivative returned, including position at zero.
         * @note Borrows its source, which must remain alive and unmodified. Any source update,
         *       assignment or move invalidates this sampler; create another after that operation.
         *       Separate samplers may concurrently read the same source without warming it.
         */
        template <int MAX_DERIVATIVE>
        class Sampler
        {
            static_assert(MAX_DERIVATIVE >= 0, "Derivative order must be nonnegative");
            using PieceMatrix = Eigen::Matrix<double, MaxCoefficients, DIM, kMatrixOptions>;
            const PPolyND *source_;
            std::array<PieceMatrix, MAX_DERIVATIVE + 1> coefficients_;
            int segment_ = -1;

            void preparePiece(int segment)
            {
                segment_ = segment;
                const int count = source_->num_coeffs_;
                for (int d = 0; d <= MAX_DERIVATIVE && d < count; ++d)
                    for (int k = d; k < count; ++k)
                        coefficients_[d].row(k - d) = source_->derivativeFactor(k, d) *
                            source_->coefficients_.row(segment * count + k);
            }

        public:
            using Sample = std::array<Vector, MAX_DERIVATIVE + 1>;

            /** @brief Bind a sampler without changing or warming the source.
             * @param source Borrowed immutable polynomial; must outlive this sampler.
             * @note Dynamic-order polynomials allocate piece storage here; fixed-order ones do not. */
            explicit Sampler(const PPolyND &source) : source_(&source)
            {
                if constexpr (!kHasFixedCapacity)
                    for (int d = 0; d <= MAX_DERIVATIVE && d < source.num_coeffs_; ++d)
                        coefficients_[d].resize(source.num_coeffs_ - d, DIM);
            }
            Sampler(PPolyND &&) = delete;
            Sampler(const PPolyND &&) = delete;

            /** @brief Sample position through MAX_DERIVATIVE on the source time axis, in seconds.
             * @param t Query time; backward jumps relocate, forward queries advance piece by piece.
             * @return Derivatives in increasing order, or all NaNs for an invalid source/time.
             * @note Interior knots use the right piece; domain endpoints use their adjacent piece;
             *       outside queries extrapolate. No allocation occurs here. Only this sampler is mutated. */
            Sample evaluate(double t)
            {
                Sample values;
                if (!source_->is_initialized_ || !std::isfinite(t))
                {
                    for (auto &value : values)
                        value.setConstant(std::numeric_limits<double>::quiet_NaN());
                    return values;
                }
                int segment = segment_;
                if (segment < 0 || t < source_->breakpoints_[segment])
                    segment = source_->findSegment(t);
                else
                    while (segment + 1 < source_->num_segments_ &&
                           t >= source_->breakpoints_[segment + 1])
                        ++segment;
                if (segment != segment_) preparePiece(segment);
                const double local_time = t - source_->breakpoints_[segment];
                for (int d = 0; d <= MAX_DERIVATIVE; ++d)
                {
                    const int count = source_->num_coeffs_ - d;
                    if (count <= 0)
                        values[d].setZero();
                    else
                    {
                        values[d] = coefficients_[d].row(count - 1).transpose();
                        for (int k = count - 2; k >= 0; --k)
                            values[d] = values[d] * local_time + coefficients_[d].row(k).transpose();
                    }
                }
                return values;
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

            Segment(const PPolyND *parent, int idx) : parent_(parent), idx_(idx) {}

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
             * @return Requested derivative, NaNs for invalid time, zero above degree.
             * @throws std::invalid_argument For a negative derivative order. */
            Vector evaluateLocal(double t, Deriv type = Deriv::Pos) const
            {
                return evaluateLocal(t, static_cast<int>(type));
            }

            Vector evaluateLocal(double t, int derivative_order) const
            {
                if (derivative_order < 0)
                    throw std::invalid_argument("Derivative order must be nonnegative");
                if (!parent_->is_initialized_ || !std::isfinite(t))
                    return Vector::Constant(std::numeric_limits<double>::quiet_NaN());
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
         * @note Invalid input produces an empty invalid value. Actual count must fit capacity. */
        PPolyND(const std::vector<double> &breakpoints,
                const CoefficientMatrix &coefficients, int coefficient_count)
        { initializeInternal(breakpoints, coefficients, coefficient_count); }

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
        /** @brief Borrow a segment with bounds checking; source mutation invalidates it. */
        Segment at(int idx) const &
        {
            if (idx < 0 || idx >= num_segments_)
                throw std::out_of_range("Segment index out of range");
            return Segment(this, idx);
        }

        /** @brief Query the trajectory time axis in seconds, extrapolating outside its domain.
         * @return Requested physical derivative; NaNs for invalid source/time, zero above degree.
         * @throws std::invalid_argument For a negative derivative order. No allocation or shared write occurs. */
        Vector evaluate(double t, int derivative_order) const
        {
            if (derivative_order < 0)
                throw std::invalid_argument("Derivative order must be nonnegative");
            if (!is_initialized_ || !std::isfinite(t))
                return Vector::Constant(std::numeric_limits<double>::quiet_NaN());
            if (derivative_order >= num_coeffs_)
                return Vector::Zero();
            int segment_idx = findSegment(t);
            double dt = t - breakpoints_[segment_idx];

            return evaluateSegmentHorner(segment_idx, dt, derivative_order);
        }

        Vector evaluate(double t, Deriv type = Deriv::Pos) const
        {
            return evaluate(t, static_cast<int>(type));
        }

        /** @brief Evaluate derivatives 0..MAX_DERIVATIVE with one segment lookup and no shared writes.
         * @param t Time on this spline's axis; endpoints use the adjacent end piece, interior knots use the right piece.
         * @return Fixed-size derivative array; invalid spline/time produces NaN. Outside times extrapolate the end piece.
         * @note Const calls are reentrant; update requires exclusive ownership. No heap allocation occurs during evaluation. */
        template <int MAX_DERIVATIVE>
        std::array<Vector, MAX_DERIVATIVE + 1> evaluateDerivatives(double t) const
        {
            static_assert(MAX_DERIVATIVE >= 0, "Derivative order must be nonnegative");
            std::array<Vector, MAX_DERIVATIVE + 1> values;
            if (!is_initialized_ || !std::isfinite(t))
            {
                for (auto &value : values) value.setConstant(std::numeric_limits<double>::quiet_NaN());
                return values;
            }
            const int segment = findSegment(t);
            const double dt = t - breakpoints_[segment];
            for (int derivative = 0; derivative <= MAX_DERIVATIVE; ++derivative)
                values[derivative] = evaluateSegmentHorner(segment, dt, derivative);
            return values;
        }

        /** @brief Evaluate one derivative into caller-owned rows without allocating.
         * @param times Query times on the trajectory axis.
         * @param[out] output Matrix with times.size() rows and DIM columns.
         * @param derivative_order Nonnegative physical derivative order.
         * @throws std::invalid_argument For a wrong output shape or negative order. */
        void evaluateInto(const std::vector<double> &times, Eigen::Ref<CoefficientMatrix> output,
                          int derivative_order = 0) const
        {
            if (output.rows() != static_cast<Eigen::Index>(times.size()) || derivative_order < 0)
                throw std::invalid_argument("Invalid batch shape or derivative order");
            for (std::size_t i = 0; i < times.size(); ++i)
                output.row(static_cast<Eigen::Index>(i)) = evaluate(times[i], derivative_order).transpose();
        }

        /** @brief Allocating batch convenience; each element uses evaluate()'s time and error contract. */
        SplineVector<Vector> evaluate(const std::vector<double> &t, int derivative_order) const
        {
            if (derivative_order < 0) throw std::invalid_argument("Derivative order must be nonnegative");
            SplineVector<Vector> results;
            results.reserve(t.size());
            for (double time : t)
                results.push_back(evaluate(time, derivative_order));
            return results;
        }

        SplineVector<Vector> evaluate(const std::vector<double> &t, Deriv type = Deriv::Pos) const
        {
            return evaluate(t, static_cast<int>(type));
        }





        double length(double dt = 0.01) const
        {
            return length(startTime(), endTime(), dt);
        }

        double length(double start_t, double end_t, double dt = 0.01) const
        {
            std::vector<double> time_sequence = generateTimeSequence(start_t, end_t, dt);

            double total_length = 0.0;
            for (size_t i = 0; i < time_sequence.size() - 1; ++i)
            {
                double t_current = time_sequence[i];
                double t_next = time_sequence[i + 1];
                double dt_actual = t_next - t_current;

                Vector velocity = evaluate(t_current, Deriv::Vel);
                total_length += velocity.norm() * dt_actual;
            }
            return total_length;
        }

        /** @brief Replace owning data, invalidating borrowed views; invalid input clears the value. */
        void update(const std::vector<double> &breakpoints, const CoefficientMatrix &coefficients, int num_coefficients)
        {
            initializeInternal(breakpoints, coefficients, num_coefficients);
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

        /** @brief Generate increasing query times, including both endpoints.
         * @return One value for a zero-length interval.
         * @throws std::invalid_argument For nonfinite inputs, nonpositive step or reversed interval.
         * @throws std::length_error For an unrepresentable sample count or advancing step. */
        std::vector<double> generateTimeSequence(double start_t, double end_t, double dt) const
        {
            if (!std::isfinite(start_t) || !std::isfinite(end_t) || !std::isfinite(dt) ||
                dt <= 0.0 || end_t < start_t)
                throw std::invalid_argument("Time sequence requires a finite forward interval and positive step");
            const double steps = std::floor((end_t - start_t) / dt);
            if (!std::isfinite(steps) || steps > static_cast<double>(std::numeric_limits<int>::max() - 2) ||
                (start_t < end_t && start_t + dt <= start_t))
                throw std::length_error("Time sequence cannot be represented");
            std::vector<double> result;
            result.reserve(static_cast<std::size_t>(steps) + 2);
            result.push_back(start_t);
            for (int i = 1; i <= static_cast<int>(steps); ++i)
            {
                const double t = start_t + i * dt;
                if (t >= end_t) break;
                if (t <= result.back()) throw std::length_error("Time sequence cannot advance");
                result.push_back(t);
            }
            if (end_t != start_t) result.push_back(end_t);
            return result;
        }

        std::vector<double> generateTimeSequence(double dt) const
        {
            return generateTimeSequence(startTime(), endTime(), dt);
        }

        /** @brief Return an owning derivative polynomial; negative orders throw invalid_argument. */
        PPolyND derivative(int derivative_order = 1) const
        {
            if (derivative_order < 0)
                throw std::invalid_argument("Derivative order must be nonnegative");
            if (num_segments_ == 0 || derivative_order >= num_coeffs_)
            {
                if (num_segments_ == 0)
                    return PPolyND();
                CoefficientMatrix zero_coeffs = CoefficientMatrix::Zero(num_segments_, DIM);
                return PPolyND(breakpoints_, zero_coeffs, 1);
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
            return PPolyND(breakpoints_, new_coeffs, new_order);
        }

        /** @brief Extract and rebase an interval using exact coefficient translation.
         * @param start Start on the original time axis, within the source domain.
         * @param end End on the original axis, strictly after start and within the source domain.
         * @return Zero-origin spline, or an uninitialized value for invalid input/nonfinite output.
         * @note Preserves piece boundaries and coefficients outside the first clipped piece; const and reentrant. */
        PPolyND extractInterval(double start, double end) const
        {
            if (!is_initialized_ || !std::isfinite(start) || !std::isfinite(end) ||
                start < startTime() || end > endTime() || end <= start) return {};
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
                times.push_back(times.back() + std::min(end, breakpoints_[i + 1]) - std::max(start, breakpoints_[i]));
            return PPolyND(times, coefficients, num_coeffs_);
        }

        /** @brief Append two zero-origin splines at coefficient level without a continuity policy.
         * @param suffix Same-order zero-origin spline appended after this value.
         * @return Concatenated spline, or an uninitialized value for invalid input/output.
         * @note Callers own seam acceptance policy; inputs are read-only, no fitting or resampling occurs. */
        PPolyND appendRebased(const PPolyND &suffix) const
        {
            if (!is_initialized_ || !suffix.is_initialized_ || startTime() != 0.0 ||
                suffix.startTime() != 0.0 || num_coeffs_ != suffix.num_coeffs_) return {};
            std::vector<double> times = breakpoints_;
            for (auto it = suffix.breakpoints_.begin() + 1; it != suffix.breakpoints_.end(); ++it)
                times.push_back(endTime() + *it);
            CoefficientMatrix coefficients(coefficients_.rows() + suffix.coefficients_.rows(), DIM);
            coefficients.topRows(coefficients_.rows()) = coefficients_;
            coefficients.bottomRows(suffix.coefficients_.rows()) = suffix.coefficients_;
            return PPolyND(times, coefficients, num_coeffs_);
        }

        /** @brief Construct an owning zero polynomial; coefficient count must fit the capacity. */
        static PPolyND zero(const std::vector<double> &breakpoints, int num_coefficients = 1)
        {
            if (num_coefficients <= 0 || (kHasFixedCapacity && num_coefficients > kCapacity) ||
                breakpoints.size() < 2 || breakpoints.size() - 1 >
                    static_cast<std::size_t>(std::numeric_limits<int>::max() / num_coefficients)) return {};
            int num_segments = static_cast<int>(breakpoints.size()) - 1;
            CoefficientMatrix zero_coeffs = CoefficientMatrix::Zero(num_segments * num_coefficients, DIM);
            return PPolyND(breakpoints, zero_coeffs, num_coefficients);
        }

        /** @brief Construct an owning constant polynomial, or an invalid value for invalid input. */
        static PPolyND constant(const std::vector<double> &breakpoints, const Vector &constant_value)
        {
            if (breakpoints.size() < 2 || breakpoints.size() - 1 >
                static_cast<std::size_t>(std::numeric_limits<int>::max())) return {};
            int num_segments = breakpoints.size() > 1 ? static_cast<int>(breakpoints.size()) - 1 : 0;
            CoefficientMatrix coeffs = CoefficientMatrix::Zero(num_segments, DIM);
            for (int i = 0; i < num_segments; ++i)
            {
                coeffs.row(i) = constant_value.transpose();
            }
            return PPolyND(breakpoints, coeffs, 1);
        }

    private:
        template <int, int> friend class MinDerivativeSplineND;

        /** @brief Prepare exclusive solver access; invalidate before writing any numeric data. */
        template <int MinDerivativeOrder>
        bool beginSpline(const std::vector<double> &durations, const CoefficientMatrix &points,
                         const BoundaryConditions<DIM> &boundary, double start, int count)
        {
            is_initialized_ = false;
            bool valid = !durations.empty() && durations.size() <=
                static_cast<std::size_t>(std::numeric_limits<int>::max() / count) &&
                points.rows() == static_cast<Eigen::Index>(durations.size()) + 1 &&
                points.allFinite() && std::isfinite(start) &&
                boundary.start_velocity.allFinite() && boundary.end_velocity.allFinite();
            if constexpr (MinDerivativeOrder >= 3)
                valid = valid && boundary.start_acceleration.allFinite() && boundary.end_acceleration.allFinite();
            if constexpr (MinDerivativeOrder >= 4)
                valid = valid && boundary.start_jerk.allFinite() && boundary.end_jerk.allFinite();
            if (!valid) { clear(); return false; }
            breakpoints_.resize(durations.size() + 1);
            breakpoints_[0] = start;
            for (std::size_t i = 0; i < durations.size(); ++i)
            {
                const double next = breakpoints_[i] + durations[i];
                if (!std::isfinite(durations[i]) || durations[i] <= 0.0 ||
                    !std::isfinite(next) || next <= breakpoints_[i])
                { clear(); return false; }
                breakpoints_[i + 1] = next;
            }
            num_segments_ = static_cast<int>(durations.size());
            num_coeffs_ = count;
            coefficients_.resize(num_segments_ * count, DIM);
            return true;
        }

        /** @brief Publish only finite coefficients; no copy of knots or coefficients is made. */
        void finishSpline()
        {
            if (!coefficients_.allFinite()) clear();
            else is_initialized_ = true;
        }

        inline void initializeInternal(const std::vector<double> &breakpoints,
                                       const CoefficientMatrix &coefficients,
                                       int num_coefficients)
        {
            const bool valid = breakpoints.size() >= 2 && num_coefficients > 0 &&
                breakpoints.size() - 1 <= static_cast<size_t>(std::numeric_limits<int>::max() / num_coefficients) &&
                (!kHasFixedCapacity || num_coefficients <= kCapacity) &&
                coefficients.rows() == static_cast<Eigen::Index>(breakpoints.size() - 1) * num_coefficients &&
                coefficients.allFinite() &&
                std::all_of(breakpoints.begin(), breakpoints.end(), [](double t) { return std::isfinite(t); }) &&
                std::adjacent_find(breakpoints.begin(), breakpoints.end(),
                    [](double a, double b) { return b <= a; }) == breakpoints.end();
            if (!valid)
            {
                num_segments_ = num_coeffs_ = 0;
                is_initialized_ = false;
                breakpoints_.clear();
                coefficients_.resize(0, DIM);
                return;
            }
            breakpoints_ = breakpoints;
            coefficients_ = coefficients;
            num_coeffs_ = num_coefficients;
            num_segments_ = static_cast<int>(breakpoints.size()) - 1;
            is_initialized_ = true;
        }

        inline int findSegment(double t) const
        {
            if (num_segments_ == 0 || breakpoints_.empty())
                return 0;
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
        CoefficientMatrix inner_points;
        Eigen::VectorXd durations;
        BoundaryGradient<DIM> start, end;
        /** @brief Allocate output storage for a topology; does not initialize values. */
        void resetTopology(int segments)
        {
            if (segments < 0) throw std::invalid_argument("Negative segment count");
            inner_points.resize(std::max(0, segments - 1), DIM);
            durations.resize(segments);
        }
        void setZero()
        {
            inner_points.setZero(); durations.setZero();
            start = {}; end = {};
        }
    };

    namespace detail
    {
        /** @brief Convert absolute knots to durations; empty input yields an invalid empty topology. */
        inline void durationsFromKnots(const std::vector<double> &knots,
                                       std::vector<double> &durations, double &start)
        {
            start = knots.empty() ? 0.0 : knots.front();
            durations.resize(knots.empty() ? 0 : knots.size() - 1);
            for (std::size_t i = 0; i < durations.size(); ++i)
                durations[i] = knots[i + 1] - knots[i];
        }
    }

} // namespace SplineTrajectory

#endif
