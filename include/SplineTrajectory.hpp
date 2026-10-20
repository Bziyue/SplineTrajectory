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
#include <stdexcept>
#include <utility>

namespace SplineTrajectory
{
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
        template <int> friend class CubicSplineND;
        template <int> friend class QuinticSplineND;
        template <int> friend class SepticSplineND;

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
            if (!coefficients_.array().isFinite().all()) clear();
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
        using RowVector = typename PPolyND<DIM>::RowVector;
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

        /** @brief Accumulate one coordinate of a position partial without forming a vector.
         * @param knot Waypoint index in [0, N], with N given by the prepared durations size.
         * @param coordinate Spatial coordinate index in [0, DIM).
         * @param partial Physical position partial in the caller's coordinates.
         * @note Requires a prepared N-segment topology and exclusive access to this output. */
        void addWaypoint(int knot, int coordinate, double partial)
        {
            if (knot == 0)
                start.p(coordinate) += partial;
            else if (knot == durations.size())
                end.p(coordinate) += partial;
            else
                inner_points(knot - 1, coordinate) += partial;
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

        /** @brief Validate adjoint inputs and initialize reusable physical gradient storage.
         * @param polynomial Published spline in the caller's coordinates and seconds; borrowed for this call.
         * @param coefficient_partials Piece-major partials with respect to ascending physical-time coefficients.
         * @param duration_partials Direct duration partials at fixed coefficients; borrowed for this call.
         * @param[out] gradient Total gradient storage; matching allocations are reused.
         * @param[out] adjoint Construction-prepared knot workspace; cleared without resizing.
         * @throws std::invalid_argument If the polynomial is invalid or partial shapes do not match.
         * @note Validates before modifying outputs. Requires exclusive access to gradient and adjoint;
         *       the knot workspace must already match the polynomial topology. */
        template <int DIM, int CoefficientCount>
        inline void prepareAdjoint(const PPolyND<DIM, CoefficientCount> &polynomial,
                                   const typename PPolyND<DIM>::CoefficientMatrix &coefficient_partials,
                                   const Eigen::VectorXd &duration_partials,
                                   SplineGradients<DIM> &gradient,
                                   typename PPolyND<DIM>::CoefficientMatrix &adjoint)
        {
            const int segments = polynomial.numSegments();
            if (!polynomial.isValid() || coefficient_partials.rows() != segments * CoefficientCount ||
                duration_partials.size() != segments)
                throw std::invalid_argument("Adjoint requires a valid spline and matching partial shapes");

            // Preserve direct duration partials while clearing all accumulated contributions.
            gradient.resetTopology(segments);
            gradient.setZero();
            gradient.durations = duration_partials;
            adjoint.setZero();
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
    } // namespace detail

    // Cubic splines: minimum acceleration.

    /** @brief Minimum-acceleration interpolation with a specialized O(N) scalar tridiagonal solver.
     * @note Minimize the squared second physical derivative integral with fixed waypoint
     *       positions and endpoint derivatives of orders 1 through 1. Coordinates
     *       are caller-defined; durations use seconds. Polynomial coefficients and knots
     *       have one owner. Updates and backward() require exclusive access; const energy
     *       and polynomial queries may run concurrently on an unchanged valid object.
     *       Copies own independent data. Moving invalidates borrowed views and the source polynomial.
     *       Numeric construction failure clears the polynomial; inspect isValid(). */
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
         * @note Inputs are copied; invalid input or numeric output leaves an invalid polynomial. */
        CubicSplineND(const std::vector<double> &t_points,
                      const CoefficientMatrix &spatial_points,
                      const BoundaryConditions<DIM> &boundary = BoundaryConditions<DIM>())
            : spatial_points_(spatial_points), boundary_(boundary)
        {
            detail::durationsFromKnots(t_points, time_segments_, start_time_);
            updateSplineInternal();
        }

        /** @brief Construct from N positive durations, N+1 waypoint rows and a common time origin.
         * @param time_segments Durations in seconds. @param spatial_points Physical waypoint rows.
         * @param start_time Absolute time origin in seconds.
         * @param boundary Clamped physical endpoint derivatives in the caller's frame.
         * @note Inputs are copied; invalid input or numeric output leaves an invalid polynomial. */
        CubicSplineND(const std::vector<double> &time_segments,
                      const CoefficientMatrix &spatial_points,
                      double start_time,
                      const BoundaryConditions<DIM> &boundary = BoundaryConditions<DIM>())
            : time_segments_(time_segments), spatial_points_(spatial_points), boundary_(boundary),
              start_time_(start_time)
        {
            updateSplineInternal();
        }

        /** @brief Replace interpolation data using absolute knots; invalid input clears the polynomial.
         * @param t_points Absolute times in seconds. @param spatial_points Physical waypoint rows.
         * @param boundary Clamped physical endpoint derivatives in the caller's frame.
         * @note Copies inputs, reuses fixed-topology storage and invalidates all borrowed views. */
        void update(const std::vector<double> &t_points,
                    const CoefficientMatrix &spatial_points,
                    const BoundaryConditions<DIM> &boundary = BoundaryConditions<DIM>())
        {
            spatial_points_ = spatial_points;
            boundary_ = boundary;
            detail::durationsFromKnots(t_points, time_segments_, start_time_);
            updateSplineInternal();
        }

        /** @brief Replace interpolation data using positive durations and a common time origin.
         * @param time_segments Durations in seconds. @param spatial_points Physical waypoint rows.
         * @param start_time Absolute time origin in seconds.
         * @param boundary Clamped physical endpoint derivatives in the caller's frame.
         * @note Copies inputs, reuses fixed-topology storage and invalidates all borrowed views.
         *       Invalid shapes, times, boundary data or numeric output clear the polynomial. */
        void update(const std::vector<double> &time_segments,
                    const CoefficientMatrix &spatial_points,
                    double start_time,
                    const BoundaryConditions<DIM> &boundary = BoundaryConditions<DIM>())
        {
            time_segments_ = time_segments;
            spatial_points_ = spatial_points;
            boundary_ = boundary;
            start_time_ = start_time;
            updateSplineInternal();
        }

        /** @brief Inspect the published polynomial. @return True only for a valid finite trajectory. */
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
         * @return Energy in physical units; NaN when the spline is invalid. */
        double energy() const
        {
            if (!trajectory_.isValid())
                return std::numeric_limits<double>::quiet_NaN();

            double total_energy = 0.0;
            for (int i = 0; i < num_segments_; ++i)
            {
                const double T = time_powers_[i].h;
                if (T <= 0)
                    continue;

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
         */
        void energyCoefficientPartials(CoefficientMatrix &gdC) const
        {
            if (!trajectory_.isValid()) throw std::logic_error("Energy gradients require a valid spline");
            gdC.resize(num_segments_ * 4, DIM);
            gdC.setZero();

            for (int i = 0; i < num_segments_; ++i)
            {
                const double T = time_powers_[i].h;
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
            CoefficientMatrix gdC;
            energyCoefficientPartials(gdC);
            return gdC;
        }

        /**
         * @brief Compute partial gradient of energy w.r.t. segment durations.
         *
         * @param[out] gdT Overwritten partial gradient ∂E/∂T, size num_segments.
         *         Element i is the direct partial derivative w.r.t. T[i].
         * @note Local physical-time coefficients are held fixed; spline interpolation constraints are not differentiated.
         */
        void energyDurationPartials(Eigen::VectorXd &gdT) const
        {
            if (!trajectory_.isValid()) throw std::logic_error("Energy gradients require a valid spline");
            gdT.resize(num_segments_);

            for (int i = 0; i < num_segments_; ++i)
            {
                const double T = time_powers_[i].h;

                const RowVector c2 = trajectory_.coefficients_.row(i * 4 + 2);
                const RowVector c3 = trajectory_.coefficients_.row(i * 4 + 3);

                RowVector acc_end = 2.0 * c2 + (6.0 * T) * c3;

                gdT(i) = acc_end.squaredNorm();
            }
        }

        /** @brief Allocate direct energy duration partials. @return Physical duration gradients at fixed coefficients. */
        Eigen::VectorXd energyDurationPartials() const
        {
            Eigen::VectorXd gdT;
            energyDurationPartials(gdT);
            return gdT;
        }

        /**
         * @brief Compute full gradient of energy w.r.t. segment durations.
         *
         * @param[out] grad Full gradient dE/dT, size num_segments.
         *         Includes both direct and indirect dependencies via chain rule.
         */
        void energyDurationGradient(Eigen::VectorXd &grad) const
        {
            if (!trajectory_.isValid()) throw std::logic_error("Energy gradients require a valid spline");
            grad.resize(num_segments_);

            for (int i = 0; i < num_segments_; ++i)
            {
                const RowVector c1 = trajectory_.coefficients_.row(i * 4 + 1);
                const RowVector c2 = trajectory_.coefficients_.row(i * 4 + 2);
                const RowVector c3 = trajectory_.coefficients_.row(i * 4 + 3);

                double term_acc = 4.0 * c2.squaredNorm();
                double term_jv = 12.0 * c1.dot(c3);

                grad(i) = -term_acc + term_jv;
            }
        }

        /** @brief Allocate total energy duration gradients. @return Gradients including interpolation dependence. */
        Eigen::VectorXd energyDurationGradient() const
        {
            Eigen::VectorXd grad;
            energyDurationGradient(grad);
            return grad;
        }

        /**
         * @brief Compute full gradient of energy w.r.t. inner waypoints only.
         * @param[out] grad Full gradient dE/dP for inner points only.
         *         Size (N-1) × DIM (excludes boundary points P0 and PN).
         */
        void energyWaypointGradient(CoefficientMatrix &grad) const
        {
            if (!trajectory_.isValid()) throw std::logic_error("Energy gradients require a valid spline");
            if (num_segments_ < 1)
            {
                grad.resize(0, DIM);
                return;
            }

            int num_rows = std::max(0, num_segments_ - 1);
            grad.resize(num_rows, DIM);

            for (int i = 1; i < num_segments_; ++i)
            {
                const RowVector c3_L = trajectory_.coefficients_.row((i - 1) * 4 + 3);
                const RowVector c3_R = trajectory_.coefficients_.row(i * 4 + 3);
                grad.row(i - 1) = 12.0 * (c3_R - c3_L);
            }
        }

        /** @brief Allocate total energy waypoint gradients. @return Interior physical position gradients. */
        CoefficientMatrix energyWaypointGradient() const
        {
            CoefficientMatrix grad;
            energyWaypointGradient(grad);
            return grad;
        }

        /**
         * @brief Compute partial gradient of energy w.r.t. boundary conditions (Position, Velocity).
         * @return BoundaryGradients containing gradients for start/end pos and vel.
         */
        BoundaryGradients energyBoundaryGradient() const
        {
            if (!trajectory_.isValid()) throw std::logic_error("Energy gradients require a valid spline");
            BoundaryGradients res;
            if (num_segments_ < 1)
                return res;

            const RowVector c2_first = trajectory_.coefficients_.row(2);
            const RowVector c3_first = trajectory_.coefficients_.row(3);

            res.start.p = 12.0 * c3_first.transpose();
            res.start.v = -4.0 * c2_first.transpose();

            int last_idx = num_segments_ - 1;
            double T = time_segments_[last_idx];

            const RowVector c2_last = trajectory_.coefficients_.row(last_idx * 4 + 2);
            const RowVector c3_last = trajectory_.coefficients_.row(last_idx * 4 + 3);

            RowVector acc_end = 2.0 * c2_last + 6.0 * c3_last * T;
            RowVector jerk_end = 6.0 * c3_last;

            res.end.p = -2.0 * jerk_end.transpose();
            res.end.v = 2.0 * acc_end.transpose();

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
            energyGradient(res);
            return res;
        }

        /**
         * @brief Compute full energy gradient combining time, inner point, and boundary gradients (reference overload).
         *
         * @param[out] grads Overwritten physical gradients; correctly sized buffers are reused
         */
        void energyGradient(Gradients &grads) const
        {
            energyWaypointGradient(grads.inner_points);
            energyDurationGradient(grads.durations);
            BoundaryGradients boundary = energyBoundaryGradient();
            grads.start = boundary.start;
            grads.end = boundary.end;
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

            backwardImpl(partialGradByCoeffs, partialGradByTimes, res);

            return res;
        }

        /**
         * @brief Propagate gradients from coefficients/times to waypoints and boundaries (reference overload).
         *
         * @param partialGradByCoeffs Input partial gradient ∂L/∂C, size (4N) × DIM.
         * @param partialGradByTimes  Input partial gradient ∂L/∂T, size N.
         * @param[out] grads Overwritten physical gradients; correctly sized buffers are reused
         */
        void backward(const CoefficientMatrix &partialGradByCoeffs,
                      const Eigen::VectorXd &partialGradByTimes,
                      Gradients &grads)
        {
            backwardImpl(partialGradByCoeffs, partialGradByTimes, grads);
        }

    private:
        // Construction: validate, prepare caches, solve knot derivatives, publish coefficients.
        /** @brief Rebuild from physical data; failed construction clears the published polynomial.
         * @note Requires exclusive access. Storage is reused when the topology is unchanged. */
        void updateSplineInternal()
        {
            // Validate before allocating or modifying the numerical workspaces.
            if (!trajectory_.template beginSpline<kMinDerivativeOrder>(
                    time_segments_, spatial_points_, boundary_, start_time_, kCoefficientCount))
            {
                num_segments_ = 0;
                return;
            }
            num_segments_ = static_cast<int>(time_segments_.size());

            prepareWorkspace();
            precomputeTimePowers();
            precomputeSlopes();

            // Solve the free knot velocities, then publish finite polynomial coefficients.
            solveInternalDerivatives();
            writeCoefficients();
            trajectory_.finishSpline();
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

                for (int coordinate = 0; coordinate < DIM; ++coordinate)
                {
                    double rhs = 3.0 * (duration_ratio * segment_slopes_(knot - 1, coordinate) +
                                          segment_slopes_(knot, coordinate));
                    // The preceding row is either clamped or already forward-substituted.
                    rhs -= duration_ratio * derivatives_(knot - 1, coordinate);
                    if (knot + 1 == num_segments_)
                        rhs -= derivatives_(num_segments_, coordinate);
                    derivatives_(knot, coordinate) = inverse * rhs;
                }
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

        // Adjoint: pull back coefficients, reuse factors, accumulate physical gradients.
        /** @brief Pull back the Hermite coefficient map through the cached velocity solve.
         * @note With A(T)q = b(T,P,boundary), solve A^T lambda = dL/dq and accumulate
         *       lambda^T (db/dtheta - dA/dtheta*q) plus direct coefficient-map partials.
         *       Input duration partials hold local power coefficients fixed. Outputs are
         *       overwritten; correctly sized output and adjoint storage are reused.
         * @param coefficient_partials Physical coefficient partials, in piece-major ascending order.
         * @param duration_partials Direct duration partials holding coefficients fixed.
         * @param[out] gradient Overwritten total gradients in the caller's coordinates. */
        void backwardImpl(const CoefficientMatrix &coefficient_partials,
                          const Eigen::VectorXd &duration_partials,
                          Gradients &gradient)
        {
            detail::prepareAdjoint(trajectory_, coefficient_partials, duration_partials, gradient, adjoint_);

            // Local Hermite map, cached velocity solve, then endpoint corrections.
            pullbackCoefficients(coefficient_partials, gradient);
            solveAdjoint(gradient);
            // A fully clamped segment has no free knot derivatives to correct.
            if (num_segments_ > 1)
                accumulateAdjoint(gradient);
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
     *       Numeric construction failure clears the polynomial; inspect isValid(). */
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
         * @note Inputs are copied; invalid input or numeric output leaves an invalid polynomial. */
        QuinticSplineND(const std::vector<double> &t_points,
                        const CoefficientMatrix &spatial_points,
                        const BoundaryConditions<DIM> &boundary = BoundaryConditions<DIM>())
            : spatial_points_(spatial_points),
              boundary_(boundary)
        {
            detail::durationsFromKnots(t_points, time_segments_, start_time_);
            updateSplineInternal();
        }

        /** @brief Construct from N positive durations, N+1 waypoint rows and a common time origin.
         * @param time_segments Durations in seconds. @param spatial_points Physical waypoint rows.
         * @param start_time Absolute time origin in seconds.
         * @param boundary Clamped physical endpoint derivatives in the caller's frame.
         * @note Inputs are copied; invalid input or numeric output leaves an invalid polynomial. */
        QuinticSplineND(const std::vector<double> &time_segments,
                        const CoefficientMatrix &spatial_points,
                        double start_time,
                        const BoundaryConditions<DIM> &boundary = BoundaryConditions<DIM>())
            : time_segments_(time_segments),
              start_time_(start_time),
              spatial_points_(spatial_points),
              boundary_(boundary)
        {
            updateSplineInternal();
        }

        /** @brief Replace interpolation data using absolute knots; invalid input clears the polynomial.
         * @param t_points Absolute times in seconds. @param spatial_points Physical waypoint rows.
         * @param boundary Clamped physical endpoint derivatives in the caller's frame.
         * @note Copies inputs, reuses fixed-topology storage and invalidates all borrowed views. */
        void update(const std::vector<double> &t_points,
                    const CoefficientMatrix &spatial_points,
                    const BoundaryConditions<DIM> &boundary = BoundaryConditions<DIM>())
        {
            spatial_points_ = spatial_points;
            boundary_ = boundary;
            detail::durationsFromKnots(t_points, time_segments_, start_time_);
            updateSplineInternal();
        }

        /** @brief Replace interpolation data using positive durations and a common time origin.
         * @param time_segments Durations in seconds. @param spatial_points Physical waypoint rows.
         * @param start_time Absolute time origin in seconds.
         * @param boundary Clamped physical endpoint derivatives in the caller's frame.
         * @note Copies inputs, reuses fixed-topology storage and invalidates all borrowed views.
         *       Invalid shapes, times, boundary data or numeric output clear the polynomial. */
        void update(const std::vector<double> &time_segments,
                    const CoefficientMatrix &spatial_points,
                    double start_time,
                    const BoundaryConditions<DIM> &boundary = BoundaryConditions<DIM>())
        {
            time_segments_ = time_segments;
            spatial_points_ = spatial_points;
            boundary_ = boundary;
            start_time_ = start_time;
            updateSplineInternal();
        }

        /** @brief Inspect the published polynomial. @return True only for a valid finite trajectory. */
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
         * @return Energy in physical units; NaN when the spline is invalid. */
        double energy() const
        {
            if (!trajectory_.isValid())
                return std::numeric_limits<double>::quiet_NaN();

            double total_energy = 0.0;
            for (int i = 0; i < num_segments_; ++i)
            {
                const double T = time_segments_[i];
                if (T <= 0)
                    continue;

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
         */
        void energyCoefficientPartials(CoefficientMatrix &gdC) const
        {
            if (!trajectory_.isValid()) throw std::logic_error("Energy gradients require a valid spline");
            gdC.resize(num_segments_ * 6, DIM);
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
            CoefficientMatrix gdC;
            energyCoefficientPartials(gdC);
            return gdC;
        }

        /**
         * @brief Compute partial gradient of energy w.r.t. segment durations.
         *
         * @param[out] gdT Overwritten partial gradient ∂E/∂T, size num_segments.
         *         Element i is the direct partial derivative w.r.t. T[i].
         * @note Local physical-time coefficients are held fixed; spline interpolation constraints are not differentiated.
         */
        void energyDurationPartials(Eigen::VectorXd &gdT) const
        {
            if (!trajectory_.isValid()) throw std::logic_error("Energy gradients require a valid spline");
            gdT.resize(num_segments_);

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
            Eigen::VectorXd gdT;
            energyDurationPartials(gdT);
            return gdT;
        }

        /**
         * @brief Compute full gradient of energy w.r.t. segment durations.
         *
         * @param[out] grad Full gradient dE/dT, size num_segments.
         *         Includes both direct and indirect dependencies via chain rule.
         */
        void energyDurationGradient(Eigen::VectorXd &grad) const
        {
            if (!trajectory_.isValid()) throw std::logic_error("Energy gradients require a valid spline");
            grad.resize(num_segments_);

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

                grad(i) = -term_jerk + term_sa - term_cv;
            }
        }

        /** @brief Allocate total energy duration gradients. @return Gradients including interpolation dependence. */
        Eigen::VectorXd energyDurationGradient() const
        {
            Eigen::VectorXd grad;
            energyDurationGradient(grad);
            return grad;
        }

        /**
         * @brief Compute full gradient of energy w.r.t. inner waypoints only.
         * @param[out] grad Full gradient dE/dP for inner points only.
         *         Size (N-1) × DIM (excludes boundary points P0 and PN).
         */
        void energyWaypointGradient(CoefficientMatrix &grad) const
        {
            if (!trajectory_.isValid()) throw std::logic_error("Energy gradients require a valid spline");
            if (num_segments_ < 1)
            {
                grad.resize(0, DIM);
                return;
            }

            int num_rows = std::max(0, num_segments_ - 1);
            grad.resize(num_rows, DIM);

            for (int i = 1; i < num_segments_; ++i)
            {
                const RowVector c5_L = trajectory_.coefficients_.row((i - 1) * 6 + 5);
                const RowVector c5_R = trajectory_.coefficients_.row(i * 6 + 5);
                grad.row(i - 1) = 240.0 * (c5_L - c5_R);
            }
        }

        /** @brief Allocate total energy waypoint gradients. @return Interior physical position gradients. */
        CoefficientMatrix energyWaypointGradient() const
        {
            CoefficientMatrix grad;
            energyWaypointGradient(grad);
            return grad;
        }

        /**
         * @brief Compute partial gradient of energy w.r.t. boundary conditions (Pos, Vel, Acc).
         * @return BoundaryGradients containing gradients.
         */
        BoundaryGradients energyBoundaryGradient() const
        {
            if (!trajectory_.isValid()) throw std::logic_error("Energy gradients require a valid spline");
            BoundaryGradients res;
            if (num_segments_ < 1)
                return res;

            const RowVector c3_first = trajectory_.coefficients_.row(3);
            const RowVector c4_first = trajectory_.coefficients_.row(4);
            const RowVector c5_first = trajectory_.coefficients_.row(5);

            res.start.p = -240.0 * c5_first.transpose();
            res.start.v = 48.0 * c4_first.transpose();
            res.start.a = -12.0 * c3_first.transpose();

            int last_idx = num_segments_ - 1;
            double T = time_segments_[last_idx];
            double T2 = T * T;

            const RowVector c3_last = trajectory_.coefficients_.row(last_idx * 6 + 3);
            const RowVector c4_last = trajectory_.coefficients_.row(last_idx * 6 + 4);
            const RowVector c5_last = trajectory_.coefficients_.row(last_idx * 6 + 5);

            RowVector jerk_end = 6.0 * c3_last + 24.0 * c4_last * T + 60.0 * c5_last * T2;
            RowVector snap_end = 24.0 * c4_last + 120.0 * c5_last * T;
            RowVector crackle_end = 120.0 * c5_last;

            res.end.p = 2.0 * crackle_end.transpose();
            res.end.v = -2.0 * snap_end.transpose();
            res.end.a = 2.0 * jerk_end.transpose();

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
            energyGradient(res);
            return res;
        }

        /**
         * @brief Compute full energy gradient combining time, inner point, and boundary gradients (reference overload).
         *
         * @param[out] grads Overwritten physical gradients; correctly sized buffers are reused
         */
        void energyGradient(Gradients &grads) const
        {
            energyWaypointGradient(grads.inner_points);
            energyDurationGradient(grads.durations);
            BoundaryGradients boundary = energyBoundaryGradient();
            grads.start = boundary.start;
            grads.end = boundary.end;
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

            backwardImpl(partialGradByCoeffs, partialGradByTimes, res);

            return res;
        }

        /**
         * @brief Propagate gradients from coefficients/times to waypoints and boundaries (reference overload).
         *
         * @param partialGradByCoeffs Input partial gradient ∂L/∂C, size (6N) × DIM.
         * @param partialGradByTimes  Input partial gradient ∂L/∂T, size N.
         * @param[out] grads Overwritten physical gradients; correctly sized buffers are reused
         */
        void backward(const CoefficientMatrix &partialGradByCoeffs,
                      const Eigen::VectorXd &partialGradByTimes,
                      Gradients &grads)
        {
            backwardImpl(partialGradByCoeffs, partialGradByTimes, grads);
        }

    private:
        // Construction: validate, prepare caches, solve knot derivatives, publish coefficients.
        /** @brief Rebuild from physical data; failed construction clears the published polynomial.
         * @note Requires exclusive access. Storage is reused when the topology is unchanged. */
        void updateSplineInternal()
        {
            // Validate before allocating or modifying the numerical workspaces.
            if (!trajectory_.template beginSpline<kMinDerivativeOrder>(
                    time_segments_, spatial_points_, boundary_, start_time_, kCoefficientCount))
            {
                num_segments_ = 0;
                return;
            }
            num_segments_ = static_cast<int>(time_segments_.size());

            prepareWorkspace();
            precomputeTimePowers();

            // Assemble and solve knot stationarity, then publish finite coefficients only.
            if (!solveInternalDerivatives())
            {
                trajectory_.clear();
                return;
            }
            writeCoefficients();
            trajectory_.finishSpline();
        }

        /** @brief Reuse topology-dependent storage and refresh the clamped endpoint derivatives.
         * @note Requires validated input. Same-topology updates skip all cache resizing;
         *       the first adjoint reuses storage prepared during construction. */
        void prepareWorkspace()
        {
            if (time_powers_.size() != time_segments_.size())
                resizeWorkspace();

            // Boundary values change independently of the segment count.
            if (num_segments_ == 1)
                return;
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
            // A single segment is fully specified by its boundary states.
            if (num_segments_ == 1)
            {
                derivatives_.resize(0, DIM);
                factors_.clear();
            }
            else
            {
                derivatives_.resize((num_segments_ + 1) * kDerivativesPerKnot, DIM);
                factors_.resize(num_segments_ - 1);
            }
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

            for (int coordinate = 0; coordinate < DIM; ++coordinate)
            {
                const double delta_left = spatial_points_(knot, coordinate) - spatial_points_(knot - 1, coordinate);
                const double delta_right = spatial_points_(knot + 1, coordinate) - spatial_points_(knot, coordinate);
                double rhs0 = 360.0 * left.h4_inv * delta_left + 360.0 * right.h4_inv * delta_right;
                double rhs1 = -60.0 * left.h3_inv * delta_left + 60.0 * right.h3_inv * delta_right;
                if (knot == 1)
                {
                    rhs0 -= previous(0, 0) * derivatives_(0, coordinate) + previous(0, 1) * derivatives_(1, coordinate);
                    rhs1 -= previous(1, 0) * derivatives_(0, coordinate) + previous(1, 1) * derivatives_(1, coordinate);
                }
                if (knot + 1 == num_segments_)
                {
                    rhs0 -= 168.0 * right.h3_inv * derivatives_(num_segments_ * 2, coordinate)
                        - 24.0 * right.h2_inv * derivatives_(num_segments_ * 2 + 1, coordinate);
                    rhs1 -= 24.0 * right.h2_inv * derivatives_(num_segments_ * 2, coordinate)
                        - 3.0 * right.h_inv * derivatives_(num_segments_ * 2 + 1, coordinate);
                }
                derivatives_(knot * 2, coordinate) = rhs0;
                derivatives_(knot * 2 + 1, coordinate) = rhs1;
            }
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
                // Single-segment curves have no free knot derivatives or primal workspace.
                const RowVector velocity = num_segments_ == 1 ? RowVector(boundary_.start_velocity.transpose()) :
                    RowVector(derivatives_.row(left_row));
                const RowVector acceleration = num_segments_ == 1 ? RowVector(boundary_.start_acceleration.transpose()) :
                    RowVector(derivatives_.row(left_row + 1));
                const RowVector end_velocity = num_segments_ == 1 ? RowVector(boundary_.end_velocity.transpose()) :
                    RowVector(derivatives_.row(right_row));
                const RowVector end_acceleration = num_segments_ == 1 ? RowVector(boundary_.end_acceleration.transpose()) :
                    RowVector(derivatives_.row(right_row + 1));
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

        // Adjoint: pull back coefficients, reuse factors, accumulate physical gradients.
        /** @brief Pull back one objective through coefficient generation and knot stationarity.
         * @param coefficient_partials Physical coefficient partials in piece-major ascending order.
         * @param duration_partials Direct duration partials with coefficients held fixed, in seconds.
         * @param[out] gradient Overwritten total physical gradients; prepared storage is reused.
         * @note Requires a valid spline, matching shapes and exclusive access to this object. */
        void backwardImpl(const CoefficientMatrix &coefficient_partials,
                          const Eigen::VectorXd &duration_partials,
                          Gradients &gradient)
        {
            detail::prepareAdjoint(trajectory_, coefficient_partials, duration_partials, gradient, adjoint_);

            // Local map, cached linear solve, then endpoint corrections.
            pullbackCoefficients(coefficient_partials, gradient);
            solveAdjoint(gradient);
            // A fully clamped segment has no free knot derivatives to correct.
            if (num_segments_ > 1)
                accumulateAdjoint(gradient);
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
                const RowVector left_velocity = num_segments_ == 1 ? RowVector(boundary_.start_velocity.transpose()) :
                    RowVector(derivatives_.row(left_row));
                const RowVector left_acceleration = num_segments_ == 1 ? RowVector(boundary_.start_acceleration.transpose()) :
                    RowVector(derivatives_.row(left_row + 1));
                const RowVector right_velocity = num_segments_ == 1 ? RowVector(boundary_.end_velocity.transpose()) :
                    RowVector(derivatives_.row(right_row));
                const RowVector right_acceleration = num_segments_ == 1 ? RowVector(boundary_.end_acceleration.transpose()) :
                    RowVector(derivatives_.row(right_row + 1));

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

            for (int knot = 1; knot < num_segments_; ++knot)
                factors_[knot - 1].forward(adjoint_, knot * kDerivativesPerKnot);
            backSubstitute(adjoint_);
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
     *       Numeric construction failure clears the polynomial; inspect isValid(). */
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
         * @note Inputs are copied; invalid input or numeric output leaves an invalid polynomial. */
        SepticSplineND(const std::vector<double> &t_points,
                       const CoefficientMatrix &spatial_points,
                       const BoundaryConditions<DIM> &boundary = BoundaryConditions<DIM>())
            : spatial_points_(spatial_points),
              boundary_(boundary)
        {
            detail::durationsFromKnots(t_points, time_segments_, start_time_);
            updateSplineInternal();
        }

        /** @brief Construct from N positive durations, N+1 waypoint rows and a common time origin.
         * @param time_segments Durations in seconds. @param spatial_points Physical waypoint rows.
         * @param start_time Absolute time origin in seconds.
         * @param boundary Clamped physical endpoint derivatives in the caller's frame.
         * @note Inputs are copied; invalid input or numeric output leaves an invalid polynomial. */
        SepticSplineND(const std::vector<double> &time_segments,
                       const CoefficientMatrix &spatial_points,
                       double start_time,
                       const BoundaryConditions<DIM> &boundary = BoundaryConditions<DIM>())
            : time_segments_(time_segments),
              start_time_(start_time),
              spatial_points_(spatial_points),
              boundary_(boundary)
        {
            updateSplineInternal();
        }

        /** @brief Replace interpolation data using absolute knots; invalid input clears the polynomial.
         * @param t_points Absolute times in seconds. @param spatial_points Physical waypoint rows.
         * @param boundary Clamped physical endpoint derivatives in the caller's frame.
         * @note Copies inputs, reuses fixed-topology storage and invalidates all borrowed views. */
        void update(const std::vector<double> &t_points,
                    const CoefficientMatrix &spatial_points,
                    const BoundaryConditions<DIM> &boundary = BoundaryConditions<DIM>())
        {
            spatial_points_ = spatial_points;
            boundary_ = boundary;
            detail::durationsFromKnots(t_points, time_segments_, start_time_);
            updateSplineInternal();
        }

        /** @brief Replace interpolation data using positive durations and a common time origin.
         * @param time_segments Durations in seconds. @param spatial_points Physical waypoint rows.
         * @param start_time Absolute time origin in seconds.
         * @param boundary Clamped physical endpoint derivatives in the caller's frame.
         * @note Copies inputs, reuses fixed-topology storage and invalidates all borrowed views.
         *       Invalid shapes, times, boundary data or numeric output clear the polynomial. */
        void update(const std::vector<double> &time_segments,
                    const CoefficientMatrix &spatial_points,
                    double start_time,
                    const BoundaryConditions<DIM> &boundary = BoundaryConditions<DIM>())
        {
            time_segments_ = time_segments;
            spatial_points_ = spatial_points;
            boundary_ = boundary;
            start_time_ = start_time;
            updateSplineInternal();
        }

        /** @brief Inspect the published polynomial. @return True only for a valid finite trajectory. */
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
         * @return Energy in physical units; NaN when the spline is invalid. */
        double energy() const
        {
            if (!trajectory_.isValid())
                return std::numeric_limits<double>::quiet_NaN();

            double total_energy = 0.0;
            for (int i = 0; i < num_segments_; ++i)
            {
                const double T = time_segments_[i];
                if (T <= 0)
                    continue;

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
         */
        void energyCoefficientPartials(CoefficientMatrix &gdC) const
        {
            if (!trajectory_.isValid()) throw std::logic_error("Energy gradients require a valid spline");
            gdC.resize(num_segments_ * 8, DIM);
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
            CoefficientMatrix gdC;
            energyCoefficientPartials(gdC);
            return gdC;
        }

        /**
         * @brief Compute partial gradient of energy w.r.t. segment durations.
         *
         * @param[out] gdT Overwritten partial gradient ∂E/∂T, size num_segments.
         *         Element i is the direct partial derivative w.r.t. T[i].
         * @note Local physical-time coefficients are held fixed; spline interpolation constraints are not differentiated.
         */
        void energyDurationPartials(Eigen::VectorXd &gdT) const
        {
            if (!trajectory_.isValid()) throw std::logic_error("Energy gradients require a valid spline");
            gdT.resize(num_segments_);

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
            Eigen::VectorXd gdT;
            energyDurationPartials(gdT);
            return gdT;
        }

        /**
         * @brief Compute full gradient of energy w.r.t. segment durations.
         *
         * @param[out] grad Full gradient dE/dT, size num_segments.
         *         Includes both direct and indirect dependencies via chain rule.
         */
        void energyDurationGradient(Eigen::VectorXd &grad) const
        {
            if (!trajectory_.isValid()) throw std::logic_error("Energy gradients require a valid spline");
            grad.resize(num_segments_);

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

                grad(i) = -term_snap + term_cj - term_pa + term_dv;
            }
        }

        /** @brief Allocate total energy duration gradients. @return Gradients including interpolation dependence. */
        Eigen::VectorXd energyDurationGradient() const
        {
            Eigen::VectorXd grad;
            energyDurationGradient(grad);
            return grad;
        }

        /**
         * @brief Compute full gradient of energy w.r.t. inner waypoints only.
         * @param[out] grad Full gradient dE/dP for inner points only.
         *         Size (N-1) × DIM (excludes boundary points P0 and PN).
         */
        void energyWaypointGradient(CoefficientMatrix &grad) const
        {
            if (!trajectory_.isValid()) throw std::logic_error("Energy gradients require a valid spline");
            if (num_segments_ < 1)
            {
                grad.resize(0, DIM);
                return;
            }

            int num_rows = std::max(0, num_segments_ - 1);
            grad.resize(num_rows, DIM);

            for (int i = 1; i < num_segments_; ++i)
            {
                const RowVector c7_L = trajectory_.coefficients_.row((i - 1) * 8 + 7);
                const RowVector c7_R = trajectory_.coefficients_.row(i * 8 + 7);
                grad.row(i - 1) = 10080.0 * (c7_R - c7_L);
            }
        }

        /** @brief Allocate total energy waypoint gradients. @return Interior physical position gradients. */
        CoefficientMatrix energyWaypointGradient() const
        {
            CoefficientMatrix grad;
            energyWaypointGradient(grad);
            return grad;
        }

        /**
         * @brief Compute partial gradient of energy w.r.t. boundary conditions (Pos, Vel, Acc, Jerk).
         * @return BoundaryGradients containing gradients.
         */
        BoundaryGradients energyBoundaryGradient() const
        {
            if (!trajectory_.isValid()) throw std::logic_error("Energy gradients require a valid spline");
            BoundaryGradients res;
            if (num_segments_ < 1)
                return res;

            const RowVector c4_first = trajectory_.coefficients_.row(4);
            const RowVector c5_first = trajectory_.coefficients_.row(5);
            const RowVector c6_first = trajectory_.coefficients_.row(6);
            const RowVector c7_first = trajectory_.coefficients_.row(7);

            res.start.p = 10080.0 * c7_first.transpose();
            res.start.v = -1440.0 * c6_first.transpose();
            res.start.a = 240.0 * c5_first.transpose();
            res.start.j = -48.0 * c4_first.transpose();

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

            res.end.p = -2.0 * d7_end.transpose();
            res.end.v = 2.0 * pop_end.transpose();
            res.end.a = -2.0 * crackle_end.transpose();
            res.end.j = 2.0 * snap_end.transpose();

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
            energyGradient(res);
            return res;
        }

        /**
         * @brief Compute full energy gradient combining time, inner point, and boundary gradients (reference overload).
         *
         * @param[out] grads Overwritten physical gradients; correctly sized buffers are reused
         */
        void energyGradient(Gradients &grads) const
        {
            energyWaypointGradient(grads.inner_points);
            energyDurationGradient(grads.durations);
            BoundaryGradients boundary = energyBoundaryGradient();
            grads.start = boundary.start;
            grads.end = boundary.end;
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

            backwardImpl(partialGradByCoeffs, partialGradByTimes, res);

            return res;
        }

        /**
         * @brief Propagate gradients from coefficients/times to waypoints and boundaries (reference overload).
         *
         * @param partialGradByCoeffs Input partial gradient ∂L/∂C, size (8N) × DIM.
         * @param partialGradByTimes  Input partial gradient ∂L/∂T, size N.
         * @param[out] grads Overwritten physical gradients; correctly sized buffers are reused
         */
        void backward(const CoefficientMatrix &partialGradByCoeffs,
                      const Eigen::VectorXd &partialGradByTimes,
                      Gradients &grads)
        {
            backwardImpl(partialGradByCoeffs, partialGradByTimes, grads);
        }

    private:
        // Construction: validate, prepare caches, solve knot derivatives, publish coefficients.
        /** @brief Rebuild from physical data; failed construction clears the published polynomial.
         * @note Requires exclusive access. Storage is reused when the topology is unchanged. */
        void updateSplineInternal()
        {
            // Validate before allocating or modifying the numerical workspaces.
            if (!trajectory_.template beginSpline<kMinDerivativeOrder>(
                    time_segments_, spatial_points_, boundary_, start_time_, kCoefficientCount))
            {
                num_segments_ = 0;
                return;
            }
            num_segments_ = static_cast<int>(time_segments_.size());

            prepareWorkspace();
            precomputeTimePowers();

            // Assemble and solve knot stationarity, then publish finite coefficients only.
            if (!solveInternalDerivatives())
            {
                trajectory_.clear();
                return;
            }
            writeCoefficients();
            trajectory_.finishSpline();
        }

        /** @brief Reuse topology-dependent storage and refresh the clamped endpoint derivatives.
         * @note Requires validated input. Same-topology updates skip all cache resizing;
         *       the first adjoint reuses storage prepared during construction. */
        void prepareWorkspace()
        {
            if (time_powers_.size() != time_segments_.size())
                resizeWorkspace();

            // Boundary values change independently of the segment count.
            if (num_segments_ == 1)
                return;
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
            // A single segment is fully specified by its boundary states.
            if (num_segments_ == 1)
            {
                derivatives_.resize(0, DIM);
                factors_.clear();
            }
            else
            {
                derivatives_.resize((num_segments_ + 1) * kDerivativesPerKnot, DIM);
                factors_.resize(num_segments_ - 1);
            }
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

            for (int coordinate = 0; coordinate < DIM; ++coordinate)
            {
                const double delta_left = spatial_points_(knot, coordinate) - spatial_points_(knot - 1, coordinate);
                const double delta_right = spatial_points_(knot + 1, coordinate) - spatial_points_(knot, coordinate);
                double rhs0 = 50400.0 * left.h6_inv * delta_left + 50400.0 * right.h6_inv * delta_right;
                double rhs1 = -10080.0 * left.h5_inv * delta_left + 10080.0 * right.h5_inv * delta_right;
                double rhs2 = 840.0 * left.h4_inv * delta_left + 840.0 * right.h4_inv * delta_right;
                if (knot == 1)
                {
                    rhs0 -= previous(0, 0) * derivatives_(0, coordinate)
                        + previous(0, 1) * derivatives_(1, coordinate)
                        + previous(0, 2) * derivatives_(2, coordinate);
                    rhs1 -= previous(1, 0) * derivatives_(0, coordinate)
                        + previous(1, 1) * derivatives_(1, coordinate)
                        + previous(1, 2) * derivatives_(2, coordinate);
                    rhs2 -= previous(2, 0) * derivatives_(0, coordinate)
                        + previous(2, 1) * derivatives_(1, coordinate)
                        + previous(2, 2) * derivatives_(2, coordinate);
                }
                if (knot + 1 == num_segments_)
                {
                    rhs0 -= 24480.0 * right.h5_inv * derivatives_(num_segments_ * 3, coordinate)
                        - 4680.0 * right.h4_inv * derivatives_(num_segments_ * 3 + 1, coordinate)
                        + 360.0 * right.h3_inv * derivatives_(num_segments_ * 3 + 2, coordinate);
                    rhs1 -= 4680.0 * right.h4_inv * derivatives_(num_segments_ * 3, coordinate)
                        - 840.0 * right.h3_inv * derivatives_(num_segments_ * 3 + 1, coordinate)
                        + 60.0 * right.h2_inv * derivatives_(num_segments_ * 3 + 2, coordinate);
                    rhs2 -= 360.0 * right.h3_inv * derivatives_(num_segments_ * 3, coordinate)
                        - 60.0 * right.h2_inv * derivatives_(num_segments_ * 3 + 1, coordinate)
                        + 4.0 * right.h_inv * derivatives_(num_segments_ * 3 + 2, coordinate);
                }
                derivatives_(knot * 3, coordinate) = rhs0;
                derivatives_(knot * 3 + 1, coordinate) = rhs1;
                derivatives_(knot * 3 + 2, coordinate) = rhs2;
            }
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

                // Single-segment curves have no free knot derivatives or primal workspace.
                const RowVector velocity = num_segments_ == 1 ? RowVector(boundary_.start_velocity.transpose()) :
                    RowVector(derivatives_.row(left_row));
                const RowVector acceleration = num_segments_ == 1 ? RowVector(boundary_.start_acceleration.transpose()) :
                    RowVector(derivatives_.row(left_row + 1));
                const RowVector jerk = num_segments_ == 1 ? RowVector(boundary_.start_jerk.transpose()) :
                    RowVector(derivatives_.row(left_row + 2));
                const RowVector end_velocity = num_segments_ == 1 ? RowVector(boundary_.end_velocity.transpose()) :
                    RowVector(derivatives_.row(right_row));
                const RowVector end_acceleration = num_segments_ == 1 ? RowVector(boundary_.end_acceleration.transpose()) :
                    RowVector(derivatives_.row(right_row + 1));
                const RowVector end_jerk = num_segments_ == 1 ? RowVector(boundary_.end_jerk.transpose()) :
                    RowVector(derivatives_.row(right_row + 2));
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

        // Adjoint: pull back coefficients, reuse factors, accumulate physical gradients.
        /** @brief Pull back one objective through coefficient generation and knot stationarity.
         * @param coefficient_partials Physical coefficient partials in piece-major ascending order.
         * @param duration_partials Direct duration partials with coefficients held fixed, in seconds.
         * @param[out] gradient Overwritten total physical gradients; prepared storage is reused.
         * @note Requires a valid spline, matching shapes and exclusive access to this object. */
        void backwardImpl(const CoefficientMatrix &coefficient_partials,
                          const Eigen::VectorXd &duration_partials,
                          Gradients &gradient)
        {
            detail::prepareAdjoint(trajectory_, coefficient_partials, duration_partials, gradient, adjoint_);

            // Local map, cached linear solve, then endpoint corrections.
            pullbackCoefficients(coefficient_partials, gradient);
            solveAdjoint(gradient);
            // A fully clamped segment has no free knot derivatives to correct.
            if (num_segments_ > 1)
                accumulateAdjoint(gradient);
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

                // Borrow contiguous derivative rows; a fully clamped segment reads its boundary inputs.
                const Eigen::Map<const RowVector> left_velocity(num_segments_ == 1 ? boundary_.start_velocity.data() :
                    derivatives_.row(left_row).data());
                const Eigen::Map<const RowVector> left_acceleration(num_segments_ == 1 ? boundary_.start_acceleration.data() :
                    derivatives_.row(left_row + 1).data());
                const Eigen::Map<const RowVector> left_jerk(num_segments_ == 1 ? boundary_.start_jerk.data() :
                    derivatives_.row(left_row + 2).data());
                const Eigen::Map<const RowVector> right_velocity(num_segments_ == 1 ? boundary_.end_velocity.data() :
                    derivatives_.row(right_row).data());
                const Eigen::Map<const RowVector> right_acceleration(num_segments_ == 1 ? boundary_.end_acceleration.data() :
                    derivatives_.row(right_row + 1).data());
                const Eigen::Map<const RowVector> right_jerk(num_segments_ == 1 ? boundary_.end_jerk.data() :
                    derivatives_.row(right_row + 2).data());

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

            for (int knot = 1; knot < num_segments_; ++knot)
                factors_[knot - 1].forward(adjoint_, knot * kDerivativesPerKnot);
            backSubstitute(adjoint_);
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
