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
 * @brief Owning piecewise polynomials, independent samplers and specialized minimum-derivative splines. */

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

    /** @brief Minimum-acceleration interpolation with a specialized O(N) scalar tridiagonal solver.
     * @note Minimize the squared 2-th physical derivative integral with fixed waypoint
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
        BoundaryConditions<DIM> boundary_velocities_;
        int num_segments_;
        double start_time_;
        Polynomial trajectory_;

        CoefficientMatrix internal_derivatives_;
        CoefficientMatrix point_diffs_;
        CoefficientMatrix scaled_point_diffs_;
        Eigen::VectorXd cached_c_prime_;
        Eigen::VectorXd cached_inv_denoms_;
        CoefficientMatrix ws_lambda_;

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

        CubicSplineND() : num_segments_(0), start_time_(0.0) {}

        /** @brief Construct from N+1 strictly increasing finite knots and N+1 waypoint rows.
         * @note Inputs are copied; invalid input or numeric output leaves an invalid polynomial. */
        CubicSplineND(const std::vector<double> &t_points,
                      const CoefficientMatrix &spatial_points,
                      const BoundaryConditions<DIM> &boundary_velocities = BoundaryConditions<DIM>())
            : spatial_points_(spatial_points), boundary_velocities_(boundary_velocities)
        {
            detail::durationsFromKnots(t_points, time_segments_, start_time_);
            updateSplineInternal();
        }

        /** @brief Construct from N positive durations, N+1 waypoint rows and a common time origin.
         * @note Inputs are copied; invalid input or numeric output leaves an invalid polynomial. */
        CubicSplineND(const std::vector<double> &time_segments,
                      const CoefficientMatrix &spatial_points,
                      double start_time,
                      const BoundaryConditions<DIM> &boundary_velocities = BoundaryConditions<DIM>())
            : time_segments_(time_segments), spatial_points_(spatial_points), boundary_velocities_(boundary_velocities),
              start_time_(start_time)
        {
            updateSplineInternal();
        }

        /** @brief Replace interpolation data using absolute knots; invalid input clears the polynomial.
         * @note Copies inputs, reuses fixed-topology storage and invalidates all borrowed views. */
        void update(const std::vector<double> &t_points,
                    const CoefficientMatrix &spatial_points,
                    const BoundaryConditions<DIM> &boundary_velocities = BoundaryConditions<DIM>())
        {
            spatial_points_ = spatial_points;
            boundary_velocities_ = boundary_velocities;
            detail::durationsFromKnots(t_points, time_segments_, start_time_);
            updateSplineInternal();
        }

        /** @brief Replace interpolation data using positive durations and a common time origin.
         * @note Copies inputs, reuses fixed-topology storage and invalidates all borrowed views.
         *       Invalid shapes, times, boundary data or numeric output clear the polynomial. */
        void update(const std::vector<double> &time_segments,
                    const CoefficientMatrix &spatial_points,
                    double start_time,
                    const BoundaryConditions<DIM> &boundary_velocities = BoundaryConditions<DIM>())
        {
            time_segments_ = time_segments;
            spatial_points_ = spatial_points;
            boundary_velocities_ = boundary_velocities;
            start_time_ = start_time;
            updateSplineInternal();
        }

        bool isValid() const { return trajectory_.isValid(); }
        int dimension() const { return DIM; }

        double startTime() const
        {
            return trajectory_.startTime();
        }

        double endTime() const
        {
            return trajectory_.endTime();
        }

        double duration() const
        {
            return trajectory_.duration();
        }

        size_t numPoints() const
        {
            return static_cast<size_t>(spatial_points_.rows());
        }

        int numSegments() const
        {
            return trajectory_.numSegments();
        }

        const CoefficientMatrix &waypoints() const { return spatial_points_; }
        const std::vector<double> &durations() const { return time_segments_; }
        const std::vector<double> &breakpoints() const { return trajectory_.breakpoints_; }
        const BoundaryConditions<DIM> &boundary() const { return boundary_velocities_; }

        /** @brief Borrow the polynomial until the next update, assignment or move. */
        const Polynomial &polynomial() const & { return trajectory_; }
        const Polynomial &polynomial() const && = delete;
        /** @brief Make an independent owning snapshot, allocating its data. */
        Polynomial copyPolynomial() const { return trajectory_; }

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

        /** @brief Integral of the squared minimum derivative; returns NaN for an invalid spline. */
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

        Eigen::VectorXd energyDurationPartials() const
        {
            Eigen::VectorXd gdT;
            energyDurationPartials(gdT);
            return gdT;
        }

        /**
         * @brief Compute full gradient of energy w.r.t. segment durations.
         *
         * @return VectorXd Full gradient dE/dT, size num_segments.
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

        Eigen::VectorXd energyDurationGradient() const
        {
            Eigen::VectorXd grad;
            energyDurationGradient(grad);
            return grad;
        }

        /**
         * @brief Compute full gradient of energy w.r.t. inner waypoints only.
         * @return CoefficientMatrix Full gradient dE/dP for inner points only.
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

            backwardImpl(partialGradByCoeffs, partialGradByTimes,
                                  res.inner_points, res.durations, res.start, res.end);

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
            backwardImpl(partialGradByCoeffs, partialGradByTimes,
                                  grads.inner_points, grads.durations, grads.start, grads.end);
        }

    private:
        /** @brief Pull back the coefficient map through the cached continuity solve.
         * @note With A(T)q = b(T,P,boundary), solve A^T lambda = dL/dq and accumulate
         *       lambda^T (db/dtheta - dA/dtheta*q) plus direct coefficient-map partials.
         *       Input duration partials hold local power coefficients fixed. Outputs are
         *       overwritten; correctly sized output and adjoint storage are reused. */
        void backwardImpl(const CoefficientMatrix &partialGradByCoeffs,
                                   const Eigen::VectorXd &partialGradByTimes,
                                   CoefficientMatrix &innerPointsGrad,
                                   Eigen::VectorXd &gradByTimes,
                                   BoundaryGradient &startGrads,
                                   BoundaryGradient &endGrads)
        {
            const int n = num_segments_;
            const CoefficientMatrix &M = internal_derivatives_;

            if (!trajectory_.isValid() || partialGradByCoeffs.rows() != n * kCoefficientCount ||
                partialGradByTimes.size() != n)
                throw std::invalid_argument("Adjoint requires a valid spline and matching partial shapes");
            gradByTimes = partialGradByTimes;
            startGrads = BoundaryGradient();
            endGrads = BoundaryGradient();

            if (n > 1)
            {
                innerPointsGrad.resize(n - 1, DIM);
                innerPointsGrad.setZero();
            }
            else
            {
                innerPointsGrad.resize(0, DIM);
            }

            auto add_point_grad = [&](int idx, const RowVector &grad)
            {
                if (idx == 0)
                {
                    startGrads.p += grad.transpose();
                }
                else if (idx == n)
                {
                    endGrads.p += grad.transpose();
                }
                else
                {
                    innerPointsGrad.row(idx - 1) += grad;
                }
            };

            ws_lambda_.resize(n + 1, DIM);
            ws_lambda_.setZero();

            for (int i = 0; i < n; ++i)
            {
                const auto &tp = time_powers_[i];
                const double h_i = tp.h;
                const double h_inv = tp.h_inv;
                const double h2_inv = tp.h2_inv;

                const RowVector g_c0 = partialGradByCoeffs.row(i * 4 + 0);
                const RowVector g_c1 = partialGradByCoeffs.row(i * 4 + 1);
                const RowVector g_c2 = partialGradByCoeffs.row(i * 4 + 2);
                const RowVector g_c3 = partialGradByCoeffs.row(i * 4 + 3);

                const RowVector grad_p_i = g_c0 - g_c1 * h_inv;
                const RowVector grad_p_i1 = g_c1 * h_inv;
                add_point_grad(i, grad_p_i);
                add_point_grad(i + 1, grad_p_i1);

                ws_lambda_.row(i) -= g_c1 * (h_i / 3.0);
                ws_lambda_.row(i + 1) -= g_c1 * (h_i / 6.0);
                ws_lambda_.row(i) += g_c2 * 0.5;
                ws_lambda_.row(i) -= g_c3 * (h_inv / 6.0);
                ws_lambda_.row(i + 1) += g_c3 * (h_inv / 6.0);

                const RowVector dP_row = point_diffs_.row(i);
                const RowVector term_dC1_dh = -dP_row * h2_inv - (2.0 * M.row(i) + M.row(i + 1)) / 6.0;
                const RowVector term_dC3_dh = -(M.row(i + 1) - M.row(i)) * (h2_inv / 6.0);

                gradByTimes(i) += g_c1.dot(term_dC1_dh) + g_c3.dot(term_dC3_dh);
            }

            solveWithCachedLU(ws_lambda_);

            for (int k = 0; k < n; ++k)
            {
                const auto &tp = time_powers_[k];
                double h2_inv = tp.h2_inv;
                const RowVector dP_row = point_diffs_.row(k);

                const RowVector common_term = ws_lambda_.row(k) - ws_lambda_.row(k + 1);

                const RowVector grad_R_P = 6.0 * tp.h_inv * common_term;
                add_point_grad(k + 1, grad_R_P);
                add_point_grad(k, -grad_R_P);

                const RowVector grad_R_h = -6.0 * dP_row * h2_inv;
                gradByTimes(k) += common_term.dot(grad_R_h);

                const RowVector M_k = M.row(k);
                const RowVector M_k1 = M.row(k + 1);

                const RowVector term_k = 2.0 * M_k + M_k1;
                const RowVector term_k1 = M_k + 2.0 * M_k1;

                gradByTimes(k) -= ws_lambda_.row(k).dot(term_k);
                gradByTimes(k) -= ws_lambda_.row(k + 1).dot(term_k1);
            }
            startGrads.v = -6.0 * ws_lambda_.row(0).transpose();

            endGrads.v = 6.0 * ws_lambda_.row(n).transpose();
        }

        /**
         * @brief Rebuild the interpolating spline and expose the generated polynomial's validity.
         * @note Requires validated input shapes and exclusive ownership; finite inputs can still produce
         *       invalid coefficients or time knots, in which case isValid() becomes false.
         */
        inline void updateSplineInternal()
        {
            if (!trajectory_.template beginSpline<kMinDerivativeOrder>(
                    time_segments_, spatial_points_, boundary_velocities_, start_time_, kCoefficientCount))
            { num_segments_ = 0; return; }
            num_segments_ = static_cast<int>(time_segments_.size());
            // Prepare adjoint storage with the topology so the first pullback does not allocate.
            ws_lambda_.resize(num_segments_ + 1, DIM);
            precomputeTimePowers();
            precomputePointDiffs();
            solveSplineInPlace();
            trajectory_.finishSpline();
            // A finite input can still produce invalid coefficients or collapsed breakpoints.
        }

        void precomputeTimePowers()
        {
            int n = static_cast<int>(time_segments_.size());
            time_powers_.resize(n);

            for (int i = 0; i < n; ++i)
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

        void precomputePointDiffs()
        {
            point_diffs_.resize(num_segments_, DIM);
            for (int i = 0; i < num_segments_; ++i)
            {
                point_diffs_.row(i) = spatial_points_.row(i + 1) - spatial_points_.row(i);
            }
        }

        void solveSplineInPlace()
        {
            const int n = num_segments_;

            scaled_point_diffs_.resize(n, DIM);
            for (int i = 0; i < n; ++i)
            {
                scaled_point_diffs_.row(i) =
                    point_diffs_.row(i) * time_powers_[i].h_inv;
            }

            internal_derivatives_.resize(n + 1, DIM);
            CoefficientMatrix &M = internal_derivatives_;

            if (n >= 2)
            {
                M.block(1, 0, n - 1, DIM) =
                    6.0 * (scaled_point_diffs_.bottomRows(n - 1) -
                           scaled_point_diffs_.topRows(n - 1));
            }
            M.row(0) =
                6.0 * (scaled_point_diffs_.row(0) -
                       boundary_velocities_.start_velocity.transpose());
            M.row(n) =
                6.0 * (boundary_velocities_.end_velocity.transpose() -
                       scaled_point_diffs_.row(n - 1));

            computeLUAndSolve(M);


            for (int i = 0; i < n; ++i)
            {
                const auto &tp = time_powers_[i];
                double h_i = tp.h;
                double h_inv = tp.h_inv;

                trajectory_.coefficients_.row(i * 4 + 0) = spatial_points_.row(i);

                trajectory_.coefficients_.row(i * 4 + 1) =
                    scaled_point_diffs_.row(i) -
                    (h_i / 6.0) *
                        (2.0 * M.row(i) + M.row(i + 1));

                trajectory_.coefficients_.row(i * 4 + 2) = M.row(i) * 0.5;

                trajectory_.coefficients_.row(i * 4 + 3) =
                    (M.row(i + 1) - M.row(i)) *
                    (h_inv / 6.0);
            }
        }

        template <typename MatType>
        void computeLUAndSolve(MatType &M)
        {
            const int n_seg = num_segments_;
            const int n_mat = n_seg + 1;

            cached_c_prime_.resize(n_mat - 1);
            cached_inv_denoms_.resize(n_mat);

            double main_0 = 2.0 * time_powers_[0].h;
            double inv = 1.0 / main_0;
            cached_inv_denoms_(0) = inv;

            double upper_0 = time_powers_[0].h;
            cached_c_prime_(0) = upper_0 * inv;
            M.row(0) *= inv;

            for (int i = 1; i < n_mat - 1; ++i)
            {
                double h_prev = time_powers_[i - 1].h;
                double h_curr = time_powers_[i].h;

                double main_i = 2.0 * (h_prev + h_curr);
                double lower_prev = h_prev;
                double upper_i = h_curr;

                double denom = main_i - lower_prev * cached_c_prime_(i - 1);
                double inv_d = 1.0 / denom;

                cached_inv_denoms_(i) = inv_d;
                cached_c_prime_(i) = upper_i * inv_d;

                M.row(i).noalias() -= lower_prev * M.row(i - 1);
                M.row(i) *= inv_d;
            }

            if (n_mat >= 2)
            {
                int i = n_mat - 1;
                double h_last = time_powers_[n_seg - 1].h;
                double main_last = 2.0 * h_last;
                double lower_prev = h_last;

                double denom = main_last - lower_prev * cached_c_prime_(i - 1);
                double inv_d = 1.0 / denom;
                cached_inv_denoms_(i) = inv_d;

                M.row(i).noalias() -= lower_prev * M.row(i - 1);
                M.row(i) *= inv_d;
            }

            for (int i = n_mat - 2; i >= 0; --i)
            {
                M.row(i).noalias() -= cached_c_prime_(i) * M.row(i + 1);
            }
        }

        template <typename MatType>
        void solveWithCachedLU(MatType &X)
        {
            const int n = static_cast<int>(cached_inv_denoms_.size());

            X.row(0) *= cached_inv_denoms_(0);

            for (int i = 1; i < n; ++i)
            {
                X.row(i).noalias() -= time_powers_[i - 1].h * X.row(i - 1);
                X.row(i) *= cached_inv_denoms_(i);
            }

            for (int i = n - 2; i >= 0; --i)
            {
                X.row(i).noalias() -= cached_c_prime_(i) * X.row(i + 1);
            }
        }

    };

    /** @brief Minimum-jerk interpolation with a specialized O(N) 2-by-2 block tridiagonal solver.
     * @note Minimize the squared 3-th physical derivative integral with fixed waypoint
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
        CoefficientMatrix point_diffs_;

        BoundaryConditions<DIM> boundary_;

        int num_segments_{0};

        Polynomial trajectory_;

        using BlockMatrix2x2Storage = Eigen::Matrix<double, Eigen::Dynamic, 4, Eigen::RowMajor>;
        BlockMatrix2x2Storage D_inv_cache_;
        BlockMatrix2x2Storage U_blocks_cache_;
        BlockMatrix2x2Storage L_blocks_cache_;
        BlockMatrix2x2Storage D_inv_T_mul_L_next_T_cache_;
        CoefficientMatrix internal_vel_;
        CoefficientMatrix internal_acc_;

        struct TimePowers
        {
            double h;
            double h_inv;
            double h2_inv;
            double h3_inv;
            double h4_inv;
            double h5_inv;
            double h6_inv;
        };

        std::vector<TimePowers> time_powers_;
        CoefficientMatrix ws_rhs_mod_;
        CoefficientMatrix ws_solution_;
        CoefficientMatrix ws_lambda_;
        Eigen::Matrix<double, Eigen::Dynamic, 2 * DIM, Eigen::RowMajor> ws_gd_internal_;

    private:
        /**
         * @brief Rebuild the interpolating spline and expose the generated polynomial's validity.
         * @note Requires validated input shapes and exclusive ownership; finite inputs can still produce
         *       invalid coefficients or time knots, in which case isValid() becomes false.
         */
        void updateSplineInternal()
        {
            if (!trajectory_.template beginSpline<kMinDerivativeOrder>(
                    time_segments_, spatial_points_, boundary_, start_time_, kCoefficientCount))
            { num_segments_ = 0; return; }
            num_segments_ = static_cast<int>(time_segments_.size());
            // Prepare adjoint storage with the topology so the first pullback does not allocate.
            ws_lambda_.resize(std::max(0, num_segments_ - 1) * 2, DIM);
            ws_gd_internal_.resize(num_segments_ + 1, 2 * DIM);
            precomputeTimePowers();
            precomputePointDiffs();
            solveQuinticInPlace();
            trajectory_.finishSpline();
            // A finite input can still produce invalid coefficients or collapsed breakpoints.
        }

    public:
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW

        QuinticSplineND() = default;

        /** @brief Construct from N+1 strictly increasing finite knots and N+1 waypoint rows.
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

        bool isValid() const { return trajectory_.isValid(); }
        int dimension() const { return DIM; }

        double startTime() const
        {
            return trajectory_.startTime();
        }

        double endTime() const
        {
            return trajectory_.endTime();
        }

        double duration() const
        {
            return trajectory_.duration();
        }

        size_t numPoints() const
        {
            return static_cast<size_t>(spatial_points_.rows());
        }

        int numSegments() const
        {
            return trajectory_.numSegments();
        }

        const CoefficientMatrix &waypoints() const { return spatial_points_; }
        const std::vector<double> &durations() const { return time_segments_; }
        const std::vector<double> &breakpoints() const { return trajectory_.breakpoints_; }
        const BoundaryConditions<DIM> &boundary() const { return boundary_; }

        /** @brief Borrow the polynomial until the next update, assignment or move. */
        const Polynomial &polynomial() const & { return trajectory_; }
        const Polynomial &polynomial() const && = delete;
        /** @brief Make an independent owning snapshot, allocating its data. */
        Polynomial copyPolynomial() const { return trajectory_; }

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

        /** @brief Integral of the squared minimum derivative; returns NaN for an invalid spline. */
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

        Eigen::VectorXd energyDurationPartials() const
        {
            Eigen::VectorXd gdT;
            energyDurationPartials(gdT);
            return gdT;
        }

        /**
         * @brief Compute full gradient of energy w.r.t. segment durations.
         *
         * @return VectorXd Full gradient dE/dT, size num_segments.
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

        Eigen::VectorXd energyDurationGradient() const
        {
            Eigen::VectorXd grad;
            energyDurationGradient(grad);
            return grad;
        }

        /**
         * @brief Compute full gradient of energy w.r.t. inner waypoints only.
         * @return CoefficientMatrix Full gradient dE/dP for inner points only.
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

            backwardImpl(partialGradByCoeffs, partialGradByTimes,
                                  res.inner_points, res.durations, res.start, res.end);

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
            backwardImpl(partialGradByCoeffs, partialGradByTimes,
                                  grads.inner_points, grads.durations, grads.start, grads.end);
        }

    private:
        /** @brief Pull back the coefficient map through the cached continuity solve.
         * @note With A(T)q = b(T,P,boundary), solve A^T lambda = dL/dq and accumulate
         *       lambda^T (db/dtheta - dA/dtheta*q) plus direct coefficient-map partials.
         *       Input duration partials hold local power coefficients fixed. Outputs are
         *       overwritten; correctly sized output and adjoint storage are reused. */
        void backwardImpl(const CoefficientMatrix &partialGradByCoeffs,
                                   const Eigen::VectorXd &partialGradByTimes,
                                   CoefficientMatrix &innerPointsGrad,
                                   Eigen::VectorXd &gradByTimes,
                                   BoundaryGradient &startGrads,
                                   BoundaryGradient &endGrads)
        {
            const int n = num_segments_;
            const int n_pts = static_cast<int>(spatial_points_.rows());

            if (!trajectory_.isValid() || partialGradByCoeffs.rows() != n * kCoefficientCount ||
                partialGradByTimes.size() != n)
                throw std::invalid_argument("Adjoint requires a valid spline and matching partial shapes");
            gradByTimes = partialGradByTimes;
            startGrads = BoundaryGradient();
            endGrads = BoundaryGradient();

            if (n > 1)
            {
                innerPointsGrad.resize(n_pts - 2, DIM);
                innerPointsGrad.setZero();
            }
            else
            {
                innerPointsGrad.resize(0, DIM);
            }

            auto add_point_grad = [&](int idx, const RowVector &grad)
            {
                if (idx == 0)
                {
                    startGrads.p += grad.transpose();
                }
                else if (idx == n)
                {
                    endGrads.p += grad.transpose();
                }
                else
                {
                    innerPointsGrad.row(idx - 1) += grad;
                }
            };

            ws_gd_internal_.resize(n_pts, 2 * DIM);
            ws_gd_internal_.setZero();

            auto add_grad_d = [&](int idx, const RowVector &d_vel, const RowVector &d_acc)
            {
                ws_gd_internal_.row(idx).segment(0, DIM) += d_vel;
                ws_gd_internal_.row(idx).segment(DIM, DIM) += d_acc;
            };

            for (int i = 0; i < n; ++i)
            {
                const auto &tp = time_powers_[i];
                const int coeff_idx = i * 6;

                const RowVector &gc0 = partialGradByCoeffs.row(coeff_idx + 0);
                const RowVector &gc1 = partialGradByCoeffs.row(coeff_idx + 1);
                const RowVector &gc2 = partialGradByCoeffs.row(coeff_idx + 2);
                const RowVector &gc3 = partialGradByCoeffs.row(coeff_idx + 3);
                const RowVector &gc4 = partialGradByCoeffs.row(coeff_idx + 4);
                const RowVector &gc5 = partialGradByCoeffs.row(coeff_idx + 5);

                double k_P3 = 10.0 * tp.h3_inv;
                double k_P4 = -15.0 * tp.h4_inv;
                double k_P5 = 6.0 * tp.h5_inv;

                RowVector sum_grad_P = gc3 * k_P3 + gc4 * k_P4 + gc5 * k_P5;
                add_point_grad(i, gc0 - sum_grad_P);
                add_point_grad(i + 1, sum_grad_P);

                RowVector grad_v_curr = gc1;
                grad_v_curr += gc3 * (-6.0 * tp.h2_inv);
                grad_v_curr += gc4 * (8.0 * tp.h3_inv);
                grad_v_curr += gc5 * (-3.0 * tp.h4_inv);

                RowVector grad_v_next = gc3 * (-4.0 * tp.h2_inv);
                grad_v_next += gc4 * (7.0 * tp.h3_inv);
                grad_v_next += gc5 * (-3.0 * tp.h4_inv);

                RowVector grad_a_curr = gc2 * 0.5;
                grad_a_curr += gc3 * (-1.5 * tp.h_inv);
                grad_a_curr += gc4 * (1.5 * tp.h2_inv);
                grad_a_curr += gc5 * (-0.5 * tp.h3_inv);

                RowVector grad_a_next = gc3 * (0.5 * tp.h_inv);
                grad_a_next += gc4 * (-1.0 * tp.h2_inv);
                grad_a_next += gc5 * (0.5 * tp.h3_inv);

                add_grad_d(i, grad_v_curr, grad_a_curr);
                add_grad_d(i + 1, grad_v_next, grad_a_next);

                const RowVector &V_curr = internal_vel_.row(i);
                const RowVector &V_next = internal_vel_.row(i + 1);
                const RowVector &A_curr = internal_acc_.row(i);
                const RowVector &A_next = internal_acc_.row(i + 1);

                const RowVector P_diff = point_diffs_.row(i);

                RowVector dc3_dh = -30.0 * tp.h4_inv * P_diff +
                                       12.0 * tp.h3_inv * V_curr + 8.0 * tp.h3_inv * V_next +
                                       1.5 * tp.h2_inv * A_curr - 0.5 * tp.h2_inv * A_next;

                RowVector dc4_dh = 60.0 * tp.h5_inv * P_diff -
                                       24.0 * tp.h4_inv * V_curr - 21.0 * tp.h4_inv * V_next -
                                       3.0 * tp.h3_inv * A_curr + 2.0 * tp.h3_inv * A_next;

                RowVector dc5_dh = -30.0 * tp.h6_inv * P_diff +
                                       12.0 * tp.h5_inv * V_curr + 12.0 * tp.h5_inv * V_next +
                                       1.5 * tp.h4_inv * A_curr - 1.5 * tp.h4_inv * A_next;

                gradByTimes(i) += gc3.dot(dc3_dh) + gc4.dot(dc4_dh) + gc5.dot(dc5_dh);
            }

            Eigen::Matrix<double, 2, DIM> raw_start_grad;
            raw_start_grad.row(0) = ws_gd_internal_.row(0).segment(0, DIM);
            raw_start_grad.row(1) = ws_gd_internal_.row(0).segment(DIM, DIM);

            Eigen::Matrix<double, 2, DIM> raw_end_grad;
            raw_end_grad.row(0) = ws_gd_internal_.row(n).segment(0, DIM);
            raw_end_grad.row(1) = ws_gd_internal_.row(n).segment(DIM, DIM);

            const int num_blocks = n - 1;
            if (num_blocks > 0)
            {
                ws_lambda_.resize(num_blocks * 2, DIM);

                for (int i = 0; i < num_blocks; ++i)
                {
                    ws_lambda_.row(2 * i) = ws_gd_internal_.row(i + 1).segment(0, DIM);
                    ws_lambda_.row(2 * i + 1) = ws_gd_internal_.row(i + 1).segment(DIM, DIM);
                }

                multiplyStoredBlock2x2T_2xN(D_inv_cache_, 0,
                                            ws_lambda_.template middleRows<2>(0),
                                            ws_lambda_.template middleRows<2>(0));

                for (int i = 0; i < num_blocks - 1; ++i)
                {
                    subMultiplyStoredBlock2x2T_2xN(U_blocks_cache_, i,
                                                   ws_lambda_.template middleRows<2>(2 * i),
                                                   ws_lambda_.template middleRows<2>(2 * (i + 1)));
                    multiplyStoredBlock2x2T_2xN(D_inv_cache_, i + 1,
                                                ws_lambda_.template middleRows<2>(2 * (i + 1)),
                                                ws_lambda_.template middleRows<2>(2 * (i + 1)));
                }

                for (int i = num_blocks - 2; i >= 0; --i)
                {
                    subMultiplyStoredBlock2x2_2xN(D_inv_T_mul_L_next_T_cache_, i,
                                                  ws_lambda_.template middleRows<2>(2 * (i + 1)),
                                                  ws_lambda_.template middleRows<2>(2 * i));
                }

                for (int i = 0; i < num_blocks; ++i)
                {
                    const RowVector lam_snap = ws_lambda_.row(2 * i);
                    const RowVector lam_jerk = ws_lambda_.row(2 * i + 1);

                    const int m = i + 1;
                    const auto &tp_L = time_powers_[m - 1];
                    const auto &tp_R = time_powers_[m];

                    const RowVector dP_L = point_diffs_.row(m - 1);
                    const RowVector dP_R = point_diffs_.row(m);
                    const RowVector &V_prev = internal_vel_.row(m - 1);
                    const RowVector &V_curr = internal_vel_.row(m);
                    const RowVector &V_next = internal_vel_.row(m + 1);

                    const RowVector &A_prev = internal_acc_.row(m - 1);
                    const RowVector &A_curr = internal_acc_.row(m);
                    const RowVector &A_next = internal_acc_.row(m + 1);

                    const double dD00_dhL = 576.0 * tp_L.h4_inv;
                    const double dD01_dhL = -72.0 * tp_L.h3_inv;
                    const double dD10_dhL = 72.0 * tp_L.h3_inv;
                    const double dD11_dhL = -9.0 * tp_L.h2_inv;

                    const double dL00_dhL = 504.0 * tp_L.h4_inv;
                    const double dL01_dhL = 48.0 * tp_L.h3_inv;
                    const double dL10_dhL = 48.0 * tp_L.h3_inv;
                    const double dL11_dhL = 3.0 * tp_L.h2_inv;

                    const double dD00_dhR = 576.0 * tp_R.h4_inv;
                    const double dD01_dhR = 72.0 * tp_R.h3_inv;
                    const double dD10_dhR = -72.0 * tp_R.h3_inv;
                    const double dD11_dhR = -9.0 * tp_R.h2_inv;

                    const double dU00_dhR = 504.0 * tp_R.h4_inv;
                    const double dU01_dhR = -48.0 * tp_R.h3_inv;
                    const double dU10_dhR = -48.0 * tp_R.h3_inv;
                    const double dU11_dhR = 3.0 * tp_R.h2_inv;

                    const RowVector drhs0_dhL = 1440.0 * dP_L * tp_L.h5_inv;
                    const RowVector drhs1_dhL = 180.0 * dP_L * tp_L.h4_inv;

                    RowVector rhs0_L = drhs0_dhL -
                                           (dD00_dhL * V_curr + dD01_dhL * A_curr +
                                            dL00_dhL * V_prev + dL01_dhL * A_prev);
                    RowVector rhs1_L = drhs1_dhL -
                                           (dD10_dhL * V_curr + dD11_dhL * A_curr +
                                            dL10_dhL * V_prev + dL11_dhL * A_prev);

                    gradByTimes(m - 1) += lam_snap.dot(rhs0_L) + lam_jerk.dot(rhs1_L);

                    const RowVector drhs0_dhR = 1440.0 * dP_R * tp_R.h5_inv;
                    const RowVector drhs1_dhR = -180.0 * dP_R * tp_R.h4_inv;

                    RowVector rhs0_R = drhs0_dhR -
                                           (dD00_dhR * V_curr + dD01_dhR * A_curr +
                                            dU00_dhR * V_next + dU01_dhR * A_next);
                    RowVector rhs1_R = drhs1_dhR -
                                           (dD10_dhR * V_curr + dD11_dhR * A_curr +
                                            dU10_dhR * V_next + dU11_dhR * A_next);

                    gradByTimes(m) += lam_snap.dot(rhs0_R) + lam_jerk.dot(rhs1_R);

                    double dr4_dp_next = -360.0 * tp_R.h4_inv;
                    double dr4_dp_curr = 360.0 * (tp_R.h4_inv - tp_L.h4_inv);
                    double dr4_dp_prev = 360.0 * tp_L.h4_inv;

                    double dr3_dp_next = 60.0 * tp_R.h3_inv;
                    double dr3_dp_curr = -60.0 * (tp_R.h3_inv + tp_L.h3_inv);
                    double dr3_dp_prev = 60.0 * tp_L.h3_inv;

                    RowVector grad_P_next = lam_snap * dr4_dp_next + lam_jerk * dr3_dp_next;
                    RowVector grad_P_curr = lam_snap * dr4_dp_curr + lam_jerk * dr3_dp_curr;
                    RowVector grad_P_prev = lam_snap * dr4_dp_prev + lam_jerk * dr3_dp_prev;

                    add_point_grad(i + 2, grad_P_next);
                    add_point_grad(i + 1, grad_P_curr);
                    add_point_grad(i, grad_P_prev);
                }

                Eigen::Matrix<double, 2, DIM> correction_start;
                multiplyStoredBlock2x2T_2xN(L_blocks_cache_, 0,
                                            ws_lambda_.template middleRows<2>(0),
                                            correction_start);
                raw_start_grad -= correction_start;

                Eigen::Matrix<double, 2, DIM> correction_end;
                multiplyStoredBlock2x2T_2xN(U_blocks_cache_, num_blocks - 1,
                                            ws_lambda_.template middleRows<2>(2 * (num_blocks - 1)),
                                            correction_end);
                raw_end_grad -= correction_end;
            }

            startGrads.v = raw_start_grad.row(0).transpose();
            startGrads.a = raw_start_grad.row(1).transpose();

            endGrads.v = raw_end_grad.row(0).transpose();
            endGrads.a = raw_end_grad.row(1).transpose();
        }
        void precomputeTimePowers()
        {
            int n = static_cast<int>(time_segments_.size());
            time_powers_.resize(n);

            for (int i = 0; i < n; ++i)
            {
                double h = time_segments_[i];
                double iv = 1.0 / h;
                double iv2 = iv * iv;
                double iv3 = iv2 * iv;

                time_powers_[i].h = h;
                time_powers_[i].h_inv = iv;
                time_powers_[i].h2_inv = iv2;
                time_powers_[i].h3_inv = iv3;
                time_powers_[i].h4_inv = iv3 * iv;
                time_powers_[i].h5_inv = iv3 * iv2;
                time_powers_[i].h6_inv = iv3 * iv3;
            }
        }

        void precomputePointDiffs()
        {
            point_diffs_.resize(num_segments_, DIM);
            for (int i = 0; i < num_segments_; ++i)
            {
                point_diffs_.row(i) = spatial_points_.row(i + 1) - spatial_points_.row(i);
            }
        }

        static inline void Inverse2x2(const Eigen::Matrix2d &A, Eigen::Matrix2d &A_inv_out)
        {
            const double a = A(0, 0), b = A(0, 1), c = A(1, 0), d = A(1, 1);
            const double det = a * d - b * c;
            const double inv_det = 1.0 / det;

            A_inv_out(0, 0) = d * inv_det;
            A_inv_out(0, 1) = -b * inv_det;
            A_inv_out(1, 0) = -c * inv_det;
            A_inv_out(1, 1) = a * inv_det;
        }

        static inline void setBlock2x2(BlockMatrix2x2Storage &storage, int i, const Eigen::Matrix2d &M)
        {
            storage(i, 0) = M(0, 0);
            storage(i, 1) = M(0, 1);
            storage(i, 2) = M(1, 0);
            storage(i, 3) = M(1, 1);
        }

        inline void MultiplyStoredBlock2x2(const BlockMatrix2x2Storage &A_storage, int idx_a,
                                           const BlockMatrix2x2Storage &B_storage, int idx_b,
                                           Eigen::Matrix2d &C_out) const noexcept
        {
            const double a00 = A_storage(idx_a, 0), a01 = A_storage(idx_a, 1);
            const double a10 = A_storage(idx_a, 2), a11 = A_storage(idx_a, 3);
            const double b00 = B_storage(idx_b, 0), b01 = B_storage(idx_b, 1);
            const double b10 = B_storage(idx_b, 2), b11 = B_storage(idx_b, 3);
            C_out(0, 0) = a00 * b00 + a01 * b10;
            C_out(0, 1) = a00 * b01 + a01 * b11;
            C_out(1, 0) = a10 * b00 + a11 * b10;
            C_out(1, 1) = a10 * b01 + a11 * b11;
        }

        template <typename BlockOut, typename BlockIn>
        inline void multiplyStoredBlock2x2_2xN(const BlockMatrix2x2Storage &A_storage, int idx,
                                               const BlockIn &B, BlockOut &&C_out) const noexcept
        {
            const double a00 = A_storage(idx, 0), a01 = A_storage(idx, 1);
            const double a10 = A_storage(idx, 2), a11 = A_storage(idx, 3);
            for (int j = 0; j < DIM; ++j)
            {
                const double b0j = B(0, j);
                const double b1j = B(1, j);
                C_out(0, j) = a00 * b0j + a01 * b1j;
                C_out(1, j) = a10 * b0j + a11 * b1j;
            }
        }

        template <typename BlockOut, typename BlockIn>
        inline void multiplyStoredBlock2x2T_2xN(const BlockMatrix2x2Storage &A_storage, int idx,
                                                const BlockIn &B, BlockOut &&C_out) const noexcept
        {
            const double a00 = A_storage(idx, 0), a01 = A_storage(idx, 1);
            const double a10 = A_storage(idx, 2), a11 = A_storage(idx, 3);
            for (int j = 0; j < DIM; ++j)
            {
                const double b0j = B(0, j);
                const double b1j = B(1, j);
                C_out(0, j) = a00 * b0j + a10 * b1j;
                C_out(1, j) = a01 * b0j + a11 * b1j;
            }
        }

        template <typename BlockOut, typename BlockIn>
        inline void subMultiplyStoredBlock2x2T_2xN(const BlockMatrix2x2Storage &A_storage, int idx,
                                                   const BlockIn &B, BlockOut &&C_out) const noexcept
        {
            const double a00 = A_storage(idx, 0), a01 = A_storage(idx, 1);
            const double a10 = A_storage(idx, 2), a11 = A_storage(idx, 3);
            for (int j = 0; j < DIM; ++j)
            {
                const double b0j = B(0, j);
                const double b1j = B(1, j);
                C_out(0, j) -= a00 * b0j + a10 * b1j;
                C_out(1, j) -= a01 * b0j + a11 * b1j;
            }
        }

        template <typename BlockOut, typename BlockIn>
        inline void subMultiplyStoredBlock2x2_2xN(const BlockMatrix2x2Storage &A_storage, int idx,
                                                  const BlockIn &B, BlockOut &&C_out) const noexcept
        {
            const double a00 = A_storage(idx, 0), a01 = A_storage(idx, 1);
            const double a10 = A_storage(idx, 2), a11 = A_storage(idx, 3);
            for (int j = 0; j < DIM; ++j)
            {
                const double b0j = B(0, j);
                const double b1j = B(1, j);
                C_out(0, j) -= a00 * b0j + a01 * b1j;
                C_out(1, j) -= a10 * b0j + a11 * b1j;
            }
        }

        void solveInternalDerivatives(const CoefficientMatrix &P, CoefficientMatrix &p_out, CoefficientMatrix &q_out)
        {
            const int n = static_cast<int>(P.rows());
            p_out.resize(n, DIM);
            q_out.resize(n, DIM);

            p_out.row(0) = boundary_.start_velocity.transpose();
            q_out.row(0) = boundary_.start_acceleration.transpose();
            p_out.row(n - 1) = boundary_.end_velocity.transpose();
            q_out.row(n - 1) = boundary_.end_acceleration.transpose();

            const int num_blocks = n - 2;
            if (num_blocks <= 0)
                return;

            Eigen::Matrix<double, 2, DIM> B_left, B_right;
            B_left.row(0) = boundary_.start_velocity.transpose();
            B_left.row(1) = boundary_.start_acceleration.transpose();
            B_right.row(0) = boundary_.end_velocity.transpose();
            B_right.row(1) = boundary_.end_acceleration.transpose();

            U_blocks_cache_.resize(num_blocks, 4);
            D_inv_cache_.resize(num_blocks, 4);
            L_blocks_cache_.resize(num_blocks, 4);
            D_inv_T_mul_L_next_T_cache_.resize(std::max(0, num_blocks - 1), 4);
            ws_rhs_mod_.resize(num_blocks * 2, DIM);

            for (int i = 0; i < num_blocks; ++i)
            {
                const int k = i + 2;
                const auto &tp_L = time_powers_[k - 2];
                const auto &tp_R = time_powers_[k - 1];

                auto rhs_block = ws_rhs_mod_.template middleRows<2>(2 * i);
                const RowVector dP_L = point_diffs_.row(k - 2);
                const RowVector dP_R = point_diffs_.row(k - 1);
                rhs_block.row(0) = -360.0 * (dP_R * tp_R.h4_inv + dP_L * tp_L.h4_inv);
                rhs_block.row(1) = 60.0 * (dP_R * tp_R.h3_inv - dP_L * tp_L.h3_inv);

                Eigen::Matrix2d D;
                D << -192.0 * (tp_L.h3_inv + tp_R.h3_inv), 36.0 * (tp_L.h2_inv - tp_R.h2_inv),
                    -36.0 * (tp_L.h2_inv - tp_R.h2_inv), 9.0 * (tp_L.h_inv + tp_R.h_inv);

                Eigen::Matrix2d L;
                L << -168.0 * tp_L.h3_inv, -24.0 * tp_L.h2_inv,
                    -24.0 * tp_L.h2_inv, -3.0 * tp_L.h_inv;
                setBlock2x2(L_blocks_cache_, i, L);

                Eigen::Matrix2d U;
                U << -168.0 * tp_R.h3_inv, 24.0 * tp_R.h2_inv,
                    24.0 * tp_R.h2_inv, -3.0 * tp_R.h_inv;
                setBlock2x2(U_blocks_cache_, i, U);

                if (i == 0)
                {
                    rhs_block.noalias() -= L * B_left;
                }
                else
                {
                    Eigen::Matrix2d X;
                    MultiplyStoredBlock2x2(D_inv_cache_, i - 1, U_blocks_cache_, i - 1, X);
                    Eigen::Matrix<double, 2, DIM> Y;
                    multiplyStoredBlock2x2_2xN(D_inv_cache_, i - 1, ws_rhs_mod_.template middleRows<2>(2 * (i - 1)), Y);
                    D.noalias() -= L * X;
                    rhs_block.noalias() -= L * Y;
                }

                if (k == n - 1)
                {
                    rhs_block.noalias() -= U * B_right;
                }

                Eigen::Matrix2d D_inv;
                Inverse2x2(D, D_inv);
                setBlock2x2(D_inv_cache_, i, D_inv);

                if (i > 0)
                {
                    Eigen::Matrix2d L_mul_D_prev_inv;
                    MultiplyStoredBlock2x2(L_blocks_cache_, i, D_inv_cache_, i - 1, L_mul_D_prev_inv);
                    setBlock2x2(D_inv_T_mul_L_next_T_cache_, i - 1, L_mul_D_prev_inv.transpose());
                }
            }

            ws_solution_.resize(num_blocks * 2, DIM);

            multiplyStoredBlock2x2_2xN(D_inv_cache_, num_blocks - 1,
                                       ws_rhs_mod_.template middleRows<2>(2 * (num_blocks - 1)),
                                       ws_solution_.template middleRows<2>(2 * (num_blocks - 1)));

            for (int i = num_blocks - 2; i >= 0; --i)
            {
                auto sol_block = ws_solution_.template middleRows<2>(2 * i);
                auto sol_next = ws_solution_.template middleRows<2>(2 * (i + 1));
                auto rhs_i = ws_rhs_mod_.template middleRows<2>(2 * i);

                const double u00 = U_blocks_cache_(i, 0), u01 = U_blocks_cache_(i, 1);
                const double u10 = U_blocks_cache_(i, 2), u11 = U_blocks_cache_(i, 3);
                Eigen::Matrix<double, 2, DIM> rhs_temp;
                for (int j = 0; j < DIM; ++j)
                {
                    const double s0 = sol_next(0, j), s1 = sol_next(1, j);
                    rhs_temp(0, j) = rhs_i(0, j) - (u00 * s0 + u01 * s1);
                    rhs_temp(1, j) = rhs_i(1, j) - (u10 * s0 + u11 * s1);
                }
                multiplyStoredBlock2x2_2xN(D_inv_cache_, i, rhs_temp, sol_block);
            }

            for (int i = 0; i < num_blocks; ++i)
            {
                p_out.row(i + 1) = ws_solution_.row(2 * i);
                q_out.row(i + 1) = ws_solution_.row(2 * i + 1);
            }
        }

        void solveQuinticInPlace()
        {
            const int n = num_segments_;

            solveInternalDerivatives(spatial_points_, internal_vel_, internal_acc_);


            for (int i = 0; i < n; ++i)
            {
                const auto &tp = time_powers_[i];

                const RowVector c0 = spatial_points_.row(i);
                const RowVector c1 = internal_vel_.row(i);
                const RowVector c2 = internal_acc_.row(i) * 0.5;

                const RowVector rhs1 = point_diffs_.row(i) - c1 * tp.h - c2 * (tp.h * tp.h);
                const RowVector rhs2 = internal_vel_.row(i + 1) - c1 - (2.0 * c2) * tp.h;
                const RowVector rhs3 = internal_acc_.row(i + 1) - (2.0 * c2);

                const RowVector c3 = (10.0 * tp.h3_inv) * rhs1 - (4.0 * tp.h2_inv) * rhs2 + (0.5 * tp.h_inv) * rhs3;
                const RowVector c4 = (-15.0 * tp.h4_inv) * rhs1 + (7.0 * tp.h3_inv) * rhs2 - (tp.h2_inv) * rhs3;
                const RowVector c5 = (6.0 * tp.h5_inv) * rhs1 - (3.0 * tp.h4_inv) * rhs2 + (0.5 * tp.h3_inv) * rhs3;

                trajectory_.coefficients_.row(i * 6 + 0) = c0;
                trajectory_.coefficients_.row(i * 6 + 1) = c1;
                trajectory_.coefficients_.row(i * 6 + 2) = c2;
                trajectory_.coefficients_.row(i * 6 + 3) = c3;
                trajectory_.coefficients_.row(i * 6 + 4) = c4;
                trajectory_.coefficients_.row(i * 6 + 5) = c5;
            }
        }

    };

    /** @brief Minimum-snap interpolation with a specialized O(N) 3-by-3 block tridiagonal solver.
     * @note Minimize the squared 4-th physical derivative integral with fixed waypoint
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
        CoefficientMatrix point_diffs_;

        BoundaryConditions<DIM> boundary_;

        int num_segments_{0};

        Polynomial trajectory_;

        struct TimePowers
        {
            double h;
            double h_inv;  // h^-1
            double h2_inv; // h^-2
            double h3_inv; // h^-3
            double h4_inv; // h^-4
            double h5_inv; // h^-5
            double h6_inv; // h^-6
            double h7_inv; // h^-7
        };
        std::vector<TimePowers> time_powers_;

        using BlockMatrix3x3Storage = Eigen::Matrix<double, Eigen::Dynamic, 9, Eigen::RowMajor>;
        BlockMatrix3x3Storage D_inv_cache_;
        BlockMatrix3x3Storage U_blocks_cache_;
        BlockMatrix3x3Storage L_blocks_cache_;
        BlockMatrix3x3Storage D_inv_T_mul_L_next_T_cache_;

        CoefficientMatrix ws_rhs_mod_;
        CoefficientMatrix ws_solution_;
        CoefficientMatrix ws_lambda_;
        Eigen::Matrix<double, Eigen::Dynamic, 3 * DIM, Eigen::RowMajor> ws_gd_internal_;

        CoefficientMatrix internal_vel_;
        CoefficientMatrix internal_acc_;
        CoefficientMatrix internal_jerk_;

    private:
        /**
         * @brief Rebuild the interpolating spline and expose the generated polynomial's validity.
         * @note Requires validated input shapes and exclusive ownership; finite inputs can still produce
         *       invalid coefficients or time knots, in which case isValid() becomes false.
         */
        void updateSplineInternal()
        {
            if (!trajectory_.template beginSpline<kMinDerivativeOrder>(
                    time_segments_, spatial_points_, boundary_, start_time_, kCoefficientCount))
            { num_segments_ = 0; return; }
            num_segments_ = static_cast<int>(time_segments_.size());
            // Prepare adjoint storage with the topology so the first pullback does not allocate.
            ws_lambda_.resize(std::max(0, num_segments_ - 1) * 3, DIM);
            ws_gd_internal_.resize(num_segments_ + 1, 3 * DIM);
            precomputeTimePowers();
            precomputePointDiffs();
            solveSepticInPlace();
            trajectory_.finishSpline();
            // A finite input can still produce invalid coefficients or collapsed breakpoints.
        }

    public:
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW

        SepticSplineND() = default;

        /** @brief Construct from N+1 strictly increasing finite knots and N+1 waypoint rows.
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

        bool isValid() const { return trajectory_.isValid(); }
        int dimension() const { return DIM; }

        double startTime() const
        {
            return trajectory_.startTime();
        }

        double endTime() const
        {
            return trajectory_.endTime();
        }

        double duration() const
        {
            return trajectory_.duration();
        }

        size_t numPoints() const
        {
            return static_cast<size_t>(spatial_points_.rows());
        }

        int numSegments() const
        {
            return trajectory_.numSegments();
        }

        const CoefficientMatrix &waypoints() const { return spatial_points_; }
        const std::vector<double> &durations() const { return time_segments_; }
        const std::vector<double> &breakpoints() const { return trajectory_.breakpoints_; }
        const BoundaryConditions<DIM> &boundary() const { return boundary_; }

        /** @brief Borrow the polynomial until the next update, assignment or move. */
        const Polynomial &polynomial() const & { return trajectory_; }
        const Polynomial &polynomial() const && = delete;
        /** @brief Make an independent owning snapshot, allocating its data. */
        Polynomial copyPolynomial() const { return trajectory_; }

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

        /** @brief Integral of the squared minimum derivative; returns NaN for an invalid spline. */
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

        Eigen::VectorXd energyDurationPartials() const
        {
            Eigen::VectorXd gdT;
            energyDurationPartials(gdT);
            return gdT;
        }

        /**
         * @brief Compute full gradient of energy w.r.t. segment durations.
         *
         * @return VectorXd Full gradient dE/dT, size num_segments.
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

        Eigen::VectorXd energyDurationGradient() const
        {
            Eigen::VectorXd grad;
            energyDurationGradient(grad);
            return grad;
        }

        /**
         * @brief Compute full gradient of energy w.r.t. inner waypoints only.
         * @return CoefficientMatrix Full gradient dE/dP for inner points only.
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

            backwardImpl(partialGradByCoeffs, partialGradByTimes,
                                  res.inner_points, res.durations, res.start, res.end);

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
            backwardImpl(partialGradByCoeffs, partialGradByTimes,
                                  grads.inner_points, grads.durations, grads.start, grads.end);
        }

    private:
        /** @brief Pull back the coefficient map through the cached continuity solve.
         * @note With A(T)q = b(T,P,boundary), solve A^T lambda = dL/dq and accumulate
         *       lambda^T (db/dtheta - dA/dtheta*q) plus direct coefficient-map partials.
         *       Input duration partials hold local power coefficients fixed. Outputs are
         *       overwritten; correctly sized output and adjoint storage are reused. */
        void backwardImpl(const CoefficientMatrix &partialGradByCoeffs,
                                   const Eigen::VectorXd &partialGradByTimes,
                                   CoefficientMatrix &innerPointsGrad,
                                   Eigen::VectorXd &gradByTimes,
                                   BoundaryGradient &startGrads,
                                   BoundaryGradient &endGrads)
        {
            const int n = num_segments_;
            const int n_pts = static_cast<int>(spatial_points_.rows());

            if (!trajectory_.isValid() || partialGradByCoeffs.rows() != n * kCoefficientCount ||
                partialGradByTimes.size() != n)
                throw std::invalid_argument("Adjoint requires a valid spline and matching partial shapes");
            gradByTimes = partialGradByTimes;
            startGrads = BoundaryGradient();
            endGrads = BoundaryGradient();

            if (n > 1)
            {
                innerPointsGrad.resize(n_pts - 2, DIM);
                innerPointsGrad.setZero();
            }
            else
            {
                innerPointsGrad.resize(0, DIM);
            }

            auto add_point_grad = [&](int idx, const RowVector &grad)
            {
                if (idx == 0)
                {
                    startGrads.p += grad.transpose();
                }
                else if (idx == n)
                {
                    endGrads.p += grad.transpose();
                }
                else
                {
                    innerPointsGrad.row(idx - 1) += grad;
                }
            };

            ws_gd_internal_.resize(n_pts, 3 * DIM);
            ws_gd_internal_.setZero();

            auto add_grad_d = [&](int idx, const RowVector &d_vel, const RowVector &d_acc, const RowVector &d_jerk)
            {
                ws_gd_internal_.row(idx).segment(0, DIM) += d_vel;
                ws_gd_internal_.row(idx).segment(DIM, DIM) += d_acc;
                ws_gd_internal_.row(idx).segment(2 * DIM, DIM) += d_jerk;
            };

            for (int i = 0; i < n; ++i)
            {
                const auto &tp = time_powers_[i];
                const int coeff_idx = i * 8;

                const RowVector &gc0 = partialGradByCoeffs.row(coeff_idx + 0);
                const RowVector &gc1 = partialGradByCoeffs.row(coeff_idx + 1);
                const RowVector &gc2 = partialGradByCoeffs.row(coeff_idx + 2);
                const RowVector &gc3 = partialGradByCoeffs.row(coeff_idx + 3);
                const RowVector &gc4 = partialGradByCoeffs.row(coeff_idx + 4);
                const RowVector &gc5 = partialGradByCoeffs.row(coeff_idx + 5);
                const RowVector &gc6 = partialGradByCoeffs.row(coeff_idx + 6);
                const RowVector &gc7 = partialGradByCoeffs.row(coeff_idx + 7);

                double k_P4 = -(210.0 * tp.h4_inv) / 6.0;
                double k_P5 = (168.0 * tp.h5_inv) / 2.0;
                double k_P6 = -(420.0 * tp.h6_inv) / 6.0;
                double k_P7 = (120.0 * tp.h7_inv) / 6.0;

                RowVector sum_grad_P = gc4 * k_P4 + gc5 * k_P5 + gc6 * k_P6 + gc7 * k_P7;
                add_point_grad(i, gc0 + sum_grad_P);
                add_point_grad(i + 1, -sum_grad_P);

                add_grad_d(i, gc1, 0.5 * gc2, (1.0 / 6.0) * gc3);

                RowVector grad_v_curr = gc4 * (-20.0 * tp.h3_inv) +
                                            gc5 * (45.0 * tp.h4_inv) +
                                            gc6 * (-36.0 * tp.h5_inv) +
                                            gc7 * (10.0 * tp.h6_inv);

                RowVector grad_v_next = gc4 * (-15.0 * tp.h3_inv) +
                                            gc5 * (39.0 * tp.h4_inv) +
                                            gc6 * (-34.0 * tp.h5_inv) +
                                            gc7 * (10.0 * tp.h6_inv);

                RowVector grad_a_curr = gc4 * (-5.0 * tp.h2_inv) +
                                            gc5 * (10.0 * tp.h3_inv) +
                                            gc6 * (-7.5 * tp.h4_inv) +
                                            gc7 * (2.0 * tp.h5_inv);

                RowVector grad_a_next = gc4 * (2.5 * tp.h2_inv) +
                                            gc5 * (-7.0 * tp.h3_inv) +
                                            gc6 * (6.5 * tp.h4_inv) +
                                            gc7 * (-2.0 * tp.h5_inv);

                RowVector grad_j_curr = gc4 * (-2.0 / 3.0 * tp.h_inv) +
                                            gc5 * (tp.h2_inv) +
                                            gc6 * (-2.0 / 3.0 * tp.h3_inv) +
                                            gc7 * (1.0 / 6.0 * tp.h4_inv);

                RowVector grad_j_next = gc4 * (-1.0 / 6.0 * tp.h_inv) +
                                            gc5 * (0.5 * tp.h2_inv) +
                                            gc6 * (-0.5 * tp.h3_inv) +
                                            gc7 * (1.0 / 6.0 * tp.h4_inv);

                add_grad_d(i, grad_v_curr, grad_a_curr, grad_j_curr);
                add_grad_d(i + 1, grad_v_next, grad_a_next, grad_j_next);

                {
                    const RowVector &P_curr = spatial_points_.row(i);
                    const RowVector &P_next = spatial_points_.row(i + 1);
                    const RowVector &V_curr = internal_vel_.row(i);
                    const RowVector &V_next = internal_vel_.row(i + 1);
                    const RowVector &A_curr = internal_acc_.row(i);
                    const RowVector &A_next = internal_acc_.row(i + 1);
                    const RowVector &J_curr = internal_jerk_.row(i);
                    const RowVector &J_next = internal_jerk_.row(i + 1);
                    if constexpr (DIM <= 3)
                    {
                        double h8_inv = tp.h7_inv * tp.h_inv;
                        double dot_dc4 = 0.0;
                        double dot_dc5 = 0.0;
                        double dot_dc6 = 0.0;
                        double dot_dc7 = 0.0;

                        for (int j = 0; j < DIM; ++j)
                        {
                            const double dPj = P_curr(j) - P_next(j);
                            const double dc4 = ((840.0 * dPj * tp.h5_inv) +
                                                (360.0 * V_curr(j) * tp.h4_inv) + (270.0 * V_next(j) * tp.h4_inv) +
                                                (60.0 * A_curr(j) * tp.h3_inv) - (30.0 * A_next(j) * tp.h3_inv) +
                                                (4.0 * J_curr(j) * tp.h2_inv) + (J_next(j) * tp.h2_inv)) /
                                               6.0;

                            const double dc5 = ((-840.0 * dPj * tp.h6_inv) -
                                                (360.0 * V_curr(j) * tp.h5_inv) - (312.0 * V_next(j) * tp.h5_inv) -
                                                (60.0 * A_curr(j) * tp.h4_inv) + (42.0 * A_next(j) * tp.h4_inv) -
                                                (4.0 * J_curr(j) * tp.h3_inv) - (2.0 * J_next(j) * tp.h3_inv)) /
                                               2.0;

                            const double dc6 = ((2520.0 * dPj * tp.h7_inv) +
                                                (1080.0 * V_curr(j) * tp.h6_inv) + (1020.0 * V_next(j) * tp.h6_inv) +
                                                (180.0 * A_curr(j) * tp.h5_inv) - (156.0 * A_next(j) * tp.h5_inv) +
                                                (12.0 * J_curr(j) * tp.h4_inv) + (9.0 * J_next(j) * tp.h4_inv)) /
                                               6.0;

                            const double dc7 = ((-840.0 * dPj * h8_inv) -
                                                (360.0 * V_curr(j) * tp.h7_inv) - (360.0 * V_next(j) * tp.h7_inv) -
                                                (60.0 * A_curr(j) * tp.h6_inv) + (60.0 * A_next(j) * tp.h6_inv) -
                                                (4.0 * J_curr(j) * tp.h5_inv) - (4.0 * J_next(j) * tp.h5_inv)) /
                                               6.0;

                            dot_dc4 += gc4(j) * dc4;
                            dot_dc5 += gc5(j) * dc5;
                            dot_dc6 += gc6(j) * dc6;
                            dot_dc7 += gc7(j) * dc7;
                        }

                        gradByTimes(i) += dot_dc4 + dot_dc5 + dot_dc6 + dot_dc7;
                    }
                    else
                    {
                        RowVector dP = P_curr - P_next;

                        RowVector dc4_dh = ((840.0 * dP * tp.h5_inv) +
                                                (360.0 * V_curr * tp.h4_inv) + (270.0 * V_next * tp.h4_inv) +
                                                (60.0 * A_curr * tp.h3_inv) - (30.0 * A_next * tp.h3_inv) +
                                                (4.0 * J_curr * tp.h2_inv) + (J_next * tp.h2_inv)) /
                                               6.0;

                        RowVector dc5_dh = ((-840.0 * dP * tp.h6_inv) -
                                                (360.0 * V_curr * tp.h5_inv) - (312.0 * V_next * tp.h5_inv) -
                                                (60.0 * A_curr * tp.h4_inv) + (42.0 * A_next * tp.h4_inv) -
                                                (4.0 * J_curr * tp.h3_inv) - (2.0 * J_next * tp.h3_inv)) /
                                               2.0;

                        RowVector dc6_dh = ((2520.0 * dP * tp.h7_inv) +
                                                (1080.0 * V_curr * tp.h6_inv) + (1020.0 * V_next * tp.h6_inv) +
                                                (180.0 * A_curr * tp.h5_inv) - (156.0 * A_next * tp.h5_inv) +
                                                (12.0 * J_curr * tp.h4_inv) + (9.0 * J_next * tp.h4_inv)) /
                                               6.0;

                        double h8_inv = tp.h7_inv * tp.h_inv;
                        RowVector dc7_dh = ((-840.0 * dP * h8_inv) -
                                                (360.0 * V_curr * tp.h7_inv) - (360.0 * V_next * tp.h7_inv) -
                                                (60.0 * A_curr * tp.h6_inv) + (60.0 * A_next * tp.h6_inv) -
                                                (4.0 * J_curr * tp.h5_inv) - (4.0 * J_next * tp.h5_inv)) /
                                               6.0;

                        gradByTimes(i) += gc4.dot(dc4_dh) + gc5.dot(dc5_dh) + gc6.dot(dc6_dh) + gc7.dot(dc7_dh);
                    }
                }
            }

            Eigen::Matrix<double, 3, DIM> raw_start_grad;
            raw_start_grad.row(0) = ws_gd_internal_.row(0).segment(0, DIM);
            raw_start_grad.row(1) = ws_gd_internal_.row(0).segment(DIM, DIM);
            raw_start_grad.row(2) = ws_gd_internal_.row(0).segment(2 * DIM, DIM);

            Eigen::Matrix<double, 3, DIM> raw_end_grad;
            raw_end_grad.row(0) = ws_gd_internal_.row(n).segment(0, DIM);
            raw_end_grad.row(1) = ws_gd_internal_.row(n).segment(DIM, DIM);
            raw_end_grad.row(2) = ws_gd_internal_.row(n).segment(2 * DIM, DIM);

            const int num_blocks = n - 1;
            if (num_blocks > 0)
            {
                ws_lambda_.resize(num_blocks * 3, DIM);

                for (int i = 0; i < num_blocks; ++i)
                {
                    ws_lambda_.row(3 * i) = ws_gd_internal_.row(i + 1).segment(0, DIM);
                    ws_lambda_.row(3 * i + 1) = ws_gd_internal_.row(i + 1).segment(DIM, DIM);
                    ws_lambda_.row(3 * i + 2) = ws_gd_internal_.row(i + 1).segment(2 * DIM, DIM);
                }

                multiplyStoredBlock3x3T_3xN(D_inv_cache_, 0,
                                            ws_lambda_.template middleRows<3>(0),
                                            ws_lambda_.template middleRows<3>(0));

                for (int i = 0; i < num_blocks - 1; ++i)
                {
                    subMultiplyStoredBlock3x3T_3xN(U_blocks_cache_, i,
                                                   ws_lambda_.template middleRows<3>(3 * i),
                                                   ws_lambda_.template middleRows<3>(3 * (i + 1)));
                    multiplyStoredBlock3x3T_3xN(D_inv_cache_, i + 1,
                                                ws_lambda_.template middleRows<3>(3 * (i + 1)),
                                                ws_lambda_.template middleRows<3>(3 * (i + 1)));
                }

                for (int i = num_blocks - 2; i >= 0; --i)
                {
                    subMultiplyStoredBlock3x3_3xN(D_inv_T_mul_L_next_T_cache_, i,
                                                  ws_lambda_.template middleRows<3>(3 * (i + 1)),
                                                  ws_lambda_.template middleRows<3>(3 * i));
                }

                for (int i = 0; i < num_blocks; ++i)
                {
                    const int m = i + 1;
                    const auto &tp_L = time_powers_[m - 1];
                    const auto &tp_R = time_powers_[m];

                    const RowVector &P_prev = spatial_points_.row(m - 1);
                    const RowVector &P_curr = spatial_points_.row(m);
                    const RowVector &P_next = spatial_points_.row(m + 1);
                    const RowVector &V_prev = internal_vel_.row(m - 1);
                    const RowVector &V_curr = internal_vel_.row(m);
                    const RowVector &V_next = internal_vel_.row(m + 1);

                    const RowVector &A_prev = internal_acc_.row(m - 1);
                    const RowVector &A_curr = internal_acc_.row(m);
                    const RowVector &A_next = internal_acc_.row(m + 1);

                    const RowVector &J_prev = internal_jerk_.row(m - 1);
                    const RowVector &J_curr = internal_jerk_.row(m);
                    const RowVector &J_next = internal_jerk_.row(m + 1);

                    const RowVector lam_3 = ws_lambda_.row(3 * i);
                    const RowVector lam_4 = ws_lambda_.row(3 * i + 1);
                    const RowVector lam_5 = ws_lambda_.row(3 * i + 2);

                    const double dD00_dhL = -1440.0 * tp_L.h4_inv;
                    const double dD01_dhL = 240.0 * tp_L.h3_inv;
                    const double dD02_dhL = -16.0 * tp_L.h2_inv;
                    const double dD10_dhL = -21600.0 * tp_L.h5_inv;
                    const double dD11_dhL = 3600.0 * tp_L.h4_inv;
                    const double dD12_dhL = -240.0 * tp_L.h3_inv;
                    const double dD20_dhL = -129600.0 * tp_L.h6_inv;
                    const double dD21_dhL = 21600.0 * tp_L.h5_inv;
                    const double dD22_dhL = -1440.0 * tp_L.h4_inv;

                    const double dL00_dhL = -1080.0 * tp_L.h4_inv;
                    const double dL01_dhL = -120.0 * tp_L.h3_inv;
                    const double dL02_dhL = -4.0 * tp_L.h2_inv;
                    const double dL10_dhL = -18720.0 * tp_L.h5_inv;
                    const double dL11_dhL = -2520.0 * tp_L.h4_inv;
                    const double dL12_dhL = -120.0 * tp_L.h3_inv;
                    const double dL20_dhL = -122400.0 * tp_L.h6_inv;
                    const double dL21_dhL = -18720.0 * tp_L.h5_inv;
                    const double dL22_dhL = -1080.0 * tp_L.h4_inv;

                    const double dD00_dhR = -1440.0 * tp_R.h4_inv;
                    const double dD01_dhR = -240.0 * tp_R.h3_inv;
                    const double dD02_dhR = -16.0 * tp_R.h2_inv;
                    const double dD10_dhR = 21600.0 * tp_R.h5_inv;
                    const double dD11_dhR = 3600.0 * tp_R.h4_inv;
                    const double dD12_dhR = 240.0 * tp_R.h3_inv;
                    const double dD20_dhR = -129600.0 * tp_R.h6_inv;
                    const double dD21_dhR = -21600.0 * tp_R.h5_inv;
                    const double dD22_dhR = -1440.0 * tp_R.h4_inv;

                    const double dU00_dhR = -1080.0 * tp_R.h4_inv;
                    const double dU01_dhR = 120.0 * tp_R.h3_inv;
                    const double dU02_dhR = -4.0 * tp_R.h2_inv;
                    const double dU10_dhR = 18720.0 * tp_R.h5_inv;
                    const double dU11_dhR = -2520.0 * tp_R.h4_inv;
                    const double dU12_dhR = 120.0 * tp_R.h3_inv;
                    const double dU20_dhR = -122400.0 * tp_R.h6_inv;
                    const double dU21_dhR = 18720.0 * tp_R.h5_inv;
                    const double dU22_dhR = -1080.0 * tp_R.h4_inv;
                    if constexpr (DIM <= 3)
                    {
                        double grad_hL = 0.0;
                        double grad_hR = 0.0;
                        for (int j = 0; j < DIM; ++j)
                        {
                            const double p_prev = P_prev(j);
                            const double p_curr = P_curr(j);
                            const double p_next = P_next(j);
                            const double v_prev = V_prev(j);
                            const double v_curr = V_curr(j);
                            const double v_next = V_next(j);
                            const double a_prev = A_prev(j);
                            const double a_curr = A_curr(j);
                            const double a_next = A_next(j);
                            const double j_prev = J_prev(j);
                            const double j_curr = J_curr(j);
                            const double j_next = J_next(j);

                            const double rhs0_L = -3360.0 * (p_curr - p_prev) * tp_L.h5_inv -
                                                  (dD00_dhL * v_curr + dD01_dhL * a_curr + dD02_dhL * j_curr +
                                                   dL00_dhL * v_prev + dL01_dhL * a_prev + dL02_dhL * j_prev);
                            const double rhs1_L = -50400.0 * (p_curr - p_prev) * tp_L.h6_inv -
                                                  (dD10_dhL * v_curr + dD11_dhL * a_curr + dD12_dhL * j_curr +
                                                   dL10_dhL * v_prev + dL11_dhL * a_prev + dL12_dhL * j_prev);
                            const double rhs2_L = -302400.0 * (p_curr - p_prev) * tp_L.h7_inv -
                                                  (dD20_dhL * v_curr + dD21_dhL * a_curr + dD22_dhL * j_curr +
                                                   dL20_dhL * v_prev + dL21_dhL * a_prev + dL22_dhL * j_prev);

                            const double rhs0_R = -3360.0 * (p_next - p_curr) * tp_R.h5_inv -
                                                  (dD00_dhR * v_curr + dD01_dhR * a_curr + dD02_dhR * j_curr +
                                                   dU00_dhR * v_next + dU01_dhR * a_next + dU02_dhR * j_next);
                            const double rhs1_R = -50400.0 * (p_curr - p_next) * tp_R.h6_inv -
                                                  (dD10_dhR * v_curr + dD11_dhR * a_curr + dD12_dhR * j_curr +
                                                   dU10_dhR * v_next + dU11_dhR * a_next + dU12_dhR * j_next);
                            const double rhs2_R = -302400.0 * (p_next - p_curr) * tp_R.h7_inv -
                                                  (dD20_dhR * v_curr + dD21_dhR * a_curr + dD22_dhR * j_curr +
                                                   dU20_dhR * v_next + dU21_dhR * a_next + dU22_dhR * j_next);

                            grad_hL += lam_3(j) * rhs0_L + lam_4(j) * rhs1_L + lam_5(j) * rhs2_L;
                            grad_hR += lam_3(j) * rhs0_R + lam_4(j) * rhs1_R + lam_5(j) * rhs2_R;
                        }
                        gradByTimes(m - 1) += grad_hL;
                        gradByTimes(m) += grad_hR;
                    }
                    else
                    {
                        RowVector rhs0_L = -3360.0 * (P_curr - P_prev) * tp_L.h5_inv -
                                               (dD00_dhL * V_curr + dD01_dhL * A_curr + dD02_dhL * J_curr +
                                                dL00_dhL * V_prev + dL01_dhL * A_prev + dL02_dhL * J_prev);
                        RowVector rhs1_L = -50400.0 * (P_curr - P_prev) * tp_L.h6_inv -
                                               (dD10_dhL * V_curr + dD11_dhL * A_curr + dD12_dhL * J_curr +
                                                dL10_dhL * V_prev + dL11_dhL * A_prev + dL12_dhL * J_prev);
                        RowVector rhs2_L = -302400.0 * (P_curr - P_prev) * tp_L.h7_inv -
                                               (dD20_dhL * V_curr + dD21_dhL * A_curr + dD22_dhL * J_curr +
                                                dL20_dhL * V_prev + dL21_dhL * A_prev + dL22_dhL * J_prev);

                        RowVector rhs0_R = -3360.0 * (P_next - P_curr) * tp_R.h5_inv -
                                               (dD00_dhR * V_curr + dD01_dhR * A_curr + dD02_dhR * J_curr +
                                                dU00_dhR * V_next + dU01_dhR * A_next + dU02_dhR * J_next);
                        RowVector rhs1_R = -50400.0 * (P_curr - P_next) * tp_R.h6_inv -
                                               (dD10_dhR * V_curr + dD11_dhR * A_curr + dD12_dhR * J_curr +
                                                dU10_dhR * V_next + dU11_dhR * A_next + dU12_dhR * J_next);
                        RowVector rhs2_R = -302400.0 * (P_next - P_curr) * tp_R.h7_inv -
                                               (dD20_dhR * V_curr + dD21_dhR * A_curr + dD22_dhR * J_curr +
                                                dU20_dhR * V_next + dU21_dhR * A_next + dU22_dhR * J_next);

                        gradByTimes(m - 1) += lam_3.dot(rhs0_L) + lam_4.dot(rhs1_L) + lam_5.dot(rhs2_L);
                        gradByTimes(m) += lam_3.dot(rhs0_R) + lam_4.dot(rhs1_R) + lam_5.dot(rhs2_R);
                    }

                    double dr3_dp_next = 840.0 * tp_R.h4_inv;
                    double dr4_dp_next = -10080.0 * tp_R.h5_inv;
                    double dr5_dp_next = 50400.0 * tp_R.h6_inv;
                    RowVector grad_P_next = lam_3 * dr3_dp_next + lam_4 * dr4_dp_next + lam_5 * dr5_dp_next;

                    double dr3_dp_curr = -840.0 * (tp_R.h4_inv - tp_L.h4_inv);
                    double dr4_dp_curr = 10080.0 * (tp_R.h5_inv + tp_L.h5_inv);
                    double dr5_dp_curr = -50400.0 * (tp_R.h6_inv - tp_L.h6_inv);
                    RowVector grad_P_curr = lam_3 * dr3_dp_curr + lam_4 * dr4_dp_curr + lam_5 * dr5_dp_curr;

                    double dr3_dp_prev = -840.0 * tp_L.h4_inv;
                    double dr4_dp_prev = -10080.0 * tp_L.h5_inv;
                    double dr5_dp_prev = -50400.0 * tp_L.h6_inv;
                    RowVector grad_P_prev = lam_3 * dr3_dp_prev + lam_4 * dr4_dp_prev + lam_5 * dr5_dp_prev;

                    add_point_grad(i + 2, grad_P_next);
                    add_point_grad(i + 1, grad_P_curr);
                    add_point_grad(i, grad_P_prev);
                }

                Eigen::Matrix<double, 3, DIM> correction_start;
                multiplyStoredBlock3x3T_3xN(L_blocks_cache_, 0,
                                            ws_lambda_.template middleRows<3>(0),
                                            correction_start);
                raw_start_grad -= correction_start;

                Eigen::Matrix<double, 3, DIM> correction_end;
                multiplyStoredBlock3x3T_3xN(U_blocks_cache_, num_blocks - 1,
                                            ws_lambda_.template middleRows<3>(3 * (num_blocks - 1)),
                                            correction_end);
                raw_end_grad -= correction_end;
            }

            startGrads.v = raw_start_grad.row(0).transpose();
            startGrads.a = raw_start_grad.row(1).transpose();
            startGrads.j = raw_start_grad.row(2).transpose();

            endGrads.v = raw_end_grad.row(0).transpose();
            endGrads.a = raw_end_grad.row(1).transpose();
            endGrads.j = raw_end_grad.row(2).transpose();
        }
        void precomputeTimePowers()
        {
            int n = static_cast<int>(time_segments_.size());
            time_powers_.resize(n);

            for (int i = 0; i < n; ++i)
            {
                double h = time_segments_[i];
                double iv = 1.0 / h;
                double iv2 = iv * iv;
                double iv3 = iv2 * iv;
                double iv4 = iv3 * iv;

                time_powers_[i].h = h;
                time_powers_[i].h_inv = iv;
                time_powers_[i].h2_inv = iv2;
                time_powers_[i].h3_inv = iv3;
                time_powers_[i].h4_inv = iv4;
                time_powers_[i].h5_inv = iv4 * iv;
                time_powers_[i].h6_inv = iv4 * iv2;
                time_powers_[i].h7_inv = iv4 * iv3;
            }
        }

        void precomputePointDiffs()
        {
            point_diffs_.resize(num_segments_, DIM);
            for (int i = 0; i < num_segments_; ++i)
            {
                point_diffs_.row(i) = spatial_points_.row(i + 1) - spatial_points_.row(i);
            }
        }

        static inline void Inverse3x3(const Eigen::Matrix3d &A, Eigen::Matrix3d &A_inv_out)
        {

            const double a00 = A(0, 0), a01 = A(0, 1), a02 = A(0, 2),
                         a10 = A(1, 0), a11 = A(1, 1), a12 = A(1, 2),
                         a20 = A(2, 0), a21 = A(2, 1), a22 = A(2, 2);

            const double c00 = a11 * a22 - a12 * a21;
            const double c01 = -(a10 * a22 - a12 * a20);
            const double c02 = a10 * a21 - a11 * a20;

            const double c10 = -(a01 * a22 - a02 * a21);
            const double c11 = a00 * a22 - a02 * a20;
            const double c12 = -(a00 * a21 - a01 * a20);

            const double c20 = a01 * a12 - a02 * a11;
            const double c21 = -(a00 * a12 - a02 * a10);
            const double c22 = a00 * a11 - a01 * a10;

            const double det = a00 * c00 + a01 * c01 + a02 * c02;
            const double inv_det = 1.0 / det;

            A_inv_out(0, 0) = c00 * inv_det;
            A_inv_out(0, 1) = c10 * inv_det;
            A_inv_out(0, 2) = c20 * inv_det;

            A_inv_out(1, 0) = c01 * inv_det;
            A_inv_out(1, 1) = c11 * inv_det;
            A_inv_out(1, 2) = c21 * inv_det;

            A_inv_out(2, 0) = c02 * inv_det;
            A_inv_out(2, 1) = c12 * inv_det;
            A_inv_out(2, 2) = c22 * inv_det;
        }

        static inline void setBlock3x3(BlockMatrix3x3Storage &storage, int i, const Eigen::Matrix3d &M)
        {
            storage(i, 0) = M(0, 0);
            storage(i, 1) = M(0, 1);
            storage(i, 2) = M(0, 2);
            storage(i, 3) = M(1, 0);
            storage(i, 4) = M(1, 1);
            storage(i, 5) = M(1, 2);
            storage(i, 6) = M(2, 0);
            storage(i, 7) = M(2, 1);
            storage(i, 8) = M(2, 2);
        }

        inline void MultiplyStoredBlock3x3(const BlockMatrix3x3Storage &A_storage, int idx_a,
                                           const BlockMatrix3x3Storage &B_storage, int idx_b,
                                           Eigen::Matrix3d &C_out) const noexcept
        {
            const double a00 = A_storage(idx_a, 0), a01 = A_storage(idx_a, 1), a02 = A_storage(idx_a, 2);
            const double a10 = A_storage(idx_a, 3), a11 = A_storage(idx_a, 4), a12 = A_storage(idx_a, 5);
            const double a20 = A_storage(idx_a, 6), a21 = A_storage(idx_a, 7), a22 = A_storage(idx_a, 8);
            const double b00 = B_storage(idx_b, 0), b01 = B_storage(idx_b, 1), b02 = B_storage(idx_b, 2);
            const double b10 = B_storage(idx_b, 3), b11 = B_storage(idx_b, 4), b12 = B_storage(idx_b, 5);
            const double b20 = B_storage(idx_b, 6), b21 = B_storage(idx_b, 7), b22 = B_storage(idx_b, 8);
            C_out(0, 0) = a00 * b00 + a01 * b10 + a02 * b20;
            C_out(0, 1) = a00 * b01 + a01 * b11 + a02 * b21;
            C_out(0, 2) = a00 * b02 + a01 * b12 + a02 * b22;
            C_out(1, 0) = a10 * b00 + a11 * b10 + a12 * b20;
            C_out(1, 1) = a10 * b01 + a11 * b11 + a12 * b21;
            C_out(1, 2) = a10 * b02 + a11 * b12 + a12 * b22;
            C_out(2, 0) = a20 * b00 + a21 * b10 + a22 * b20;
            C_out(2, 1) = a20 * b01 + a21 * b11 + a22 * b21;
            C_out(2, 2) = a20 * b02 + a21 * b12 + a22 * b22;
        }

        template <typename BlockOut, typename BlockIn>
        inline void multiplyStoredBlock3x3_3xN(const BlockMatrix3x3Storage &A_storage, int idx,
                                               const BlockIn &B, BlockOut &&C_out) const noexcept
        {
            const double a00 = A_storage(idx, 0), a01 = A_storage(idx, 1), a02 = A_storage(idx, 2);
            const double a10 = A_storage(idx, 3), a11 = A_storage(idx, 4), a12 = A_storage(idx, 5);
            const double a20 = A_storage(idx, 6), a21 = A_storage(idx, 7), a22 = A_storage(idx, 8);
            for (int j = 0; j < DIM; ++j)
            {
                const double b0j = B(0, j);
                const double b1j = B(1, j);
                const double b2j = B(2, j);
                C_out(0, j) = a00 * b0j + a01 * b1j + a02 * b2j;
                C_out(1, j) = a10 * b0j + a11 * b1j + a12 * b2j;
                C_out(2, j) = a20 * b0j + a21 * b1j + a22 * b2j;
            }
        }

        template <typename BlockOut, typename BlockIn>
        inline void multiplyStoredBlock3x3T_3xN(const BlockMatrix3x3Storage &A_storage, int idx,
                                                const BlockIn &B, BlockOut &&C_out) const noexcept
        {
            const double a00 = A_storage(idx, 0), a01 = A_storage(idx, 1), a02 = A_storage(idx, 2);
            const double a10 = A_storage(idx, 3), a11 = A_storage(idx, 4), a12 = A_storage(idx, 5);
            const double a20 = A_storage(idx, 6), a21 = A_storage(idx, 7), a22 = A_storage(idx, 8);
            for (int j = 0; j < DIM; ++j)
            {
                const double b0j = B(0, j);
                const double b1j = B(1, j);
                const double b2j = B(2, j);
                C_out(0, j) = a00 * b0j + a10 * b1j + a20 * b2j;
                C_out(1, j) = a01 * b0j + a11 * b1j + a21 * b2j;
                C_out(2, j) = a02 * b0j + a12 * b1j + a22 * b2j;
            }
        }

        template <typename BlockOut, typename BlockIn>
        inline void subMultiplyStoredBlock3x3T_3xN(const BlockMatrix3x3Storage &A_storage, int idx,
                                                   const BlockIn &B, BlockOut &&C_out) const noexcept
        {
            const double a00 = A_storage(idx, 0), a01 = A_storage(idx, 1), a02 = A_storage(idx, 2);
            const double a10 = A_storage(idx, 3), a11 = A_storage(idx, 4), a12 = A_storage(idx, 5);
            const double a20 = A_storage(idx, 6), a21 = A_storage(idx, 7), a22 = A_storage(idx, 8);
            for (int j = 0; j < DIM; ++j)
            {
                const double b0j = B(0, j);
                const double b1j = B(1, j);
                const double b2j = B(2, j);
                C_out(0, j) -= a00 * b0j + a10 * b1j + a20 * b2j;
                C_out(1, j) -= a01 * b0j + a11 * b1j + a21 * b2j;
                C_out(2, j) -= a02 * b0j + a12 * b1j + a22 * b2j;
            }
        }

        template <typename BlockOut, typename BlockIn>
        inline void subMultiplyStoredBlock3x3_3xN(const BlockMatrix3x3Storage &A_storage, int idx,
                                                  const BlockIn &B, BlockOut &&C_out) const noexcept
        {
            const double a00 = A_storage(idx, 0), a01 = A_storage(idx, 1), a02 = A_storage(idx, 2);
            const double a10 = A_storage(idx, 3), a11 = A_storage(idx, 4), a12 = A_storage(idx, 5);
            const double a20 = A_storage(idx, 6), a21 = A_storage(idx, 7), a22 = A_storage(idx, 8);
            for (int j = 0; j < DIM; ++j)
            {
                const double b0j = B(0, j);
                const double b1j = B(1, j);
                const double b2j = B(2, j);
                C_out(0, j) -= a00 * b0j + a01 * b1j + a02 * b2j;
                C_out(1, j) -= a10 * b0j + a11 * b1j + a12 * b2j;
                C_out(2, j) -= a20 * b0j + a21 * b1j + a22 * b2j;
            }
        }

    private:
        void solveInternalDerivatives(const CoefficientMatrix &P,
                                      CoefficientMatrix &p_out,
                                      CoefficientMatrix &q_out,
                                      CoefficientMatrix &s_out)
        {
            const int n = static_cast<int>(P.rows());
            p_out.resize(n, DIM);
            q_out.resize(n, DIM);
            s_out.resize(n, DIM);

            p_out.row(0) = boundary_.start_velocity.transpose();
            q_out.row(0) = boundary_.start_acceleration.transpose();
            s_out.row(0) = boundary_.start_jerk.transpose();

            p_out.row(n - 1) = boundary_.end_velocity.transpose();
            q_out.row(n - 1) = boundary_.end_acceleration.transpose();
            s_out.row(n - 1) = boundary_.end_jerk.transpose();

            const int num_blocks = n - 2;
            if (num_blocks <= 0)
                return;

            Eigen::Matrix<double, 3, DIM> B_left, B_right;
            B_left.row(0) = boundary_.start_velocity.transpose();
            B_left.row(1) = boundary_.start_acceleration.transpose();
            B_left.row(2) = boundary_.start_jerk.transpose();

            B_right.row(0) = boundary_.end_velocity.transpose();
            B_right.row(1) = boundary_.end_acceleration.transpose();
            B_right.row(2) = boundary_.end_jerk.transpose();

            U_blocks_cache_.resize(num_blocks, 9);
            D_inv_cache_.resize(num_blocks, 9);
            L_blocks_cache_.resize(num_blocks, 9);
            D_inv_T_mul_L_next_T_cache_.resize(std::max(0, num_blocks - 1), 9);
            ws_rhs_mod_.resize(num_blocks * 3, DIM);

            for (int i = 0; i < num_blocks; ++i)
            {
                const int k = i + 2;
                const auto &tp_L = time_powers_[k - 2];
                const auto &tp_R = time_powers_[k - 1];

                Eigen::Matrix<double, 1, DIM> r3 = 840.0 * ((P.row(k) - P.row(k - 1)) * tp_R.h4_inv +
                                                            (P.row(k - 1) - P.row(k - 2)) * tp_L.h4_inv);
                Eigen::Matrix<double, 1, DIM> r4 = 10080.0 * ((P.row(k - 1) - P.row(k)) * tp_R.h5_inv +
                                                              (P.row(k - 1) - P.row(k - 2)) * tp_L.h5_inv);
                Eigen::Matrix<double, 1, DIM> r5 = 50400.0 * ((P.row(k) - P.row(k - 1)) * tp_R.h6_inv +
                                                              (P.row(k - 1) - P.row(k - 2)) * tp_L.h6_inv);
                Eigen::Matrix<double, 3, DIM> r;
                r.row(0) = r3;
                r.row(1) = r4;
                r.row(2) = r5;

                Eigen::Matrix3d D;
                D << 480.0 * (tp_L.h3_inv + tp_R.h3_inv), 120.0 * (tp_R.h2_inv - tp_L.h2_inv), 16.0 * (tp_L.h_inv + tp_R.h_inv),
                    5400.0 * (tp_L.h4_inv - tp_R.h4_inv), -1200.0 * (tp_L.h3_inv + tp_R.h3_inv), 120.0 * (tp_L.h2_inv - tp_R.h2_inv),
                    25920.0 * (tp_L.h5_inv + tp_R.h5_inv), 5400.0 * (tp_R.h4_inv - tp_L.h4_inv), 480.0 * (tp_L.h3_inv + tp_R.h3_inv);

                Eigen::Matrix3d L;
                L << 360.0 * tp_L.h3_inv, 60.0 * tp_L.h2_inv, 4.0 * tp_L.h_inv,
                    4680.0 * tp_L.h4_inv, 840.0 * tp_L.h3_inv, 60.0 * tp_L.h2_inv,
                    24480.0 * tp_L.h5_inv, 4680.0 * tp_L.h4_inv, 360.0 * tp_L.h3_inv;
                setBlock3x3(L_blocks_cache_, i, L);

                Eigen::Matrix3d U;
                U << 360.0 * tp_R.h3_inv, -60.0 * tp_R.h2_inv, 4.0 * tp_R.h_inv,
                    -4680.0 * tp_R.h4_inv, 840.0 * tp_R.h3_inv, -60.0 * tp_R.h2_inv,
                    24480.0 * tp_R.h5_inv, -4680.0 * tp_R.h4_inv, 360.0 * tp_R.h3_inv;
                setBlock3x3(U_blocks_cache_, i, U);

                if (i == 0)
                {
                    r.noalias() -= L * B_left;
                }
                else
                {
                    Eigen::Matrix3d X;
                    MultiplyStoredBlock3x3(D_inv_cache_, i - 1, U_blocks_cache_, i - 1, X);
                    Eigen::Matrix<double, 3, DIM> Y;
                    multiplyStoredBlock3x3_3xN(D_inv_cache_, i - 1, ws_rhs_mod_.template middleRows<3>(3 * (i - 1)), Y);
                    D.noalias() -= L * X;
                    r.noalias() -= L * Y;
                }

                if (i == num_blocks - 1)
                {
                    r.noalias() -= U * B_right;
                }

                Eigen::Matrix3d D_inv;
                Inverse3x3(D, D_inv);
                setBlock3x3(D_inv_cache_, i, D_inv);

                if (i > 0)
                {
                    Eigen::Matrix3d L_mul_D_prev_inv;
                    MultiplyStoredBlock3x3(L_blocks_cache_, i, D_inv_cache_, i - 1, L_mul_D_prev_inv);
                    setBlock3x3(D_inv_T_mul_L_next_T_cache_, i - 1, L_mul_D_prev_inv.transpose());
                }
                ws_rhs_mod_.template middleRows<3>(3 * i) = r;
            }

            ws_solution_.resize(num_blocks * 3, DIM);

            multiplyStoredBlock3x3_3xN(D_inv_cache_, num_blocks - 1,
                                       ws_rhs_mod_.template middleRows<3>(3 * (num_blocks - 1)),
                                       ws_solution_.template middleRows<3>(3 * (num_blocks - 1)));

            for (int i = num_blocks - 2; i >= 0; --i)
            {
                auto sol_block = ws_solution_.template middleRows<3>(3 * i);
                auto sol_next = ws_solution_.template middleRows<3>(3 * (i + 1));
                auto rhs_i = ws_rhs_mod_.template middleRows<3>(3 * i);

                const double u00 = U_blocks_cache_(i, 0), u01 = U_blocks_cache_(i, 1), u02 = U_blocks_cache_(i, 2);
                const double u10 = U_blocks_cache_(i, 3), u11 = U_blocks_cache_(i, 4), u12 = U_blocks_cache_(i, 5);
                const double u20 = U_blocks_cache_(i, 6), u21 = U_blocks_cache_(i, 7), u22 = U_blocks_cache_(i, 8);
                Eigen::Matrix<double, 3, DIM> rhs_temp;
                for (int j = 0; j < DIM; ++j)
                {
                    const double s0 = sol_next(0, j), s1 = sol_next(1, j), s2 = sol_next(2, j);
                    rhs_temp(0, j) = rhs_i(0, j) - (u00 * s0 + u01 * s1 + u02 * s2);
                    rhs_temp(1, j) = rhs_i(1, j) - (u10 * s0 + u11 * s1 + u12 * s2);
                    rhs_temp(2, j) = rhs_i(2, j) - (u20 * s0 + u21 * s1 + u22 * s2);
                }
                multiplyStoredBlock3x3_3xN(D_inv_cache_, i, rhs_temp, sol_block);
            }

            for (int i = 0; i < num_blocks; ++i)
            {
                const int row = i + 1;
                p_out.row(row) = ws_solution_.row(3 * i);
                q_out.row(row) = ws_solution_.row(3 * i + 1);
                s_out.row(row) = ws_solution_.row(3 * i + 2);
            }
        }

        void solveSepticInPlace()
        {
            const int n = num_segments_;

            solveInternalDerivatives(spatial_points_, internal_vel_, internal_acc_, internal_jerk_);


            for (int i = 0; i < n; ++i)
            {
                const auto &tp = time_powers_[i];

                const RowVector c0 = spatial_points_.row(i);
                const RowVector c1 = internal_vel_.row(i);
                const RowVector c2 = internal_acc_.row(i) * 0.5;
                const RowVector c3 = internal_jerk_.row(i) / 6.0;

                const RowVector &V_curr = internal_vel_.row(i);
                const RowVector &V_next = internal_vel_.row(i + 1);
                const RowVector &A_curr = internal_acc_.row(i);
                const RowVector &A_next = internal_acc_.row(i + 1);
                const RowVector &J_curr = internal_jerk_.row(i);
                const RowVector &J_next = internal_jerk_.row(i + 1);
                const RowVector P_diff = -point_diffs_.row(i);

                const RowVector c4 = -(210.0 * P_diff * tp.h4_inv +
                                           120.0 * V_curr * tp.h3_inv +
                                           90.0 * V_next * tp.h3_inv +
                                           30.0 * A_curr * tp.h2_inv -
                                           15.0 * A_next * tp.h2_inv +
                                           4.0 * J_curr * tp.h_inv +
                                           J_next * tp.h_inv) /
                                         6.0;

                const RowVector c5 = (168.0 * P_diff * tp.h5_inv +
                                          90.0 * V_curr * tp.h4_inv +
                                          78.0 * V_next * tp.h4_inv +
                                          20.0 * A_curr * tp.h3_inv -
                                          14.0 * A_next * tp.h3_inv +
                                          2.0 * J_curr * tp.h2_inv +
                                          J_next * tp.h2_inv) /
                                         2.0;

                const RowVector c6 = -(420.0 * P_diff * tp.h6_inv +
                                           216.0 * V_curr * tp.h5_inv +
                                           204.0 * V_next * tp.h5_inv +
                                           45.0 * A_curr * tp.h4_inv -
                                           39.0 * A_next * tp.h4_inv +
                                           4.0 * J_curr * tp.h3_inv +
                                           3.0 * J_next * tp.h3_inv) /
                                         6.0;

                const RowVector c7 = (120.0 * P_diff * tp.h7_inv +
                                          60.0 * V_curr * tp.h6_inv +
                                          60.0 * V_next * tp.h6_inv +
                                          12.0 * A_curr * tp.h5_inv -
                                          12.0 * A_next * tp.h5_inv +
                                          J_curr * tp.h4_inv +
                                          J_next * tp.h4_inv) /
                                         6.0;

                trajectory_.coefficients_.row(i * 8 + 0) = c0;
                trajectory_.coefficients_.row(i * 8 + 1) = c1;
                trajectory_.coefficients_.row(i * 8 + 2) = c2;
                trajectory_.coefficients_.row(i * 8 + 3) = c3;
                trajectory_.coefficients_.row(i * 8 + 4) = c4;
                trajectory_.coefficients_.row(i * 8 + 5) = c5;
                trajectory_.coefficients_.row(i * 8 + 6) = c6;
                trajectory_.coefficients_.row(i * 8 + 7) = c7;
            }
        }

    };

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
