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
 * @brief Owning piecewise polynomials and a shared minimum-derivative spline kernel. */
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
#include <type_traits>

// Piecewise polynomials, sampling and shared physical data types.

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

// Unit-interval Hermite maps and orthogonal energy constants.
// Endpoint columns are [p, v, ...] at the start followed by those at the end.
// H maps endpoint derivatives to ascending powers; Q = H^T G H integrates the
// squared S-th derivative on [0,1]. E converts that derivative to shifted
// Legendre modes with squared norms 1/(2j+1). These compile-time tables require
// no per-instance storage or initialization.

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

// Cached block tridiagonal factorization and solves.

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

// Shared minimum-derivative spline construction, energy and adjoint pipeline.

// Keep compile-time index visitors inside their numeric kernel on GCC and Clang.
#if defined(__GNUC__) || defined(__clang__)
#define SPLINE_INLINE_VISITOR __attribute__((always_inline))
#else
#define SPLINE_INLINE_VISITOR
#endif
namespace SplineTrajectory {
/** @brief Minimum squared S-th derivative spline, specialized at compile time by order.
 * @tparam DIM Positive spatial dimension. @tparam S Minimum derivative order: 2, 3 or 4.
 * @note Waypoints use caller-defined coordinates and durations use seconds. One object owns
 * its polynomial, factors and workspaces. Updates and backward require exclusive access;
 * const queries on an unchanged object are reentrant. Copies own independent storage.
 * Invalid input or a failed factorization clears the polynomial. No regularization is applied. */
template<int DIM, int S>
class MinDerivativeSplineND
{
    static_assert(DIM > 0 && S >= 2 && S <= 4, "Supported spline dimensions are positive; orders are 2, 3 and 4");
    // M free derivatives per knot, K endpoint values / power coefficients, E polynomial degree.
    static constexpr int M = S - 1, K = 2 * S, E = 2 * S - 1;
    using Kernel = detail::HermiteKernel<S>;
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    using Vector = Eigen::Matrix<double, DIM, 1>;
    using RowVector = Eigen::Matrix<double, 1, DIM>;
    static constexpr int kMatrixOptions = DIM == 1 ? Eigen::ColMajor : Eigen::RowMajor;
    using CoefficientMatrix = Eigen::Matrix<double, Eigen::Dynamic, DIM, kMatrixOptions>;
    using Polynomial = PPolyND<DIM, K>;
    using BoundaryGradient = SplineTrajectory::BoundaryGradient<DIM>;
    using BoundaryGradients = BoundaryGradientPair<DIM>;
    using Gradients = SplineGradients<DIM>;
    static constexpr int kDimension = DIM, kDegree = E, kCoefficientCount = K, kMinDerivativeOrder = S;
private:
    using Block = Eigen::Matrix<double, M, M, (M == 1 ? Eigen::ColMajor : Eigen::RowMajor)>;
    std::vector<double> times_;
    CoefficientMatrix points_;
    BoundaryConditions<DIM> boundary_;
    Polynomial polynomial_;
    double start_ = 0.0;
    int segments_ = 0;
    // Knot-major derivative rows keep each v/a/j block and all spatial coordinates together.
    CoefficientMatrix derivatives_, adjoint_;
    std::vector<std::array<double, K>> inverse_powers_;
    detail::BlockLDLT<M> factor_;

public:
    /** @brief Construct an empty, invalid spline. */
    MinDerivativeSplineND() = default;
    /** @brief Copy interpolation inputs expressed as absolute knots.
     * @param knots N+1 finite, strictly increasing times in seconds.
     * @param points N+1 waypoint rows. @param boundary Physical endpoint derivatives. */
    MinDerivativeSplineND(const std::vector<double> &knots, const CoefficientMatrix &points,
                          const BoundaryConditions<DIM> &boundary = {})
    { update(knots, points, boundary); }
    /** @brief Copy interpolation inputs expressed as durations.
     * @param times N positive finite durations in seconds. @param points N+1 waypoint rows.
     * @param start Common time origin in seconds. @param boundary Physical endpoint derivatives. */
    MinDerivativeSplineND(const std::vector<double> &times, const CoefficientMatrix &points,
                          double start, const BoundaryConditions<DIM> &boundary = {})
    { update(times, points, start, boundary); }
    /** @brief Replace data using absolute knots; invalid input clears the polynomial.
     * @param knots Strictly increasing times. @param points Waypoints in the caller's frame.
     * @param boundary Physical endpoint derivatives. @note Invalidates all borrowed views. */
    void update(const std::vector<double> &knots, const CoefficientMatrix &points,
                const BoundaryConditions<DIM> &boundary = {})
    {
        detail::durationsFromKnots(knots, times_, start_);
        points_ = points; boundary_ = boundary;
        rebuild();
    }
    /** @brief Replace data using durations, reusing storage for an unchanged topology.
     * @param times Positive durations in seconds. @param points N+1 waypoint rows.
     * @param start Common time origin. @param boundary Physical endpoint derivatives.
     * @note Invalidates borrowed views; invalid input or numeric breakdown clears the polynomial. */
    void update(const std::vector<double> &times, const CoefficientMatrix &points, double start,
                const BoundaryConditions<DIM> &boundary = {})
    {
        times_ = times; points_ = points; start_ = start; boundary_ = boundary;
        rebuild();
    }
    /** @brief Inspect construction validity. @return True only for a finite published polynomial. */
    bool isValid() const { return polynomial_.isValid(); }
    /** @brief Spatial dimension. @return DIM. */
    int dimension() const { return DIM; }
    /** @brief Absolute start time. @return Seconds, or zero for an invalid polynomial. */
    double startTime() const { return polynomial_.startTime(); }
    /** @brief Absolute end time. @return Seconds, or zero for an invalid polynomial. */
    double endTime() const { return polynomial_.endTime(); }
    /** @brief Total duration. @return Seconds, or zero for an invalid polynomial. */
    double duration() const { return polynomial_.duration(); }
    /** @brief Input waypoint count. @return Number of stored rows. */
    size_t numPoints() const { return static_cast<size_t>(points_.rows()); }
    /** @brief Published segment count. @return Zero for an invalid polynomial. */
    int numSegments() const { return polynomial_.numSegments(); }
    /** @brief Borrow physical waypoints. @return Storage valid until update, assignment or move. */
    const CoefficientMatrix &waypoints() const { return points_; }
    /** @brief Borrow input durations. @return Seconds, valid until update, assignment or move. */
    const std::vector<double> &durations() const { return times_; }
    /** @brief Borrow absolute knots. @return Seconds, valid until update, assignment or move. */
    const std::vector<double> &breakpoints() const { return polynomial_.breakpoints(); }
    /** @brief Borrow endpoint derivatives. @return Physical values valid until the next mutation. */
    const BoundaryConditions<DIM> &boundary() const { return boundary_; }
    /** @brief Borrow the polynomial. @return View valid until update, assignment or move. */
    const Polynomial &polynomial() const & { return polynomial_; }
    /** @brief Prevent a borrowed polynomial from outliving a temporary spline. */
    const Polynomial &polynomial() const && = delete;
    /** @brief Copy an independent owning polynomial. @return Snapshot with independent lifetime. */
    Polynomial copyPolynomial() const { return polynomial_; }

    /** @brief Build physical-time basis rows through the fifth derivative.
     * @param t Local time in seconds. @param[out] p Position basis. @param[out] v Velocity basis.
     * @param[out] a Acceleration basis. @param[out] j Jerk basis. @param[out] s Snap basis.
     * @param[out] c Crackle basis. Derivatives beyond the polynomial degree are zero. */
    static void computeBasisFunctions(double t, Eigen::Matrix<double,1,K> &p,
        Eigen::Matrix<double,1,K> &v, Eigen::Matrix<double,1,K> &a,
        Eigen::Matrix<double,1,K> &j, Eigen::Matrix<double,1,K> &s, Eigen::Matrix<double,1,K> &c)
    {
        std::array<Eigen::Matrix<double,1,K> *,6> rows{{&p,&v,&a,&j,&s,&c}};
        detail::staticFor<0,6>([&](auto derivative) SPLINE_INLINE_VISITOR {
            constexpr int r = decltype(derivative)::value;
            rows[r]->setZero();
            double power = 1.0;
            detail::staticFor<r,K>([&](auto degree) SPLINE_INLINE_VISITOR {
                constexpr int k = decltype(degree)::value;
                (*rows[r])(k) = detail::factorial(k) / detail::factorial(k-r) * power;
                power *= t;
            });
        });
    }

    /** @brief Integrate squared minimum derivative using a nonnegative orthogonal expansion.
     * @return Energy in physical units; NaN when the spline is invalid. */
    double energy() const
    {
        if (!isValid()) return std::numeric_limits<double>::quiet_NaN();
        double value = 0.0;
        for (int i = 0; i < segments_; ++i)
        {
            double segment = 0.0;
            detail::staticFor<0,S>([&](auto mode) SPLINE_INLINE_VISITOR {
                constexpr int r = decltype(mode)::value;
                segment += (1.0/(2*r+1))*energyMode<r>(i).squaredNorm();
            });
            value += times_[i]*segment;
        }
        return value;
    }
    /** @brief Differentiate energy holding physical coefficients fixed.
     * @param[out] gradient N direct duration partials; existing storage is reused. */
    void energyDurationPartials(Eigen::VectorXd &gradient) const
    {
        requireValid(); gradient.resize(segments_);
        for (int i = 0; i < segments_; ++i)
            gradient(i) = derivative<S>(i, times_[i]).squaredNorm();
    }
    /** @brief Allocate direct duration partials. @return N physical partial derivatives. */
    Eigen::VectorXd energyDurationPartials() const
    { Eigen::VectorXd result; energyDurationPartials(result); return result; }
    /** @brief Differentiate energy holding durations fixed.
     * @param[out] gradient Physical power-coefficient partials in piece-major order. */
    void energyCoefficientPartials(CoefficientMatrix &gradient) const
    {
        requireValid(); gradient.setZero(segments_*K, DIM);
        for (int i = 0; i < segments_; ++i)
        {
            const auto modes = energyModes(i);
            double power = times_[i];
            detail::staticFor<0,S>([&](auto column) SPLINE_INLINE_VISITOR {
                constexpr int j = decltype(column)::value;
                RowVector row = RowVector::Zero();
                detail::staticFor<0,j+1>([&](auto mode) SPLINE_INLINE_VISITOR {
                    constexpr int r = decltype(mode)::value;
                    row += (2.0 * Kernel::E[r][j] / (2*r+1)) * modes.row(r);
                });
                gradient.row(i*K+S+j) = power * row;
                power *= times_[i];
            });
        }
    }
    /** @brief Allocate coefficient partials. @return Physical coefficient gradient matrix. */
    CoefficientMatrix energyCoefficientPartials() const
    { CoefficientMatrix result; energyCoefficientPartials(result); return result; }
    /** @brief Evaluate the Hamiltonian duration gradient of the optimized spline.
     * @param[out] gradient Total physical duration gradients, without an adjoint solve. */
    void energyDurationGradient(Eigen::VectorXd &gradient) const
    {
        requireValid(); gradient.resize(segments_);
        for (int i = 0; i < segments_; ++i)
        {
            const auto coefficients = polynomial_.coefficients_.template middleRows<K>(i*K);
            double value = -std::pow(detail::factorial(S),2) * coefficients.row(S).squaredNorm();
            detail::staticFor<1,S>([&](auto order) SPLINE_INLINE_VISITOR {
                constexpr int r = decltype(order)::value;
                constexpr double scale = ((S-r+1)%2 ? -2.0 : 2.0) * detail::factorial(r) * detail::factorial(K-r);
                value += scale * coefficients.row(r).dot(coefficients.row(K-r));
            });
            gradient(i) = value;
        }
    }
    /** @brief Allocate total duration gradients. @return N derivatives in physical units. */
    Eigen::VectorXd energyDurationGradient() const
    { Eigen::VectorXd result; energyDurationGradient(result); return result; }
    /** @brief Differentiate energy at interior waypoints using highest-derivative jumps.
     * @param[out] gradient N-1 waypoint gradients; endpoint positions are excluded. */
    void energyWaypointGradient(CoefficientMatrix &gradient) const
    {
        requireValid(); gradient.resize(segments_-1, DIM);
        constexpr double scale = (S%2 ? 2.0 : -2.0) * detail::factorial(E);
        for (int i = 1; i < segments_; ++i)
            gradient.row(i-1) = scale * (polynomial_.coefficients_.row(i*K-1) - polynomial_.coefficients_.row((i+1)*K-1));
    }
    /** @brief Allocate interior waypoint gradients. @return N-1 physical gradient rows. */
    CoefficientMatrix energyWaypointGradient() const
    { CoefficientMatrix result; energyWaypointGradient(result); return result; }
    /** @brief Evaluate energy boundary gradients through integration by parts.
     * @return Start/end position and derivative gradients; unused orders remain zero. */
    BoundaryGradients energyBoundaryGradient() const
    {
        requireValid(); BoundaryGradients result;
        detail::staticFor<0,S>([&](auto order) SPLINE_INLINE_VISITOR {
            constexpr int r = decltype(order)::value;
            constexpr double scale = (S-r)%2 ? -2.0 : 2.0;
            boundaryComponent<r>(result.start) = scale * detail::factorial(E-r) * polynomial_.coefficients_.row(E-r).transpose();
            boundaryComponent<r>(result.end) = -scale * derivative<E-r>(segments_-1, times_.back()).transpose();
        });
        return result;
    }
    /** @brief Compute all analytic energy gradients without a general adjoint.
     * @param[out] gradient Overwritten physical gradients; fixed-topology storage is reused. */
    void energyGradient(Gradients &gradient) const
    {
        energyWaypointGradient(gradient.inner_points);
        energyDurationGradient(gradient.durations);
        const auto boundary = energyBoundaryGradient();
        gradient.start = boundary.start; gradient.end = boundary.end;
    }
    /** @brief Allocate all energy gradients. @return Independently owned gradient buffers. */
    Gradients energyGradient() const
    { Gradients result; energyGradient(result); return result; }

    /** @brief Pull back coefficient and independent duration partials through interpolation.
     * @param coefficients Physical local-power partials, K*N rows in the polynomial's order.
     * @param times Independent duration partials holding those coefficients fixed.
     * @param[out] gradient Overwritten physical gradients; prepared storage is reused.
     * @note Requires a valid spline and matching shapes. The factorization is shared with construction. */
    void backward(const CoefficientMatrix &coefficients, const Eigen::VectorXd &times, Gradients &gradient)
    {
        if (!isValid() || coefficients.rows() != segments_*K || times.size() != segments_)
            throw std::invalid_argument("Adjoint requires a valid spline and matching partial shapes");
        gradient.resetTopology(segments_); gradient.setZero(); gradient.durations = times;
        adjoint_.setZero();

        // Transpose the local Hermite map; time derivatives hold endpoint physical derivatives fixed.
        for (int i = 0; i < segments_; ++i)
        {
            const auto &inv = inverse_powers_[i];
            double time_partial = 0.0;
            detail::staticFor<S,K>([&](auto degree) SPLINE_INLINE_VISITOR {
                constexpr int k = decltype(degree)::value;
                time_partial -= k * coefficients.row(i*K+k).dot(polynomial_.coefficients_.row(i*K+k));
            });
            detail::staticFor<1,K>([&](auto column) SPLINE_INLINE_VISITOR {
                constexpr int j = decltype(column)::value, r = j%S;
                RowVector partial = coefficientPullback<j>(i, coefficients, std::make_index_sequence<S>{});
                if constexpr (r == 0)
                {
                    addPoint(gradient,i,coefficients.row(i*K)-partial);
                    addPoint(gradient,i+1,partial);
                }
                else
                {
                    constexpr int side = j/S;
                    time_partial += r*partial.dot(derivatives_.row((i+side)*M+r-1));
                    if constexpr (side == 0)
                        partial += (1.0/detail::factorial(r))*coefficients.row(i*K+r);
                    adjoint_.row((i+side)*M+r-1) += partial;
                }
            });
            gradient.durations(i) += inv[1]*time_partial;
        }
        detail::staticFor<1,S>([&](auto order) SPLINE_INLINE_VISITOR {
            constexpr int r = decltype(order)::value;
            boundaryComponent<r>(gradient.start) = adjoint_.row(r-1).transpose();
            boundaryComponent<r>(gradient.end) = adjoint_.row(segments_*M+r-1).transpose();
        });
        adjoint_.template topRows<M>().setZero();
        adjoint_.template bottomRows<M>().setZero();
        factor_.solve(adjoint_, M);

        // The adjoint is itself a Hermite polynomial with zero waypoint values.
        // Integration by parts gives its endpoint forces; the polarized Hamiltonian
        // contracts the time derivative without forming dA/dT or a dense local Hessian.
        for (int i = 0; i < segments_; ++i)
        {
            Eigen::Matrix<double,S,DIM,kMatrixOptions> high;
            detail::staticFor<S,K>([&](auto degree) SPLINE_INLINE_VISITOR {
                constexpr int k = decltype(degree)::value;
                high.row(k-S) = derivativeCoefficient<k>(i, adjoint_, std::make_index_sequence<2*M>{});
            });
            constexpr double position_scale = (S%2 ? 1.0 : -1.0)*detail::factorial(E);
            const RowVector point = position_scale*high.row(S-1);
            addPoint(gradient,i,point); addPoint(gradient,i+1,-point);
            double time_partial = detail::factorial(S)*detail::factorial(S) *
                polynomial_.coefficients_.row(i*K+S).dot(high.row(0));
            detail::staticFor<1,S>([&](auto order) SPLINE_INLINE_VISITOR {
                constexpr int r = decltype(order)::value;
                constexpr double sign = (S-r+1)%2 ? -1.0 : 1.0;
                time_partial -= sign*detail::factorial(K-r) * (
                    adjoint_.row(i*M+r-1).dot(polynomial_.coefficients_.row(i*K+K-r)) +
                    detail::factorial(r)*polynomial_.coefficients_.row(i*K+r).dot(high.row(S-r)));
                constexpr double boundary_sign = (S-r)%2 ? -1.0 : 1.0;
                if (i == 0)
                    boundaryComponent<r>(gradient.start) -= boundary_sign*detail::factorial(E-r)*high.row(S-1-r).transpose();
                if (i+1 == segments_)
                {
                    RowVector end = RowVector::Zero();
                    detail::staticFor<0,r+1>([&](auto index) SPLINE_INLINE_VISITOR {
                        constexpr int k = E-decltype(index)::value;
                        end = end*times_[i] + (detail::factorial(k)/detail::factorial(k-E+r))*high.row(k-S);
                    });
                    boundaryComponent<r>(gradient.end) += boundary_sign*end.transpose();
                }
            });
            gradient.durations(i) += time_partial;
        }
    }
    /** @brief Allocate and return a coefficient-map pullback.
     * @param coefficients Physical coefficient partials. @param times Independent duration partials.
     * @return Independently owned physical gradients. */
    Gradients backward(const CoefficientMatrix &coefficients, const Eigen::VectorXd &times)
    { Gradients result; backward(coefficients,times,result); return result; }

private:
    /** @brief Require a published polynomial before numeric gradient queries. */
    void requireValid() const
    { if (!isValid()) throw std::logic_error("Energy gradients require a valid spline"); }
    /** @brief Select a physical boundary gradient at compile time.
     * @param[in,out] value Boundary gradient. @return Mutable component of order R. */
    template<int R> static Vector &boundaryComponent(BoundaryGradient &value)
    {
        if constexpr (R == 0) return value.p;
        else if constexpr (R == 1) return value.v;
        else if constexpr (R == 2) return value.a;
        else return value.j;
    }
    /** @brief Add a waypoint partial to its public output location.
     * @param[in,out] gradient Physical output gradients. @param index Knot index.
     * @param value Row to accumulate in the caller's coordinates. */
    void addPoint(Gradients &gradient, int index, const RowVector &value) const
    {
        if (index == 0) gradient.start.p += value.transpose();
        else if (index == segments_) gradient.end.p += value.transpose();
        else gradient.inner_points.row(index-1) += value;
    }
    /** @brief Evaluate a fixed Hermite derivative row using scalar arithmetic.
     * @param segment Valid piece. @param values Knot-major physical derivative rows.
     * @param indices Compile-time endpoint derivative indices. @return High power coefficient.
     * @note Position differences are added separately; adjoint waypoint values are identically zero. */
    template<int Degree, std::size_t... Index>
    EIGEN_ALWAYS_INLINE RowVector derivativeCoefficient(int segment, const CoefficientMatrix &values,
                                                       std::index_sequence<Index...> indices) const
    {
        (void)indices;
        const auto &inv = inverse_powers_[segment];
        RowVector result;
        for (int d=0; d<DIM; ++d)
            result(d) = (((Kernel::H[Degree][(Index/M)*S+Index%M+1]*inv[Degree-Index%M-1]) *
                          values((segment+Index/M)*M+Index%M,d)) + ...);
        return result;
    }
    /** @brief Evaluate one row of the transposed Hermite map without expression temporaries.
     * @param segment Valid piece. @param partials Physical coefficient partials.
     * @param indices High polynomial degrees. @return Endpoint partial from high coefficients. */
    template<int Column, std::size_t... Index>
    EIGEN_ALWAYS_INLINE RowVector coefficientPullback(int segment, const CoefficientMatrix &partials,
                                                     std::index_sequence<Index...> indices) const
    {
        (void)indices;
        const auto &inv = inverse_powers_[segment];
        RowVector result;
        for (int d=0; d<DIM; ++d)
            result(d) = (((Kernel::H[S+Index][Column]*inv[S+Index-Column%S]) *
                          partials(segment*K+S+Index,d)) + ...);
        return result;
    }
    /** @brief Evaluate one derivative with Horner's rule in local physical time.
     * @param segment Valid segment index. @param t Local seconds. @return Physical derivative row. */
    template<int R> RowVector derivative(int segment, double t) const
    {
        RowVector value = RowVector::Zero();
        detail::staticFor<0,K-R>([&](auto index) SPLINE_INLINE_VISITOR {
            constexpr int k = E-decltype(index)::value;
            value = value*t + (detail::factorial(k)/detail::factorial(k-R)) * polynomial_.coefficients_.row(segment*K+k);
        });
        return value;
    }
    /** @brief Evaluate one shifted-Legendre mode with Horner's rule.
     * @param segment Valid segment index. @return Mode R in physical derivative units. */
    template<int R> EIGEN_ALWAYS_INLINE RowVector energyMode(int segment) const
    {
        RowVector mode = Kernel::E[R][S-1]*polynomial_.coefficients_.row(segment*K+E);
        detail::staticFor<0,S-1-R>([&](auto index) SPLINE_INLINE_VISITOR {
            constexpr int j = S-2-decltype(index)::value;
            mode = mode*times_[segment] + Kernel::E[R][j]*polynomial_.coefficients_.row(segment*K+S+j);
        });
        double power = 1.0;
        for (int j = 0; j < R; ++j) power *= times_[segment];
        return power*mode;
    }
    /** @brief Gather orthogonal energy modes for the coefficient partials.
     * @param segment Valid segment index. @return Modes in physical derivative units. */
    Eigen::Matrix<double,S,DIM,kMatrixOptions> energyModes(int segment) const
    {
        Eigen::Matrix<double,S,DIM,kMatrixOptions> modes;
        detail::staticFor<0,S>([&](auto index) SPLINE_INLINE_VISITOR {
            constexpr int r = decltype(index)::value;
            modes.row(r) = energyMode<r>(segment);
        });
        return modes;
    }
    /** @brief Validate inputs, allocate once per topology, then assemble, solve and publish.
     * @note Failed numeric construction clears the polynomial and never exposes stale coefficients. */
    void rebuild()
    {
        if (!polynomial_.template beginSpline<S>(times_,points_,boundary_,start_,K))
        { segments_ = 0; return; }
        segments_ = static_cast<int>(times_.size());
        inverse_powers_.resize(segments_);
        derivatives_.resize((segments_+1)*M,DIM);
        adjoint_.resize((segments_+1)*M,DIM);
        factor_.resize(segments_-1);
        for (int i = 0; i < segments_; ++i)
        {
            auto &inv = inverse_powers_[i]; inv[0] = 1.0; inv[1] = 1.0/times_[i];
            for (int k = 2; k < K; ++k) inv[k] = inv[k-1]*inv[1];
        }
        derivatives_.row(0) = boundary_.start_velocity.transpose();
        derivatives_.row(segments_*M) = boundary_.end_velocity.transpose();
        if constexpr (S >= 3)
        {
            derivatives_.row(1) = boundary_.start_acceleration.transpose();
            derivatives_.row(segments_*M+1) = boundary_.end_acceleration.transpose();
        }
        if constexpr (S >= 4)
        {
            derivatives_.row(2) = boundary_.start_jerk.transpose();
            derivatives_.row(segments_*M+2) = boundary_.end_jerk.transpose();
        }
        if (!assemble()) { polynomial_.clear(); return; }
        factor_.solve(derivatives_,M);
        writeCoefficients();
        polynomial_.finishSpline();
    }
    /** @brief Assemble stationarity of the exact segment energy, in physical derivative coordinates.
     * @return True if every symmetric Schur block factors; false on numeric breakdown. */
    bool assemble()
    {
        for (int knot = 1; knot < segments_; ++knot)
        {
            const auto &left = inverse_powers_[knot-1], &right = inverse_powers_[knot];
            const RowVector dl = points_.row(knot)-points_.row(knot-1);
            const RowVector dr = points_.row(knot+1)-points_.row(knot);
            Block diagonal, previous;
            detail::staticFor<1,S>([&](auto row) SPLINE_INLINE_VISITOR {
                constexpr int r = decltype(row)::value;
                auto rhs = derivatives_.row(knot*M+r-1);
                rhs = -(Kernel::Q[S+r][S]*left[E-r])*dl - (Kernel::Q[r][S]*right[E-r])*dr;
                detail::staticFor<1,S>([&](auto column) SPLINE_INLINE_VISITOR {
                    constexpr int c = decltype(column)::value;
                    diagonal(r-1,c-1) = Kernel::Q[S+r][S+c]*left[E-r-c] + Kernel::Q[r][c]*right[E-r-c];
                    previous(r-1,c-1) = Kernel::Q[S+r][c]*left[E-r-c];
                    if (knot == 1) rhs -= previous(r-1,c-1)*derivatives_.row(c-1);
                    if (knot+1 == segments_)
                        rhs -= (Kernel::Q[r][S+c]*right[E-r-c])*derivatives_.row(segments_*M+c-1);
                });
            });
            if (!factor_.factor(knot-1,diagonal,previous)) return false;
        }
        return true;
    }
    /** @brief Apply one unit-interval Hermite row to scalar endpoint data.
     * @param data Endpoints relative to the start position, with derivatives scaled by T^r.
     * @param indices Nonconstant endpoint columns. @return Normalized power coefficient. */
    template<int Degree, std::size_t... Index>
    static EIGEN_ALWAYS_INLINE double hermiteValue(const std::array<double,K> &data,
                                                  std::index_sequence<Index...> indices)
    {
        (void)indices;
        return ((Kernel::H[Degree][Index+1]*data[Index+1]) + ...);
    }
    /** @brief Form coefficients in normalized time before converting to physical powers.
     * @note Each coordinate's endpoint values stay in registers across all polynomial degrees. */
    void writeCoefficients()
    {
        for (int i=0; i<segments_; ++i)
        {
            const auto &inv = inverse_powers_[i];
            for (int d=0; d<DIM; ++d)
            {
                std::array<double,K> data;
                data[0]=0.0; data[S]=points_(i+1,d)-points_(i,d);
                polynomial_.coefficients_(i*K,d)=points_(i,d);
                double power=1.0;
                detail::staticFor<1,S>([&](auto order) SPLINE_INLINE_VISITOR {
                    constexpr int r=decltype(order)::value;
                    power*=times_[i];
                    data[r]=power*derivatives_(i*M+r-1,d);
                    data[S+r]=power*derivatives_((i+1)*M+r-1,d);
                    polynomial_.coefficients_(i*K+r,d)=(1.0/detail::factorial(r))*derivatives_(i*M+r-1,d);
                });
                detail::staticFor<S,K>([&](auto degree) SPLINE_INLINE_VISITOR {
                    constexpr int k=decltype(degree)::value;
                    polynomial_.coefficients_(i*K+k,d)=inv[k]*hermiteValue<k>(data,std::make_index_sequence<K-1>{});
                });
            }
        }
    }

};
} // namespace SplineTrajectory
#undef SPLINE_INLINE_VISITOR

// Public spline aliases.

namespace SplineTrajectory
{
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

    template <int DIM> using CubicSplineND = MinDerivativeSplineND<DIM, 2>;
    template <int DIM> using QuinticSplineND = MinDerivativeSplineND<DIM, 3>;
    template <int DIM> using SepticSplineND = MinDerivativeSplineND<DIM, 4>;
    /** @brief Select a supported minimum-derivative spline at compile time. */
    template <int DIM, int S> struct MinDerivativeSplineSelector
    { using type = MinDerivativeSplineND<DIM, S>; };

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
