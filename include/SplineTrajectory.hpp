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

#ifndef SPLINE_TRAJECTORY_HPP
#define SPLINE_TRAJECTORY_HPP

#include <Eigen/Dense>
#include <vector>
#include <iostream>
#include <algorithm>

namespace SplineTrajectory
{
    template <typename T>
    using SplineVector = std::vector<T, Eigen::aligned_allocator<T>>;

    template <int DIM>
    struct BoundaryConditions
    {
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW
        using VectorType = Eigen::Matrix<double, DIM, 1>;
        VectorType start_velocity;
        VectorType start_acceleration;
        VectorType end_velocity;
        VectorType end_acceleration;

        BoundaryConditions()
            : start_velocity(VectorType::Zero()),
              start_acceleration(VectorType::Zero()),
              end_velocity(VectorType::Zero()),
              end_acceleration(VectorType::Zero())
        {
        }

        BoundaryConditions(const VectorType &start_velocity,
                           const VectorType &end_velocity)
            : start_velocity(start_velocity),
              start_acceleration(VectorType::Zero()),
              end_velocity(end_velocity),
              end_acceleration(VectorType::Zero())
        {
        }

        BoundaryConditions(const VectorType &start_velocity,
                           const VectorType &start_acceleration,
                           const VectorType &end_velocity,
                           const VectorType &end_acceleration)
            : start_velocity(start_velocity),
              start_acceleration(start_acceleration),
              end_velocity(end_velocity),
              end_acceleration(end_acceleration)
        {
        }
    };

    template <int DIM>
    struct SegmentedTimeSequence
    {
        struct SegmentInfo
        {
            int segment_idx;
            double segment_start;
            std::vector<double> times;
            std::vector<double> relative_times;
        };

        std::vector<SegmentInfo> segments;

        size_t getTotalSize() const
        {
            size_t total = 0;
            for (const auto &seg : segments)
            {
                total += seg.times.size();
            }
            return total;
        }
    };

    template <int DIM>
    class PPolyND
    {
    public:
        using VectorType = Eigen::Matrix<double, DIM, 1>;
        using MatrixType = Eigen::Matrix<double, Eigen::Dynamic, DIM>;
        using SegmentedTimeSeq = SegmentedTimeSequence<DIM>;
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    private:
        std::vector<double> breakpoints_;
        MatrixType coefficients_;
        int num_segments_;
        int order_;
        bool is_initialized_;

        mutable int cached_segment_idx_;
        mutable bool cache_valid_;

        mutable SplineVector<VectorType> cached_coeffs_;

        mutable std::vector<std::vector<double>> derivative_factors_cache_;

        mutable std::vector<bool> derivative_factors_computed_;

    public:
        PPolyND() : num_segments_(0), order_(0), is_initialized_(false),
                    cached_segment_idx_(0), cache_valid_(false) {}

        PPolyND(const std::vector<double> &breakpoints,
                const MatrixType &coefficients,
                int order)
            : is_initialized_(false), cached_segment_idx_(0), cache_valid_(false)
        {
            initializeInternal(breakpoints, coefficients, order);
        }

        void update(const std::vector<double> &breakpoints,
                    const MatrixType &coefficients,
                    int order)
        {
            initializeInternal(breakpoints, coefficients, order);
        }

        bool isInitialized() const { return is_initialized_; }
        int getDimension() const { return DIM; }
        int getOrder() const { return order_; }
        int getNumSegments() const { return num_segments_; }

        void clearCache() const
        {
            cache_valid_ = false;
            cached_segment_idx_ = 0;

            derivative_factors_computed_.assign(order_, false);
        }

        VectorType evaluate(double t, int derivative_order = 0) const
        {
            if (derivative_order >= order_)
                return VectorType::Zero();

            ensureDerivativeFactorsComputed(derivative_order);

            int segment_idx = findSegmentCached(t);
            double dt = t - breakpoints_[segment_idx];

            VectorType result = VectorType::Zero();
            double dt_power = 1.0;

            for (int k = derivative_order; k < order_; ++k)
            {
                const double coeff_factor = derivative_factors_cache_[derivative_order][k];
                const VectorType &coeff = cached_coeffs_[k];

                result += (coeff_factor * coeff) * dt_power;
                dt_power *= dt;
            }

            return result;
        }

        SplineVector<VectorType> evaluate(const std::vector<double> &t, int derivative_order = 0) const
        {
            if (t.empty())
                return {};

            SplineVector<VectorType> results;
            results.reserve(t.size());

            for (double time : t)
            {
                results.push_back(evaluate(time, derivative_order));
            }

            return results;
        }

        SplineVector<VectorType> evaluate(double start_t, double end_t, double dt, int derivative_order = 0) const
        {
            auto segmented_seq = generateSegmentedTimeSequence(start_t, end_t, dt);
            return evaluateSegmented(segmented_seq, derivative_order);
        }

        SegmentedTimeSeq generateSegmentedTimeSequence(double start_t, double end_t, double dt) const
        {
            SegmentedTimeSeq segmented_seq;

            if (start_t > end_t || dt <= 0.0)
                return segmented_seq;

            double current_t = start_t;
            int current_segment_idx = findSegment(current_t);

            while (current_t <= end_t)
            {
                double segment_start = breakpoints_[current_segment_idx];
                double segment_end = (current_segment_idx < num_segments_ - 1)
                                         ? breakpoints_[current_segment_idx + 1]
                                         : std::numeric_limits<double>::max();

                typename SegmentedTimeSeq::SegmentInfo segment_info;
                segment_info.segment_idx = current_segment_idx;
                segment_info.segment_start = segment_start;

                while (current_t <= end_t && current_t < segment_end)
                {
                    segment_info.times.push_back(current_t);
                    segment_info.relative_times.push_back(current_t - segment_start);
                    current_t += dt;
                }

                if (!segment_info.times.empty())
                {
                    segmented_seq.segments.push_back(std::move(segment_info));
                }

                if (current_segment_idx < num_segments_ - 1)
                {
                    current_segment_idx++;
                }
                else
                {
                    break;
                }
            }

            if (segmented_seq.segments.empty() || segmented_seq.segments.back().times.back() < end_t)
            {
                int end_seg_idx = findSegment(end_t);
                if (!segmented_seq.segments.empty() && end_seg_idx == segmented_seq.segments.back().segment_idx)
                {
                    auto &last_segment = segmented_seq.segments.back();
                    last_segment.times.push_back(end_t);
                    last_segment.relative_times.push_back(end_t - last_segment.segment_start);
                }
                else
                {
                    typename SegmentedTimeSeq::SegmentInfo end_segment;
                    end_segment.segment_idx = end_seg_idx;
                    end_segment.segment_start = breakpoints_[end_seg_idx];
                    end_segment.times.push_back(end_t);
                    end_segment.relative_times.push_back(end_t - end_segment.segment_start);
                    segmented_seq.segments.push_back(std::move(end_segment));
                }
            }

            return segmented_seq;
        }

        SplineVector<VectorType> evaluateSegmented(const SegmentedTimeSeq &segmented_seq, int derivative_order = 0) const
        {
            if (derivative_order >= order_)
            {
                SplineVector<VectorType> results(segmented_seq.getTotalSize(), VectorType::Zero());
                return results;
            }

            ensureDerivativeFactorsComputed(derivative_order);

            SplineVector<VectorType> results;
            results.reserve(segmented_seq.getTotalSize());

            for (const auto &segment_info : segmented_seq.segments)
            {

                SplineVector<VectorType> segment_coeffs(order_);
                for (int k = 0; k < order_; ++k)
                {
                    segment_coeffs[k] = coefficients_.row(segment_info.segment_idx * order_ + k);
                }

                for (double dt : segment_info.relative_times)
                {
                    VectorType result = VectorType::Zero();
                    double dt_power = 1.0;

                    for (int k = derivative_order; k < order_; ++k)
                    {
                        const double coeff_factor = derivative_factors_cache_[derivative_order][k];
                        result += (coeff_factor * segment_coeffs[k]) * dt_power;
                        dt_power *= dt;
                    }

                    results.push_back(result);
                }
            }

            return results;
        }

        std::vector<double> generateTimeSequence(double start_t, double end_t, double dt) const
        {
            std::vector<double> time_sequence;
            double current_t = start_t;

            while (current_t <= end_t)
            {
                time_sequence.push_back(current_t);
                current_t += dt;
            }

            if (time_sequence.empty() || time_sequence.back() < end_t)
            {
                time_sequence.push_back(end_t);
            }

            return time_sequence;
        }

        std::vector<double> generateTimeSequence(double dt) const
        {
            return generateTimeSequence(getStartTime(), getEndTime(), dt);
        }

        VectorType getPos(double t) const { return evaluate(t, 0); }
        VectorType getVel(double t) const { return evaluate(t, 1); }
        VectorType getAcc(double t) const { return evaluate(t, 2); }
        VectorType getJerk(double t) const { return evaluate(t, 3); }
        VectorType getSnap(double t) const { return evaluate(t, 4); }

        SplineVector<VectorType> getPos(const std::vector<double> &t) const { return evaluate(t, 0); }
        SplineVector<VectorType> getVel(const std::vector<double> &t) const { return evaluate(t, 1); }
        SplineVector<VectorType> getAcc(const std::vector<double> &t) const { return evaluate(t, 2); }
        SplineVector<VectorType> getJerk(const std::vector<double> &t) const { return evaluate(t, 3); }
        SplineVector<VectorType> getSnap(const std::vector<double> &t) const { return evaluate(t, 4); }

        SplineVector<VectorType> getPos(double start_t, double end_t, double dt) const { return evaluate(start_t, end_t, dt, 0); }
        SplineVector<VectorType> getVel(double start_t, double end_t, double dt) const { return evaluate(start_t, end_t, dt, 1); }
        SplineVector<VectorType> getAcc(double start_t, double end_t, double dt) const { return evaluate(start_t, end_t, dt, 2); }
        SplineVector<VectorType> getJerk(double start_t, double end_t, double dt) const { return evaluate(start_t, end_t, dt, 3); }
        SplineVector<VectorType> getSnap(double start_t, double end_t, double dt) const { return evaluate(start_t, end_t, dt, 4); }

        double getTrajectoryLength(double dt = 0.01) const
        {
            double start_time = getStartTime();
            double end_time = getEndTime();
            return getTrajectoryLength(start_time, end_time, dt);
        }

        double getTrajectoryLength(double start_t, double end_t, double dt = 0.01) const
        {
            std::vector<double> time_sequence = generateTimeSequence(start_t, end_t, dt);

            double total_length = 0.0;
            for (size_t i = 0; i < time_sequence.size() - 1; ++i)
            {
                double t_current = time_sequence[i];
                double t_next = time_sequence[i + 1];
                double dt_actual = t_next - t_current;

                VectorType velocity = getVel(t_current);
                total_length += velocity.norm() * dt_actual;
            }
            return total_length;
        }

        double getCumulativeLength(double t, double dt = 0.01) const
        {
            return getTrajectoryLength(getStartTime(), t, dt);
        }

        PPolyND derivative(int derivative_order = 1) const
        {
            if (derivative_order >= order_)
            {
                MatrixType zero_coeffs = MatrixType::Zero(num_segments_, DIM);
                return PPolyND(breakpoints_, zero_coeffs, 1);
            }

            int new_order = order_ - derivative_order;
            MatrixType new_coeffs(num_segments_ * new_order, DIM);

            for (int seg = 0; seg < num_segments_; ++seg)
            {
                for (int k = 0; k < new_order; ++k)
                {

                    int orig_k = k + derivative_order;

                    double coeff_factor = 1.0;
                    for (int j = 0; j < derivative_order; ++j)
                    {
                        coeff_factor *= (orig_k - j);
                    }

                    VectorType orig_coeff = coefficients_.row(seg * order_ + orig_k);
                    new_coeffs.row(seg * new_order + k) = coeff_factor * orig_coeff;
                }
            }

            return PPolyND(breakpoints_, new_coeffs, new_order);
        }

        double getStartTime() const
        {
            return breakpoints_.front();
        }

        double getEndTime() const
        {
            return breakpoints_.back();
        }

        double getDuration() const
        {
            return breakpoints_.back() - breakpoints_.front();
        }

        std::vector<double> getBreakpoints() { return breakpoints_; }
        MatrixType getCoefficients() { return coefficients_; }

        static PPolyND zero(const std::vector<double> &breakpoints, int order = 1)
        {
            int num_segments = breakpoints.size() - 1;
            MatrixType zero_coeffs = MatrixType::Zero(num_segments * order, DIM);
            return PPolyND(breakpoints, zero_coeffs, order);
        }

        static PPolyND constant(const std::vector<double> &breakpoints,
                                const VectorType &constant_value)
        {
            int num_segments = breakpoints.size() - 1;
            MatrixType coeffs = MatrixType::Zero(num_segments, DIM);

            for (int i = 0; i < num_segments; ++i)
            {
                coeffs.row(i) = constant_value.transpose();
            }

            return PPolyND(breakpoints, coeffs, 1);
        }

    private:
        inline void initializeInternal(const std::vector<double> &breakpoints,
                                       const MatrixType &coefficients,
                                       int order)
        {
            breakpoints_ = breakpoints;
            coefficients_ = coefficients;
            order_ = order;
            num_segments_ = breakpoints_.size() - 1;

            cached_coeffs_.resize(order_);
            for (int i = 0; i < order_; ++i)
            {
                cached_coeffs_[i] = VectorType::Zero();
            }

            derivative_factors_cache_.assign(order_, std::vector<double>());
            derivative_factors_computed_.assign(order_, false);

            is_initialized_ = true;
            clearCache();
        }

        inline void ensureDerivativeFactorsComputed(int derivative_order) const
        {
            if (!derivative_factors_computed_[derivative_order])
            {
                derivative_factors_cache_[derivative_order].resize(order_);
                for (int k = derivative_order; k < order_; ++k)
                {
                    double factor = 1.0;
                    for (int j = 0; j < derivative_order; ++j)
                    {
                        factor *= (k - j);
                    }
                    derivative_factors_cache_[derivative_order][k] = factor;
                }
                derivative_factors_computed_[derivative_order] = true;
            }
        }

        inline int findSegment(double t) const
        {
            if (t <= breakpoints_.front())
                return 0;
            if (t >= breakpoints_.back())
                return num_segments_ - 1;

            auto it = std::upper_bound(breakpoints_.begin(), breakpoints_.end(), t);
            return std::distance(breakpoints_.begin(), it) - 1;
        }

        inline int findSegmentCached(double t) const
        {
            if (t <= breakpoints_.front())
            {
                updateCache(0);
                return 0;
            }

            if (t >= breakpoints_.back())
            {
                updateCache(num_segments_ - 1);
                return num_segments_ - 1;
            }

            if (cache_valid_ &&
                t >= breakpoints_[cached_segment_idx_] &&
                t < breakpoints_[cached_segment_idx_ + 1])
            {
                return cached_segment_idx_;
            }

            if (cache_valid_ && cached_segment_idx_ + 1 < num_segments_)
            {
                int next_idx = cached_segment_idx_ + 1;
                if (t >= breakpoints_[next_idx] &&
                    (next_idx + 1 >= num_segments_ || t < breakpoints_[next_idx + 1]))
                {
                    updateCache(next_idx);
                    return next_idx;
                }
            }

            int segment_idx = findSegment(t);
            updateCache(segment_idx);
            return segment_idx;
        }

        inline void updateCache(int segment_idx) const
        {
            cached_segment_idx_ = segment_idx;

            for (int k = 0; k < order_; ++k)
            {
                cached_coeffs_[k] = coefficients_.row(segment_idx * order_ + k);
            }

            cache_valid_ = true;
        }
    };

    template <int DIM>
    class CubicSplineND
    {
    public:
        using VectorType = Eigen::Matrix<double, DIM, 1>;
        using RowVectorType = Eigen::Matrix<double, 1, DIM>;
        using MatrixType = Eigen::Matrix<double, Eigen::Dynamic, DIM>;
        static constexpr int kStorageOrder = (DIM == 1) ? Eigen::ColMajor : Eigen::RowMajor;
        using WorkMat = Eigen::Matrix<double, Eigen::Dynamic, DIM, kStorageOrder>;

    private:
        std::vector<double> time_segments_;
        SplineVector<VectorType> spatial_points_;
        BoundaryConditions<DIM> boundary_velocities_;
        int num_segments_;
        MatrixType coeffs_;
        bool is_initialized_;
        double start_time_;
        std::vector<double> cumulative_times_;
        PPolyND<DIM> trajectory_;

    public:
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW

        CubicSplineND() : num_segments_(0), is_initialized_(false), start_time_(0.0) {}

        CubicSplineND(const std::vector<double> &t_points,
                      const SplineVector<VectorType> &spatial_points,
                      const BoundaryConditions<DIM> &boundary_velocities = BoundaryConditions<DIM>())
            : spatial_points_(spatial_points), boundary_velocities_(boundary_velocities), is_initialized_(false)
        {
            convertTimePointsToSegments(t_points);
            updateSplineInternal();
        }

        CubicSplineND(const std::vector<double> &time_segments,
                      const SplineVector<VectorType> &spatial_points,
                      double start_time,
                      const BoundaryConditions<DIM> &boundary_velocities = BoundaryConditions<DIM>())
            : time_segments_(time_segments), spatial_points_(spatial_points), boundary_velocities_(boundary_velocities),
              is_initialized_(false), start_time_(start_time)
        {
            updateSplineInternal();
        }

        void update(const std::vector<double> &t_points,
                    const SplineVector<VectorType> &spatial_points,
                    const BoundaryConditions<DIM> &boundary_velocities = BoundaryConditions<DIM>())
        {
            spatial_points_ = spatial_points;
            boundary_velocities_ = boundary_velocities;
            convertTimePointsToSegments(t_points);
            updateSplineInternal();
        }

        void update(const std::vector<double> &time_segments,
                    const SplineVector<VectorType> &spatial_points,
                    double start_time,
                    const BoundaryConditions<DIM> &boundary_velocities = BoundaryConditions<DIM>())
        {
            time_segments_ = time_segments;
            spatial_points_ = spatial_points;
            boundary_velocities_ = boundary_velocities;
            start_time_ = start_time;
            updateSplineInternal();
        }

        bool isInitialized() const { return is_initialized_; }
        int getDimension() const { return DIM; }

        double getStartTime() const
        {
            return start_time_;
        }

        double getEndTime() const
        {
            return cumulative_times_.back();
        }

        double getDuration() const
        {
            return cumulative_times_.back() - start_time_;
        }

        size_t getNumPoints() const
        {
            return spatial_points_.size();
        }

        int getNumSegments() const
        {
            return num_segments_;
        }

        SplineVector<VectorType> getSpacePoints() const { return spatial_points_; }
        std::vector<double> getTimeSegments() const { return time_segments_; }
        std::vector<double> getCumulativeTimes() const { return cumulative_times_; }
        BoundaryConditions<DIM> getBoundaryConditions() const { return boundary_velocities_; }

        const PPolyND<DIM> &getTrajectory() const { return trajectory_; }
        PPolyND<DIM> getTrajectoryCopy() const { return trajectory_; }
        PPolyND<DIM> getPPoly() const { return trajectory_; }

        double getEnergy() const
        {
            double total_energy = 0.0;
            for (int i = 0; i < num_segments_; ++i)
            {
                const double T = time_segments_[i];
                if (T <= 0)
                    continue;

                const double T2 = T * T;
                const double T3 = T2 * T;

                // c2, c3
                // p(t) = c0 + c1t + c2t^2 + c3t^3
                RowVectorType c = coeffs_.row(i * 4 + 2);
                RowVectorType d = coeffs_.row(i * 4 + 3);

                total_energy += 12.0 * d.squaredNorm() * T3 +
                                12.0 * c.dot(d) * T2 +
                                4.0 * c.squaredNorm() * T;
            }
            return total_energy;
        }

    private:
        inline void updateSplineInternal()
        {
            num_segments_ = static_cast<int>(time_segments_.size());
            updateCumulativeTimes();
            coeffs_ = solveSpline();
            is_initialized_ = true;
            initializePPoly();
        }

        void convertTimePointsToSegments(const std::vector<double> &t_points)
        {
            start_time_ = t_points.front();
            time_segments_.clear();
            time_segments_.reserve(t_points.size() - 1);
            for (size_t i = 1; i < t_points.size(); ++i)
                time_segments_.push_back(t_points[i] - t_points[i - 1]);
        }

        void updateCumulativeTimes()
        {
            if (num_segments_ <= 0)
                return;
            cumulative_times_.resize(num_segments_ + 1);
            cumulative_times_[0] = start_time_;
            for (int i = 0; i < num_segments_; ++i)
            {
                cumulative_times_[i + 1] = cumulative_times_[i] + time_segments_[i];
            }
        }

        template <typename MatType>
        static void solveTridiagonalInPlace(const Eigen::VectorXd &lower,
                                            const Eigen::VectorXd &main,
                                            const Eigen::VectorXd &upper,
                                            MatType &M /* (n x DIM) */)
        {
            const int n = static_cast<int>(main.size());
            if (n <= 0)
                return;

            Eigen::VectorXd c_prime(n - 1);

            double inv = 1.0 / main(0);
            c_prime(0) = upper(0) * inv;
            M.row(0) *= inv;

            for (int i = 1; i < n - 1; ++i)
            {
                double denom = main(i) - lower(i - 1) * c_prime(i - 1);
                double inv_d = 1.0 / denom;
                c_prime(i) = upper(i) * inv_d;

                M.row(i).noalias() -= lower(i - 1) * M.row(i - 1);
                M.row(i) *= inv_d;
            }

            if (n >= 2)
            {
                double denom = main(n - 1) - lower(n - 2) * c_prime(n - 2);
                double inv_d = 1.0 / denom;
                M.row(n - 1).noalias() -= lower(n - 2) * M.row(n - 2);
                M.row(n - 1) *= inv_d;
            }

            for (int i = n - 2; i >= 0; --i)
            {
                M.row(i).noalias() -= c_prime(i) * M.row(i + 1);
            }
        }

        MatrixType solveSpline()
        {
            const int n_pts = static_cast<int>(spatial_points_.size());
            const int n = num_segments_;
            Eigen::Map<const Eigen::VectorXd> h(time_segments_.data(), n);

            WorkMat P(n_pts, DIM);
            for (int i = 0; i < n_pts; ++i)
                P.row(i) = spatial_points_[i].transpose();

            WorkMat p_diff_h = (P.bottomRows(n) - P.topRows(n)).array().colwise() / h.array();

            WorkMat D(n + 1, DIM);
            if (n >= 2)
            {
                D.block(1, 0, n - 1, DIM) = 6.0 * (p_diff_h.bottomRows(n - 1) - p_diff_h.topRows(n - 1));
            }
            D.row(0) = 6.0 * (p_diff_h.row(0) - boundary_velocities_.start_velocity.transpose());
            D.row(n) = 6.0 * (boundary_velocities_.end_velocity.transpose() - p_diff_h.row(n - 1));

            Eigen::VectorXd A_main(n + 1);
            Eigen::VectorXd A_lower(n);
            Eigen::VectorXd A_upper(n);

            if (n > 1)
            {
                A_lower.segment(0, n - 1) = h.segment(0, n - 1);
                A_upper.segment(1, n - 1) = h.segment(1, n - 1);
                A_main.segment(1, n - 1) =
                    2.0 * (h.segment(0, n - 1).array() + h.segment(1, n - 1).array()).matrix();
            }
            A_main(0) = 2.0 * h(0);
            A_upper(0) = h(0);
            A_main(n) = 2.0 * h(n - 1);
            A_lower(n - 1) = h(n - 1);

            solveTridiagonalInPlace(A_lower, A_main, A_upper, D);
            WorkMat &M = D;

            WorkMat a = P.topRows(n);
            WorkMat M_i = M.topRows(n);
            WorkMat M_i1 = M.bottomRows(n);

            const auto diag_h_over_6 = (h.array() / 6.0).matrix().asDiagonal();
            WorkMat b = p_diff_h - diag_h_over_6 * (2.0 * M_i + M_i1);
            WorkMat c = M_i * 0.5;
            const auto diag_inv_6h = (1.0 / (6.0 * h.array())).matrix().asDiagonal();
            WorkMat d = diag_inv_6h * (M_i1 - M_i);

            MatrixType coeffs(n * 4, DIM);

            for (int i = 0; i < n; ++i)
            {
                coeffs.row(i * 4 + 0) = a.row(i);
                coeffs.row(i * 4 + 1) = b.row(i);
                coeffs.row(i * 4 + 2) = c.row(i);
                coeffs.row(i * 4 + 3) = d.row(i);
            }

            return coeffs;
        }

        void initializePPoly()
        {

            trajectory_.update(cumulative_times_, coeffs_, 4);
        }
    };

    template <int DIM>
    class QuinticSplineND
    {
    public:
        using VectorType = Eigen::Matrix<double, DIM, 1>;
        using RowVectorType = Eigen::Matrix<double, 1, DIM>;
        using MatrixType = Eigen::Matrix<double, Eigen::Dynamic, DIM>;

    private:
        std::vector<double> time_segments_;
        std::vector<double> cumulative_times_;
        double start_time_{0.0};

        SplineVector<VectorType> spatial_points_;

        BoundaryConditions<DIM> boundary_;

        int num_segments_{0};
        bool is_initialized_{false};

        MatrixType coeffs_;
        PPolyND<DIM> trajectory_;

    private:
        void updateSplineInternal()
        {
            num_segments_ = static_cast<int>(time_segments_.size());
            updateCumulativeTimes();
            coeffs_ = solveQuintic();
            is_initialized_ = true;
            initializePPoly();
        }

    public:
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW

        QuinticSplineND() = default;

        QuinticSplineND(const std::vector<double> &t_points,
                        const SplineVector<VectorType> &spatial_points,
                        const BoundaryConditions<DIM> &boundary = BoundaryConditions<DIM>())
            : spatial_points_(spatial_points),
              boundary_(boundary)
        {
            convertTimePointsToSegments(t_points);
            updateSplineInternal();
        }

        QuinticSplineND(const std::vector<double> &time_segments,
                        const SplineVector<VectorType> &spatial_points,
                        double start_time,
                        const BoundaryConditions<DIM> &boundary = BoundaryConditions<DIM>())
            : time_segments_(time_segments),
              start_time_(start_time),
              spatial_points_(spatial_points),
              boundary_(boundary)
        {
            updateSplineInternal();
        }

        void update(const std::vector<double> &t_points,
                    const SplineVector<VectorType> &spatial_points,
                    const BoundaryConditions<DIM> &boundary = BoundaryConditions<DIM>())
        {
            spatial_points_ = spatial_points;
            boundary_ = boundary;
            convertTimePointsToSegments(t_points);
            updateSplineInternal();
        }

        void update(const std::vector<double> &time_segments,
                                const SplineVector<VectorType> &spatial_points,
                                double start_time,
                                const BoundaryConditions<DIM> &boundary = BoundaryConditions<DIM>())
        {
            time_segments_ = time_segments;
            spatial_points_ = spatial_points;
            boundary_ = boundary;
            start_time_ = start_time;
            updateSplineInternal();
        }

        bool isInitialized() const { return is_initialized_; }
        int getDimension() const { return DIM; }

        double getStartTime() const
        {
            return start_time_;
        }

        double getEndTime() const
        {
            return cumulative_times_.back();
        }

        double getDuration() const
        {
            return cumulative_times_.back() - start_time_;
        }

        size_t getNumPoints() const
        {
            return spatial_points_.size();
        }

        int getNumSegments() const
        {
            return num_segments_;
        }

        SplineVector<VectorType> getSpacePoints() const { return spatial_points_; }
        std::vector<double> getTimeSegments() const { return time_segments_; }
        std::vector<double> getCumulativeTimes() const { return cumulative_times_; }
        BoundaryConditions<DIM> getBoundaryConditions() const { return boundary_; }

        const PPolyND<DIM> &getTrajectory() const { return trajectory_; }
        PPolyND<DIM> getTrajectoryCopy() const { return trajectory_; }
        PPolyND<DIM> getPPoly() const { return trajectory_; }

        double getEnergy() const
        {
            if (!is_initialized_)
            {
                return 0.0;
            }

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

                // c3, c4, c5
                // p(t) = c0 + c1*t + c2*t^2 + c3*t^3 + c4*t^4 + c5*t^5
                RowVectorType c3 = coeffs_.row(i * 6 + 3);
                RowVectorType c4 = coeffs_.row(i * 6 + 4);
                RowVectorType c5 = coeffs_.row(i * 6 + 5);

                total_energy += 36.0 * c3.squaredNorm() * T +
                                144.0 * c4.dot(c3) * T2 +
                                192.0 * c4.squaredNorm() * T3 +
                                240.0 * c5.dot(c3) * T3 +
                                720.0 * c5.dot(c4) * T4 +
                                720.0 * c5.squaredNorm() * T5;
            }
            return total_energy;
        }


    private:
        void convertTimePointsToSegments(const std::vector<double> &t_points)
        {
            start_time_ = t_points.front();
            time_segments_.clear();
            time_segments_.reserve(t_points.size() - 1);
            for (size_t i = 1; i < t_points.size(); ++i)
                time_segments_.push_back(t_points[i] - t_points[i - 1]);
        }

        void updateCumulativeTimes()
        {
            if (num_segments_ <= 0)
                return;
            cumulative_times_.resize(num_segments_ + 1);
            cumulative_times_[0] = start_time_;
            for (int i = 0; i < num_segments_; ++i)
                cumulative_times_[i + 1] = cumulative_times_[i] + time_segments_[i];
        }

        template <typename DerivedB>
        static inline Eigen::Matrix<double, 2, DerivedB::ColsAtCompileTime>
        solve2x2(const Eigen::Matrix2d &A, const Eigen::MatrixBase<DerivedB> &B)
        {
            static_assert(DerivedB::RowsAtCompileTime == 2, "B must have 2 rows");
            const double a = A(0, 0), b = A(0, 1), c = A(1, 0), d = A(1, 1);
            const double det = a * d - b * c;

            const double inv_det = 1.0 / det;

            Eigen::Matrix<double, 2, DerivedB::ColsAtCompileTime> result;
            result.row(0) = (d * B.row(0) - b * B.row(1)) * inv_det;
            result.row(1) = (-c * B.row(0) + a * B.row(1)) * inv_det;

            return result;
        }

        void solveInternalDerivatives(const MatrixType &P,
                                      const Eigen::VectorXd &h,
                                      MatrixType &p_out,
                                      MatrixType &q_out)
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

            std::vector<Eigen::Matrix2d, Eigen::aligned_allocator<Eigen::Matrix2d>> L_blocks;
            L_blocks.reserve(std::max(0, num_blocks - 1));
            std::vector<Eigen::Matrix2d, Eigen::aligned_allocator<Eigen::Matrix2d>> D_blocks;
            D_blocks.reserve(num_blocks);
            std::vector<Eigen::Matrix2d, Eigen::aligned_allocator<Eigen::Matrix2d>> U_blocks;
            U_blocks.reserve(std::max(0, num_blocks - 1));
            std::vector<Eigen::Matrix<double, 2, DIM>, Eigen::aligned_allocator<Eigen::Matrix<double, 2, DIM>>> rhs_blocks;
            rhs_blocks.reserve(num_blocks);

            for (int i = 0; i < num_blocks; ++i)
            {
                const int k = i + 2;
                const double hL = h(k - 2);
                const double hR = h(k - 1);

                const double hL_inv = 1.0 / hL;
                const double hL2_inv = hL_inv * hL_inv;
                const double hL3_inv = hL2_inv * hL_inv;
                const double hL4_inv = hL3_inv * hL_inv;

                const double hR_inv = 1.0 / hR;
                const double hR2_inv = hR_inv * hR_inv;
                const double hR3_inv = hR2_inv * hR_inv;
                const double hR4_inv = hR3_inv * hR_inv;

                Eigen::Matrix<double, 1, DIM> r3 = 60.0 * ((P.row(k) - P.row(k - 1)) * hR3_inv -
                                                           (P.row(k - 1) - P.row(k - 2)) * hL3_inv);
                Eigen::Matrix<double, 1, DIM> r4 = 360.0 * ((P.row(k - 1) - P.row(k)) * hR4_inv +
                                                            (P.row(k - 2) - P.row(k - 1)) * hL4_inv);
                Eigen::Matrix<double, 2, DIM> r;
                r.row(0) = r3;
                r.row(1) = r4;

                Eigen::Matrix2d L;
                L << -24.0 * hL2_inv, -3.0 * hL_inv,
                    -168.0 * hL3_inv, -24.0 * hL2_inv;

                Eigen::Matrix2d D;
                D << -36.0 * hL2_inv + 36.0 * hR2_inv, 9.0 * (hL_inv + hR_inv),
                    -192.0 * (hL3_inv + hR3_inv), 36.0 * (hL2_inv - hR2_inv);

                Eigen::Matrix2d U;
                U << 24.0 * hR2_inv, -3.0 * hR_inv,
                    -168.0 * hR3_inv, 24.0 * hR2_inv;

                if (k == 2)
                {
                    r.noalias() -= L * B_left;
                }
                else
                {
                    L_blocks.push_back(L);
                }

                if (k == n - 1)
                {
                    r.noalias() -= U * B_right;
                }
                else
                {
                    U_blocks.push_back(U);
                }

                D_blocks.push_back(D);
                rhs_blocks.push_back(r);
            }

            std::vector<Eigen::Matrix2d, Eigen::aligned_allocator<Eigen::Matrix2d>> D_mod;
            D_mod.reserve(num_blocks);
            std::vector<Eigen::Matrix<double, 2, DIM>, Eigen::aligned_allocator<Eigen::Matrix<double, 2, DIM>>> rhs_mod;
            rhs_mod.reserve(num_blocks);

            D_mod.push_back(D_blocks[0]);
            rhs_mod.push_back(rhs_blocks[0]);

            for (int i = 1; i < num_blocks; ++i)
            {
                const Eigen::Matrix2d &D_prev = D_mod[i - 1];
                const Eigen::Matrix2d &L = L_blocks[i - 1];
                const Eigen::Matrix2d &U = U_blocks[i - 1];

                const Eigen::Matrix2d X = solve2x2(D_prev, U);
                const Eigen::Matrix<double, 2, DIM> Y = solve2x2(D_prev, rhs_mod[i - 1]);

                const Eigen::Matrix2d D_new = D_blocks[i] - L * X;
                const Eigen::Matrix<double, 2, DIM> rhs_new = rhs_blocks[i] - L * Y;

                D_mod.push_back(D_new);
                rhs_mod.push_back(rhs_new);
            }

            std::vector<Eigen::Matrix<double, 2, DIM>, Eigen::aligned_allocator<Eigen::Matrix<double, 2, DIM>>> solution(num_blocks);
            solution[num_blocks - 1] = solve2x2(D_mod.back(), rhs_mod.back());
            for (int i = num_blocks - 2; i >= 0; --i)
            {
                const Eigen::Matrix<double, 2, DIM> rhs_temp = rhs_mod[i] - U_blocks[i] * solution[i + 1];
                solution[i] = solve2x2(D_mod[i], rhs_temp);
            }

            for (int i = 0; i < num_blocks; ++i)
            {
                const int row = i + 1;
                p_out.row(row) = solution[i].row(0);
                q_out.row(row) = solution[i].row(1);
            }
        }

        MatrixType solveQuintic()
        {
            const int n_pts = static_cast<int>(spatial_points_.size());
            const int n = num_segments_;

            Eigen::Map<const Eigen::VectorXd> h(time_segments_.data(), n);

            MatrixType P(n_pts, DIM);
            for (int i = 0; i < n_pts; ++i)
            {
                P.row(i) = spatial_points_[i].transpose();
            }

            MatrixType p_nodes, q_nodes;
            solveInternalDerivatives(P, h, p_nodes, q_nodes);

            MatrixType coeffs(n * 6, DIM);

            for (int i = 0; i < n; ++i)
            {
                const double hi = h(i);
                const double h_inv = 1.0 / hi;
                const double h2_inv = h_inv * h_inv;
                const double h3_inv = h2_inv * h_inv;

                const RowVectorType c0 = P.row(i);
                const RowVectorType c1 = p_nodes.row(i);
                const RowVectorType c2 = q_nodes.row(i) * 0.5;

                const RowVectorType rhs1 = P.row(i + 1) - c0 - c1 * hi - c2 * (hi * hi);
                const RowVectorType rhs2 = p_nodes.row(i + 1) - c1 - (2.0 * c2) * hi;
                const RowVectorType rhs3 = q_nodes.row(i + 1) - (2.0 * c2);

                const RowVectorType c3 = (10.0 * h3_inv) * rhs1 - (4.0 * h2_inv) * rhs2 + (0.5 * h_inv) * rhs3;
                const RowVectorType c4 = (-15.0 * h3_inv * h_inv) * rhs1 + (7.0 * h3_inv) * rhs2 - (h2_inv)*rhs3;
                const RowVectorType c5 = (6.0 * h3_inv * h2_inv) * rhs1 - (3.0 * h3_inv * h_inv) * rhs2 + (0.5 * h3_inv) * rhs3;

                coeffs.row(i * 6 + 0) = c0;
                coeffs.row(i * 6 + 1) = c1;
                coeffs.row(i * 6 + 2) = c2;
                coeffs.row(i * 6 + 3) = c3;
                coeffs.row(i * 6 + 4) = c4;
                coeffs.row(i * 6 + 5) = c5;
            }

            return coeffs;
        }

        void initializePPoly()
        {
            trajectory_.update(cumulative_times_, coeffs_, 6);
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
    using PointSequence4d = SplineVector<SplinePoint4d>;
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

} // namespace SplineTrajectory

#endif // SPLINE_TRAJECTORY_HPP