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
        VectorType start_jerk;
        VectorType end_jerk;

        BoundaryConditions()
            : start_velocity(VectorType::Zero()),
              start_acceleration(VectorType::Zero()),
              end_velocity(VectorType::Zero()),
              end_acceleration(VectorType::Zero()),
              start_jerk(VectorType::Zero()),
              end_jerk(VectorType::Zero())
        {
        }

        BoundaryConditions(const VectorType &start_velocity,
                           const VectorType &end_velocity)
            : start_velocity(start_velocity),
              start_acceleration(VectorType::Zero()),
              end_velocity(end_velocity),
              end_acceleration(VectorType::Zero()),
              start_jerk(VectorType::Zero()),
              end_jerk(VectorType::Zero())
        {
        }

        BoundaryConditions(const VectorType &start_velocity,
                           const VectorType &start_acceleration,
                           const VectorType &end_velocity,
                           const VectorType &end_acceleration)
            : start_velocity(start_velocity),
              start_acceleration(start_acceleration),
              end_velocity(end_velocity),
              end_acceleration(end_acceleration),
              start_jerk(VectorType::Zero()),
              end_jerk(VectorType::Zero())
        {
        }

        BoundaryConditions(const VectorType &start_velocity,
                           const VectorType &start_acceleration,
                           const VectorType &end_velocity,
                           const VectorType &end_acceleration,
                           const VectorType &start_jerk,
                           const VectorType &end_jerk)
            : start_velocity(start_velocity),
              start_acceleration(start_acceleration),
              end_velocity(end_velocity),
              end_acceleration(end_acceleration),
              start_jerk(start_jerk),
              end_jerk(end_jerk)
        {
        }
    };

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
        static constexpr int kMatrixOptions = (DIM == 1) ? Eigen::ColMajor : Eigen::RowMajor;
        using MatrixType = Eigen::Matrix<double, Eigen::Dynamic, DIM, kMatrixOptions>;
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

        SegmentedTimeSequence generateSegmentedTimeSequence(double start_t, double end_t, double dt) const
        {
            SegmentedTimeSequence segmented_seq;

            if (start_t <= end_t && dt > 0.0)
            {
                double current_t = start_t;
                int current_segment_idx = findSegment(current_t);

                while (current_t <= end_t)
                {
                    double segment_start = breakpoints_[current_segment_idx];
                    double segment_end = (current_segment_idx < num_segments_ - 1)
                                             ? breakpoints_[current_segment_idx + 1]
                                             : std::numeric_limits<double>::max();

                    typename SegmentedTimeSequence::SegmentInfo segment_info;
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
                        typename SegmentedTimeSequence::SegmentInfo end_segment;
                        end_segment.segment_idx = end_seg_idx;
                        end_segment.segment_start = breakpoints_[end_seg_idx];
                        end_segment.times.push_back(end_t);
                        end_segment.relative_times.push_back(end_t - end_segment.segment_start);
                        segmented_seq.segments.push_back(std::move(end_segment));
                    }
                }
            }
            return segmented_seq;
        }

        SplineVector<VectorType> evaluateSegmented(const SegmentedTimeSequence &segmented_seq, int derivative_order = 0) const
        {
            SplineVector<VectorType> results;
            const size_t total_size = segmented_seq.getTotalSize();
            results.reserve(total_size);

            if (derivative_order >= order_)
            {
                results.resize(total_size, VectorType::Zero());
            }
            else
            {
                ensureDerivativeFactorsComputed(derivative_order);

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

        // Single time point evaluation
        VectorType getPos(double t) const { return evaluate(t, 0); }
        VectorType getVel(double t) const { return evaluate(t, 1); }
        VectorType getAcc(double t) const { return evaluate(t, 2); }
        VectorType getJerk(double t) const { return evaluate(t, 3); }
        VectorType getSnap(double t) const { return evaluate(t, 4); }

        // Multiple time points evaluation
        SplineVector<VectorType> getPos(const std::vector<double> &t) const { return evaluate(t, 0); }
        SplineVector<VectorType> getVel(const std::vector<double> &t) const { return evaluate(t, 1); }
        SplineVector<VectorType> getAcc(const std::vector<double> &t) const { return evaluate(t, 2); }
        SplineVector<VectorType> getJerk(const std::vector<double> &t) const { return evaluate(t, 3); }
        SplineVector<VectorType> getSnap(const std::vector<double> &t) const { return evaluate(t, 4); }

        // Time range evaluation(internal segmentation)
        SplineVector<VectorType> getPos(double start_t, double end_t, double dt) const { return evaluate(start_t, end_t, dt, 0); }
        SplineVector<VectorType> getVel(double start_t, double end_t, double dt) const { return evaluate(start_t, end_t, dt, 1); }
        SplineVector<VectorType> getAcc(double start_t, double end_t, double dt) const { return evaluate(start_t, end_t, dt, 2); }
        SplineVector<VectorType> getJerk(double start_t, double end_t, double dt) const { return evaluate(start_t, end_t, dt, 3); }
        SplineVector<VectorType> getSnap(double start_t, double end_t, double dt) const { return evaluate(start_t, end_t, dt, 4); }

        SplineVector<VectorType> getPos(const SegmentedTimeSequence &segmented_seq) const { return evaluateSegmented(segmented_seq, 0); }
        SplineVector<VectorType> getVel(const SegmentedTimeSequence &segmented_seq) const { return evaluateSegmented(segmented_seq, 1); }
        SplineVector<VectorType> getAcc(const SegmentedTimeSequence &segmented_seq) const { return evaluateSegmented(segmented_seq, 2); }
        SplineVector<VectorType> getJerk(const SegmentedTimeSequence &segmented_seq) const { return evaluateSegmented(segmented_seq, 3); }
        SplineVector<VectorType> getSnap(const SegmentedTimeSequence &segmented_seq) const { return evaluateSegmented(segmented_seq, 4); }

        double getTrajectoryLength(double dt = 0.01) const
        {
            return getTrajectoryLength(getStartTime(), getEndTime(), dt);
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

        const std::vector<double> &getBreakpoints() const { return breakpoints_; }
        const MatrixType &getCoefficients() const { return coefficients_; }

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
        static constexpr int kMatrixOptions = (DIM == 1) ? Eigen::ColMajor : Eigen::RowMajor;
        using MatrixType = Eigen::Matrix<double, Eigen::Dynamic, DIM, kMatrixOptions>;

        struct Gradients
        {
            MatrixType points;
            Eigen::VectorXd times;
        };

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

        MatrixType internal_derivatives_;
        Eigen::VectorXd cached_c_prime_;
        Eigen::VectorXd cached_inv_denoms_;

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

        const SplineVector<VectorType> &getSpacePoints() const { return spatial_points_; }
        const std::vector<double> &getTimeSegments() const { return time_segments_; }
        const std::vector<double> &getCumulativeTimes() const { return cumulative_times_; }
        const BoundaryConditions<DIM> &getBoundaryConditions() const { return boundary_velocities_; }

        const PPolyND<DIM> &getTrajectory() const { return trajectory_; }
        PPolyND<DIM> getTrajectoryCopy() const { return trajectory_; }
        const PPolyND<DIM> &getPPoly() const { return trajectory_; }
        PPolyND<DIM> getPPolyCopy() const { return trajectory_; }

        double getEnergy() const
        {
            if (!is_initialized_)
                return 0.0;

            double total_energy = 0.0;
            for (int i = 0; i < num_segments_; ++i)
            {
                const double T = time_powers_[i].h;
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

        /**
         * @brief Computes the partial gradient of the energy (acceleration cost) w.r.t polynomial coefficients.
         * @param gdC [Output] Matrix of size (num_segments * 4) x DIM.
         */
        void getEnergyPartialGradByCoeffs(MatrixType &gdC) const
        {
            gdC.resize(num_segments_ * 4, DIM);
            gdC.setZero();

            for (int i = 0; i < num_segments_; ++i)
            {
                const double T = time_powers_[i].h;
                double T2 = T * T;
                double T3 = T2 * T;

                // Coefficients c2, c3
                // Energy = 4*c2^2*T + 12*c2*c3*T^2 + 12*c3^2*T^3
                const RowVectorType c2 = coeffs_.row(i * 4 + 2);
                const RowVectorType c3 = coeffs_.row(i * 4 + 3);

                // dE/dc2 = 8*c2*T + 12*c3*T^2
                gdC.row(i * 4 + 2) = 8.0 * c2 * T + 12.0 * c3 * T2;

                // dE/dc3 = 12*c2*T^2 + 24*c3*T^3
                gdC.row(i * 4 + 3) = 12.0 * c2 * T2 + 24.0 * c3 * T3;
            }
        }

        /**
         * @brief Computes the partial gradient of the energy (acceleration cost) with respect to the segment duration T.
         * By applying the Fundamental Theorem of Calculus, the derivative of the energy integral with respect to its
         * upper limit T is simply the squared norm of the acceleration vector at the end of the segment.
         * This avoids complex polynomial expansion.
         * @param gdT [Output] Gradient vector of size num_segments_.
         */
        void getEnergyPartialGradByTimes(Eigen::VectorXd &gdT) const
        {
            gdT.resize(num_segments_);

            for (int i = 0; i < num_segments_; ++i)
            {
                const double T = time_powers_[i].h;

                const RowVectorType c2 = coeffs_.row(i * 4 + 2);
                const RowVectorType c3 = coeffs_.row(i * 4 + 3);

                RowVectorType acc_end = 2.0 * c2 + (6.0 * T) * c3;

                gdT(i) = acc_end.squaredNorm();
            }
        }

        Eigen::VectorXd getEnergyGradTimes() const
        {
            Eigen::VectorXd grad(num_segments_);

            for (int i = 0; i < num_segments_; ++i)
            {
                const RowVectorType c1 = coeffs_.row(i * 4 + 1);
                const RowVectorType c2 = coeffs_.row(i * 4 + 2);
                const RowVectorType c3 = coeffs_.row(i * 4 + 3);

                double term_acc = 4.0 * c2.squaredNorm();
                double term_jv = 12.0 * c1.dot(c3);

                grad(i) = -term_acc + term_jv;
            }
            return grad;
        }

        MatrixType getEnergyGradInnerP() const
        {
            MatrixType grad(std::max(0, num_segments_ - 1), DIM);
            for (int i = 1; i < num_segments_; ++i)
            {
                const RowVectorType c3_L = coeffs_.row((i - 1) * 4 + 3);
                const RowVectorType c3_R = coeffs_.row(i * 4 + 3);
                grad.row(i - 1) = 12.0 * (c3_R - c3_L);
            }
            return grad;
        }

        /**
         * @brief Propagates gradients from polynomial coefficients to waypoints and time segments.
         *
         * @param includeEndpoints If false (default), only returns gradients for inner waypoints (rows 1 to N-1),
         * excluding the start and end points, which is consistent with MINCO and other trajectory optimization libraries.
         * If true, returns gradients for ALL waypoints including start and end points.
         */
        void propagateGrad(const MatrixType &partialGradByCoeffs,
                           const Eigen::VectorXd &partialGradByTimes,
                           MatrixType &gradByPoints,
                           Eigen::VectorXd &gradByTimes,
                           bool includeEndpoints = false)
        {
            const int n = num_segments_;
            const int dim = DIM;

            const MatrixType &M = internal_derivatives_; // Size (N+1) x DIM

            gradByPoints = MatrixType::Zero(n + 1, dim);
            gradByTimes = partialGradByTimes;

            MatrixType gradM = MatrixType::Zero(n + 1, dim);

            for (int i = 0; i < n; ++i)
            {
                const auto &tp = time_powers_[i];
                double h_i = tp.h;
                double h_inv = tp.h_inv;
                double h2_inv = tp.h2_inv;

                Eigen::Matrix<double, 1, dim> g_c0 = partialGradByCoeffs.row(i * 4 + 0);
                Eigen::Matrix<double, 1, dim> g_c1 = partialGradByCoeffs.row(i * 4 + 1);
                Eigen::Matrix<double, 1, dim> g_c2 = partialGradByCoeffs.row(i * 4 + 2);
                Eigen::Matrix<double, 1, dim> g_c3 = partialGradByCoeffs.row(i * 4 + 3);

                gradByPoints.row(i) += g_c0;
                gradByPoints.row(i) -= g_c1 * h_inv;
                gradByPoints.row(i + 1) += g_c1 * h_inv;

                gradM.row(i) -= g_c1 * (h_i / 3.0);
                gradM.row(i + 1) -= g_c1 * (h_i / 6.0);
                gradM.row(i) += g_c2 * 0.5;
                gradM.row(i) -= g_c3 * (h_inv / 6.0);
                gradM.row(i + 1) += g_c3 * (h_inv / 6.0);

                VectorType dP = spatial_points_[i + 1] - spatial_points_[i];
                VectorType term_dC1_dh = -dP * h2_inv - (2.0 * M.row(i) + M.row(i + 1)).transpose() / 6.0;
                VectorType term_dC3_dh = -(M.row(i + 1) - M.row(i)).transpose() * (h2_inv / 6.0);

                gradByTimes(i) += g_c1.dot(term_dC1_dh) + g_c3.dot(term_dC3_dh);
            }

            MatrixType lambda(n + 1, dim);
            lambda = gradM;
            solveWithCachedLU(lambda);

            for (int k = 0; k < n; ++k)
            {
                const auto &tp = time_powers_[k];
                double h2_inv = tp.h2_inv;
                VectorType dP = spatial_points_[k + 1] - spatial_points_[k];

                VectorType common_term = lambda.row(k) - lambda.row(k + 1);

                VectorType grad_R_P = 6.0 * tp.h_inv * common_term;
                gradByPoints.row(k + 1) += grad_R_P;
                gradByPoints.row(k) -= grad_R_P;

                VectorType grad_R_h = -6.0 * dP * h2_inv;
                gradByTimes(k) += common_term.dot(grad_R_h);

                VectorType M_k = M.row(k);
                VectorType M_k1 = M.row(k + 1);

                VectorType term_k = 2.0 * M_k + M_k1;
                VectorType term_k1 = M_k + 2.0 * M_k1;

                gradByTimes(k) -= lambda.row(k).dot(term_k);
                gradByTimes(k) -= lambda.row(k + 1).dot(term_k1);
            }

            if (!includeEndpoints && n > 1)
            {
                gradByPoints = gradByPoints.middleRows(1, n - 1).eval();
            }
        }

        Gradients propagateGrad(const MatrixType &partialGradByCoeffs,
                                const Eigen::VectorXd &partialGradByTimes,
                                bool includeEndpoints = false)
        {
            Gradients result;
            propagateGrad(partialGradByCoeffs, partialGradByTimes, result.points, result.times, includeEndpoints);
            return result;
        }

    private:
        inline void updateSplineInternal()
        {
            num_segments_ = static_cast<int>(time_segments_.size());
            updateCumulativeTimes();
            precomputeTimePowers();
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
        // -----------------------

        MatrixType solveSpline()
        {
            const int n = num_segments_;

            MatrixType p_diff_h(n, DIM);
            for (int i = 0; i < n; ++i)
            {
                p_diff_h.row(i) = (spatial_points_[i + 1] - spatial_points_[i]).transpose() * time_powers_[i].h_inv;
            }

            internal_derivatives_.resize(n + 1, DIM);
            MatrixType &M = internal_derivatives_;

            if (n >= 2)
            {
                M.block(1, 0, n - 1, DIM) = 6.0 * (p_diff_h.bottomRows(n - 1) - p_diff_h.topRows(n - 1));
            }
            M.row(0) = 6.0 * (p_diff_h.row(0) - boundary_velocities_.start_velocity.transpose());
            M.row(n) = 6.0 * (boundary_velocities_.end_velocity.transpose() - p_diff_h.row(n - 1));

            computeLUAndSolve(M);

            MatrixType coeffs(n * 4, DIM);

            for (int i = 0; i < n; ++i)
            {
                const auto &tp = time_powers_[i];
                double h_i = tp.h;
                double h_inv = tp.h_inv;

                coeffs.row(i * 4 + 0) = spatial_points_[i].transpose();

                coeffs.row(i * 4 + 1) = p_diff_h.row(i) - (h_i / 6.0) * (2.0 * M.row(i) + M.row(i + 1));

                coeffs.row(i * 4 + 2) = M.row(i) * 0.5;

                coeffs.row(i * 4 + 3) = (M.row(i + 1) - M.row(i)) * (h_inv / 6.0);
            }

            return coeffs;
        }

        template <typename MatType>
        void computeLUAndSolve(MatType &M /* (n+1 x DIM) */)
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
        static constexpr int kMatrixOptions = (DIM == 1) ? Eigen::ColMajor : Eigen::RowMajor;
        using MatrixType = Eigen::Matrix<double, Eigen::Dynamic, DIM, kMatrixOptions>;

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

        std::vector<Eigen::Matrix2d, Eigen::aligned_allocator<Eigen::Matrix2d>> D_inv_cache_;
        std::vector<Eigen::Matrix2d, Eigen::aligned_allocator<Eigen::Matrix2d>> U_blocks_cache_;
        std::vector<Eigen::Matrix2d, Eigen::aligned_allocator<Eigen::Matrix2d>> L_blocks_cache_;
        MatrixType internal_vel_;
        MatrixType internal_acc_;

        struct TimePowers
        {
            double h;
            double h_inv;  // h^-1
            double h2_inv; // h^-2
            double h3_inv; // h^-3
            double h4_inv; // h^-4
            double h5_inv; // h^-5
            double h6_inv; // h^-6
        };
        std::vector<TimePowers> time_powers_;
        std::vector<Eigen::Matrix<double, 2, DIM>, Eigen::aligned_allocator<Eigen::Matrix<double, 2, DIM>>> ws_d_rhs_mod_;
        std::vector<Eigen::Matrix<double, 2, DIM>, Eigen::aligned_allocator<Eigen::Matrix<double, 2, DIM>>> ws_current_lambda_;
        std::vector<Eigen::Matrix<double, 2, DIM>, Eigen::aligned_allocator<Eigen::Matrix<double, 2, DIM>>> ws_rhs_mod_;
        std::vector<Eigen::Matrix<double, 2, DIM>, Eigen::aligned_allocator<Eigen::Matrix<double, 2, DIM>>> ws_solution_;

    private:
        void updateSplineInternal()
        {
            num_segments_ = static_cast<int>(time_segments_.size());
            updateCumulativeTimes();
            precomputeTimePowers();
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

        const SplineVector<VectorType> &getSpacePoints() const { return spatial_points_; }
        const std::vector<double> &getTimeSegments() const { return time_segments_; }
        const std::vector<double> &getCumulativeTimes() const { return cumulative_times_; }
        const BoundaryConditions<DIM> &getBoundaryConditions() const { return boundary_; }

        const PPolyND<DIM> &getTrajectory() const { return trajectory_; }
        PPolyND<DIM> getTrajectoryCopy() const { return trajectory_; }
        const PPolyND<DIM> &getPPoly() const { return trajectory_; }
        PPolyND<DIM> getPPolyCopy() const { return trajectory_; }

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
        /**
         * @brief Computes the partial gradient of the energy (jerk cost) w.r.t polynomial coefficients.
         * @param gdC [Output] Matrix of size (num_segments * 6) x DIM.
         */
        void getEnergyPartialGradByCoeffs(MatrixType &gdC) const
        {
            gdC.resize(num_segments_ * 6, DIM);
            gdC.setZero();

            for (int i = 0; i < num_segments_; ++i)
            {
                double T = time_segments_[i];
                double T2 = T * T;
                double T3 = T2 * T;
                double T4 = T3 * T;
                double T5 = T4 * T;

                const RowVectorType c3 = coeffs_.row(i * 6 + 3);
                const RowVectorType c4 = coeffs_.row(i * 6 + 4);
                const RowVectorType c5 = coeffs_.row(i * 6 + 5);

                // dE/dc3
                gdC.row(i * 6 + 3) = 72.0 * c3 * T +
                                     144.0 * c4 * T2 +
                                     240.0 * c5 * T3;

                // dE/dc4
                gdC.row(i * 6 + 4) = 144.0 * c3 * T2 +
                                     384.0 * c4 * T3 +
                                     720.0 * c5 * T4;

                // dE/dc5
                gdC.row(i * 6 + 5) = 240.0 * c3 * T3 +
                                     720.0 * c4 * T4 +
                                     1440.0 * c5 * T5;
            }
        }

        /**
         * @brief Computes the partial gradient of the energy (jerk cost) with respect to the segment duration T.
         * By applying the Fundamental Theorem of Calculus, the derivative of the energy integral with respect to
         * its upper limit T is simply the squared norm of the jerk vector at the end of the segment. This avoids
         * complex polynomial expansion.
         * @param gdT [Output] Gradient vector to store the explicit time gradients.
         */
        void getEnergyPartialGradByTimes(Eigen::VectorXd &gdT) const
        {
            gdT.resize(num_segments_);

            for (int i = 0; i < num_segments_; ++i)
            {
                const double T = time_segments_[i];

                const RowVectorType c3 = coeffs_.row(i * 6 + 3);
                const RowVectorType c4 = coeffs_.row(i * 6 + 4);
                const RowVectorType c5 = coeffs_.row(i * 6 + 5);

                RowVectorType jerk_end = 6.0 * c3 + T * (24.0 * c4 + (60.0 * T) * c5);

                gdT(i) = jerk_end.squaredNorm();
            }
        }

        Eigen::VectorXd getEnergyGradTimes() const
        {
            Eigen::VectorXd grad(num_segments_);

            for (int i = 0; i < num_segments_; ++i)
            {
                const RowVectorType c1 = coeffs_.row(i * 6 + 1);
                const RowVectorType c2 = coeffs_.row(i * 6 + 2);
                const RowVectorType c3 = coeffs_.row(i * 6 + 3);
                const RowVectorType c4 = coeffs_.row(i * 6 + 4);
                const RowVectorType c5 = coeffs_.row(i * 6 + 5);

                double term_jerk = 36.0 * c3.squaredNorm();
                double term_sa = 96.0 * c2.dot(c4);
                double term_cv = 240.0 * c1.dot(c5);

                grad(i) = -term_jerk + term_sa - term_cv;
            }
            return grad;
        }
        MatrixType getEnergyGradInnerP() const
        {
            MatrixType grad(std::max(0, num_segments_ - 1), DIM);
            for (int i = 1; i < num_segments_; ++i)
            {
                const RowVectorType c5_L = coeffs_.row((i - 1) * 6 + 5);
                const RowVectorType c5_R = coeffs_.row(i * 6 + 5);
                grad.row(i - 1) = 240.0 * (c5_L - c5_R);
            }
            return grad;
        }

        struct Gradients
        {
            MatrixType points;
            Eigen::VectorXd times;
        };

        /**
         * @brief Computes gradients w.r.t. waypoints and time segments using the Analytic Adjoint Method.
         *
         * Efficiently propagates gradients by reusing the cached matrix factorization ($O(N)$ complexity)
         * instead of differentiating the linear system explicitly.
         *
         * **Mathematical Principle:**
         * 1. **Explicit Chain Rule:** Propagate $\partial E/\partial \mathbf{c}$ to positions ($P$) and internal derivatives ($\mathbf{d} = v, a$).
         * 2. **Adjoint Correction:** Solve $M^T \boldsymbol{\lambda} = \nabla_{\mathbf{d}} E$ to find Lagrange multipliers $\boldsymbol{\lambda}$ enforcing continuity.
         * 3. **Implicit Time Gradient:** Account for continuity breakage due to time scaling by accumulating $\boldsymbol{\lambda} \cdot \dot{\mathbf{d}}$ (Jerk, Snap).
         *
         * @param partialGradByCoeffs Input gradient w.r.t polynomial coefficients (6N x D).
         * @param partialGradByTimes  Input gradient w.r.t segment durations.
         * @param[out] gradByPoints   Output gradients for waypoints (positions).
         * @param[out] gradByTimes    Output gradients for segment durations.
         * @param includeEndpoints    If true, includes gradients for start and end points.
         */
        void propagateGrad(const MatrixType &partialGradByCoeffs,
                           const Eigen::VectorXd &partialGradByTimes,
                           MatrixType &gradByPoints,
                           Eigen::VectorXd &gradByTimes,
                           bool includeEndpoints = false)
        {
            const int n = num_segments_;
            const int n_pts = static_cast<int>(spatial_points_.size());

            gradByPoints = MatrixType::Zero(n_pts, DIM);
            gradByTimes = partialGradByTimes;

            Eigen::Matrix<double, Eigen::Dynamic, 2 * DIM, Eigen::RowMajor> gd_internal;
            gd_internal.resize(n_pts, 2 * DIM);
            gd_internal.setZero();

            auto add_grad_d = [&](int idx, const RowVectorType &d_vel, const RowVectorType &d_acc)
            {
                gd_internal.row(idx).segment(0, DIM) += d_vel;
                gd_internal.row(idx).segment(DIM, DIM) += d_acc;
            };

            for (int i = 0; i < n; ++i)
            {
                const auto &tp = time_powers_[i];
                const int coeff_idx = i * 6;

                const RowVectorType &gc0 = partialGradByCoeffs.row(coeff_idx + 0);
                const RowVectorType &gc1 = partialGradByCoeffs.row(coeff_idx + 1);
                const RowVectorType &gc2 = partialGradByCoeffs.row(coeff_idx + 2);
                const RowVectorType &gc3 = partialGradByCoeffs.row(coeff_idx + 3);
                const RowVectorType &gc4 = partialGradByCoeffs.row(coeff_idx + 4);
                const RowVectorType &gc5 = partialGradByCoeffs.row(coeff_idx + 5);

                gradByPoints.row(i) += gc0;

                double k_P3 = 10.0 * tp.h3_inv;
                double k_P4 = -15.0 * tp.h4_inv;
                double k_P5 = 6.0 * tp.h5_inv;

                RowVectorType sum_grad_P = gc3 * k_P3 + gc4 * k_P4 + gc5 * k_P5;
                gradByPoints.row(i + 1) += sum_grad_P;
                gradByPoints.row(i) -= sum_grad_P;

                RowVectorType grad_v_curr = gc1;
                grad_v_curr += gc3 * (-6.0 * tp.h2_inv);
                grad_v_curr += gc4 * (8.0 * tp.h3_inv);
                grad_v_curr += gc5 * (-3.0 * tp.h4_inv);

                RowVectorType grad_v_next = gc3 * (-4.0 * tp.h2_inv);
                grad_v_next += gc4 * (7.0 * tp.h3_inv);
                grad_v_next += gc5 * (-3.0 * tp.h4_inv);

                RowVectorType grad_a_curr = gc2 * 0.5;
                grad_a_curr += gc3 * (-1.5 * tp.h_inv);
                grad_a_curr += gc4 * (1.5 * tp.h2_inv);
                grad_a_curr += gc5 * (-0.5 * tp.h3_inv);

                RowVectorType grad_a_next = gc3 * (0.5 * tp.h_inv);
                grad_a_next += gc4 * (-1.0 * tp.h2_inv);
                grad_a_next += gc5 * (0.5 * tp.h3_inv);

                add_grad_d(i, grad_v_curr, grad_a_curr);
                add_grad_d(i + 1, grad_v_next, grad_a_next);

                const RowVectorType &P_curr = spatial_points_[i].transpose();
                const RowVectorType &P_next = spatial_points_[i + 1].transpose();
                const RowVectorType &V_curr = internal_vel_.row(i);
                const RowVectorType &V_next = internal_vel_.row(i + 1);
                const RowVectorType &A_curr = internal_acc_.row(i);
                const RowVectorType &A_next = internal_acc_.row(i + 1);

                RowVectorType P_diff = P_next - P_curr;

                RowVectorType dc3_dh = -30.0 * tp.h4_inv * P_diff +
                                       12.0 * tp.h3_inv * V_curr + 8.0 * tp.h3_inv * V_next +
                                       1.5 * tp.h2_inv * A_curr - 0.5 * tp.h2_inv * A_next;

                RowVectorType dc4_dh = 60.0 * tp.h5_inv * P_diff -
                                       24.0 * tp.h4_inv * V_curr - 21.0 * tp.h4_inv * V_next -
                                       3.0 * tp.h3_inv * A_curr + 2.0 * tp.h3_inv * A_next;

                RowVectorType dc5_dh = -30.0 * tp.h6_inv * P_diff +
                                       12.0 * tp.h5_inv * V_curr + 12.0 * tp.h5_inv * V_next +
                                       1.5 * tp.h4_inv * A_curr - 1.5 * tp.h4_inv * A_next;

                gradByTimes(i) += gc3.dot(dc3_dh) + gc4.dot(dc4_dh) + gc5.dot(dc5_dh);
            }

            const int num_blocks = n - 1;
            if (num_blocks > 0)
            {
                std::vector<Eigen::Matrix<double, 2, DIM>> lambda(num_blocks);

                for (int i = 0; i < num_blocks; ++i)
                {
                    lambda[i].row(0) = gd_internal.row(i + 1).segment(0, DIM);
                    lambda[i].row(1) = gd_internal.row(i + 1).segment(DIM, DIM);
                }

                {
                    Eigen::Matrix<double, 2, DIM> tmp = lambda[0];
                    Multiply2x2T_2xN(D_inv_cache_[0], tmp, lambda[0]);
                }

                for (int i = 0; i < num_blocks - 1; ++i)
                {
                    Eigen::Matrix<double, 2, DIM> update_term;
                    Multiply2x2T_2xN(U_blocks_cache_[i], lambda[i], update_term);
                    lambda[i + 1] -= update_term;

                    Eigen::Matrix<double, 2, DIM> tmp = lambda[i + 1];
                    Multiply2x2T_2xN(D_inv_cache_[i + 1], tmp, lambda[i + 1]);
                }

                for (int i = num_blocks - 2; i >= 0; --i)
                {
                    Eigen::Matrix<double, 2, DIM> update_term;
                    Multiply2x2T_2xN(L_blocks_cache_[i + 1], lambda[i + 1], update_term);

                    Eigen::Matrix<double, 2, DIM> scaled_update;
                    Multiply2x2T_2xN(D_inv_cache_[i], update_term, scaled_update);

                    lambda[i] -= scaled_update;
                }

                for (int i = 0; i < num_blocks; ++i)
                {
                    const int seg_idx = i;
                    const double T = time_segments_[seg_idx];

                    const int coeff_offset = seg_idx * 6;
                    const RowVectorType c4 = coeffs_.row(coeff_offset + 4);
                    const RowVectorType c5 = coeffs_.row(coeff_offset + 5);

                    const RowVectorType Snap_at_T = 24.0 * c4 + 120.0 * c5 * T;
                    const RowVectorType Crackle_at_T = 120.0 * c5;

                    const RowVectorType &lam_snap = lambda[i].row(0);
                    const RowVectorType &lam_jerk = lambda[i].row(1);

                    gradByTimes(seg_idx) += (lam_snap.dot(Crackle_at_T) + lam_jerk.dot(Snap_at_T));

                    const int k = i + 2;
                    const auto &tp_L = time_powers_[k - 2];
                    const auto &tp_R = time_powers_[k - 1];

                    double dr4_dp_next = 360.0 * tp_R.h5_inv;
                    double dr4_dp_curr = -360.0 * (tp_R.h5_inv + tp_L.h5_inv);
                    double dr4_dp_prev = 360.0 * tp_L.h5_inv;

                    double dr3_dp_next = 60.0 * tp_R.h4_inv;
                    double dr3_dp_curr = -60.0 * (tp_R.h4_inv + tp_L.h4_inv);
                    double dr3_dp_prev = 60.0 * tp_L.h4_inv;

                    RowVectorType grad_P_next = lam_snap * dr4_dp_next + lam_jerk * dr3_dp_next;
                    RowVectorType grad_P_curr = lam_snap * dr4_dp_curr + lam_jerk * dr3_dp_curr;
                    RowVectorType grad_P_prev = lam_snap * dr4_dp_prev + lam_jerk * dr3_dp_prev;

                    gradByPoints.row(i + 2) += grad_P_next;
                    gradByPoints.row(i + 1) += grad_P_curr;
                    gradByPoints.row(i) += grad_P_prev;
                }
            }

            if (!includeEndpoints && n > 1)
            {
                gradByPoints = gradByPoints.middleRows(1, n - 1).eval();
            }
        }

        /**
         * @brief Convenience wrapper for the Analytic Adjoint Method returning a Gradients structure.
         *
         * Wraps the in-place implementation to return point and time gradients in a single structure.
         * Uses the same $O(N)$ cached matrix factorization strategy to avoid explicit differentiation.
         *
         * @param partialGradByCoeffs Input gradient w.r.t polynomial coefficients (6N x D).
         * @param partialGradByTimes  Input gradient w.r.t segment durations.
         * @param includeEndpoints    If true, includes gradients for start and end points.
         * @return Gradients Structure containing gradients for waypoints (positions) and times.
         */
        Gradients propagateGrad(const MatrixType &partialGradByCoeffs,
                                const Eigen::VectorXd &partialGradByTimes,
                                bool includeEndpoints = false)
        {
            Gradients result;
            propagateGrad(partialGradByCoeffs, partialGradByTimes, result.points, result.times, includeEndpoints);
            return result;
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

        static inline void Multiply2x2(const Eigen::Matrix2d &A, const Eigen::Matrix2d &B, Eigen::Matrix2d &C_out) noexcept
        {
            const double a00 = A(0, 0), a01 = A(0, 1);
            const double a10 = A(1, 0), a11 = A(1, 1);
            const double b00 = B(0, 0), b01 = B(0, 1);
            const double b10 = B(1, 0), b11 = B(1, 1);

            C_out(0, 0) = a00 * b00 + a01 * b10;
            C_out(0, 1) = a00 * b01 + a01 * b11;
            C_out(1, 0) = a10 * b00 + a11 * b10;
            C_out(1, 1) = a10 * b01 + a11 * b11;
        }

        template <int N>
        static inline void Multiply2x2_2xN(const Eigen::Matrix2d &A, const Eigen::Matrix<double, 2, N> &B,
                                           Eigen::Matrix<double, 2, N> &C_out) noexcept
        {
            const double a00 = A(0, 0), a01 = A(0, 1);
            const double a10 = A(1, 0), a11 = A(1, 1);

            for (int j = 0; j < N; ++j)
            {
                const double b0j = B(0, j);
                const double b1j = B(1, j);
                C_out(0, j) = a00 * b0j + a01 * b1j;
                C_out(1, j) = a10 * b0j + a11 * b1j;
            }
        }

        template <int N>
        static inline void Multiply2x2T_2xN(const Eigen::Matrix2d &A, const Eigen::Matrix<double, 2, N> &B,
                                            Eigen::Matrix<double, 2, N> &C_out) noexcept
        {
            const double a00 = A(0, 0), a01 = A(0, 1);
            const double a10 = A(1, 0), a11 = A(1, 1);

            for (int j = 0; j < N; ++j)
            {
                const double b0j = B(0, j);
                const double b1j = B(1, j);

                C_out(0, j) = a00 * b0j + a10 * b1j;
                C_out(1, j) = a01 * b0j + a11 * b1j;
            }
        }

        void solveInternalDerivatives(const MatrixType &P, MatrixType &p_out, MatrixType &q_out)
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

            U_blocks_cache_.resize(std::max(0, num_blocks - 1));
            D_inv_cache_.resize(num_blocks);
            L_blocks_cache_.resize(num_blocks);
            ws_rhs_mod_.resize(num_blocks);

            for (int i = 0; i < num_blocks; ++i)
            {
                const int k = i + 2;
                const auto &tp_L = time_powers_[k - 2];
                const auto &tp_R = time_powers_[k - 1];

                Eigen::Matrix<double, 1, DIM> r3 = 60.0 * ((P.row(k) - P.row(k - 1)) * tp_R.h3_inv - (P.row(k - 1) - P.row(k - 2)) * tp_L.h3_inv);
                Eigen::Matrix<double, 1, DIM> r4 = 360.0 * ((P.row(k - 1) - P.row(k)) * tp_R.h4_inv + (P.row(k - 2) - P.row(k - 1)) * tp_L.h4_inv);

                Eigen::Matrix<double, 2, DIM> r;
                r.row(0) = r4;
                r.row(1) = r3;

                Eigen::Matrix2d D;
                D << -192.0 * (tp_L.h3_inv + tp_R.h3_inv), 36.0 * (tp_L.h2_inv - tp_R.h2_inv),
                    -36.0 * (tp_L.h2_inv - tp_R.h2_inv), 9.0 * (tp_L.h_inv + tp_R.h_inv);

                Eigen::Matrix2d L;
                L << -168.0 * tp_L.h3_inv, -24.0 * tp_L.h2_inv,
                    -24.0 * tp_L.h2_inv, -3.0 * tp_L.h_inv;
                L_blocks_cache_[i] = L;

                if (k < n - 1)
                {
                    Eigen::Matrix2d U;
                    U << -168.0 * tp_R.h3_inv, 24.0 * tp_R.h2_inv,
                        24.0 * tp_R.h2_inv, -3.0 * tp_R.h_inv;
                    U_blocks_cache_[i] = U;
                }

                if (i == 0) // k == 2
                {
                    r.noalias() -= L * B_left;
                }
                else
                {
                    Eigen::Matrix2d X;
                    Multiply2x2(D_inv_cache_[i - 1], U_blocks_cache_[i - 1], X);
                    Eigen::Matrix<double, 2, DIM> Y;
                    Multiply2x2_2xN(D_inv_cache_[i - 1], ws_rhs_mod_[i - 1], Y);

                    D.noalias() -= L * X;
                    r.noalias() -= L * Y;
                }

                if (k == n - 1)
                {
                    Eigen::Matrix2d U_last;
                    U_last << -168.0 * tp_R.h3_inv, 24.0 * tp_R.h2_inv,
                        24.0 * tp_R.h2_inv, -3.0 * tp_R.h_inv;
                    r.noalias() -= U_last * B_right;
                }

                Inverse2x2(D, D_inv_cache_[i]);
                ws_rhs_mod_[i] = r;
            }

            ws_solution_.resize(num_blocks);

            Multiply2x2_2xN(D_inv_cache_[num_blocks - 1], ws_rhs_mod_[num_blocks - 1], ws_solution_[num_blocks - 1]);

            for (int i = num_blocks - 2; i >= 0; --i)
            {
                const Eigen::Matrix<double, 2, DIM> rhs_temp = ws_rhs_mod_[i] - U_blocks_cache_[i] * ws_solution_[i + 1];
                Multiply2x2_2xN(D_inv_cache_[i], rhs_temp, ws_solution_[i]);
            }

            for (int i = 0; i < num_blocks; ++i)
            {
                p_out.row(i + 1) = ws_solution_[i].row(0);
                q_out.row(i + 1) = ws_solution_[i].row(1);
            }
        }

        MatrixType solveQuintic()
        {
            const int n_pts = static_cast<int>(spatial_points_.size());
            const int n = num_segments_;

            MatrixType P(n_pts, DIM);
            for (int i = 0; i < n_pts; ++i)
            {
                P.row(i) = spatial_points_[i].transpose();
            }

            solveInternalDerivatives(P, internal_vel_, internal_acc_);

            MatrixType coeffs(n * 6, DIM);

            for (int i = 0; i < n; ++i)
            {
                const auto &tp = time_powers_[i];

                const RowVectorType c0 = P.row(i);
                const RowVectorType c1 = internal_vel_.row(i);
                const RowVectorType c2 = internal_acc_.row(i) * 0.5;

                const RowVectorType rhs1 = P.row(i + 1) - c0 - c1 * tp.h - c2 * (tp.h * tp.h);
                const RowVectorType rhs2 = internal_vel_.row(i + 1) - c1 - (2.0 * c2) * tp.h;
                const RowVectorType rhs3 = internal_acc_.row(i + 1) - (2.0 * c2);

                const RowVectorType c3 = (10.0 * tp.h3_inv) * rhs1 - (4.0 * tp.h2_inv) * rhs2 + (0.5 * tp.h_inv) * rhs3;
                const RowVectorType c4 = (-15.0 * tp.h4_inv) * rhs1 + (7.0 * tp.h3_inv) * rhs2 - (tp.h2_inv) * rhs3;
                const RowVectorType c5 = (6.0 * tp.h5_inv) * rhs1 - (3.0 * tp.h4_inv) * rhs2 + (0.5 * tp.h3_inv) * rhs3;

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

    template <int DIM>
    class SepticSplineND
    {
    public:
        using VectorType = Eigen::Matrix<double, DIM, 1>;
        using RowVectorType = Eigen::Matrix<double, 1, DIM>;
        static constexpr int kMatrixOptions = (DIM == 1) ? Eigen::ColMajor : Eigen::RowMajor;
        using MatrixType = Eigen::Matrix<double, Eigen::Dynamic, DIM, kMatrixOptions>;

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

        std::vector<Eigen::Matrix3d, Eigen::aligned_allocator<Eigen::Matrix3d>> D_inv_cache_;
        std::vector<Eigen::Matrix3d, Eigen::aligned_allocator<Eigen::Matrix3d>> U_blocks_cache_;
        std::vector<Eigen::Matrix3d, Eigen::aligned_allocator<Eigen::Matrix3d>> L_blocks_cache_;

        std::vector<Eigen::Matrix<double, 3, DIM>, Eigen::aligned_allocator<Eigen::Matrix<double, 3, DIM>>> ws_rhs_mod_;
        std::vector<Eigen::Matrix<double, 3, DIM>, Eigen::aligned_allocator<Eigen::Matrix<double, 3, DIM>>> ws_d_rhs_mod_;
        std::vector<Eigen::Matrix<double, 3, DIM>, Eigen::aligned_allocator<Eigen::Matrix<double, 3, DIM>>> ws_current_lambda_;
        std::vector<Eigen::Matrix<double, 3, DIM>, Eigen::aligned_allocator<Eigen::Matrix<double, 3, DIM>>> ws_solution_;

        MatrixType internal_vel_;
        MatrixType internal_acc_;
        MatrixType internal_jerk_;

    private:
        void updateSplineInternal()
        {
            num_segments_ = static_cast<int>(time_segments_.size());
            updateCumulativeTimes();
            precomputeTimePowers();
            coeffs_ = solveSepticSpline();
            is_initialized_ = true;
            initializePPoly();
        }

    public:
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW

        SepticSplineND() = default;

        SepticSplineND(const std::vector<double> &t_points,
                       const SplineVector<VectorType> &spatial_points,
                       const BoundaryConditions<DIM> &boundary = BoundaryConditions<DIM>())
            : spatial_points_(spatial_points),
              boundary_(boundary)
        {
            convertTimePointsToSegments(t_points);
            updateSplineInternal();
        }

        SepticSplineND(const std::vector<double> &time_segments,
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

        const SplineVector<VectorType> &getSpacePoints() const { return spatial_points_; }
        const std::vector<double> &getTimeSegments() const { return time_segments_; }
        const std::vector<double> &getCumulativeTimes() const { return cumulative_times_; }
        const BoundaryConditions<DIM> &getBoundaryConditions() const { return boundary_; }

        const PPolyND<DIM> &getTrajectory() const { return trajectory_; }
        PPolyND<DIM> getTrajectoryCopy() const { return trajectory_; }
        const PPolyND<DIM> &getPPoly() const { return trajectory_; }
        PPolyND<DIM> getPPolyCopy() const { return trajectory_; }

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
                const double T6 = T4 * T2;
                const double T7 = T4 * T3;

                // c4, c5, c6, c7
                // p(t) = c0 + c1*t + c2*t^2 + c3*t^3 + c4*t^4 + c5*t^5 + c6*t^6 + c7*t^7
                RowVectorType c4 = coeffs_.row(i * 8 + 4);
                RowVectorType c5 = coeffs_.row(i * 8 + 5);
                RowVectorType c6 = coeffs_.row(i * 8 + 6);
                RowVectorType c7 = coeffs_.row(i * 8 + 7);

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
         * @brief Computes the partial gradient of the energy (snap cost) w.r.t polynomial coefficients.
         * @param gdC [Output] Matrix of size (num_segments * 8) x DIM.
         */
        void getEnergyPartialGradByCoeffs(MatrixType &gdC) const
        {
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

                const RowVectorType c4 = coeffs_.row(i * 8 + 4);
                const RowVectorType c5 = coeffs_.row(i * 8 + 5);
                const RowVectorType c6 = coeffs_.row(i * 8 + 6);
                const RowVectorType c7 = coeffs_.row(i * 8 + 7);

                // dE/dc4
                gdC.row(i * 8 + 4) = 1152.0 * c4 * T +
                                     2880.0 * c5 * T2 +
                                     5760.0 * c6 * T3 +
                                     10080.0 * c7 * T4;

                // dE/dc5
                gdC.row(i * 8 + 5) = 2880.0 * c4 * T2 +
                                     9600.0 * c5 * T3 +
                                     21600.0 * c6 * T4 +
                                     40320.0 * c7 * T5;

                // dE/dc6
                gdC.row(i * 8 + 6) = 5760.0 * c4 * T3 +
                                     21600.0 * c5 * T4 +
                                     51840.0 * c6 * T5 +
                                     100800.0 * c7 * T6;

                // dE/dc7
                gdC.row(i * 8 + 7) = 10080.0 * c4 * T4 +
                                     40320.0 * c5 * T5 +
                                     100800.0 * c6 * T6 +
                                     201600.0 * c7 * T7;
            }
        }

        /**
         * @brief Computes the partial gradient of the energy (snap cost) with respect to the segment duration T.
         * By applying the Fundamental Theorem of Calculus, the derivative of the energy integral with respect to
         * its upper limit T is simply the squared norm of the snap vector at the end of the segment.
         * This avoids complex polynomial expansion.
         * @param gdT [Output] Gradient vector to store the explicit time gradients.
         */
        void getEnergyPartialGradByTimes(Eigen::VectorXd &gdT) const
        {
            gdT.resize(num_segments_);

            for (int i = 0; i < num_segments_; ++i)
            {
                const double T = time_segments_[i];

                const RowVectorType c4 = coeffs_.row(i * 8 + 4);
                const RowVectorType c5 = coeffs_.row(i * 8 + 5);
                const RowVectorType c6 = coeffs_.row(i * 8 + 6);
                const RowVectorType c7 = coeffs_.row(i * 8 + 7);

                RowVectorType snap_end = 24.0 * c4 + T * (120.0 * c5 + T * (360.0 * c6 + (840.0 * T) * c7));

                gdT(i) = snap_end.squaredNorm();
            }
        }

        Eigen::VectorXd getEnergyGradTimes() const
        {
            Eigen::VectorXd grad(num_segments_);

            for (int i = 0; i < num_segments_; ++i)
            {

                const int offset = i * 8;
                const RowVectorType c1 = coeffs_.row(offset + 1);
                const RowVectorType c2 = coeffs_.row(offset + 2);
                const RowVectorType c3 = coeffs_.row(offset + 3);
                const RowVectorType c4 = coeffs_.row(offset + 4);
                const RowVectorType c5 = coeffs_.row(offset + 5);
                const RowVectorType c6 = coeffs_.row(offset + 6);
                const RowVectorType c7 = coeffs_.row(offset + 7);

                double term_snap = 576.0 * c4.squaredNorm();
                double term_cj = 1440.0 * c3.dot(c5);
                double term_pa = 2880.0 * c2.dot(c6);
                double term_dv = 10080.0 * c1.dot(c7);

                grad(i) = -term_snap + term_cj - term_pa + term_dv;
            }
            return grad;
        }
        MatrixType getEnergyGradInnerP() const
        {
            MatrixType grad(std::max(0, num_segments_ - 1), DIM);
            for (int i = 1; i < num_segments_; ++i)
            {
                const RowVectorType c7_L = coeffs_.row((i - 1) * 8 + 7);
                const RowVectorType c7_R = coeffs_.row(i * 8 + 7);
                grad.row(i - 1) = 10080.0 * (c7_R - c7_L);
            }
            return grad;
        }

        /**
         * @brief Computes gradients w.r.t. waypoints and time segments using the Analytic Adjoint Method.
         *
         * Efficiently propagates gradients by reusing the cached matrix factorization ($O(N)$ complexity)
         * instead of differentiating the linear system explicitly.
         *
         * **Mathematical Principle:**
         * 1. **Explicit Chain Rule:** Propagate $\partial E/\partial \mathbf{c}$ to positions ($P$) and internal derivatives ($\mathbf{d} = v, a, j$).
         * 2. **Adjoint Correction:** Solve $M^T \boldsymbol{\lambda} = \nabla_{\mathbf{d}} E$ to find Lagrange multipliers $\boldsymbol{\lambda}$ enforcing continuity.
         * 3. **Implicit Time Gradient:** Account for continuity breakage due to time scaling by accumulating $\boldsymbol{\lambda} \cdot \dot{\mathbf{d}}$ (Snap, Crackle, Pop).
         *
         * @param partialGradByCoeffs Input gradient w.r.t polynomial coefficients (8N x D).
         * @param partialGradByTimes  Input gradient w.r.t segment durations.
         * @param[out] gradByPoints   Output gradients for waypoints (positions).
         * @param[out] gradByTimes    Output gradients for segment durations.
         * @param includeEndpoints    If true, includes gradients for start and end points.
         */
        void propagateGrad(const MatrixType &partialGradByCoeffs,
                           const Eigen::VectorXd &partialGradByTimes,
                           MatrixType &gradByPoints,
                           Eigen::VectorXd &gradByTimes,
                           bool includeEndpoints = false)
        {
            const int n = num_segments_;
            const int n_pts = static_cast<int>(spatial_points_.size());

            gradByPoints = MatrixType::Zero(n_pts, DIM);
            gradByTimes = partialGradByTimes;

            Eigen::Matrix<double, Eigen::Dynamic, 3 * DIM, Eigen::RowMajor> gd_internal;
            gd_internal.resize(n_pts, 3 * DIM);
            gd_internal.setZero();

            auto add_grad_d = [&](int idx, const RowVectorType &d_vel, const RowVectorType &d_acc, const RowVectorType &d_jerk)
            {
                gd_internal.row(idx).segment(0, DIM) += d_vel;
                gd_internal.row(idx).segment(DIM, DIM) += d_acc;
                gd_internal.row(idx).segment(2 * DIM, DIM) += d_jerk;
            };

            for (int i = 0; i < n; ++i)
            {
                const auto &tp = time_powers_[i];
                const int coeff_idx = i * 8;

                const RowVectorType &gc0 = partialGradByCoeffs.row(coeff_idx + 0);
                const RowVectorType &gc1 = partialGradByCoeffs.row(coeff_idx + 1);
                const RowVectorType &gc2 = partialGradByCoeffs.row(coeff_idx + 2);
                const RowVectorType &gc3 = partialGradByCoeffs.row(coeff_idx + 3);
                const RowVectorType &gc4 = partialGradByCoeffs.row(coeff_idx + 4);
                const RowVectorType &gc5 = partialGradByCoeffs.row(coeff_idx + 5);
                const RowVectorType &gc6 = partialGradByCoeffs.row(coeff_idx + 6);
                const RowVectorType &gc7 = partialGradByCoeffs.row(coeff_idx + 7);

                gradByPoints.row(i) += gc0;

                double k_P4 = -(210.0 * tp.h4_inv) / 6.0;
                double k_P5 = (168.0 * tp.h5_inv) / 2.0;
                double k_P6 = -(420.0 * tp.h6_inv) / 6.0;
                double k_P7 = (120.0 * tp.h7_inv) / 6.0;

                RowVectorType sum_grad_P = gc4 * k_P4 + gc5 * k_P5 + gc6 * k_P6 + gc7 * k_P7;
                gradByPoints.row(i) += sum_grad_P;
                gradByPoints.row(i + 1) -= sum_grad_P;

                add_grad_d(i, gc1, 0.5 * gc2, (1.0 / 6.0) * gc3);

                RowVectorType grad_v_curr = gc4 * (-20.0 * tp.h3_inv) +
                                            gc5 * (45.0 * tp.h4_inv) +
                                            gc6 * (-36.0 * tp.h5_inv) +
                                            gc7 * (10.0 * tp.h6_inv);

                RowVectorType grad_v_next = gc4 * (-15.0 * tp.h3_inv) +
                                            gc5 * (39.0 * tp.h4_inv) +
                                            gc6 * (-34.0 * tp.h5_inv) +
                                            gc7 * (10.0 * tp.h6_inv);

                RowVectorType grad_a_curr = gc4 * (-5.0 * tp.h2_inv) +
                                            gc5 * (10.0 * tp.h3_inv) +
                                            gc6 * (-7.5 * tp.h4_inv) +
                                            gc7 * (2.0 * tp.h5_inv);

                RowVectorType grad_a_next = gc4 * (2.5 * tp.h2_inv) +
                                            gc5 * (-7.0 * tp.h3_inv) +
                                            gc6 * (6.5 * tp.h4_inv) +
                                            gc7 * (-2.0 * tp.h5_inv);

                RowVectorType grad_j_curr = gc4 * (-2.0 / 3.0 * tp.h_inv) +
                                            gc5 * (tp.h2_inv) +
                                            gc6 * (-2.0 / 3.0 * tp.h3_inv) +
                                            gc7 * (1.0 / 6.0 * tp.h4_inv);

                RowVectorType grad_j_next = gc4 * (-1.0 / 6.0 * tp.h_inv) +
                                            gc5 * (0.5 * tp.h2_inv) +
                                            gc6 * (-0.5 * tp.h3_inv) +
                                            gc7 * (1.0 / 6.0 * tp.h4_inv);

                add_grad_d(i, grad_v_curr, grad_a_curr, grad_j_curr);
                add_grad_d(i + 1, grad_v_next, grad_a_next, grad_j_next);

                {
                    const RowVectorType &P_curr = spatial_points_[i].transpose();
                    const RowVectorType &P_next = spatial_points_[i + 1].transpose();
                    const RowVectorType &V_curr = internal_vel_.row(i);
                    const RowVectorType &V_next = internal_vel_.row(i + 1);
                    const RowVectorType &A_curr = internal_acc_.row(i);
                    const RowVectorType &A_next = internal_acc_.row(i + 1);
                    const RowVectorType &J_curr = internal_jerk_.row(i);
                    const RowVectorType &J_next = internal_jerk_.row(i + 1);

                    RowVectorType dP = P_curr - P_next;

                    RowVectorType dc4_dh = ((840.0 * dP * tp.h5_inv) +
                                            (360.0 * V_curr * tp.h4_inv) + (270.0 * V_next * tp.h4_inv) +
                                            (60.0 * A_curr * tp.h3_inv) - (30.0 * A_next * tp.h3_inv) +
                                            (4.0 * J_curr * tp.h2_inv) + (J_next * tp.h2_inv)) /
                                           6.0;

                    RowVectorType dc5_dh = ((-840.0 * dP * tp.h6_inv) -
                                            (360.0 * V_curr * tp.h5_inv) - (312.0 * V_next * tp.h5_inv) -
                                            (60.0 * A_curr * tp.h4_inv) + (42.0 * A_next * tp.h4_inv) -
                                            (4.0 * J_curr * tp.h3_inv) - (2.0 * J_next * tp.h3_inv)) /
                                           2.0;

                    RowVectorType dc6_dh = ((2520.0 * dP * tp.h7_inv) +
                                            (1080.0 * V_curr * tp.h6_inv) + (1020.0 * V_next * tp.h6_inv) +
                                            (180.0 * A_curr * tp.h5_inv) - (156.0 * A_next * tp.h5_inv) +
                                            (12.0 * J_curr * tp.h4_inv) + (9.0 * J_next * tp.h4_inv)) /
                                           6.0;

                    double h8_inv = tp.h7_inv * tp.h_inv;
                    RowVectorType dc7_dh = ((-840.0 * dP * h8_inv) -
                                            (360.0 * V_curr * tp.h7_inv) - (360.0 * V_next * tp.h7_inv) -
                                            (60.0 * A_curr * tp.h6_inv) + (60.0 * A_next * tp.h6_inv) -
                                            (4.0 * J_curr * tp.h5_inv) - (4.0 * J_next * tp.h5_inv)) /
                                           6.0;

                    gradByTimes(i) += gc4.dot(dc4_dh) + gc5.dot(dc5_dh) + gc6.dot(dc6_dh) + gc7.dot(dc7_dh);
                }
            }
            const int num_blocks = n - 1;
            if (num_blocks > 0)
            {
                std::vector<Eigen::Matrix<double, 3, DIM>> lambda(num_blocks);

                for (int i = 0; i < num_blocks; ++i)
                {
                    lambda[i].row(0) = gd_internal.row(i + 1).segment(0, DIM);
                    lambda[i].row(1) = gd_internal.row(i + 1).segment(DIM, DIM);
                    lambda[i].row(2) = gd_internal.row(i + 1).segment(2 * DIM, DIM);
                }

                {
                    Eigen::Matrix<double, 3, DIM> tmp = lambda[0];
                    Multiply3x3T_3xN(D_inv_cache_[0], tmp, lambda[0]);
                }
                for (int i = 0; i < num_blocks - 1; ++i)
                {
                    Eigen::Matrix<double, 3, DIM> update_term;
                    Multiply3x3T_3xN(U_blocks_cache_[i], lambda[i], update_term);
                    lambda[i + 1] -= update_term;

                    Eigen::Matrix<double, 3, DIM> tmp = lambda[i + 1];
                    Multiply3x3T_3xN(D_inv_cache_[i + 1], tmp, lambda[i + 1]);
                }

                for (int i = num_blocks - 2; i >= 0; --i)
                {
                    Eigen::Matrix<double, 3, DIM> update_term;
                    Multiply3x3T_3xN(L_blocks_cache_[i + 1], lambda[i + 1], update_term);

                    Eigen::Matrix<double, 3, DIM> scaled_update;
                    Multiply3x3T_3xN(D_inv_cache_[i], update_term, scaled_update);

                    lambda[i] -= scaled_update;
                }

                for (int i = 0; i < num_blocks; ++i)
                {
                    const int seg_idx = i;
                    const double T = time_segments_[seg_idx];
                    const double T2 = T * T;

                    const int coeff_offset = seg_idx * 8;
                    const RowVectorType c5 = coeffs_.row(coeff_offset + 5);
                    const RowVectorType c6 = coeffs_.row(coeff_offset + 6);
                    const RowVectorType c7 = coeffs_.row(coeff_offset + 7);

                    RowVectorType dSnap_dT = 120.0 * c5 + 720.0 * c6 * T + 2520.0 * c7 * T2;
                    RowVectorType dCrackle_dT = 720.0 * c6 + 5040.0 * c7 * T;
                    RowVectorType dPop_dT = 5040.0 * c7;

                    const RowVectorType &lam_snap = lambda[i].row(0);
                    const RowVectorType &lam_crackle = lambda[i].row(1);
                    const RowVectorType &lam_pop = lambda[i].row(2);

                    gradByTimes(seg_idx) += (lam_snap.dot(dSnap_dT) +
                                             lam_crackle.dot(dCrackle_dT) +
                                             lam_pop.dot(dPop_dT));

                    const int k = i + 2;
                    const auto &tp_L = time_powers_[k - 2];
                    const auto &tp_R = time_powers_[k - 1];

                    double dr3_dp_next = 840.0 * tp_R.h4_inv;
                    double dr4_dp_next = -10080.0 * tp_R.h5_inv;
                    double dr5_dp_next = 50400.0 * tp_R.h6_inv;
                    RowVectorType grad_P_next = lam_snap * dr3_dp_next + lam_crackle * dr4_dp_next + lam_pop * dr5_dp_next;
                    gradByPoints.row(i + 2) += grad_P_next;

                    double dr3_dp_curr = -840.0 * (tp_R.h4_inv - tp_L.h4_inv);
                    double dr4_dp_curr = 10080.0 * (tp_R.h5_inv + tp_L.h5_inv);
                    double dr5_dp_curr = -50400.0 * (tp_R.h6_inv - tp_L.h6_inv);
                    RowVectorType grad_P_curr = lam_snap * dr3_dp_curr + lam_crackle * dr4_dp_curr + lam_pop * dr5_dp_curr;
                    gradByPoints.row(i + 1) += grad_P_curr;

                    double dr3_dp_prev = -840.0 * tp_L.h4_inv;
                    double dr4_dp_prev = -10080.0 * tp_L.h5_inv;
                    double dr5_dp_prev = -50400.0 * tp_L.h6_inv;
                    RowVectorType grad_P_prev = lam_snap * dr3_dp_prev + lam_crackle * dr4_dp_prev + lam_pop * dr5_dp_prev;
                    gradByPoints.row(i) += grad_P_prev;
                }
            }

            if (!includeEndpoints && n > 1)
            {
                gradByPoints = gradByPoints.middleRows(1, n - 1).eval();
            }
        }
        struct Gradients
        {
            MatrixType points;
            Eigen::VectorXd times;
        };

        /**
         * @brief Convenience wrapper for the Analytic Adjoint Method returning a Gradients structure.
         *
         * Wraps the in-place implementation to return point and time gradients in a single structure.
         * Uses the same $O(N)$ cached matrix factorization strategy to avoid explicit differentiation.
         *
         * @param partialGradByCoeffs Input gradient w.r.t polynomial coefficients (8N x D).
         * @param partialGradByTimes  Input gradient w.r.t segment durations.
         * @param includeEndpoints    If true, includes gradients for start and end points.
         * @return Gradients Structure containing gradients for waypoints (positions) and times.
         */
        Gradients propagateGrad(const MatrixType &partialGradByCoeffs,
                                const Eigen::VectorXd &partialGradByTimes,
                                bool includeEndpoints = false)
        {
            Gradients result;
            propagateGrad(partialGradByCoeffs, partialGradByTimes, result.points, result.times, includeEndpoints);
            return result;
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

        static inline void Multiply3x3(const Eigen::Matrix3d &A, const Eigen::Matrix3d &B, Eigen::Matrix3d &C_out) noexcept
        {
            const double a00 = A(0, 0), a01 = A(0, 1), a02 = A(0, 2);
            const double a10 = A(1, 0), a11 = A(1, 1), a12 = A(1, 2);
            const double a20 = A(2, 0), a21 = A(2, 1), a22 = A(2, 2);

            const double b00 = B(0, 0), b01 = B(0, 1), b02 = B(0, 2);
            const double b10 = B(1, 0), b11 = B(1, 1), b12 = B(1, 2);
            const double b20 = B(2, 0), b21 = B(2, 1), b22 = B(2, 2);

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

        template <int N>
        static inline void Multiply3x3_3xN(const Eigen::Matrix3d &A, const Eigen::Matrix<double, 3, N> &B,
                                           Eigen::Matrix<double, 3, N> &C_out) noexcept
        {
            const double a00 = A(0, 0), a01 = A(0, 1), a02 = A(0, 2);
            const double a10 = A(1, 0), a11 = A(1, 1), a12 = A(1, 2);
            const double a20 = A(2, 0), a21 = A(2, 1), a22 = A(2, 2);

            for (int j = 0; j < N; ++j)
            {
                const double b0j = B(0, j);
                const double b1j = B(1, j);
                const double b2j = B(2, j);

                C_out(0, j) = a00 * b0j + a01 * b1j + a02 * b2j;
                C_out(1, j) = a10 * b0j + a11 * b1j + a12 * b2j;
                C_out(2, j) = a20 * b0j + a21 * b1j + a22 * b2j;
            }
        }

        template <int N>
        static inline void Multiply3x3T_3xN(const Eigen::Matrix3d &A, const Eigen::Matrix<double, 3, N> &B,
                                            Eigen::Matrix<double, 3, N> &C_out) noexcept
        {
            const double a00 = A(0, 0), a01 = A(0, 1), a02 = A(0, 2);
            const double a10 = A(1, 0), a11 = A(1, 1), a12 = A(1, 2);
            const double a20 = A(2, 0), a21 = A(2, 1), a22 = A(2, 2);

            for (int j = 0; j < N; ++j)
            {
                const double b0j = B(0, j);
                const double b1j = B(1, j);
                const double b2j = B(2, j);

                C_out(0, j) = a00 * b0j + a10 * b1j + a20 * b2j;
                C_out(1, j) = a01 * b0j + a11 * b1j + a21 * b2j;
                C_out(2, j) = a02 * b0j + a12 * b1j + a22 * b2j;
            }
        }

    private:
        void solveInternalDerivatives(const MatrixType &P,
                                      MatrixType &p_out,
                                      MatrixType &q_out,
                                      MatrixType &s_out)
        {
            const int n = static_cast<int>(P.rows());
            p_out.resize(n, DIM);
            q_out.resize(n, DIM);
            s_out.resize(n, DIM);

            // Boundary conditions
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

            U_blocks_cache_.resize(std::max(0, num_blocks - 1));
            D_inv_cache_.resize(num_blocks);
            L_blocks_cache_.resize(num_blocks);
            ws_rhs_mod_.resize(num_blocks);

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

                L_blocks_cache_[i] = L;

                if (i < num_blocks - 1)
                {
                    Eigen::Matrix3d U;
                    U << 360.0 * tp_R.h3_inv, -60.0 * tp_R.h2_inv, 4.0 * tp_R.h_inv,
                        -4680.0 * tp_R.h4_inv, 840.0 * tp_R.h3_inv, -60.0 * tp_R.h2_inv,
                        24480.0 * tp_R.h5_inv, -4680.0 * tp_R.h4_inv, 360.0 * tp_R.h3_inv;

                    U_blocks_cache_[i] = U;
                }

                if (i == 0)
                {
                    r.noalias() -= L * B_left;
                }
                else
                {
                    const Eigen::Matrix3d &D_prev_inv = D_inv_cache_[i - 1];
                    const Eigen::Matrix3d &U_prev = U_blocks_cache_[i - 1];
                    const Eigen::Matrix<double, 3, DIM> &r_prev = ws_rhs_mod_[i - 1];

                    Eigen::Matrix3d X;
                    Multiply3x3(D_prev_inv, U_prev, X);
                    Eigen::Matrix<double, 3, DIM> Y;
                    Multiply3x3_3xN(D_prev_inv, r_prev, Y);

                    D.noalias() -= L * X;
                    r.noalias() -= L * Y;
                }

                if (i == num_blocks - 1)
                {
                    Eigen::Matrix3d U_last;
                    U_last << 360.0 * tp_R.h3_inv, -60.0 * tp_R.h2_inv, 4.0 * tp_R.h_inv,
                        -4680.0 * tp_R.h4_inv, 840.0 * tp_R.h3_inv, -60.0 * tp_R.h2_inv,
                        24480.0 * tp_R.h5_inv, -4680.0 * tp_R.h4_inv, 360.0 * tp_R.h3_inv;

                    r.noalias() -= U_last * B_right;
                }

                Inverse3x3(D, D_inv_cache_[i]);
                ws_rhs_mod_[i] = r;
            }

            ws_solution_.resize(num_blocks);

            Multiply3x3_3xN(D_inv_cache_[num_blocks - 1], ws_rhs_mod_[num_blocks - 1], ws_solution_[num_blocks - 1]);

            for (int i = num_blocks - 2; i >= 0; --i)
            {
                const Eigen::Matrix<double, 3, DIM> rhs_temp = ws_rhs_mod_[i] - U_blocks_cache_[i] * ws_solution_[i + 1];
                Multiply3x3_3xN(D_inv_cache_[i], rhs_temp, ws_solution_[i]);
            }

            for (int i = 0; i < num_blocks; ++i)
            {
                const int row = i + 1;
                p_out.row(row) = ws_solution_[i].row(0);
                q_out.row(row) = ws_solution_[i].row(1);
                s_out.row(row) = ws_solution_[i].row(2);
            }
        }

        MatrixType solveSepticSpline()
        {
            const int n_pts = static_cast<int>(spatial_points_.size());
            const int n = num_segments_;

            // 1. Construct Points Matrix
            MatrixType P(n_pts, DIM);
            for (int i = 0; i < n_pts; ++i)
            {
                P.row(i) = spatial_points_[i].transpose();
            }

            solveInternalDerivatives(P, internal_vel_, internal_acc_, internal_jerk_);

            MatrixType coeffs(n * 8, DIM);

            for (int i = 0; i < n; ++i)
            {
                const auto &tp = time_powers_[i];

                const RowVectorType c0 = P.row(i);
                const RowVectorType c1 = internal_vel_.row(i);
                const RowVectorType c2 = internal_acc_.row(i) * 0.5;
                const RowVectorType c3 = internal_jerk_.row(i) / 6.0;

                const RowVectorType &P_curr = P.row(i);
                const RowVectorType &P_next = P.row(i + 1);
                const RowVectorType &V_curr = internal_vel_.row(i);
                const RowVectorType &V_next = internal_vel_.row(i + 1);
                const RowVectorType &A_curr = internal_acc_.row(i);
                const RowVectorType &A_next = internal_acc_.row(i + 1);
                const RowVectorType &J_curr = internal_jerk_.row(i);
                const RowVectorType &J_next = internal_jerk_.row(i + 1);

                const RowVectorType c4 = -(210.0 * (P_curr - P_next) * tp.h4_inv +
                                           120.0 * V_curr * tp.h3_inv +
                                           90.0 * V_next * tp.h3_inv +
                                           30.0 * A_curr * tp.h2_inv -
                                           15.0 * A_next * tp.h2_inv +
                                           4.0 * J_curr * tp.h_inv +
                                           J_next * tp.h_inv) /
                                         6.0;

                const RowVectorType c5 = (168.0 * (P_curr - P_next) * tp.h5_inv +
                                          90.0 * V_curr * tp.h4_inv +
                                          78.0 * V_next * tp.h4_inv +
                                          20.0 * A_curr * tp.h3_inv -
                                          14.0 * A_next * tp.h3_inv +
                                          2.0 * J_curr * tp.h2_inv +
                                          J_next * tp.h2_inv) /
                                         2.0;

                const RowVectorType c6 = -(420.0 * (P_curr - P_next) * tp.h6_inv +
                                           216.0 * V_curr * tp.h5_inv +
                                           204.0 * V_next * tp.h5_inv +
                                           45.0 * A_curr * tp.h4_inv -
                                           39.0 * A_next * tp.h4_inv +
                                           4.0 * J_curr * tp.h3_inv +
                                           3.0 * J_next * tp.h3_inv) /
                                         6.0;

                const RowVectorType c7 = (120.0 * (P_curr - P_next) * tp.h7_inv +
                                          60.0 * V_curr * tp.h6_inv +
                                          60.0 * V_next * tp.h6_inv +
                                          12.0 * A_curr * tp.h5_inv -
                                          12.0 * A_next * tp.h5_inv +
                                          J_curr * tp.h4_inv +
                                          J_next * tp.h4_inv) /
                                         6.0;

                coeffs.row(i * 8 + 0) = c0;
                coeffs.row(i * 8 + 1) = c1;
                coeffs.row(i * 8 + 2) = c2;
                coeffs.row(i * 8 + 3) = c3;
                coeffs.row(i * 8 + 4) = c4;
                coeffs.row(i * 8 + 5) = c5;
                coeffs.row(i * 8 + 6) = c6;
                coeffs.row(i * 8 + 7) = c7;
            }

            return coeffs;
        }

        void initializePPoly()
        {
            trajectory_.update(cumulative_times_, coeffs_, 8);
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