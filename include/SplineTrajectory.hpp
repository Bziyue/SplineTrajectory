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
         * @brief Computes the partial gradient of the energy (acceleration cost) w.r.t segment durations.
         * @param gdT [Output] Vector of size num_segments.
         */
        void getEnergyPartialGradByTimes(Eigen::VectorXd &gdT) const
        {
            gdT.resize(num_segments_);

            for (int i = 0; i < num_segments_; ++i)
            {
                const double T = time_powers_[i].h;
                double T2 = T * T;

                const RowVectorType c2 = coeffs_.row(i * 4 + 2);
                const RowVectorType c3 = coeffs_.row(i * 4 + 3);

                // dE/dT = 4*c2^2 + 24*c2*c3*T + 36*c3^2*T^2
                gdT(i) = 4.0 * c2.squaredNorm() +
                         24.0 * c2.dot(c3) * T +
                         36.0 * c3.squaredNorm() * T2;
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
         * @brief Computes the partial gradient of the energy (jerk cost) w.r.t segment durations.
         * @param gdT [Output] Vector of size num_segments.
         */
        void getEnergyPartialGradByTimes(Eigen::VectorXd &gdT) const
        {
            gdT.resize(num_segments_);

            for (int i = 0; i < num_segments_; ++i)
            {
                double T = time_segments_[i];
                double T2 = T * T;
                double T3 = T2 * T;
                double T4 = T3 * T;

                const RowVectorType c3 = coeffs_.row(i * 6 + 3);
                const RowVectorType c4 = coeffs_.row(i * 6 + 4);
                const RowVectorType c5 = coeffs_.row(i * 6 + 5);

                // dE/dT
                gdT(i) = 36.0 * c3.squaredNorm() +
                         288.0 * c3.dot(c4) * T +
                         576.0 * c4.squaredNorm() * T2 +
                         720.0 * c3.dot(c5) * T2 +
                         2880.0 * c4.dot(c5) * T3 +
                         3600.0 * c5.squaredNorm() * T4;
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
         * @brief Propagates gradients from polynomial coefficients to waypoints and time segments.
         *
         * This function computes gradients for ALL waypoints, including the start and end points.
         *
         * @param gradByPoints [Output] Matrix of size (num_segments + 1) x DIM.
         * Contains gradients for P_0, P_1, ..., P_N.
         * @param gradByTimes  [Output] Vector of size num_segments. Gradients w.r.t duration of each segment.
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
            const int n_pts = static_cast<int>(spatial_points_.size());

            gradByPoints = MatrixType::Zero(n_pts, DIM);
            gradByTimes = partialGradByTimes;

            MatrixType g_x = MatrixType::Zero(n_pts * 2, DIM);

            for (int i = 0; i < n; ++i)
            {
                const auto &tp = time_powers_[i];
                double hi = tp.h;

                const RowVectorType gc0 = partialGradByCoeffs.row(i * 6 + 0);
                const RowVectorType gc1 = partialGradByCoeffs.row(i * 6 + 1);
                const RowVectorType gc2 = partialGradByCoeffs.row(i * 6 + 2);
                const RowVectorType gc3 = partialGradByCoeffs.row(i * 6 + 3);
                const RowVectorType gc4 = partialGradByCoeffs.row(i * 6 + 4);
                const RowVectorType gc5 = partialGradByCoeffs.row(i * 6 + 5);

                const RowVectorType P_curr = spatial_points_[i].transpose();
                const RowVectorType P_next = spatial_points_[i + 1].transpose();
                const RowVectorType v_curr = internal_vel_.row(i);
                const RowVectorType a_curr = internal_acc_.row(i);

                const RowVectorType R1 = P_next - P_curr - v_curr * hi - 0.5 * a_curr * hi * hi;
                const RowVectorType R2 = internal_vel_.row(i + 1) - v_curr - a_curr * hi;
                const RowVectorType R3 = internal_acc_.row(i + 1) - a_curr;

                gradByPoints.row(i) += gc0;
                RowVectorType dJ_dR1 = gc3 * (10.0 * tp.h3_inv) + gc4 * (-15.0 * tp.h4_inv) + gc5 * (6.0 * tp.h5_inv);
                gradByPoints.row(i + 1) += dJ_dR1;
                gradByPoints.row(i) -= dJ_dR1;

                g_x.row(2 * i) += gc1;
                g_x.row(2 * i + 1) += 0.5 * gc2;

                RowVectorType dJ_dR2 = gc3 * (-4.0 * tp.h2_inv) + gc4 * (7.0 * tp.h3_inv) + gc5 * (-3.0 * tp.h4_inv);
                RowVectorType dJ_dR3 = gc3 * (0.5 * tp.h_inv) + gc4 * (-1.0 * tp.h2_inv) + gc5 * (0.5 * tp.h3_inv);

                g_x.row(2 * i) += dJ_dR1 * (-hi);
                g_x.row(2 * i + 1) += dJ_dR1 * (-0.5 * hi * hi);

                g_x.row(2 * (i + 1)) += dJ_dR2;
                g_x.row(2 * i) -= dJ_dR2;
                g_x.row(2 * i + 1) += dJ_dR2 * (-hi);

                g_x.row(2 * (i + 1) + 1) += dJ_dR3;
                g_x.row(2 * i + 1) -= dJ_dR3;

                RowVectorType dc3_dh = -30.0 * tp.h4_inv * R1 + 8.0 * tp.h3_inv * R2 - 0.5 * tp.h2_inv * R3 - 10.0 * tp.h3_inv * v_curr - 6.0 * tp.h2_inv * a_curr;
                RowVectorType dc4_dh = 60.0 * tp.h5_inv * R1 - 21.0 * tp.h4_inv * R2 + 2.0 * tp.h3_inv * R3 + 15.0 * tp.h4_inv * v_curr + 8.0 * tp.h3_inv * a_curr;
                RowVectorType dc5_dh = -30.0 * tp.h6_inv * R1 + 12.0 * tp.h5_inv * R2 - 1.5 * tp.h4_inv * R3 - 6.0 * tp.h5_inv * v_curr - 3.0 * tp.h4_inv * a_curr;

                gradByTimes(i) += gc3.dot(dc3_dh) + gc4.dot(dc4_dh) + gc5.dot(dc5_dh);
            }

            const int num_blocks = n - 1;
            if (num_blocks <= 0)
                return;

            ws_d_rhs_mod_.resize(num_blocks);
            ws_current_lambda_.resize(num_blocks);

            for (auto &m : ws_d_rhs_mod_)
                m.setZero();

            for (int i = 0; i < num_blocks; ++i)
            {
                ws_current_lambda_[i].row(0) = g_x.row(2 * (i + 1));
                ws_current_lambda_[i].row(1) = g_x.row(2 * (i + 1) + 1);
            }

            for (int i = 0; i < num_blocks - 1; ++i)
            {
                Eigen::Matrix<double, 2, DIM> term = D_inv_cache_[i].transpose() * ws_current_lambda_[i];
                ws_d_rhs_mod_[i] += term;
                ws_current_lambda_[i + 1] -= U_blocks_cache_[i].transpose() * term;
            }
            ws_d_rhs_mod_[num_blocks - 1] += D_inv_cache_.back().transpose() * ws_current_lambda_.back();

            for (int i = num_blocks - 1; i > 0; --i)
            {
                const Eigen::Matrix2d &L = L_blocks_cache_[i];

                Eigen::Matrix<double, 2, DIM> term = D_inv_cache_[i - 1].transpose() * (L.transpose() * ws_d_rhs_mod_[i]);
                ws_d_rhs_mod_[i - 1] -= term;
            }

            for (int i = 0; i < num_blocks; ++i)
            {
                const int k = i + 2;
                const Eigen::Matrix<double, 1, DIM> lam_v = ws_d_rhs_mod_[i].row(0);
                const Eigen::Matrix<double, 1, DIM> lam_a = ws_d_rhs_mod_[i].row(1);

                const auto &tp_L = time_powers_[k - 2];
                const auto &tp_R = time_powers_[k - 1];

                gradByPoints.row(k) += lam_v * (60.0 * tp_R.h3_inv) + lam_a * (-360.0 * tp_R.h4_inv);
                gradByPoints.row(k - 1) += lam_v * (-60.0 * (tp_R.h3_inv + tp_L.h3_inv)) + lam_a * (360.0 * (tp_R.h4_inv - tp_L.h4_inv));
                gradByPoints.row(k - 2) += lam_v * (60.0 * tp_L.h3_inv) + lam_a * (360.0 * tp_L.h4_inv);

                const RowVectorType dP_R = spatial_points_[k].transpose() - spatial_points_[k - 1].transpose();
                const RowVectorType dP_L = spatial_points_[k - 1].transpose() - spatial_points_[k - 2].transpose();

                const RowVectorType v_prev = internal_vel_.row(k - 1);
                const RowVectorType a_prev = internal_acc_.row(k - 1);
                const RowVectorType v_curr = internal_vel_.row(k);
                const RowVectorType a_curr = internal_acc_.row(k);
                const RowVectorType v_next = internal_vel_.row(k + 1);
                const RowVectorType a_next = internal_acc_.row(k + 1);

                double term_rhs_hR = lam_v.dot(dP_R * (-180.0 * tp_R.h4_inv)) +
                                     lam_a.dot(dP_R * (1440.0 * tp_R.h5_inv));
                double dD_R_row0 = -72.0 * tp_R.h3_inv;
                double dD_R_row0_a = -9.0 * tp_R.h2_inv;
                double dD_R_row1 = 576.0 * tp_R.h4_inv;
                double dD_R_row1_a = 72.0 * tp_R.h3_inv;

                double term_LHS_D_hR = lam_v.dot(v_curr * dD_R_row0 + a_curr * dD_R_row0_a) +
                                       lam_a.dot(v_curr * dD_R_row1 + a_curr * dD_R_row1_a);

                double dU_row0 = -48.0 * tp_R.h3_inv;
                double dU_row0_a = 3.0 * tp_R.h2_inv;
                double dU_row1 = 504.0 * tp_R.h4_inv;
                double dU_row1_a = -48.0 * tp_R.h3_inv;

                double term_LHS_U_hR = lam_v.dot(v_next * dU_row0 + a_next * dU_row0_a) +
                                       lam_a.dot(v_next * dU_row1 + a_next * dU_row1_a);

                gradByTimes(k - 1) += term_rhs_hR - (term_LHS_D_hR + term_LHS_U_hR);

                double term_rhs_hL = lam_v.dot(dP_L * (180.0 * tp_L.h4_inv)) +
                                     lam_a.dot(dP_L * (1440.0 * tp_L.h5_inv));

                double dL_row0 = 48.0 * tp_L.h3_inv;
                double dL_row0_a = 3.0 * tp_L.h2_inv;
                double dL_row1 = 504.0 * tp_L.h4_inv;
                double dL_row1_a = 48.0 * tp_L.h3_inv;

                double term_LHS_L_hL = lam_v.dot(v_prev * dL_row0 + a_prev * dL_row0_a) +
                                       lam_a.dot(v_prev * dL_row1 + a_prev * dL_row1_a);

                double dD_L_row0 = 72.0 * tp_L.h3_inv;
                double dD_L_row0_a = -9.0 * tp_L.h2_inv;
                double dD_L_row1 = 576.0 * tp_L.h4_inv;
                double dD_L_row1_a = -72.0 * tp_L.h3_inv;

                double term_LHS_D_hL = lam_v.dot(v_curr * dD_L_row0 + a_curr * dD_L_row0_a) +
                                       lam_a.dot(v_curr * dD_L_row1 + a_curr * dD_L_row1_a);

                gradByTimes(k - 2) += term_rhs_hL - (term_LHS_L_hL + term_LHS_D_hL);
            }

            if (!includeEndpoints && n > 1)
            {
                gradByPoints = gradByPoints.middleRows(1, n - 1).eval();
            }
        }
        /**
         * @brief Propagates gradients from polynomial coefficients to waypoints and time segments.
         *
         * @param includeEndpoints If false (default), only returns gradients for inner waypoints (rows 1 to N-1),
         * excluding the start and end points, which is consistent with MINCO and other trajectory optimization libraries.
         * If true, returns gradients for ALL waypoints including start and end points.
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

        void solveInternalDerivatives(const MatrixType &P,
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
                r.row(0) = r3;
                r.row(1) = r4;

                Eigen::Matrix2d D;
                D << -36.0 * tp_L.h2_inv + 36.0 * tp_R.h2_inv, 9.0 * (tp_L.h_inv + tp_R.h_inv),
                    -192.0 * (tp_L.h3_inv + tp_R.h3_inv), 36.0 * (tp_L.h2_inv - tp_R.h2_inv);

                Eigen::Matrix2d L;
                L << -24.0 * tp_L.h2_inv, -3.0 * tp_L.h_inv,
                    -168.0 * tp_L.h3_inv, -24.0 * tp_L.h2_inv;
                L_blocks_cache_[i] = L;

                Eigen::Matrix2d U;
                if (k < n - 1)
                {
                    U << 24.0 * tp_R.h2_inv, -3.0 * tp_R.h_inv,
                        -168.0 * tp_R.h3_inv, 24.0 * tp_R.h2_inv;
                    U_blocks_cache_[i] = U;
                }

                if (i == 0) // k == 2
                {
                    r.noalias() -= L * B_left;
                }
                else
                {
                    const Eigen::Matrix2d X = D_inv_cache_[i - 1] * U_blocks_cache_[i - 1];
                    const Eigen::Matrix<double, 2, DIM> Y = D_inv_cache_[i - 1] * ws_rhs_mod_[i - 1];

                    D.noalias() -= L * X;
                    r.noalias() -= L * Y;
                }

                if (k == n - 1)
                {
                    Eigen::Matrix2d U_last;
                    U_last << 24.0 * tp_R.h2_inv, -3.0 * tp_R.h_inv,
                        -168.0 * tp_R.h3_inv, 24.0 * tp_R.h2_inv;
                    r.noalias() -= U_last * B_right;
                }

                Inverse2x2(D, D_inv_cache_[i]);
                ws_rhs_mod_[i] = r;
            }

            ws_solution_.resize(num_blocks);

            ws_solution_[num_blocks - 1] = D_inv_cache_[num_blocks - 1] * ws_rhs_mod_[num_blocks - 1];

            for (int i = num_blocks - 2; i >= 0; --i)
            {
                const Eigen::Matrix<double, 2, DIM> rhs_temp = ws_rhs_mod_[i] - U_blocks_cache_[i] * ws_solution_[i + 1];
                ws_solution_[i] = D_inv_cache_[i] * rhs_temp;
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
         * @brief Computes the partial gradient of the energy (snap cost) w.r.t segment durations.
         * @param gdT [Output] Vector of size num_segments.
         */
        void getEnergyPartialGradByTimes(Eigen::VectorXd &gdT) const
        {
            gdT.resize(num_segments_);

            for (int i = 0; i < num_segments_; ++i)
            {
                double T = time_segments_[i];
                double T2 = T * T;
                double T3 = T2 * T;
                double T4 = T3 * T;
                double T5 = T4 * T;
                double T6 = T5 * T;

                const RowVectorType c4 = coeffs_.row(i * 8 + 4);
                const RowVectorType c5 = coeffs_.row(i * 8 + 5);
                const RowVectorType c6 = coeffs_.row(i * 8 + 6);
                const RowVectorType c7 = coeffs_.row(i * 8 + 7);

                // dE/dT
                gdT(i) = 576.0 * c4.squaredNorm() +
                         5760.0 * c4.dot(c5) * T +
                         14400.0 * c5.squaredNorm() * T2 +
                         17280.0 * c4.dot(c6) * T2 +
                         86400.0 * c5.dot(c6) * T3 +
                         40320.0 * c4.dot(c7) * T3 +
                         129600.0 * c6.squaredNorm() * T4 +
                         201600.0 * c5.dot(c7) * T4 +
                         604800.0 * c6.dot(c7) * T5 +
                         705600.0 * c7.squaredNorm() * T6;
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
         * @brief Propagates gradients from polynomial coefficients to waypoints and time segments.
         *
         * This function computes gradients for ALL waypoints, including the start and end points.
         *
         * @param gradByPoints [Output] Matrix of size (num_segments + 1) x DIM.
         * Contains gradients for P_0, P_1, ..., P_N.
         * @param gradByTimes  [Output] Vector of size num_segments. Gradients w.r.t duration of each segment.
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
            const int n_pts = static_cast<int>(spatial_points_.size());

            gradByPoints = MatrixType::Zero(n_pts, DIM);
            gradByTimes = partialGradByTimes;

            MatrixType g_x = MatrixType::Zero(n_pts * 3, DIM);

            for (int i = 0; i < n; ++i)
            {
                const auto &tp = time_powers_[i];

                const RowVectorType gc0 = partialGradByCoeffs.row(i * 8 + 0);
                const RowVectorType gc1 = partialGradByCoeffs.row(i * 8 + 1);
                const RowVectorType gc2 = partialGradByCoeffs.row(i * 8 + 2);
                const RowVectorType gc3 = partialGradByCoeffs.row(i * 8 + 3);
                const RowVectorType gc4 = partialGradByCoeffs.row(i * 8 + 4);
                const RowVectorType gc5 = partialGradByCoeffs.row(i * 8 + 5);
                const RowVectorType gc6 = partialGradByCoeffs.row(i * 8 + 6);
                const RowVectorType gc7 = partialGradByCoeffs.row(i * 8 + 7);

                gradByPoints.row(i) += gc0;
                g_x.row(3 * i + 0) += gc1;
                g_x.row(3 * i + 1) += 0.5 * gc2;
                g_x.row(3 * i + 2) += (1.0 / 6.0) * gc3;

                gradByPoints.row(i) += gc4 * (-35.0 * tp.h4_inv) + gc5 * (84.0 * tp.h5_inv) +
                                       gc6 * (-70.0 * tp.h6_inv) + gc7 * (20.0 * tp.h7_inv);

                gradByPoints.row(i + 1) += gc4 * (35.0 * tp.h4_inv) + gc5 * (-84.0 * tp.h5_inv) +
                                           gc6 * (70.0 * tp.h6_inv) + gc7 * (-20.0 * tp.h7_inv);

                g_x.row(3 * i) += gc4 * (-20.0 * tp.h3_inv) + gc5 * (45.0 * tp.h4_inv) +
                                  gc6 * (-36.0 * tp.h5_inv) + gc7 * (10.0 * tp.h6_inv);
                g_x.row(3 * (i + 1)) += gc4 * (-15.0 * tp.h3_inv) + gc5 * (39.0 * tp.h4_inv) +
                                        gc6 * (-34.0 * tp.h5_inv) + gc7 * (10.0 * tp.h6_inv);

                g_x.row(3 * i + 1) += gc4 * (-5.0 * tp.h2_inv) + gc5 * (10.0 * tp.h3_inv) +
                                      gc6 * (-7.5 * tp.h4_inv) + gc7 * (2.0 * tp.h5_inv);

                g_x.row(3 * (i + 1) + 1) += gc4 * (2.5 * tp.h2_inv) + gc5 * (-7.0 * tp.h3_inv) +
                                            gc6 * (6.5 * tp.h4_inv) + gc7 * (-2.0 * tp.h5_inv);

                g_x.row(3 * i + 2) += gc4 * (-2.0 / 3.0 * tp.h_inv) + gc5 * (tp.h2_inv) +
                                      gc6 * (-2.0 / 3.0 * tp.h3_inv) + gc7 * (1.0 / 6.0 * tp.h4_inv);

                g_x.row(3 * (i + 1) + 2) += gc4 * (-1.0 / 6.0 * tp.h_inv) + gc5 * (0.5 * tp.h2_inv) +
                                            gc6 * (-0.5 * tp.h3_inv) + gc7 * (1.0 / 6.0 * tp.h4_inv);

                RowVectorType dc4 = (60.0 * internal_acc_.row(i) * tp.h3_inv - 30.0 * internal_acc_.row(i + 1) * tp.h3_inv +
                                     4.0 * internal_jerk_.row(i) * tp.h2_inv + internal_jerk_.row(i + 1) * tp.h2_inv +
                                     840.0 * spatial_points_[i].transpose() * tp.h5_inv - 840.0 * spatial_points_[i + 1].transpose() * tp.h5_inv +
                                     360.0 * internal_vel_.row(i) * tp.h4_inv + 270.0 * internal_vel_.row(i + 1) * tp.h4_inv) *
                                    (1.0 / 6.0);

                RowVectorType dc5 = (-30.0 * internal_acc_.row(i) * tp.h3_inv + 21.0 * internal_acc_.row(i + 1) * tp.h3_inv -
                                     2.0 * internal_jerk_.row(i) * tp.h2_inv - internal_jerk_.row(i + 1) * tp.h2_inv -
                                     420.0 * spatial_points_[i].transpose() * tp.h5_inv + 420.0 * spatial_points_[i + 1].transpose() * tp.h5_inv -
                                     180.0 * internal_vel_.row(i) * tp.h4_inv - 156.0 * internal_vel_.row(i + 1) * tp.h4_inv) *
                                    tp.h_inv;

                RowVectorType dc6 = (60.0 * internal_acc_.row(i) * tp.h3_inv - 52.0 * internal_acc_.row(i + 1) * tp.h3_inv +
                                     4.0 * internal_jerk_.row(i) * tp.h2_inv + 3.0 * internal_jerk_.row(i + 1) * tp.h2_inv +
                                     840.0 * spatial_points_[i].transpose() * tp.h5_inv - 840.0 * spatial_points_[i + 1].transpose() * tp.h5_inv +
                                     360.0 * internal_vel_.row(i) * tp.h4_inv + 340.0 * internal_vel_.row(i + 1) * tp.h4_inv) *
                                    (0.5 * tp.h_inv * tp.h_inv);

                RowVectorType dc7 = 2.0 * (-15.0 * internal_acc_.row(i) * tp.h3_inv + 15.0 * internal_acc_.row(i + 1) * tp.h3_inv - internal_jerk_.row(i) * tp.h2_inv - internal_jerk_.row(i + 1) * tp.h2_inv - 210.0 * spatial_points_[i].transpose() * tp.h5_inv + 210.0 * spatial_points_[i + 1].transpose() * tp.h5_inv - 90.0 * internal_vel_.row(i) * tp.h4_inv - 90.0 * internal_vel_.row(i + 1) * tp.h4_inv) *
                                    (1.0 / 3.0 * tp.h3_inv);

                gradByTimes(i) += gc4.dot(dc4) + gc5.dot(dc5) + gc6.dot(dc6) + gc7.dot(dc7);
            }

            const int num_blocks = n - 1;
            if (num_blocks > 0)
            {
                ws_d_rhs_mod_.resize(num_blocks);
                ws_current_lambda_.resize(num_blocks);

                for (auto &m : ws_d_rhs_mod_)
                    m.setZero();

                for (int i = 0; i < num_blocks; ++i)
                {
                    ws_current_lambda_[i].row(0) = g_x.row(3 * (i + 1) + 0);
                    ws_current_lambda_[i].row(1) = g_x.row(3 * (i + 1) + 1);
                    ws_current_lambda_[i].row(2) = g_x.row(3 * (i + 1) + 2);
                }

                for (int i = 0; i < num_blocks - 1; ++i)
                {
                    Eigen::Matrix<double, 3, DIM> term = D_inv_cache_[i].transpose() * ws_current_lambda_[i];
                    ws_d_rhs_mod_[i] += term;
                    ws_current_lambda_[i + 1] -= U_blocks_cache_[i].transpose() * term;
                }
                ws_d_rhs_mod_[num_blocks - 1] += D_inv_cache_.back().transpose() * ws_current_lambda_.back();

                for (int i = num_blocks - 1; i > 0; --i)
                {
                    const Eigen::Matrix3d &L = L_blocks_cache_[i];
                    Eigen::Matrix<double, 3, DIM> term = D_inv_cache_[i - 1].transpose() * (L.transpose() * ws_d_rhs_mod_[i]);
                    ws_d_rhs_mod_[i - 1] -= term;
                }

                for (int i = 0; i < num_blocks; ++i)
                {
                    const int node_curr = i + 1;
                    const int node_prev = i;
                    const int node_next = i + 2;

                    const Eigen::Matrix<double, 1, DIM> lam_snap = ws_d_rhs_mod_[i].row(0);
                    const Eigen::Matrix<double, 1, DIM> lam_crackle = ws_d_rhs_mod_[i].row(1);
                    const Eigen::Matrix<double, 1, DIM> lam_pop = ws_d_rhs_mod_[i].row(2);

                    const auto &tp_L = time_powers_[i];
                    const auto &tp_R = time_powers_[i + 1];

                    double dRes0_dP_curr = 840.0 * tp_R.h4_inv - 840.0 * tp_L.h4_inv;
                    double dRes1_dP_curr = -10080.0 * tp_R.h5_inv - 10080.0 * tp_L.h5_inv;
                    double dRes2_dP_curr = 50400.0 * tp_R.h6_inv - 50400.0 * tp_L.h6_inv;

                    gradByPoints.row(node_curr) -= lam_snap * dRes0_dP_curr +
                                                   lam_crackle * dRes1_dP_curr +
                                                   lam_pop * dRes2_dP_curr;

                    double dRes0_dP_prev = 840.0 * tp_L.h4_inv;
                    double dRes1_dP_prev = 10080.0 * tp_L.h5_inv;
                    double dRes2_dP_prev = 50400.0 * tp_L.h6_inv;

                    gradByPoints.row(node_prev) -= lam_snap * dRes0_dP_prev +
                                                   lam_crackle * dRes1_dP_prev +
                                                   lam_pop * dRes2_dP_prev;

                    double dRes0_dP_next = -840.0 * tp_R.h4_inv;
                    double dRes1_dP_next = 10080.0 * tp_R.h5_inv;
                    double dRes2_dP_next = -50400.0 * tp_R.h6_inv;

                    gradByPoints.row(node_next) -= lam_snap * dRes0_dP_next +
                                                   lam_crackle * dRes1_dP_next +
                                                   lam_pop * dRes2_dP_next;

                    RowVectorType P_curr = spatial_points_[node_curr].transpose();
                    RowVectorType P_prev = spatial_points_[node_prev].transpose();
                    RowVectorType P_next = spatial_points_[node_next].transpose();

                    RowVectorType V_curr = internal_vel_.row(node_curr);
                    RowVectorType V_prev = internal_vel_.row(node_prev);
                    RowVectorType V_next = internal_vel_.row(node_next);

                    RowVectorType A_curr = internal_acc_.row(node_curr);
                    RowVectorType A_prev = internal_acc_.row(node_prev);
                    RowVectorType A_next = internal_acc_.row(node_next);

                    RowVectorType J_curr = internal_jerk_.row(node_curr);
                    RowVectorType J_prev = internal_jerk_.row(node_prev);
                    RowVectorType J_next = internal_jerk_.row(node_next);

                    RowVectorType term_dRes0_dhL = 4.0 * (60.0 * A_curr * tp_L.h3_inv - 30.0 * A_prev * tp_L.h3_inv -
                                                          4.0 * J_curr * tp_L.h2_inv - J_prev * tp_L.h2_inv +
                                                          840.0 * P_curr * tp_L.h5_inv - 840.0 * P_prev * tp_L.h5_inv -
                                                          360.0 * V_curr * tp_L.h4_inv - 270.0 * V_prev * tp_L.h4_inv);

                    RowVectorType term_dRes1_dhL = 120.0 * (30.0 * A_curr * tp_L.h4_inv - 21.0 * A_prev * tp_L.h4_inv -
                                                            2.0 * J_curr * tp_L.h3_inv - J_prev * tp_L.h3_inv +
                                                            420.0 * P_curr * tp_L.h6_inv - 420.0 * P_prev * tp_L.h6_inv -
                                                            180.0 * V_curr * tp_L.h5_inv - 156.0 * V_prev * tp_L.h5_inv);

                    RowVectorType term_dRes2_dhL = 360.0 * (60.0 * A_curr * tp_L.h5_inv - 52.0 * A_prev * tp_L.h5_inv -
                                                            4.0 * J_curr * tp_L.h4_inv - 3.0 * J_prev * tp_L.h4_inv +
                                                            840.0 * P_curr * tp_L.h7_inv - 840.0 * P_prev * tp_L.h7_inv -
                                                            360.0 * V_curr * tp_L.h6_inv - 340.0 * V_prev * tp_L.h6_inv);

                    gradByTimes(i) -= lam_snap.dot(term_dRes0_dhL) +
                                      lam_crackle.dot(term_dRes1_dhL) +
                                      lam_pop.dot(term_dRes2_dhL);

                    RowVectorType term_dRes0_dhR = 4.0 * (-60.0 * A_curr * tp_R.h3_inv + 30.0 * A_next * tp_R.h3_inv -
                                                          4.0 * J_curr * tp_R.h2_inv - J_next * tp_R.h2_inv -
                                                          840.0 * P_curr * tp_R.h5_inv + 840.0 * P_next * tp_R.h5_inv -
                                                          360.0 * V_curr * tp_R.h4_inv - 270.0 * V_next * tp_R.h4_inv);

                    RowVectorType term_dRes1_dhR = 120.0 * (30.0 * A_curr * tp_R.h4_inv - 21.0 * A_next * tp_R.h4_inv +
                                                            2.0 * J_curr * tp_R.h3_inv + J_next * tp_R.h3_inv +
                                                            420.0 * P_curr * tp_R.h6_inv - 420.0 * P_next * tp_R.h6_inv +
                                                            180.0 * V_curr * tp_R.h5_inv + 156.0 * V_next * tp_R.h5_inv);

                    RowVectorType term_dRes2_dhR = 360.0 * (-60.0 * A_curr * tp_R.h5_inv + 52.0 * A_next * tp_R.h5_inv -
                                                            4.0 * J_curr * tp_R.h4_inv - 3.0 * J_next * tp_R.h4_inv -
                                                            840.0 * P_curr * tp_R.h7_inv + 840.0 * P_next * tp_R.h7_inv -
                                                            360.0 * V_curr * tp_R.h6_inv - 340.0 * V_next * tp_R.h6_inv);

                    gradByTimes(i + 1) -= lam_snap.dot(term_dRes0_dhR) +
                                          lam_crackle.dot(term_dRes1_dhR) +
                                          lam_pop.dot(term_dRes2_dhR);
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
         * @brief Propagates gradients from polynomial coefficients to waypoints and time segments.
         *
         * This function computes gradients for ALL waypoints, including the start and end points.
         *
         * @param gradByPoints [Output] Matrix of size (num_segments + 1) x DIM.
         * Contains gradients for P_0, P_1, ..., P_N.
         * @param gradByTimes  [Output] Vector of size num_segments. Gradients w.r.t duration of each segment.
         * @param includeEndpoints If false (default), only returns gradients for inner waypoints (rows 1 to N-1),
         * excluding the start and end points, which is consistent with MINCO and other trajectory optimization libraries.
         * If true, returns gradients for ALL waypoints including start and end points.
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

                    const Eigen::Matrix3d X = D_prev_inv * U_prev;
                    const Eigen::Matrix<double, 3, DIM> Y = D_prev_inv * r_prev;

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

            ws_solution_[num_blocks - 1] = D_inv_cache_[num_blocks - 1] * ws_rhs_mod_[num_blocks - 1];

            for (int i = num_blocks - 2; i >= 0; --i)
            {
                const Eigen::Matrix<double, 3, DIM> rhs_temp = ws_rhs_mod_[i] - U_blocks_cache_[i] * ws_solution_[i + 1];
                ws_solution_[i] = D_inv_cache_[i] * rhs_temp;
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