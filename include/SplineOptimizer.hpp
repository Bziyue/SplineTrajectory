#ifndef SPLINE_OPTIMIZER_HPP
#define SPLINE_OPTIMIZER_HPP

#include "SplineTrajectory.hpp"
#include <algorithm>
#include <vector>
#include <cmath>
#include <iostream>
#include <iomanip>
#include <memory>
#include <concepts>
#include <type_traits> 
#include <utility>   
#include <sstream>  

namespace SplineTrajectory
{
    namespace Concepts
    {
        // Time map protocol:
        // toTime(tau), toTau(time_value), backward(tau, time_value, gradT)
        template <typename MapT>
        concept TimeMapLike = requires(const MapT &map,
                                       double tau,
                                       double time_value,
                                       double gradT)
        {
            { map.toTime(tau) } -> std::convertible_to<double>;
            { map.toTau(time_value) } -> std::convertible_to<double>;
            { map.backward(tau, time_value, gradT) } -> std::convertible_to<double>;
        };

        // Spatial map protocol:
        // getUnconstrainedDim(index), toPhysical(xi, index),
        // toUnconstrained(p, index), backwardGrad(xi, grad_p, index)
        template <typename T, int DIM>
        concept SpatialMapLike = requires(const T &map,
                                          const Eigen::VectorXd &xi,
                                          const Eigen::VectorXd &grad_p,
                                          const Eigen::Matrix<double, DIM, 1> &p,
                                          int index)
        {
            { map.getUnconstrainedDim(index) } -> std::convertible_to<int>;
            { map.toPhysical(xi, index) } -> std::convertible_to<Eigen::VectorXd>;
            { map.toUnconstrained(p, index) } -> std::convertible_to<Eigen::VectorXd>;
            { map.backwardGrad(xi, grad_p, index) } -> std::convertible_to<Eigen::VectorXd>;
        };

        // Executor protocol:
        // operator()(int start, int end, Func f)
        template <typename T>
        concept ExecutorLike = requires(const T &executor)
        {
            executor(0, 1, [](int) {});
        };

        // Time cost protocol:
        // operator()(const vector<double>& Ts, VectorXd& grad) -> double
        template <typename T>
        concept TimeCostLike = requires(T &&f, const std::vector<double> &Ts, Eigen::VectorXd &grad)
        {
            { std::forward<T>(f)(Ts, grad) } -> std::convertible_to<double>;
        };

        // Waypoints cost protocol:
        // operator()(const WaypointsType&, MatrixXd& grad_q) -> double
        template <typename T, typename WaypointsType>
        concept WaypointsCostLike = requires(T &&f,
                                             const WaypointsType &waypoints,
                                             Eigen::Matrix<double, -1, -1> &grad_q)
        {
            { std::forward<T>(f)(waypoints, grad_q) } -> std::convertible_to<double>;
        };

        // Integral cost protocol:
        // operator()(t, t_global, segment_index, p, v, a, j, s, gp, gv, ga, gj, gs, gt) -> double
        template <typename T, typename VecT>
        concept IntegralCostLike = requires(T &&f,
                                            double t,
                                            double t_global,
                                            int segment_index,
                                            const VecT &p,
                                            const VecT &v,
                                            const VecT &a,
                                            const VecT &j,
                                            const VecT &s,
                                            VecT &gp,
                                            VecT &gv,
                                            VecT &ga,
                                            VecT &gj,
                                            VecT &gs,
                                            double &gt)
        {
            { std::forward<T>(f)(t, t_global, segment_index, p, v, a, j, s, gp, gv, ga, gj, gs, gt) } -> std::convertible_to<double>;
        };
    } // namespace Concepts

    /**
     * @brief SerialExecutor
     * Runs the loop sequentially on the current thread.
     * Default executor, zero overhead, no dependencies.
     */
    struct SerialExecutor
    {
        template <typename Func>
        void operator()(int start, int end, Func &&f) const
        {
            for (int i = start; i < end; ++i)
            {
                f(i);
            }
        }
    };

    /**
     * @brief OpenMPExecutor
     * Runs the loop in parallel using OpenMP.
     * Requires compilation with -fopenmp. 
     * If compiled without OpenMP, falls back to serial execution automatically.
     */
    struct OpenMPExecutor
    {
        template <typename Func>
        void operator()(int start, int end, Func &&f) const
        {
#if defined(_OPENMP)
            #pragma omp parallel for schedule(static)
            for (int i = start; i < end; ++i)
            {
                f(i);
            }
#else
            // Fallback if OpenMP is not available
            for (int i = start; i < end; ++i)
            {
                f(i);
            }
#endif
        }
    };

    struct IdentityTimeMap
    {
        double toTime(double tau) const { return tau; }
        double toTau(double T) const { return T; }
        // T = tau => dT/dtau = 1 => grad_tau = gradT * 1
        double backward(double tau, double T, double gradT) const { return gradT; }
    };

    struct QuadInvTimeMap
    {
        double toTime(double tau) const
        {
            return tau > 0
                ? ((0.5 * tau + 1.0) * tau + 1.0)
                : (1.0 / ((0.5 * tau - 1.0) * tau + 1.0));
        }

        double toTau(double T) const
        {
            return T > 1.0
                   ? (std::sqrt(2.0 * T - 1.0) - 1.0)
                   : (1.0 - std::sqrt(2.0 / T - 1.0));
        }

        double backward(double tau, double T, double gradT) const
        {
            if (tau > 0)
            {
                return gradT * (tau + 1.0);
            }
            else
            {
                double den = (0.5 * tau - 1.0) * tau + 1.0;
                return gradT * (1.0 - tau) / (den * den);
            }
        }
    };

    /**
     * @brief IdentitySpatialMap
     * Default unconstrained mapping (xi = p).
     */
    template <int DIM>
    struct IdentitySpatialMap
    {
        using VectorType = Eigen::Matrix<double, DIM, 1>;

        int getUnconstrainedDim(int index) const { return DIM; }

        VectorType toPhysical(const VectorType& xi, int index) const
        {
            return xi;
        }

        VectorType toUnconstrained(const VectorType& p, int index) const
        {
            return p;
        }

        VectorType backwardGrad(const VectorType& xi, const VectorType& grad_p, int index) const
        {
            return grad_p;
        }
    };

    struct VoidWaypointsCost
    {
        template <typename WaypointsType, typename GradMatrixType>
        double operator()(const WaypointsType & /*waypoints*/, GradMatrixType & /*grad_q*/) const
        {
            return 0.0;
        }
    };

    struct OptimizationFlags
    {
        bool start_p = false;
        bool start_v = false;
        bool start_a = false;
        bool start_j = false;
        bool end_p = false;
        bool end_v = false;
        bool end_a = false;
        bool end_j = false;
    };

    template <int DIM,
              typename SplineType = QuinticSplineND<DIM>,
              typename TimeMap = QuadInvTimeMap,
              typename SpatialMap = IdentitySpatialMap<DIM>>
    requires Concepts::TimeMapLike<TimeMap> &&
             Concepts::SpatialMapLike<SpatialMap, DIM>
    class SplineOptimizer
    {
    public:
        using VectorType = typename SplineType::VectorType;
        using MatrixType = typename SplineType::MatrixType;
        using WaypointsType = MatrixType;

        /**
         * @brief Workspace holds all mutable state required during optimization.
         * This allows the Optimizer to be stateless and thread-safe.
         */
        struct Workspace
        {
            EIGEN_MAKE_ALIGNED_OPERATOR_NEW

            SplineType spline;
            std::vector<double> cache_times;
            WaypointsType cache_waypoints;

            Eigen::VectorXd cache_gdT;
            MatrixType cache_gdC;
            Eigen::VectorXd user_gdT_buffer;

            typename SplineType::Gradients grads;
            typename SplineType::Gradients energy_grads;

            Eigen::VectorXd explicit_time_grad_buffer;
            MatrixType discrete_grad_q_buffer;
            std::vector<double> segment_start_times;
            std::vector<double> segment_costs;

            void resize(int num_segments)
            {
                if (static_cast<int>(cache_times.size()) != num_segments)
                {
                    cache_times.resize(num_segments);
                    cache_waypoints.resize(num_segments + 1, DIM);
                    cache_gdT.resize(num_segments);
                    user_gdT_buffer.resize(num_segments);
                    cache_gdC.resize(num_segments * SplineType::COEFF_NUM, DIM);
                    explicit_time_grad_buffer.resize(num_segments);
                    discrete_grad_q_buffer.resize(num_segments + 1, DIM);
                    segment_start_times.resize(num_segments);
                    segment_costs.resize(num_segments);
                }
            }
        };

    private:
        struct SpatialVariableLayout
        {
            int point_index = 0;
            int offset = 0;
            int dof = 0;
        };

        std::vector<double> ref_times_;
        WaypointsType ref_waypoints_;
        BoundaryConditions<DIM> ref_bc_;
        double start_time_ = 0.0;

        OptimizationFlags flags_;
        int num_segments_ = 0;
        bool is_valid_ = false;

        double rho_energy_ = 0.0;
        int integral_num_steps_ = 64;

        TimeMap default_time_map_;
        SpatialMap default_spatial_map_;

        const TimeMap* active_time_map_ = nullptr;
        const SpatialMap* active_spatial_map_ = nullptr;

        mutable std::unique_ptr<Workspace> internal_ws_;
        
        mutable std::string last_error_message_;
        mutable std::vector<SpatialVariableLayout> spatial_layout_;
        mutable int derivatives_offset_ = 0;
        mutable int total_dimension_ = 0;
        mutable bool layout_dirty_ = true;
        
        /**
         * @brief Helper method to retrieve or create the internal workspace.
         * @return Pointer to the internal workspace.
         */
        Workspace* getOrCreateInternalWorkspace() const
        {
            if (!internal_ws_)
            {
                internal_ws_ = std::make_unique<Workspace>();
            }
            return internal_ws_.get();
        }

        void markLayoutDirty()
        {
            layout_dirty_ = true;
        }

        bool isSpatialOptimized(int idx) const
        {
            if (idx == 0)
            {
                return flags_.start_p;
            }
            if (idx == num_segments_)
            {
                return flags_.end_p;
            }
            return true;
        }

        int countOptimizedDerivativeBlocks() const
        {
            int blocks = 0;
            if (flags_.start_v) ++blocks;
            if constexpr (SplineType::ORDER >= 5)
            {
                if (flags_.start_a) ++blocks;
            }
            if constexpr (SplineType::ORDER >= 7)
            {
                if (flags_.start_j) ++blocks;
            }

            if (flags_.end_v) ++blocks;
            if constexpr (SplineType::ORDER >= 5)
            {
                if (flags_.end_a) ++blocks;
            }
            if constexpr (SplineType::ORDER >= 7)
            {
                if (flags_.end_j) ++blocks;
            }
            return blocks;
        }

        void rebuildLayoutCache() const
        {
            spatial_layout_.clear();

            if (num_segments_ <= 0)
            {
                derivatives_offset_ = 0;
                total_dimension_ = 0;
                layout_dirty_ = false;
                return;
            }

            int offset = num_segments_;
            for (int i = 0; i <= num_segments_; ++i)
            {
                if (!isSpatialOptimized(i))
                {
                    continue;
                }

                const int dof = active_spatial_map_->getUnconstrainedDim(i);
                spatial_layout_.push_back(SpatialVariableLayout{i, offset, dof});
                offset += dof;
            }

            derivatives_offset_ = offset;
            total_dimension_ = derivatives_offset_ + countOptimizedDerivativeBlocks() * DIM;
            layout_dirty_ = false;
        }

        void ensureLayoutCache() const
        {
            if (!layout_dirty_)
            {
                return;
            }
            rebuildLayoutCache();
        }
        
        static constexpr double MIN_VALID_DURATION = 1e-3; // 1 ms

    public:
        SplineOptimizer()
        {
            active_time_map_ = &default_time_map_;
            active_spatial_map_ = &default_spatial_map_;
        }

        SplineOptimizer(const SplineOptimizer &other)
            : ref_times_(other.ref_times_),
              ref_waypoints_(other.ref_waypoints_),
              ref_bc_(other.ref_bc_),
              start_time_(other.start_time_),
              flags_(other.flags_),
              num_segments_(other.num_segments_),
              is_valid_(other.is_valid_),
              rho_energy_(other.rho_energy_),
              integral_num_steps_(other.integral_num_steps_),
              default_time_map_(other.default_time_map_),
              default_spatial_map_(other.default_spatial_map_),
              spatial_layout_(other.spatial_layout_),
              derivatives_offset_(other.derivatives_offset_),
              total_dimension_(other.total_dimension_),
              layout_dirty_(other.layout_dirty_)
        {
            active_time_map_ = (other.active_time_map_ == &other.default_time_map_)
                              ? &default_time_map_
                              : other.active_time_map_;
            active_spatial_map_ = (other.active_spatial_map_ == &other.default_spatial_map_)
                                  ? &default_spatial_map_
                                  : other.active_spatial_map_;

            if (other.internal_ws_)
                internal_ws_ = std::unique_ptr<Workspace>(new Workspace(*other.internal_ws_));
        }

        SplineOptimizer &operator=(const SplineOptimizer &other)
        {
            if (this != &other)
            {
                ref_times_ = other.ref_times_;
                ref_waypoints_ = other.ref_waypoints_;
                ref_bc_ = other.ref_bc_;
                start_time_ = other.start_time_;
                flags_ = other.flags_;
                num_segments_ = other.num_segments_;
                is_valid_ = other.is_valid_;
                rho_energy_ = other.rho_energy_;
                integral_num_steps_ = other.integral_num_steps_;
                default_time_map_ = other.default_time_map_;
                default_spatial_map_ = other.default_spatial_map_;
                spatial_layout_ = other.spatial_layout_;
                derivatives_offset_ = other.derivatives_offset_;
                total_dimension_ = other.total_dimension_;
                layout_dirty_ = other.layout_dirty_;

                active_time_map_ = (other.active_time_map_ == &other.default_time_map_)
                                  ? &default_time_map_
                                  : other.active_time_map_;
                active_spatial_map_ = (other.active_spatial_map_ == &other.default_spatial_map_)
                                      ? &default_spatial_map_
                                      : other.active_spatial_map_;

                if (other.internal_ws_)
                    internal_ws_ = std::unique_ptr<Workspace>(new Workspace(*other.internal_ws_));
                else
                    internal_ws_.reset();
            }
            return *this;
        }

        /**
         * @brief Set the TimeMap to use for time transformations.
         * @param map Pointer to a TimeMap instance (can be nullptr to reset to default).
         * The optimizer does not take ownership; the map must remain valid.
         */
        void setTimeMap(const TimeMap* map)
        {
            active_time_map_ = (map != nullptr) ? map : &default_time_map_;
        }

        /**
         * @brief Set the SpatialMap to use for spatial transformations.
         * @param map Pointer to a SpatialMap instance (can be nullptr to reset to default).
         * The optimizer does not take ownership; the map must remain valid.
         */
        void setSpatialMap(const SpatialMap* map)
        {
            active_spatial_map_ = (map != nullptr) ? map : &default_spatial_map_;
            markLayoutDirty();
        }

        /**
         * @brief Initialize using Absolute Time Points.
         * Converts time points to segments.
         * @return true if the initial state is valid, false otherwise.
         */
        bool setInitState(const std::vector<double> &t_points,
                          const WaypointsType &waypoints,
                          const BoundaryConditions<DIM> &bc)
        {
            last_error_message_.clear();
            
            if (t_points.empty())
            {
                last_error_message_ = "Input time points vector is empty";
                is_valid_ = false;
                return false;
            }

            std::vector<double> time_segments;
            time_segments.reserve(t_points.size() - 1);
            for (size_t i = 1; i < t_points.size(); ++i)
            {
                time_segments.push_back(t_points[i] - t_points[i - 1]);
            }

            return setInitState(time_segments, waypoints, t_points.front(), bc);
        }

        /**
         * @brief Initialize using Time Segments (Durations).
         * @return true if the initial state is valid, false otherwise.
         */
        bool setInitState(const std::vector<double> &time_segments,
                          const WaypointsType &waypoints,
                          double start_time,
                          const BoundaryConditions<DIM> &bc)
        {
            last_error_message_.clear();
            
            start_time_ = start_time;
            ref_times_ = time_segments;
            ref_waypoints_ = waypoints;
            ref_bc_ = bc;
            num_segments_ = static_cast<int>(ref_times_.size());
            markLayoutDirty();

            is_valid_ = checkValidity();
            
            if (is_valid_) {
                if (internal_ws_)
                    internal_ws_->resize(num_segments_);
            }

            return is_valid_;
        }

        void setOptimizationFlags(const OptimizationFlags &flags)
        {
            flags_ = flags;
            markLayoutDirty();
        }
        void setEnergyWeights(double rho_energy) { rho_energy_ = rho_energy; }
        void setIntegralNumSteps(int steps) { integral_num_steps_ = steps; }

        /**
         * @brief Check if the current optimization state is valid.
         * @return true if valid, false otherwise.
         */
        bool isValid() const { return is_valid_; }

        /**
         * @brief Convertible to bool to check validity.
         */
        operator bool() const { return is_valid_; }
        
        /**
         * @brief Get the last error message from validation checks.
         * @return Error message string (empty if no error).
         */
        std::string getLastError() const { return last_error_message_; }

        /**
         * @brief Perform a thorough validity check on segments, times, waypoints and BCs.
         * Aggregates ALL errors instead of stopping at the first one.
         * @param msg_out Optional output pointer to receive detailed error message.
         * @return true if valid, false otherwise.
         */
        bool checkValidity(std::string* msg_out = nullptr) const
        {
            std::vector<std::string> errors;
            
            if (num_segments_ <= 0) {
                errors.push_back("Invalid segment count: num_segments_ <= 0");
            }
            
            if (ref_times_.size() != static_cast<size_t>(num_segments_)) {
                errors.push_back("Size mismatch: ref_times_.size() = " + std::to_string(ref_times_.size()) + 
                                 " != num_segments_ = " + std::to_string(num_segments_));
            }
            
            if (ref_waypoints_.rows() != num_segments_ + 1) {
                errors.push_back("Size mismatch: ref_waypoints_.rows() = " + std::to_string(ref_waypoints_.rows()) + 
                                 " != num_segments_ + 1 = " + std::to_string(num_segments_ + 1));
            }
            
            if (!std::isfinite(start_time_)) {
                errors.push_back("Start time is not finite (NaN or Inf)");
            }
            
            for (size_t i = 0; i < ref_times_.size(); ++i) {
                double t = ref_times_[i];
                if (!std::isfinite(t)) {
                    errors.push_back("Time segment [" + std::to_string(i) + "] is not finite: " + std::to_string(t));
                } else if (t < MIN_VALID_DURATION) {
                    errors.push_back("Time segment [" + std::to_string(i) + "] is too small: " + 
                                     std::to_string(t) + " < " + std::to_string(MIN_VALID_DURATION));
                }
            }
            
            for (int i = 0; i < ref_waypoints_.rows(); ++i) {
                if (!ref_waypoints_.row(i).array().isFinite().all()) {
                    errors.push_back("Waypoint row [" + std::to_string(i) + "] contains NaN or Inf");
                }
            }
            
            if (!ref_bc_.start_velocity.array().isFinite().all()) {
                errors.push_back("Start velocity contains NaN or Inf");
            }
            if (!ref_bc_.end_velocity.array().isFinite().all()) {
                errors.push_back("End velocity contains NaN or Inf");
            }
            
            if constexpr (SplineType::ORDER >= 5) {
                if (!ref_bc_.start_acceleration.array().isFinite().all()) {
                    errors.push_back("Start acceleration contains NaN or Inf");
                }
                if (!ref_bc_.end_acceleration.array().isFinite().all()) {
                    errors.push_back("End acceleration contains NaN or Inf");
                }
            }
            
            if constexpr (SplineType::ORDER >= 7) {
                if (!ref_bc_.start_jerk.array().isFinite().all()) {
                    errors.push_back("Start jerk contains NaN or Inf");
                }
                if (!ref_bc_.end_jerk.array().isFinite().all()) {
                    errors.push_back("End jerk contains NaN or Inf");
                }
            }
            
            if (!errors.empty()) {
                reportError(msg_out, errors);
                return false;
            }
            
            if (msg_out) {
                msg_out->clear();
            }
            
            return true;
        }

        int getDimension() const { return calculateDimension(); }

        /**
         * @brief Generate initial guess x based on reference state.
         * Applies 'toUnconstrained' mapping using unified layout logic.
         */
        Eigen::VectorXd generateInitialGuess() const
        {
            ensureLayoutCache();
            int dim = total_dimension_;
            Eigen::VectorXd x(dim);
            for (int i = 0; i < num_segments_; ++i)
            {
                x(i) = active_time_map_->toTau(ref_times_[i]);
            }

            for (const auto &var : spatial_layout_)
            {
                x.segment(var.offset, var.dof) =
                    active_spatial_map_->toUnconstrained(ref_waypoints_.row(var.point_index).transpose(), var.point_index);
            }

            BoundaryConditions<DIM> bc = ref_bc_;
            int offset = derivatives_offset_;
            auto apply_derivatives = [&](auto&& op) {
                if (flags_.start_v) op(bc.start_velocity);
                if constexpr (SplineType::ORDER >= 5) if (flags_.start_a) op(bc.start_acceleration);
                if constexpr (SplineType::ORDER >= 7) if (flags_.start_j) op(bc.start_jerk);
                
                if (flags_.end_v) op(bc.end_velocity);
                if constexpr (SplineType::ORDER >= 5) if (flags_.end_a) op(bc.end_acceleration);
                if constexpr (SplineType::ORDER >= 7) if (flags_.end_j) op(bc.end_jerk);
            };

            apply_derivatives([&](const VectorType& v) {
                x.segment<DIM>(offset) = v;
                offset += DIM;
            });

            return x;
        }

        /**
         * @brief Primary evaluate: 3 cost functions with optional Workspace and Executor.
         * @param ws Workspace pointer (default nullptr to use internal workspace).
         * @param executor Executor instance (default SerialExecutor).
         */
        template <typename TimeCostFunc, typename WaypointsCostFunc, typename IntegralCostFunc, 
                  typename Executor = SerialExecutor>
        requires Concepts::TimeCostLike<TimeCostFunc> &&
                 Concepts::WaypointsCostLike<WaypointsCostFunc, WaypointsType> &&
                 Concepts::IntegralCostLike<IntegralCostFunc, VectorType> &&
                 Concepts::ExecutorLike<Executor>
        double evaluate(const Eigen::VectorXd &x, Eigen::VectorXd &grad_out,
                        TimeCostFunc &&time_cost_func,
                        WaypointsCostFunc &&waypoints_cost_func,
                        IntegralCostFunc &&integral_cost_func,
                        Workspace *ws = nullptr,
                        const Executor& executor = Executor()) const
        {
            using WCF = std::decay_t<WaypointsCostFunc>;
            ensureLayoutCache();

            Workspace& ws_ref = (ws != nullptr) ? *ws : *getOrCreateInternalWorkspace();
            ws_ref.resize(num_segments_);
            grad_out.setZero(x.size());
            double total_cost = 0.0;
            int n_inner = std::max(0, num_segments_ - 1);

            for (int i = 0; i < num_segments_; ++i)
            {
                ws_ref.cache_times[i] = active_time_map_->toTime(x(i));
            }

            ws_ref.cache_waypoints = ref_waypoints_;
            for (const auto &var : spatial_layout_)
            {
                ws_ref.cache_waypoints.row(var.point_index) =
                    active_spatial_map_->toPhysical(x.segment(var.offset, var.dof), var.point_index).transpose();
            }

            BoundaryConditions<DIM> current_bc = ref_bc_;
            int offset = derivatives_offset_;
            auto apply_derivatives_forward = [&](auto&& op) {

                if (flags_.start_v) op(current_bc.start_velocity);
                if constexpr (SplineType::ORDER >= 5) if (flags_.start_a) op(current_bc.start_acceleration);
                if constexpr (SplineType::ORDER >= 7) if (flags_.start_j) op(current_bc.start_jerk);
                
                if (flags_.end_v) op(current_bc.end_velocity);
                if constexpr (SplineType::ORDER >= 5) if (flags_.end_a) op(current_bc.end_acceleration);
                if constexpr (SplineType::ORDER >= 7) if (flags_.end_j) op(current_bc.end_jerk);
            };

            apply_derivatives_forward([&](VectorType& target) {
                target = x.template segment<DIM>(offset);
                offset += DIM;
            });

            ws_ref.spline.update(ws_ref.cache_times, ws_ref.cache_waypoints, start_time_, current_bc);

            ws_ref.user_gdT_buffer.setZero();
            ws_ref.cache_gdT.setZero();
            total_cost += time_cost_func(ws_ref.cache_times, ws_ref.user_gdT_buffer);
            ws_ref.cache_gdT += ws_ref.user_gdT_buffer;

            ws_ref.cache_gdC.setZero();
            calculateIntegralCost(ws_ref, ws_ref.cache_gdC, ws_ref.cache_gdT, total_cost, 
                                  std::forward<IntegralCostFunc>(integral_cost_func),
                                  executor);

            ws_ref.spline.propagateGrad(ws_ref.cache_gdC, ws_ref.cache_gdT, ws_ref.grads);

            if constexpr (!std::is_same_v<WCF, VoidWaypointsCost>)
            {
                ws_ref.discrete_grad_q_buffer.setZero();

                double dw_cost = waypoints_cost_func(ws_ref.cache_waypoints, ws_ref.discrete_grad_q_buffer);
                total_cost += dw_cost;

                ws_ref.grads.start.p += ws_ref.discrete_grad_q_buffer.row(0).transpose();
                if (n_inner > 0) {
                    ws_ref.grads.inner_points += ws_ref.discrete_grad_q_buffer.block(1, 0, n_inner, DIM);
                }
                ws_ref.grads.end.p += ws_ref.discrete_grad_q_buffer.row(num_segments_).transpose();
            }

            if (rho_energy_ > 0)
            {
                double energy = ws_ref.spline.getEnergy();
                total_cost += rho_energy_ * energy;

                ws_ref.spline.getEnergyGrad(ws_ref.energy_grads);

                ws_ref.grads.times += rho_energy_ * ws_ref.energy_grads.times;
                if (n_inner > 0)
                    ws_ref.grads.inner_points += rho_energy_ * ws_ref.energy_grads.inner_points;

                ws_ref.grads.start.p += rho_energy_ * ws_ref.energy_grads.start.p;
                ws_ref.grads.start.v += rho_energy_ * ws_ref.energy_grads.start.v;
                if constexpr (SplineType::ORDER >= 5) ws_ref.grads.start.a += rho_energy_ * ws_ref.energy_grads.start.a;
                if constexpr (SplineType::ORDER >= 7) ws_ref.grads.start.j += rho_energy_ * ws_ref.energy_grads.start.j;

                ws_ref.grads.end.p += rho_energy_ * ws_ref.energy_grads.end.p;
                ws_ref.grads.end.v += rho_energy_ * ws_ref.energy_grads.end.v;
                if constexpr (SplineType::ORDER >= 5) ws_ref.grads.end.a += rho_energy_ * ws_ref.energy_grads.end.a;
                if constexpr (SplineType::ORDER >= 7) ws_ref.grads.end.j += rho_energy_ * ws_ref.energy_grads.end.j;
            }

            // Time gradient: backward through time map
            for (int i = 0; i < num_segments_; ++i)
            {
                double tau = x(i);
                double T = ws_ref.cache_times[i];
                double gradT = ws_ref.grads.times(i);
                grad_out(i) = active_time_map_->backward(tau, T, gradT);
            }

            // Spatial gradient: unified loop with backward mapping
            for (const auto &var : spatial_layout_)
            {
                const auto xi = x.segment(var.offset, var.dof);

                if (var.point_index == 0)
                {
                    grad_out.segment(var.offset, var.dof) =
                        active_spatial_map_->backwardGrad(xi, ws_ref.grads.start.p, 0);
                }
                else if (var.point_index == num_segments_)
                {
                    grad_out.segment(var.offset, var.dof) =
                        active_spatial_map_->backwardGrad(xi, ws_ref.grads.end.p, var.point_index);
                }
                else
                {
                    VectorType inner_grad = ws_ref.grads.inner_points.row(var.point_index - 1).transpose();
                    grad_out.segment(var.offset, var.dof) =
                        active_spatial_map_->backwardGrad(xi, inner_grad, var.point_index);
                }
            }

            offset = derivatives_offset_;
            auto apply_derivatives_backward = [&](auto&& op) {
                if (flags_.start_v) op(ws_ref.grads.start.v);
                if constexpr (SplineType::ORDER >= 5) if (flags_.start_a) op(ws_ref.grads.start.a);
                if constexpr (SplineType::ORDER >= 7) if (flags_.start_j) op(ws_ref.grads.start.j);
                
                if (flags_.end_v) op(ws_ref.grads.end.v);
                if constexpr (SplineType::ORDER >= 5) if (flags_.end_a) op(ws_ref.grads.end.a);
                if constexpr (SplineType::ORDER >= 7) if (flags_.end_j) op(ws_ref.grads.end.j);
            };

            apply_derivatives_backward([&](const VectorType& g) {
                grad_out.template segment<DIM>(offset) = g;
                offset += DIM;
            });

            return total_cost;
        }

        /**
         * @brief Secondary evaluate: 2 cost functions (no waypoints cost).
         * Forwards to primary evaluate with VoidWaypointsCost.
         */
        template <typename TimeCostFunc, typename IntegralCostFunc, typename Executor = SerialExecutor>
        requires Concepts::TimeCostLike<TimeCostFunc> &&
                 Concepts::IntegralCostLike<IntegralCostFunc, VectorType> &&
                 Concepts::ExecutorLike<Executor>
        double evaluate(const Eigen::VectorXd &x, Eigen::VectorXd &grad_out,
                        TimeCostFunc &&time_cost_func, 
                        IntegralCostFunc &&integral_cost_func,
                        Workspace *ws = nullptr,
                        const Executor& executor = Executor()) const
        {
            return evaluate(x, grad_out,
                            std::forward<TimeCostFunc>(time_cost_func),
                            VoidWaypointsCost(), 
                            std::forward<IntegralCostFunc>(integral_cost_func),
                            ws, executor);
        }



        const SplineType *getOptimalSpline() const
        {
            if (internal_ws_)
                return &(internal_ws_->spline);
            return nullptr;
        }

        struct GradientCheckResult
        {
            bool valid = false;          
            double error_norm = 0.0;      
            double rel_error = 0.0;       
            Eigen::VectorXd analytical;   
            Eigen::VectorXd numerical;   
            
            std::string makeReport() const {
                std::stringstream ss;
                if (valid) 
                    ss << "Gradient Check PASSED! Norm: " << error_norm << "\n";
                else
                    ss << "Gradient Check FAILED! Norm: " << error_norm << "\n";
                return ss.str();
            }
        };

        /**
         * @brief Primary checkGradients: 3 cost functions with optional Workspace.
         * @param ws Workspace pointer (default nullptr to use internal workspace).
         */
        template <typename TFunc, typename WFunc, typename IFunc>
        requires Concepts::TimeCostLike<TFunc> &&
                 Concepts::WaypointsCostLike<WFunc, WaypointsType> &&
                 Concepts::IntegralCostLike<IFunc, VectorType>
        GradientCheckResult checkGradients(const Eigen::VectorXd &x, 
                            TFunc &&tf, WFunc &&wf, IFunc &&ifc, 
                            Workspace *ws = nullptr,
                            double eps = 1e-6,
                            double tol = 1e-4)
        {
            GradientCheckResult res;
            
            Workspace& ws_ref = (ws != nullptr) ? *ws : *getOrCreateInternalWorkspace();
            
            res.analytical.resize(x.size());
            evaluate(x, res.analytical, tf, wf, ifc, &ws_ref);

            res.numerical.resize(x.size());
            Eigen::VectorXd dummy_grad(x.size());

            Eigen::VectorXd x_temp = x;

            for (int i = 0; i < x.size(); ++i)
            {
                double old_val = x_temp(i);
                
                x_temp(i) = old_val + eps;
                double c_p = evaluate(x_temp, dummy_grad, tf, wf, ifc, &ws_ref);
                
                x_temp(i) = old_val - eps;
                double c_m = evaluate(x_temp, dummy_grad, tf, wf, ifc, &ws_ref);
                
                x_temp(i) = old_val;

                res.numerical(i) = (c_p - c_m) / (2 * eps);
            }

            evaluate(x, res.analytical, tf, wf, ifc, &ws_ref);

            Eigen::VectorXd diff = res.analytical - res.numerical;
            res.error_norm = diff.norm();

            double grad_norm = res.analytical.norm();
            res.rel_error = (grad_norm > 1e-9) ? (res.error_norm / grad_norm) : res.error_norm;

            res.valid = (res.error_norm < tol);

            return res;
        }

        /**
         * @brief Secondary checkGradients: 2 cost functions (no waypoints cost).
         * Forwards to primary checkGradients with VoidWaypointsCost.
         */
        template <typename TFunc, typename IFunc>
        requires Concepts::TimeCostLike<TFunc> &&
                 Concepts::IntegralCostLike<IFunc, VectorType>
        GradientCheckResult checkGradients(const Eigen::VectorXd &x, 
                            TFunc &&tf, IFunc &&ifc, 
                            Workspace *ws = nullptr,
                            double eps = 1e-6,
                            double tol = 1e-4)
        {
            return checkGradients(x, 
                                  std::forward<TFunc>(tf), 
                                  VoidWaypointsCost(),
                                  std::forward<IFunc>(ifc), 
                                  ws,
                                  eps,
                                  tol);
        }


    private:
        /**
         * @brief Helper to safely format and report error messages.
         * Aggregates multiple error strings into a single message.
         * @param out Optional output pointer to store error message
         * @param errors Vector of individual error messages
         */
        void reportError(std::string* out, const std::vector<std::string>& errors) const
        {
            if (errors.empty()) return;
            
            std::stringstream ss;
            ss << "[SplineOptimizer Validation Failed] Found " << errors.size() << " error(s):\n";
            for (size_t i = 0; i < errors.size(); ++i) {
                ss << "  [" << (i + 1) << "] " << errors[i] << "\n";
            }
            
            std::string msg = ss.str();
            
            last_error_message_ = msg;
            
            if (out) {
                *out = msg;
            }
        }
        
        int calculateDimension() const
        {
            ensureLayoutCache();
            return total_dimension_;
        }

        template <typename IntegralFunc, typename Executor>
        void calculateIntegralCost(Workspace &ws, MatrixType &gdC, Eigen::VectorXd &gdT, double &cost, 
                                   IntegralFunc &&integral_cost, const Executor& executor) const
        {
            const auto &coeffs = ws.spline.getTrajectory().getCoefficients();
            
            double running_time = start_time_;
            for(int i = 0; i < num_segments_; ++i) {
                ws.segment_start_times[i] = running_time;
                running_time += ws.cache_times[i];
            }

            std::fill(ws.segment_costs.begin(), ws.segment_costs.end(), 0.0);

            ws.explicit_time_grad_buffer.setZero();

            int K = integral_num_steps_;
            double inv_K = 1.0 / K;
            
            executor(0, num_segments_, [&](int i) {
                double T = ws.cache_times[i];
                double dt = T * inv_K;
                int base_row = i * SplineType::COEFF_NUM;
                
                Eigen::Matrix<double, SplineType::COEFF_NUM, DIM> coeff_block = 
                     coeffs.template block<SplineType::COEFF_NUM, DIM>(base_row, 0); 
                
                double local_acc_cost = 0.0;
                double local_acc_gdT = 0.0;
                double local_acc_explicit_time_grad = 0.0;

                Eigen::Matrix<double, SplineType::COEFF_NUM, DIM> local_acc_gdC;
                local_acc_gdC.setZero();

                Eigen::Matrix<double, 1, SplineType::COEFF_NUM> b_p, b_v, b_a, b_j, b_s, b_c;
                double current_segment_start_time = ws.segment_start_times[i];

                for (int k = 0; k <= K; ++k)
                {
                    double alpha = (double)k * inv_K;
                    double t = alpha * T;

                    double weight_trap = (k == 0 || k == K) ? 0.5 : 1.0;
                    double common_weight = weight_trap * dt;

                    double t_global = current_segment_start_time + t;

                    SplineType::computeBasisFunctions(t, b_p, b_v, b_a, b_j, b_s, b_c);
                    
                    VectorType p, v, a, j, s, c;
                    p.transpose().noalias() = b_p * coeff_block;
                    v.transpose().noalias() = b_v * coeff_block;
                    a.transpose().noalias() = b_a * coeff_block;
                    j.transpose().noalias() = b_j * coeff_block;
                    s.transpose().noalias() = b_s * coeff_block;
                    c.transpose().noalias() = b_c * coeff_block;

                    VectorType gp = VectorType::Zero();
                    VectorType gv = VectorType::Zero();
                    VectorType ga = VectorType::Zero();
                    VectorType gj = VectorType::Zero();
                    VectorType gs = VectorType::Zero();
                    double gt = 0.0;

                    double c_val = integral_cost(t, t_global, i, p, v, a, j, s, gp, gv, ga, gj, gs, gt);

                    local_acc_cost += c_val * common_weight;

                    local_acc_gdC.noalias() += (
                        b_p.transpose() * gp.transpose() + 
                        b_v.transpose() * gv.transpose() + 
                        b_a.transpose() * ga.transpose() + 
                        b_j.transpose() * gj.transpose() + 
                        b_s.transpose() * gs.transpose()
                    ) * common_weight;

                    local_acc_gdT += c_val * weight_trap * inv_K;

                    double drift_grad = gp.dot(v) + gv.dot(a) + ga.dot(j) + gj.dot(s) + gs.dot(c);
                    local_acc_gdT += drift_grad * alpha * common_weight;

                    local_acc_gdT += gt * alpha * common_weight;
                    local_acc_explicit_time_grad += gt * common_weight;
                }

                ws.segment_costs[i] = local_acc_cost;
                
                gdT(i) += local_acc_gdT;
                ws.explicit_time_grad_buffer(i) += local_acc_explicit_time_grad;
                
                gdC.template block(base_row, 0, SplineType::COEFF_NUM, DIM) += local_acc_gdC;
            });

            for(int i = 0; i < num_segments_; ++i) {
                cost += ws.segment_costs[i];
            }

            double accumulator = 0.0;
            for (int i = num_segments_ - 1; i > 0; --i)
            {
                accumulator += ws.explicit_time_grad_buffer(i);
                gdT(i - 1) += accumulator;
            }
        }
    };
}
#endif
