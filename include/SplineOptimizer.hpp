#ifndef SPLINE_OPTIMIZER_HPP
#define SPLINE_OPTIMIZER_HPP

#include "SplineTrajectory.hpp"
#include <algorithm>
#include <vector>
#include <cmath>
#include <iostream>
#include <iomanip>
#include <memory>
#include <type_traits> 
#include <utility>   
#include <sstream>  

namespace SplineTrajectory
{
    // =========================================================================
    //  INTERFACE DOCUMENTATION (Reference Only)
    //  These definitions are provided for documentation purposes. 
    //  User-defined types (Functors/Lambdas) do not need to inherit from them, 
    //  but must satisfy the function signatures verified by TypeTraits.
    // =========================================================================

#if 0
    /**
     * @brief TimeMap Interface Protocol
     * Defines how the unconstrained optimization variable (tau) maps to physical time (T).
     */
    struct TimeMapProtocol
    {
        // Convert unconstrained variable 'tau' to physical time 'T'
        double toTime(double tau) const;
        // Convert physical time 'T' to unconstrained variable 'tau'
        double toTau(double T) const;
        // Compute gradient w.r.t tau given gradient w.r.t T (Chain Rule)
        // Returns: dCost/dtau = (dCost/dT) * (dT/dtau)
        double backward(double tau, double T, double gradT) const;
    };

    /**
     * @brief SpatialMap Interface Protocol [NEW]
     * Defines how the unconstrained variable (xi) maps to physical position (p).
     * Unlike TimeMap, this acts on an instance level to hold constraints (e.g., Polyhedrons).
     *
     * Global Absolute Indexing:
     * - index = 0: The Start Point of the trajectory
     * - index = 1 to N-1: The Inner Waypoints
     * - index = N: The End Point of the trajectory
     */
    struct SpatialMapProtocol
    {
        // Get dimension of unconstrained variable xi for the point at global 'index'
        int getUnconstrainedDim(int index) const;

        // Forward: xi (unconstrained) -> p (physical)
        Eigen::VectorXd toPhysical(const Eigen::VectorXd& xi, int index) const;

        // Backward: p (physical) -> xi (unconstrained), used for initial guess
        Eigen::VectorXd toUnconstrained(const Eigen::VectorXd& p, int index) const;

        // Gradient: dCost/dxi = (dCost/dp) * (dp/dxi)
        Eigen::VectorXd backwardGrad(const Eigen::VectorXd& xi, const Eigen::VectorXd& grad_p, int index) const;
    };

    /**
     * @brief IntegralCostFunc Protocol
     * Functor to compute trajectory integral cost.
     * Returns the scalar cost value.
     */
    struct IntegralCostProtocol
    {
        /**
         * @param t         Relative time inside the segment [0, T]
         * @param t_global  Global time from start of trajectory    
         * @param i         Segment index
         * @param p         Position vector (dim)
         * @param v         Velocity vector (dim)
         * @param a         Acceleration vector (dim)
         * @param j         Jerk vector (dim)
         * @param s         Snap vector (dim)
         * @param gp        [Output] Gradient w.r.t Position
         * @param gv        [Output] Gradient w.r.t Velocity
         * @param ga        [Output] Gradient w.r.t Acceleration
         * @param gj        [Output] Gradient w.r.t Jerk
         * @param gs        [Output] Gradient w.r.t Snap
         * @param gt        [Output] Gradient w.r.t Explicit Time (e.g. dynamic obstacles)
         * @return          Scalar cost value to be accumulated
         */
        double operator()(double t, double t_global, int i,
                          const Eigen::VectorXd &p, const Eigen::VectorXd &v,
                          const Eigen::VectorXd &a, const Eigen::VectorXd &j, const Eigen::VectorXd &s,
                          Eigen::VectorXd &gp, Eigen::VectorXd &gv, Eigen::VectorXd &ga,
                          Eigen::VectorXd &gj, Eigen::VectorXd &gs, double &gt) const;
    };

    /**
     * @brief TimeCostFunc Protocol
     * Functor to compute cost based on ALL segment durations.
     * Allows for global time constraints (e.g., total time).
     */
    struct TimeCostProtocol
    {
        /**
         * @param Ts    Physical durations of all segments (std::vector<double>)
         * @param grad  [Output] Gradient of cost w.r.t each T (Eigen::VectorXd ref)
         * @return      Cost value
         */
        double operator()(const std::vector<double>& Ts, Eigen::VectorXd &grad) const;
    };

    /**
     * @brief WaypointsCostProtocol
     * Functor to compute cost based on DISCRETE waypoints (q).
     * This is separate from the integral cost along the continuous curve.
     */
    struct WaypointsCostProtocol
    {
        /**
         * @param waypoints  The full list of waypoints [q0, q1, ... qN] (Physical space)
         * @param grad_q     [Output] Gradient w.r.t each waypoint (Matrix: (N+1) x DIM)
         * Rows correspond to q0, q1, ... qN
         * @return           Cost value
         */
        double operator()(const SplineVector<Eigen::Matrix<double, Eigen::Dynamic, 1>> &waypoints, 
                          Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> &grad_q) const;
    };
#endif

    namespace TypeTraits
    {
        template <typename...>
        using void_t = void;

        // --- TimeMap Traits ---
        template <typename T, typename = void>
        struct HasTimeMapInterface : std::false_type {};

        template <typename T>
        struct HasTimeMapInterface<T, void_t<
            decltype(static_cast<double>(std::declval<T>().toTime(std::declval<double>()))),
            decltype(static_cast<double>(std::declval<T>().toTau(std::declval<double>()))),
            decltype(static_cast<double>(std::declval<T>().backward(std::declval<double>(), std::declval<double>(), std::declval<double>())))
        >> : std::true_type {};

        // --- SpatialMap Traits ---
        template <typename T, int DIM, typename = void>
        struct HasSpatialMapInterface : std::false_type {};

        template <typename T, int DIM>
        struct HasSpatialMapInterface<T, DIM, void_t<
            decltype(static_cast<int>(std::declval<T>().getUnconstrainedDim(std::declval<int>()))),
            decltype(std::declval<T>().toPhysical(std::declval<Eigen::VectorXd>(), std::declval<int>())),
            decltype(std::declval<T>().toUnconstrained(std::declval<Eigen::VectorXd>(), std::declval<int>())),
            decltype(std::declval<T>().backwardGrad(std::declval<Eigen::VectorXd>(),
                                                    std::declval<Eigen::VectorXd>(),
                                                    std::declval<int>()))
        >> : std::true_type {};

        template <typename T, typename = void>
        struct HasExecutorInterface : std::false_type {};

        template <typename T>
        struct HasExecutorInterface<T, void_t<
            decltype(std::declval<T>()(
                std::declval<int>(),       
                std::declval<int>(),       
                std::declval<void(*)(int)>()                  
            ))
        >> : std::true_type {};

        // --- Cost Func Traits ---
        template <typename T, typename = void>
        struct HasTimeCostInterface : std::false_type {};

        template <typename T>
        struct HasTimeCostInterface<T, void_t<
            decltype(static_cast<double>(std::declval<T>()(
                std::declval<const std::vector<double>&>(), // All times
                std::declval<Eigen::VectorXd &>()           // Gradient vector
            )))
        >> : std::true_type {};

        template <typename T, typename WaypointsType, typename = void>
        struct HasWaypointsCostInterface : std::false_type {};

        template <typename T, typename WaypointsType>
        struct HasWaypointsCostInterface<T, WaypointsType, void_t<
            decltype(static_cast<double>(std::declval<T>()(
                std::declval<const WaypointsType &>(),                // Waypoints
                std::declval<Eigen::Matrix<double, -1, -1> &>()       // Gradient Matrix (Dynamic)
            )))
        >> : std::true_type {};

        template <typename T, typename VecT, typename = void>
        struct HasIntegralCostInterface : std::false_type {};

        template <typename T, typename VecT>
        struct HasIntegralCostInterface<T, VecT, void_t<
            decltype(static_cast<double>(std::declval<T>()(
                std::declval<double>(),                  // t (relative)
                std::declval<double>(),                  // t_global
                std::declval<int>(),                     // segment_index
                std::declval<const VecT &>(),            // p
                std::declval<const VecT &>(),            // v
                std::declval<const VecT &>(),            // a
                std::declval<const VecT &>(),            // j
                std::declval<const VecT &>(),            // s
                std::declval<VecT &>(),                  // gp
                std::declval<VecT &>(),                  // gv
                std::declval<VecT &>(),                  // ga
                std::declval<VecT &>(),                  // gj
                std::declval<VecT &>(),                  // gs
                std::declval<double &>()                 // gt
            )))
        >> : std::true_type {};
    }

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
    class SplineOptimizer
    {
        static_assert(TypeTraits::HasTimeMapInterface<TimeMap>::value,
                      "\n[SplineOptimizer Error] The provided 'TimeMap' type does not satisfy the required interface.\n"
                      "It must implement const member methods:\n"
                      "  double toTime(double tau) const;\n"
                      "  double toTau(double T) const;\n"
                      "  double backward(double tau, double T, double gradT) const;\n");

        static_assert(TypeTraits::HasSpatialMapInterface<SpatialMap, DIM>::value,
                      "\n[SplineOptimizer Error] The provided 'SpatialMap' type does not satisfy the required interface.\n"
                      "It must implement toPhysical, toUnconstrained, and backwardGrad methods.\n");

    public:
        using VectorType = typename SplineType::VectorType;
        using MatrixType = typename SplineType::MatrixType;
        using WaypointsType = SplineVector<VectorType>;

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

            void resize(int num_segments)
            {
                if (static_cast<int>(cache_times.size()) != num_segments)
                {
                    cache_times.resize(num_segments);
                    cache_waypoints.resize(num_segments + 1);
                    cache_gdT.resize(num_segments);
                    user_gdT_buffer.resize(num_segments);
                    cache_gdC.resize(num_segments * SplineType::COEFF_NUM, DIM);
                    explicit_time_grad_buffer.resize(num_segments);
                    discrete_grad_q_buffer.resize(num_segments + 1, DIM);
                }
            }
        };

    private:
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
              default_spatial_map_(other.default_spatial_map_)
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

            is_valid_ = checkValidity();
            
            if (is_valid_) {
                if (internal_ws_)
                    internal_ws_->resize(num_segments_);
            }

            return is_valid_;
        }

        void setOptimizationFlags(const OptimizationFlags &flags) { flags_ = flags; }
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
            
            if (ref_waypoints_.size() != static_cast<size_t>(num_segments_ + 1)) {
                errors.push_back("Size mismatch: ref_waypoints_.size() = " + std::to_string(ref_waypoints_.size()) + 
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
            
            for (size_t i = 0; i < ref_waypoints_.size(); ++i) {
                if (!ref_waypoints_[i].array().isFinite().all()) {
                    errors.push_back("Waypoint [" + std::to_string(i) + "] contains NaN or Inf");
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
            int dim = calculateDimension();
            Eigen::VectorXd x(dim);
            int offset = 0;

            for (double t : ref_times_)
                x(offset++) = active_time_map_->toTau(t);

            auto is_spatial_optimized = [&](int idx) {
                if (idx == 0) return flags_.start_p;
                if (idx == num_segments_) return flags_.end_p;
                return true; 
            };

            for (int i = 0; i <= num_segments_; ++i)
            {
                if (is_spatial_optimized(i)) {
                    int dof = active_spatial_map_->getUnconstrainedDim(i);
                    x.segment(offset, dof) = active_spatial_map_->toUnconstrained(ref_waypoints_[i], i);
                    offset += dof;
                }
            }

            BoundaryConditions<DIM> bc = ref_bc_;
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
        double evaluate(const Eigen::VectorXd &x, Eigen::VectorXd &grad_out,
                        TimeCostFunc &&time_cost_func,
                        WaypointsCostFunc &&waypoints_cost_func,
                        IntegralCostFunc &&integral_cost_func,
                        Workspace *ws = nullptr,
                        const Executor& executor = Executor()) const
        {
            using TCF = typename std::decay<TimeCostFunc>::type;
            using WCF = typename std::decay<WaypointsCostFunc>::type;
            using SCF = typename std::decay<IntegralCostFunc>::type;

            static_assert(TypeTraits::HasTimeCostInterface<TCF>::value,
                          "\n[SplineOptimizer Error] 'TimeCostFunc' signature mismatch.\n"
                          "Required: double operator()(const vector<double>& Ts, VectorXd &grad)\n");

            static_assert(TypeTraits::HasWaypointsCostInterface<WCF, WaypointsType>::value,
                          "\n[SplineOptimizer Error] 'WaypointsCostFunc' signature mismatch.\n"
                          "Required: double operator()(const WaypointsType& qs, MatrixXd &grad_q)\n");

            static_assert(TypeTraits::HasIntegralCostInterface<SCF, VectorType>::value,
                          "\n[SplineOptimizer Error] 'IntegralCostFunc' signature mismatch.\n");

            static_assert(TypeTraits::HasExecutorInterface<Executor>::value,
                          "\n[SplineOptimizer Error] 'Executor' signature mismatch.\n"
                          "The Executor must implement:\n"
                          "  void operator()(int start, int end, Func&& f) const;\n"
                          "where 'f' is a callable accepting an 'int' index.\n");

            Workspace& ws_ref = (ws != nullptr) ? *ws : *getOrCreateInternalWorkspace();
            ws_ref.resize(num_segments_);
            grad_out.setZero(x.size());
            double total_cost = 0.0;
            int n_inner = std::max(0, num_segments_ - 1);

            int offset = 0;

            for (int i = 0; i < num_segments_; ++i)
                ws_ref.cache_times[i] = active_time_map_->toTime(x(offset++));

            auto is_spatial_optimized = [&](int idx) {
                if (idx == 0) return flags_.start_p;
                if (idx == num_segments_) return flags_.end_p;
                return true;
            };

            for (int i = 0; i <= num_segments_; ++i)
            {
                if (is_spatial_optimized(i)) {
                    int dof = active_spatial_map_->getUnconstrainedDim(i);
                    Eigen::VectorXd xi = x.segment(offset, dof);
                    ws_ref.cache_waypoints[i] = active_spatial_map_->toPhysical(xi, i);
                    offset += dof;
                } else {
                    ws_ref.cache_waypoints[i] = ref_waypoints_[i];
                }
            }

            BoundaryConditions<DIM> current_bc = ref_bc_;
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

                ws_ref.grads.start.p += ws_ref.discrete_grad_q_buffer.row(0);
                if (n_inner > 0) {
                    ws_ref.grads.inner_points += ws_ref.discrete_grad_q_buffer.block(1, 0, n_inner, DIM);
                }
                ws_ref.grads.end.p += ws_ref.discrete_grad_q_buffer.row(num_segments_);
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

            offset = 0;

            // Time gradient: backward through time map
            for (int i = 0; i < num_segments_; ++i)
            {
                double tau = x(offset);
                double T = ws_ref.cache_times[i];
                double gradT = ws_ref.grads.times(i);
                grad_out(offset) = active_time_map_->backward(tau, T, gradT);
                offset++;
            }

            // Spatial gradient: unified loop with backward mapping
            for (int i = 0; i <= num_segments_; ++i)
            {
                if (is_spatial_optimized(i)) {
                    int dof = active_spatial_map_->getUnconstrainedDim(i);
                    Eigen::VectorXd xi = x.segment(offset, dof);

                    Eigen::VectorXd grad_p_phys;
                    if (i == 0) {
                        grad_p_phys = ws_ref.grads.start.p;
                    } else if (i == num_segments_) {
                        grad_p_phys = ws_ref.grads.end.p;
                    } else {
                        grad_p_phys = ws_ref.grads.inner_points.row(i - 1);
                    }

                    // Chain Rule: dCost/dxi = dCost/dp * dp/dxi
                    grad_out.segment(offset, dof) = active_spatial_map_->backwardGrad(xi, grad_p_phys, i);
                    offset += dof;
                }
            }

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
            int dim = 0;

            dim += num_segments_;

            auto is_spatial_optimized = [&](int idx) {
                if (idx == 0) return flags_.start_p;
                if (idx == num_segments_) return flags_.end_p;
                return true; 
            };

            for (int i = 0; i <= num_segments_; ++i)
            {
                if (is_spatial_optimized(i)) {
                    dim += active_spatial_map_->getUnconstrainedDim(i);
                }
            }

            auto apply_derivatives = [&](auto&& op) {
                if (flags_.start_v) op();
                if constexpr (SplineType::ORDER >= 5) if (flags_.start_a) op();
                if constexpr (SplineType::ORDER >= 7) if (flags_.start_j) op();
                
                if (flags_.end_v) op();
                if constexpr (SplineType::ORDER >= 5) if (flags_.end_a) op();
                if constexpr (SplineType::ORDER >= 7) if (flags_.end_j) op();
            };

            apply_derivatives([&]() { dim += DIM; });

            return dim;
        }

        template <typename IntegralFunc, typename Executor>
        void calculateIntegralCost(Workspace &ws, MatrixType &gdC, Eigen::VectorXd &gdT, double &cost, 
                                   IntegralFunc &&integral_cost, const Executor& executor) const
        {
            const auto &coeffs = ws.spline.getTrajectory().getCoefficients();
            
            std::vector<double> segment_start_times(num_segments_);
            double running_time = start_time_;
            for(int i = 0; i < num_segments_; ++i) {
                segment_start_times[i] = running_time;
                running_time += ws.cache_times[i];
            }

            std::vector<double> segment_costs(num_segments_, 0.0);

            ws.explicit_time_grad_buffer.setZero();

            int K = integral_num_steps_;
            double inv_K = 1.0 / K;
            
            executor(0, num_segments_, [&](int i) {
                double T = ws.cache_times[i];
                double dt = T * inv_K;
                int base_row = i * SplineType::COEFF_NUM;
                
                constexpr int LocalOptions = (DIM == 1) ? Eigen::ColMajor : Eigen::RowMajor; 
                Eigen::Matrix<double, SplineType::COEFF_NUM, DIM, LocalOptions> coeff_block = 
                     coeffs.template block<SplineType::COEFF_NUM, DIM>(base_row, 0); 
                
                double local_acc_cost = 0.0;
                double local_acc_gdT = 0.0;
                double local_acc_explicit_time_grad = 0.0;

                Eigen::Matrix<double, SplineType::COEFF_NUM, DIM, LocalOptions> local_acc_gdC;
                local_acc_gdC.setZero();

                Eigen::Matrix<double, 1, SplineType::COEFF_NUM> b_p, b_v, b_a, b_j, b_s, b_c;
                double current_segment_start_time = segment_start_times[i];

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

                segment_costs[i] = local_acc_cost;
                
                gdT(i) += local_acc_gdT;
                ws.explicit_time_grad_buffer(i) += local_acc_explicit_time_grad;
                
                gdC.block(base_row, 0, SplineType::COEFF_NUM, DIM) += local_acc_gdC;
            });

            for(int i = 0; i < num_segments_; ++i) {
                cost += segment_costs[i];
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