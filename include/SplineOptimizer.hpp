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
     */
    struct SpatialMapProtocol
    {
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
            decltype(std::declval<T>().toPhysical(std::declval<Eigen::Matrix<double, DIM, 1>>(), std::declval<int>())),
            decltype(std::declval<T>().toUnconstrained(std::declval<Eigen::Matrix<double, DIM, 1>>(), std::declval<int>())),
            decltype(std::declval<T>().backwardGrad(std::declval<Eigen::Matrix<double, DIM, 1>>(), 
                                                    std::declval<Eigen::Matrix<double, DIM, 1>>(), 
                                                    std::declval<int>()))
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
        int opt_dim_ = 0;

        double rho_energy_ = 0.0;
        int integral_num_steps_ = 16;

        TimeMap default_time_map_;
        SpatialMap default_spatial_map_;

        const TimeMap* active_time_map_ = nullptr;
        const SpatialMap* active_spatial_map_ = nullptr;

        mutable std::unique_ptr<Workspace> internal_ws_;

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
              opt_dim_(other.opt_dim_),
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
                opt_dim_ = other.opt_dim_;
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
         */
        void setInitState(const std::vector<double> &t_points,
                          const WaypointsType &waypoints,
                          const BoundaryConditions<DIM> &bc)
        {
            start_time_ = t_points.front();

            ref_times_.clear();
            ref_times_.reserve(t_points.size() - 1);
            for (size_t i = 1; i < t_points.size(); ++i)
            {
                double dt = t_points[i] - t_points[i - 1];
                ref_times_.push_back(dt);
            }

            ref_waypoints_ = waypoints;
            ref_bc_ = bc;
            num_segments_ = static_cast<int>(ref_times_.size());

            if (internal_ws_)
                internal_ws_->resize(num_segments_);
        }

        /**
         * @brief Initialize using Time Segments (Durations).
         */
        void setInitState(const std::vector<double> &time_segments,
                          const WaypointsType &waypoints,
                          double start_time,
                          const BoundaryConditions<DIM> &bc)
        {
            start_time_ = start_time;
            ref_times_ = time_segments;
            ref_waypoints_ = waypoints;
            ref_bc_ = bc;
            num_segments_ = static_cast<int>(ref_times_.size());

            if (internal_ws_)
                internal_ws_->resize(num_segments_);
        }

        void setOptimizationFlags(const OptimizationFlags &flags) { flags_ = flags; }
        void setEnergyWeights(double rho_energy) { rho_energy_ = rho_energy; }
        void setIntegralNumSteps(int steps) { integral_num_steps_ = steps; }

        int getDimension() const { return calculateDimension(); }

        /**
         * @brief Generate initial guess x based on reference state.
         * Applies 'toUnconstrained' mapping for inner waypoints.
         */
        Eigen::VectorXd generateInitialGuess() const
        {
            int dim = calculateDimension();
            Eigen::VectorXd x(dim);
            int offset = 0;

            for (double t : ref_times_)
                x(offset++) = active_time_map_->toTau(t);

            for (int i = 1; i < num_segments_; ++i)
            {
                x.segment<DIM>(offset) = active_spatial_map_->toUnconstrained(ref_waypoints_[i], i - 1);
                offset += DIM;
            }

            auto push_vec = [&](const VectorType &v)
            { x.segment<DIM>(offset) = v; offset += DIM; };

            if (flags_.start_p)
                push_vec(ref_waypoints_.front());
            if (flags_.start_v)
                push_vec(ref_bc_.start_velocity);
            if constexpr (SplineType::ORDER >= 5)
            {
                if (flags_.start_a)
                    push_vec(ref_bc_.start_acceleration);
            }
            if constexpr (SplineType::ORDER >= 7)
            {
                if (flags_.start_j)
                    push_vec(ref_bc_.start_jerk);
            }

            if (flags_.end_p)
                push_vec(ref_waypoints_.back());
            if (flags_.end_v)
                push_vec(ref_bc_.end_velocity);
            if constexpr (SplineType::ORDER >= 5)
            {
                if (flags_.end_a)
                    push_vec(ref_bc_.end_acceleration);
            }
            if constexpr (SplineType::ORDER >= 7)
            {
                if (flags_.end_j)
                    push_vec(ref_bc_.end_jerk);
            }

            return x;
        }

        /**
         * @brief Thread-safe evaluate. Requires user to provide a Workspace.
         */
        template <typename TimeCostFunc, typename WaypointsCostFunc, typename IntegralCostFunc>
        double evaluate(const Eigen::VectorXd &x, Eigen::VectorXd &grad_out,
                        TimeCostFunc &&time_cost_func, 
                        WaypointsCostFunc &&waypoints_cost_func, 
                        IntegralCostFunc &&integral_cost_func,
                        Workspace &ws) const
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

            ws.resize(num_segments_);
            grad_out.setZero(x.size());
            double total_cost = 0.0;
            int n_inner = std::max(0, num_segments_ - 1);

            int offset = 0;
            for (int i = 0; i < num_segments_; ++i)
                ws.cache_times[i] = active_time_map_->toTime(x(offset++));

            for (int i = 1; i < num_segments_; ++i)
            {
                VectorType xi = x.template segment<DIM>(offset);
                ws.cache_waypoints[i] = active_spatial_map_->toPhysical(xi, i - 1);
                offset += DIM;
            }

            BoundaryConditions<DIM> current_bc = ref_bc_;
            ws.cache_waypoints[0] = ref_waypoints_[0];
            ws.cache_waypoints[num_segments_] = ref_waypoints_[num_segments_];

            auto pull_vec = [&](bool flag, VectorType &target)
            { if(flag) { target = x.template segment<DIM>(offset); offset += DIM; } };

            pull_vec(flags_.start_p, ws.cache_waypoints[0]);
            pull_vec(flags_.start_v, current_bc.start_velocity);
            if constexpr (SplineType::ORDER >= 5) pull_vec(flags_.start_a, current_bc.start_acceleration);
            if constexpr (SplineType::ORDER >= 7) pull_vec(flags_.start_j, current_bc.start_jerk);

            pull_vec(flags_.end_p, ws.cache_waypoints[num_segments_]);
            pull_vec(flags_.end_v, current_bc.end_velocity);
            if constexpr (SplineType::ORDER >= 5) pull_vec(flags_.end_a, current_bc.end_acceleration);
            if constexpr (SplineType::ORDER >= 7) pull_vec(flags_.end_j, current_bc.end_jerk);

            ws.spline.update(ws.cache_times, ws.cache_waypoints, start_time_, current_bc);

            ws.user_gdT_buffer.setZero();
            ws.cache_gdT.setZero();
            total_cost += time_cost_func(ws.cache_times, ws.user_gdT_buffer);
            ws.cache_gdT += ws.user_gdT_buffer;

            ws.cache_gdC.setZero(); 
            calculateIntegralCost(ws, ws.cache_gdC, ws.cache_gdT, total_cost, std::forward<IntegralCostFunc>(integral_cost_func));

            ws.spline.propagateGrad(ws.cache_gdC, ws.cache_gdT, ws.grads);

            if constexpr (!std::is_same_v<WCF, VoidWaypointsCost>)
            {
                ws.discrete_grad_q_buffer.setZero();

                double dw_cost = waypoints_cost_func(ws.cache_waypoints, ws.discrete_grad_q_buffer);
                total_cost += dw_cost;

                ws.grads.start.p += ws.discrete_grad_q_buffer.row(0);
                if (n_inner > 0) {
                    ws.grads.inner_points += ws.discrete_grad_q_buffer.block(1, 0, n_inner, DIM);
                }
                ws.grads.end.p += ws.discrete_grad_q_buffer.row(num_segments_);
            }

            if (rho_energy_ > 0)
            {
                double energy = ws.spline.getEnergy();
                total_cost += rho_energy_ * energy;

                ws.spline.getEnergyGrad(ws.energy_grads);

                ws.grads.times += rho_energy_ * ws.energy_grads.times;
                if (n_inner > 0)
                    ws.grads.inner_points += rho_energy_ * ws.energy_grads.inner_points;

                ws.grads.start.p += rho_energy_ * ws.energy_grads.start.p;
                ws.grads.start.v += rho_energy_ * ws.energy_grads.start.v;
                if constexpr (SplineType::ORDER >= 5) ws.grads.start.a += rho_energy_ * ws.energy_grads.start.a;
                if constexpr (SplineType::ORDER >= 7) ws.grads.start.j += rho_energy_ * ws.energy_grads.start.j;

                ws.grads.end.p += rho_energy_ * ws.energy_grads.end.p;
                ws.grads.end.v += rho_energy_ * ws.energy_grads.end.v;
                if constexpr (SplineType::ORDER >= 5) ws.grads.end.a += rho_energy_ * ws.energy_grads.end.a;
                if constexpr (SplineType::ORDER >= 7) ws.grads.end.j += rho_energy_ * ws.energy_grads.end.j;
            }

            offset = 0;

            for (int i = 0; i < num_segments_; ++i)
            {
                double tau = x(offset);
                double T = ws.cache_times[i];
                double gradT = ws.grads.times(i);
                grad_out(offset) = active_time_map_->backward(tau, T, gradT);
                offset++;
            }

            for (int i = 0; i < n_inner; ++i)
            {
                VectorType xi = x.template segment<DIM>(offset);
                VectorType grad_p = ws.grads.inner_points.row(i);

                grad_out.template segment<DIM>(offset) = active_spatial_map_->backwardGrad(xi, grad_p, i);
                offset += DIM;
            }

            auto push_grad = [&](bool flag, const VectorType &g)
            { if(flag) { grad_out.template segment<DIM>(offset) = g; offset += DIM; } };

            push_grad(flags_.start_p, ws.grads.start.p);
            push_grad(flags_.start_v, ws.grads.start.v);
            if constexpr (SplineType::ORDER >= 5) push_grad(flags_.start_a, ws.grads.start.a);
            if constexpr (SplineType::ORDER >= 7) push_grad(flags_.start_j, ws.grads.start.j);

            push_grad(flags_.end_p, ws.grads.end.p);
            push_grad(flags_.end_v, ws.grads.end.v);
            if constexpr (SplineType::ORDER >= 5) push_grad(flags_.end_a, ws.grads.end.a);
            if constexpr (SplineType::ORDER >= 7) push_grad(flags_.end_j, ws.grads.end.j);

            return total_cost;
        }

        template <typename TimeCostFunc, typename IntegralCostFunc>
        double evaluate(const Eigen::VectorXd &x, Eigen::VectorXd &grad_out,
                        TimeCostFunc &&time_cost_func, 
                        IntegralCostFunc &&integral_cost_func,
                        Workspace &ws) const
        {
            return evaluate(x, grad_out,
                            std::forward<TimeCostFunc>(time_cost_func),
                            VoidWaypointsCost(), 
                            std::forward<IntegralCostFunc>(integral_cost_func),
                            ws);
        }

        /**
         * @brief Convenience evaluate. Uses internal workspace.
         */
        template <typename TimeCostFunc, typename IntegralCostFunc>
        double evaluate(const Eigen::VectorXd &x, Eigen::VectorXd &grad_out,
                        TimeCostFunc &&time_cost_func, IntegralCostFunc &&integral_cost_func) const
        {
            if (!internal_ws_)
            {
                internal_ws_ = std::unique_ptr<Workspace>(new Workspace());
            }
            return evaluate(x, grad_out,
                            std::forward<TimeCostFunc>(time_cost_func),
                            std::forward<IntegralCostFunc>(integral_cost_func),
                            *internal_ws_);
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

        template <typename TFunc, typename WFunc, typename IFunc>
        GradientCheckResult checkGradients(const Eigen::VectorXd &x, 
                            TFunc &&tf, WFunc &&wf, IFunc &&ifc, 
                            Workspace &ws,
                            double eps = 1e-6,
                            double tol = 1e-4)
        {
            GradientCheckResult res;
            
            res.analytical.resize(x.size());
            evaluate(x, res.analytical, tf, wf, ifc, ws);

            res.numerical.resize(x.size());
            Eigen::VectorXd dummy_grad(x.size());

            Eigen::VectorXd x_temp = x;

            for (int i = 0; i < x.size(); ++i)
            {
                double old_val = x_temp(i);
                
                x_temp(i) = old_val + eps;
                double c_p = evaluate(x_temp, dummy_grad, tf, wf, ifc, ws);
                
                x_temp(i) = old_val - eps;
                double c_m = evaluate(x_temp, dummy_grad, tf, wf, ifc, ws);
                
                x_temp(i) = old_val;

                res.numerical(i) = (c_p - c_m) / (2 * eps);
            }

            evaluate(x, res.analytical, tf, wf, ifc, ws);

            Eigen::VectorXd diff = res.analytical - res.numerical;
            res.error_norm = diff.norm();

            double grad_norm = res.analytical.norm();
            res.rel_error = (grad_norm > 1e-9) ? (res.error_norm / grad_norm) : res.error_norm;

            res.valid = (res.error_norm < tol);

            return res;
        }

        template <typename TFunc, typename IFunc>
        GradientCheckResult checkGradients(const Eigen::VectorXd &x, 
                            TFunc &&tf, IFunc &&ifc, 
                            Workspace &ws,
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

        template <typename TFunc, typename IFunc>
        GradientCheckResult checkGradients(const Eigen::VectorXd &x, 
                                           TFunc &&tf, IFunc &&ifc,
                                           double eps = 1e-6,
                                           double tol = 1e-4)
        {
            if (!internal_ws_)
            {
                internal_ws_ = std::unique_ptr<Workspace>(new Workspace());
            }
            return checkGradients(x, 
                                  std::forward<TFunc>(tf), 
                                  VoidWaypointsCost(), 
                                  std::forward<IFunc>(ifc), 
                                  *internal_ws_,
                                  eps,
                                  tol);
        }

    private:
        int calculateDimension() const
        {
            int dim = num_segments_ + std::max(0, num_segments_ - 1) * DIM;
            auto check = [&](bool f)
            { if(f) dim += DIM; };
            check(flags_.start_p);
            check(flags_.start_v);
            if constexpr (SplineType::ORDER >= 5)
                check(flags_.start_a);
            if constexpr (SplineType::ORDER >= 7)
                check(flags_.start_j);
            check(flags_.end_p);
            check(flags_.end_v);
            if constexpr (SplineType::ORDER >= 5)
                check(flags_.end_a);
            if constexpr (SplineType::ORDER >= 7)
                check(flags_.end_j);
            return dim;
        }

        template <typename IntegralFunc>
        void calculateIntegralCost(Workspace &ws, MatrixType &gdC, Eigen::VectorXd &gdT, double &cost, IntegralFunc &&integral_cost) const
        {
            const auto &coeffs = ws.spline.getTrajectory().getCoefficients();
            Eigen::Matrix<double, 1, SplineType::COEFF_NUM> b_p, b_v, b_a, b_j, b_s, b_c;

            ws.explicit_time_grad_buffer.setZero();

            double current_segment_start_time = start_time_;

            int K = integral_num_steps_;
            double inv_K = 1.0 / K;

            for (int i = 0; i < num_segments_; ++i)
            {
                double T = ws.cache_times[i];
        
                double dt = T * inv_K;
                int base_row = i * SplineType::COEFF_NUM;
                auto block = coeffs.block(base_row, 0, SplineType::COEFF_NUM, DIM);

                for (int k = 0; k <= K; ++k)
                {
                    double alpha = (double)k * inv_K;
                    double t = alpha * T; 

                    double weight_trap = (k == 0 || k == K) ? 0.5 : 1.0;
                    double common_weight = weight_trap * dt;

                    double t_global = current_segment_start_time + t;

                    SplineType::computeBasisFunctions(t, b_p, b_v, b_a, b_j, b_s, b_c);

                    VectorType p = (b_p * block).transpose();
                    VectorType v = (b_v * block).transpose();
                    VectorType a = (b_a * block).transpose();
                    VectorType j = (b_j * block).transpose();
                    VectorType s = (b_s * block).transpose();
                    VectorType c = (b_c * block).transpose();

                    VectorType gp = VectorType::Zero(), gv = VectorType::Zero(), ga = VectorType::Zero(),
                               gj = VectorType::Zero(), gs = VectorType::Zero();

                    double gt = 0.0;  

                    double c_val = integral_cost(t, t_global, i, p, v, a, j, s, gp, gv, ga, gj, gs, gt);

                    if (std::abs(c_val) > 1e-12)
                    {
                        cost += c_val * common_weight;

                        for (int r = 0; r < SplineType::COEFF_NUM; ++r)
                        {
                            VectorType g = gp * b_p(r) + gv * b_v(r) + ga * b_a(r) + gj * b_j(r) + gs * b_s(r);
                            gdC.row(base_row + r) += g.transpose() * common_weight;
                        }

                        gdT(i) += c_val * weight_trap * inv_K;

                        double drift_grad = gp.dot(v) + gv.dot(a) + ga.dot(j) + gj.dot(s) + gs.dot(c); 
                        gdT(i) += drift_grad * alpha * common_weight;

                        gdT(i) += gt * alpha * common_weight;

                        ws.explicit_time_grad_buffer(i) += gt * common_weight;
                    }
                }

                current_segment_start_time += T;
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