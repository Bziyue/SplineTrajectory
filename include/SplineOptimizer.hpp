#ifndef Spline_OPTIMIZER_HPP
#define Spline_OPTIMIZER_HPP

#include "SplineTrajectory.hpp"
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
        static double toTime(double tau);
        // Convert physical time 'T' to unconstrained variable 'tau'
        static double toTau(double T);
        // Compute gradient w.r.t tau given gradient w.r.t T (Chain Rule)
        // Returns: dCost/dtau = (dCost/dT) * (dT/dtau)
        static double backward(double tau, double T, double gradT);
    };

    /**
     * @brief SpatialCostFunc Protocol
     * Functor to compute trajectory integral cost.
     * Returns the scalar cost value.
     */
    struct SpatialCostProtocol
    {
        /**
         * @param t    Relative time inside the segment [0, T]
         * @param p    Position vector (dim)
         * @param v    Velocity vector (dim)
         * @param a    Acceleration vector (dim)
         * @param j    Jerk vector (dim)
         * @param s    Snap vector (dim)
         * @param gp   [Output] Gradient w.r.t Position
         * @param gv   [Output] Gradient w.r.t Velocity
         * @param ga   [Output] Gradient w.r.t Acceleration
         * @param gj   [Output] Gradient w.r.t Jerk
         * @param gs   [Output] Gradient w.r.t Snap
         * @return     Scalar cost value to be accumulated
         */
        double operator()(double t,
                          const Eigen::VectorXd &p, const Eigen::VectorXd &v,
                          const Eigen::VectorXd &a, const Eigen::VectorXd &j, const Eigen::VectorXd &s,
                          Eigen::VectorXd &gp, Eigen::VectorXd &gv, Eigen::VectorXd &ga,
                          Eigen::VectorXd &gj, Eigen::VectorXd &gs) const;
    };

    /**
     * @brief TimeCostFunc Protocol
     * Functor to compute cost based on segment duration.
     * Returns the cost value.
     */
    struct TimeCostProtocol
    {
        /**
         * @param T    Physical duration of the segment
         * @param grad [Output] Gradient of cost w.r.t T (dL/dT)
         * @return     Cost value
         */
        double operator()(double T, double &grad) const;
    };
#endif

    namespace TypeTraits
    {
        template <typename...>
        using void_t = void;

        template <typename T, typename = void>
        struct HasTimeMapInterface : std::false_type {};

        template <typename T>
        struct HasTimeMapInterface<T, void_t<
            decltype(static_cast<double>(T::toTime(std::declval<double>()))),
            decltype(static_cast<double>(T::toTau(std::declval<double>()))),
            decltype(static_cast<double>(T::backward(std::declval<double>(), std::declval<double>(), std::declval<double>())))
        >> : std::true_type {};

        template <typename T, typename = void>
        struct HasTimeCostInterface : std::false_type {};

        template <typename T>
        struct HasTimeCostInterface<T, void_t<
            decltype(static_cast<double>(std::declval<T>()(
                std::declval<double>(),   // T
                std::declval<double &>()  // grad ref
            )))
        >> : std::true_type {};

        template <typename T, typename VecT, typename = void>
        struct HasSpatialCostInterface : std::false_type {};

        template <typename T, typename VecT>
        struct HasSpatialCostInterface<T, VecT, void_t<
            decltype(static_cast<double>(std::declval<T>()(
                std::declval<double>(),                  // t
                std::declval<const VecT &>(),            // p
                std::declval<const VecT &>(),            // v
                std::declval<const VecT &>(),            // a
                std::declval<const VecT &>(),            // j
                std::declval<const VecT &>(),            // s
                std::declval<VecT &>(),                  // gp
                std::declval<VecT &>(),                  // gv
                std::declval<VecT &>(),                  // ga
                std::declval<VecT &>(),                  // gj
                std::declval<VecT &>()                   // gs
            )))
        >> : std::true_type {};
    }

    struct IdentityTimeMap
    {
        static double toTime(double tau) { return tau; }
        static double toTau(double T) { return T; }
        // T = tau => dT/dtau = 1 => grad_tau = gradT * 1
        static double backward(double tau, double T, double gradT) { return gradT; }
    };

    struct ExpTimeMap
    {
        static double toTime(double tau) { return std::exp(tau); }
        static double toTau(double T) { return std::log(T); }
        // T = exp(tau) => dT/dtau = exp(tau) = T => grad_tau = gradT * T
        static double backward(double tau, double T, double gradT) { return gradT * T; }
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
              typename TimeMap = IdentityTimeMap>
    class SplineOptimizer
    {
        static_assert(TypeTraits::HasTimeMapInterface<TimeMap>::value,
                      "\n[SplineOptimizer Error] The provided 'TimeMap' type does not satisfy the required interface.\n"
                      "It must implement static methods:\n"
                      "  static double toTime(double tau);\n"
                      "  static double toTau(double T);\n"
                      "  static double backward(double tau, double T, double gradT);\n");

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

            typename SplineType::Gradients grads;
            typename SplineType::Gradients energy_grads;

            void resize(int num_segments)
            {
                if (static_cast<int>(cache_times.size()) != num_segments)
                {
                    cache_times.resize(num_segments);
                    cache_waypoints.resize(num_segments + 1);
                    cache_gdT.resize(num_segments);
                    cache_gdC.resize(num_segments * SplineType::COEFF_NUM, DIM);
                }
            }
        };

    private:
        std::vector<double> ref_times_; // Represents segment durations
        WaypointsType ref_waypoints_;
        BoundaryConditions<DIM> ref_bc_;
        double start_time_ = 0.0;

        OptimizationFlags flags_;
        int num_segments_ = 0;
        int opt_dim_ = 0;

        double rho_energy_ = 0.0;
        double integral_step_ = 0.05;

        // === Internal Workspace for Convenience API ===
        mutable std::unique_ptr<Workspace> internal_ws_;

    public:
        SplineOptimizer() = default;

        SplineOptimizer(const SplineOptimizer &other)
            : ref_times_(other.ref_times_),
              ref_waypoints_(other.ref_waypoints_),
              ref_bc_(other.ref_bc_),
              start_time_(other.start_time_),
              flags_(other.flags_),
              num_segments_(other.num_segments_),
              opt_dim_(other.opt_dim_),
              rho_energy_(other.rho_energy_),
              integral_step_(other.integral_step_)
        {
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
                integral_step_ = other.integral_step_;
                if (other.internal_ws_)
                    internal_ws_ = std::unique_ptr<Workspace>(new Workspace(*other.internal_ws_));
                else
                    internal_ws_.reset();
            }
            return *this;
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
        void setEnergeWeights(double rho_energy) { rho_energy_ = rho_energy; }
        void setIntegrationStep(double step) { integral_step_ = step; }

        int getDimension() const { return calculateDimension(); }

        /**
         * @brief Generate initial guess x based on reference state.
         */
        Eigen::VectorXd generateInitialGuess() const
        {
            int dim = calculateDimension();
            Eigen::VectorXd x(dim);
            int offset = 0;

            for (double t : ref_times_)
                x(offset++) = TimeMap::toTau(t);

            for (int i = 1; i < num_segments_; ++i)
            {
                x.segment<DIM>(offset) = ref_waypoints_[i];
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
        template <typename TimeCostFunc, typename SpatialCostFunc>
        double evaluate(const Eigen::VectorXd &x, Eigen::VectorXd &grad_out,
                        TimeCostFunc &&time_cost_func, SpatialCostFunc &&spatial_cost_func,
                        Workspace &ws) const
        {
            using TCF = typename std::decay<TimeCostFunc>::type;
            using SCF = typename std::decay<SpatialCostFunc>::type;

            static_assert(TypeTraits::HasTimeCostInterface<TCF>::value,
                          "\n[SplineOptimizer Error] The provided 'TimeCostFunc' does not satisfy the required interface.\n"
                          "Signature required: double operator()(double T, double &grad)\n");

            static_assert(TypeTraits::HasSpatialCostInterface<SCF, VectorType>::value,
                          "\n[SplineOptimizer Error] The provided 'SpatialCostFunc' does not satisfy the required interface.\n"
                          "Signature required: double operator()(double t, const VectorXd& p, ..., VectorXd &gp, ...)\n");

            ws.resize(num_segments_);
            grad_out.setZero(x.size());
            double total_cost = 0.0;
            int n_inner = std::max(0, num_segments_ - 1);

            int offset = 0;
            for (int i = 0; i < num_segments_; ++i)
                ws.cache_times[i] = TimeMap::toTime(x(offset++));

            for (int i = 1; i < num_segments_; ++i)
            {
                ws.cache_waypoints[i] = x.template segment<DIM>(offset);
                offset += DIM;
            }

            BoundaryConditions<DIM> current_bc = ref_bc_;
            ws.cache_waypoints[0] = ref_waypoints_[0];
            ws.cache_waypoints[num_segments_] = ref_waypoints_[num_segments_];

            auto pull_vec = [&](bool flag, VectorType &target)
            { if(flag) { target = x.template segment<DIM>(offset); offset += DIM; } };

            pull_vec(flags_.start_p, ws.cache_waypoints[0]);
            pull_vec(flags_.start_v, current_bc.start_velocity);
            if constexpr (SplineType::ORDER >= 5)
                pull_vec(flags_.start_a, current_bc.start_acceleration);
            if constexpr (SplineType::ORDER >= 7)
                pull_vec(flags_.start_j, current_bc.start_jerk);

            pull_vec(flags_.end_p, ws.cache_waypoints[num_segments_]);
            pull_vec(flags_.end_v, current_bc.end_velocity);
            if constexpr (SplineType::ORDER >= 5)
                pull_vec(flags_.end_a, current_bc.end_acceleration);
            if constexpr (SplineType::ORDER >= 7)
                pull_vec(flags_.end_j, current_bc.end_jerk);

            ws.spline.update(ws.cache_times, ws.cache_waypoints, start_time_, current_bc);

            if (!ws.spline.isInitialized())
                return 1e9;

            ws.cache_gdT.setZero();
            for (int i = 0; i < num_segments_; ++i)
            {
                double g = 0;
                double c = time_cost_func(ws.cache_times[i], g);
                total_cost += c;
                ws.cache_gdT(i) += g;
            }

            ws.cache_gdC.setZero(); 
            calculateIntegralCost(ws, ws.cache_gdC, total_cost, std::forward<SpatialCostFunc>(spatial_cost_func));

            ws.spline.propagateGrad(ws.cache_gdC, ws.cache_gdT, ws.grads);

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
                if constexpr (SplineType::ORDER >= 5)
                    ws.grads.start.a += rho_energy_ * ws.energy_grads.start.a;
                if constexpr (SplineType::ORDER >= 7)
                    ws.grads.start.j += rho_energy_ * ws.energy_grads.start.j;

                ws.grads.end.p += rho_energy_ * ws.energy_grads.end.p;
                ws.grads.end.v += rho_energy_ * ws.energy_grads.end.v;
                if constexpr (SplineType::ORDER >= 5)
                    ws.grads.end.a += rho_energy_ * ws.energy_grads.end.a;
                if constexpr (SplineType::ORDER >= 7)
                    ws.grads.end.j += rho_energy_ * ws.energy_grads.end.j;
            }

            offset = 0;
            for (int i = 0; i < num_segments_; ++i)
            {
                double tau = x(offset); // original unconstrained var
                double T = ws.cache_times[i];
                double gradT = ws.grads.times(i);
                grad_out(offset) = TimeMap::backward(tau, T, gradT);
                offset++;
            }

            for (int i = 0; i < n_inner; ++i)
            {
                grad_out.template segment<DIM>(offset) = ws.grads.inner_points.row(i);
                offset += DIM;
            }

            auto push_grad = [&](bool flag, const VectorType &g)
            { if(flag) { grad_out.template segment<DIM>(offset) = g; offset += DIM; } };

            push_grad(flags_.start_p, ws.grads.start.p);
            push_grad(flags_.start_v, ws.grads.start.v);
            if constexpr (SplineType::ORDER >= 5)
                push_grad(flags_.start_a, ws.grads.start.a);
            if constexpr (SplineType::ORDER >= 7)
                push_grad(flags_.start_j, ws.grads.start.j);

            push_grad(flags_.end_p, ws.grads.end.p);
            push_grad(flags_.end_v, ws.grads.end.v);
            if constexpr (SplineType::ORDER >= 5)
                push_grad(flags_.end_a, ws.grads.end.a);
            if constexpr (SplineType::ORDER >= 7)
                push_grad(flags_.end_j, ws.grads.end.j);

            return total_cost;
        }

        /**
         * @brief Convenience evaluate. Uses internal workspace.
         */
        template <typename TimeCostFunc, typename SpatialCostFunc>
        double evaluate(const Eigen::VectorXd &x, Eigen::VectorXd &grad_out,
                        TimeCostFunc &&time_cost_func, SpatialCostFunc &&spatial_cost_func) const
        {
            if (!internal_ws_)
            {
                internal_ws_ = std::unique_ptr<Workspace>(new Workspace());
            }
            return evaluate(x, grad_out,
                            std::forward<TimeCostFunc>(time_cost_func),
                            std::forward<SpatialCostFunc>(spatial_cost_func),
                            *internal_ws_);
        }

        const SplineType *getOptimalSpline() const
        {
            if (internal_ws_)
                return &(internal_ws_->spline);
            return nullptr;
        }

        template <typename TFunc, typename SFunc>
        bool checkGradients(const Eigen::VectorXd &x, TFunc &&tf, SFunc &&sf)
        {
            std::cout << "[SplineOptimizer] Checking Gradients..." << std::endl;
            Eigen::VectorXd analytical_grad(x.size());
            evaluate(x, analytical_grad, tf, sf);

            Eigen::VectorXd num_grad(x.size());
            Eigen::VectorXd x_p = x, x_m = x;
            double eps = 1e-6;

            for (int i = 0; i < x.size(); ++i)
            {
                x_p(i) += eps;
                x_m(i) -= eps;
                double c_p = evaluate(x_p, num_grad, tf, sf);
                double c_m = evaluate(x_m, num_grad, tf, sf);
                num_grad(i) = (c_p - c_m) / (2 * eps);
                x_p(i) = x(i);
                x_m(i) = x(i);
            }

            double error = (analytical_grad - num_grad).norm();
            std::cout << "Gradient Error Norm: " << error << std::endl;
            if (error > 1e-4)
            {
                std::cout << "Analytical: " << analytical_grad.transpose() << "\n";
                std::cout << "Numerical:  " << num_grad.transpose() << "\n";
                return false;
            }
            return true;
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

        template <typename SpatialFunc>
        void calculateIntegralCost(const Workspace &ws, MatrixType &gdC, double &cost, SpatialFunc &&spatial_cost) const
        {
            const auto &coeffs = ws.spline.getTrajectory().getCoefficients();
            Eigen::Matrix<double, 1, SplineType::COEFF_NUM> b_p, b_v, b_a, b_j, b_s;

            for (int i = 0; i < num_segments_; ++i)
            {
                double T = ws.cache_times[i];
                if (T < 1e-6)
                    continue;

                int n_steps = std::ceil(T / integral_step_);
                double dt = T / n_steps;
                int base_row = i * SplineType::COEFF_NUM;
                auto block = coeffs.block(base_row, 0, SplineType::COEFF_NUM, DIM);

                for (int k = 0; k <= n_steps; ++k)
                {
                    double weight = (k == 0 || k == n_steps) ? 0.5 * dt : dt;
                    double t = k * dt; // t is relative time in segment
                    SplineType::computeBasisFunctions(t, b_p, b_v, b_a, b_j, b_s);

                    VectorType p = (b_p * block).transpose();
                    VectorType v = (b_v * block).transpose();
                    VectorType a = (b_a * block).transpose();
                    VectorType j = (b_j * block).transpose();
                    VectorType s = (b_s * block).transpose();

                    VectorType gp = VectorType::Zero(), gv = VectorType::Zero(), ga = VectorType::Zero(),
                               gj = VectorType::Zero(), gs = VectorType::Zero();

                    double c_val = spatial_cost(t, p, v, a, j, s, gp, gv, ga, gj, gs);

                    if (c_val > 0)
                    {
                        cost += c_val * weight;
                        for (int r = 0; r < SplineType::COEFF_NUM; ++r)
                        {
                            VectorType g = gp * b_p(r) + gv * b_v(r) + ga * b_a(r) + gj * b_j(r) + gs * b_s(r);
                            gdC.row(base_row + r) += g.transpose() * weight;
                        }
                    }
                }
            }
        }
    };
}
#endif