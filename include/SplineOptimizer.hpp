#ifndef SPLINE_OPTIMIZER_HPP
#define SPLINE_OPTIMIZER_HPP

#include "SplineTrajectory.hpp"
#include <algorithm>
#include <vector>
#include <cmath>
#include <iostream>
#include <iomanip>
#include <type_traits> 
#include <utility>   
#include <sstream>  
#include <cassert>

namespace SplineTrajectory
{
    // Protocol reference lives in SplineOptimizerProtocols.md in the same directory.

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

        template <typename T, typename SplineType, int DIM, typename = void>
        struct HasTrajectoryCostInterface : std::false_type {};

        template <typename T, typename SplineType, int DIM>
        struct HasTrajectoryCostInterface<T, SplineType, DIM, void_t<
            decltype(static_cast<double>(std::declval<T>()(
                std::declval<const SplineType &>(),
                std::declval<const std::vector<double> &>(),
                std::declval<const typename SplineType::MatrixType &>(),
                std::declval<double>(),
                std::declval<const BoundaryConditions<DIM> &>(),
                std::declval<typename SplineType::Gradients &>()
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
                std::declval<int>(),                     // step_in_seg
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


        template <typename T, typename SamplesType, typename GradMatrixType, typename = void>
        struct HasSampleCostInterface : std::false_type {};

        template <typename T, typename SamplesType, typename GradMatrixType>
        struct HasSampleCostInterface<T, SamplesType, GradMatrixType, void_t<
            decltype(static_cast<double>(std::declval<T>()(
                std::declval<const SamplesType &>(),
                std::declval<GradMatrixType &>(),
                std::declval<Eigen::VectorXd &>()
            )))
        >> : std::true_type {};

        template <typename T, int DIM, typename SplineType, typename = void>
        struct HasAuxiliaryStateMapInterface : std::false_type {};

        template <typename T, int DIM, typename SplineType>
        struct HasAuxiliaryStateMapInterface<T, DIM, SplineType, void_t<
            decltype(static_cast<int>(std::declval<T>().getDimension())),
            decltype(std::declval<T>().getInitialValue(
                std::declval<const std::vector<double> &>(),
                std::declval<const typename SplineType::MatrixType &>(),
                std::declval<double>(),
                std::declval<const BoundaryConditions<DIM> &>())),
            decltype(std::declval<T>().apply(
                std::declval<const Eigen::VectorXd &>(),
                std::declval<std::vector<double> &>(),
                std::declval<typename SplineType::MatrixType &>(),
                std::declval<double &>(),
                std::declval<BoundaryConditions<DIM> &>())),
            decltype(static_cast<double>(std::declval<T>().backward(
                std::declval<const Eigen::VectorXd &>(),
                std::declval<const SplineType &>(),
                std::declval<const std::vector<double> &>(),
                std::declval<const typename SplineType::MatrixType &>(),
                std::declval<double>(),
                std::declval<const BoundaryConditions<DIM> &>(),
                std::declval<typename SplineType::Gradients &>(),
                std::declval<Eigen::VectorXd &>())))
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

    template <typename SplineType, int DIM>
    struct VoidTrajectoryCost
    {
        double operator()(const SplineType & /*spline*/,
                          const std::vector<double> & /*times*/,
                          const typename SplineType::MatrixType & /*waypoints*/,
                          double /*start_time*/,
                          const BoundaryConditions<DIM> & /*bc*/,
                          typename SplineType::Gradients & /*grads*/) const
        {
            return 0.0;
        }
    };

    template <int DIM, typename SplineType>
    struct VoidAuxiliaryStateMap
    {
        using WaypointsType = typename SplineType::MatrixType;
        using Gradients = typename SplineType::Gradients;

        int getDimension() const { return 0; }

        Eigen::VectorXd getInitialValue(const std::vector<double> & /*ref_times*/,
                                        const WaypointsType & /*ref_waypoints*/,
                                        double /*ref_start_time*/,
                                        const BoundaryConditions<DIM> & /*ref_bc*/) const
        {
            return Eigen::VectorXd();
        }

        void apply(const Eigen::VectorXd & /*z*/,
                   std::vector<double> & /*times*/,
                   WaypointsType & /*waypoints*/,
                   double & /*start_time*/,
                   BoundaryConditions<DIM> & /*bc*/) const
        {
        }

        double backward(const Eigen::VectorXd & /*z*/,
                        const SplineType & /*spline*/,
                        const std::vector<double> & /*times*/,
                        const WaypointsType & /*waypoints*/,
                        double /*start_time*/,
                        const BoundaryConditions<DIM> & /*bc*/,
                        Gradients & /*grads*/,
                        Eigen::VectorXd &grad_z) const
        {
            grad_z.resize(0);
            return 0.0;
        }
    };

    struct BoundaryDerivativeMask
    {
        bool v = false;
        bool a = false;
        bool j = false;
    };

    struct OptimizationMask
    {
        std::vector<uint8_t> time;
        std::vector<uint8_t> waypoints;
        BoundaryDerivativeMask start;
        BoundaryDerivativeMask end;
    };

    template <int DIM,
              typename SplineType = QuinticSplineND<DIM>,
              typename TimeMap = QuadInvTimeMap,
              typename SpatialMap = IdentitySpatialMap<DIM>,
              typename AuxiliaryStateMap = VoidAuxiliaryStateMap<DIM, SplineType>>
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

        static_assert(TypeTraits::HasAuxiliaryStateMapInterface<AuxiliaryStateMap, DIM, SplineType>::value,
                      "\n[SplineOptimizer Error] The provided 'AuxiliaryStateMap' type does not satisfy the required interface.\n"
                      "It must implement getDimension, getInitialValue, apply, and backward methods.\n");

    public:
        using VectorType = typename SplineType::VectorType;
        using MatrixType = typename SplineType::MatrixType;
        using WaypointsType = MatrixType;
        using SampleGradMatrix = Eigen::Matrix<double, DIM, Eigen::Dynamic>;

        struct IntegralSample
        {
            EIGEN_MAKE_ALIGNED_OPERATOR_NEW

            int seg_idx = 0;
            int step_in_seg = 0;
            int sample_buffer_index = 0;
            double alpha = 0.0;
            double t_local = 0.0;
            double t_global = 0.0;
            double trap_weight = 0.0;
            double dt = 0.0;
            Eigen::Matrix<double, 1, SplineType::COEFF_NUM> b_p;
            VectorType p = VectorType::Zero();
            VectorType v = VectorType::Zero();
        };

        using IntegralSampleBuffer = std::vector<IntegralSample, Eigen::aligned_allocator<IntegralSample>>;

        struct VoidSampleCost
        {
            template <typename SamplesType>
            double operator()(const SamplesType & /*samples*/,
                              SampleGradMatrix & /*grad_p*/,
                              Eigen::VectorXd & /*grad_t_global*/) const
            {
                return 0.0;
            }
        };

        struct WorkingState;
        struct EvaluationBuffers;
        struct Workspace;

        template <typename TimeCostFunc,
                  typename IntegralCostFunc,
                  typename WaypointsCostFunc = VoidWaypointsCost,
                  typename SampleCostFunc = VoidSampleCost,
                  typename TrajectoryCostFunc = VoidTrajectoryCost<SplineType, DIM>,
                  typename Executor = SerialExecutor>
        struct EvaluateSpec
        {
            const TimeCostFunc *time_cost = nullptr;
            const IntegralCostFunc *integral_cost = nullptr;
            const WaypointsCostFunc *waypoints_cost = nullptr;
            const SampleCostFunc *sample_cost = nullptr;
            const TrajectoryCostFunc *trajectory_cost = nullptr;
            Workspace *workspace = nullptr;
            Executor executor{};
        };

        template <typename TimeCostFunc, typename IntegralCostFunc,
                  typename Executor = SerialExecutor>
        static EvaluateSpec<TimeCostFunc, IntegralCostFunc, VoidWaypointsCost,
                            VoidSampleCost, VoidTrajectoryCost<SplineType, DIM>, Executor>
        makeEvaluateSpec(const TimeCostFunc &time_cost,
                         const IntegralCostFunc &integral_cost,
                         Workspace &workspace,
                         const Executor &executor = Executor())
        {
            EvaluateSpec<TimeCostFunc, IntegralCostFunc, VoidWaypointsCost,
                         VoidSampleCost, VoidTrajectoryCost<SplineType, DIM>, Executor> spec;
            spec.time_cost = &time_cost;
            spec.integral_cost = &integral_cost;
            spec.workspace = &workspace;
            spec.executor = executor;
            return spec;
        }

        /**
         * @brief Workspace holds all mutable state required during optimization.
         * Callers must provide one workspace per evaluation context.
         */
        struct WorkingState
        {
            EIGEN_MAKE_ALIGNED_OPERATOR_NEW

            SplineType spline;
            std::vector<double> times;
            WaypointsType waypoints;
            BoundaryConditions<DIM> bc;
            double start_time = 0.0;
            Eigen::VectorXd auxiliary_vars;

            void resize(int num_segments)
            {
                if (static_cast<int>(times.size()) != num_segments)
                {
                    times.resize(num_segments);
                    waypoints.resize(num_segments + 1, DIM);
                    auxiliary_vars.resize(0);
                }
            }
        };

        struct EvaluationBuffers
        {
            EIGEN_MAKE_ALIGNED_OPERATOR_NEW

            Eigen::VectorXd grad_times;
            MatrixType grad_coeffs;
            Eigen::VectorXd time_cost_grad_buffer;

            typename SplineType::Gradients grads;
            typename SplineType::Gradients energy_grads;

            Eigen::VectorXd global_time_grad_buffer;
            MatrixType waypoint_grad_buffer;
            SampleGradMatrix sample_position_grad_buffer;
            Eigen::VectorXd sample_time_grad_buffer;
            std::vector<double> segment_begin_times;
            std::vector<double> segment_cost_buffer;
            IntegralSampleBuffer integral_samples;
            bool record_integral_samples = false;

            void resize(int num_segments)
            {
                if (grad_times.size() != num_segments)
                {
                    grad_times.resize(num_segments);
                    time_cost_grad_buffer.resize(num_segments);
                    grad_coeffs.resize(num_segments * SplineType::COEFF_NUM, DIM);
                    global_time_grad_buffer.resize(num_segments);
                    waypoint_grad_buffer.resize(num_segments + 1, DIM);
                    sample_position_grad_buffer.resize(DIM, 0);
                    sample_time_grad_buffer.resize(0);
                    segment_begin_times.resize(num_segments);
                    segment_cost_buffer.resize(num_segments);
                    integral_samples.clear();
                }
            }
        };

        struct Workspace
        {
            EIGEN_MAKE_ALIGNED_OPERATOR_NEW

            WorkingState state;
            EvaluationBuffers buffers;

            void resize(int num_segments)
            {
                state.resize(num_segments);
                buffers.resize(num_segments);
            }
        };

        enum class ErrorCode
        {
            None = 0,
            ValidationFailed,
            InvalidOptimizerState,
            InvalidIntegralSteps,
            DimensionMismatch,
            NullWorkspace,
            NullTimeCost,
            NullIntegralCost,
            NullWaypointsCost,
            NullSampleCost,
            NullTrajectoryCost
        };

        struct ResultBase
        {
            bool ok = false;
            ErrorCode code = ErrorCode::None;
            std::string message;

            explicit operator bool() const { return ok; }
        };

        struct Status : ResultBase
        {
        };

        struct EvaluationResult : ResultBase
        {
            double cost = 0.0;
        };

    private:
        struct TimeVariableLayout
        {
            int segment_index = 0;
            int offset = 0;
        };

        struct PointVariableLayout
        {
            int point_index = 0;
            int offset = 0;
            int dof = 0;
        };

        enum class BoundaryDerivativeSlot
        {
            StartV,
            StartA,
            StartJ,
            EndV,
            EndA,
            EndJ
        };

        struct DecisionVariableLayout
        {
            std::vector<TimeVariableLayout> time;
            std::vector<PointVariableLayout> waypoints;
            int boundary_derivatives_offset = 0;
            int auxiliary_offset = 0;
            int total_dimension = 0;
        };

        std::vector<double> ref_times_;
        WaypointsType ref_waypoints_;
        BoundaryConditions<DIM> ref_bc_;
        double start_time_ = 0.0;

        OptimizationMask optimization_mask_;
        bool has_explicit_time_mask_ = false;
        bool has_explicit_waypoint_mask_ = false;
        int num_segments_ = 0;
        bool is_valid_ = false;

        double rho_energy_ = 0.0;
        int integral_num_steps_ = 64;

        TimeMap default_time_map_;
        SpatialMap default_spatial_map_;
        AuxiliaryStateMap default_auxiliary_state_map_;

        const TimeMap* active_time_map_ = nullptr;
        const SpatialMap* active_spatial_map_ = nullptr;
        const AuxiliaryStateMap* active_auxiliary_state_map_ = nullptr;

        mutable DecisionVariableLayout layout_;
        mutable bool layout_dirty_ = true;
        
        void markLayoutDirty()
        {
            layout_dirty_ = true;
        }

        void normalizeOptimizationMask()
        {
            if (num_segments_ <= 0)
            {
                return;
            }

            if (!has_explicit_time_mask_)
            {
                optimization_mask_.time.assign(num_segments_, static_cast<uint8_t>(1));
            }

            if (!has_explicit_waypoint_mask_)
            {
                optimization_mask_.waypoints.assign(num_segments_ + 1, static_cast<uint8_t>(0));
                for (int i = 1; i < num_segments_; ++i)
                {
                    optimization_mask_.waypoints[i] = static_cast<uint8_t>(1);
                }
            }
        }

        bool isTimeOptimized(int idx) const
        {
            assert(static_cast<size_t>(num_segments_) == optimization_mask_.time.size());
            return idx >= 0 &&
                   idx < static_cast<int>(optimization_mask_.time.size()) &&
                   optimization_mask_.time[idx] != 0;
        }

        bool isWaypointOptimized(int idx) const
        {
            assert(static_cast<size_t>(num_segments_ + 1) == optimization_mask_.waypoints.size());
            return idx >= 0 &&
                   idx < static_cast<int>(optimization_mask_.waypoints.size()) &&
                   optimization_mask_.waypoints[idx] != 0;
        }

        template <typename Fn>
        void forEachOptimizedBoundaryDerivativeSlot(Fn &&fn) const
        {
            if (optimization_mask_.start.v) fn(BoundaryDerivativeSlot::StartV);
            if constexpr (SplineType::ORDER >= 5)
            {
                if (optimization_mask_.start.a) fn(BoundaryDerivativeSlot::StartA);
            }
            if constexpr (SplineType::ORDER >= 7)
            {
                if (optimization_mask_.start.j) fn(BoundaryDerivativeSlot::StartJ);
            }

            if (optimization_mask_.end.v) fn(BoundaryDerivativeSlot::EndV);
            if constexpr (SplineType::ORDER >= 5)
            {
                if (optimization_mask_.end.a) fn(BoundaryDerivativeSlot::EndA);
            }
            if constexpr (SplineType::ORDER >= 7)
            {
                if (optimization_mask_.end.j) fn(BoundaryDerivativeSlot::EndJ);
            }
        }

        int countOptimizedDerivativeBlocks() const
        {
            int blocks = 0;
            if (optimization_mask_.start.v) ++blocks;
            if constexpr (SplineType::ORDER >= 5)
            {
                if (optimization_mask_.start.a) ++blocks;
            }
            if constexpr (SplineType::ORDER >= 7)
            {
                if (optimization_mask_.start.j) ++blocks;
            }

            if (optimization_mask_.end.v) ++blocks;
            if constexpr (SplineType::ORDER >= 5)
            {
                if (optimization_mask_.end.a) ++blocks;
            }
            if constexpr (SplineType::ORDER >= 7)
            {
                if (optimization_mask_.end.j) ++blocks;
            }
            return blocks;
        }

        void rebuildLayoutCache() const
        {
            layout_.time.clear();
            layout_.waypoints.clear();

            if (num_segments_ <= 0)
            {
                layout_.boundary_derivatives_offset = 0;
                layout_.auxiliary_offset = 0;
                layout_.total_dimension = 0;
                layout_dirty_ = false;
                return;
            }

            int offset = 0;
            for (int i = 0; i < num_segments_; ++i)
            {
                if (!isTimeOptimized(i))
                {
                    continue;
                }

                layout_.time.push_back(TimeVariableLayout{i, offset});
                ++offset;
            }

            for (int i = 0; i <= num_segments_; ++i)
            {
                if (!isWaypointOptimized(i))
                {
                    continue;
                }

                const int dof = active_spatial_map_->getUnconstrainedDim(i);
                layout_.waypoints.push_back(PointVariableLayout{i, offset, dof});
                offset += dof;
            }

            layout_.boundary_derivatives_offset = offset;
            layout_.auxiliary_offset = layout_.boundary_derivatives_offset + countOptimizedDerivativeBlocks() * DIM;
            layout_.total_dimension = layout_.auxiliary_offset + active_auxiliary_state_map_->getDimension();
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
            active_auxiliary_state_map_ = &default_auxiliary_state_map_;
        }

        SplineOptimizer(const SplineOptimizer &) = delete;
        SplineOptimizer &operator=(const SplineOptimizer &) = delete;
        SplineOptimizer(SplineOptimizer &&) = delete;
        SplineOptimizer &operator=(SplineOptimizer &&) = delete;

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
         * @brief Set the AuxiliaryStateMap for optional extra optimization variables.
         * @param map Pointer to an AuxiliaryStateMap instance (can be nullptr to reset to default).
         * The optimizer does not take ownership; the map must remain valid.
         */
        void setAuxiliaryStateMap(const AuxiliaryStateMap* map)
        {
            active_auxiliary_state_map_ = (map != nullptr) ? map : &default_auxiliary_state_map_;
            markLayoutDirty();
        }

        /**
         * @brief Initialize using Absolute Time Points.
         * Converts time points to segments.
         * @return Validation status for the new reference state.
         */
        Status setInitState(const std::vector<double> &t_points,
                            const WaypointsType &waypoints,
                            const BoundaryConditions<DIM> &bc)
        {
            if (t_points.empty())
            {
                is_valid_ = false;
                return makeErrorStatus(ErrorCode::ValidationFailed,
                                       "Input time points vector is empty");
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
         * @return Validation status for the new reference state.
         */
        Status setInitState(const std::vector<double> &time_segments,
                            const WaypointsType &waypoints,
                            double start_time,
                            const BoundaryConditions<DIM> &bc)
        {
            start_time_ = start_time;
            ref_times_ = time_segments;
            ref_waypoints_ = waypoints;
            ref_bc_ = bc;
            num_segments_ = static_cast<int>(ref_times_.size());
            normalizeOptimizationMask();
            markLayoutDirty();

            const Status status = checkValidity();
            is_valid_ = status.ok;
            return status;
        }

        static OptimizationMask makeFullOptimizationMask(int num_segments)
        {
            OptimizationMask mask;
            mask.time.assign(num_segments, static_cast<uint8_t>(1));
            mask.waypoints.assign(num_segments + 1, static_cast<uint8_t>(1));
            return mask;
        }

        Status setOptimizationMask(const OptimizationMask &mask)
        {
            optimization_mask_ = mask;
            has_explicit_time_mask_ = !mask.time.empty();
            has_explicit_waypoint_mask_ = !mask.waypoints.empty();
            normalizeOptimizationMask();
            markLayoutDirty();

            if (num_segments_ <= 0)
            {
                is_valid_ = false;
                return makeOkStatus();
            }

            const Status status = checkValidity();
            is_valid_ = status.ok;
            return status;
        }

        const OptimizationMask &getOptimizationMask() const { return optimization_mask_; }
        void setEnergyWeights(double rho_energy) { rho_energy_ = rho_energy; }
        Status setIntegralNumSteps(int steps)
        {
            if (steps <= 0)
            {
                return makeErrorStatus(ErrorCode::InvalidIntegralSteps,
                                       "[SplineOptimizer Error] integral_num_steps must be positive.");
            }
            integral_num_steps_ = steps;
            return makeOkStatus();
        }

        void setRecordIntegralSamples(bool enable, Workspace &workspace) const
        {
            workspace.buffers.record_integral_samples = enable;
        }

        const IntegralSampleBuffer &getRecordedIntegralSamples(const Workspace &workspace) const
        {
            return workspace.buffers.integral_samples;
        }

        /**
         * @brief Check if the current optimization state is valid.
         * @return true if valid, false otherwise.
         */
        bool isValid() const { return is_valid_; }

        /**
         * @brief Convertible to bool to check validity.
         */
        explicit operator bool() const { return is_valid_; }
        
        /**
         * @brief Perform a thorough validity check on segments, times, waypoints and BCs.
         * Aggregates ALL errors instead of stopping at the first one.
         * @return Validation status with aggregated diagnostics.
         */
        Status checkValidity() const
        {
            return validateConfiguration(true);
        }

        int getDimension() const { return calculateDimension(); }

        /**
         * @brief Generate initial guess x based on reference state.
         * Applies 'toUnconstrained' mapping using unified layout logic.
         */
        Eigen::VectorXd generateInitialGuess() const
        {
            return encodeReferenceState();
        }

        Eigen::VectorXd encodeReferenceState() const
        {
            const Eigen::VectorXd auxiliary_initial =
                active_auxiliary_state_map_->getInitialValue(ref_times_, ref_waypoints_, start_time_, ref_bc_);
            return encodeMaskedDecisionVariables(ref_times_, ref_waypoints_, ref_bc_, auxiliary_initial);
        }

        Status encodeWorkingState(const Workspace &workspace, Eigen::VectorXd &x_out) const
        {
            const auto &state = workspace.state;
            if (state.times.size() != ref_times_.size() ||
                state.waypoints.rows() != ref_waypoints_.rows())
            {
                return makeErrorStatus(
                    ErrorCode::InvalidOptimizerState,
                    "[SplineOptimizer Error] Workspace state is not initialized or does not match the current reference state.");
            }

            Eigen::VectorXd auxiliary_vars = state.auxiliary_vars;
            const int aux_dim = active_auxiliary_state_map_->getDimension();
            if (aux_dim <= 0)
            {
                auxiliary_vars.resize(0);
            }
            else if (auxiliary_vars.size() != aux_dim)
            {
                return makeErrorStatus(
                    ErrorCode::InvalidOptimizerState,
                    "[SplineOptimizer Error] Workspace auxiliary variable dimension does not match the active AuxiliaryStateMap.");
            }

            x_out = encodeMaskedDecisionVariables(workspace.state.times,
                                                  workspace.state.waypoints,
                                                  workspace.state.bc,
                                                  auxiliary_vars);
            return makeOkStatus();
        }

    private:
        template <typename TimeCostFunc,
                  typename IntegralCostFunc,
                  typename WaypointsCostFunc,
                  typename SampleCostFunc,
                  typename TrajectoryCostFunc,
                  typename Executor>
        struct ResolvedEvaluateSpec
        {
            const TimeCostFunc &time_cost;
            const IntegralCostFunc &integral_cost;
            const WaypointsCostFunc &waypoints_cost;
            const SampleCostFunc &sample_cost;
            const TrajectoryCostFunc &trajectory_cost;
            Workspace &workspace;
            const Executor &executor;
        };

        template <typename TimeCostFunc,
                  typename IntegralCostFunc,
                  typename WaypointsCostFunc,
                  typename SampleCostFunc,
                  typename TrajectoryCostFunc,
                  typename Executor>
        ResolvedEvaluateSpec<TimeCostFunc, IntegralCostFunc,
                             WaypointsCostFunc, SampleCostFunc,
                             TrajectoryCostFunc, Executor>
        resolveEvaluateSpec(
            const EvaluateSpec<TimeCostFunc, IntegralCostFunc,
                               WaypointsCostFunc, SampleCostFunc,
                               TrajectoryCostFunc, Executor> &spec) const
        {
            return {
                *spec.time_cost,
                *spec.integral_cost,
                resolveWaypointsCost(spec.waypoints_cost),
                resolveSampleCost(spec.sample_cost),
                resolveTrajectoryCost(spec.trajectory_cost),
                *spec.workspace,
                spec.executor
            };
        }

        Eigen::VectorXd encodeMaskedDecisionVariables(const std::vector<double> &times,
                                                      const WaypointsType &waypoints,
                                                      const BoundaryConditions<DIM> &bc,
                                                      const Eigen::VectorXd &auxiliary_vars) const
        {
            ensureLayoutCache();
            Eigen::VectorXd x = Eigen::VectorXd::Zero(layout_.total_dimension);

            for (const auto &var : layout_.time)
            {
                x(var.offset) = active_time_map_->toTau(times[var.segment_index]);
            }

            for (const auto &var : layout_.waypoints)
            {
                x.segment(var.offset, var.dof) =
                    active_spatial_map_->toUnconstrained(waypoints.row(var.point_index).transpose(), var.point_index);
            }

            int offset = layout_.boundary_derivatives_offset;
            forEachOptimizedBoundaryDerivativeSlot([&](BoundaryDerivativeSlot slot) {
                switch (slot)
                {
                    case BoundaryDerivativeSlot::StartV:
                        x.template segment<DIM>(offset) = bc.start_velocity;
                        break;
                    case BoundaryDerivativeSlot::StartA:
                        if constexpr (SplineType::ORDER >= 5)
                        {
                            x.template segment<DIM>(offset) = bc.start_acceleration;
                        }
                        break;
                    case BoundaryDerivativeSlot::StartJ:
                        if constexpr (SplineType::ORDER >= 7)
                        {
                            x.template segment<DIM>(offset) = bc.start_jerk;
                        }
                        break;
                    case BoundaryDerivativeSlot::EndV:
                        x.template segment<DIM>(offset) = bc.end_velocity;
                        break;
                    case BoundaryDerivativeSlot::EndA:
                        if constexpr (SplineType::ORDER >= 5)
                        {
                            x.template segment<DIM>(offset) = bc.end_acceleration;
                        }
                        break;
                    case BoundaryDerivativeSlot::EndJ:
                        if constexpr (SplineType::ORDER >= 7)
                        {
                            x.template segment<DIM>(offset) = bc.end_jerk;
                        }
                        break;
                }
                offset += DIM;
            });

            const int aux_dim = active_auxiliary_state_map_->getDimension();
            if (aux_dim > 0)
            {
                assert(auxiliary_vars.size() == aux_dim &&
                       "[SplineOptimizer Error] Auxiliary variable dimension mismatch during encoding.");
                x.segment(layout_.auxiliary_offset, aux_dim) = auxiliary_vars;
            }

            return x;
        }

        void decodeMaskedDecisionVariables(const Eigen::VectorXd &x, WorkingState &state) const
        {
            state.times = ref_times_;
            for (const auto &var : layout_.time)
            {
                state.times[var.segment_index] = active_time_map_->toTime(x(var.offset));
            }

            state.waypoints = ref_waypoints_;
            for (const auto &var : layout_.waypoints)
            {
                state.waypoints.row(var.point_index) =
                    active_spatial_map_->toPhysical(x.segment(var.offset, var.dof), var.point_index).transpose();
            }

            state.bc = ref_bc_;
            state.start_time = start_time_;

            int offset = layout_.boundary_derivatives_offset;
            forEachOptimizedBoundaryDerivativeSlot([&](BoundaryDerivativeSlot slot) {
                switch (slot)
                {
                    case BoundaryDerivativeSlot::StartV:
                        state.bc.start_velocity = x.template segment<DIM>(offset);
                        break;
                    case BoundaryDerivativeSlot::StartA:
                        if constexpr (SplineType::ORDER >= 5)
                        {
                            state.bc.start_acceleration = x.template segment<DIM>(offset);
                        }
                        break;
                    case BoundaryDerivativeSlot::StartJ:
                        if constexpr (SplineType::ORDER >= 7)
                        {
                            state.bc.start_jerk = x.template segment<DIM>(offset);
                        }
                        break;
                    case BoundaryDerivativeSlot::EndV:
                        state.bc.end_velocity = x.template segment<DIM>(offset);
                        break;
                    case BoundaryDerivativeSlot::EndA:
                        if constexpr (SplineType::ORDER >= 5)
                        {
                            state.bc.end_acceleration = x.template segment<DIM>(offset);
                        }
                        break;
                    case BoundaryDerivativeSlot::EndJ:
                        if constexpr (SplineType::ORDER >= 7)
                        {
                            state.bc.end_jerk = x.template segment<DIM>(offset);
                        }
                        break;
                }
                offset += DIM;
            });
        }

        void applyAuxiliaryVariables(const Eigen::VectorXd &x, WorkingState &state) const
        {
            const int auxiliary_dim = active_auxiliary_state_map_->getDimension();
            if (auxiliary_dim > 0)
            {
                state.auxiliary_vars = x.segment(layout_.auxiliary_offset, auxiliary_dim);
                active_auxiliary_state_map_->apply(state.auxiliary_vars,
                                                   state.times,
                                                   state.waypoints,
                                                   state.start_time,
                                                   state.bc);
            }
            else
            {
                state.auxiliary_vars.resize(0);
            }
        }

        void updateWorkingSpline(WorkingState &state) const
        {
            state.spline.update(state.times, state.waypoints, state.start_time, state.bc);
        }

        void decodeAndBuildWorkingState(const Eigen::VectorXd &x, Workspace &ws) const
        {
            decodeMaskedDecisionVariables(x, ws.state);
            applyAuxiliaryVariables(x, ws.state);
            updateWorkingSpline(ws.state);
        }

        void resetEvaluationState(Eigen::Index gradient_size,
                                  Eigen::VectorXd &grad_out,
                                  Workspace &ws) const
        {
            ws.resize(num_segments_);
            grad_out.setZero(gradient_size);
        }

        template <typename TimeCostFunc>
        double accumulateTimeCost(const TimeCostFunc &time_cost, Workspace &ws) const
        {
            auto &state = ws.state;
            auto &buffers = ws.buffers;
            buffers.time_cost_grad_buffer.setZero();
            buffers.grad_times.setZero();

            double total_cost = time_cost(state.times, buffers.time_cost_grad_buffer);
            buffers.grad_times += buffers.time_cost_grad_buffer;
            return total_cost;
        }

        template <typename IntegralCostFunc, typename SampleCostFunc, typename Executor>
        double accumulateIntegralAndSampleCosts(const IntegralCostFunc &integral_cost,
                                                const SampleCostFunc &sample_cost,
                                                Workspace &ws,
                                                const Executor &executor) const
        {
            using SampleCost = typename std::decay<SampleCostFunc>::type;
            auto &state = ws.state;
            auto &buffers = ws.buffers;

            double total_cost = 0.0;
            buffers.grad_coeffs.setZero();

            const bool need_integral_samples =
                buffers.record_integral_samples || !std::is_same_v<SampleCost, VoidSampleCost>;
            accumulateIntegralCost(ws, buffers.grad_coeffs, buffers.grad_times, total_cost,
                                   integral_cost,
                                   state.start_time,
                                   need_integral_samples,
                                   executor);

            if constexpr (!std::is_same_v<SampleCost, VoidSampleCost>)
            {
                const Eigen::Index sample_count =
                    static_cast<Eigen::Index>(buffers.integral_samples.size());

                buffers.sample_position_grad_buffer.resize(DIM, sample_count);
                buffers.sample_position_grad_buffer.setZero();
                buffers.sample_time_grad_buffer.resize(sample_count);
                buffers.sample_time_grad_buffer.setZero();

                total_cost += sample_cost(buffers.integral_samples,
                                          buffers.sample_position_grad_buffer,
                                          buffers.sample_time_grad_buffer);

                accumulateSampleCostGradients(buffers.integral_samples,
                                              buffers.sample_position_grad_buffer,
                                              buffers.sample_time_grad_buffer,
                                              buffers.grad_coeffs,
                                              buffers.grad_times);
            }

            return total_cost;
        }

        template <typename TrajectoryCostFunc>
        double accumulateTrajectoryCost(const TrajectoryCostFunc &trajectory_cost, Workspace &ws) const
        {
            auto &state = ws.state;
            auto &buffers = ws.buffers;
            return trajectory_cost(state.spline,
                                   state.times,
                                   state.waypoints,
                                   state.start_time,
                                   state.bc,
                                   buffers.grads);
        }

        template <typename WaypointsCostFunc>
        double accumulateWaypointCost(const WaypointsCostFunc &waypoints_cost, Workspace &ws) const
        {
            using WaypointsCost = typename std::decay<WaypointsCostFunc>::type;
            auto &state = ws.state;
            auto &buffers = ws.buffers;
            if constexpr (std::is_same_v<WaypointsCost, VoidWaypointsCost>)
            {
                return 0.0;
            }
            else
            {
                const int num_inner_points = std::max(0, num_segments_ - 1);

                buffers.waypoint_grad_buffer.setZero();
                const double waypoint_cost_value =
                    waypoints_cost(state.waypoints, buffers.waypoint_grad_buffer);

                buffers.grads.start.p += buffers.waypoint_grad_buffer.row(0).transpose();
                if (num_inner_points > 0)
                {
                    buffers.grads.inner_points += buffers.waypoint_grad_buffer.block(1, 0, num_inner_points, DIM);
                }
                buffers.grads.end.p += buffers.waypoint_grad_buffer.row(num_segments_).transpose();

                return waypoint_cost_value;
            }
        }

        double accumulateEnergyCost(Workspace &ws) const
        {
            auto &state = ws.state;
            auto &buffers = ws.buffers;
            if (rho_energy_ <= 0.0)
            {
                return 0.0;
            }

            const int num_inner_points = std::max(0, num_segments_ - 1);
            const double energy = state.spline.getEnergy();
            state.spline.getEnergyGrad(buffers.energy_grads);

            buffers.grads.times += rho_energy_ * buffers.energy_grads.times;
            if (num_inner_points > 0)
            {
                buffers.grads.inner_points += rho_energy_ * buffers.energy_grads.inner_points;
            }

            buffers.grads.start.p += rho_energy_ * buffers.energy_grads.start.p;
            buffers.grads.start.v += rho_energy_ * buffers.energy_grads.start.v;
            if constexpr (SplineType::ORDER >= 5) buffers.grads.start.a += rho_energy_ * buffers.energy_grads.start.a;
            if constexpr (SplineType::ORDER >= 7) buffers.grads.start.j += rho_energy_ * buffers.energy_grads.start.j;

            buffers.grads.end.p += rho_energy_ * buffers.energy_grads.end.p;
            buffers.grads.end.v += rho_energy_ * buffers.energy_grads.end.v;
            if constexpr (SplineType::ORDER >= 5) buffers.grads.end.a += rho_energy_ * buffers.energy_grads.end.a;
            if constexpr (SplineType::ORDER >= 7) buffers.grads.end.j += rho_energy_ * buffers.energy_grads.end.j;

            return rho_energy_ * energy;
        }

        double backpropagateAuxiliaryGradient(Eigen::VectorXd &grad_out, Workspace &ws) const
        {
            const int auxiliary_dim = active_auxiliary_state_map_->getDimension();
            if (auxiliary_dim <= 0)
            {
                return 0.0;
            }

            auto &state = ws.state;
            auto &buffers = ws.buffers;
            Eigen::VectorXd grad_aux;
            const double auxiliary_cost = active_auxiliary_state_map_->backward(state.auxiliary_vars,
                                                                                state.spline,
                                                                                state.times,
                                                                                state.waypoints,
                                                                                state.start_time,
                                                                                state.bc,
                                                                                buffers.grads,
                                                                                grad_aux);
            if (grad_aux.size() != auxiliary_dim)
            {
                assert(false && "[SplineOptimizer Error] Auxiliary gradient dimension mismatch.");
                grad_aux.conservativeResize(auxiliary_dim);
                grad_aux.setZero();
            }
            grad_out.segment(layout_.auxiliary_offset, auxiliary_dim) = grad_aux;
            return auxiliary_cost;
        }

        void propagateSplineGradients(Workspace &ws) const
        {
            ws.state.spline.propagateGrad(ws.buffers.grad_coeffs, ws.buffers.grad_times, ws.buffers.grads);
        }

        void writeDecisionGradient(const Eigen::VectorXd &x,
                                   Eigen::VectorXd &grad_out,
                                   const Workspace &ws) const
        {
            const auto &state = ws.state;
            const auto &buffers = ws.buffers;

            for (const auto &var : layout_.time)
            {
                const double tau = x(var.offset);
                const double T = state.times[var.segment_index];
                const double gradT = buffers.grads.times(var.segment_index);
                grad_out(var.offset) = active_time_map_->backward(tau, T, gradT);
            }

            for (const auto &var : layout_.waypoints)
            {
                const auto xi = x.segment(var.offset, var.dof);

                if (var.point_index == 0)
                {
                    grad_out.segment(var.offset, var.dof) =
                        active_spatial_map_->backwardGrad(xi, buffers.grads.start.p, 0);
                }
                else if (var.point_index == num_segments_)
                {
                    grad_out.segment(var.offset, var.dof) =
                        active_spatial_map_->backwardGrad(xi, buffers.grads.end.p, var.point_index);
                }
                else
                {
                    VectorType grad_inner_point = buffers.grads.inner_points.row(var.point_index - 1).transpose();
                    grad_out.segment(var.offset, var.dof) =
                        active_spatial_map_->backwardGrad(xi, grad_inner_point, var.point_index);
                }
            }

            int offset = layout_.boundary_derivatives_offset;
            forEachOptimizedBoundaryDerivativeSlot([&](BoundaryDerivativeSlot slot) {
                switch (slot)
                {
                    case BoundaryDerivativeSlot::StartV:
                        grad_out.template segment<DIM>(offset) = buffers.grads.start.v;
                        break;
                    case BoundaryDerivativeSlot::StartA:
                        if constexpr (SplineType::ORDER >= 5)
                        {
                            grad_out.template segment<DIM>(offset) = buffers.grads.start.a;
                        }
                        break;
                    case BoundaryDerivativeSlot::StartJ:
                        if constexpr (SplineType::ORDER >= 7)
                        {
                            grad_out.template segment<DIM>(offset) = buffers.grads.start.j;
                        }
                        break;
                    case BoundaryDerivativeSlot::EndV:
                        grad_out.template segment<DIM>(offset) = buffers.grads.end.v;
                        break;
                    case BoundaryDerivativeSlot::EndA:
                        if constexpr (SplineType::ORDER >= 5)
                        {
                            grad_out.template segment<DIM>(offset) = buffers.grads.end.a;
                        }
                        break;
                    case BoundaryDerivativeSlot::EndJ:
                        if constexpr (SplineType::ORDER >= 7)
                        {
                            grad_out.template segment<DIM>(offset) = buffers.grads.end.j;
                        }
                        break;
                }
                offset += DIM;
            });
        }

        template <typename TimeCostFunc,
                  typename IntegralCostFunc,
                  typename SampleCostFunc,
                  typename Executor>
        double accumulateForwardCosts(const TimeCostFunc &time_cost,
                                      const IntegralCostFunc &integral_cost,
                                      const SampleCostFunc &sample_cost,
                                      Workspace &ws,
                                      const Executor &executor) const
        {
            double total_cost = 0.0;
            total_cost += accumulateTimeCost(time_cost, ws);
            total_cost += accumulateIntegralAndSampleCosts(integral_cost, sample_cost, ws, executor);
            return total_cost;
        }

        template <typename WaypointsCostFunc,
                  typename TrajectoryCostFunc>
        double accumulatePostPropagationCosts(const WaypointsCostFunc &waypoints_cost,
                                             const TrajectoryCostFunc &trajectory_cost,
                                             Eigen::VectorXd &grad_out,
                                             Workspace &ws) const
        {
            double total_cost = 0.0;
            total_cost += accumulateTrajectoryCost(trajectory_cost, ws);
            total_cost += accumulateWaypointCost(waypoints_cost, ws);
            total_cost += accumulateEnergyCost(ws);
            total_cost += backpropagateAuxiliaryGradient(grad_out, ws);
            return total_cost;
        }

        template <typename TimeCostFunc,
                  typename IntegralCostFunc,
                  typename WaypointsCostFunc,
                  typename SampleCostFunc,
                  typename TrajectoryCostFunc,
                  typename Executor>
        double runEvaluation(
            const Eigen::VectorXd &x,
            Eigen::VectorXd &grad_out,
            const ResolvedEvaluateSpec<TimeCostFunc, IntegralCostFunc,
                                       WaypointsCostFunc, SampleCostFunc,
                                       TrajectoryCostFunc, Executor> &spec) const
        {
            using TimeCost = typename std::decay<TimeCostFunc>::type;
            using IntegralCost = typename std::decay<IntegralCostFunc>::type;
            using WaypointsCost = typename std::decay<WaypointsCostFunc>::type;
            using SampleCost = typename std::decay<SampleCostFunc>::type;
            using TrajectoryCost = typename std::decay<TrajectoryCostFunc>::type;

            static_assert(TypeTraits::HasTimeCostInterface<TimeCost>::value,
                          "[SplineOptimizer Error] 'TimeCostFunc' signature mismatch.");
            static_assert(TypeTraits::HasWaypointsCostInterface<WaypointsCost, WaypointsType>::value,
                          "[SplineOptimizer Error] 'WaypointsCostFunc' signature mismatch.");
            static_assert(TypeTraits::HasIntegralCostInterface<IntegralCost, VectorType>::value,
                          "[SplineOptimizer Error] 'IntegralCostFunc' signature mismatch.");
            static_assert(TypeTraits::HasSampleCostInterface<SampleCost, IntegralSampleBuffer, SampleGradMatrix>::value,
                          "[SplineOptimizer Error] 'SampleCostFunc' signature mismatch.");
            static_assert(TypeTraits::HasTrajectoryCostInterface<TrajectoryCost, SplineType, DIM>::value,
                          "[SplineOptimizer Error] 'TrajectoryCostFunc' signature mismatch.");
            static_assert(TypeTraits::HasExecutorInterface<Executor>::value,
                          "[SplineOptimizer Error] 'Executor' signature mismatch.");

            ensureLayoutCache();

            Workspace &ws = spec.workspace;
            resetEvaluationState(x.size(), grad_out, ws);
            decodeAndBuildWorkingState(x, ws);

            double total_cost =
                accumulateForwardCosts(spec.time_cost, spec.integral_cost, spec.sample_cost, ws, spec.executor);

            propagateSplineGradients(ws);
            total_cost += accumulatePostPropagationCosts(spec.waypoints_cost, spec.trajectory_cost, grad_out, ws);

            writeDecisionGradient(x, grad_out, ws);
            return total_cost;
        }

    public:

        template <typename TimeCostFunc,
                  typename IntegralCostFunc,
                  typename WaypointsCostFunc,
                  typename SampleCostFunc,
                  typename TrajectoryCostFunc,
                  typename Executor>
        EvaluationResult evaluate(
            const Eigen::VectorXd &x,
            Eigen::VectorXd &grad_out,
            const EvaluateSpec<TimeCostFunc, IntegralCostFunc,
                               WaypointsCostFunc, SampleCostFunc,
                               TrajectoryCostFunc, Executor> &spec) const
        {
            const Status status = validateEvaluateSpec(x, spec);
            if (!status)
            {
                grad_out.setZero(x.size());
                return makeErrorEvaluationResult(status.code, status.message);
            }
            return makeOkEvaluationResult(runEvaluation(x, grad_out, resolveEvaluateSpec(spec)));
        }

        /**
         * @brief Access the spline stored in a caller-provided workspace.
         */
        const SplineType &getWorkingSpline(const Workspace &workspace) const
        {
            return workspace.state.spline;
        }

        struct GradientCheckResult
        {
            bool valid = false;          
            ErrorCode code = ErrorCode::None;
            double error_norm = 0.0;      
            double rel_error = 0.0;       
            double max_abs_error = 0.0;
            Eigen::Index max_error_index = -1;
            Eigen::VectorXd analytical;   
            Eigen::VectorXd numerical;   
            std::string message;
            
            std::string makeReport() const {
                std::stringstream ss;
                ss << (valid ? "Gradient Check PASSED! " : "Gradient Check FAILED! ");
                ss << "Norm: " << error_norm;
                ss << ", MaxAbs: " << max_abs_error;
                if (max_error_index >= 0)
                {
                    ss << ", Index: " << max_error_index;
                }
                if (!message.empty())
                {
                    ss << "\n" << message;
                }
                ss << "\n";
                return ss.str();
            }
        };

        template <typename TimeCostFunc,
                  typename IntegralCostFunc,
                  typename WaypointsCostFunc,
                  typename SampleCostFunc,
                  typename TrajectoryCostFunc,
                  typename Executor>
        GradientCheckResult checkGradients(
            const Eigen::VectorXd &x,
            const EvaluateSpec<TimeCostFunc, IntegralCostFunc,
                               WaypointsCostFunc, SampleCostFunc,
                               TrajectoryCostFunc, Executor> &spec,
            double eps = 1e-6,
            double tol = 1e-4)
        {
            GradientCheckResult res;
            const Status status = validateEvaluateSpec(x, spec);
            if (!status)
            {
                res.code = status.code;
                res.message = status.message;
                return res;
            }
            auto resolved_spec = resolveEvaluateSpec(spec);

            res.analytical.resize(x.size());
            runEvaluation(x, res.analytical, resolved_spec);

            res.numerical.resize(x.size());
            computeNumericalGradient(x, eps, resolved_spec, res.numerical);

            Eigen::VectorXd diff = res.analytical - res.numerical;
            res.error_norm = diff.norm();
            if (diff.size() > 0)
            {
                Eigen::Index idx = 0;
                res.max_abs_error = diff.cwiseAbs().maxCoeff(&idx);
                res.max_error_index = idx;
            }

            double grad_norm = res.analytical.norm();
            res.rel_error = (grad_norm > 1e-9) ? (res.error_norm / grad_norm) : res.error_norm;

            res.valid = (res.error_norm < tol);

            return res;
        }


    private:
        static Status makeOkStatus()
        {
            Status status;
            status.ok = true;
            status.code = ErrorCode::None;
            return status;
        }

        static Status makeErrorStatus(ErrorCode code, std::string message)
        {
            Status status;
            status.ok = false;
            status.code = code;
            status.message = std::move(message);
            return status;
        }

        static EvaluationResult makeOkEvaluationResult(double cost)
        {
            EvaluationResult result;
            result.ok = true;
            result.code = ErrorCode::None;
            result.cost = cost;
            return result;
        }

        static EvaluationResult makeErrorEvaluationResult(ErrorCode code, std::string message)
        {
            EvaluationResult result;
            result.ok = false;
            result.code = code;
            result.message = std::move(message);
            return result;
        }

        static Status makeValidationStatus(const std::vector<std::string> &errors)
        {
            if (errors.empty())
            {
                return makeOkStatus();
            }

            std::stringstream ss;
            ss << "[SplineOptimizer Validation Failed] Found " << errors.size() << " error(s):\n";
            for (size_t i = 0; i < errors.size(); ++i)
            {
                ss << "  [" << (i + 1) << "] " << errors[i] << "\n";
            }

            return makeErrorStatus(ErrorCode::ValidationFailed, ss.str());
        }

        int calculateDimension() const
        {
            ensureLayoutCache();
            return layout_.total_dimension;
        }

        void appendMaskCapabilityErrors(std::vector<std::string> &errors) const
        {
            if constexpr (SplineType::ORDER < 5)
            {
                if (optimization_mask_.start.a || optimization_mask_.end.a)
                {
                    errors.push_back("OptimizationMask requests acceleration optimization, "
                                     "but this spline order does not expose acceleration boundary variables.");
                }
            }

            if constexpr (SplineType::ORDER < 7)
            {
                if (optimization_mask_.start.j || optimization_mask_.end.j)
                {
                    errors.push_back("OptimizationMask requests jerk optimization, "
                                     "but this spline order does not expose jerk boundary variables.");
                }
            }
        }

        void appendValidationErrors(std::vector<std::string> &errors, bool require_initialized_state) const
        {
            appendMaskCapabilityErrors(errors);

            if (require_initialized_state)
            {
                if (num_segments_ <= 0)
                {
                    errors.push_back("Invalid segment count: num_segments_ <= 0");
                }

                if (ref_times_.size() != static_cast<size_t>(num_segments_))
                {
                    errors.push_back("Size mismatch: ref_times_.size() = " + std::to_string(ref_times_.size()) +
                                     " != num_segments_ = " + std::to_string(num_segments_));
                }

                if (ref_waypoints_.rows() != num_segments_ + 1)
                {
                    errors.push_back("Size mismatch: ref_waypoints_.rows() = " + std::to_string(ref_waypoints_.rows()) +
                                     " != num_segments_ + 1 = " + std::to_string(num_segments_ + 1));
                }

                if (!std::isfinite(start_time_))
                {
                    errors.push_back("Start time is not finite (NaN or Inf)");
                }
            }

            if (num_segments_ > 0)
            {
                if (!optimization_mask_.time.empty() &&
                    optimization_mask_.time.size() != static_cast<size_t>(num_segments_))
                {
                    errors.push_back("OptimizationMask.time size mismatch: " +
                                     std::to_string(optimization_mask_.time.size()) +
                                     " != num_segments_ = " + std::to_string(num_segments_));
                }

                if (!optimization_mask_.waypoints.empty() &&
                    optimization_mask_.waypoints.size() != static_cast<size_t>(num_segments_ + 1))
                {
                    errors.push_back("OptimizationMask.waypoints size mismatch: " +
                                     std::to_string(optimization_mask_.waypoints.size()) +
                                     " != num_segments_ + 1 = " + std::to_string(num_segments_ + 1));
                }
            }
            else if (!optimization_mask_.time.empty())
            {
                errors.push_back("OptimizationMask.time can only be sized after setInitState() establishes segment count.");
            }
            else if (!optimization_mask_.waypoints.empty())
            {
                errors.push_back("OptimizationMask.waypoints can only be sized after setInitState() establishes segment count.");
            }

            if (require_initialized_state)
            {
                for (size_t i = 0; i < ref_times_.size(); ++i)
                {
                    double t = ref_times_[i];
                    if (!std::isfinite(t))
                    {
                        errors.push_back("Time segment [" + std::to_string(i) + "] is not finite: " + std::to_string(t));
                    }
                    else if (t < MIN_VALID_DURATION)
                    {
                        errors.push_back("Time segment [" + std::to_string(i) + "] is too small: " +
                                         std::to_string(t) + " < " + std::to_string(MIN_VALID_DURATION));
                    }
                }

                for (int i = 0; i < ref_waypoints_.rows(); ++i)
                {
                    if (!ref_waypoints_.row(i).array().isFinite().all())
                    {
                        errors.push_back("Waypoint row [" + std::to_string(i) + "] contains NaN or Inf");
                    }
                }

                if (!ref_bc_.start_velocity.array().isFinite().all())
                {
                    errors.push_back("Start velocity contains NaN or Inf");
                }
                if (!ref_bc_.end_velocity.array().isFinite().all())
                {
                    errors.push_back("End velocity contains NaN or Inf");
                }

                if constexpr (SplineType::ORDER >= 5)
                {
                    if (!ref_bc_.start_acceleration.array().isFinite().all())
                    {
                        errors.push_back("Start acceleration contains NaN or Inf");
                    }
                    if (!ref_bc_.end_acceleration.array().isFinite().all())
                    {
                        errors.push_back("End acceleration contains NaN or Inf");
                    }
                }

                if constexpr (SplineType::ORDER >= 7)
                {
                    if (!ref_bc_.start_jerk.array().isFinite().all())
                    {
                        errors.push_back("Start jerk contains NaN or Inf");
                    }
                    if (!ref_bc_.end_jerk.array().isFinite().all())
                    {
                        errors.push_back("End jerk contains NaN or Inf");
                    }
                }
            }
        }

        Status validateConfiguration(bool require_initialized_state) const
        {
            std::vector<std::string> errors;
            appendValidationErrors(errors, require_initialized_state);
            if (!errors.empty())
            {
                return makeValidationStatus(errors);
            }
            return makeOkStatus();
        }

        template <typename TimeCostFunc,
                  typename IntegralCostFunc,
                  typename WaypointsCostFunc,
                  typename SampleCostFunc,
                  typename TrajectoryCostFunc,
                  typename Executor>
        void computeNumericalGradient(
            const Eigen::VectorXd &x,
            double eps,
            const ResolvedEvaluateSpec<TimeCostFunc, IntegralCostFunc,
                                       WaypointsCostFunc, SampleCostFunc,
                                       TrajectoryCostFunc, Executor> &spec,
            Eigen::VectorXd &numerical_gradient) const
        {
            numerical_gradient.resize(x.size());

            Eigen::VectorXd dummy_grad(x.size());
            Eigen::VectorXd x_perturbed = x;

            for (int i = 0; i < x.size(); ++i)
            {
                const double original_value = x_perturbed(i);

                x_perturbed(i) = original_value + eps;
                const double cost_plus = runEvaluation(x_perturbed, dummy_grad, spec);

                x_perturbed(i) = original_value - eps;
                const double cost_minus = runEvaluation(x_perturbed, dummy_grad, spec);

                x_perturbed(i) = original_value;
                numerical_gradient(i) = (cost_plus - cost_minus) / (2 * eps);
            }
        }

        template <typename WaypointsCostFunc>
        const WaypointsCostFunc &resolveWaypointsCost(const WaypointsCostFunc *cost_ptr) const
        {
            if constexpr (std::is_same_v<WaypointsCostFunc, VoidWaypointsCost>)
            {
                if (cost_ptr == nullptr)
                {
                    static const VoidWaypointsCost default_cost{};
                    return default_cost;
                }
            }
            return *cost_ptr;
        }

        template <typename SampleCostFunc>
        const SampleCostFunc &resolveSampleCost(const SampleCostFunc *cost_ptr) const
        {
            if constexpr (std::is_same_v<SampleCostFunc, VoidSampleCost>)
            {
                if (cost_ptr == nullptr)
                {
                    static const VoidSampleCost default_cost{};
                    return default_cost;
                }
            }
            return *cost_ptr;
        }

        template <typename TrajectoryCostFunc>
        const TrajectoryCostFunc &resolveTrajectoryCost(const TrajectoryCostFunc *cost_ptr) const
        {
            if constexpr (std::is_same_v<TrajectoryCostFunc, VoidTrajectoryCost<SplineType, DIM>>)
            {
                if (cost_ptr == nullptr)
                {
                    static const VoidTrajectoryCost<SplineType, DIM> default_cost{};
                    return default_cost;
                }
            }
            return *cost_ptr;
        }

        template <typename TimeCostFunc,
                  typename IntegralCostFunc,
                  typename WaypointsCostFunc,
                  typename SampleCostFunc,
                  typename TrajectoryCostFunc,
                  typename Executor>
        Status validateEvaluateSpec(
            const Eigen::VectorXd &x,
            const EvaluateSpec<TimeCostFunc, IntegralCostFunc,
                               WaypointsCostFunc, SampleCostFunc,
                               TrajectoryCostFunc, Executor> &spec) const
        {
            if (!is_valid_)
            {
                return makeErrorStatus(ErrorCode::InvalidOptimizerState,
                                       "[SplineOptimizer Error] evaluate() called on an invalid optimizer state.");
            }
            if (integral_num_steps_ <= 0)
            {
                return makeErrorStatus(ErrorCode::InvalidIntegralSteps,
                                       "[SplineOptimizer Error] integral_num_steps must be positive.");
            }
            if (x.size() != getDimension())
            {
                return makeErrorStatus(ErrorCode::DimensionMismatch,
                                       "[SplineOptimizer Error] Input dimension mismatch in evaluate().");
            }
            if (spec.workspace == nullptr)
            {
                return makeErrorStatus(ErrorCode::NullWorkspace,
                                       "[SplineOptimizer Error] 'workspace' must not be null.");
            }
            if (spec.time_cost == nullptr)
            {
                return makeErrorStatus(ErrorCode::NullTimeCost,
                                       "[SplineOptimizer Error] 'time_cost' must not be null.");
            }
            if (spec.integral_cost == nullptr)
            {
                return makeErrorStatus(ErrorCode::NullIntegralCost,
                                       "[SplineOptimizer Error] 'integral_cost' must not be null.");
            }
            if constexpr (!std::is_same_v<WaypointsCostFunc, VoidWaypointsCost>)
            {
                if (spec.waypoints_cost == nullptr)
                {
                    return makeErrorStatus(ErrorCode::NullWaypointsCost,
                                           "[SplineOptimizer Error] 'waypoints_cost' must not be null.");
                }
            }
            if constexpr (!std::is_same_v<SampleCostFunc, VoidSampleCost>)
            {
                if (spec.sample_cost == nullptr)
                {
                    return makeErrorStatus(ErrorCode::NullSampleCost,
                                           "[SplineOptimizer Error] 'sample_cost' must not be null.");
                }
            }
            if constexpr (!std::is_same_v<TrajectoryCostFunc, VoidTrajectoryCost<SplineType, DIM>>)
            {
                if (spec.trajectory_cost == nullptr)
                {
                    return makeErrorStatus(ErrorCode::NullTrajectoryCost,
                                           "[SplineOptimizer Error] 'trajectory_cost' must not be null.");
                }
            }
            return makeOkStatus();
        }

        void accumulateSampleCostGradients(const IntegralSampleBuffer &samples,
                                           const SampleGradMatrix &sample_position_gradients,
                                           const Eigen::VectorXd &sample_time_gradients,
                                           MatrixType &grad_coeffs,
                                           Eigen::VectorXd &grad_times) const
        {
            const Eigen::Index sample_count = static_cast<Eigen::Index>(samples.size());
            if (sample_count == 0)
            {
                return;
            }

            if (sample_position_gradients.rows() != DIM || sample_position_gradients.cols() != sample_count)
            {
                assert(false && "[SplineOptimizer Error] Sample position gradient shape mismatch.");
                return;
            }

            Eigen::VectorXd global_time_grad = Eigen::VectorXd::Zero(num_segments_);

            for (Eigen::Index sample_idx = 0; sample_idx < sample_count; ++sample_idx)
            {
                const IntegralSample &sample = samples[sample_idx];
                const int base_row = sample.seg_idx * SplineType::COEFF_NUM;
                const VectorType grad_position = sample_position_gradients.col(sample_idx);

                grad_coeffs.template block<SplineType::COEFF_NUM, DIM>(base_row, 0).noalias() +=
                    sample.b_p.transpose() * grad_position.transpose();
                grad_times(sample.seg_idx) += grad_position.dot(sample.v) * sample.alpha;

                if (sample_idx < sample_time_gradients.size())
                {
                    const double grad_time = sample_time_gradients(sample_idx);
                    grad_times(sample.seg_idx) += grad_time * sample.alpha;
                    global_time_grad(sample.seg_idx) += grad_time;
                }
            }

            double accumulator = 0.0;
            for (int i = num_segments_ - 1; i > 0; --i)
            {
                accumulator += global_time_grad(i);
                grad_times(i - 1) += accumulator;
            }
        }

        template <typename IntegralFunc, typename Executor>
        void accumulateIntegralCost(Workspace &ws,
                                    MatrixType &grad_coeffs,
                                    Eigen::VectorXd &grad_times,
                                    double &cost,
                                    IntegralFunc &&integral_cost,
                                    double start_time,
                                    bool record_samples,
                                    const Executor& executor) const
        {
            auto &state = ws.state;
            auto &buffers = ws.buffers;
            const auto &coeffs = state.spline.getTrajectory().getCoefficients();

            double running_time = start_time;
            for(int i = 0; i < num_segments_; ++i) {
                buffers.segment_begin_times[i] = running_time;
                running_time += state.times[i];
            }

            std::fill(buffers.segment_cost_buffer.begin(), buffers.segment_cost_buffer.end(), 0.0);

            buffers.global_time_grad_buffer.setZero();

            int K = integral_num_steps_;
            double inv_K = 1.0 / K;

            if (record_samples)
            {
                buffers.integral_samples.resize(num_segments_ * (K + 1));
            }
            else
            {
                buffers.integral_samples.clear();
            }

            executor(0, num_segments_, [&](int i) {
                double T = state.times[i];
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
                double current_segment_start_time = buffers.segment_begin_times[i];

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

                    double c_val = integral_cost(t, t_global, i, k, p, v, a, j, s, gp, gv, ga, gj, gs, gt);

                    if (record_samples)
                    {
                        const int sample_index = i * (K + 1) + k;
                        IntegralSample &sample = buffers.integral_samples[sample_index];
                        sample.seg_idx = i;
                        sample.step_in_seg = k;
                        sample.sample_buffer_index = sample_index;
                        sample.alpha = alpha;
                        sample.t_local = t;
                        sample.t_global = t_global;
                        sample.trap_weight = weight_trap;
                        sample.dt = dt;
                        sample.b_p = b_p;
                        sample.p = p;
                        sample.v = v;
                    }

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

                buffers.segment_cost_buffer[i] = local_acc_cost;

                grad_times(i) += local_acc_gdT;
                buffers.global_time_grad_buffer(i) += local_acc_explicit_time_grad;

                grad_coeffs.template block(base_row, 0, SplineType::COEFF_NUM, DIM) += local_acc_gdC;
            });

            for(int i = 0; i < num_segments_; ++i) {
                cost += buffers.segment_cost_buffer[i];
            }

            double accumulator = 0.0;
            for (int i = num_segments_ - 1; i > 0; --i)
            {
                accumulator += buffers.global_time_grad_buffer(i);
                grad_times(i - 1) += accumulator;
            }
        }
    };
}
#endif
