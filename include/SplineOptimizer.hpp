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

/** @file SplineOptimizer.hpp
 * @brief Statically bound objectives with owned preparation and evaluation workspaces. */

#ifndef SPLINE_OPTIMIZER_HPP
#define SPLINE_OPTIMIZER_HPP

#include "SplineTrajectory.hpp"
#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <cassert>
#include <vector>
#include <cstdint>
#include <functional>
#include <numeric>
#include <type_traits>
#include <utility>

namespace SplineTrajectory
{
    /** @brief Cost and explicit status for callbacks and optimizer evaluations.
     * @note The optimizer clears the decision gradient and returns infinity on failure.
     *       Callbacks report failures directly; any partial output from a failed callback is discarded. */
    struct [[nodiscard]] EvaluationResult
    {
        double cost = std::numeric_limits<double>::infinity();
        Status status{ErrorCode::NotPrepared, "No successful evaluation"};
        bool ok() const noexcept { return status.ok(); }
        explicit operator bool() const noexcept { return ok(); }
    };

    /** @brief Integration location on the source time axis, measured in seconds. */
    struct IntegralPointInfo
    {
        int segment_index = 0, segment_count = 0, step_index = 0, step_count = 0;
        double alpha = 0.0, segment_duration = 0.0, step_size = 0.0;
        double segment_time = 0.0, trajectory_time = 0.0;
        bool isSegmentStart() const noexcept { return step_index == 0; }
        bool isSegmentEnd() const noexcept { return step_index == step_count; }
        bool isTrajectoryStart() const noexcept { return segment_index == 0 && isSegmentStart(); }
        bool isTrajectoryEnd() const noexcept { return segment_index + 1 == segment_count && isSegmentEnd(); }
        /** @brief Input waypoint row in [0, segment_count], or -1 between waypoints.
         * @note Both quadrature records at an interior knot identify the same waypoint. */
        int waypointIndex() const noexcept
        {
            if (isSegmentStart()) return segment_index;
            return isSegmentEnd() ? segment_index + 1 : -1;
        }
    };

    /** @brief State at one integration point; derivatives use physical seconds in the caller's frame. */
    template<int DIM>
    struct IntegralState
    {
        using Vector = Eigen::Matrix<double, DIM, 1>;
        Vector p = Vector::Zero(), v = Vector::Zero(), a = Vector::Zero();
        Vector j = Vector::Zero(), s = Vector::Zero();
    };

    /** @brief Unweighted integrand partials, cleared for each integration point.
     * @note integral() returns the unweighted integrand. The optimizer applies quadrature weights
     *       to both cost and partials. trajectory_time is the explicit time partial; state drift
     *       and the derivative of the integration measure are handled by the optimizer. */
    template<int DIM>
    struct IntegralGradient
    {
        using Vector = Eigen::Matrix<double, DIM, 1>;
        Vector p = Vector::Zero(), v = Vector::Zero(), a = Vector::Zero();
        Vector j = Vector::Zero(), s = Vector::Zero();
        double trajectory_time = 0.0;
    };

    /** @brief Direct physical duration coordinates; the optimizer checks decoded candidates for positivity. */
    struct IdentityTimeMap
    {
        /** @brief Copy a physical-time coordinate without transforming it.
         * @param value Finite coordinate in seconds; the optimizer validates decoded positivity.
         * @return The unchanged coordinate. */
        double forward(double value) const noexcept
        { return value; }
        /** @brief Copy physical seconds. @param value Caller-validated positive finite duration. @return Coordinate. */
        double inverse(double value) const noexcept { return forward(value); }
        /** @brief Pull back a finite time partial. @param tau Decision coordinate. @param time Physical seconds.
         * @param gradient Finite physical partial. @return The unchanged decision partial. */
        double backward(double tau, double time, double gradient) const noexcept
        {
            (void)tau; (void)time;
            return gradient;
        }
    };

    /** @brief Smooth positive duration map with an analytic inverse and pullback. */
    struct QuadInvTimeMap
    {
        /** @brief Map a decision coordinate to physical duration using the quadratic inverse map.
         * @param tau Finite dimensionless decision coordinate.
         * @return Computed duration in seconds; Optimizer checks its range once during decoding.
         * @note Stateless and reentrant; standalone callers own numerical-range validation. */
        double forward(double tau) const noexcept
        {
            const double time = tau > 0.0 ? (0.5 * tau + 1.0) * tau + 1.0
                                         : 1.0 / ((0.5 * tau - 1.0) * tau + 1.0);
            return time;
        }
        /** @brief Invert the duration map for initial decision coordinates.
         * @param duration Positive physical duration in seconds.
         * @return Computed coordinate; Optimizer validates the assembled initial vector once.
         * @note Stateless and reentrant; standalone callers own numerical-range validation. */
        double inverse(double duration) const noexcept
        {
            const double tau = duration > 1.0 ? std::sqrt(2.0 * duration - 1.0) - 1.0
                                             : 1.0 - std::sqrt(2.0 / duration - 1.0);
            return tau;
        }
        /** @brief Apply the duration-map chain rule without rescanning intermediate results.
         * @param tau Dimensionless decision coordinate used for the forward map.
         * @param time Physical duration in seconds; unused by this analytic derivative.
         * @param gradient Partial with respect to physical duration.
         * @return Decision partial; Optimizer checks the final accumulated gradient once.
         * @note Stateless and reentrant; standalone callers guarantee representable arithmetic. */
        double backward(double tau, double time, double gradient) const noexcept
        {
            (void)time;
            const double denominator = (0.5 * tau - 1.0) * tau + 1.0;
            const double result = tau > 0.0 ? gradient * (tau + 1.0) :
                gradient * (1.0 - tau) / (denominator * denominator);
            return result;
        }
    };

    /** @brief Cartesian coordinates with an allocation-free pullback; callers supply finite, DIM-sized inputs/outputs. */
    template<int DIM>
    struct IdentitySpatialMap
    {
        using Vector = Eigen::Matrix<double, DIM, 1>;
        /** @brief Report Cartesian decision dimension.
         * @param index Global waypoint index; this map uses the same dimension everywhere.
         * @return DIM independent coordinates. */
        int dimension(int index) const noexcept { (void)index; return DIM; }
        /** @brief Copy decision coordinates into physical Cartesian coordinates.
         * @param value Finite DIM-sized coordinates. @param index Global waypoint index; unused.
         * @return Owned vector in the caller's coordinate frame; stateless and reentrant. */
        Vector forward(const Eigen::Ref<const Eigen::VectorXd> &value, int index) const
        {
            (void)index;
            return Vector(value);
        }
        /** @brief Copy physical Cartesian coordinates into decision coordinates.
         * @param point Finite physical position. @param index Global waypoint index; unused.
         * @return Independent DIM-sized vector; stateless and reentrant. */
        Vector inverse(const Vector &point, int index) const
        {
            (void)index;
            return point;
        }
        /** @brief Overwrite a Cartesian decision partial with the physical partial.
         * @param coordinates Forward decision coordinates; unused by the identity Jacobian.
         * @param gradient Physical position partial. @param index Global waypoint index; unused.
         * @param[out] destination Exclusive DIM-sized decision slice; no allocation or resizing. */
        void backward(const Eigen::Ref<const Eigen::VectorXd> &coordinates, const Vector &gradient,
                          int index, Eigen::Ref<Eigen::VectorXd> destination) const
        {
            (void)coordinates; (void)index;
            destination = gradient;
        }
    };

    /** @brief Explicit absence of auxiliary decision variables. */
    struct NoAuxiliaryMap {};
    /** @brief Explicit objective with no user cost; minimum-derivative energy may still be enabled. */
    struct NoObjective {};

    /** @brief Owned default parameterization: positive times and Cartesian waypoints. */
    template<class Spline>
    struct DefaultParameterization
    {
        QuadInvTimeMap time;
        IdentitySpatialMap<Spline::kDimension> space;
        NoAuxiliaryMap auxiliary;
    };

    struct BoundaryDerivativeMask { bool v = false, a = false, j = false; };
    /** @brief Select direct optimization coordinates; empty arrays select all durations and interior waypoints.
     * @note Boundary derivatives default to unselected. Auxiliary coordinates are independent of this mask
     *       and may still change unselected physical parameters through their apply() mapping. */
    struct OptimizationMask
    {
        std::vector<std::uint8_t> durations, waypoints;
        BoundaryDerivativeMask start, end;

        /** @brief Select no direct coordinates for a nonnegative segment count; auxiliary coordinates remain active. */
        static OptimizationMask none(int segments)
        {
            OptimizationMask mask;
            mask.durations.assign(segments, 0);
            mask.waypoints.assign(segments + 1, 0);
            return mask;
        }

        /** @brief Select only interior waypoints as direct coordinates; auxiliary coordinates remain active. */
        static OptimizationMask interiorWaypoints(int segments)
        {
            auto mask = none(segments);
            for (int i = 1; i < segments; ++i) mask.waypoints[i] = 1;
            return mask;
        }
    };

    /** @brief Fixed solve options; changing topology or recording requires preparation. */
    struct OptimizerOptions
    {
        double energy_weight = 0.0;
        int integration_steps = 64;
        bool record_samples = false;
        double duration_weight = 0.0; ///< Linear cost on the sum of physical segment durations.
        OptimizationMask mask{};
    };

    struct DurationVariable { int segment_index = 0, offset = 0; };
    struct PointVariable { int point_index = 0, offset = 0, dof = 0; };
    /** @brief Read-only decision layout, valid until preparation, movement or destruction. */
    struct DecisionLayout
    {
        std::vector<DurationVariable> durations;
        std::vector<PointVariable> waypoints;
        int boundary_offset = 0, auxiliary_offset = 0, dimension = 0;
    };

    /** @brief Exclusive mutable physical parameters passed to an auxiliary map.
     * @note Existing topology must be preserved; references expire on return from apply(). */
    template<class Spline>
    struct MutableParameters
    {
        std::vector<double> &durations;
        typename Spline::WaypointMatrix &waypoints;
        BoundaryConditions<Spline::kDimension> &boundary;
        double &start_time;
    };

    /** @brief Borrowed physical state for coefficient/parameter costs and auxiliary pullbacks. */
    template<class Spline>
    struct ParameterView
    {
        const typename Spline::Polynomial &polynomial;
        const std::vector<double> &durations;
        const typename Spline::WaypointMatrix &waypoints;
        const BoundaryConditions<Spline::kDimension> &boundary;
        double start_time;
    };

    /** @brief Additive coefficient and independent-time partials before the spline adjoint.
     * @note durations differentiates time with local power coefficients held fixed. */
    template<class Spline>
    struct CoefficientGradientView
    {
        Eigen::Ref<typename Spline::CoefficientMatrix> coefficients;
        Eigen::Ref<Eigen::VectorXd> durations;
        double &start_time;
    };

    /** @brief Additive physical parameter gradients after the spline adjoint. */
    template<class Spline>
    struct ParameterGradientView
    {
        typename Spline::WaypointMatrix &waypoints;
        Eigen::VectorXd &durations;
        BoundaryConditions<Spline::kDimension> &boundary;
        double &start_time;
        /** @brief Add a row or column waypoint partial to the borrowed owning storage. */
        template<class Derived>
        void addWaypoint(int index, const Eigen::MatrixBase<Derived> &partial)
        { detail::addWaypointGradient(waypoints, index, partial); }
    };

    /** @brief Additive batch partials with one row per recorded quadrature point.
     * @note sample() returns the whole batch cost; no quadrature weights are applied automatically.
     *       Rows match SampleBuffer, including duplicated interior knots. Only position and explicit
     *       trajectory-time partials are supported; recorded velocities are used for the time chain rule. */
    template<class Spline>
    struct SampleGradientView
    {
        Eigen::Ref<typename Spline::WaypointMatrix> positions;
        Eigen::Ref<Eigen::VectorXd> trajectory_times;
    };

    /** @brief Borrowed decision coordinates and immutable layout for additive domain penalties. */
    struct DecisionView
    {
        const Eigen::VectorXd &variables;
        const DecisionLayout &layout;
    };

    /** @brief Serial segment execution; the concrete callable remains visible to the compiler. */
    struct SerialExecutor
    {
        template<class Function>
        void operator()(int begin, int end, Function &&function) const
        { for (int i = begin; i < end; ++i) function(i); }
    };

    /** @brief Parallel segment execution when OpenMP is enabled, otherwise serial.
     * @note The integral cost must permit concurrent calls; mutable diagnostic state must be independent.
     *       With OpenMP, exceptions must not escape a worker callback. */
    struct OpenMPExecutor
    {
        template<class Function>
        void operator()(int begin, int end, Function &&function) const
        {
#if defined(_OPENMP)
#pragma omp parallel for schedule(static)
#endif
            for (int i = begin; i < end; ++i) function(i);
        }
    };

    namespace detail
    {
        template<class T> T &unwrap(T &value) noexcept { return value; }
        template<class T> T &unwrap(const std::reference_wrapper<T> &value) noexcept { return value.get(); }
        template<class T> T &unwrap(std::reference_wrapper<T> &value) noexcept { return value.get(); }
        template<class T> using Plain = std::remove_cv_t<std::remove_reference_t<T>>;

        // Diagnose the removed stage instead of silently ignoring an old objective's time cost.
        template<class T, class = void> struct HasDurationMember : std::false_type {};
        template<class T> struct HasDurationMember<T, std::void_t<decltype(std::declval<T &>().duration)>> : std::true_type {};

        struct MissingStage {};
        template<class Result>
        inline constexpr bool isCostResult = std::is_same_v<Result, double> ||
                                             std::is_same_v<Result, EvaluationResult>;

        inline EvaluationResult toEvaluationResult(double cost) { return {cost, Status::success()}; }
        inline EvaluationResult toEvaluationResult(EvaluationResult result) { return result; }

        template<class Objective, class... Args>
        struct IntegralStage
        {
        private:
            template<class T> static auto result(int) -> decltype(std::declval<T &>().integral(std::declval<Args>()...));
            template<class> static MissingStage result(...);
            template<class T> static auto object(int)
                -> std::bool_constant<std::is_object_v<Plain<decltype(std::declval<T &>().integral)>>>;
            template<class> static std::false_type object(...);
            template<class T> static auto named(int) -> decltype((void)std::declval<T &>().integral, std::true_type{});
            template<class T> static auto named(long) -> decltype((void)&T::integral, std::true_type{});
            template<class> static std::false_type named(...);
        public:
            using Result = decltype(result<Objective>(0));
            static constexpr bool present = !std::is_same_v<Result, MissingStage> || decltype(named<Objective>(0))::value;
            static constexpr bool valid = decltype(object<Objective>(0))::value && isCostResult<Result>;
        };

        template<class Objective, class... Args>
        struct SampleStage
        {
        private:
            template<class T> static auto result(int) -> decltype(std::declval<T &>().sample(std::declval<Args>()...));
            template<class> static MissingStage result(...);
            template<class T> static auto object(int)
                -> std::bool_constant<std::is_object_v<Plain<decltype(std::declval<T &>().sample)>>>;
            template<class> static std::false_type object(...);
            template<class T> static auto named(int) -> decltype((void)std::declval<T &>().sample, std::true_type{});
            template<class T> static auto named(long) -> decltype((void)&T::sample, std::true_type{});
            template<class> static std::false_type named(...);
        public:
            using Result = decltype(result<Objective>(0));
            static constexpr bool present = !std::is_same_v<Result, MissingStage> || decltype(named<Objective>(0))::value;
            static constexpr bool valid = decltype(object<Objective>(0))::value && isCostResult<Result>;
        };

        template<class Objective, class... Args>
        struct CoefficientStage
        {
        private:
            template<class T> static auto result(int) -> decltype(std::declval<T &>().coefficient(std::declval<Args>()...));
            template<class> static MissingStage result(...);
            template<class T> static auto object(int)
                -> std::bool_constant<std::is_object_v<Plain<decltype(std::declval<T &>().coefficient)>>>;
            template<class> static std::false_type object(...);
            template<class T> static auto named(int) -> decltype((void)std::declval<T &>().coefficient, std::true_type{});
            template<class T> static auto named(long) -> decltype((void)&T::coefficient, std::true_type{});
            template<class> static std::false_type named(...);
        public:
            using Result = decltype(result<Objective>(0));
            static constexpr bool present = !std::is_same_v<Result, MissingStage> || decltype(named<Objective>(0))::value;
            static constexpr bool valid = decltype(object<Objective>(0))::value && isCostResult<Result>;
        };

        template<class Objective, class... Args>
        struct ParameterStage
        {
        private:
            template<class T> static auto result(int) -> decltype(std::declval<T &>().parameter(std::declval<Args>()...));
            template<class> static MissingStage result(...);
            template<class T> static auto object(int)
                -> std::bool_constant<std::is_object_v<Plain<decltype(std::declval<T &>().parameter)>>>;
            template<class> static std::false_type object(...);
            template<class T> static auto named(int) -> decltype((void)std::declval<T &>().parameter, std::true_type{});
            template<class T> static auto named(long) -> decltype((void)&T::parameter, std::true_type{});
            template<class> static std::false_type named(...);
        public:
            using Result = decltype(result<Objective>(0));
            static constexpr bool present = !std::is_same_v<Result, MissingStage> || decltype(named<Objective>(0))::value;
            static constexpr bool valid = decltype(object<Objective>(0))::value && isCostResult<Result>;
        };

        template<class Objective, class... Args>
        struct DecisionStage
        {
        private:
            template<class T> static auto result(int) -> decltype(std::declval<T &>().decision(std::declval<Args>()...));
            template<class> static MissingStage result(...);
            template<class T> static auto object(int)
                -> std::bool_constant<std::is_object_v<Plain<decltype(std::declval<T &>().decision)>>>;
            template<class> static std::false_type object(...);
            template<class T> static auto named(int) -> decltype((void)std::declval<T &>().decision, std::true_type{});
            template<class T> static auto named(long) -> decltype((void)&T::decision, std::true_type{});
            template<class> static std::false_type named(...);
        public:
            using Result = decltype(result<Objective>(0));
            static constexpr bool present = !std::is_same_v<Result, MissingStage> || decltype(named<Objective>(0))::value;
            static constexpr bool valid = decltype(object<Objective>(0))::value && isCostResult<Result>;
        };

        template<class T, class = void> struct IntegralDerivativeOrder : std::integral_constant<int, 4> {};
        template<class T> struct IntegralDerivativeOrder<T, std::void_t<decltype(T::kDerivativeOrder)>>
            : std::integral_constant<int, T::kDerivativeOrder> {};

        /** @brief Read a total map's result without wrapping or copying it.
         * @param value Borrowed mapping result. @return The same value for immediate consumption. */
        template<class T> const T &mappingValue(const T &value) { return value; }
        /** @brief Read a fallible map after its status has been checked.
         * @param value Successful borrowed mapping result. @return Borrow valid while value lives. */
        template<class T> const T &mappingValue(const StatusOr<T> &value) { return value.value(); }
        /** @brief Total maps cannot reject their input domain.
         * @param value Mapping result. @return Compile-time unconditional success. */
        template<class T> Status mappingStatus(const T &value) { (void)value; return Status::success(); }
        /** @brief Preserve a user map's explicit domain rejection.
         * @param value Fallible mapping result. @return Its original success or failure status. */
        template<class T> Status mappingStatus(const StatusOr<T> &value) { return value.status(); }

        /** @brief Invoke a spatial pullback with either a total or a fallible contract.
         * @param function Synchronous, exclusively borrowed pullback, returning void or Status.
         * @return Success for a total map, otherwise the map's original status. */
        template<class Function> Status invokeMapping(Function &&function)
        {
            if constexpr (std::is_void_v<std::invoke_result_t<Function>>)
            { function(); return Status::success(); }
            else return function();
        }

    }

    /** @brief Reusable, exclusively owned spline optimization workspace.
     * @tparam Spline Concrete minimum-derivative spline; dimension and degree are inferred.
     * @tparam Parameterization Owned maps, or an explicit reference_wrapper to longer-lived maps.
     * @note One instance per concurrent solve. No objective is retained between synchronous calls.
     *       prepare(), assignment and movement invalidate all borrowed state views.
     *       Objective groups callable data members named integral, sample, coefficient, parameter or decision.
     *       Each cost implements operator(); std::ref/std::cref explicitly borrow an existing cost object.
     *       The integral cost declares kDerivativeOrder (default 4).
     *       Objective itself has no stage methods or integration settings. */
    template<class Spline, class Parameterization = DefaultParameterization<Spline>>
    class SplineOptimizer
    {
        static constexpr int DIM = Spline::kDimension;
        using SplineType = Spline;
        using Maps = detail::Plain<decltype(detail::unwrap(std::declval<const Parameterization &>()))>;
        using AuxiliaryMap = detail::Plain<decltype(detail::unwrap(std::declval<const Maps &>().auxiliary))>;
        static constexpr bool has_auxiliary = !std::is_same_v<AuxiliaryMap, NoAuxiliaryMap>;

    public:
        using Vector = typename Spline::Vector;
        using CoefficientMatrix = typename Spline::CoefficientMatrix;
        using Problem = SplineInput<DIM>;
        using Polynomial = typename Spline::Polynomial;
        using Status = SplineTrajectory::Status;
        using ErrorCode = SplineTrajectory::ErrorCode;
        using Gradients = typename Spline::Gradients;
        using SampleGradMatrix = typename Spline::WaypointMatrix;

        /** @brief One recorded quadrature point, borrowed until the next evaluation or preparation. */
        struct IntegralSample
        {
            IntegralPointInfo point;
            double trap_weight = 0.0;
            Eigen::Matrix<double, 1, Spline::kCoefficientCount> b_p;
            Vector p = Vector::Zero(), v = Vector::Zero();
        };
        /** @brief Segment-major quadrature records: segment_count * (integration_steps + 1) points.
         * @note Each segment includes both endpoints. For a cost that visits each knot once, skip
         *       point.isSegmentStart() && !point.isTrajectoryStart(), keeping the original gradient rows.
         *       Skipped rows remain zero; integral() still uses both records for quadrature. */
        using SampleBuffer = SplineVector<IntegralSample>;

    private:
        template<class O> using IntegralStage = detail::IntegralStage<O, const IntegralPointInfo &, const IntegralState<DIM> &, IntegralGradient<DIM> &>;
        template<class O> using SampleStage = detail::SampleStage<O, const SampleBuffer &, SampleGradientView<Spline> &>;
        template<class O> using CoefficientStage = detail::CoefficientStage<O, const ParameterView<Spline> &, CoefficientGradientView<Spline> &>;
        template<class O> using ParameterStage = detail::ParameterStage<O, const ParameterView<Spline> &, ParameterGradientView<Spline> &>;
        template<class O> using DecisionStage = detail::DecisionStage<O, const DecisionView &, Eigen::Ref<Eigen::VectorXd>>;

        template<class O>
        static constexpr void validateObjective()
        {
            static_assert(!detail::HasDurationMember<O>::value,
                          "Objective.duration was removed; move its cost to Objective.parameter");
            static_assert(IntegralStage<O>::present || SampleStage<O>::present ||
                          CoefficientStage<O>::present || ParameterStage<O>::present ||
                          DecisionStage<O>::present || std::is_same_v<O, NoObjective>,
                          "Objective must declare a supported cost member, or explicitly use NoObjective");
            static_assert(!IntegralStage<O>::present || IntegralStage<O>::valid,
                          "Objective.integral must be a callable data member returning double or EvaluationResult");
            static_assert(!SampleStage<O>::present || SampleStage<O>::valid,
                          "Objective.sample must be a callable data member returning double or EvaluationResult");
            static_assert(!CoefficientStage<O>::present || CoefficientStage<O>::valid,
                          "Objective.coefficient must be a callable data member returning double or EvaluationResult");
            static_assert(!ParameterStage<O>::present || ParameterStage<O>::valid,
                          "Objective.parameter must be a callable data member returning double or EvaluationResult");
            static_assert(!DecisionStage<O>::present || DecisionStage<O>::valid,
                          "Objective.decision must be a callable data member returning double or EvaluationResult");
        }

        // Numeric kernel aliases are local; public dimensions come from Spline.
        using IntegralBasis = Eigen::Matrix<double, 6, Spline::kCoefficientCount, Eigen::RowMajor>;
        using SegmentCoefficients = Eigen::Matrix<double, Spline::kCoefficientCount, DIM>;
        using SegmentPowers = Eigen::Matrix<double, Spline::kCoefficientCount, 1>;
        struct PreparedData
        {
            Problem problem;
            OptimizerOptions options;
            DecisionLayout layout;
            Eigen::VectorXd initial_variables;
            SplineVector<IntegralBasis> basis;
            int num_segments = 0;
            bool sample_capability = false;
            Status status{ErrorCode::NotPrepared, "Optimizer is not prepared"};
        };
        struct WorkingState
        {
            Spline spline;
            std::vector<double> durations;
            CoefficientMatrix waypoints;
            BoundaryConditions<DIM> bc;
            double start_time = 0.0;
            Eigen::VectorXd auxiliary_vars, mapped_durations;
        };
        struct EvaluationBuffers
        {
            CoefficientMatrix grad_coeffs;
            Eigen::VectorXd grad_times, trajectory_time_grad_buffer, auxiliary_grad_buffer;
            double grad_start_time = 0.0;
            Gradients grads;
            SampleGradMatrix sample_position_grad_buffer;
            Eigen::VectorXd sample_time_grad_buffer;
            std::vector<EvaluationResult> segment_results;
            SampleBuffer integral_samples;
            bool samples_valid = false;
            void prepare(int pieces, int auxiliary, Eigen::Index samples)
            {
                samples_valid = false;
                grad_coeffs.resize(pieces * Spline::kCoefficientCount, DIM);
                grad_times.resize(pieces);
                trajectory_time_grad_buffer.resize(pieces);
                auxiliary_grad_buffer.resize(auxiliary);
                grads.setZero(pieces);
                segment_results.resize(pieces);
                integral_samples.resize(static_cast<std::size_t>(samples));
                sample_position_grad_buffer.resize(samples, DIM);
                sample_time_grad_buffer.resize(samples);
            }
        };

        Parameterization parameterization_;
        PreparedData prepared_;
        WorkingState state_;
        EvaluationBuffers buffers_;


        decltype(auto) maps() const { return detail::unwrap(parameterization_); }
        decltype(auto) timeMap() const { return detail::unwrap(maps().time); }
        decltype(auto) spaceMap() const { return detail::unwrap(maps().space); }
        decltype(auto) auxiliaryMap() const { return detail::unwrap(maps().auxiliary); }
        int auxiliaryDimension() const
        {
            if constexpr (has_auxiliary) return auxiliaryMap().dimension();
            else return 0;
        }
        static Status error(ErrorCode code, const char *message, int index = -1) noexcept
        { return {code, message, index}; }
        Status failPreparation(Status status)
        {
            prepared_.status = status;
            state_.spline.invalidate();
            return status;
        }
        EvaluationResult failEvaluation(Status status, Eigen::VectorXd &gradient)
        {
            buffers_.samples_valid = false;
            gradient.setZero();
            return {std::numeric_limits<double>::infinity(), status};
        }
        static bool finiteBoundary(const BoundaryConditions<DIM> &boundary)
        {
            bool valid = detail::allFinite(boundary.start.v) && detail::allFinite(boundary.end.v);
            if constexpr (Spline::kDegree >= 5)
                valid = valid && detail::allFinite(boundary.start.a) && detail::allFinite(boundary.end.a);
            if constexpr (Spline::kDegree >= 7)
                valid = valid && detail::allFinite(boundary.start.j) && detail::allFinite(boundary.end.j);
            return valid;
        }
        template<class Boundary, class Function>
        void forEachBoundary(Boundary &boundary, Function &&function) const
        {
            const auto visit = [&](auto &endpoint, const BoundaryDerivativeMask &mask) {
                if (mask.v) function(endpoint.v);
                if constexpr (Spline::kDegree >= 5) if (mask.a) function(endpoint.a);
                if constexpr (Spline::kDegree >= 7) if (mask.j) function(endpoint.j);
            };
            visit(boundary.start, prepared_.options.mask.start);
            visit(boundary.end, prepared_.options.mask.end);
        }
        ParameterView<Spline> parameterView() const
        {
            return {state_.spline.polynomial(), state_.durations, state_.waypoints, state_.bc, state_.start_time};
        }
        ParameterGradientView<Spline> parameterGradient()
        {
            auto &gradient = buffers_.grads;
            return {gradient.waypoints, gradient.durations, gradient.boundary, buffers_.grad_start_time};
        }

        // Initial coordinates belong to the prepared problem, not the changing trial state.
        StatusOr<Eigen::VectorXd> encodeInitialGuess() const
        {
            const auto &reference = prepared_.problem;
            Eigen::VectorXd x(prepared_.layout.dimension);
            for (const auto &variable : prepared_.layout.durations)
            {
                auto value = timeMap().inverse(reference.durations[variable.segment_index]);
                if (const auto status = detail::mappingStatus(value); !status) return status;
                x(variable.offset) = detail::mappingValue(value);
            }
            for (const auto &variable : prepared_.layout.waypoints)
            {
                auto value = spaceMap().inverse(reference.waypoints.row(variable.point_index).transpose(), variable.point_index);
                if (const auto status = detail::mappingStatus(value); !status) return status;
                if (detail::mappingValue(value).size() != variable.dof)
                    return Status{ErrorCode::DimensionMismatch, "Spatial map initial dimension mismatch", variable.point_index};
                x.segment(variable.offset, variable.dof) = detail::mappingValue(value);
            }
            int offset = prepared_.layout.boundary_offset;
            forEachBoundary(reference.boundary, [&](const auto &value) {
                x.template segment<DIM>(offset) = value;
                offset += DIM;
            });
            if constexpr (has_auxiliary)
            {
                auto value = auxiliaryMap().initial(reference);
                if (!value) return value.status();
                if (value.value().size() != state_.auxiliary_vars.size())
                    return Status{ErrorCode::DimensionMismatch, "Auxiliary map initial dimension mismatch"};
                x.segment(prepared_.layout.auxiliary_offset, value.value().size()) = value.value();
            }
            if (!detail::allFinite(x)) return Status{ErrorCode::NumericalFailure, "Initial decision coordinates are nonfinite"};
            return x;
        }

        /** @brief Validate reference topology and numerical options without changing workspace state.
         * @param problem Physical input with durations in seconds and waypoints in caller coordinates.
         * @param options Fixed integration and built-in cost settings.
         * @return Success or the first invalid-input/dimension status; maps are checked during layout setup. */
        static Status validateProblem(const Problem &problem, const OptimizerOptions &options)
        {
            if (problem.durations.empty() ||
                problem.durations.size() > static_cast<std::size_t>(std::numeric_limits<int>::max() / Spline::kCoefficientCount))
                return error(ErrorCode::InvalidInput, "Invalid segment count");
            const int n = static_cast<int>(problem.durations.size());
            if (problem.waypoints.rows() != n + 1 || problem.waypoints.cols() != DIM)
                return error(ErrorCode::DimensionMismatch, "Waypoint shape must be (segments + 1) by dimension");
            if (!detail::allFinite(problem.waypoints) || !finiteBoundary(problem.boundary) || !std::isfinite(problem.start_time))
                return error(ErrorCode::InvalidInput, "Nonfinite reference state");
            for (double duration : problem.durations)
                if (!std::isfinite(duration) || duration <= 0.0)
                    return error(ErrorCode::InvalidInput, "Durations must be finite and positive");
            if (options.integration_steps <= 0 || options.integration_steps == std::numeric_limits<int>::max() ||
                !std::isfinite(options.energy_weight) || options.energy_weight < 0.0 ||
                !std::isfinite(options.duration_weight) || options.duration_weight < 0.0)
                return error(ErrorCode::InvalidInput, "Invalid integration steps or cost weight");

            return Status::success();
        }

        /** @brief Expand the active mask and assign each decision coordinate a fixed offset.
         * @return Success or invalid mask/map dimensions; requires a validated prepared problem.
         * @note May allocate layout storage and call user maps; no numerical candidate is built. */
        Status prepareDecisionLayout()
        {
            const int n = prepared_.num_segments;
            auto &mask = prepared_.options.mask;
            if (mask.durations.empty()) mask.durations.assign(n, 1);
            if (mask.waypoints.empty())
            {
                mask.waypoints.assign(n + 1, 1);
                mask.waypoints.front() = mask.waypoints.back() = 0;
            }
            if (mask.durations.size() != static_cast<std::size_t>(n) || mask.waypoints.size() != static_cast<std::size_t>(n + 1))
                return error(ErrorCode::DimensionMismatch, "Optimization mask does not match topology");
            if constexpr (Spline::kDegree < 5)
                if (mask.start.a || mask.end.a)
                    return error(ErrorCode::InvalidInput, "This spline has no acceleration boundary variables");
            if constexpr (Spline::kDegree < 7)
                if (mask.start.j || mask.end.j)
                    return error(ErrorCode::InvalidInput, "This spline has no jerk boundary variables");

            auto &layout = prepared_.layout;
            layout.durations.clear();
            layout.waypoints.clear();
            std::int64_t offset = 0;
            for (int i = 0; i < n; ++i)
                if (mask.durations[i]) layout.durations.push_back({i, static_cast<int>(offset++)});
            for (int i = 0; i <= n; ++i)
            {
                if (!mask.waypoints[i]) continue;
                const int dof = spaceMap().dimension(i);
                if (dof <= 0 || offset + dof > std::numeric_limits<int>::max())
                    return error(ErrorCode::InvalidInput, "Invalid spatial decision dimension");
                layout.waypoints.push_back({i, static_cast<int>(offset), dof});
                offset += dof;
            }
            layout.boundary_offset = static_cast<int>(offset);
            forEachBoundary(prepared_.problem.boundary, [&](const auto &) { offset += DIM; });
            const int auxiliary = auxiliaryDimension();
            if (auxiliary < 0 || offset + auxiliary > std::numeric_limits<int>::max())
                return error(ErrorCode::InvalidInput, "Invalid auxiliary decision dimension");
            layout.auxiliary_offset = static_cast<int>(offset);
            layout.dimension = static_cast<int>(offset + auxiliary);
            return Status::success();
        }

        /** @brief Allocate evaluation buffers and precompute the normalized quadrature basis.
         * @return Success or an unsupported sample-count status; requires a valid decision layout.
         * @note Mutates owned storage during preparation only; evaluation reuses these buffers. */
        Status prepareEvaluationStorage()
        {
            const int n = prepared_.num_segments;
            const auto &options = prepared_.options;
            const int auxiliary = prepared_.layout.dimension - prepared_.layout.auxiliary_offset;
            const std::int64_t total_samples = static_cast<std::int64_t>(n) * (options.integration_steps + 1);
            if (total_samples > std::numeric_limits<int>::max())
                return error(ErrorCode::ResourceLimit, "Sample count exceeds supported indices");
            const std::int64_t sample_count = prepared_.sample_capability ? total_samples : 0;

            prepared_.basis.resize(static_cast<std::size_t>(options.integration_steps) + 1);
            for (int k = 0; k <= options.integration_steps; ++k)
            {
                Eigen::Matrix<double, 1, Spline::kCoefficientCount> p, v, a, j, snap, c;
                Spline::computeBasisFunctions(static_cast<double>(k) / options.integration_steps,
                                             p, v, a, j, snap, c);
                prepared_.basis[k] << p, v, a, j, snap, c;
            }
            buffers_.prepare(n, auxiliary, static_cast<Eigen::Index>(sample_count));
            // Fixed parameters are initialized once; only auxiliary maps can later change them.
            state_.durations = prepared_.problem.durations;
            state_.waypoints = prepared_.problem.waypoints;
            state_.bc = prepared_.problem.boundary;
            state_.start_time = prepared_.problem.start_time;
            if constexpr (has_auxiliary)
                state_.mapped_durations.resize(prepared_.layout.durations.size());
            state_.auxiliary_vars.resize(auxiliary);
            return Status::success();
        }

        /** @brief Prepare owned input, decision layout, reusable storage and the initial candidate.
         * @param problem Physical reference input, copied before changing potentially aliased state.
         * @param options Fixed integration and energy settings.
         * @param need_samples Whether the objective requires recorded sample storage.
         * @return Success or a preparation failure that leaves no usable candidate.
         * @note Requires exclusive access; allocation and user map exceptions follow prepare(). */
        Status prepareImpl(const Problem &problem, const OptimizerOptions &options, bool need_samples)
        {
            // Snapshot first: callers may pass this optimizer's current problem view.
            Problem snapshot = problem;
            prepared_.status = {ErrorCode::NotPrepared, "Optimizer is not prepared"};
            const Status input = validateProblem(snapshot, options);
            if (!input) return failPreparation(input);
            prepared_.problem = std::move(snapshot);
            prepared_.num_segments = static_cast<int>(prepared_.problem.durations.size());
            prepared_.options = options;
            prepared_.sample_capability = need_samples || options.record_samples;

            // Resolve maps and offsets before sizing the buffers consumed by evaluation.
            const Status layout = prepareDecisionLayout();
            if (!layout) return failPreparation(layout);
            const Status storage = prepareEvaluationStorage();
            if (!storage) return failPreparation(storage);

            // Encode once and build the initial candidate through the same path as later trials.
            auto initial = encodeInitialGuess();
            if (!initial) return failPreparation(initial.status());
            prepared_.initial_variables = std::move(initial).value();
            const Status built = buildCandidate(prepared_.initial_variables);
            if (!built) return failPreparation(built);
            return prepared_.status = Status::success();
        }

        /** @brief Decode trusted decision coordinates and construct one trial trajectory.
         * @param x Finite decision vector matching the prepared layout; checked by the entry point.
         * @return Success, or an explicit map, duration-domain, knot or factorization failure.
         * @pre Successful spatial/auxiliary maps preserve declared shapes and prepared topology.
         * @note Requires exclusive optimizer access. Failure invalidates the candidate spline. */
        Status buildCandidate(const Eigen::VectorXd &x)
        {
            auto reject = [&](Status status) { state_.spline.invalidate(); return status; };
            // Auxiliary maps may modify even fixed coordinates, so restore those inputs per trial.
            if constexpr (has_auxiliary)
            {
                const auto &reference = prepared_.problem;
                state_.durations = reference.durations;
                state_.waypoints = reference.waypoints;
                state_.bc = reference.boundary;
                state_.start_time = reference.start_time;
            }
            for (const auto &variable : prepared_.layout.durations)
            {
                auto value = timeMap().forward(x(variable.offset));
                if (const auto status = detail::mappingStatus(value); !status) return reject(status);
                const double duration = detail::mappingValue(value);
                if (!(duration > 0.0) || !std::isfinite(duration))
                    return reject({ErrorCode::NumericalFailure, "Mapped duration must be finite and positive", variable.segment_index});
                state_.durations[variable.segment_index] = duration;
                if constexpr (has_auxiliary) state_.mapped_durations(variable.offset) = duration;
            }
            for (const auto &variable : prepared_.layout.waypoints)
            {
                auto point = spaceMap().forward(x.segment(variable.offset, variable.dof), variable.point_index);
                if (const auto status = detail::mappingStatus(point); !status) return reject(status);
                state_.waypoints.row(variable.point_index) = detail::mappingValue(point).transpose();
            }
            int offset = prepared_.layout.boundary_offset;
            forEachBoundary(state_.bc, [&](auto &value) {
                value = x.template segment<DIM>(offset);
                offset += DIM;
            });
            if constexpr (has_auxiliary)
            {
                state_.auxiliary_vars = x.segment(prepared_.layout.auxiliary_offset, state_.auxiliary_vars.size());
                MutableParameters<Spline> state{state_.durations, state_.waypoints, state_.bc, state_.start_time};
                if (const auto status = auxiliaryMap().apply(state_.auxiliary_vars, state); !status) return reject(status);
            }
            // Time-map range is checked above; other maps promise valid physical outputs.
            // Reference data and decisions were checked at their own entry, so the spline
            // can directly construct coefficients without repeating input/output scans.
            return state_.spline.update(state_.durations, state_.waypoints, state_.start_time, state_.bc);
        }

    public:
        SplineOptimizer() = default;
        /** @brief Own maps by value, or borrow an explicitly wrapped longer-lived parameterization. */
        explicit SplineOptimizer(Parameterization parameterization) : parameterization_(std::move(parameterization)) {}
        SplineOptimizer(const SplineOptimizer &) = delete;
        SplineOptimizer &operator=(const SplineOptimizer &) = delete;
        /** @brief Transfer the solve and invalidate all prior views; the source becomes unprepared. */
        SplineOptimizer(SplineOptimizer &&other)
            : parameterization_(std::move(other.parameterization_)), prepared_(std::move(other.prepared_)),
              state_(std::move(other.state_)), buffers_(std::move(other.buffers_))
        {
            other.prepared_.status = {ErrorCode::NotPrepared, "Optimizer is not prepared"};
            other.state_ = WorkingState{};
        }
        SplineOptimizer &operator=(SplineOptimizer &&other)
        {
            if (this != &other)
            {
                parameterization_ = std::move(other.parameterization_);
                prepared_ = std::move(other.prepared_);
                state_ = std::move(other.state_);
                buffers_ = std::move(other.buffers_);
                other.prepared_.status = {ErrorCode::NotPrepared, "Optimizer is not prepared"};
                other.state_ = WorkingState{};
            }
            return *this;
        }

        /** @brief Validate and copy a problem, allocate storage, and build its initial candidate.
         * @param problem Physical input; not retained by reference.
         * @param options Fixed quadrature, built-in costs and optional recording settings.
         * @return Failure leaves the optimizer unprepared with no usable polynomial.
         * @note Allocation occurs here. Borrowed maps must remain valid throughout the solve;
         *       reprepare whenever their mapping behavior changes. */
        template<class Objective,
                 std::enable_if_t<!std::is_same_v<detail::Plain<Objective>, OptimizerOptions>, int> = 0>
        Status prepare(const Problem &problem, const Objective &objective)
        { return prepare(problem, OptimizerOptions{}, objective); }

        /** @brief Prepare all library storage needed by the objective; no objective reference is retained. */
        template<class Objective>
        Status prepare(const Problem &problem, const OptimizerOptions &options, const Objective &objective)
        {
            (void)objective;
            using O = detail::Plain<Objective>;
            validateObjective<O>();
            return prepareImpl(problem, options, SampleStage<O>::present);
        }

        bool isPrepared() const noexcept { return static_cast<bool>(prepared_.status); }
        int decisionDimension() const noexcept { return isPrepared() ? prepared_.layout.dimension : 0; }
        const Problem &problem() const & noexcept { return prepared_.problem; }
        const Problem &problem() const && noexcept = delete;
        const DecisionLayout &layout() const & noexcept { return prepared_.layout; }
        const DecisionLayout &layout() const && noexcept = delete;
        const OptimizationMask &activeMask() const & noexcept { return prepared_.options.mask; }
        const OptimizationMask &activeMask() const && noexcept = delete;
        /** @brief Borrow the last built polynomial; inspect evaluation/update status before use.
         * @note Any evaluation, preparation or movement invalidates previously obtained views/samplers. */
        const Polynomial &polynomial() const & { return state_.spline.polynomial(); }
        const Polynomial &polynomial() const && = delete;
        Polynomial copyPolynomial() const { return polynomial(); }
        /** @brief Borrow current physical parameters for diagnostics without exposing mutable storage.
         * @return Read-only references invalidated by evaluation, preparation, movement or destruction.
         * @note After a failed decode, values may be partial; the returned evaluation status determines validity.
         *       Inspect the original decision vector when replaying a rejected candidate. */
        ParameterView<Spline> parameters() const & { return parameterView(); }
        ParameterView<Spline> parameters() const && = delete;
        /** @brief Return unweighted minimum-derivative energy of the last constructed trajectory.
         * @return Finite energy in physical-time units, or an explicit state/numerical failure.
         * @note Read-only and allocation-free; a valid trial is not necessarily an accepted solution. */
        StatusOr<double> energy() const
        {
            if (!state_.spline.isValid()) return Status{ErrorCode::InvalidState, "Energy requires a valid candidate"};
            const double value = state_.spline.energy();
            if (!std::isfinite(value)) return Status{ErrorCode::NumericalFailure, "Nonfinite spline energy"};
            return value;
        }
        /** @brief Borrow records from the latest successful recording evaluation; otherwise empty. */
        const SampleBuffer &recordedSamples() const & noexcept
        {
            static const SampleBuffer empty;
            return isPrepared() && buffers_.samples_valid ? buffers_.integral_samples : empty;
        }
        const SampleBuffer &recordedSamples() const && noexcept = delete;

        /** @brief Copy the initial coordinates encoded once during preparation.
         * @return Owning vector, independent of subsequent trial evaluations and caller edits.
         * @pre prepare() succeeded and isPrepared() remains true; checked by assertion.
         * @note Allocates the returned vector. Rejected trials do not invalidate preparation;
         *       failed preparation and moving from the optimizer invalidate this precondition. */
        Eigen::VectorXd initialGuess() const
        {
            assert(isPrepared() && "Prepare the optimizer before requesting an initial guess");
            return prepared_.initial_variables;
        }

        /** @brief Rebuild a candidate without evaluating costs; use the solver's final accepted vector after solving.
         * @param x Accepted coordinates, not necessarily the solver's final trial callback.
         * @return Failure invalidates the working polynomial; its storage and preparation remain reusable. */
#if defined(__GNUC__) || defined(__clang__)
        __attribute__((noinline))
#endif
        Status update(const Eigen::VectorXd &x)
        {
            buffers_.samples_valid = false;
            if (!isPrepared() || x.size() != decisionDimension() || !detail::allFinite(x))
            {
                state_.spline.invalidate();
                if (!isPrepared()) return prepared_.status;
                if (x.size() != decisionDimension()) return error(ErrorCode::DimensionMismatch, "Decision vector has wrong size");
                return error(ErrorCode::InvalidInput, "Nonfinite decision vector");
            }
            return buildCandidate(x);
        }

        /** @brief Evaluate one candidate with statically bound named objective stages.
         * @param x Prepared-size finite decision vector.
         * @param[out] gradient Owning decision gradient; resized as needed and cleared on failure. Must not alias x.
         * @param[in,out] objective Named cost members borrowed only for this synchronous call.
         * @param executor Segment executor; parallel calls require a thread-safe integral member.
         * @return Finite cost and success, or infinity with a failure status.
         * @note Prepared fixed-topology evaluation reuses storage; an unmatched owning output may allocate. User callbacks own their allocation
         *       and exception behavior. Preparation must include the objective, with default or explicit options. */
        template<class Objective, class Executor = SerialExecutor>
        EvaluationResult evaluate(const Eigen::VectorXd &x, Eigen::VectorXd &gradient,
                                  Objective &objective, const Executor &executor = {})
        {
            buffers_.samples_valid = false;
            using O = detail::Plain<Objective>;
            validateObjective<O>();
            if (!isPrepared()) return failEvaluation(prepared_.status, gradient);
            gradient.resize(decisionDimension());
            if (x.size() != decisionDimension())
            {
                state_.spline.invalidate();
                return failEvaluation(error(ErrorCode::DimensionMismatch, "Decision vector must match the prepared layout"), gradient);
            }
            if (!detail::allFinite(x))
            {
                state_.spline.invalidate();
                return failEvaluation(error(ErrorCode::InvalidInput, "Nonfinite decision vector"), gradient);
            }
            if constexpr (SampleStage<O>::present)
                if (!prepared_.sample_capability)
                    return failEvaluation(error(ErrorCode::SamplingNotPrepared, "Prepare with the objective before evaluating its sample stage"), gradient);

            // Decode once, then accumulate all independent coefficient/time partials before the adjoint.
            gradient.setZero();
            const Status built = buildCandidate(x);
            if (!built) return failEvaluation(built, gradient);

            auto &b = buffers_;
            b.grad_coeffs.setZero();
            b.grad_times.setZero();
            b.grad_start_time = 0.0;
            double cost = 0.0;
            constexpr bool needs_integral = IntegralStage<O>::present;
            if (const auto status = accumulateCoefficientCosts(objective, executor, cost); !status)
                return failEvaluation(status, gradient);

            // Inactive coefficient-stage costs need no spline adjoint. Test components
            // exactly: a norm can underflow and incorrectly erase nonzero partials.
            bool needs_adjoint = false;
            if constexpr (needs_integral || SampleStage<O>::present || CoefficientStage<O>::present)
                needs_adjoint = (b.grad_coeffs.array() != 0.0).any();
            if (needs_adjoint)
                state_.spline.backward(b.grad_coeffs, b.grad_times, b.grads);
            else
                detail::initializeGradients(b.grad_times, b.grads);

            // Physical costs and auxiliary maps act before ordinary time/space coordinate pullbacks.
            cost += accumulateEnergy();
            if (const auto status = accumulateParameterCosts(objective, cost, gradient); !status)
                return failEvaluation(status, gradient);
            if (const auto status = writeDecisionGradient(x, gradient); !status)
                return failEvaluation(status, gradient);

            // Direct decision penalties add to the mapped gradient; publish only finite results.
            if constexpr (DecisionStage<O>::valid)
            {
                const auto result = detail::toEvaluationResult(
                    objective.decision(DecisionView{x, prepared_.layout}, Eigen::Ref<Eigen::VectorXd>(gradient)));
                if (!result) return failEvaluation(result.status, gradient);
                cost += result.cost;
            }
            if (!std::isfinite(cost) || !detail::allFinite(gradient) || !std::isfinite(b.grad_start_time))
                return failEvaluation(error(ErrorCode::NumericalFailure, "Nonfinite objective or decision gradient"), gradient);
            return {cost, Status::success()};
        }

    private:
        /** @brief Accumulate integral, sample and coefficient costs before the spline adjoint.
         * @param[in,out] objective Statically bound callbacks borrowed for this evaluation only.
         * @param executor Segment executor; parallel integral callbacks must be thread-safe.
         * @param[in,out] cost Additive scalar objective; owned partial buffers were cleared by evaluate().
         * @note Uses the current candidate and prepared buffers; no library allocation occurs.
         * @return Success with the documented output, or an explicit input/state/numerical failure. */
        template<class Objective, class Executor>
        Status accumulateCoefficientCosts(Objective &objective, const Executor &executor, double &cost)
        {
            using O = detail::Plain<Objective>;
            auto &b = buffers_;
            const bool record_samples = SampleStage<O>::present || prepared_.options.record_samples;
            constexpr bool needs_integral = IntegralStage<O>::present;
            if constexpr (needs_integral)
            {
                const auto result = evaluateIntegral(objective, executor, record_samples);
                if (!result) return result.status;
                cost += result.cost;
            }
            else if (record_samples)
            {
                recordSamples(executor);
            }
            if constexpr (SampleStage<O>::valid)
            {
                b.sample_position_grad_buffer.setZero();
                b.sample_time_grad_buffer.setZero();
                SampleGradientView<Spline> output{b.sample_position_grad_buffer, b.sample_time_grad_buffer};
                const auto result = detail::toEvaluationResult(objective.sample(b.integral_samples, output));
                if (!result) return result.status;
                cost += result.cost;
                accumulateSampleCostGradients();
            }
            if constexpr (CoefficientStage<O>::valid)
            {
                auto input = parameterView();
                CoefficientGradientView<Spline> output{b.grad_coeffs, b.grad_times, b.grad_start_time};
                const auto result = detail::toEvaluationResult(objective.coefficient(input, output));
                if (!result) return result.status;
                cost += result.cost;
            }
            return Status::success();
        }

        /** @brief Add physical-parameter costs and reverse auxiliary maps after the spline adjoint.
         * @param[in,out] objective Borrowed callbacks for direct physical-parameter costs.
         * @param[in,out] cost Additive scalar objective.
         * @param[in,out] gradient Pre-sized decision gradient; writes only auxiliary coordinates here.
         * @note Mutates owned physical gradients before ordinary time/space map pullbacks.
         * @return Success with the documented output, or an explicit input/state/numerical failure. */
        template<class Objective>
        Status accumulateParameterCosts(Objective &objective, double &cost, Eigen::VectorXd &gradient)
        {
            using O = detail::Plain<Objective>;
            auto &b = buffers_;
            const double duration_weight = prepared_.options.duration_weight;
            if (duration_weight != 0.0)
            {
                b.grads.durations.array() += duration_weight;
                cost += duration_weight * std::accumulate(state_.durations.begin(), state_.durations.end(), 0.0);
            }
            if constexpr (ParameterStage<O>::valid)
            {
                auto input = parameterView();
                auto output = parameterGradient();
                const auto result = detail::toEvaluationResult(objective.parameter(input, output));
                if (!result) return result.status;
                cost += result.cost;
            }
            if constexpr (has_auxiliary)
            {
                b.auxiliary_grad_buffer.setZero();
                auto input = parameterView();
                auto output = parameterGradient();
                const Status status = auxiliaryMap().backward(state_.auxiliary_vars, input, output,
                                                Eigen::Ref<Eigen::VectorXd>(b.auxiliary_grad_buffer));
                if (!status) return status;
                gradient.segment(prepared_.layout.auxiliary_offset, b.auxiliary_grad_buffer.size()) = b.auxiliary_grad_buffer;
            }
            return Status::success();
        }

        template<class Objective, class Executor>
        EvaluationResult evaluateIntegral(Objective &objective, const Executor &executor, bool record_samples)
        {
            using O = detail::Plain<Objective>;
            EvaluationResult result{0.0, Status::success()};
            if constexpr (IntegralStage<O>::valid)
            {
                auto &integral = detail::unwrap(objective.integral);
                using Cost = detail::Plain<decltype(integral)>;
                constexpr int order = detail::IntegralDerivativeOrder<Cost>::value;
                static_assert(order >= 0 && order <= 4, "Integral kDerivativeOrder must be between zero and four");
                if constexpr (order >= 0 && order <= 4)
                {
                    result = accumulateIntegralCost<order>(integral, record_samples, executor);
                    buffers_.samples_valid = record_samples && result.status.ok();
                }
            }
            return result;
        }

        /** @brief Add energy directly to the prepared physical gradients.
         * @return Weighted energy; the candidate was successfully built before this stage.
         * @note No temporary gradient, resizing, or repeated runtime validity check. */
        double accumulateEnergy()
        {
            const double weight = prepared_.options.energy_weight;
            if (weight == 0.0) return 0.0;
            state_.spline.energyGradientAdd(buffers_.grads, weight);
            return weight * state_.spline.energy();
        }

        Status writeDecisionGradient(const Eigen::VectorXd &x, Eigen::VectorXd &gradient) const
        {
            const auto &g = buffers_.grads;
            for (const auto &variable : prepared_.layout.durations)
            {
                const double tau = x(variable.offset);
                // Auxiliary pullbacks return derivatives in the original mapped coordinates.
                // A time map using its physical argument must see the duration before apply().
                double mapped_duration = state_.durations[variable.segment_index];
                if constexpr (has_auxiliary)
                    mapped_duration = state_.mapped_durations(variable.offset);
                const auto result = timeMap().backward(tau, mapped_duration, g.durations(variable.segment_index));
                if (const auto status = detail::mappingStatus(result); !status) return status;
                gradient(variable.offset) = detail::mappingValue(result);
            }
            for (const auto &variable : prepared_.layout.waypoints)
            {
                const Vector physical = g.waypoints.row(variable.point_index).transpose();
                const auto status = detail::invokeMapping([&] {
                    return spaceMap().backward(x.segment(variable.offset, variable.dof), physical,
                        variable.point_index, gradient.segment(variable.offset, variable.dof));
                });
                if (!status) return status;
            }
            int offset = prepared_.layout.boundary_offset;
            forEachBoundary(g.boundary, [&](const auto &value) {
                gradient.template segment<DIM>(offset) = value;
                offset += DIM;
            });
            return Status::success();
        }

        /** @brief Reverse recorded position/trajectory-time partials into owned coefficient/time buffers.
         * @note Requires sample-capable preparation and callback outputs with their fixed shapes.
         *       Reuses exclusive buffers without allocation; explicit time partials are always accumulated. */
        void accumulateSampleCostGradients()
        {
            const auto &samples = buffers_.integral_samples;
            const auto &sample_position_gradients = buffers_.sample_position_grad_buffer;
            const auto &sample_time_gradients = buffers_.sample_time_grad_buffer;
            auto &grad_coeffs = buffers_.grad_coeffs;
            auto &grad_times = buffers_.grad_times;
            auto &trajectory_time_grad = buffers_.trajectory_time_grad_buffer;
            const Eigen::Index sample_count = static_cast<Eigen::Index>(samples.size());
            trajectory_time_grad.setZero();

            for (Eigen::Index sample_idx = 0; sample_idx < sample_count; ++sample_idx)
            {
                const IntegralSample &sample = samples[sample_idx];
                const int segment_index = sample.point.segment_index;
                const int base_row = segment_index * SplineType::kCoefficientCount;
                const Vector grad_position = sample_position_gradients.row(sample_idx).transpose();

                // Exact component tests preserve tiny nonzero partials and propagate NaNs.
                if ((grad_position.array() != 0.0).any())
                {
                    grad_coeffs.template block<SplineType::kCoefficientCount, DIM>(base_row, 0).noalias() +=
                        sample.b_p.transpose() * grad_position.transpose();
                    grad_times(segment_index) += grad_position.dot(sample.v) * sample.point.alpha;
                }

                const double grad_time = sample_time_gradients(sample_idx);
                grad_times(segment_index) += grad_time * sample.point.alpha;
                trajectory_time_grad(segment_index) += grad_time;
            }

            accumulateTrajectoryTimeGradients();
        }

        /** @brief Reverse explicit trajectory-time partials to earlier durations and the common start time.
         * @note Each buffer entry sums the current stage's partials within one segment. Called once
         *       per stage, after local-duration terms are added; preserves reverse accumulation order. */
        void accumulateTrajectoryTimeGradients()
        {
            double accumulator = 0.0;
            for (int i = prepared_.num_segments - 1; i > 0; --i)
            {
                accumulator += buffers_.trajectory_time_grad_buffer(i);
                buffers_.grad_times(i - 1) += accumulator;
            }
            buffers_.grad_start_time += buffers_.trajectory_time_grad_buffer.sum();
        }

        /** @brief Express a segment's coefficients on the unit time interval.
         * @param segment Valid candidate segment index.
         * @param[out] coefficients Fixed-size normalized coefficient block.
         * @param[out] powers Ascending physical duration powers used by the pullback and records.
         * @note Reads the current candidate only; separate outputs permit parallel segment execution. */
        void prepareIntegralSegment(int segment, SegmentCoefficients &coefficients, SegmentPowers &powers) const
        {
            const double duration = state_.durations[segment];
            powers(0) = 1.0;
            for (int k = 1; k < Spline::kCoefficientCount; ++k) powers(k) = powers(k - 1) * duration;
            const auto source = state_.spline.polynomial().coefficients().template middleRows<Spline::kCoefficientCount>(
                segment * Spline::kCoefficientCount);
            for (int k = 0; k < Spline::kCoefficientCount; ++k) coefficients.row(k) = powers(k) * source.row(k);
        }

        /** @brief Store one prepared quadrature point for a later sample cost or diagnostics.
         * @param point Physical location with its segment and quadrature indices.
         * @param powers Duration powers mapping the position basis to physical time.
         * @param position Physical position. @param velocity Physical velocity per second.
         * @note Writes one disjoint record in preallocated storage; no shared accumulation. */
        void recordSample(const IntegralPointInfo &point, const SegmentPowers &powers,
                          const Vector &position, const Vector &velocity)
        {
            auto &sample = buffers_.integral_samples[point.segment_index * (point.step_count + 1) + point.step_index];
            sample.point = point;
            sample.trap_weight = point.isSegmentStart() || point.isSegmentEnd() ? 0.5 : 1.0;
            sample.b_p = prepared_.basis[point.step_index].row(0).array() * powers.transpose().array();
            sample.p = position;
            sample.v = velocity;
        }

        /** @brief Record positions and velocities when no integral objective is present.
         * @param executor Segment executor with disjoint writes to prepared sample storage.
         * @note Requires a valid candidate and sample-capable preparation. No cost/gradient work occurs. */
        template<class Executor>
        void recordSamples(const Executor &executor)
        {
            const int steps = prepared_.options.integration_steps;
            const double inverse_steps = 1.0 / steps;
            executor(0, prepared_.num_segments, [&](int segment) {
                SegmentCoefficients coefficients;
                SegmentPowers powers;
                prepareIntegralSegment(segment, coefficients, powers);
                const double duration = state_.durations[segment];
                for (int k = 0; k <= steps; ++k)
                {
                    const double alpha = k * inverse_steps;
                    const double segment_time = alpha * duration;
                    const Eigen::Matrix<double, 2, DIM> values = prepared_.basis[k].template topRows<2>() * coefficients;
                    const IntegralPointInfo point{segment, prepared_.num_segments, k, steps, alpha,
                        duration, duration * inverse_steps, segment_time, state_.spline.breakpoints()[segment] + segment_time};
                    recordSample(point, powers, values.row(0).transpose(), values.row(1).transpose() / duration);
                }
            });
            buffers_.samples_valid = true;
        }

        /** @brief Integrate a statically bound cost and accumulate its coefficient/time partials.
         * @param[in,out] integral_cost Borrowed integrand with unweighted physical-state partials.
         * @param record_samples Whether to fill already prepared sample storage.
         * @param executor Segment executor with disjoint per-segment writes.
         * @return Trapezoid-rule cost, or the first callback failure in segment order.
         * @note Uses exclusive owned buffers; parallel callbacks must be thread-safe. No allocation. */
        template <int derivative_order, typename IntegralFunc, typename Executor>
#if defined(__GNUC__) || defined(__clang__)
        // Keep statically bound sample/flatness call chains visible inside the quadrature loop.
        // GCC 9 otherwise outlines them after structured state binding, increasing measured planner cost.
        __attribute__((flatten))
#endif
        EvaluationResult accumulateIntegralCost(IntegralFunc &integral_cost, bool record_samples,
                                      const Executor &executor)
        {
            auto &state = state_;
            auto &buffers = buffers_;
            auto &grad_coeffs = buffers.grad_coeffs;
            auto &grad_times = buffers.grad_times;
            std::fill(buffers.segment_results.begin(), buffers.segment_results.end(), EvaluationResult{0.0, Status::success()});

            buffers.trajectory_time_grad_buffer.setZero();

            int K = prepared_.options.integration_steps;
            double inv_K = 1.0 / K;

            executor(0, prepared_.num_segments, [&](int i) {
                double T = state.durations[i];
                double dt = T * inv_K;
                int base_row = i * SplineType::kCoefficientCount;

                SegmentCoefficients coeff_block;
                SegmentPowers time_powers;
                prepareIntegralSegment(i, coeff_block, time_powers);
                const double inverse_time = 1.0 / T;
                const double inverse_time2 = inverse_time * inverse_time;
                const double inverse_time3 = inverse_time2 * inverse_time;
                const double inverse_time4 = inverse_time3 * inverse_time;
                [[maybe_unused]] const double inverse_time5 = inverse_time4 * inverse_time;

                double local_acc_cost = 0.0;
                double local_acc_gdT = 0.0;
                double local_acc_explicit_time_grad = 0.0;

                Eigen::Matrix<double, SplineType::kCoefficientCount, DIM> local_acc_gdC;
                local_acc_gdC.setZero();

                double current_segment_start_time = state.spline.breakpoints()[i];

                for (int k = 0; k <= K; ++k)
                {
                    double alpha = (double)k * inv_K;
                    double t = alpha * T;

                    double weight_trap = (k == 0 || k == K) ? 0.5 : 1.0;
                    double common_weight = weight_trap * dt;

                    double t_global = current_segment_start_time + t;

                    const auto &basis = prepared_.basis[k];

                    static_assert(derivative_order >= 0 && derivative_order <= 4,
                                  "Integral costs support derivative orders zero through four.");
                    IntegralState<DIM> sample_state;
                    auto &p = sample_state.p; auto &v = sample_state.v; auto &a = sample_state.a;
                    auto &j = sample_state.j; auto &s = sample_state.s;
                    [[maybe_unused]] Vector c = Vector::Zero();
                    // One fixed-size product evaluates the requested states and their time drift.
                    const Eigen::Matrix<double, derivative_order + 2, DIM> values =
                        basis.template topRows<derivative_order + 2>() * coeff_block;
                    p = values.row(0).transpose();
                    v = inverse_time * values.row(1).transpose();
                    if constexpr (derivative_order >= 1) a = inverse_time2 * values.row(2).transpose();
                    if constexpr (derivative_order >= 2) j = inverse_time3 * values.row(3).transpose();
                    if constexpr (derivative_order >= 3) s = inverse_time4 * values.row(4).transpose();
                    if constexpr (derivative_order >= 4) c = inverse_time5 * values.row(5).transpose();

                    IntegralGradient<DIM> partials;
                    auto &gp = partials.p; auto &gv = partials.v; auto &ga = partials.a;
                    auto &gj = partials.j; auto &gs = partials.s; auto &gt = partials.trajectory_time;
                    const IntegralPointInfo point{i, prepared_.num_segments, k, K,
                                                  alpha, T, dt, t, t_global};
                    // Successful callbacks guarantee valid partials. Only explicit rejection
                    // stops this segment; the final objective/gradient is checked once outside.
                    const auto evaluated = detail::toEvaluationResult(integral_cost(point, sample_state, partials));
                    if (!evaluated)
                    {
                        Status status = evaluated.status;
                        if (status.index < 0) status.index = i * (K + 1) + k;
                        buffers.segment_results[i] = {std::numeric_limits<double>::infinity(), status};
                        return; // Only this segment stops; workers own disjoint result slots.
                    }
                    const double c_val = evaluated.cost;

                    local_acc_cost += c_val * common_weight;

                    // Gather physical partials once, then accumulate one basis-transpose product.
                    // No norm/tolerance test discards tiny nonzero gradients.
                    Eigen::Matrix<double, derivative_order + 1, DIM> weighted_partials;
                    weighted_partials.row(0) = common_weight * gp.transpose();
                    if constexpr (derivative_order >= 1) weighted_partials.row(1) = (common_weight * inverse_time) * gv.transpose();
                    if constexpr (derivative_order >= 2) weighted_partials.row(2) = (common_weight * inverse_time2) * ga.transpose();
                    if constexpr (derivative_order >= 3) weighted_partials.row(3) = (common_weight * inverse_time3) * gj.transpose();
                    if constexpr (derivative_order >= 4) weighted_partials.row(4) = (common_weight * inverse_time4) * gs.transpose();
                    local_acc_gdC.noalias() += basis.template topRows<derivative_order + 1>().transpose() * weighted_partials;

                    local_acc_gdT += c_val * weight_trap * inv_K;
                    double drift_grad = gp.dot(v);
                    if constexpr (derivative_order >= 1) drift_grad += gv.dot(a);
                    if constexpr (derivative_order >= 2) drift_grad += ga.dot(j);
                    if constexpr (derivative_order >= 3) drift_grad += gj.dot(s);
                    if constexpr (derivative_order >= 4) drift_grad += gs.dot(c);
                    local_acc_gdT += drift_grad * alpha * common_weight;

                    local_acc_gdT += gt * alpha * common_weight;
                    local_acc_explicit_time_grad += gt * common_weight;

                    if (record_samples) recordSample(point, time_powers, p, v);
                }

                buffers.segment_results[i] = {local_acc_cost, Status::success()};

                grad_times(i) += local_acc_gdT;
                buffers.trajectory_time_grad_buffer(i) += local_acc_explicit_time_grad;

                for (int power = 0; power < SplineType::kCoefficientCount; ++power)
                    grad_coeffs.row(base_row + power) += time_powers(power) * local_acc_gdC.row(power);
            });

            // Reduce in segment order after workers finish, then reverse the trajectory-time chain.
            double cost = 0.0;
            for (int i = 0; i < prepared_.num_segments; ++i)
            {
                if (!buffers.segment_results[i]) return buffers.segment_results[i];
                cost += buffers.segment_results[i].cost;
            }
            accumulateTrajectoryTimeGradients();
            return {cost, Status::success()};
        }

    };

    /** @brief Deduce an owned or explicitly borrowed parameterization type. */
    template<class Spline, class Parameterization>
    auto makeOptimizer(Parameterization &&maps)
    { return SplineOptimizer<Spline, std::decay_t<Parameterization>>(std::forward<Parameterization>(maps)); }

    /** @brief Owning finite-difference diagnostics; never part of normal objective evaluation. */
    struct GradientCheckResult
    {
        Status status{ErrorCode::NotPrepared, "No successful evaluation"};
        Eigen::VectorXd analytical, numerical;
        double error_norm = std::numeric_limits<double>::infinity();
        double relative_error = std::numeric_limits<double>::infinity();
        double max_absolute_error = std::numeric_limits<double>::infinity();
        Eigen::Index max_error_index = -1;
        explicit operator bool() const noexcept { return static_cast<bool>(status); }
    };

    /** @brief Compare a complete objective pullback to central differences and restore the base candidate.
     * @param[in,out] optimizer Prepared, exclusively owned workspace.
     * @param x Base decision coordinates.
     * @param[in,out] objective Borrowed costs, valid for all perturbations and the final restoration.
     * @param step Positive perturbation in decision units.
     * @param tolerance Absolute tolerance on the Euclidean gradient error.
     * @return Owning diagnostics. Failed candidates propagate status; the base is restored even on failure.
     * @note This diagnostic allocates and evaluates the objective repeatedly. User exceptions propagate. */
    template<class Optimizer, class Objective>
    GradientCheckResult checkGradients(Optimizer &optimizer, const Eigen::VectorXd &x, Objective &objective,
                                      double step = 1e-6, double tolerance = 1e-4)
    {
        GradientCheckResult result;
        if (!std::isfinite(step) || step <= 0.0 || !std::isfinite(tolerance) || tolerance <= 0.0)
        {
            result.status = {ErrorCode::InvalidInput, "Gradient-check step and tolerance must be finite and positive"};
            return result;
        }
        result.analytical.resize(x.size());
        result.numerical.setZero(x.size());
        const auto base = optimizer.evaluate(x, result.analytical, objective);
        result.status = base.status;
        if (!base) return result;
        Eigen::VectorXd perturbed = x, scratch(x.size());
        for (Eigen::Index i = 0; i < x.size(); ++i)
        {
            perturbed(i) = x(i) + step;
            const auto plus = optimizer.evaluate(perturbed, scratch, objective);
            if (!plus) { result.status = plus.status; break; }
            perturbed(i) = x(i) - step;
            const auto minus = optimizer.evaluate(perturbed, scratch, objective);
            if (!minus) { result.status = minus.status; break; }
            result.numerical(i) = (plus.cost - minus.cost) / (2.0 * step);
            perturbed(i) = x(i);
        }
        const auto restored = optimizer.evaluate(x, scratch, objective);
        if (!restored) result.status = restored.status;
        if (!result.status) return result;
        const Eigen::VectorXd difference = result.analytical - result.numerical;
        result.error_norm = difference.norm();
        result.relative_error = result.error_norm / std::max(1e-9, result.analytical.norm());
        result.max_absolute_error = difference.size() ? difference.cwiseAbs().maxCoeff(&result.max_error_index) : 0.0;
        if (result.error_norm > tolerance)
            result.status = {ErrorCode::GradientMismatch, "Analytical and numerical gradients differ"};
        return result;
    }
}
#endif
