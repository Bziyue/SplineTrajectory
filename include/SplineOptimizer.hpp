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
#include <stdexcept>
#include <vector>
#include <cassert>
#include <cstdint>
#include <functional>
#include <optional>
#include <type_traits>
#include <utility>

namespace SplineTrajectory
{
    /** @brief Failure categories shared by optimizer preparation and evaluation. */
    enum class OptimizationError
    {
        None,
        NotPrepared,
        InvalidInput,
        DimensionMismatch,
        NumericalFailure,
        SamplingNotPrepared,
        GradientMismatch
    };

    /** @brief Allocation-free status; message refers to immutable library text. */
    struct OptimizationStatus
    {
        OptimizationError code = OptimizationError::NotPrepared;
        int index = -1;
        const char *message = "Optimizer is not prepared";
        OptimizationStatus() = default;
        OptimizationStatus(OptimizationError value, const char *text, int location = -1) noexcept
            : code(value), index(location), message(text) {}
        explicit operator bool() const noexcept { return code == OptimizationError::None; }
        static OptimizationStatus success() noexcept { return {OptimizationError::None, ""}; }
    };

    /** @brief Cost and status. Failure carries infinity; the decision gradient is cleared. */
    struct EvaluationResult
    {
        double cost = std::numeric_limits<double>::infinity();
        OptimizationStatus status;
        explicit operator bool() const noexcept { return static_cast<bool>(status); }
    };

    /** @brief Integration location on the source time axis, measured in seconds. */
    struct IntegralPointInfo
    {
        int segment_index = 0, segment_count = 0, step_index = 0, step_count = 0;
        double alpha = 0.0, segment_duration = 0.0, step_size = 0.0;
        double local_time = 0.0, global_time = 0.0;
        bool isSegmentStart() const noexcept { return step_index == 0; }
        bool isSegmentEnd() const noexcept { return step_index == step_count; }
        bool isTrajectoryStart() const noexcept { return segment_index == 0 && isSegmentStart(); }
        bool isTrajectoryEnd() const noexcept { return segment_index + 1 == segment_count && isSegmentEnd(); }
        int interiorBoundaryIndex() const noexcept
        {
            if (isSegmentEnd() && segment_index + 1 < segment_count) return segment_index;
            return isSegmentStart() && segment_index > 0 ? segment_index - 1 : -1;
        }
    };

    /** @brief Physical-time derivatives in the caller's coordinate frame. */
    template<int DIM>
    struct SampleState
    {
        using Vector = Eigen::Matrix<double, DIM, 1>;
        Vector p = Vector::Zero(), v = Vector::Zero(), a = Vector::Zero();
        Vector j = Vector::Zero(), s = Vector::Zero();
    };

    /** @brief Unweighted integrand partials, cleared for each sample.
     * @note time is the explicit global-time partial; quadrature and state drift belong to the optimizer. */
    template<int DIM>
    struct SampleGradient
    {
        using Vector = Eigen::Matrix<double, DIM, 1>;
        Vector p = Vector::Zero(), v = Vector::Zero(), a = Vector::Zero();
        Vector j = Vector::Zero(), s = Vector::Zero();
        double time = 0.0;
    };

    /** @brief Direct physical duration coordinates; candidates must remain positive. */
    struct IdentityTimeMap
    {
        double toTime(double value) const noexcept { return value; }
        double toTau(double value) const noexcept { return value; }
        double backward(double, double, double gradient) const noexcept { return gradient; }
    };

    /** @brief Smooth positive duration map with an analytic inverse and pullback. */
    struct QuadInvTimeMap
    {
        double toTime(double tau) const noexcept
        {
            return tau > 0.0 ? (0.5 * tau + 1.0) * tau + 1.0
                             : 1.0 / ((0.5 * tau - 1.0) * tau + 1.0);
        }
        double toTau(double duration) const noexcept
        {
            return duration > 1.0 ? std::sqrt(2.0 * duration - 1.0) - 1.0
                                  : 1.0 - std::sqrt(2.0 / duration - 1.0);
        }
        double backward(double tau, double, double gradient) const noexcept
        {
            if (tau > 0.0) return gradient * (tau + 1.0);
            const double denominator = (0.5 * tau - 1.0) * tau + 1.0;
            return gradient * (1.0 - tau) / (denominator * denominator);
        }
    };

    /** @brief Cartesian waypoint coordinates with an allocation-free pullback. */
    template<int DIM>
    struct IdentitySpatialMap
    {
        using Vector = Eigen::Matrix<double, DIM, 1>;
        int dimension(int) const noexcept { return DIM; }
        Vector toPhysical(const Eigen::Ref<const Eigen::VectorXd> &value, int) const { return value; }
        Vector toUnconstrained(const Vector &point, int) const { return point; }
        void backwardInto(const Eigen::Ref<const Eigen::VectorXd> &, const Vector &gradient,
                          int, Eigen::Ref<Eigen::VectorXd> destination) const { destination = gradient; }
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
    /** @brief Active coordinates; empty masks select all durations and only interior waypoints. */
    struct OptimizationMask
    {
        std::vector<std::uint8_t> time, waypoints;
        BoundaryDerivativeMask start, end;
    };

    /** @brief Numerical input copied by prepare(); times use seconds on one common axis. */
    template<class Spline>
    struct SplineProblem
    {
        std::vector<double> durations;
        typename Spline::CoefficientMatrix waypoints;
        BoundaryConditions<Spline::kDimension> boundary;
        double start_time = 0.0;
        std::optional<OptimizationMask> mask;
    };

    /** @brief Fixed solve options; changing topology or recording requires preparation. */
    struct OptimizerOptions
    {
        double energy_weight = 0.0;
        int integration_steps = 64;
        bool record_samples = false;
    };

    struct TimeVariable { int segment_index = 0, offset = 0; };
    struct PointVariable { int point_index = 0, offset = 0, dof = 0; };
    /** @brief Read-only decision layout, valid until preparation, movement or destruction. */
    struct DecisionLayout
    {
        std::vector<TimeVariable> time;
        std::vector<PointVariable> waypoints;
        int boundary_offset = 0, auxiliary_offset = 0, dimension = 0;
    };

    /** @brief Exclusive mutable physical parameters passed to an auxiliary map.
     * @note Existing topology must be preserved; references expire on return from apply(). */
    template<class Spline>
    struct MutableParameters
    {
        std::vector<double> &durations;
        typename Spline::CoefficientMatrix &waypoints;
        BoundaryConditions<Spline::kDimension> &boundary;
        double &start_time;
    };

    /** @brief Borrowed physical state for coefficient/parameter costs and auxiliary pullbacks. */
    template<class Spline>
    struct ParameterView
    {
        const typename Spline::Polynomial &polynomial;
        const std::vector<double> &durations;
        const typename Spline::CoefficientMatrix &waypoints;
        const BoundaryConditions<Spline::kDimension> &boundary;
        double start_time;
    };

    /** @brief Additive coefficient and independent-time partials before the spline adjoint.
     * @note durations differentiates time with local power coefficients held fixed. */
    template<class Spline>
    struct CoefficientGradient
    {
        Eigen::Ref<typename Spline::CoefficientMatrix> coefficients;
        Eigen::Ref<Eigen::VectorXd> durations;
        double &start_time;
    };

    /** @brief Additive physical parameter gradients after the spline adjoint. */
    template<class Spline>
    struct ParameterGradient
    {
        Eigen::Ref<typename Spline::CoefficientMatrix> inner_points;
        Eigen::Ref<Eigen::VectorXd> durations;
        typename Spline::BoundaryGradient &start, &end;
        double &start_time;
        /** @brief Add a waypoint partial using the global index, including both endpoints. */
        void addWaypoint(int index, const typename Spline::Vector &gradient)
        {
            if (index < 0 || index > inner_points.rows() + 1)
                throw std::out_of_range("Waypoint gradient index is outside the trajectory");
            if (index == 0) start.p += gradient;
            else if (index == inner_points.rows() + 1) end.p += gradient;
            else inner_points.row(index - 1) += gradient.transpose();
        }
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

        template<class T, class = void> struct HasDuration : std::false_type {};
        template<class T> struct HasDuration<T, std::void_t<decltype(std::declval<T &>().duration)>> : std::true_type {};
        template<class T, class = void> struct HasIntegral : std::false_type {};
        template<class T> struct HasIntegral<T, std::void_t<decltype(std::declval<T &>().integral)>> : std::true_type {};
        template<class T, class = void> struct HasSample : std::false_type {};
        template<class T> struct HasSample<T, std::void_t<decltype(std::declval<T &>().sample)>> : std::true_type {};
        template<class T, class = void> struct HasCoefficient : std::false_type {};
        template<class T> struct HasCoefficient<T, std::void_t<decltype(std::declval<T &>().coefficient)>> : std::true_type {};
        template<class T, class = void> struct HasParameter : std::false_type {};
        template<class T> struct HasParameter<T, std::void_t<decltype(std::declval<T &>().parameter)>> : std::true_type {};
        template<class T, class = void> struct HasDecision : std::false_type {};
        template<class T> struct HasDecision<T, std::void_t<decltype(std::declval<T &>().decision)>> : std::true_type {};
        template<class T, class = void> struct HasBeginEvaluation : std::false_type {};
        template<class T> struct HasBeginEvaluation<T, std::void_t<decltype(std::declval<T &>().beginEvaluation())>> : std::true_type {};
        template<class T, class = void> struct IntegralDerivativeOrder : std::integral_constant<int, 4> {};
        template<class T> struct IntegralDerivativeOrder<T, std::void_t<decltype(T::kDerivativeOrder)>>
            : std::integral_constant<int, T::kDerivativeOrder> {};

        struct ZeroIntegral
        {
            static constexpr int kDerivativeOrder = 0;
            template<int DIM>
            double operator()(const IntegralPointInfo &, const SampleState<DIM> &, SampleGradient<DIM> &) const
            { return 0.0; }
        };
    }

    /** @brief Reusable, exclusively owned spline optimization workspace.
     * @tparam Spline Concrete minimum-derivative spline; dimension and degree are inferred.
     * @tparam Parameterization Owned maps, or an explicit reference_wrapper to longer-lived maps.
     * @note One instance per concurrent solve. No objective is retained between synchronous calls.
     *       prepare(), assignment and movement invalidate all borrowed state views. */
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
        using Problem = SplineProblem<Spline>;
        using Polynomial = typename Spline::Polynomial;
        using Status = OptimizationStatus;
        using ErrorCode = OptimizationError;
        using Gradients = typename Spline::Gradients;
        using SampleGradMatrix = Eigen::Matrix<double, DIM, Eigen::Dynamic>;

        /** @brief One recorded quadrature point, borrowed until the next evaluation or preparation. */
        struct IntegralSample
        {
            IntegralPointInfo point;
            double trap_weight = 0.0;
            Eigen::Matrix<double, 1, Spline::kCoefficientCount> b_p;
            Vector p = Vector::Zero(), v = Vector::Zero();
        };
        using SampleBuffer = SplineVector<IntegralSample>;

    private:
        // Numeric kernel aliases are local; public dimensions come from Spline.
        using IntegralSampleBuffer = SampleBuffer;
        struct IntegralBasis
        {
            Eigen::Matrix<double, 1, Spline::kCoefficientCount> p, v, a, j, s, c;
        };
        struct PreparedData
        {
            Problem problem;
            OptimizerOptions options;
            OptimizationMask mask;
            DecisionLayout layout;
            Eigen::VectorXd initial_variables;
            SplineVector<IntegralBasis> basis;
            int num_segments = 0;
            bool sample_capability = false;
            Status status;
        };
        struct WorkingState
        {
            Spline spline;
            std::vector<double> durations;
            CoefficientMatrix waypoints;
            BoundaryConditions<DIM> bc;
            double start_time = 0.0;
            Eigen::VectorXd auxiliary_vars;
        };
        struct EvaluationBuffers
        {
            CoefficientMatrix grad_coeffs;
            Eigen::VectorXd grad_times, global_time_grad_buffer, auxiliary_grad_buffer;
            double grad_start_time = 0.0;
            Gradients grads, energy_grads;
            SampleGradMatrix sample_position_grad_buffer;
            Eigen::VectorXd sample_time_grad_buffer;
            std::vector<double> segment_begin_times, segment_cost_buffer;
            SampleBuffer integral_samples;
            bool samples_valid = false;
            void prepare(int pieces, int auxiliary, Eigen::Index samples)
            {
                samples_valid = false;
                grad_coeffs.resize(pieces * Spline::kCoefficientCount, DIM);
                grad_times.resize(pieces);
                global_time_grad_buffer.resize(pieces);
                auxiliary_grad_buffer.resize(auxiliary);
                grads.resetTopology(pieces);
                energy_grads.resetTopology(pieces);
                segment_begin_times.resize(pieces);
                segment_cost_buffer.resize(pieces);
                integral_samples.resize(static_cast<std::size_t>(samples));
                sample_position_grad_buffer.resize(DIM, samples);
                sample_time_grad_buffer.resize(samples);
            }
        };

        Parameterization parameterization_;
        PreparedData prepared_;
        WorkingState state_;
        EvaluationBuffers buffers_;
        Status last_status_;

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
            prepared_.status = last_status_ = status;
            state_.spline = Spline{};
            return status;
        }
        EvaluationResult failEvaluation(Status status, Eigen::VectorXd &gradient)
        {
            last_status_ = status;
            buffers_.samples_valid = false;
            gradient.setZero();
            return {std::numeric_limits<double>::infinity(), status};
        }
        static bool finiteBoundary(const BoundaryConditions<DIM> &boundary)
        {
            bool valid = boundary.start_velocity.allFinite() && boundary.end_velocity.allFinite();
            if constexpr (Spline::kDegree >= 5)
                valid = valid && boundary.start_acceleration.allFinite() && boundary.end_acceleration.allFinite();
            if constexpr (Spline::kDegree >= 7)
                valid = valid && boundary.start_jerk.allFinite() && boundary.end_jerk.allFinite();
            return valid;
        }
        template<class Function>
        void forEachBoundary(Function &&function) const
        {
            const auto &mask = prepared_.mask;
            if (mask.start.v) function(0);
            if constexpr (Spline::kDegree >= 5) if (mask.start.a) function(1);
            if constexpr (Spline::kDegree >= 7) if (mask.start.j) function(2);
            if (mask.end.v) function(3);
            if constexpr (Spline::kDegree >= 5) if (mask.end.a) function(4);
            if constexpr (Spline::kDegree >= 7) if (mask.end.j) function(5);
        }
        template<class Boundary>
        static decltype(auto) boundaryVector(Boundary &boundary, int slot)
        {
            switch (slot)
            {
            case 0: return (boundary.start_velocity);
            case 1: return (boundary.start_acceleration);
            case 2: return (boundary.start_jerk);
            case 3: return (boundary.end_velocity);
            case 4: return (boundary.end_acceleration);
            default: return (boundary.end_jerk);
            }
        }
        const Vector &boundaryGradient(int slot) const
        {
            const auto &endpoint = slot < 3 ? buffers_.grads.start : buffers_.grads.end;
            if constexpr (Spline::kDegree >= 7) if (slot % 3 == 2) return endpoint.j;
            if constexpr (Spline::kDegree >= 5) if (slot % 3 == 1) return endpoint.a;
            return endpoint.v;
        }
        ParameterView<Spline> parameterView() const
        {
            return {state_.spline.polynomial(), state_.durations, state_.waypoints, state_.bc, state_.start_time};
        }
        ParameterGradient<Spline> parameterGradient()
        {
            auto &gradient = buffers_.grads;
            return {gradient.inner_points, gradient.durations, gradient.start, gradient.end, buffers_.grad_start_time};
        }

        // Initial coordinates belong to the prepared problem, not the changing trial state.
        Eigen::VectorXd encodeInitialGuess() const
        {
            const auto &reference = prepared_.problem;
            Eigen::VectorXd x(prepared_.layout.dimension);
            for (const auto &variable : prepared_.layout.time)
                x(variable.offset) = timeMap().toTau(reference.durations[variable.segment_index]);
            for (const auto &variable : prepared_.layout.waypoints)
            {
                const auto value = spaceMap().toUnconstrained(reference.waypoints.row(variable.point_index).transpose(), variable.point_index);
                if (value.size() != variable.dof) throw std::invalid_argument("Spatial map initial dimension mismatch");
                x.segment(variable.offset, variable.dof) = value;
            }
            int offset = prepared_.layout.boundary_offset;
            forEachBoundary([&](int slot) {
                x.template segment<DIM>(offset) = boundaryVector(reference.boundary, slot);
                offset += DIM;
            });
            if constexpr (has_auxiliary)
            {
                const auto value = auxiliaryMap().initial(reference);
                if (value.size() != state_.auxiliary_vars.size()) throw std::invalid_argument("Auxiliary map initial dimension mismatch");
                x.segment(prepared_.layout.auxiliary_offset, value.size()) = value;
            }
            if (!x.allFinite()) throw std::invalid_argument("Initial decision coordinates are nonfinite");
            return x;
        }

        Status prepareImpl(const Problem &problem, OptimizerOptions options, bool need_samples) try
        {
            // Copy before modifying state: problem may be this optimizer's read-only problem view.
            Problem snapshot = problem;
            prepared_.status = Status{};
            if (snapshot.durations.empty() ||
                snapshot.durations.size() > static_cast<std::size_t>(std::numeric_limits<int>::max() / Spline::kCoefficientCount))
                return failPreparation(error(ErrorCode::InvalidInput, "Invalid segment count"));
            const int n = static_cast<int>(snapshot.durations.size());
            if (snapshot.waypoints.rows() != n + 1 || snapshot.waypoints.cols() != DIM)
                return failPreparation(error(ErrorCode::DimensionMismatch, "Waypoint shape must be (segments + 1) by dimension"));
            if (!snapshot.waypoints.allFinite() || !finiteBoundary(snapshot.boundary) || !std::isfinite(snapshot.start_time))
                return failPreparation(error(ErrorCode::InvalidInput, "Nonfinite reference state"));
            for (double duration : snapshot.durations)
                if (!std::isfinite(duration) || duration <= 0.0)
                    return failPreparation(error(ErrorCode::InvalidInput, "Durations must be finite and positive"));
            if (options.integration_steps <= 0 || options.integration_steps == std::numeric_limits<int>::max() ||
                !std::isfinite(options.energy_weight) || options.energy_weight < 0.0)
                return failPreparation(error(ErrorCode::InvalidInput, "Invalid integration steps or energy weight"));

            prepared_.problem = std::move(snapshot);
            prepared_.num_segments = n;
            prepared_.options = options;
            prepared_.sample_capability = need_samples || options.record_samples;
            prepared_.mask = prepared_.problem.mask.value_or(OptimizationMask{});
            auto &mask = prepared_.mask;
            if (mask.time.empty()) mask.time.assign(n, 1);
            if (mask.waypoints.empty())
            {
                mask.waypoints.assign(n + 1, 1);
                mask.waypoints.front() = mask.waypoints.back() = 0;
            }
            if (mask.time.size() != static_cast<std::size_t>(n) || mask.waypoints.size() != static_cast<std::size_t>(n + 1))
                return failPreparation(error(ErrorCode::DimensionMismatch, "Optimization mask does not match topology"));
            if constexpr (Spline::kDegree < 5)
                if (mask.start.a || mask.end.a)
                    return failPreparation(error(ErrorCode::InvalidInput, "This spline has no acceleration boundary variables"));
            if constexpr (Spline::kDegree < 7)
                if (mask.start.j || mask.end.j)
                    return failPreparation(error(ErrorCode::InvalidInput, "This spline has no jerk boundary variables"));

            auto &layout = prepared_.layout;
            layout.time.clear();
            layout.waypoints.clear();
            std::int64_t offset = 0;
            for (int i = 0; i < n; ++i)
                if (mask.time[i]) layout.time.push_back({i, static_cast<int>(offset++)});
            for (int i = 0; i <= n; ++i)
            {
                if (!mask.waypoints[i]) continue;
                const int dof = spaceMap().dimension(i);
                if (dof <= 0 || offset + dof > std::numeric_limits<int>::max())
                    return failPreparation(error(ErrorCode::InvalidInput, "Invalid spatial decision dimension"));
                layout.waypoints.push_back({i, static_cast<int>(offset), dof});
                offset += dof;
            }
            layout.boundary_offset = static_cast<int>(offset);
            forEachBoundary([&](int) { offset += DIM; });
            const int auxiliary = auxiliaryDimension();
            if (auxiliary < 0 || offset + auxiliary > std::numeric_limits<int>::max())
                return failPreparation(error(ErrorCode::InvalidInput, "Invalid auxiliary decision dimension"));
            layout.auxiliary_offset = static_cast<int>(offset);
            layout.dimension = static_cast<int>(offset + auxiliary);
            const std::int64_t sample_count = prepared_.sample_capability ?
                static_cast<std::int64_t>(n) * (options.integration_steps + 1) : 0;
            if (sample_count > std::numeric_limits<int>::max())
                return failPreparation(error(ErrorCode::InvalidInput, "Sample count exceeds supported indices"));

            prepared_.basis.resize(static_cast<std::size_t>(options.integration_steps) + 1);
            for (int k = 0; k <= options.integration_steps; ++k)
            {
                auto &basis = prepared_.basis[k];
                Spline::computeBasisFunctions(static_cast<double>(k) / options.integration_steps,
                                             basis.p, basis.v, basis.a, basis.j, basis.s, basis.c);
            }
            buffers_.prepare(n, auxiliary, static_cast<Eigen::Index>(sample_count));
            state_.durations.resize(n);
            state_.waypoints.resize(n + 1, DIM);
            state_.auxiliary_vars.resize(auxiliary);
            prepared_.status = Status::success();
            try
            {
                prepared_.initial_variables = encodeInitialGuess();
                const Status built = buildCandidate(prepared_.initial_variables);
                if (!built) return failPreparation(built);
            }
            catch (const std::invalid_argument &)
            { return failPreparation(error(ErrorCode::InvalidInput, "Parameterization returned invalid initial coordinates")); }
            return last_status_ = Status::success();
        }

        catch (const std::invalid_argument &)
        { return failPreparation(error(ErrorCode::InvalidInput, "Parameterization rejected preparation")); }
        catch (const std::length_error &)
        { return failPreparation(error(ErrorCode::InvalidInput, "Prepared storage exceeds supported sizes")); }

        /** @brief Decode and construct exactly once, rejecting invalid data before coefficient costs. */
        Status buildCandidate(const Eigen::VectorXd &x)
        {
            auto reject = [&](const char *message, int index = -1) {
                state_.spline = Spline{};
                return error(ErrorCode::NumericalFailure, message, index);
            };
            const auto &reference = prepared_.problem;
            state_.durations = reference.durations;
            state_.waypoints = reference.waypoints;
            state_.bc = reference.boundary;
            state_.start_time = reference.start_time;
            try
            {
                for (const auto &variable : prepared_.layout.time)
                    state_.durations[variable.segment_index] = timeMap().toTime(x(variable.offset));
                for (const auto &variable : prepared_.layout.waypoints)
                {
                    const auto point = spaceMap().toPhysical(x.segment(variable.offset, variable.dof), variable.point_index);
                    if (point.size() != DIM) return reject("Spatial map returned the wrong physical dimension");
                    state_.waypoints.row(variable.point_index) = point.transpose();
                }
                int offset = prepared_.layout.boundary_offset;
                forEachBoundary([&](int slot) {
                    boundaryVector(state_.bc, slot) = x.template segment<DIM>(offset);
                    offset += DIM;
                });
                if constexpr (has_auxiliary)
                {
                    state_.auxiliary_vars = x.segment(prepared_.layout.auxiliary_offset, state_.auxiliary_vars.size());
                    MutableParameters<Spline> state{state_.durations, state_.waypoints, state_.bc, state_.start_time};
                    auxiliaryMap().apply(state_.auxiliary_vars, state);
                }
            }
            catch (const std::invalid_argument &) { return reject("Parameterization rejected the candidate"); }
            if (state_.durations.size() != static_cast<std::size_t>(prepared_.num_segments) ||
                state_.waypoints.rows() != prepared_.num_segments + 1 || state_.waypoints.cols() != DIM)
                return reject("Parameterization changed the prepared topology");
            if (!state_.waypoints.allFinite() || !finiteBoundary(state_.bc) || !std::isfinite(state_.start_time))
                return reject("Nonfinite mapped physical state");
            double current = state_.start_time;
            for (int i = 0; i < prepared_.num_segments; ++i)
            {
                const double duration = state_.durations[i];
                const double next = current + duration;
                if (!std::isfinite(duration) || duration <= 0.0 || !std::isfinite(next) || next <= current)
                    return reject("Nonpositive duration or unrepresentable cumulative time knot", i);
                current = next;
            }
            state_.spline.update(state_.durations, state_.waypoints, state_.start_time, state_.bc);
            if (!state_.spline.isValid()) return reject("Spline construction produced nonfinite coefficients");
            return Status::success();
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
              state_(std::move(other.state_)), buffers_(std::move(other.buffers_)), last_status_(other.last_status_)
        {
            other.prepared_.status = other.last_status_ = Status{};
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
                last_status_ = other.last_status_;
                other.prepared_.status = other.last_status_ = Status{};
                other.state_ = WorkingState{};
            }
            return *this;
        }

        /** @brief Validate and copy a problem, allocate storage, and build its initial candidate.
         * @param problem Physical input; not retained by reference.
         * @param options Fixed quadrature, energy and optional recording settings.
         * @return Failure leaves the optimizer unprepared with no usable polynomial.
         * @note Allocation occurs here. Borrowed maps must remain valid throughout the solve;
         *       reprepare whenever their mapping behavior changes. */
        Status prepare(const Problem &problem, OptimizerOptions options = {})
        { return prepareImpl(problem, options, false); }

        /** @brief Also prepare storage required by the objective's sample stage.
         * @param objective Inspected by type only; no objective reference is retained. */
        template<class Objective>
        Status prepare(const Problem &problem, OptimizerOptions options, const Objective &objective)
        {
            (void)objective;
            return prepareImpl(problem, options, detail::HasSample<Objective>::value);
        }

        bool isPrepared() const noexcept { return static_cast<bool>(prepared_.status); }
        int dimension() const noexcept { return isPrepared() ? prepared_.layout.dimension : 0; }
        const Problem &problem() const noexcept { return prepared_.problem; }
        const DecisionLayout &layout() const noexcept { return prepared_.layout; }
        const OptimizationMask &activeMask() const noexcept { return prepared_.mask; }
        Status lastStatus() const noexcept { return last_status_; }
        /** @brief Borrow the last built polynomial; inspect evaluation/updateAccepted status before use.
         * @note Any evaluation, preparation or movement invalidates previously obtained views/samplers. */
        const Polynomial &polynomial() const & { return state_.spline.polynomial(); }
        const Polynomial &polynomial() const && = delete;
        Polynomial copyPolynomial() const { return polynomial(); }
        /** @brief Borrow current physical parameters for diagnostics without exposing mutable storage.
         * @return Read-only references invalidated by evaluation, preparation, movement or destruction.
         * @note After a failed decode, values may be partial; lastStatus() determines validity.
         *       Inspect the original decision vector when replaying a rejected candidate. */
        ParameterView<Spline> parameters() const & { return parameterView(); }
        ParameterView<Spline> parameters() const && = delete;
        /** @brief Return unweighted minimum-derivative energy of the last constructed trajectory.
         * @return Energy in physical-time units, or NaN when the polynomial is invalid.
         * @note Read-only and allocation-free; a valid trial is not necessarily an accepted solution. */
        double energy() const { return state_.spline.energy(); }
        /** @brief Borrow records from the latest successful recording evaluation; otherwise empty. */
        const SampleBuffer &recordedSamples() const noexcept
        {
            static const SampleBuffer empty;
            return isPrepared() && buffers_.samples_valid ? buffers_.integral_samples : empty;
        }

        /** @brief Copy the initial coordinates encoded once during preparation.
         * @return Owning vector, independent of subsequent trial evaluations and caller edits.
         * @throws std::logic_error If preparation has not succeeded.
         * @note Allocates the returned vector; does not repeat potentially expensive inverse maps. */
        Eigen::VectorXd initialGuess() const
        {
            if (!isPrepared()) throw std::logic_error("Prepare the optimizer before requesting an initial guess");
            return prepared_.initial_variables;
        }

        /** @brief Rebuild the external solver's accepted vector without evaluating the objective.
         * @param x Accepted coordinates, not necessarily the solver's final trial callback.
         * @return Failure clears the working polynomial; prepared configuration remains reusable. */
#if defined(__GNUC__) || defined(__clang__)
        __attribute__((noinline))
#endif
        Status updateAccepted(const Eigen::VectorXd &x)
        {
            buffers_.samples_valid = false;
            if (!isPrepared() || x.size() != dimension() || !x.allFinite())
            {
                state_.spline = Spline{};
                return last_status_ = error(ErrorCode::InvalidInput, "Invalid accepted decision vector or unprepared optimizer");
            }
            return last_status_ = buildCandidate(x);
        }

        /** @brief Evaluate one candidate with statically bound named objective stages.
         * @param x Prepared-size finite decision vector.
         * @param[out] gradient Pre-sized decision gradient; cleared on failure, never resized here.
         * @param[in,out] objective Named cost members borrowed only for this synchronous call.
         * @param executor Segment executor; parallel calls require a thread-safe integral member.
         * @return Finite cost and success, or infinity with a failure status.
         * @note Fixed-topology library evaluation does not allocate. User callbacks own their allocation
         *       and exception behavior. A sample member requires prepare(problem, options, objective). */
        template<class Objective, class Executor = SerialExecutor>
        EvaluationResult evaluate(const Eigen::VectorXd &x, Eigen::VectorXd &gradient,
                                  Objective &objective, const Executor &executor = {})
        {
            buffers_.samples_valid = false;
            using O = detail::Plain<Objective>;
            static_assert(detail::HasDuration<O>::value || detail::HasIntegral<O>::value ||
                          detail::HasSample<O>::value || detail::HasCoefficient<O>::value ||
                          detail::HasParameter<O>::value || detail::HasDecision<O>::value ||
                          std::is_same_v<O, NoObjective>,
                          "Objective must declare a supported cost member, or explicitly use NoObjective");
            if (!isPrepared()) return failEvaluation(prepared_.status, gradient);
            if (x.size() != dimension() || gradient.size() != dimension())
            {
                state_.spline = Spline{};
                return failEvaluation(error(ErrorCode::DimensionMismatch, "Decision vector and output gradient must be pre-sized"), gradient);
            }
            if (!x.allFinite())
            {
                state_.spline = Spline{};
                return failEvaluation(error(ErrorCode::InvalidInput, "Nonfinite decision vector"), gradient);
            }
            if constexpr (detail::HasSample<O>::value)
                if (!prepared_.sample_capability)
                    return failEvaluation(error(ErrorCode::SamplingNotPrepared, "Prepare with the objective before evaluating its sample stage"), gradient);
            gradient.setZero();
            const Status built = buildCandidate(x);
            if (!built) return failEvaluation(built, gradient);

            auto &b = buffers_;
            b.grad_coeffs.setZero();
            b.grad_times.setZero();
            b.grad_start_time = 0.0;
            double cost = 0.0;
            if constexpr (detail::HasDuration<O>::value)
            {
                auto &function = detail::unwrap(objective.duration);
                using Function = decltype(function);
                using Output = Eigen::Ref<Eigen::VectorXd>;
                constexpr bool valid = std::is_invocable_r_v<double, Function, const std::vector<double> &, Output>;
                static_assert(valid, "Objective.duration must return double and accept (const vector<double>&, Eigen::Ref<VectorXd>)");
                if constexpr (valid) cost += function(state_.durations, Output(b.grad_times));
            }

            const bool record_samples = detail::HasSample<O>::value || prepared_.options.record_samples;
            constexpr bool needs_integral = detail::HasIntegral<O>::value;
            if constexpr (needs_integral)
                cost += evaluateIntegral(detail::unwrap(objective.integral), executor, record_samples);
            else if (record_samples)
            {
                detail::ZeroIntegral integral;
                cost += evaluateIntegral(integral, executor, true);
            }
            if constexpr (detail::HasSample<O>::value)
            {
                b.sample_position_grad_buffer.setZero();
                b.sample_time_grad_buffer.setZero();
                auto &function = detail::unwrap(objective.sample);
                using PositionOutput = Eigen::Ref<SampleGradMatrix>;
                using TimeOutput = Eigen::Ref<Eigen::VectorXd>;
                constexpr bool valid = std::is_invocable_r_v<double, decltype(function), const SampleBuffer &, PositionOutput, TimeOutput>;
                static_assert(valid, "Objective.sample must accept (const SampleBuffer&, Eigen::Ref<SampleGradMatrix>, Eigen::Ref<VectorXd>) and return double");
                if constexpr (valid)
                {
                    cost += function(b.integral_samples, PositionOutput(b.sample_position_grad_buffer), TimeOutput(b.sample_time_grad_buffer));
                    accumulateSampleCostGradients(b.integral_samples, b.sample_position_grad_buffer,
                                                  b.sample_time_grad_buffer, b.grad_coeffs, b.grad_times,
                                                  b.global_time_grad_buffer, prepared_.num_segments);
                    b.grad_start_time += b.global_time_grad_buffer.sum();
                }
            }
            if constexpr (detail::HasCoefficient<O>::value)
            {
                auto &function = detail::unwrap(objective.coefficient);
                constexpr bool valid = std::is_invocable_r_v<double, decltype(function), const ParameterView<Spline> &, CoefficientGradient<Spline> &>;
                static_assert(valid, "Objective.coefficient must return double and accept (const ParameterView<Spline>&, CoefficientGradient<Spline>&)");
                if constexpr (valid)
                {
                    auto input = parameterView();
                    CoefficientGradient<Spline> output{b.grad_coeffs, b.grad_times, b.grad_start_time};
                    cost += function(input, output);
                }
            }
            if (!std::isfinite(cost) || !b.grad_coeffs.allFinite() || !b.grad_times.allFinite() || !std::isfinite(b.grad_start_time))
                return failEvaluation(error(ErrorCode::NumericalFailure, "Nonfinite coefficient-stage objective or partials"), gradient);

            if constexpr (needs_integral || detail::HasSample<O>::value || detail::HasCoefficient<O>::value)
                state_.spline.backward(b.grad_coeffs, b.grad_times, b.grads);
            else
            {
                b.grads.setZero();
                b.grads.durations = b.grad_times;
            }
            cost += accumulateEnergy();
            if constexpr (detail::HasParameter<O>::value)
            {
                auto &function = detail::unwrap(objective.parameter);
                constexpr bool valid = std::is_invocable_r_v<double, decltype(function), const ParameterView<Spline> &, ParameterGradient<Spline> &>;
                static_assert(valid, "Objective.parameter must return double and accept (const ParameterView<Spline>&, ParameterGradient<Spline>&)");
                if constexpr (valid)
                {
                    auto input = parameterView();
                    auto output = parameterGradient();
                    cost += function(input, output);
                }
            }
            if constexpr (has_auxiliary)
            {
                b.auxiliary_grad_buffer.setZero();
                auto input = parameterView();
                auto output = parameterGradient();
                cost += auxiliaryMap().backward(state_.auxiliary_vars, input, output,
                                                Eigen::Ref<Eigen::VectorXd>(b.auxiliary_grad_buffer));
                gradient.segment(prepared_.layout.auxiliary_offset, b.auxiliary_grad_buffer.size()) = b.auxiliary_grad_buffer;
            }
            writeDecisionGradient(x, gradient);
            if constexpr (detail::HasDecision<O>::value)
            {
                auto &function = detail::unwrap(objective.decision);
                using Output = Eigen::Ref<Eigen::VectorXd>;
                constexpr bool valid = std::is_invocable_r_v<double, decltype(function), const DecisionView &, Output>;
                static_assert(valid, "Objective.decision must return double and accept (const DecisionView&, Eigen::Ref<VectorXd>)");
                if constexpr (valid) cost += function(DecisionView{x, prepared_.layout}, Output(gradient));
            }
            if (!std::isfinite(cost) || !gradient.allFinite() || !std::isfinite(b.grad_start_time))
                return failEvaluation(error(ErrorCode::NumericalFailure, "Nonfinite objective or decision gradient"), gradient);
            last_status_ = Status::success();
            return {cost, last_status_};
        }

        /** @brief Scalar adapter for an external solver; inspect lastStatus() on infinity.
         * @note This calls the same evaluation path and does not retain the objective. */
        template<class Objective, class Executor = SerialExecutor>
        double costAndGradient(const Eigen::VectorXd &x, Eigen::VectorXd &gradient,
                               Objective &objective, const Executor &executor = {})
        { return evaluate(x, gradient, objective, executor).cost; }

    private:
        template<class Integral, class Executor>
        double evaluateIntegral(Integral &integral, const Executor &executor, bool record_samples)
        {
            constexpr bool valid = std::is_invocable_r_v<double, Integral &, const IntegralPointInfo &,
                                                        const SampleState<DIM> &, SampleGradient<DIM> &>;
            static_assert(valid, "Objective.integral must return double and accept (const IntegralPointInfo&, const SampleState<DIM>&, SampleGradient<DIM>&)");
            constexpr int derivative_order = detail::IntegralDerivativeOrder<detail::Plain<Integral>>::value;
            static_assert(derivative_order >= 0 && derivative_order <= 4,
                          "Objective.integral kDerivativeOrder must be between zero and four");
            double cost = 0.0;
            if constexpr (valid && derivative_order >= 0 && derivative_order <= 4)
            {
                if constexpr (detail::HasBeginEvaluation<Integral>::value) integral.beginEvaluation();
                accumulateIntegralCost(buffers_.grad_coeffs, buffers_.grad_times, cost, integral,
                                       state_.start_time, record_samples, executor);
                buffers_.grad_start_time += buffers_.global_time_grad_buffer.sum();
                buffers_.samples_valid = record_samples;
            }
            return cost;
        }

        double accumulateEnergy()
        {
            const double weight = prepared_.options.energy_weight;
            if (weight == 0.0) return 0.0;
            auto &g = buffers_.grads;
            auto &e = buffers_.energy_grads;
            const double energy = state_.spline.energy();
            state_.spline.energyGradient(e);
            g.inner_points += weight * e.inner_points;
            g.durations += weight * e.durations;
            g.start.p += weight * e.start.p;
            g.end.p += weight * e.end.p;
            g.start.v += weight * e.start.v;
            g.end.v += weight * e.end.v;
            if constexpr (Spline::kDegree >= 5)
            {
                g.start.a += weight * e.start.a;
                g.end.a += weight * e.end.a;
            }
            if constexpr (Spline::kDegree >= 7)
            {
                g.start.j += weight * e.start.j;
                g.end.j += weight * e.end.j;
            }
            return weight * energy;
        }

        void writeDecisionGradient(const Eigen::VectorXd &x, Eigen::VectorXd &gradient) const
        {
            const auto &g = buffers_.grads;
            for (const auto &variable : prepared_.layout.time)
            {
                const double tau = x(variable.offset);
                // Auxiliary pullbacks return derivatives in the original mapped coordinates.
                // A time map using its physical argument must see the duration before apply().
                double mapped_duration = state_.durations[variable.segment_index];
                if constexpr (has_auxiliary) mapped_duration = timeMap().toTime(tau);
                gradient(variable.offset) = timeMap().backward(tau, mapped_duration, g.durations(variable.segment_index));
            }
            for (const auto &variable : prepared_.layout.waypoints)
            {
                Vector physical;
                if (variable.point_index == 0) physical = g.start.p;
                else if (variable.point_index == prepared_.num_segments) physical = g.end.p;
                else physical = g.inner_points.row(variable.point_index - 1).transpose();
                spaceMap().backwardInto(x.segment(variable.offset, variable.dof), physical,
                                       variable.point_index, gradient.segment(variable.offset, variable.dof));
            }
            int offset = prepared_.layout.boundary_offset;
            forEachBoundary([&](int slot) {
                gradient.template segment<DIM>(offset) = boundaryGradient(slot);
                offset += DIM;
            });
        }

        void accumulateSampleCostGradients(const IntegralSampleBuffer &samples,
                                           const SampleGradMatrix &sample_position_gradients,
                                           const Eigen::VectorXd &sample_time_gradients,
                                           CoefficientMatrix &grad_coeffs,
                                           Eigen::VectorXd &grad_times,
                                           Eigen::VectorXd &global_time_grad,
                                           int num_segments) const
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

            global_time_grad.setZero();

            for (Eigen::Index sample_idx = 0; sample_idx < sample_count; ++sample_idx)
            {
                const IntegralSample &sample = samples[sample_idx];
                const int segment_index = sample.point.segment_index;
                const int base_row = segment_index * SplineType::kCoefficientCount;
                const Vector grad_position = sample_position_gradients.col(sample_idx);

                grad_coeffs.template block<SplineType::kCoefficientCount, DIM>(base_row, 0).noalias() +=
                    sample.b_p.transpose() * grad_position.transpose();
                grad_times(segment_index) += grad_position.dot(sample.v) * sample.point.alpha;

                if (sample_idx < sample_time_gradients.size())
                {
                    const double grad_time = sample_time_gradients(sample_idx);
                    grad_times(segment_index) += grad_time * sample.point.alpha;
                    global_time_grad(segment_index) += grad_time;
                }
            }

            double accumulator = 0.0;
            for (int i = num_segments - 1; i > 0; --i)
            {
                accumulator += global_time_grad(i);
                grad_times(i - 1) += accumulator;
            }
        }

        template <typename IntegralFunc, typename Executor>
#if defined(__GNUC__) || defined(__clang__)
        // Keep statically bound sample/flatness call chains visible inside the quadrature loop.
        // GCC 9 otherwise outlines them after structured state binding, increasing measured planner cost.
        __attribute__((flatten))
#endif
        void accumulateIntegralCost(CoefficientMatrix &grad_coeffs,
                                    Eigen::VectorXd &grad_times,
                                    double &cost,
                                    IntegralFunc &&integral_cost,
                                    double start_time,
                                    bool record_samples,
                                    const Executor& executor)
        {
            auto &state = state_;
            auto &buffers = buffers_;
            const auto &coeffs = state.spline.polynomial().coefficients();

            double running_time = start_time;
            for(int i = 0; i < prepared_.num_segments; ++i) {
                buffers.segment_begin_times[i] = running_time;
                running_time += state.durations[i];
            }

            std::fill(buffers.segment_cost_buffer.begin(), buffers.segment_cost_buffer.end(), 0.0);

            buffers.global_time_grad_buffer.setZero();

            int K = prepared_.options.integration_steps;
            double inv_K = 1.0 / K;

            executor(0, prepared_.num_segments, [&](int i) {
                double T = state.durations[i];
                double dt = T * inv_K;
                int base_row = i * SplineType::kCoefficientCount;

                Eigen::Matrix<double, SplineType::kCoefficientCount, DIM> coeff_block =
                    coeffs.template block<SplineType::kCoefficientCount, DIM>(base_row, 0);

                // A fixed normalized quadrature rule is shared by all pieces/evaluations.
                // Scale coefficients once per piece; time drift below remains at fixed physical coefficients.
                Eigen::Matrix<double, SplineType::kCoefficientCount, 1> time_powers;
                time_powers(0) = 1.0;
                for (int power = 1; power < SplineType::kCoefficientCount; ++power)
                    time_powers(power) = time_powers(power - 1) * T;
                for (int power = 0; power < SplineType::kCoefficientCount; ++power)
                    coeff_block.row(power) *= time_powers(power);
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

                double current_segment_start_time = buffers.segment_begin_times[i];

                for (int k = 0; k <= K; ++k)
                {
                    double alpha = (double)k * inv_K;
                    double t = alpha * T;

                    double weight_trap = (k == 0 || k == K) ? 0.5 : 1.0;
                    double common_weight = weight_trap * dt;

                    double t_global = current_segment_start_time + t;

                    const auto &basis = prepared_.basis[k];
                    const auto &b_p = basis.p; const auto &b_v = basis.v; const auto &b_a = basis.a;
                    const auto &b_j = basis.j; const auto &b_s = basis.s; const auto &b_c = basis.c;

                    using Cost = std::decay_t<IntegralFunc>;
                    constexpr int derivative_order = detail::IntegralDerivativeOrder<Cost>::value;
                    static_assert(derivative_order >= 0 && derivative_order <= 4,
                                  "Integral costs support derivative orders zero through four.");
                    SampleState<DIM> sample_state;
                    auto &p = sample_state.p; auto &v = sample_state.v; auto &a = sample_state.a;
                    auto &j = sample_state.j; auto &s = sample_state.s;
                    [[maybe_unused]] Vector c = Vector::Zero();
                    p.transpose().noalias() = b_p * coeff_block;
                    v.transpose().noalias() = (b_v * coeff_block) * inverse_time;
                    if constexpr (derivative_order >= 1) a.transpose().noalias() = (b_a * coeff_block) * inverse_time2;
                    if constexpr (derivative_order >= 2) j.transpose().noalias() = (b_j * coeff_block) * inverse_time3;
                    if constexpr (derivative_order >= 3) s.transpose().noalias() = (b_s * coeff_block) * inverse_time4;
                    if constexpr (derivative_order >= 4) c.transpose().noalias() = (b_c * coeff_block) * inverse_time5;

                    SampleGradient<DIM> partials;
                    auto &gp = partials.p; auto &gv = partials.v; auto &ga = partials.a;
                    auto &gj = partials.j; auto &gs = partials.s; auto &gt = partials.time;
                    const IntegralPointInfo point{i, prepared_.num_segments, k, K,
                                                  alpha, T, dt, t, t_global};
                    const double c_val = integral_cost(point, sample_state, partials);

                    if (record_samples)
                    {
                        const int sample_index = i * (K + 1) + k;
                        IntegralSample &sample = buffers.integral_samples[sample_index];
                        sample.point = point;
                        sample.trap_weight = weight_trap;
                        sample.b_p = b_p.array() * time_powers.transpose().array();
                        sample.p = p;
                        sample.v = v;
                    }

                    local_acc_cost += c_val * common_weight;

                    local_acc_gdC.noalias() += b_p.transpose() * gp.transpose() * common_weight;
                    if constexpr (derivative_order >= 1)
                        local_acc_gdC.noalias() += b_v.transpose() * gv.transpose() * (common_weight * inverse_time);
                    if constexpr (derivative_order >= 2)
                        local_acc_gdC.noalias() += b_a.transpose() * ga.transpose() * (common_weight * inverse_time2);
                    if constexpr (derivative_order >= 3)
                        local_acc_gdC.noalias() += b_j.transpose() * gj.transpose() * (common_weight * inverse_time3);
                    if constexpr (derivative_order >= 4)
                        local_acc_gdC.noalias() += b_s.transpose() * gs.transpose() * (common_weight * inverse_time4);

                    local_acc_gdT += c_val * weight_trap * inv_K;
                    double drift_grad = gp.dot(v);
                    if constexpr (derivative_order >= 1) drift_grad += gv.dot(a);
                    if constexpr (derivative_order >= 2) drift_grad += ga.dot(j);
                    if constexpr (derivative_order >= 3) drift_grad += gj.dot(s);
                    if constexpr (derivative_order >= 4) drift_grad += gs.dot(c);
                    local_acc_gdT += drift_grad * alpha * common_weight;

                    local_acc_gdT += gt * alpha * common_weight;
                    local_acc_explicit_time_grad += gt * common_weight;
                }

                buffers.segment_cost_buffer[i] = local_acc_cost;

                grad_times(i) += local_acc_gdT;
                buffers.global_time_grad_buffer(i) += local_acc_explicit_time_grad;

                for (int power = 0; power < SplineType::kCoefficientCount; ++power)
                    grad_coeffs.row(base_row + power) += time_powers(power) * local_acc_gdC.row(power);
            });

            for(int i = 0; i < prepared_.num_segments; ++i) {
                cost += buffers.segment_cost_buffer[i];
            }

            double accumulator = 0.0;
            for (int i = prepared_.num_segments - 1; i > 0; --i)
            {
                accumulator += buffers.global_time_grad_buffer(i);
                grad_times(i - 1) += accumulator;
            }
        }

    };

    /** @brief Deduce an owned or explicitly borrowed parameterization type. */
    template<class Spline, class Parameterization>
    auto makeOptimizer(Parameterization &&maps)
    { return SplineOptimizer<Spline, std::decay_t<Parameterization>>(std::forward<Parameterization>(maps)); }

    /** @brief Owning finite-difference diagnostics; never part of normal objective evaluation. */
    struct GradientCheckResult
    {
        OptimizationStatus status;
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
            result.status = {OptimizationError::InvalidInput, "Gradient-check step and tolerance must be finite and positive"};
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
            result.status = {OptimizationError::GradientMismatch, "Analytical and numerical gradients differ"};
        return result;
    }
}
#endif
