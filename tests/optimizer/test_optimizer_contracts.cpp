#define EIGEN_RUNTIME_NO_MALLOC
#include "TestSupport.hpp"
#include "SplineOptimizer.hpp"
#include "AllocationCounter.hpp"
#include "SplineConvexHull.hpp"
#include <iostream>
#include <stdexcept>

using spline_test::require;

struct DurationCost
{
    double weight = 0.2;
    template<class Curve>
    SplineTrajectory::EvaluationResult operator()(const SplineTrajectory::ParameterView<Curve> &state,
        SplineTrajectory::ParameterGradientView<Curve> &output) const
    {
        const auto &times = state.durations;
        auto &gradient = output.durations;
        gradient.array() += weight;
        const double cost = weight * std::accumulate(times.begin(), times.end(), 0.0);
        return {cost, SplineTrajectory::Status::success()};
    }
};

template<int ORDER>
struct StateCost
{
    static constexpr int kDerivativeOrder = ORDER;
    SplineTrajectory::EvaluationResult operator()(const SplineTrajectory::IntegralPointInfo &point,
                      const SplineTrajectory::IntegralState<3> &state,
                      SplineTrajectory::IntegralGradient<3> &gradient) const
    {
        double cost = 0.01 * point.trajectory_time * point.trajectory_time;
        gradient.trajectory_time += 0.02 * point.trajectory_time;
        gradient.p += state.p;
        cost += 0.5 * state.p.squaredNorm();
        if constexpr (ORDER >= 1) { gradient.v += 0.3 * state.v; cost += 0.15 * state.v.squaredNorm(); }
        if constexpr (ORDER >= 2) { gradient.a += 0.02 * state.a; cost += 0.01 * state.a.squaredNorm(); }
        if constexpr (ORDER >= 3) { gradient.j += 0.002 * state.j; cost += 0.001 * state.j.squaredNorm(); }
        if constexpr (ORDER >= 4) { gradient.s += 0.0002 * state.s; cost += 0.0001 * state.s.squaredNorm(); }
        return {cost, SplineTrajectory::Status::success()};
    }
};

/** @brief A non-default-constructible coefficient objective using a writable gradient view. */
struct CoefficientCost
{
    explicit CoefficientCost(double value) : weight(value) {}
    double weight;
    template<class Spline>
    SplineTrajectory::EvaluationResult operator()(const SplineTrajectory::ParameterView<Spline> &state,
                      SplineTrajectory::CoefficientGradientView<Spline> &gradient) const
    {
        gradient.coefficients += weight * state.polynomial.coefficients();
        double cost = 0.5 * weight * state.polynomial.coefficients().squaredNorm();
        for (int i = 0; i < gradient.durations.size(); ++i)
        {
            gradient.durations(i) += 0.02 * state.durations[i];
            cost += 0.01 * state.durations[i] * state.durations[i];
        }
        return {cost, SplineTrajectory::Status::success()};
    }
};

struct ParameterCost
{
    template<class Spline>
    SplineTrajectory::EvaluationResult operator()(const SplineTrajectory::ParameterView<Spline> &state,
                      SplineTrajectory::ParameterGradientView<Spline> &gradient) const
    {
        for (int i = 0; i < state.waypoints.rows(); ++i)
            gradient.addWaypoint(i, 0.04 * state.waypoints.row(i).transpose());
        gradient.boundary.start.v += 0.02 * state.boundary.start.v;
        return {DurationCost{}(state, gradient).cost + 0.02 * state.waypoints.squaredNorm() + 0.01 * state.boundary.start.v.squaredNorm(), SplineTrajectory::Status::success()};
    }
};

struct DecisionCost
{
    SplineTrajectory::EvaluationResult operator()(const SplineTrajectory::DecisionView &view, Eigen::Ref<Eigen::VectorXd> gradient) const
    {
        gradient += 0.006 * view.variables;
        return {0.003 * view.variables.squaredNorm(), SplineTrajectory::Status::success()};
    }
};

template<class Spline>
auto problem()
{
    SplineTrajectory::SplineInput<Spline::kDimension> result;
    result.durations = {1.4, 1.7};
    result.start_time = 0.7;
    result.waypoints.resize(3, 3);
    result.waypoints << 0,0,0, 1,0.3,-0.2, 2,-0.1,0.4;
    result.boundary.start.v << 0.1,0.2,-0.1;
    return result;
}

template<class Spline, int ORDER>
void checkCompleteGradient()
{
    using Optimizer = SplineTrajectory::SplineOptimizer<Spline>;
    auto reference = problem<Spline>();
    SplineTrajectory::OptimizationMask mask;
    mask.durations.assign(2, 1);
    mask.waypoints.assign(3, 1);
    mask.start.v = mask.end.v = true;
    if constexpr (Spline::kDegree >= 5) mask.start.a = mask.end.a = true;
    if constexpr (Spline::kDegree >= 7) mask.start.j = mask.end.j = true;
    Optimizer optimizer;
    SplineTrajectory::OptimizerOptions options;
    options.mask = mask;
    options.integration_steps = 5;
    options.energy_weight = 0.01;
    options.duration_weight = 0.3;
    struct Objective
    {
        StateCost<ORDER> integral;
        CoefficientCost coefficient{1e-4};
        ParameterCost parameter;
        DecisionCost decision;
    } objective;
    require(bool(optimizer.prepare(reference, options, objective)), "prepare complete objective");
    const auto x = optimizer.initialGuess();
    const auto check = SplineTrajectory::checkGradients(optimizer, x, objective, 1e-5, 2e-4);
    if (!check) std::cerr << check.status.message << ": " << check.error_norm << '\n';
    require(bool(check) && check.max_absolute_error < 2e-4, "complete coefficient/parameter/decision gradient");
    const auto expected = optimizer.copyPolynomial();
    require(optimizer.polynomial().coefficients().isApprox(expected.coefficients()), "gradient check restores candidate");
    Eigen::VectorXd gradient(x.size());

    // Preparation must cover the first evaluation, not only warmed repetitions.
    Optimizer fresh;
    require(bool(fresh.prepare(reference, options, objective)), "prepare fresh allocation check");
    allocation_test::begin();
    Eigen::internal::set_is_malloc_allowed(false);
    for (int i = 0; i < 10; ++i)
        require(bool(fresh.evaluate(x, gradient, objective)), "allocation-free complete objective");
    Eigen::internal::set_is_malloc_allowed(true);
    require(allocation_test::end() == 0, "No standard allocation in prepared evaluation");
    const auto serial = optimizer.evaluate(x, gradient, objective);
    const auto expected_gradient = gradient;
    const auto parallel = optimizer.evaluate(x, gradient, objective, SplineTrajectory::OpenMPExecutor{});
    require(bool(parallel) && serial.cost == parallel.cost && gradient.isApprox(expected_gradient, 1e-12),
            "executor preserves objective and gradient");
}

void checkEnergyOnly()
{
    using Spline = SplineTrajectory::QuinticSplineND<3>;
    SplineTrajectory::SplineOptimizer<Spline> optimizer;
    SplineTrajectory::OptimizerOptions options;
    options.energy_weight = 0.3;
    options.record_samples = true;
    require(bool(optimizer.prepare(problem<Spline>(), options, SplineTrajectory::NoObjective{})), "prepare energy and recording");
    SplineTrajectory::NoObjective objective;
    const auto x = optimizer.initialGuess();
    const auto checked = SplineTrajectory::checkGradients(optimizer, x, objective, 1e-5, 1e-4);
    require(bool(checked), "energy-only gradient");
    require(!optimizer.recordedSamples().empty(), "record without an integrand");
    Eigen::VectorXd gradient(x.size());
    allocation_test::begin();
    Eigen::internal::set_is_malloc_allowed(false);
    require(bool(optimizer.evaluate(x, gradient, objective)), "allocation-free explicit recording");
    Eigen::internal::set_is_malloc_allowed(true);
    require(allocation_test::end() == 0, "No standard allocation in prepared evaluation");
}

/** @brief Move the common start time and share a scale across all durations. */
struct SharedMap
{
    int dimension() const { return 2; }
    template<class Problem> SplineTrajectory::StatusOr<Eigen::VectorXd> initial(const Problem &) const { return Eigen::VectorXd(Eigen::VectorXd::Zero(2)); }
    template<class Spline>
    SplineTrajectory::Status apply(const Eigen::Ref<const Eigen::VectorXd> &x, SplineTrajectory::MutableParameters<Spline> &state) const
    {
        state.start_time += x(0);
        for (double &duration : state.durations) duration *= std::exp(x(1));

        return SplineTrajectory::Status::success();
    }
    template<class Spline>
    SplineTrajectory::Status backward(const Eigen::Ref<const Eigen::VectorXd> &x,
                    const SplineTrajectory::ParameterView<Spline> &state,
                    SplineTrajectory::ParameterGradientView<Spline> &gradient,
                    Eigen::Ref<Eigen::VectorXd> output) const
    {
        output(0) += gradient.start_time;
        for (int i = 0; i < gradient.durations.size(); ++i)
            output(1) += gradient.durations(i) * state.durations[i];
        gradient.durations *= std::exp(x(1));
        return SplineTrajectory::Status::success();
    }
};

template<class Spline>
void checkDurationWeight()
{
    using namespace SplineTrajectory;
    struct Parameterization
    {
        QuadInvTimeMap time;
        IdentitySpatialMap<3> space;
        SharedMap auxiliary;
    };
    using Optimizer = SplineOptimizer<Spline, Parameterization>;
    NoObjective empty;
    for (int active_durations : {0, 1, 2})
    {
        auto input = problem<Spline>();
        OptimizationMask mask;
        mask.durations = {active_durations > 0, active_durations > 1};
        for (double weight : {0.0, 0.2, 1.3})
        {
            Optimizer builtin, callback;
            OptimizerOptions options;
            options.mask = mask;
            options.energy_weight = 0.01;
            struct Objective { DurationCost parameter; } objective{{weight}};
            require(bool(callback.prepare(input, options, objective)), "prepare reference duration callback");
            options.duration_weight = weight;
            require(bool(builtin.prepare(input, options, empty)), "prepare built-in duration weight");
            auto x = builtin.initialGuess();
            x.array() += 0.05;
            x(builtin.layout().auxiliary_offset) = 0.2;
            x(builtin.layout().auxiliary_offset + 1) = -0.3;
            Eigen::VectorXd gradient(x.size()), reference_gradient(x.size());
            const auto expected = callback.evaluate(x, reference_gradient, objective);
            require(bool(expected), "reference duration evaluation");

            // Include the first prepared evaluation, with nonidentity maps and fixed duration slots.
            allocation_test::begin();
            Eigen::internal::set_is_malloc_allowed(false);
            for (int iteration = 0; iteration < 3; ++iteration)
            {
                const auto result = builtin.evaluate(x, gradient, empty);
                require(bool(result) && std::abs(result.cost - expected.cost) < 1e-12,
                        "built-in duration cost equals parameter callback");
                require((gradient - reference_gradient).norm() < 1e-11,
                        "duration gradient passes through auxiliary and time maps");
            }
            Eigen::internal::set_is_malloc_allowed(true);
            require(allocation_test::end() == 0, "built-in duration cost does not allocate");
            const auto checked = checkGradients(builtin, x, empty, 1e-5, 2e-4);
            require(bool(checked), "built-in energy and duration finite differences");
        }
    }

    Optimizer optimizer;
    for (double weight : {-1.0, std::numeric_limits<double>::infinity(),
                          std::numeric_limits<double>::quiet_NaN()})
    {
        OptimizerOptions options;
        options.duration_weight = weight;
        const auto status = optimizer.prepare(problem<Spline>(), options, empty);
        require(!status && status.code == ErrorCode::InvalidInput && !optimizer.isPrepared(),
                "duration weight must be finite and nonnegative");
    }
}

struct ClockCoefficient
{
    template<class Spline>
    SplineTrajectory::EvaluationResult operator()(const SplineTrajectory::ParameterView<Spline> &state,
                      SplineTrajectory::CoefficientGradientView<Spline> &gradient) const
    { gradient.start_time += 3.0; return {3.0 * state.start_time, SplineTrajectory::Status::success()}; }
};

struct ClockSample
{
    template<class Samples, class Spline>
    SplineTrajectory::EvaluationResult operator()(const Samples &samples, SplineTrajectory::SampleGradientView<Spline> &gradient) const
    {
        double cost = 0.0;
        for (int i = 0; i < static_cast<int>(samples.size()); ++i)
        {
            cost += 0.005 * samples[i].point.trajectory_time * samples[i].point.trajectory_time;
            gradient.trajectory_times(i) += 0.01 * samples[i].point.trajectory_time;
            gradient.positions.row(i) += 0.002 * samples[i].p.transpose();
            cost += 0.001 * samples[i].p.squaredNorm();
        }
        return {cost, SplineTrajectory::Status::success()};
    }
};

void checkSharedVariables()
{
    using namespace SplineTrajectory;
    using Spline = QuinticSplineND<3>;
    struct Parameterization
    {
        QuadInvTimeMap time;
        IdentitySpatialMap<3> space;
        SharedMap auxiliary;
    };
    struct Objective
    {
        DurationCost parameter;
        StateCost<2> integral;
        ClockCoefficient coefficient;
        ClockSample sample;
    } objective;
    SplineOptimizer<Spline, Parameterization> optimizer;
    OptimizerOptions options;
    options.integration_steps = 5;
    options.duration_weight = 0.4;
    require(bool(optimizer.prepare(problem<Spline>(), options, SplineTrajectory::NoObjective{})), "prepare without sample capability");
    auto x = optimizer.initialGuess();
    Eigen::VectorXd gradient(x.size());
    const auto missing = optimizer.evaluate(x, gradient, objective);
    require(!missing && missing.status.code == ErrorCode::SamplingNotPrepared, "sample requires explicit preparation");
    require(bool(optimizer.prepare(problem<Spline>(), options, objective)), "prepare sample objective");
    require(optimizer.recordedSamples().empty(), "Preparation must not expose unfilled sample records");
    x(optimizer.layout().auxiliary_offset) = 0.2;
    x(optimizer.layout().auxiliary_offset + 1) = 0.1;
    const auto checked = checkGradients(optimizer, x, objective, 1e-5, 2e-4);
    if (!checked) std::cerr << checked.status.message << ": " << checked.error_norm << '\n';
    require(bool(checked), "combined start time, shared duration and sample gradient");
    allocation_test::begin();
    Eigen::internal::set_is_malloc_allowed(false);
    require(bool(optimizer.evaluate(x, gradient, objective)), "allocation-free shared-variable objective");
    Eigen::internal::set_is_malloc_allowed(true);
    require(allocation_test::end() == 0, "No standard allocation in prepared evaluation");
    require(!optimizer.recordedSamples().empty(), "Sample objective records current values");
    NoObjective empty;
    allocation_test::begin();
    Eigen::internal::set_is_malloc_allowed(false);
    require(bool(optimizer.evaluate(x, gradient, empty)), "Switch to an objective without sample recording");
    require(optimizer.recordedSamples().empty(), "Absent sample stage must not expose old records");
    require(bool(optimizer.evaluate(x, gradient, objective)), "Restore sample objective without preparation");
    Eigen::internal::set_is_malloc_allowed(true);
    require(allocation_test::end() == 0, "Objective switching must reuse prepared capacity");
    require(bool(optimizer.update(x)) && optimizer.recordedSamples().empty(),
            "Accepted-state update invalidates records from a trial evaluation");

}

int main()
{
    return spline_test::run("test_optimizer_contracts", []
    {
        checkCompleteGradient<SplineTrajectory::CubicSplineND<3>, 0>();
        checkCompleteGradient<SplineTrajectory::QuinticSplineND<3>, 2>();
        checkCompleteGradient<SplineTrajectory::SepticSplineND<3>, 3>();
        checkCompleteGradient<SplineTrajectory::SepticSplineND<3>, 4>();
        checkEnergyOnly();
        checkDurationWeight<SplineTrajectory::CubicSplineND<3>>();
        checkDurationWeight<SplineTrajectory::QuinticSplineND<3>>();
        checkDurationWeight<SplineTrajectory::SepticSplineND<3>>();
        checkSharedVariables();
    });
}
