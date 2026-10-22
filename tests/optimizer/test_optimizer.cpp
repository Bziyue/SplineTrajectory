#define EIGEN_RUNTIME_NO_MALLOC
#include "TestSupport.hpp"
#include "SplineOptimizer.hpp"
#include "SplineConvexHull.hpp"
#include <iostream>
#include <stdexcept>
#include <type_traits>

namespace
{
using Spline = SplineTrajectory::QuinticSplineND<3>;
using Optimizer = SplineTrajectory::SplineOptimizer<Spline>;
using Matrix = Spline::CoefficientMatrix;

using spline_test::require;

template<class Curve = Spline>
SplineTrajectory::SplineInput<Curve::kDimension> makeProblem()
{
    SplineTrajectory::SplineInput<Curve::kDimension> problem;
    problem.durations = {1.2, 1.5};
    problem.start_time = 2.0;
    problem.waypoints.resize(3, 3);
    problem.waypoints << 0,0,0, 1,0.2,0.1, 2,0,0;
    return problem;
}

struct DurationCost
{
    template<class Curve>
    SplineTrajectory::EvaluationResult operator()(const SplineTrajectory::ParameterView<Curve> &state,
        SplineTrajectory::ParameterGradientView<Curve> &output) const
    {
        const auto &times = state.durations;
        auto &gradient = output.durations;
        double cost = 0.0;
        for (double duration : times) cost += duration;
        gradient.array() += 1.0;
        return {cost, SplineTrajectory::Status::success()};
    }
};

struct ScalingSpatialMap
{
    explicit ScalingSpatialMap(double factor) : scale(factor) {}
    double scale;
    mutable int inverse_calls = 0;
    int dimension(int) const { return 3; }
    SplineTrajectory::StatusOr<Eigen::Vector3d> forward(const Eigen::Ref<const Eigen::VectorXd> &x, int) const { return Eigen::Vector3d(scale * x); }
    SplineTrajectory::StatusOr<Eigen::Vector3d> inverse(const Eigen::Vector3d &point, int) const { ++inverse_calls; return Eigen::Vector3d(point / scale); }
    SplineTrajectory::Status backward(const Eigen::Ref<const Eigen::VectorXd> &, const Eigen::Vector3d &gradient,
                      int, Eigen::Ref<Eigen::VectorXd> output) const { output = scale * gradient;
        return SplineTrajectory::Status::success();
    }
};

struct Parameterization
{
    SplineTrajectory::IdentityTimeMap time;
    ScalingSpatialMap space;
    SplineTrajectory::NoAuxiliaryMap auxiliary;
};

void testOwnershipAndMovement()
{
    using Mapped = SplineTrajectory::SplineOptimizer<Spline, Parameterization>;
    static_assert(!std::is_default_constructible_v<Mapped>, "Non-default maps must not be silently constructed");
    static_assert(!std::is_copy_constructible_v<Mapped> && !std::is_copy_assignable_v<Mapped>, "A workspace has one owner");
    static_assert(std::is_move_constructible_v<Mapped>, "Owned workspaces support movement");
    Parameterization maps{{}, ScalingSpatialMap(2.0), {}};
    Mapped optimizer(maps);
    maps.space.scale = 7.0;
    const auto problem = makeProblem();
    require(bool(optimizer.prepare(problem, SplineTrajectory::NoObjective{})), "owned map preparation");
    const auto x = optimizer.initialGuess();
    require(std::abs(x(optimizer.layout().waypoints.front().offset) - 0.5) < 1e-14,
            "owned maps are independent of later caller changes");
    struct Objective { DurationCost parameter; } objective;
    Eigen::VectorXd gradient(x.size());
    auto moved = std::move(optimizer);
    require(!optimizer.isPrepared() && moved.isPrepared(), "move invalidates source preparation");
    require(!optimizer.polynomial().isValid(), "moved-from polynomial is invalid");
    require(bool(moved.evaluate(x, gradient, objective)), "moved workspace remains usable");
    Mapped assigned(Parameterization{{}, ScalingSpatialMap(3.0), {}});
    assigned = std::move(moved);
    require(!moved.isPrepared() && bool(assigned.evaluate(x, gradient, objective)), "move assignment transfers workspace");
    auto borrowed = SplineTrajectory::makeOptimizer<Spline>(std::cref(maps));
    require(bool(borrowed.prepare(problem, SplineTrajectory::NoObjective{})), "explicitly borrowed parameterization");
    const auto borrowed_x = borrowed.initialGuess();
    auto editable_initial = borrowed.initialGuess();
    editable_initial.setZero();
    require(maps.space.inverse_calls == 1, "initial coordinates are encoded only during preparation");
    require(borrowed.initialGuess().isApprox(borrowed_x, 0.0), "initial guess returns an independent copy");
    require(bool(borrowed.evaluate(borrowed_x, gradient, objective)), "evaluate the prepared initial coordinates");
    require(maps.space.inverse_calls == 1, "trial evaluation does not encode coordinates");
    auto changed = problem;
    changed.waypoints(1, 0) += 1.0;
    require(bool(borrowed.prepare(changed, SplineTrajectory::NoObjective{})) && maps.space.inverse_calls == 2,
            "a new problem refreshes its initial coordinate snapshot");
    require(std::abs(borrowed.initialGuess()(borrowed.layout().waypoints.front().offset) - 2.0 / 7.0) < 1e-14,
            "repreparation replaces the previous initial coordinates");
    require(std::abs(borrowed_x(borrowed.layout().waypoints.front().offset) - 1.0 / 7.0) < 1e-14,
            "borrowed maps use the supplied object");
    require(bool(assigned.prepare(assigned.problem(), SplineTrajectory::NoObjective{})), "prepare accepts the optimizer's own immutable problem view");
}

void testInputAndAcceptedState()
{
    Optimizer optimizer;
    auto problem = makeProblem();
    SplineTrajectory::OptimizationMask mask;
    mask.durations = {1};
    SplineTrajectory::OptimizerOptions options;
    options.mask = mask;
    require(!optimizer.prepare(problem, options, SplineTrajectory::NoObjective{}), "reject malformed mask");
    require(!optimizer.isPrepared() && !optimizer.polynomial().isValid(), "failed preparation invalidates state");
    require(bool(optimizer.prepare(problem, SplineTrajectory::NoObjective{})), "prepare valid problem after rejection");
    auto x = optimizer.initialGuess();
    Eigen::VectorXd gradient;
    struct Objective { DurationCost parameter; } objective;
    require(bool(optimizer.evaluate(x, gradient, objective)) && gradient.size() == x.size(),
            "owning output is resized on first evaluation");
    gradient.resize(x.size());
    require(bool(optimizer.evaluate(x, gradient, objective)), "evaluate after sizing output");
    const auto accepted = optimizer.copyPolynomial();
    const auto original = x;
    x(0) += 0.4;
    require(bool(optimizer.evaluate(x, gradient, objective)), "evaluate a later trial");
    require(bool(optimizer.update(original)), "restore the solver's accepted vector");
    require(optimizer.polynomial().coefficients().isApprox(accepted.coefficients(), 1e-13),
            "accepted state must not equal the last rejected trial");
    x(0) = std::numeric_limits<double>::quiet_NaN();
    require(!optimizer.update(x) && !optimizer.polynomial().isValid(), "invalid acceptance clears polynomial");

    auto mapped = SplineTrajectory::makeOptimizer<Spline>(Parameterization{{}, ScalingSpatialMap(1.0), {}});
    require(bool(mapped.prepare(problem, SplineTrajectory::NoObjective{})), "prepare identity durations");
    x = mapped.initialGuess();
    x(0) = -1.0;
    require(!mapped.evaluate(x, gradient, objective), "reject nonpositive decoded duration");
    auto check = SplineTrajectory::checkGradients(optimizer, original, objective, -1.0);
    require(!check, "reject negative gradient-check step");
}

struct RecordingIntegral
{
    static constexpr int kDerivativeOrder = 0;
    int calls = 0;
    SplineTrajectory::EvaluationResult operator()(const SplineTrajectory::IntegralPointInfo &,
                      const SplineTrajectory::IntegralState<3> &state,
                      SplineTrajectory::IntegralGradient<3> &gradient)
    {
        ++calls;
        gradient.p += 0.002 * state.p;
        return {0.001 * state.p.squaredNorm(), SplineTrajectory::Status::success()};
    }
};

void testRecordingAndBorrowedObjective()
{
    Optimizer optimizer;
    SplineTrajectory::OptimizerOptions options;
    options.integration_steps = 4;
    options.record_samples = true;
    const auto problem = makeProblem();
    require(bool(optimizer.prepare(problem, options, SplineTrajectory::NoObjective{})), "prepare sample recording");
    auto x = optimizer.initialGuess();
    Eigen::VectorXd gradient(x.size());
    RecordingIntegral integral;
    struct Objective { std::reference_wrapper<RecordingIntegral> integral; } objective{std::ref(integral)};
    require(bool(optimizer.evaluate(x, gradient, objective)), "borrow integral by concrete type");
    require(integral.calls == 10, "borrowed integral visits each quadrature point");
    const auto &samples = optimizer.recordedSamples();
    require(samples.size() == 10 && samples.front().point.isTrajectoryStart() &&
            samples.back().point.isTrajectoryEnd(), "recorded sample topology");
    require(samples.front().point.waypointIndex() == 0 && samples.back().point.waypointIndex() == 2,
            "endpoint indices match input waypoint rows");
    require(samples[4].point.waypointIndex() == 1 && samples[5].point.waypointIndex() == 1,
            "both quadrature endpoints identify their common waypoint row");
    require(samples[1].point.waypointIndex() == -1, "non-waypoint sample has no waypoint index");
    require(samples.front().point.trajectory_time == problem.start_time, "recording uses the source time axis");
    require(samples.front().trap_weight == 0.5 && samples[1].trap_weight == 1.0, "trapezoidal endpoint weights");
    const auto expected = optimizer.copyPolynomial();
    const auto checked = SplineTrajectory::checkGradients(optimizer, x, objective, 1e-6, 1e-4);
    require(bool(checked) && optimizer.polynomial().coefficients().isApprox(expected.coefficients(), 1e-13),
            "gradient diagnostics restore the base candidate");
}

/** @brief Tetrahedral softmax coordinates with a redundant vertex on alternate waypoints.
 * @note Varying decision dimensions exercise the layout needed by polytope maps. */
struct PolytopeSpatialMap
{
    int dimension(int index) const { return 4 + index % 2; }
    static Eigen::Vector3d vertex(Eigen::Index index)
    {
        Eigen::Vector3d value = Eigen::Vector3d::Zero();
        if (index >= 1 && index <= 3) value(index - 1) = 1.0;
        return value;
    }
    SplineTrajectory::StatusOr<Eigen::Vector3d> forward(const Eigen::Ref<const Eigen::VectorXd> &x, int) const
    {
        const double shift = x.maxCoeff();
        double sum = 0.0;
        Eigen::Vector3d point = Eigen::Vector3d::Zero();
        for (Eigen::Index i = 0; i < x.size(); ++i)
        {
            const double weight = std::exp(x(i) - shift);
            sum += weight;
            point += weight * vertex(i);
        }
        return Eigen::Vector3d(point / sum);
    }
    SplineTrajectory::StatusOr<Eigen::VectorXd> inverse(const Eigen::Vector3d &point, int index) const
    {
        Eigen::VectorXd x(dimension(index));
        const double origin_weight = (1.0 - point.sum()) / (x.size() - 3);
        x.setConstant(std::log(origin_weight));
        for (int i = 1; i <= 3; ++i) x(i) = std::log(point(i - 1));
        return Eigen::VectorXd(x);
    }
    SplineTrajectory::Status backward(const Eigen::Ref<const Eigen::VectorXd> &x, const Eigen::Vector3d &gradient,
                      int index, Eigen::Ref<Eigen::VectorXd> output) const
    {
        const auto mapped = forward(x, index);
        if (!mapped) return mapped.status();
        const Eigen::Vector3d &point = mapped.value();
        const double shift = x.maxCoeff();
        double sum = 0.0;
        for (Eigen::Index i = 0; i < x.size(); ++i) sum += std::exp(x(i) - shift);
        for (Eigen::Index i = 0; i < x.size(); ++i)
            output(i) = std::exp(x(i) - shift) / sum * gradient.dot(vertex(i) - point);

        return SplineTrajectory::Status::success();
    }
};

void testVariableDimensionSpatialMap()
{
    struct Maps
    {
        SplineTrajectory::QuadInvTimeMap time;
        PolytopeSpatialMap space;
        SplineTrajectory::NoAuxiliaryMap auxiliary;
    };
    auto optimizer = SplineTrajectory::makeOptimizer<Spline>(Maps{});
    auto problem = makeProblem();
    problem.durations = {1.2, 1.5, 1.1};
    problem.waypoints.resize(4, 3);
    problem.waypoints << 0,0,0, 0.2,0.1,0.1, 0.3,0.2,0.2, 1,0,0;
    SplineTrajectory::OptimizerOptions options;
    options.energy_weight = 0.01;
    options.integration_steps = 8;
    require(bool(optimizer.prepare(problem, options, SplineTrajectory::NoObjective{})), "prepare variable-dimension polytope maps");
    require(optimizer.layout().waypoints[0].dof == 5 && optimizer.layout().waypoints[1].dof == 4,
            "spatial dimensions belong to individual global waypoint indices");
    auto x = optimizer.initialGuess();
    x.tail(9).array() += 0.1;
    x(optimizer.layout().waypoints[0].offset + 1) += 0.2;
    Eigen::VectorXd gradient(x.size());
    struct Objective { RecordingIntegral integral; } objective;
    Eigen::internal::set_is_malloc_allowed(false);
    const auto result = optimizer.evaluate(x, gradient, objective);
    Eigen::internal::set_is_malloc_allowed(true);
    require(bool(result), "polytope pullback writes pre-sized variable-width slices without allocation");
    require(bool(SplineTrajectory::checkGradients(optimizer, x, objective, 1e-6, 1e-5)),
            "variable-dimension spatial and duration pullbacks match finite differences");
}

/** @brief Count actual constructions and adjoints, independently of objective adapters. */
struct CountingSpline : Spline
{
    inline static int builds = 0, adjoints = 0;
    /** @brief Observe the optimizer's ordinary public construction call. */
    template<class... Arguments>
    decltype(auto) update(Arguments &&...arguments)
    {
        ++builds;
        return Spline::update(std::forward<Arguments>(arguments)...);
    }
    /** @brief Observe the optimizer's ordinary public adjoint call. */
    template<class... Arguments>
    decltype(auto) backward(Arguments &&...arguments)
    {
        ++adjoints;
        return Spline::backward(std::forward<Arguments>(arguments)...);
    }
};

struct HullCost
{
    SplineTrajectory::ConvexHullWorkspace<3> workspace;
    Matrix control_gradient;
    explicit HullCost(const Spline::Polynomial &polynomial)
        : workspace(SplineTrajectory::toBezier(polynomial))
    { control_gradient.resize(workspace.controls().rows(), 3); }

    template<class Curve>
    SplineTrajectory::EvaluationResult operator()(const SplineTrajectory::ParameterView<Curve> &state,
                      SplineTrajectory::CoefficientGradientView<Curve> &gradient)
    {
        workspace.update(state.polynomial);
        control_gradient = 0.01 * workspace.controls();
        workspace.backwardAdd(control_gradient, gradient.coefficients, gradient.durations);
        return {0.005 * workspace.controls().squaredNorm(), SplineTrajectory::Status::success()};
    }
};

void testOneConstructionAndAdjoint()
{
    SplineTrajectory::SplineOptimizer<CountingSpline> optimizer;
    require(bool(optimizer.prepare(makeProblem<CountingSpline>(), SplineTrajectory::NoObjective{})), "prepare counted spline");
    struct Objective
    {
        RecordingIntegral integral;
        HullCost coefficient;
    } objective{{}, HullCost(optimizer.polynomial())};
    const auto x = optimizer.initialGuess();
    Eigen::VectorXd gradient(x.size());
    CountingSpline::builds = CountingSpline::adjoints = 0;
    Eigen::internal::set_is_malloc_allowed(false);
    const auto result = optimizer.evaluate(x, gradient, objective);
    Eigen::internal::set_is_malloc_allowed(true);
    require(bool(result) && CountingSpline::builds == 1 && CountingSpline::adjoints == 1,
            "integral and hull gradients share exactly one construction and adjoint");
    const auto checked = SplineTrajectory::checkGradients(optimizer, x, objective, 2e-6, 2e-3);
    require(bool(checked), "hull and integral objective finite differences");
}
} // namespace

int main()
{
    return spline_test::run("test_optimizer", []
    {
        testOwnershipAndMovement();
        testInputAndAcceptedState();
        testRecordingAndBorrowedObjective();
        testVariableDimensionSpatialMap();
        testOneConstructionAndAdjoint();
    });
}
