#include <cassert>
#include <cmath>
#include <iostream>
#include <string>
#include <vector>

#include "ConvexHullBasis.hpp"
#include "SplineOptimizer.hpp"

namespace
{
template <typename T>
void expectTrue(bool condition, const T &message)
{
    if (!condition)
    {
        std::cerr << message << std::endl;
        std::abort();
    }
}

struct LinearTimeCost
{
    double operator()(const std::vector<double> &times, Eigen::VectorXd &grad) const
    {
        grad = Eigen::VectorXd::Ones(static_cast<Eigen::Index>(times.size()));
        double total = 0.0;
        for (double t : times)
        {
            total += t;
        }
        return total;
    }
};

struct ZeroIntegralCost
{
    using Vec = Eigen::Vector3d;

    double operator()(const SplineTrajectory::IntegralPointInfo &point,
                      const Vec &p,
                      const Vec &v,
                      const Vec &a,
                      const Vec &j,
                      const Vec &s,
                      Vec &gp,
                      Vec &gv,
                      Vec &ga,
                      Vec &gj,
                      Vec &gs,
                      double &gt) const
    {
        (void)point;
        (void)p;
        (void)v;
        (void)a;
        (void)j;
        (void)s;
        gp.setZero();
        gv.setZero();
        ga.setZero();
        gj.setZero();
        gs.setZero();
        gt = 0.0;
        return 0.0;
    }
};

struct HookedZeroIntegralCost : ZeroIntegralCost
{
    void beginEvaluation() const
    {
        ++begin_calls;
    }

    mutable int begin_calls = 0;
};

struct RecordingZeroIntegralCost : ZeroIntegralCost
{
    using Vec = Eigen::Vector3d;

    double operator()(const SplineTrajectory::IntegralPointInfo &point,
                      const Vec &p, const Vec &v, const Vec &a,
                      const Vec &j, const Vec &s,
                      Vec &gp, Vec &gv, Vec &ga, Vec &gj, Vec &gs,
                      double &gt) const
    {
        points.push_back(point);
        return ZeroIntegralCost::operator()(point, p, v, a, j, s,
                                            gp, gv, ga, gj, gs, gt);
    }

    mutable std::vector<SplineTrajectory::IntegralPointInfo> points;
};

struct ZeroWaypointsCost
{
    template <typename WaypointsType, typename GradMatrixType>
    double operator()(const WaypointsType &waypoints,
                      GradMatrixType &grad_q) const
    {
        grad_q = GradMatrixType::Zero(waypoints.rows(), waypoints.cols());
        return 0.0;
    }
};

struct ZeroSampleCost
{
    template <typename SamplesType>
    double operator()(const SamplesType &samples,
                      Eigen::Matrix<double, 3, Eigen::Dynamic> &grad_p,
                      Eigen::VectorXd &grad_t_global) const
    {
        grad_p = Eigen::Matrix<double, 3, Eigen::Dynamic>::Zero(3, static_cast<Eigen::Index>(samples.size()));
        grad_t_global = Eigen::VectorXd::Zero(static_cast<Eigen::Index>(samples.size()));
        return 0.0;
    }
};

struct ZeroTrajectoryCost
{
    double operator()(const SplineTrajectory::QuinticSplineND<3> &spline,
                      const std::vector<double> &times,
                      const Eigen::Matrix<double, Eigen::Dynamic, 3, Eigen::RowMajor> &waypoints,
                      double start_time,
                      const SplineTrajectory::BoundaryConditions<3> &bc,
                      SplineTrajectory::QuinticSplineND<3>::Gradients &grads) const
    {
        (void)spline;
        (void)times;
        (void)waypoints;
        (void)start_time;
        (void)bc;
        (void)grads;
        return 0.0;
    }
};

struct QuadraticHullCoefficientCost
{
    using Spline = SplineTrajectory::QuinticSplineND<3>;
    using Matrix = Spline::MatrixType;
    using Hull = SplineTrajectory::ConvexHullRepresentation<3>;

    mutable Hull workspace;
    mutable Matrix control_gradients;

    double operator()(Spline &spline,
                      const std::vector<double> &,
                      double,
                      Matrix &coefficient_gradients,
                      Eigen::VectorXd &duration_gradients,
                      double &) const
    {
        const auto &polynomial = spline.getPPoly();
        if (!workspace.kernel() ||
            workspace.numSourceSegments() !=
                polynomial.getNumSegments())
        {
            workspace.resetTopology(
                polynomial,
                SplineTrajectory::ConvexHullBasis::MINVO,
                1, 2);
            control_gradients.resize(
                workspace.controls().rows(), 3);
        }
        workspace.update(polynomial);
        control_gradients = workspace.controls();
        workspace.backwardAdd(
            control_gradients, coefficient_gradients,
            duration_gradients);
        return 0.5 * workspace.controls().squaredNorm();
    }
};

template <int DIM>
struct ScalingSpatialMap
{
    using VectorType = Eigen::Matrix<double, DIM, 1>;

    explicit ScalingSpatialMap(double scale_in = 1.0) : scale(scale_in) {}

    int getUnconstrainedDim(int index) const
    {
        (void)index;
        return DIM;
    }

    VectorType toPhysical(const VectorType &xi, int index) const
    {
        (void)index;
        return scale * xi;
    }

    VectorType toUnconstrained(const VectorType &p, int index) const
    {
        (void)index;
        return p / scale;
    }

    VectorType backwardGrad(const VectorType &xi, const VectorType &grad_p, int index) const
    {
        (void)xi;
        (void)index;
        return scale * grad_p;
    }

    double scale = 1.0;
};

template <typename Optimizer>
typename Optimizer::ProblemDefinition makeProblem(int num_segments,
                                                  bool full_waypoint_mask = false)
{
    typename Optimizer::ProblemDefinition problem;
    problem.time_segments.assign(num_segments, 1.0);
    problem.waypoints.resize(num_segments + 1, 3);
    for (int i = 0; i <= num_segments; ++i)
    {
        problem.waypoints.row(i) << 1.0 + i, 2.0 - 0.5 * i, -0.25 * i;
    }
    problem.start_time = 0.25;

    if (full_waypoint_mask)
    {
        problem.mask = Optimizer::makeFullOptimizationMask(num_segments);
    }

    return problem;
}

void testPrepareRejectsBadMask()
{
    using Optimizer = SplineTrajectory::SplineOptimizer<3>;
    Optimizer optimizer;
    Optimizer::OptimizationContext context;
    auto problem = makeProblem<Optimizer>(2);

    SplineTrajectory::OptimizationMask bad_mask;
    bad_mask.time.assign(1, static_cast<uint8_t>(1));
    bad_mask.waypoints.assign(3, static_cast<uint8_t>(1));
    problem.mask = bad_mask;

    const auto status = optimizer.prepareContext(problem, context);
    expectTrue(!status, "prepareContext should reject mismatched mask sizes.");
}

void testPreparedContextSnapshotsSpatialMap()
{
    using Optimizer = SplineTrajectory::SplineOptimizer<3,
                                                        SplineTrajectory::QuinticSplineND<3>,
                                                        SplineTrajectory::QuadInvTimeMap,
                                                        ScalingSpatialMap<3> >;

    ScalingSpatialMap<3> map_prepare(1.0);
    ScalingSpatialMap<3> map_after(2.0);

    Optimizer optimizer;
    typename Optimizer::OptimizerConfig config;
    config.spatial_map = &map_prepare;
    expectTrue(static_cast<bool>(optimizer.setConfig(config)), "setConfig with prepare map should succeed.");

    typename Optimizer::OptimizationContext context;
    auto problem = makeProblem<Optimizer>(1, true);
    expectTrue(static_cast<bool>(optimizer.prepareContext(problem, context)), "prepareContext should succeed.");

    const Eigen::VectorXd x_before = optimizer.generateInitialGuess(context);

    config.spatial_map = &map_after;
    expectTrue(static_cast<bool>(optimizer.setConfig(config)), "setConfig with post-prepare map should succeed.");

    const Eigen::VectorXd x_after = optimizer.generateInitialGuess(context);
    expectTrue((x_before - x_after).norm() < 1e-12,
               "Prepared context should keep using the spatial map snapshot captured during prepareContext.");
}

void testEvaluateRejectsInvalidDecodedState()
{
    using Optimizer = SplineTrajectory::SplineOptimizer<3,
                                                        SplineTrajectory::QuinticSplineND<3>,
                                                        SplineTrajectory::IdentityTimeMap>;

    Optimizer optimizer;
    typename Optimizer::OptimizationContext context;
    auto problem = makeProblem<Optimizer>(1);
    expectTrue(static_cast<bool>(optimizer.prepareContext(problem, context)),
               "prepareContext should succeed for IdentityTimeMap test.");

    LinearTimeCost time_cost;
    ZeroIntegralCost integral_cost;
    Eigen::VectorXd grad;
    Eigen::VectorXd x = optimizer.generateInitialGuess(context);
    x(0) = -0.5;

    const auto spec = Optimizer::makeEvaluateSpec(time_cost, integral_cost);
    const auto result = optimizer.evaluate(context, x, grad, spec);
    expectTrue(!result, "evaluate should reject negative durations after decode.");
}

void testCheckGradientsRestoresWorkingState()
{
    using Optimizer = SplineTrajectory::SplineOptimizer<3,
                                                        SplineTrajectory::QuinticSplineND<3>,
                                                        SplineTrajectory::IdentityTimeMap>;

    Optimizer optimizer;
    typename Optimizer::OptimizationContext context;
    auto problem = makeProblem<Optimizer>(1);
    expectTrue(static_cast<bool>(optimizer.prepareContext(problem, context)),
               "prepareContext should succeed for gradient-check test.");

    LinearTimeCost time_cost;
    ZeroIntegralCost integral_cost;
    ZeroWaypointsCost waypoints_cost;
    ZeroSampleCost sample_cost;
    ZeroTrajectoryCost trajectory_cost;

    Eigen::VectorXd x = optimizer.generateInitialGuess(context);
    auto spec = Optimizer::makeEvaluateSpec(time_cost, integral_cost)
                    .withWaypointsCost(waypoints_cost)
                    .withSampleCost(sample_cost)
                    .withTrajectoryCost(trajectory_cost);

    auto check = optimizer.checkGradients(context, x, spec, 1e-6, 1e-3);
    expectTrue(check.valid, "checkGradients should succeed on the simple smooth test problem.");

    const auto &times = optimizer.getWorkingSpline(context).getTimeSegments();
    expectTrue(times.size() == 1, "Working spline should contain the original single time segment.");
    expectTrue(std::abs(times.front() - problem.time_segments.front()) < 1e-12,
               "checkGradients should restore the working spline state to the original decision vector.");
}

void testPreparedEvaluationAndWorkingStateSynchronization()
{
    using Optimizer = SplineTrajectory::SplineOptimizer<3>;
    Optimizer optimizer;
    Optimizer::OptimizationContext checked_context;
    Optimizer::OptimizationContext prepared_context;
    auto problem = makeProblem<Optimizer>(3);
    expectTrue(static_cast<bool>(optimizer.prepareContext(problem, checked_context)),
               "Checked context preparation should succeed.");
    expectTrue(static_cast<bool>(optimizer.prepareContext(problem, prepared_context)),
               "Prepared context preparation should succeed.");

    LinearTimeCost time_cost;
    HookedZeroIntegralCost checked_integral_cost;
    HookedZeroIntegralCost prepared_integral_cost;
    const auto checked_spec = Optimizer::makeEvaluateSpec(time_cost, checked_integral_cost);
    const auto prepared_spec = Optimizer::makeEvaluateSpec(time_cost, prepared_integral_cost);
    Eigen::VectorXd x = optimizer.generateInitialGuess(checked_context);
    Eigen::VectorXd checked_grad;
    Eigen::VectorXd prepared_grad;

    const auto checked = optimizer.evaluate(checked_context, x, checked_grad, checked_spec);
    const double prepared_cost =
        optimizer.evaluatePrepared(prepared_context, x, prepared_grad, prepared_spec);
    expectTrue(static_cast<bool>(checked), "Checked evaluation should succeed.");
    expectTrue(std::abs(checked.cost - prepared_cost) < 1e-14,
               "Prepared evaluation must preserve the checked cost.");
    expectTrue((checked_grad - prepared_grad).norm() < 1e-14,
               "Prepared evaluation must preserve the checked gradient.");
    expectTrue(checked_integral_cost.begin_calls == 1 &&
                   prepared_integral_cost.begin_calls == 1,
               "Integral beginEvaluation hook should run exactly once per evaluation.");

    x(0) += 0.2;
    const auto sync = optimizer.synchronizeWorkingState(prepared_context, x);
    expectTrue(static_cast<bool>(sync), "Working-state synchronization should succeed.");
    const double expected_time = SplineTrajectory::QuadInvTimeMap().toTime(x(0));
    const auto &synced_times = optimizer.getWorkingSpline(prepared_context).getTimeSegments();
    expectTrue(std::abs(synced_times.front() - expected_time) < 1e-14,
               "Working-state synchronization should rebuild the spline without a cost evaluation.");
}

void testIntegralPointMetadata()
{
    using Optimizer = SplineTrajectory::SplineOptimizer<3>;
    Optimizer optimizer;
    auto config = optimizer.getActiveConfig();
    config.integral_num_steps = 2;
    expectTrue(static_cast<bool>(optimizer.setConfig(config)),
               "Two-step integration config should be valid.");

    Optimizer::OptimizationContext context;
    auto problem = makeProblem<Optimizer>(2);
    expectTrue(static_cast<bool>(optimizer.prepareContext(problem, context)),
               "Metadata test context preparation should succeed.");

    LinearTimeCost time_cost;
    RecordingZeroIntegralCost integral_cost;
    const auto spec = Optimizer::makeEvaluateSpec(time_cost, integral_cost);
    Eigen::VectorXd x = optimizer.generateInitialGuess(context);
    Eigen::VectorXd grad;
    const auto result = optimizer.evaluate(context, x, grad, spec);
    expectTrue(static_cast<bool>(result), "Metadata test evaluation should succeed.");
    expectTrue(integral_cost.points.size() == 6,
               "Two segments with K=2 must produce six quadrature visits.");

    const auto &left_end = integral_cost.points[2];
    const auto &right_start = integral_cost.points[3];
    expectTrue(left_end.segment_index == 0 && left_end.step_index == 2 &&
                   left_end.interiorBoundaryIndex() == 0,
               "Left segment endpoint must map to the internal boundary exactly.");
    expectTrue(right_start.segment_index == 1 && right_start.step_index == 0 &&
                   right_start.interiorBoundaryIndex() == 0,
               "Right segment start must map to the same internal boundary exactly.");
    expectTrue(integral_cost.points.front().isTrajectoryStart() &&
                   integral_cost.points.back().isTrajectoryEnd(),
               "Trajectory endpoint metadata must be exact integer topology.");
    expectTrue(std::abs(left_end.global_time - right_start.global_time) < 1e-14,
               "Both visits of one internal boundary must share global time.");
    expectTrue(std::abs(left_end.alpha - 1.0) < 1e-14 &&
                   std::abs(right_start.alpha) < 1e-14,
               "Quadrature metadata must expose the existing normalized time.");
    expectTrue(std::abs(left_end.step_size * left_end.step_count -
                            left_end.segment_duration) < 1e-14,
               "Quadrature step size must come from the optimizer's decoded duration.");
}

void testCoefficientCostUsesSingleSplineAdjoint()
{
    using Optimizer = SplineTrajectory::SplineOptimizer<
        3, SplineTrajectory::MinDerivativeSplineND<3, 3>,
        SplineTrajectory::IdentityTimeMap>;
    Optimizer optimizer;
    Optimizer::OptimizationContext context;
    auto problem = makeProblem<Optimizer>(2, true);
    expectTrue(
        static_cast<bool>(
            optimizer.prepareContext(problem, context)),
        "Coefficient-cost context preparation should succeed.");

    LinearTimeCost time_cost;
    ZeroIntegralCost integral_cost;
    QuadraticHullCoefficientCost hull_cost;
    const auto spec =
        Optimizer::makeEvaluateSpec(time_cost, integral_cost)
            .withCoefficientCost(hull_cost);
    const Eigen::VectorXd x =
        optimizer.generateInitialGuess(context);
    const auto check =
        optimizer.checkGradients(context, x, spec, 2e-6, 2e-3);
    expectTrue(
        check.valid && check.max_abs_error < 2e-4,
        "CoefficientCost hull gradients must pass the optimizer-level finite-difference check.");
}
} // namespace

int main()
{
    testPrepareRejectsBadMask();
    testPreparedContextSnapshotsSpatialMap();
    testEvaluateRejectsInvalidDecodedState();
    testCheckGradientsRestoresWorkingState();
    testPreparedEvaluationAndWorkingStateSynchronization();
    testIntegralPointMetadata();
    testCoefficientCostUsesSingleSplineAdjoint();

    std::cout << "test_spline_optimizer passed" << std::endl;
    return 0;
}
