#include <cassert>
#include <cmath>
#include <iostream>
#include <string>
#include <vector>

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

    double operator()(double t,
                      double t_global,
                      int segment_index,
                      int step_in_seg,
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
        (void)t;
        (void)t_global;
        (void)segment_index;
        (void)step_in_seg;
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
} // namespace

int main()
{
    testPrepareRejectsBadMask();
    testPreparedContextSnapshotsSpatialMap();
    testEvaluateRejectsInvalidDecodedState();
    testCheckGradientsRestoresWorkingState();

    std::cout << "test_spline_optimizer passed" << std::endl;
    return 0;
}
