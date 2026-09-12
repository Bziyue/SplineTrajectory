#include "SplineOptimizer.hpp"
#include "SplineConvexHull.hpp"
#include <iostream>

namespace
{
using Spline = SplineTrajectory::SepticSplineND<3>;

struct ExponentialTimeMap
{
    double toTime(double value) const { return std::exp(value); }
    double toTau(double duration) const { return std::log(duration); }
    double backward(double, double duration, double gradient) const { return duration * gradient; }
};

/** @brief Smooth coordinates inside a box; seed points must lie strictly inside it. */
struct BoxCoordinates
{
    Eigen::Vector3d extent{4.0, 3.0, 2.0};
    int dimension(int) const { return 3; }
    Eigen::Vector3d toPhysical(const Eigen::Ref<const Eigen::VectorXd> &x, int) const
    { return (extent.array() * x.array().tanh()).matrix(); }
    Eigen::Vector3d toUnconstrained(const Eigen::Vector3d &point, int) const
    { return (point.array() / extent.array()).unaryExpr([](double value) { return std::atanh(value); }).matrix(); }
    void backwardInto(const Eigen::Ref<const Eigen::VectorXd> &x, const Eigen::Vector3d &gradient,
                      int, Eigen::Ref<Eigen::VectorXd> output) const
    { output = (gradient.array() * extent.array() * (1.0 - x.array().tanh().square())).matrix(); }
};

/** @brief Two auxiliary coordinates: common start-time offset and logarithmic duration scale. */
struct SharedCoordinates
{
    int dimension() const { return 2; }
    Eigen::VectorXd initial(const SplineTrajectory::SplineProblem<Spline> &) const
    { return Eigen::VectorXd::Zero(2); }
    void apply(const Eigen::Ref<const Eigen::VectorXd> &x, SplineTrajectory::MutableParameters<Spline> &state) const
    {
        state.start_time += x(0);
        for (double &duration : state.durations) duration *= std::exp(x(1));
    }
    double backward(const Eigen::Ref<const Eigen::VectorXd> &x,
                    const SplineTrajectory::ParameterView<Spline> &state,
                    SplineTrajectory::ParameterGradient<Spline> &gradient,
                    Eigen::Ref<Eigen::VectorXd> output) const
    {
        output(0) += gradient.start_time;
        for (int i = 0; i < gradient.durations.size(); ++i)
            output(1) += gradient.durations(i) * state.durations[i];
        gradient.durations *= std::exp(x(1));
        return 0.0;
    }
};

struct ProblemParameterization
{
    ExponentialTimeMap time;
    BoxCoordinates space;
    SharedCoordinates auxiliary;
};

struct IntegralCost
{
    static constexpr int kDerivativeOrder = 1;
    double operator()(const SplineTrajectory::IntegralPointInfo &point,
                      const SplineTrajectory::SampleState<3> &state,
                      SplineTrajectory::SampleGradient<3> &gradient) const
    {
        gradient.p += 0.02 * state.p;
        gradient.v += 0.04 * state.v;
        gradient.time += 0.02 * point.global_time;
        return 0.01 * state.p.squaredNorm() + 0.02 * state.v.squaredNorm() +
               0.01 * point.global_time * point.global_time;
    }
};

/** @brief Quadratic control-point objective; prepare its owned hull storage outside callbacks. */
struct CoefficientCost
{
    SplineTrajectory::ConvexHullWorkspace<3> hull;
    Spline::CoefficientMatrix control_gradient;
    explicit CoefficientCost(const Spline::Polynomial &polynomial)
        : hull(SplineTrajectory::toBezier(polynomial, 0, 1))
    { control_gradient.resize(hull.controls().rows(), 3); }
    double operator()(const SplineTrajectory::ParameterView<Spline> &state,
                      SplineTrajectory::CoefficientGradient<Spline> &gradient)
    {
        hull.update(state.polynomial);
        control_gradient = 0.002 * hull.controls();
        hull.backwardAdd(control_gradient, gradient.coefficients, gradient.durations);
        return 0.001 * hull.controls().squaredNorm();
    }
};

struct ProblemObjective
{
    IntegralCost integral;
    CoefficientCost coefficient;
};
}

int main()
{
    SplineTrajectory::SplineOptimizer<Spline, ProblemParameterization> optimizer;
    decltype(optimizer)::Problem problem;
    problem.durations = {1.5, 1.7};
    problem.start_time = 0.5;
    problem.waypoints.resize(3, 3);
    problem.waypoints << 0,0,0, 1,0.3,0.2, 2,0,0;
    SplineTrajectory::OptimizerOptions options;
    options.energy_weight = 0.003;
    options.integration_steps = 10;
    const auto prepared = optimizer.prepare(problem, options);
    if (!prepared) { std::cerr << prepared.message << '\n'; return 1; }
    ProblemObjective objective{{}, CoefficientCost(optimizer.polynomial())};
    auto x = optimizer.initialGuess();
    x(optimizer.layout().auxiliary_offset) = 0.2;
    x(optimizer.layout().auxiliary_offset + 1) = 0.1;
    const auto check = SplineTrajectory::checkGradients(optimizer, x, objective, 1e-5, 2e-4);
    if (!check) { std::cerr << check.status.message << ": " << check.error_norm << '\n'; return 1; }
    std::cout << "dimension=" << optimizer.dimension() << " maximum_gradient_error=" << check.max_absolute_error << '\n';
}
