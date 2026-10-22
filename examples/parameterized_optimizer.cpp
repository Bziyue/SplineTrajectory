#include <stdexcept>
#include "SplineOptimizer.hpp"
#include "SplineConvexHull.hpp"
#include <iostream>

namespace
{
using Spline = SplineTrajectory::SepticSplineND<3>;

struct ExponentialTimeMap
{
    SplineTrajectory::StatusOr<double> forward(double value) const
    {
        const double duration = std::exp(value);
        if (!std::isfinite(duration) || duration <= 0.0)
            return SplineTrajectory::Status{SplineTrajectory::ErrorCode::NumericalFailure, "Unrepresentable exponential duration"};
        return duration;
    }
    SplineTrajectory::StatusOr<double> inverse(double duration) const
    {
        if (!std::isfinite(duration) || duration <= 0.0)
            return SplineTrajectory::Status{SplineTrajectory::ErrorCode::InvalidInput, "Expected positive finite duration"};
        return std::log(duration);
    }
    SplineTrajectory::StatusOr<double> backward(double, double duration, double gradient) const
    {
        const double result = duration * gradient;
        if (!std::isfinite(result))
            return SplineTrajectory::Status{SplineTrajectory::ErrorCode::NumericalFailure, "Nonfinite time gradient"};
        return result;
    }
};

/** @brief Smooth coordinates inside a box; seed points must lie strictly inside it. */
struct BoxCoordinates
{
    Eigen::Vector3d extent{4.0, 3.0, 2.0};
    int dimension(int) const { return 3; }
    SplineTrajectory::StatusOr<Eigen::Vector3d> forward(const Eigen::Ref<const Eigen::VectorXd> &x, int) const
    {
        const Eigen::Vector3d result = (extent.array() * x.array().tanh()).matrix();
        if (!result.allFinite())
            return SplineTrajectory::Status{SplineTrajectory::ErrorCode::NumericalFailure, "Nonfinite box coordinate"};
        return result;
    }
    SplineTrajectory::StatusOr<Eigen::Vector3d> inverse(const Eigen::Vector3d &point, int) const
    {
        if ((point.array().abs() >= extent.array()).any())
            return SplineTrajectory::Status{SplineTrajectory::ErrorCode::InvalidInput, "Initial point must be strictly inside the box"};
        return Eigen::Vector3d((point.array() / extent.array()).unaryExpr([](double value) { return std::atanh(value); }).matrix());
    }
    SplineTrajectory::Status backward(const Eigen::Ref<const Eigen::VectorXd> &x, const Eigen::Vector3d &gradient,
                      int, Eigen::Ref<Eigen::VectorXd> output) const
    { output = (gradient.array() * extent.array() * (1.0 - x.array().tanh().square())).matrix();
        return SplineTrajectory::Status::success();
    }
};

/** @brief Two auxiliary coordinates: common start-time offset and logarithmic duration scale. */
struct SharedCoordinates
{
    int dimension() const { return 2; }
    SplineTrajectory::StatusOr<Eigen::VectorXd> initial(const SplineTrajectory::SplineInput<Spline::kDimension> &) const
    { return Eigen::VectorXd(Eigen::VectorXd::Zero(2)); }
    SplineTrajectory::Status apply(const Eigen::Ref<const Eigen::VectorXd> &x, SplineTrajectory::MutableParameters<Spline> &state) const
    {
        state.start_time += x(0);
        const double scale = std::exp(x(1));
        if (!std::isfinite(state.start_time) || !std::isfinite(scale) || scale <= 0.0)
            return {SplineTrajectory::ErrorCode::NumericalFailure, "Unrepresentable auxiliary transform"};
        for (double &duration : state.durations)
        {
            duration *= scale;
            if (!std::isfinite(duration) || duration <= 0.0)
                return {SplineTrajectory::ErrorCode::NumericalFailure, "Unrepresentable scaled duration"};
        }

        return SplineTrajectory::Status::success();
    }
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
        gradient.trajectory_time += 0.02 * point.trajectory_time;
        return 0.01 * state.p.squaredNorm() + 0.02 * state.v.squaredNorm() +
               0.01 * point.trajectory_time * point.trajectory_time;
    }
};

/** @brief Quadratic control-point objective; prepare its owned hull storage outside callbacks. */
struct CoefficientCost
{
    SplineTrajectory::ConvexHullWorkspace<3> hull;
    Spline::CoefficientMatrix control_gradient;
    explicit CoefficientCost(int segments)
    {
        hull.prepare(segments, Spline::kCoefficientCount, {SplineTrajectory::ConvexHullBasis::Bezier, 0, 1});
        control_gradient.resize(hull.controls().rows(), 3);
    }
    double operator()(const SplineTrajectory::ParameterView<Spline> &state,
                      SplineTrajectory::CoefficientGradientView<Spline> &gradient)
    {
        hull.update(state.polynomial);
        control_gradient = 0.002 * hull.controls();
        hull.backwardAdd(control_gradient, gradient);
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
    ProblemObjective objective{{}, CoefficientCost(static_cast<int>(problem.durations.size()))};
    const auto prepared = optimizer.prepare(problem, options, objective);
    if (!prepared) { std::cerr << prepared.message << '\n'; return 1; }
    auto x = optimizer.initialGuess();
    x(optimizer.layout().auxiliary_offset) = 0.2;
    x(optimizer.layout().auxiliary_offset + 1) = 0.1;
    const auto check = SplineTrajectory::checkGradients(optimizer, x, objective, 1e-5, 2e-4);
    if (!check) { std::cerr << check.status.message << ": " << check.error_norm << '\n'; return 1; }
    std::cout << "dimension=" << optimizer.decisionDimension() << " maximum_gradient_error=" << check.max_absolute_error << '\n';
}
