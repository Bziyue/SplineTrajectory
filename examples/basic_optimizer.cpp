#include "SplineOptimizer.hpp"
#include <iostream>

namespace
{
using Spline = SplineTrajectory::QuinticSplineND<2>;
using Optimizer = SplineTrajectory::SplineOptimizer<Spline>;

struct DurationCost
{
    double operator()(const std::vector<double> &durations, Eigen::Ref<Eigen::VectorXd> gradient) const
    {
        gradient.array() += 0.2;
        double cost = 0.0;
        for (double duration : durations) cost += 0.2 * duration;
        return cost;
    }
};

struct IntegralCost
{
    static constexpr int kDerivativeOrder = 1;
    double operator()(const SplineTrajectory::IntegralPointInfo &,
                      const SplineTrajectory::SampleState<2> &state,
                      SplineTrajectory::SampleGradient<2> &gradient) const
    {
        gradient.v += 0.1 * state.v;
        return 0.05 * state.v.squaredNorm();
    }
};

struct ProblemObjective
{
    DurationCost duration;
    IntegralCost integral;
};
}

int main()
{
    Optimizer optimizer;
    Optimizer::Problem problem;
    problem.durations = {1.5, 1.5};
    problem.waypoints.resize(3, 2);
    problem.waypoints << 0, 0, 1, 0.5, 2, 0;
    SplineTrajectory::OptimizerOptions options;
    options.integration_steps = 16;
    options.energy_weight = 0.01;
    const auto prepared = optimizer.prepare(problem, options);
    if (!prepared) { std::cerr << prepared.message << '\n'; return 1; }

    ProblemObjective objective;
    auto x = optimizer.initialGuess();
    Eigen::VectorXd gradient(x.size()), trial_gradient(x.size());
    // A small external descent loop illustrates the callback contract; the library does not own a solver.
    for (int iteration = 0; iteration < 40; ++iteration)
    {
        const auto result = optimizer.evaluate(x, gradient, objective);
        if (!result) { std::cerr << result.status.message << '\n'; return 1; }
        if (gradient.norm() < 1e-6) break;
        double step = 0.1;
        for (int search = 0; search < 20; ++search)
        {
            const Eigen::VectorXd trial = x - step * gradient;
            const auto candidate = optimizer.evaluate(trial, trial_gradient, objective);
            if (candidate && candidate.cost <= result.cost - 1e-4 * step * gradient.squaredNorm())
            { x = trial; break; }
            step *= 0.5;
        }
    }
    // A rejected line-search trial may have been the last callback.
    const auto accepted = optimizer.updateAccepted(x);
    if (!accepted) { std::cerr << accepted.message << '\n'; return 1; }
    auto sampler = optimizer.polynomial().makeSampler<1>();
    const auto middle = sampler.evaluate(0.5 * optimizer.polynomial().duration());
    std::cout << "duration=" << optimizer.polynomial().duration()
              << " midpoint=" << middle[0].transpose() << '\n';
}
