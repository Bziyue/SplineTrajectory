#include "support/Penalties.hpp"

namespace example = spline_examples;
struct Objective
{
    example::DurationCost duration;
    example::DynamicsCost integral;
};

int main(int argc, char **argv) try
{
    auto problem = example::makeProblem();
    problem.durations = {1.2, 1.4, 1.2};
    // Keep the spatial path fixed and optimize its segment durations.
    SplineTrajectory::OptimizationMask mask;
    mask.waypoints.assign(problem.waypoints.rows(), 0);
    problem.mask = mask;
    example::Optimizer optimizer;
    const auto prepared = optimizer.prepare(problem, example::options());
    if (!prepared) throw std::runtime_error(prepared.message);

    Objective objective;
    example::minimize(optimizer, objective, example::checkRequested(argc, argv));
    example::printDynamics(optimizer.polynomial());
    return 0;
}
catch (const std::exception &error)
{
    std::cerr << error.what() << '\n';
    return 1;
}
