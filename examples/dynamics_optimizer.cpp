#include "support/Penalties.hpp"

namespace example = spline_examples;
struct Objective
{
    example::DynamicsCost integral;
};

int main(int argc, char **argv) try
{
    auto problem = example::makeProblem();
    problem.durations = {1.2, 1.4, 1.2};
    // Keep the spatial path fixed and optimize its segment durations.
    auto options = example::options();
    options.mask = SplineTrajectory::OptimizationMask::none(static_cast<int>(problem.durations.size()));
    options.mask.durations.assign(problem.durations.size(), 1);
    example::Optimizer optimizer;
    Objective objective;
    const auto prepared = optimizer.prepare(problem, options, objective);
    if (!prepared) throw std::runtime_error(prepared.message);

    example::minimize(optimizer, objective, example::checkRequested(argc, argv));
    example::printDynamics(optimizer.polynomial());
    return 0;
}
catch (const std::exception &error)
{
    std::cerr << error.what() << '\n';
    return 1;
}
