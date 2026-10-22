#include "support/OptimizationExample.hpp"

namespace example = spline_examples;

int main(int argc, char **argv) try
{
    example::Optimizer optimizer;
    SplineTrajectory::NoObjective objective;
    const auto prepared = optimizer.prepare(example::makeProblem(), example::options(), objective);
    if (!prepared) throw std::runtime_error(prepared.message);

    // Options enable minimum-jerk energy and a linear cost on total duration.
    example::minimize(optimizer, objective, example::checkRequested(argc, argv));
    const auto &trajectory = optimizer.polynomial();
    const auto midpoint = trajectory.evaluate(trajectory.startTime() + trajectory.duration() / 2.0);
    std::cout << "midpoint: " << midpoint.transpose() << '\n';
    return 0;
}
catch (const std::exception &error)
{
    std::cerr << error.what() << '\n';
    return 1;
}
