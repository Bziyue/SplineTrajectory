#include "support/OptimizationExample.hpp"

namespace example = spline_examples;
struct Objective { example::DurationCost duration; };

int main(int argc, char **argv) try
{
    example::Optimizer optimizer;
    const auto prepared = optimizer.prepare(example::makeProblem(), example::options());
    if (!prepared) throw std::runtime_error(prepared.message);

    // Minimum-jerk energy is enabled by options; add a linear cost on total duration.
    Objective objective;
    example::minimize(optimizer, objective, example::checkRequested(argc, argv));
    const auto midpoint = optimizer.polynomial().evaluate(optimizer.polynomial().duration() / 2.0);
    std::cout << "midpoint: " << midpoint.transpose() << '\n';
    return 0;
}
catch (const std::exception &error)
{
    std::cerr << error.what() << '\n';
    return 1;
}
