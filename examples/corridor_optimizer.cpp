#include "support/Penalties.hpp"

namespace example = spline_examples;
/** @brief Integrate segment-specific corridor and derivative-bound penalties. */
struct CorridorCost
{
    static constexpr int kDerivativeOrder = 2;
    std::array<example::Box, 3> boxes = example::corridor();
    example::DynamicsCost dynamics;
    SplineTrajectory::EvaluationResult operator()(const SplineTrajectory::IntegralPointInfo &point,
                      const SplineTrajectory::SampleState<2> &state,
                      SplineTrajectory::SampleGradient<2> &gradient) const
    {
        const auto result = dynamics(point, state, gradient);
        if (!result) return result;
        return {boxes[point.segment_index].penalty(state.p, 5000.0, gradient.p)
             + result.cost, SplineTrajectory::Status::success()};
    }
};
struct Objective { example::DurationCost duration; CorridorCost integral; };

int main(int argc, char **argv) try
{
    auto problem = example::makeProblem();
    problem.waypoints(1, 1) = 0.85;
    example::Optimizer optimizer;
    const auto prepared = optimizer.prepare(problem, example::options());
    if (!prepared) throw std::runtime_error(prepared.message);
    Objective objective;
    example::minimize(optimizer, objective, example::checkRequested(argc, argv));

    double violation = 0.0;
    for (int piece = 0; piece < optimizer.polynomial().numSegments(); ++piece)
    {
        const auto segment = optimizer.polynomial().at(piece).value();
        for (int i = 0; i <= 200; ++i)
        {
            example::Vector scratch = example::Vector::Zero();
            violation = std::max(violation, std::sqrt(objective.integral.boxes[piece].penalty(
                segment.evaluateLocal(segment.duration() * i / 200.0).value(), 1.0, scratch)));
        }
    }
    std::cout << "sampled corridor violation: " << violation << '\n';
    example::printDynamics(optimizer.polynomial());
    return 0;
}
catch (const std::exception &error)
{
    std::cerr << error.what() << '\n';
    return 1;
}
