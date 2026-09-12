#include "SplineConvexHull.hpp"
#include "support/Penalties.hpp"

namespace example = spline_examples;
/** @brief Penalize Bezier position, velocity and acceleration controls in convex sets.
 * @note Each objective owns its mutable hull workspaces. A soft penalty does not enforce exact feasibility. */
struct HullCost
{
    using Matrix = example::Spline::CoefficientMatrix;
    std::array<SplineTrajectory::ConvexHullWorkspace<2>, 3> hulls;
    std::array<Matrix, 3> control_gradients;
    std::array<example::Box, 3> boxes = example::corridor();

    explicit HullCost(const example::Spline::Polynomial &polynomial)
    {
        for (int derivative = 0; derivative < 3; ++derivative)
        {
            hulls[derivative].configure(polynomial, {SplineTrajectory::ConvexHullBasis::Bezier, derivative, 1});
            control_gradients[derivative].resize(hulls[derivative].controls().rows(), 2);
        }
    }

    double operator()(const SplineTrajectory::ParameterView<example::Spline> &state,
                      SplineTrajectory::CoefficientGradient<example::Spline> &gradient)
    {
        double cost = 0.0;
        for (int derivative = 0; derivative < 3; ++derivative)
        {
            auto &hull = hulls[derivative];
            auto &partials = control_gradients[derivative];
            hull.update(state.polynomial);
            partials.setZero();
            for (int piece = 0; piece < hull.numPieces(); ++piece)
            {
                const auto &box = boxes[hull.pieceInfo(piece).source_segment];
                for (int local = 0; local < hull.controlsPerPiece(); ++local)
                {
                    const int row = piece * hull.controlsPerPiece() + local;
                    const example::Vector value = hull.controls().row(row).transpose();
                    example::Vector partial = example::Vector::Zero();
                    cost += derivative == 0 ? box.penalty(value, 5000.0, partial)
                        : example::normPenalty(value, derivative == 1 ? 2.5 : 4.0, 100.0, partial);
                    partials.row(row) = partial.transpose();
                }
            }
            // Include explicit duration partials from conversion, in addition to coefficient partials.
            hull.backwardAdd(partials, gradient.coefficients, gradient.durations);
        }
        return cost;
    }
};
struct Objective { example::DurationCost duration; HullCost coefficient; };

int main(int argc, char **argv) try
{
    auto problem = example::makeProblem();
    problem.waypoints(1, 1) = 0.85;
    example::Optimizer optimizer;
    const auto prepared = optimizer.prepare(problem, example::options());
    if (!prepared) throw std::runtime_error(prepared.message);

    Objective objective{{}, HullCost(optimizer.polynomial())};
    example::minimize(optimizer, objective, example::checkRequested(argc, argv));
    // Inspect controls of the accepted polynomial, rather than the last line-search trial.
    double corridor_violation = 0.0;
    for (int derivative = 0; derivative < 3; ++derivative)
    {
        auto &hull = objective.coefficient.hulls[derivative];
        hull.update(optimizer.polynomial());
        if (derivative != 0)
        {
            std::cout << "control bound on " << (derivative == 1 ? "speed: " : "acceleration: ")
                      << hull.controls().rowwise().norm().maxCoeff() << '\n';
            continue;
        }
        for (int piece = 0; piece < hull.numPieces(); ++piece)
            for (int local = 0; local < hull.controlsPerPiece(); ++local)
            {
                example::Vector scratch = example::Vector::Zero();
                const example::Vector point = hull.pieceControls(piece).row(local).transpose();
                corridor_violation = std::max(corridor_violation, std::sqrt(
                    objective.coefficient.boxes[hull.pieceInfo(piece).source_segment].penalty(point, 1.0, scratch)));
            }
    }
    std::cout << "control corridor violation: " << corridor_violation << '\n';
    return 0;
}
catch (const std::exception &error)
{
    std::cerr << error.what() << '\n';
    return 1;
}
