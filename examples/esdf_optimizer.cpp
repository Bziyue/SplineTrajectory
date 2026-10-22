#include "support/Penalties.hpp"

namespace example = spline_examples;
/** @brief Analytic circle ESDF in metres; replace this query with a map's distance and spatial gradient. */
struct CircleDistance
{
    example::Vector center{3.0, 0.0};
    double radius = 0.7;
    double query(const example::Vector &point, example::Vector &gradient) const
    {
        const example::Vector delta = point - center;
        const double length = delta.norm();
        // The ESDF is nondifferentiable at the centre; choose the zero subgradient there.
        gradient = length > 0.0 ? example::Vector(delta / length) : example::Vector::Zero();
        return length - radius;
    }
};
struct EsdfCost
{
    static constexpr int kDerivativeOrder = 2;
    CircleDistance field;
    example::DynamicsCost dynamics;
    double clearance = 0.2;
    double operator()(const SplineTrajectory::IntegralPointInfo &point,
                      const SplineTrajectory::IntegralState<2> &state,
                      SplineTrajectory::IntegralGradient<2> &gradient) const
    {
        double cost = dynamics(point, state, gradient);
        example::Vector direction;
        const double violation = clearance - field.query(state.p, direction);
        if (violation > 0.0)
        {
            cost += 5000.0 * violation * violation;
            gradient.p -= 10000.0 * violation * direction;
        }
        return cost;
    }
};
struct Objective { EsdfCost integral; };

int main(int argc, char **argv) try
{
    auto problem = example::makeProblem();
    problem.waypoints << 0, 0, 2, 0.25, 4, 0.25, 6, 0;
    example::Optimizer optimizer;
    Objective objective;
    const auto prepared = optimizer.prepare(problem, example::options(), objective);
    if (!prepared) throw std::runtime_error(prepared.message);
    example::minimize(optimizer, objective, example::checkRequested(argc, argv));

    const auto &trajectory = optimizer.polynomial();
    auto sampler = trajectory.makeSampler<0>();
    double minimum_distance = std::numeric_limits<double>::infinity();
    for (int i = 0; i <= 1000; ++i)
    {
        example::Vector scratch;
        const auto position = sampler.evaluate(trajectory.startTime() + trajectory.duration() * i / 1000.0);
        minimum_distance = std::min(minimum_distance, objective.integral.field.query(position, scratch));
    }
    std::cout << "minimum sampled ESDF: " << minimum_distance << '\n';
    if (minimum_distance <= 0.0) throw std::runtime_error("The example trajectory intersects the obstacle");
    example::printDynamics(optimizer.polynomial());
    return 0;
}
catch (const std::exception &error)
{
    std::cerr << error.what() << '\n';
    return 1;
}
