#pragma once
#include "SplineOptimizer.hpp"
#include <gcopter/lbfgs.hpp>
#include <algorithm>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <stdexcept>
#include <string>

namespace spline_examples
{
using Spline = SplineTrajectory::QuinticSplineND<2>;
using Optimizer = SplineTrajectory::SplineOptimizer<Spline>;
using Vector = Eigen::Vector2d;

struct DurationCost
{
    double weight = 0.5;
    double operator()(const std::vector<double> &durations, Eigen::Ref<Eigen::VectorXd> gradient) const
    {
        gradient.array() += weight;
        return weight * std::accumulate(durations.begin(), durations.end(), 0.0);
    }
};

inline Optimizer::Problem makeProblem()
{
    Optimizer::Problem problem;
    problem.durations = {1.5, 1.5, 1.5};
    problem.waypoints.resize(4, 2);
    problem.waypoints << 0.0, 0.0, 2.0, 0.2, 4.0, 0.9, 6.0, 1.0;
    return problem;
}

inline SplineTrajectory::OptimizerOptions options()
{
    SplineTrajectory::OptimizerOptions result;
    result.energy_weight = 0.02;
    result.integration_steps = 32;
    return result;
}

inline bool checkRequested(int argc, char **argv)
{
    if (argc == 1) return false;
    if (argc == 2 && std::string(argv[1]) == "--check-gradients") return true;
    throw std::invalid_argument("Usage: example [--check-gradients]");
}

/** @brief Minimize a prepared objective with GCOPTER L-BFGS and publish its accepted trajectory.
 * @param[in,out] optimizer Prepared, exclusively owned optimizer.
 * @param[in,out] objective Cost callbacks, borrowed only for this synchronous solve.
 * @param check_gradient Check the initial decision gradient before solving when true.
 * @return Accepted objective value; throws on evaluation or solver failure.
 */
template<class Engine, class Objective>
double minimize(Engine &optimizer, Objective &objective, bool check_gradient = false)
{
    auto x = optimizer.initialGuess();
    if (check_gradient)
    {
        const auto checked = SplineTrajectory::checkGradients(optimizer, x, objective, 2e-6, 2e-4);
        if (!checked)
        {
            std::cerr << "gradient error: " << checked.error_norm << ", relative: " << checked.relative_error << '\n';
            throw std::runtime_error(std::string("Objective gradient check failed: ") + checked.status.message);
        }
    }
    struct CallbackData { Engine &optimizer; Objective &objective; } data{optimizer, objective};
    auto callback = [](void *instance, const Eigen::VectorXd &variables, Eigen::VectorXd &gradient)
    {
        auto &context = *static_cast<CallbackData *>(instance);
        return context.optimizer.costAndGradient(variables, gradient, context.objective);
    };
    Eigen::VectorXd gradient(x.size());
    const double initial_cost = callback(&data, x, gradient);
    if (!std::isfinite(initial_cost)) throw std::runtime_error("Invalid initial objective");
    lbfgs::lbfgs_parameter_t parameters;
    parameters.mem_size = 64;
    parameters.past = 5;
    parameters.delta = 1e-9;
    parameters.g_epsilon = 1e-6;
    parameters.max_iterations = 1000;
    parameters.max_linesearch = 128;
    double cost = initial_cost;
    const int code = lbfgs::lbfgs_optimize(x, cost, callback, nullptr, nullptr, &data, parameters);
    if (code < 0) throw std::runtime_error(lbfgs::lbfgs_strerror(code));
    cost = callback(&data, x, gradient);
    if (!std::isfinite(cost) || !gradient.allFinite() || cost > initial_cost + 1e-8 * std::max(1.0, std::abs(initial_cost)))
        throw std::runtime_error("Invalid final objective");
    const auto accepted = optimizer.updateAccepted(x);
    if (!accepted) throw std::runtime_error(accepted.message);
    std::cout << std::setprecision(6) << "cost: " << initial_cost << " -> " << cost
              << ", duration: " << optimizer.polynomial().duration()
              << ", solver: " << lbfgs::lbfgs_strerror(code) << '\n';
    return cost;
}

/** @brief Print sampled derivative maxima; these are observations, not continuous bounds. */
template<class Polynomial>
void printDynamics(const Polynomial &polynomial)
{
    auto sampler = polynomial.template makeSampler<2>();
    double speed = 0.0, acceleration = 0.0;
    for (int i = 0; i <= 1000; ++i)
    {
        const auto state = sampler.evaluate(polynomial.startTime() + polynomial.duration() * i / 1000.0);
        speed = std::max(speed, state[1].norm());
        acceleration = std::max(acceleration, state[2].norm());
    }
    std::cout << "sampled peak speed: " << speed << ", acceleration: " << acceleration << '\n';
}
}
