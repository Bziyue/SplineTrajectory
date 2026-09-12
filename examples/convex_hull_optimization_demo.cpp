#include <Eigen/Dense>

#include <algorithm>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <functional>
#include <iomanip>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "SplineConvexHull.hpp"
#include "SplineOptimizer.hpp"

namespace
{
using Spline = SplineTrajectory::QuinticSplineND<2>;
using Optimizer = SplineTrajectory::SplineOptimizer<Spline>;
using Matrix = Spline::CoefficientMatrix;
using Vector = Eigen::Vector2d;
using Hull = SplineTrajectory::ConvexHullWorkspace<2>;

struct Circle
{
    Vector center;
    double radius = 0.0;
};

struct Box
{
    Vector lower;
    Vector upper;
};

struct LinearTimeCost
{
    double weight = 0.4;

    double operator()(const std::vector<double> &times,
                      Eigen::Ref<Eigen::VectorXd> gradient) const
    {
        gradient.array() += weight;
        double total = 0.0;
        for (double duration : times)
            total += duration;
        return weight * total;
    }
};

struct HullTrajectoryCost
{
    enum class Environment
    {
        ESDF,
        Corridor
    };

    Environment environment = Environment::ESDF;
    std::vector<Circle> circles;
    std::vector<Box> boxes;
    double clearance = 0.22;
    double esdf_control_buffer = 0.03;
    double corridor_control_buffer = 0.002;
    double dynamics_control_buffer = 0.005;
    double max_speed = 3.2;
    double max_acceleration = 5.0;
    double position_weight = 400.0;
    double speed_weight = 100.0;
    double acceleration_weight = 30.0;
    int position_subdivision_depth = 2;
    int dynamics_subdivision_depth = 1;
    Hull position_workspace;
    Hull velocity_workspace;
    Hull acceleration_workspace;
    Matrix position_gradient;
    Matrix velocity_gradient;
    Matrix acceleration_gradient;

    static double addNormUpperPenalty(const Matrix &controls,
                                      double upper,
                                      double weight,
                                      Matrix &gradients)
    {
        double cost = 0.0;
        for (Eigen::Index row = 0; row < controls.rows(); ++row)
        {
            const Vector value = controls.row(row).transpose();
            const double norm = value.norm();
            const double violation = norm - upper;
            if (violation > 0.0 && norm > 1e-12)
            {
                cost += weight * violation * violation;
                gradients.row(row) +=
                    (2.0 * weight * violation / norm * value).transpose();
            }
        }
        return cost;
    }

    double addPositionPenalty(const Hull &position,
                              Matrix &control_gradients) const
    {
        double cost = 0.0;
        for (int piece = 0; piece < position.numPieces(); ++piece)
        {
            const auto &info = position.pieceInfo(piece);
            const int offset = piece * position.controlsPerPiece();
            for (int local = 0; local < position.controlsPerPiece(); ++local)
            {
                const int row = offset + local;
                const Vector point = position.controls().row(row).transpose();

                if (environment == Environment::ESDF)
                {
                    for (const Circle &circle : circles)
                    {
                        const Vector delta = point - circle.center;
                        const double distance = delta.norm();
                        const double violation =
                            circle.radius + clearance +
                            esdf_control_buffer - distance;
                        if (violation > 0.0)
                        {
                            const Vector direction =
                                distance > 1e-10 ? delta / distance
                                                 : Vector(0.0, 1.0);
                            cost += position_weight * violation * violation;
                            control_gradients.row(row) +=
                                (-2.0 * position_weight * violation *
                                 direction).transpose();
                        }
                    }
                }
                else
                {
                    const Box &box = boxes.at(info.source_segment);
                    for (int axis = 0; axis < 2; ++axis)
                    {
                        const double lower =
                            box.lower(axis) + corridor_control_buffer;
                        const double upper =
                            box.upper(axis) - corridor_control_buffer;
                        if (point(axis) < lower)
                        {
                            const double violation =
                                lower - point(axis);
                            cost += position_weight * violation * violation;
                            control_gradients(row, axis) -=
                                2.0 * position_weight * violation;
                        }
                        else if (point(axis) > upper)
                        {
                            const double violation =
                                point(axis) - upper;
                            cost += position_weight * violation * violation;
                            control_gradients(row, axis) +=
                                2.0 * position_weight * violation;
                        }
                    }
                }
            }
        }
        return cost;
    }

    static void prepareWorkspace(Hull &workspace,
                                 Matrix &control_gradient,
                                 const Spline::Polynomial &polynomial,
                                 int derivative,
                                 int subdivision_depth)
    {
        const SplineTrajectory::HullOptions options{
            SplineTrajectory::ConvexHullBasis::Bezier, derivative, subdivision_depth};
        if (!workspace.matches(polynomial, options))
        {
            workspace.configure(polynomial, options);
            control_gradient.resize(
                workspace.controls().rows(), 2);
        }
        workspace.update(polynomial);
        control_gradient.setZero();
    }

    void prepare(const Spline::Polynomial &polynomial)
    {
        prepareWorkspace(position_workspace, position_gradient,
                         polynomial, 0, position_subdivision_depth);
        prepareWorkspace(velocity_workspace, velocity_gradient,
                         polynomial, 1, dynamics_subdivision_depth);
        prepareWorkspace(acceleration_workspace,
                         acceleration_gradient, polynomial, 2,
                         dynamics_subdivision_depth);

    }

    double operator()(const SplineTrajectory::ParameterView<Spline> &state,
                      SplineTrajectory::CoefficientGradient<Spline> &gradient)
    {
        position_workspace.update(state.polynomial);
        velocity_workspace.update(state.polynomial);
        acceleration_workspace.update(state.polynomial);
        position_gradient.setZero();
        velocity_gradient.setZero();
        acceleration_gradient.setZero();
        auto &coefficient_gradients = gradient.coefficients;
        auto &duration_gradients = gradient.durations;
        double cost =
            addPositionPenalty(position_workspace, position_gradient);
        cost += addNormUpperPenalty(
            velocity_workspace.controls(),
            max_speed - dynamics_control_buffer,
            speed_weight, velocity_gradient);
        cost += addNormUpperPenalty(
            acceleration_workspace.controls(),
            max_acceleration - dynamics_control_buffer,
            acceleration_weight, acceleration_gradient);

        position_workspace.backwardAdd(
            position_gradient, coefficient_gradients,
            duration_gradients);
        velocity_workspace.backwardAdd(
            velocity_gradient, coefficient_gradients,
            duration_gradients);
        acceleration_workspace.backwardAdd(
            acceleration_gradient, coefficient_gradients,
            duration_gradients);
        return cost;
    }
};

struct LBFGSResult
{
    int iterations = 0;
    const char *termination = "iteration_limit";
    double final_cost = 0.0;
    double gradient_inf_norm = 0.0;
    std::vector<double> history;
};

template <typename Evaluate>
LBFGSResult minimizeLBFGS(Eigen::VectorXd &x,
                          const Evaluate &evaluate,
                          int max_iterations = 300,
                          int memory = 10)
{
    LBFGSResult result;
    Eigen::VectorXd gradient(x.size());
    double cost = evaluate(x, gradient);
    result.history.push_back(cost);

    std::vector<Eigen::VectorXd> s_history;
    std::vector<Eigen::VectorXd> y_history;
    std::vector<double> inverse_curvature;

    for (int iteration = 0; iteration < max_iterations; ++iteration)
    {
        result.iterations = iteration;
        if (!std::isfinite(cost) || !gradient.allFinite())
            throw std::runtime_error("Non-finite objective encountered in L-BFGS.");
        if (gradient.lpNorm<Eigen::Infinity>() < 2e-5)
        {
            result.termination = "gradient_tolerance";
            break;
        }

        Eigen::VectorXd direction;
        if (s_history.empty())
        {
            direction = -gradient;
        }
        else
        {
            Eigen::VectorXd q = gradient;
            std::vector<double> alpha(s_history.size(), 0.0);
            for (int i = static_cast<int>(s_history.size()) - 1; i >= 0; --i)
            {
                alpha[i] = inverse_curvature[i] * s_history[i].dot(q);
                q.noalias() -= alpha[i] * y_history[i];
            }
            const Eigen::VectorXd &last_s = s_history.back();
            const Eigen::VectorXd &last_y = y_history.back();
            const double scale =
                last_s.dot(last_y) / last_y.squaredNorm();
            Eigen::VectorXd r = std::max(1e-6, scale) * q;
            for (std::size_t i = 0; i < s_history.size(); ++i)
            {
                const double beta =
                    inverse_curvature[i] * y_history[i].dot(r);
                r.noalias() += s_history[i] * (alpha[i] - beta);
            }
            direction = -r;
        }

        double directional_derivative = gradient.dot(direction);
        if (!(directional_derivative < -1e-12))
        {
            direction = -gradient;
            directional_derivative = -gradient.squaredNorm();
            s_history.clear();
            y_history.clear();
            inverse_curvature.clear();
        }

        const double max_step_norm = 3.0;
        if (direction.norm() > max_step_norm)
        {
            direction *= max_step_norm / direction.norm();
            directional_derivative = gradient.dot(direction);
        }

        double step = 1.0;
        Eigen::VectorXd trial_x;
        Eigen::VectorXd trial_gradient(x.size());
        double trial_cost = std::numeric_limits<double>::infinity();
        for (int line_search = 0; line_search < 30; ++line_search)
        {
            trial_x = x + step * direction;
            trial_cost = evaluate(trial_x, trial_gradient);
            if (std::isfinite(trial_cost) &&
                trial_cost <= cost +
                                  1e-4 * step * directional_derivative)
                break;
            step *= 0.5;
        }
        if (step < 1e-9 || !std::isfinite(trial_cost))
        {
            result.termination = "line_search_failed";
            break;
        }

        Eigen::VectorXd s = trial_x - x;
        Eigen::VectorXd y = trial_gradient - gradient;
        const double curvature = s.dot(y);
        if (curvature > 1e-10 * s.norm() * y.norm())
        {
            if (static_cast<int>(s_history.size()) == memory)
            {
                s_history.erase(s_history.begin());
                y_history.erase(y_history.begin());
                inverse_curvature.erase(inverse_curvature.begin());
            }
            s_history.push_back(s);
            y_history.push_back(y);
            inverse_curvature.push_back(1.0 / curvature);
        }

        x = std::move(trial_x);
        gradient = std::move(trial_gradient);
        cost = trial_cost;
        result.history.push_back(cost);
        result.iterations = iteration + 1;
    }

    result.final_cost = cost;
    result.gradient_inf_norm =
        gradient.lpNorm<Eigen::Infinity>();
    return result;
}

void writeTrajectory(const std::filesystem::path &path,
                     const Spline::Polynomial &spline)
{
    std::ofstream stream(path);
    stream << std::setprecision(17);
    stream << "t,x,y,speed,acceleration\n";
    constexpr int samples = 500;
    const auto &polynomial = spline;
    for (int i = 0; i <= samples; ++i)
    {
        const double alpha = static_cast<double>(i) / samples;
        const double time =
            spline.startTime() + alpha * spline.duration();
        const Vector position = polynomial.evaluate(time, 0);
        const Vector velocity = polynomial.evaluate(time, 1);
        const Vector acceleration = polynomial.evaluate(time, 2);
        stream << time << ',' << position.x() << ',' << position.y()
               << ',' << velocity.norm() << ','
               << acceleration.norm() << '\n';
    }
}

void writeTrajectoryPrefix(const std::filesystem::path &path,
                           const Spline::Polynomial &spline,
                           int source_segments)
{
    std::ofstream stream(path);
    stream << std::setprecision(17);
    stream << "t,source_segment,x,y\n";
    const auto &polynomial = spline;
    const auto &breakpoints = polynomial.breakpoints();
    const int segment_count =
        std::min(source_segments, polynomial.numSegments());
    constexpr int samples_per_segment = 160;
    for (int segment = 0; segment < segment_count; ++segment)
    {
        for (int sample = 0; sample <= samples_per_segment; ++sample)
        {
            if (segment > 0 && sample == 0)
                continue;
            const double alpha =
                static_cast<double>(sample) / samples_per_segment;
            const double time =
                breakpoints[segment] +
                alpha *
                    (breakpoints[segment + 1] -
                     breakpoints[segment]);
            const Vector position = polynomial.evaluate(time, 0);
            stream << time << ',' << segment << ','
                   << position.x() << ',' << position.y() << '\n';
        }
    }
}

void writeControls(const std::filesystem::path &path,
                   const Hull &representation,
                   int source_segments =
                       std::numeric_limits<int>::max())
{
    std::ofstream stream(path);
    stream << std::setprecision(17);
    stream << "piece,source_segment,local_control,x,y\n";
    for (int piece = 0; piece < representation.numPieces(); ++piece)
    {
        if (representation.pieceInfo(piece).source_segment >=
            source_segments)
            continue;
        const auto controls = representation.pieceControls(piece);
        for (int local = 0; local < representation.controlsPerPiece(); ++local)
        {
            stream << piece << ','
                   << representation.pieceInfo(piece).source_segment << ','
                   << local << ',' << controls(local, 0) << ','
                   << controls(local, 1) << '\n';
        }
    }
}

void writeHistory(const std::filesystem::path &path,
                  const std::vector<double> &history)
{
    std::ofstream stream(path);
    stream << std::setprecision(17);
    stream << "iteration,cost\n";
    for (std::size_t i = 0; i < history.size(); ++i)
        stream << i << ',' << history[i] << '\n';
}

void writeEnvironment(const std::filesystem::path &path,
                      const HullTrajectoryCost &cost)
{
    std::ofstream stream(path);
    stream << std::setprecision(17);
    if (cost.environment == HullTrajectoryCost::Environment::ESDF)
    {
        stream << "type,cx,cy,radius,clearance\n";
        for (const Circle &circle : cost.circles)
            stream << "circle," << circle.center.x() << ','
                   << circle.center.y() << ',' << circle.radius
                   << ',' << cost.clearance << '\n';
    }
    else
    {
        stream << "type,xmin,ymin,xmax,ymax\n";
        for (const Box &box : cost.boxes)
            stream << "box," << box.lower.x() << ','
                   << box.lower.y() << ',' << box.upper.x()
                   << ',' << box.upper.y() << '\n';
    }
}

Optimizer::Problem makeProblem(
    const std::vector<Vector, Eigen::aligned_allocator<Vector>> &points,
    double initial_duration)
{
    Optimizer::Problem problem;
    const int segments = static_cast<int>(points.size()) - 1;
    problem.durations.assign(segments, initial_duration);
    problem.waypoints.resize(segments + 1, 2);
    for (int i = 0; i <= segments; ++i)
        problem.waypoints.row(i) = points[i].transpose();

    SplineTrajectory::OptimizationMask mask;
    mask.time.assign(segments, static_cast<uint8_t>(1));
    mask.waypoints.assign(segments + 1, static_cast<uint8_t>(0));
    for (int i = 1; i < segments; ++i)
        mask.waypoints[i] = static_cast<uint8_t>(1);
    problem.mask = mask;
    return problem;
}

void runScenario(const std::string &name,
                 const Optimizer::Problem &problem,
                 HullTrajectoryCost hull_cost,
                 const std::filesystem::path &output_directory)
{
    Optimizer optimizer;
    SplineTrajectory::OptimizerOptions config;
    config.energy_weight = 0.003;
    config.integration_steps = 20;

    const auto prepare_status = optimizer.prepare(problem, config);
    if (!prepare_status)
        throw std::runtime_error(prepare_status.message);

    Eigen::VectorXd x = optimizer.initialGuess();
    const Eigen::VectorXd initial_x = x;
    LinearTimeCost time_cost;
    hull_cost.prepare(optimizer.polynomial());
    struct Objective
    {
        LinearTimeCost &duration;
        HullTrajectoryCost &coefficient;
    } objective{time_cost, hull_cost};

    Eigen::VectorXd scratch_gradient(x.size());
    const auto initial_result = optimizer.evaluate(initial_x, scratch_gradient, objective);
    if (!initial_result) throw std::runtime_error(initial_result.status.message);
    writeTrajectory(output_directory / (name + "_initial_trajectory.csv"),
                    optimizer.polynomial());

    const auto gradient_check =
        SplineTrajectory::checkGradients(optimizer, x, objective, 1e-6, 2e-3);
    std::cout << name << " gradient check: max_abs="
              << gradient_check.max_absolute_error << ", relative="
              << gradient_check.relative_error << '\n';
    if (!gradient_check)
        throw std::runtime_error(name + " optimizer gradient check failed.");

    // A C-style solver can receive this ordinary named object as its user data.
    struct CallbackData
    {
        Optimizer &optimizer;
        Objective &objective;
        static double evaluate(void *instance, const Eigen::VectorXd &decision, Eigen::VectorXd &gradient)
        {
            auto &data = *static_cast<CallbackData *>(instance);
            return data.optimizer.costAndGradient(decision, gradient, data.objective);
        }
    } callback{optimizer, objective};
    auto evaluate = [&](const Eigen::VectorXd &decision, Eigen::VectorXd &gradient)
    { return CallbackData::evaluate(&callback, decision, gradient); };
    const LBFGSResult result = minimizeLBFGS(x, evaluate);

    const auto synchronize_status =
        optimizer.updateAccepted(x);
    if (!synchronize_status)
        throw std::runtime_error(synchronize_status.message);
    const Spline::Polynomial &spline = optimizer.polynomial();
    const auto position_bezier =
        SplineTrajectory::toBezier(
            spline, 0,
            hull_cost.position_subdivision_depth);
    const auto position_minvo =
        SplineTrajectory::toMINVO(spline, 0);
    const auto velocity =
        SplineTrajectory::toBezier(
            spline, 1,
            hull_cost.dynamics_subdivision_depth);
    const auto acceleration =
        SplineTrajectory::toBezier(
            spline, 2,
            hull_cost.dynamics_subdivision_depth);

    writeTrajectory(output_directory / (name + "_trajectory.csv"), spline);
    writeControls(output_directory / (name + "_bezier.csv"),
                  position_bezier);
    writeControls(output_directory / (name + "_minvo.csv"),
                  position_minvo);
    writeHistory(output_directory / (name + "_history.csv"),
                 result.history);
    writeEnvironment(output_directory / (name + "_environment.csv"),
                     hull_cost);

    if (name == "esdf")
    {
        constexpr int displayed_segments = 3;
        writeTrajectoryPrefix(
            output_directory / "bezier_subdivision_trajectory.csv",
            spline, displayed_segments);
        for (int depth = 0; depth <= 4; ++depth)
        {
            const auto bezier_subdivision =
                SplineTrajectory::toBezier(
                    spline, 0, depth);
            writeControls(
                output_directory /
                    ("bezier_subdivision_depth_" +
                     std::to_string(depth) + ".csv"),
                bezier_subdivision, displayed_segments);

            const auto minvo_subdivision =
                SplineTrajectory::toMINVO(
                    spline, 0, depth);
            writeControls(
                output_directory /
                    ("minvo_subdivision_depth_" +
                     std::to_string(depth) + ".csv"),
                minvo_subdivision, displayed_segments);
        }
    }

    double maximum_speed_control = 0.0;
    for (Eigen::Index row = 0; row < velocity.controls().rows(); ++row)
        maximum_speed_control = std::max(
            maximum_speed_control, velocity.controls().row(row).norm());
    double maximum_acceleration_control = 0.0;
    for (Eigen::Index row = 0; row < acceleration.controls().rows(); ++row)
        maximum_acceleration_control = std::max(
            maximum_acceleration_control,
            acceleration.controls().row(row).norm());

    std::cout << name << ": termination=" << result.termination << ", iterations=" << result.iterations
              << ", cost=" << result.final_cost
              << ", duration=" << spline.duration()
              << ", max |v control|=" << maximum_speed_control
              << ", max |a control|=" << maximum_acceleration_control
              << ", |grad|inf=" << result.gradient_inf_norm << '\n';
}
}

int main(int argc, char **argv)
{
    try
    {
        const std::filesystem::path output_directory =
            argc > 1 ? std::filesystem::path(argv[1])
                     : std::filesystem::path("convex_hull_demo_output");
        std::filesystem::create_directories(output_directory);

        const std::vector<Vector, Eigen::aligned_allocator<Vector>> esdf_points{
            Vector(0.0, 0.0), Vector(2.0, 0.15), Vector(4.0, 0.20),
            Vector(6.0, -0.15), Vector(8.0, 0.10), Vector(10.0, 0.0)};
        HullTrajectoryCost esdf_cost;
        esdf_cost.environment = HullTrajectoryCost::Environment::ESDF;
        esdf_cost.circles = {
            {Vector(3.6, 0.0), 0.78},
            {Vector(6.4, 0.0), 0.82}};
        runScenario("esdf", makeProblem(esdf_points, 1.15),
                    esdf_cost, output_directory);

        const std::vector<Vector, Eigen::aligned_allocator<Vector>> corridor_points{
            Vector(0.0, -1.0), Vector(2.0, -1.0), Vector(4.0, 1.0),
            Vector(6.0, 1.0), Vector(8.0, -1.0), Vector(10.0, -1.0)};
        HullTrajectoryCost corridor_cost;
        corridor_cost.environment =
            HullTrajectoryCost::Environment::Corridor;
        corridor_cost.boxes = {
            {Vector(-0.3, -1.8), Vector(2.4, -0.2)},
            {Vector(1.6, -1.5), Vector(4.4, 1.55)},
            {Vector(3.6, 0.2), Vector(6.4, 1.8)},
            {Vector(5.6, -1.5), Vector(8.4, 1.55)},
            {Vector(7.6, -1.8), Vector(10.3, -0.2)}};
        runScenario("corridor", makeProblem(corridor_points, 1.25),
                    corridor_cost, output_directory);

        std::cout << "Wrote demo data to "
                  << std::filesystem::absolute(output_directory) << '\n';
        return 0;
    }
    catch (const std::exception &error)
    {
        std::cerr << "convex_hull_optimization_demo failed: "
                  << error.what() << '\n';
        return 1;
    }
}
