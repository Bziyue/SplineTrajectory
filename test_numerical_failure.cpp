#include "SplineOptimizer.hpp"

#include <cmath>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <string>

namespace
{
using Spline = SplineTrajectory::SepticSplineND<3>;
using Optimizer = SplineTrajectory::SplineOptimizer<
    3, Spline, SplineTrajectory::QuadInvTimeMap>;

void Require(bool condition, const char *message)
{
  if (!condition)
  {
    std::cerr << message << '\n';
    std::abort();
  }
}

Optimizer::ProblemDefinition MakeProblem()
{
  Optimizer::ProblemDefinition problem;
  problem.time_segments = {1.0, 1.0};
  problem.waypoints.resize(3, 3);
  problem.waypoints << 0, 0, 0, 1, 0, 0, 2, 0, 0;
  return problem;
}

/** @brief Check that all maintained spline families expose failed polynomial construction.
 * @tparam Curve Cubic, quintic or septic spline in three dimensions.
 * @note Finite input deliberately collapses time knots or overflows solved coefficients. */
template <class Curve>
void CheckSplineValidity()
{
  const auto problem = MakeProblem();
  Curve curve(problem.time_segments, problem.waypoints, 0.0);
  Require(curve.isInitialized(), "valid initial curve");

  curve.update(problem.time_segments, problem.waypoints, 2e16);
  Require(!curve.isInitialized() && !curve.getTrajectory().isInitialized(),
          "outer spline must reject collapsed polynomial time knots");

  auto points = problem.waypoints;
  points(0, 0) = -std::numeric_limits<double>::max();
  points(1, 0) = std::numeric_limits<double>::max();
  curve.update(problem.time_segments, points, 0.0);
  Require(!curve.isInitialized() && !curve.getTrajectory().isInitialized(),
          "outer spline must reject nonfinite solved coefficients");

  curve.update(problem.time_segments, problem.waypoints, 0.0);
  Require(curve.isInitialized() && curve.getTrajectory().evaluate(0.5).allFinite(),
          "curve must recover after invalid updates");
}

/** @brief Reject bad trials before costs/adjoints, preserve error status and allow a valid retry.
 * @note Runs unchanged with and without NDEBUG; checks do not rely on assert. */
void CheckEvaluationFailures()
{
  Optimizer optimizer;
  Optimizer::OptimizationContext context;
  const auto problem = MakeProblem();
  Require(bool(optimizer.prepareContext(problem, context)), "prepare valid context");
  const auto good = optimizer.generateInitialGuess(context);
  Eigen::VectorXd gradient;
  int time_calls = 0;
  int integral_calls = 0;
  bool nonfinite_cost = false;
  auto time_cost = [&](const std::vector<double> &times, Eigen::VectorXd &g) {
    ++time_calls;
    g.setOnes();
    return nonfinite_cost ? std::numeric_limits<double>::infinity() : times[0] + times[1];
  };
  auto integral_cost = [&](const auto &, const auto &, const auto &, const auto &,
                           const auto &, const auto &, auto &, auto &, auto &, auto &,
                           auto &, double &) {
    ++integral_calls;
    return 0.0;
  };
  const auto spec = Optimizer::makeEvaluateSpec(time_cost, integral_cost);
  const double baseline = optimizer.evaluatePrepared(context, good, gradient, spec);
  const Eigen::VectorXd baseline_gradient = gradient;
  Require(std::isfinite(baseline) && gradient.allFinite(), "valid baseline evaluation");
  Require(optimizer.checkGradients(context, good, spec).valid, "normal gradient check remains valid");

  // First trial reproduces the September 11 empty-coefficient read with the real duration map.
  for (int kind = 0; kind < 4; ++kind)
  {
    Eigen::VectorXd bad = good;
    if (kind == 0) bad(0) = 2e8;
    if (kind == 1) bad(0) = std::numeric_limits<double>::max();
    if (kind == 2)
      bad(context.prepared.layout.waypoints.front().offset) = std::numeric_limits<double>::max();
    if (kind == 3) bad(0) = std::numeric_limits<double>::quiet_NaN();
    const int previous_time_calls = time_calls;
    const int previous_integral_calls = integral_calls;
    gradient.setOnes(good.size());
    const auto result = optimizer.evaluate(context, bad, gradient, spec);
    Require(!result && std::isinf(result.cost) && result.cost > 0 && gradient.isZero(),
            "checked evaluation must return failure, positive infinity and zero gradient");
    Require(!optimizer.getWorkingSpline(context).isInitialized(), "failed build cannot expose stale curve");
    Require(context.isValid(), "trial failure must not poison prepared configuration");
    Require(time_calls == previous_time_calls && integral_calls == previous_integral_calls,
            "invalid build must not enter cost callbacks");
    if (kind == 0)
      Require(result.message.find("segment=1") != std::string::npos, "collapsed knot index retained");

    Optimizer::Status status;
    gradient.setOnes();
    const double rejected = optimizer.evaluatePrepared(context, bad, gradient, spec, &status);
    Require(!status && !status.message.empty() && std::isinf(rejected) && gradient.isZero(),
            "scalar adapter must preserve status and reject invalid trials in Release too");
    Require(!optimizer.synchronizeWorkingState(context, bad), "synchronization must reject same trial");
    Require(!optimizer.getWorkingSpline(context).getTrajectory().isInitialized(), "invalid synchronized curve");

    Require(optimizer.evaluatePrepared(context, good, gradient, spec) == baseline &&
                gradient.isApprox(baseline_gradient, 1e-12),
            "valid retry must reproduce original cost and gradient");
  }

  nonfinite_cost = true;
  const auto invalid_cost = optimizer.evaluate(context, good, gradient, spec);
  Require(!invalid_cost && std::isinf(invalid_cost.cost) && gradient.isZero(), "nonfinite callback is failure");
  nonfinite_cost = false;
  Require(std::isfinite(optimizer.evaluatePrepared(context, good, gradient, spec)), "recover after bad cost");

  auto invalid_seed = problem;
  invalid_seed.time_segments = {2e16, 1.0};
  Require(!optimizer.prepareContext(invalid_seed, context), "prepare must reject collapsed seed knots");
  Require(!context.isValid() && !optimizer.getWorkingSpline(context).isInitialized(), "invalid seed state");
  invalid_seed = problem;
  invalid_seed.waypoints(1, 0) = std::numeric_limits<double>::max();
  Require(!optimizer.prepareContext(invalid_seed, context), "prepare must reject nonfinite seed coefficients");
  Require(bool(optimizer.prepareContext(problem, context)), "reprepare after invalid seed");
  Require(bool(optimizer.synchronizeWorkingState(context, good)), "synchronize valid retry");
}
} // namespace

/** @brief Exercise structural numerical failure under both Debug and NDEBUG builds.
 * @return Zero on success; failed conditions abort independently of compiler assertions. */
int main()
{
  CheckSplineValidity<SplineTrajectory::CubicSplineND<3>>();
  CheckSplineValidity<SplineTrajectory::QuinticSplineND<3>>();
  CheckSplineValidity<Spline>();
  CheckEvaluationFailures();
  std::cout << "spline_numerical_failure_test: PASS\n";
}
