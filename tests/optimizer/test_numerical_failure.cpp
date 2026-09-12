#include "TestSupport.hpp"
#include "SplineOptimizer.hpp"

#include <cmath>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <string>

namespace
{
using Spline = SplineTrajectory::SepticSplineND<3>;
using Optimizer = SplineTrajectory::SplineOptimizer<Spline>;

using spline_test::require;

Optimizer::Problem MakeProblem()
{
  Optimizer::Problem problem;
  problem.durations = {1.0, 1.0};
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
  Curve curve(problem.durations, problem.waypoints, 0.0);
  require(curve.isValid(), "valid initial curve");

  curve.update(problem.durations, problem.waypoints, 2e16);
  require(!curve.isValid() && !curve.polynomial().isValid(),
          "outer spline must reject collapsed polynomial time knots");

  auto points = problem.waypoints;
  points(0, 0) = -std::numeric_limits<double>::max();
  points(1, 0) = std::numeric_limits<double>::max();
  curve.update(problem.durations, points, 0.0);
  require(!curve.isValid() && !curve.polynomial().isValid(),
          "outer spline must reject nonfinite solved coefficients");

  curve.update(problem.durations, problem.waypoints, 0.0);
  require(curve.isValid() && curve.polynomial().evaluate(0.5).allFinite(),
          "curve must recover after invalid updates");
}

/** @brief Reject bad trials before costs/adjoints, preserve error status and allow a valid retry.
 * @note Runs unchanged with and without NDEBUG; checks do not rely on assert. */
void CheckEvaluationFailures()
{
  Optimizer optimizer;
  const auto problem = MakeProblem();
  require(bool(optimizer.prepare(problem)), "prepare valid context");
  const auto good = optimizer.initialGuess();
  Eigen::VectorXd gradient(good.size());
  int time_calls = 0;
  int integral_calls = 0;
  bool nonfinite_cost = false;
  auto time_cost = [&](const std::vector<double> &times, Eigen::Ref<Eigen::VectorXd> g) {
    ++time_calls;
    g.setOnes();
    return nonfinite_cost ? std::numeric_limits<double>::infinity() : times[0] + times[1];
  };
  auto integral_cost = [&](const auto &, const auto &, auto &) {
    ++integral_calls;
    return 0.0;
  };
  struct Objective
  {
    decltype(time_cost) &duration;
    decltype(integral_cost) &integral;
  } objective{time_cost, integral_cost};
  const double baseline = optimizer.costAndGradient(good, gradient, objective);
  const Eigen::VectorXd baseline_gradient = gradient;
  require(std::isfinite(baseline) && gradient.allFinite(), "valid baseline evaluation");
  require(bool(SplineTrajectory::checkGradients(optimizer, good, objective)), "normal gradient check remains valid");

  // First trial reproduces the September 11 empty-coefficient read with the real duration map.
  for (int kind = 0; kind < 4; ++kind)
  {
    Eigen::VectorXd bad = good;
    if (kind == 0) bad(0) = 2e8;
    if (kind == 1) bad(0) = std::numeric_limits<double>::max();
    if (kind == 2)
      bad(optimizer.layout().waypoints.front().offset) = std::numeric_limits<double>::max();
    if (kind == 3) bad(0) = std::numeric_limits<double>::quiet_NaN();
    const int previous_time_calls = time_calls;
    const int previous_integral_calls = integral_calls;
    gradient.setOnes(good.size());
    const auto result = optimizer.evaluate(bad, gradient, objective);
    require(!result && std::isinf(result.cost) && result.cost > 0 && gradient.isZero(),
            "checked evaluation must return failure, positive infinity and zero gradient");
    require(!optimizer.polynomial().isValid(), "failed build cannot expose stale curve");
    require(optimizer.isPrepared(), "trial failure must not poison prepared configuration");
    require(time_calls == previous_time_calls && integral_calls == previous_integral_calls,
            "invalid build must not enter cost callbacks");
    if (kind == 0)
      require(result.status.index == 1, "collapsed knot index retained");

    Optimizer::Status status;
    gradient.setOnes();
    const double rejected = optimizer.costAndGradient(bad, gradient, objective);
    status = optimizer.lastStatus();
    require(!status && status.message[0] != 0 && std::isinf(rejected) && gradient.isZero(),
            "scalar adapter must preserve status and reject invalid trials in Release too");
    require(!optimizer.updateAccepted(bad), "synchronization must reject same trial");
    require(!optimizer.polynomial().isValid(), "invalid synchronized curve");

    require(optimizer.costAndGradient(good, gradient, objective) == baseline &&
                gradient.isApprox(baseline_gradient, 1e-12),
            "valid retry must reproduce original cost and gradient");
  }

  nonfinite_cost = true;
  const auto invalid_cost = optimizer.evaluate(good, gradient, objective);
  require(!invalid_cost && std::isinf(invalid_cost.cost) && gradient.isZero(), "nonfinite callback is failure");
  nonfinite_cost = false;
  require(std::isfinite(optimizer.costAndGradient(good, gradient, objective)), "recover after bad cost");

  auto invalid_seed = problem;
  invalid_seed.durations = {2e16, 1.0};
  require(!optimizer.prepare(invalid_seed), "prepare must reject collapsed seed knots");
  require(!optimizer.isPrepared() && !optimizer.polynomial().isValid(), "invalid seed state");
  invalid_seed = problem;
  invalid_seed.waypoints(1, 0) = std::numeric_limits<double>::max();
  require(!optimizer.prepare(invalid_seed), "prepare must reject nonfinite seed coefficients");
  require(bool(optimizer.prepare(problem)), "reprepare after invalid seed");
  require(bool(optimizer.updateAccepted(good)), "synchronize valid retry");
}
} // namespace

/** @brief Exercise structural numerical failure and recovery.
 * @return Zero on success; one on a failed condition. */
int main()
{
    return spline_test::run("test_numerical_failure", []
    {
        CheckSplineValidity<SplineTrajectory::CubicSplineND<3>>();
        CheckSplineValidity<SplineTrajectory::QuinticSplineND<3>>();
        CheckSplineValidity<Spline>();
        CheckEvaluationFailures();
    });
}
