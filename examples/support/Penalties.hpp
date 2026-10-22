#pragma once
#include "OptimizationExample.hpp"
#include <array>

namespace spline_examples
{
/** @brief Add a squared hinge penalty for an upper bound on the squared norm.
 * @param[in,out] gradient Accumulated partial with respect to value.
 */
inline double normPenalty(const Vector &value, double limit, double weight, Vector &gradient)
{
    const double violation = value.squaredNorm() - limit * limit;
    if (violation <= 0.0) return 0.0;
    gradient += 4.0 * weight * violation * value;
    return weight * violation * violation;
}

struct Box
{
    Vector lower, upper;
    /** @brief Add the squared positive half-space residuals and their position partials. */
    double penalty(const Vector &point, double weight, Vector &gradient) const
    {
        double cost = 0.0;
        for (int axis = 0; axis < 2; ++axis)
        {
            const double residual = point(axis) < lower(axis) ? point(axis) - lower(axis)
                : point(axis) > upper(axis) ? point(axis) - upper(axis) : 0.0;
            cost += weight * residual * residual;
            gradient(axis) += 2.0 * weight * residual;
        }
        return cost;
    }
};
inline std::array<Box, 3> corridor()
{
    return {{{Vector(-0.3, -0.4), Vector(2.3, 0.6)},
             {Vector(1.7, -0.4), Vector(4.3, 1.4)},
             {Vector(3.7, 0.4), Vector(6.3, 1.4)}}};
}

struct DynamicsCost
{
    static constexpr int kDerivativeOrder = 2;
    double max_speed = 2.5;
    double max_acceleration = 4.0;
    double weight = 100.0;
    double operator()(const SplineTrajectory::IntegralPointInfo &,
                      const SplineTrajectory::SampleState<2> &state,
                      SplineTrajectory::SampleGradient<2> &gradient) const
    {
        return normPenalty(state.v, max_speed, weight, gradient.v)
             + normPenalty(state.a, max_acceleration, weight, gradient.a);
    }
};
}
