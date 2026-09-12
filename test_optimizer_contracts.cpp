#define EIGEN_RUNTIME_NO_MALLOC
#include "SplineOptimizer.hpp"
#include "SplineConvexHull.hpp"
#include <iostream>
#include <stdexcept>

void require(bool condition, const char *message)
{
    if (!condition) throw std::runtime_error(message);
}

struct TimeCost
{
    double operator()(const std::vector<double> &times, Eigen::VectorXd &gradient) const
    {
        gradient.setConstant(0.2);
        double cost = 0;
        for (double t : times) cost += 0.2 * t;
        return cost;
    }
};

template <int ORDER>
struct StateCost
{
    static constexpr int kDerivativeOrder = ORDER;
    double operator()(const SplineTrajectory::IntegralPointInfo &point,
                      const SplineTrajectory::SampleState<3> &state,
                      SplineTrajectory::SampleGradient<3> &gradient) const
    {
        double cost = 0.01 * point.global_time * point.global_time;
        gradient.time = 0.02 * point.global_time;
        gradient.p = state.p;
        cost += 0.5 * state.p.squaredNorm();
        if constexpr (ORDER >= 1) { gradient.v = 0.3 * state.v; cost += 0.15 * state.v.squaredNorm(); }
        if constexpr (ORDER >= 2) { gradient.a = 0.02 * state.a; cost += 0.01 * state.a.squaredNorm(); }
        if constexpr (ORDER >= 3) { gradient.j = 0.002 * state.j; cost += 0.001 * state.j.squaredNorm(); }
        if constexpr (ORDER >= 4) { gradient.s = 0.0002 * state.s; cost += 0.0001 * state.s.squaredNorm(); }
        return cost;
    }
};

/** @brief Non-default-constructible coefficient cost, exercising concrete callable binding. */
struct CoefficientCost
{
    explicit CoefficientCost(double weight) : weight(weight) {}
    double weight;
    template <class Spline, class Matrix>
    double operator()(Spline &spline, const std::vector<double> &times, double,
                      Matrix &gc, Eigen::VectorXd &gt, double &) const
    {
        gc += weight * spline.getTrajectory().getCoefficients();
        double cost = 0.5 * weight * spline.getTrajectory().getCoefficients().squaredNorm();
        for (int i = 0; i < gt.size(); ++i) { gt(i) += 0.02 * times[i]; cost += 0.01 * times[i] * times[i]; }
        return cost;
    }
};

template <class Spline, int ORDER>
void checkGradients()
{
    using Optimizer = SplineTrajectory::SplineOptimizer<3, Spline>;
    Optimizer optimizer;
    typename Optimizer::ProblemDefinition problem;
    problem.time_segments = {1.4, 1.7};
    problem.start_time = 0.7;
    problem.waypoints.resize(3, 3);
    problem.waypoints << 0,0,0, 1,0.3,-0.2, 2,-0.1,0.4;
    problem.bc.start_velocity << 0.1,0.2,-0.1;
    auto mask = Optimizer::makeFullOptimizationMask(2);
    mask.start.v = mask.end.v = true;
    if constexpr (Spline::ORDER >= 5) mask.start.a = mask.end.a = true;
    if constexpr (Spline::ORDER >= 7) mask.start.j = mask.end.j = true;
    problem.mask = mask;
    typename Optimizer::OptimizerConfig config;
    config.integral_num_steps = 5;
    config.rho_energy = 0.01;
    typename Optimizer::OptimizationContext context;
    require(bool(optimizer.prepareContext(problem, context, config)), "prepare typed cost");
    TimeCost time;
    StateCost<ORDER> state;
    CoefficientCost coefficient(1e-4);
    auto spec = Optimizer::makeEvaluateSpec(time, state).withCoefficientCost(coefficient)
        .withExecutor(SplineTrajectory::SerialExecutor{});
    auto x = optimizer.generateInitialGuess(context);
    auto check = optimizer.checkGradients(context, x, spec, 1e-5, 2e-4);
    require(check.valid && check.max_abs_error < 2e-4, "structured coefficient/state gradient");
    Eigen::VectorXd gradient;
    require(bool(optimizer.evaluate(context, x, gradient, spec)), "typed cost evaluation");
    Eigen::internal::set_is_malloc_allowed(false);
    for (int i = 0; i < 10; ++i)
        require(bool(optimizer.evaluate(context, x, gradient, spec)), "reused context evaluation");
    Eigen::internal::set_is_malloc_allowed(true);
}

void checkEnergyOnly()
{
    using Opt = SplineTrajectory::SplineOptimizer<3>;
    Opt opt; Opt::ProblemDefinition problem; Opt::OptimizationContext context;
    problem.time_segments = {1.2,1.5};
    problem.waypoints.resize(3,3); problem.waypoints << 0,0,0, 1,0.1,0.2, 2,0,0;
    Opt::OptimizerConfig config; config.rho_energy = 0.3;
    require(bool(opt.prepareContext(problem,context,config)), "prepare energy-only");
    TimeCost time; SplineTrajectory::VoidIntegralCost integral;
    auto spec=Opt::makeEvaluateSpec(time,integral);
    const auto x=opt.generateInitialGuess(context);
    const auto result=opt.checkGradients(context,x,spec,1e-5,1e-4);
    require(result.valid && result.max_abs_error < 1e-4, "energy-only gradient");
    opt.setRecordIntegralSamples(true,context);
    Eigen::VectorXd gradient;
    require(bool(opt.evaluate(context,x,gradient,spec)), "recording with absent integrand");
    require(!opt.getRecordedIntegralSamples(context).empty(), "explicit recording remains available");
}

struct ShiftMap
{
    int getDimension() const { return 1; }
    template<class Matrix, class Boundary>
    Eigen::VectorXd getInitialValue(const std::vector<double> &, const Matrix &, double, const Boundary &) const
    { return Eigen::VectorXd::Zero(1); }
    template<class Matrix, class Boundary>
    void apply(const Eigen::VectorXd &x, std::vector<double> &, Matrix &, double &start, Boundary &) const
    { start += x(0); }
    template<class Spline, class Matrix, class Boundary, class Gradients>
    double backward(const Eigen::VectorXd &, const Spline &, const std::vector<double> &,
                    const Matrix &, double, const Boundary &, Gradients &, double start_gradient,
                    Eigen::VectorXd &gradient) const
    { gradient.setConstant(1, start_gradient); return 0.0; }
};

struct ClockCoefficientCost
{
    template<class Spline, class Matrix>
    double operator()(Spline &, const std::vector<double> &, double start, Matrix &,
                      Eigen::VectorXd &, double &gradient) const
    { gradient += 3.0; return 3.0 * start; }
};

struct ClockSampleCost
{
    template<class Samples, class Matrix>
    double operator()(const Samples &samples, Matrix &, Eigen::VectorXd &gradient) const
    {
        double cost = 0.0;
        for (int i = 0; i < int(samples.size()); ++i)
        {
            cost += 0.005 * samples[i].point.global_time * samples[i].point.global_time;
            gradient(i) = 0.01 * samples[i].point.global_time;
        }
        return cost;
    }
};

void checkExplicitStartTime()
{
    using namespace SplineTrajectory;
    using Opt = SplineOptimizer<3, QuinticSplineND<3>, QuadInvTimeMap, IdentitySpatialMap<3>, ShiftMap>;
    Opt opt; Opt::ProblemDefinition problem; Opt::OptimizationContext context;
    problem.time_segments = {1.2,1.5}; problem.start_time = 0.7;
    problem.waypoints.resize(3,3); problem.waypoints << 0,0,0, 1,0.1,0.2, 2,0,0;
    Opt::OptimizerConfig config; config.integral_num_steps = 5;
    require(bool(opt.prepareContext(problem,context,config)), "prepare start-time gradient");
    TimeCost time; StateCost<2> integral; ClockCoefficientCost coefficient; ClockSampleCost sample;
    const auto spec = Opt::makeEvaluateSpec(time,integral).withSampleCost(sample).withCoefficientCost(coefficient);
    const auto x = opt.generateInitialGuess(context);
    const auto result = opt.checkGradients(context,x,spec,1e-5,1e-4);
    require(result.valid && result.max_abs_error < 1e-4, "combined explicit start-time gradient");
}

int main()
{
    checkGradients<SplineTrajectory::CubicSplineND<3>, 0>();
    checkGradients<SplineTrajectory::QuinticSplineND<3>, 2>();
    checkGradients<SplineTrajectory::SepticSplineND<3>, 3>();
    checkGradients<SplineTrajectory::SepticSplineND<3>, 4>();
    checkEnergyOnly();
    checkExplicitStartTime();
    std::cout << "optimizer contracts passed\n";
}
