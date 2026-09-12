#include "SplineOptimizer.hpp"
#include "gcopter/minco.hpp"
#ifdef BENCH_LBFGS_HEADER
#include BENCH_LBFGS_HEADER
#endif
#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <numeric>

#ifndef BENCH_BACKEND
#define BENCH_BACKEND 0
#endif
#ifndef BENCH_DEGREE
#define BENCH_DEGREE 7
#endif
#ifndef BENCH_LABEL
#define BENCH_LABEL "structured"
#endif
#if BENCH_DEGREE == 7
using Spline = SplineTrajectory::SepticSplineND<3>;
using Minco = minco::MINCO_S4NU;
#else
using Spline = SplineTrajectory::QuinticSplineND<3>;
using Minco = minco::MINCO_S3NU;
#endif
using Optimizer = SplineTrajectory::SplineOptimizer<3, Spline>;
using Matrix = Spline::MatrixType;
using Clock = std::chrono::steady_clock;
constexpr int count = BENCH_DEGREE + 1;
constexpr int steps = 16;
constexpr double energy_weight = 0.01;

struct TimeCost
{
    double operator()(const std::vector<double> &times, Eigen::VectorXd &g) const
    {
        g.setConstant(0.2);
        return 0.2 * std::accumulate(times.begin(), times.end(), 0.0);
    }
};
struct IntegralCost
{
    static constexpr int kDerivativeOrder = 3;
    double operator()(const SplineTrajectory::IntegralPointInfo &,
                      const Eigen::Vector3d &p, const Eigen::Vector3d &v,
                      const Eigen::Vector3d &a, const Eigen::Vector3d &j, const Eigen::Vector3d &,
                      Eigen::Vector3d &gp, Eigen::Vector3d &gv, Eigen::Vector3d &ga,
                      Eigen::Vector3d &gj, Eigen::Vector3d &gs, double &gt) const
    {
        gp = p; gv = 0.3 * v; ga = 0.02 * a; gj = 0.002 * j; gs.setZero(); gt = 0;
        return 0.5*p.squaredNorm() + 0.15*v.squaredNorm() + 0.01*a.squaredNorm() + 0.001*j.squaredNorm();
    }
};

/** @brief Fixed physical objective shared by MINCO, direct spline, and optimizer paths. */
class Problem
{
public:
    const int pieces;
    Optimizer optimizer;
    Optimizer::OptimizationContext context;
    Optimizer::ProblemDefinition problem;
    TimeCost time_cost;
    IntegralCost integral_cost;
    std::vector<double> times;
    Matrix points;
    Spline spline;
    Minco minco;
#if BENCH_BACKEND == 1
    Eigen::MatrixX3d coeff_grad, energy_coeff_grad;
    Eigen::VectorXd energy_time_grad;
#else
    Matrix coeff_grad;
#endif
    Eigen::VectorXd time_grad, physical_time_grad;
    Eigen::Matrix3Xd inner, point_grad;
    Spline::Gradients spline_gradients, energy_gradients;

    explicit Problem(int n) : pieces(n)
    {
        times.assign(n, 1.2);
        points.resize(n + 1, 3);
        for (int i = 0; i <= n; ++i) points.row(i) << 0.3*i, std::sin(0.7*i), 0.2*std::cos(0.4*i);
        problem.time_segments = times; problem.waypoints = points;
#if BENCH_BACKEND == 0
        Optimizer::OptimizerConfig config; config.integral_num_steps = steps; config.rho_energy = energy_weight;
#ifdef BENCH_LEGACY_CONTEXT
        optimizer.setConfig(config);
        auto status = optimizer.prepareContext(problem, context);
#else
        auto status = optimizer.prepareContext(problem, context, config);
#endif
        if (!status) throw std::runtime_error("benchmark preparation failed");
#elif BENCH_BACKEND == 1
        Eigen::Matrix<double, 3, (BENCH_DEGREE + 1)/2> head, tail;
        head.setZero(); tail.setZero(); head.col(0) = points.row(0).transpose(); tail.col(0) = points.row(n).transpose();
        minco.setConditions(head, tail, n);
        energy_coeff_grad.resize(n * count, 3); energy_time_grad.resize(n);
#else
        spline.update(times, points, 0.0, problem.bc);
#endif
        coeff_grad.resize(n * count, 3); time_grad.resize(n); physical_time_grad.resize(n);
        inner.resize(3, n - 1); point_grad.resize(3, n - 1);
    }

    Eigen::VectorXd initialGuess() const
    {
        Eigen::VectorXd x(pieces + 3*(pieces-1));
        for (int i=0;i<pieces;++i) x(i)=SplineTrajectory::QuadInvTimeMap{}.toTau(times[i]);
        for (int i=1;i<pieces;++i) x.segment<3>(pieces+3*(i-1))=points.row(i).transpose();
        return x;
    }

    double evaluate(const Eigen::VectorXd &x, Eigen::VectorXd &gradient)
    {
#if BENCH_BACKEND == 0
        const auto spec = Optimizer::makeEvaluateSpec(time_cost, integral_cost);
        return optimizer.evaluate(context, x, gradient, spec).cost;
#else
        SplineTrajectory::QuadInvTimeMap time_map;
        for (int i = 0; i < pieces; ++i) times[i] = time_map.toTime(x(i));
        for (int i = 1; i < pieces; ++i)
        {
            points.row(i) = x.segment<3>(pieces + 3*(i-1)).transpose();
            inner.col(i-1) = points.row(i).transpose();
        }
#if BENCH_BACKEND == 1
        minco.setParameters(inner, Eigen::Map<const Eigen::VectorXd>(times.data(), pieces));
        const auto &coefficients = minco.getCoeffs();
#else
        spline.update(times, points, 0.0, problem.bc);
        const auto &coefficients = spline.getTrajectory().getCoefficients();
#endif
        coeff_grad.setZero(); time_grad.setConstant(0.2);
        double cost = 0.2 * std::accumulate(times.begin(), times.end(), 0.0);
        for (int i = 0; i < pieces; ++i)
        {
            const Eigen::Matrix<double, count, 3> block = coefficients.template middleRows<count>(i * count);
            for (int k = 0; k <= steps; ++k)
            {
                const double alpha = double(k)/steps;
                const double weight = (k == 0 || k == steps) ? 0.5 : 1.0;
                const double common = weight * times[i]/steps;
                Eigen::Matrix<double, 1, count> bp,bv,ba,bj,bs,bc;
                Spline::computeBasisFunctions(alpha*times[i],bp,bv,ba,bj,bs,bc);
                const Eigen::Vector3d p = (bp*block).transpose(), v=(bv*block).transpose();
                const Eigen::Vector3d a=(ba*block).transpose(), j=(bj*block).transpose(), s=(bs*block).transpose();
                Eigen::Vector3d gp,gv,ga,gj,gs; double gt;
                const double local = integral_cost({},p,v,a,j,s,gp,gv,ga,gj,gs,gt);
                cost += common*local;
                coeff_grad.template middleRows<count>(i*count).noalias() +=
                    (bp.transpose()*gp.transpose()+bv.transpose()*gv.transpose()+ba.transpose()*ga.transpose()+bj.transpose()*gj.transpose())*common;
                time_grad(i) += local*weight/steps + (gp.dot(v)+gv.dot(a)+ga.dot(j)+gj.dot(s))*alpha*common;
            }
        }
#if BENCH_BACKEND == 1
        double energy; minco.getEnergy(energy);
        auto &gc = energy_coeff_grad; auto &gt = energy_time_grad;
        minco.getEnergyPartialGradByCoeffs(gc); minco.getEnergyPartialGradByTimes(gt);
        coeff_grad += energy_weight * gc; time_grad += energy_weight * gt; cost += energy_weight * energy;
        minco.propogateGrad(coeff_grad, time_grad, point_grad, physical_time_grad);
#else
        spline.propagateGrad(coeff_grad, time_grad, spline_gradients);
        spline.getEnergyGrad(energy_gradients); cost += energy_weight*spline.getEnergy();
        physical_time_grad = spline_gradients.times + energy_weight*energy_gradients.times;
        point_grad = (spline_gradients.inner_points + energy_weight*energy_gradients.inner_points).transpose();
#endif
        gradient.resize(x.size());
        for(int i=0;i<pieces;++i) gradient(i)=time_map.backward(x(i),times[i],physical_time_grad(i));
        for(int i=1;i<pieces;++i) gradient.segment<3>(pieces+3*(i-1))=point_grad.col(i-1);
        return cost;
#endif
    }
};

int main(int argc, char **argv)
{
    const int repeats = argc > 1 ? std::atoi(argv[1]) : 2000;
#ifdef BENCH_LBFGS_HEADER
    if (argc > 2)
    {
        std::cout << "backend,degree,pieces,trial,solve_us,evaluations,iterations,line_search_evaluations,code,cost,gradient_inf\n" << std::setprecision(17);
        for (int n : {2,8,32})
            for (int trial=0; trial<repeats; ++trial)
            {
                Problem problem(n); auto x=problem.initialGuess();
                struct Callback { Problem *problem; int evaluations=0; int iterations=0; int line_search_evaluations=0; } callback{&problem};
                auto evaluate=[](void *data,const Eigen::VectorXd &x,Eigen::VectorXd &g) {
                    auto &call=*static_cast<Callback *>(data); ++call.evaluations;
                    return call.problem->evaluate(x,g);
                };
                auto progress=[](void *data,const Eigen::VectorXd &,const Eigen::VectorXd &,
                                 double,double,int iteration,int line_evaluations) {
                    auto &call=*static_cast<Callback *>(data);
                    call.iterations=iteration; call.line_search_evaluations+=line_evaluations;
                    return 0;
                };
                lbfgs::lbfgs_parameter_t parameters;
                parameters.mem_size=16; parameters.past=3; parameters.delta=1e-6;
                parameters.g_epsilon=1e-8; parameters.max_iterations=500;
                double cost; const auto start=Clock::now();
                const int code=lbfgs::lbfgs_optimize(x,cost,evaluate,nullptr,progress,&callback,parameters);
                const double us=std::chrono::duration<double,std::micro>(Clock::now()-start).count();
                Eigen::VectorXd gradient; problem.evaluate(x,gradient);
                std::cout<<BENCH_LABEL<<','<<BENCH_DEGREE<<','<<n<<','<<trial<<','<<us<<','
                         <<callback.evaluations<<','<<callback.iterations<<','<<callback.line_search_evaluations<<','<<code<<','<<cost<<','<<gradient.lpNorm<Eigen::Infinity>()<<'\n';
            }
        return 0;
    }
#endif
    std::cout << "backend,degree,pieces,prepare_us,first_us,p50_us,p95_us,cost,gradient_checksum\n" << std::setprecision(17);
    for (int n : {2,8,32,128})
    {
        const auto begin=Clock::now(); Problem problem(n);
        auto x=problem.initialGuess(); Eigen::VectorXd gradient;
        const auto prepared=Clock::now(); const double cost=problem.evaluate(x,gradient); const auto first=Clock::now();
        const Eigen::VectorXd expected=gradient;
        std::vector<double> timings;
        for(int round=0;round<15;++round)
        {
            const auto start=Clock::now();
            for(int j=0;j<repeats;++j) problem.evaluate(x,gradient);
            timings.push_back(std::chrono::duration<double,std::micro>(Clock::now()-start).count()/repeats);
        }
        if(!std::isfinite(cost)||!gradient.allFinite()||!gradient.isApprox(expected,1e-13)) return 1;
        std::sort(timings.begin(),timings.end());
        std::cout<<BENCH_LABEL<<','<<BENCH_DEGREE<<','<<n<<','
          <<std::chrono::duration<double,std::micro>(prepared-begin).count()<<','
          <<std::chrono::duration<double,std::micro>(first-prepared).count()<<','
          <<timings[7]<<','<<timings[14]<<','<<cost<<','<<gradient.sum()<<'\n';
    }
}
