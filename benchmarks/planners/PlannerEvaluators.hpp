#include <stdexcept>
#pragma once
#include "PlannerCosts.hpp"
#if PLANNER_GCOPTER
#include <frozen_gcopter/minco.hpp>
#include <frozen_gcopter/lbfgs.hpp>
#else
#include <traj_opt/minco.h>
#include <utils/optimization/lbfgs.h>
#endif
#include <type_traits>

namespace planner_benchmark {
#if PLANNER_GCOPTER
using NativeMinco = minco::MINCO_S3NU;
namespace Solver = lbfgs;
#else
using NativeMinco = traj_opt::MINCO_S4NU;
using Solver = math_utils::lbfgs;
#endif
inline constexpr int coefficients = Spline::kCoefficientCount;

inline Eigen::VectorXd initialGuess(const ProblemData &p,const Geometry &geometry) {
    Eigen::VectorXd x(p.dimension());
    const Eigen::Map<const Eigen::VectorXd> durations(p.durations.data(),p.segments);
    Eigen::VectorXd tau(p.segments);
#if PLANNER_GCOPTER
    Native::backwardT(durations,tau);
#else
    Native::backwardMapTToTau(durations,tau);
#endif
    x.head(p.segments)=tau;
    SpatialMap map{&p,&geometry};
    int offset=p.segments;
    for(int i=1;i<p.segments;++i) {
        const int dimension=map.dimension(i);
        x.segment(offset,dimension)=map.toUnconstrained(p.waypoints.row(i).transpose(),i).value();
        offset+=dimension;
    }
    return x;
}

/** @brief The original coefficient quadrature and physical adjoint, with MINCO or a direct spline kernel.
 * @note The native mapping, quadrature and L-BFGS remain project-specific; only the selected interpolation kernel changes. */
template<bool UseSpline>
class DirectEvaluator {
    using Kernel = std::conditional_t<UseSpline,Spline,NativeMinco>;
    using CoefficientMatrix = std::conditional_t<UseSpline,Matrix,Eigen::MatrixX3d>;
    const ProblemData &problem_;
    const Geometry &geometry_;
    Kernel kernel_;
    flatness::FlatnessMap flatmap_;
    Eigen::VectorXd tau_,xi_,times_,partial_times_,gradient_times_,penalty_log_,attractor_distances_;
    Eigen::Matrix3Xd points_,gradient_points_,attractors_;
    CoefficientMatrix partial_coefficients_;
    Matrix all_points_;
    std::vector<double> spline_times_;
    typename Spline::Gradients spline_gradients_;
    SplineTrajectory::BoundaryConditions<3> boundary_;
    void build(const Eigen::VectorXd &x) {
        tau_=x.head(problem_.segments); xi_=x.tail(problem_.spatialDimension());
#if PLANNER_GCOPTER
        Native::forwardT(tau_,times_);
#else
        Native::forwardMapTauToT(tau_,times_);
#endif
        if(problem_.usesPolytopeMap()) Native::forwardP(xi_,geometry_.vertex_indices,geometry_.vertices,points_);
        else points_=Eigen::Map<const Eigen::Matrix3Xd>(xi_.data(),3,problem_.segments-1);
        if constexpr (UseSpline) {
            for(int i=0;i<problem_.segments;++i) spline_times_[i]=times_(i);
            all_points_.middleRows(1,problem_.segments-1)=points_.transpose();
            if (const auto status = kernel_.update(spline_times_,all_points_,0.0,boundary_); !status) { throw std::runtime_error(status.message); }
        } else kernel_.setParameters(points_,times_);
    }
    const auto &coefficientsView() const {
        if constexpr(UseSpline) return kernel_.polynomial().coefficients();
        else return kernel_.getCoeffs();
    }
public:
    static constexpr const char *label=UseSpline?"spline_direct":"minco";
    DirectEvaluator(const ProblemData &p,const Geometry &geometry):problem_(p),geometry_(geometry) {
        const int n=p.segments;
        tau_.resize(n); xi_.resize(p.spatialDimension()); times_.resize(n); partial_times_.resize(n); gradient_times_.resize(n);
        points_.resize(3,n-1); gradient_points_.resize(3,n-1); partial_coefficients_.resize(n*coefficients,3);
        penalty_log_.setZero(8); attractor_distances_.setZero(n-1); attractors_=p.waypoints.middleRows(1,n-1).transpose();
        flatmap_.reset(p.physical[0],p.physical[1],p.physical[2],p.physical[3],p.physical[4],p.physical[5]);
        if constexpr(UseSpline) {
            all_points_=p.waypoints; spline_times_=p.durations; boundary_=splineProblem(p).boundary;
            spline_gradients_.resetTopology(n);
            if (const auto status = kernel_.update(spline_times_,all_points_,0.0,boundary_); !status) { throw std::runtime_error(status.message); }
        } else {
            Eigen::Matrix<double,3,coefficients/2> head,tail;
            head.col(0)=p.waypoints.row(0).transpose(); tail.col(0)=p.waypoints.row(n).transpose();
            head.rightCols(coefficients/2-1)=p.start_derivatives.leftCols(coefficients/2-1);
            tail.rightCols(coefficients/2-1)=p.end_derivatives.leftCols(coefficients/2-1);
            kernel_.setConditions(head,tail,n);
        }
    }
    double evaluate(const Eigen::VectorXd &x,Eigen::VectorXd &gradient) {
        build(x);
        double energy;
        if constexpr(UseSpline) {
            if(!kernel_.isValid()) { gradient.setZero(); return std::numeric_limits<double>::infinity(); }
            energy=kernel_.energy();
            kernel_.energyCoefficientPartials(partial_coefficients_); kernel_.energyDurationPartials(partial_times_);
        } else {
            kernel_.getEnergy(energy);
            kernel_.getEnergyPartialGradByCoeffs(partial_coefficients_); kernel_.getEnergyPartialGradByTimes(partial_times_);
        }
        double cost=problem_.energy*energy;
        partial_coefficients_*=problem_.energy; partial_times_*=problem_.energy;
#if PLANNER_GCOPTER
        Native::attachPenaltyFunctional(times_,coefficientsView(),geometry_.plane_indices,geometry_.planes,
            problem_.smooth,problem_.integration,problem_.gcopter_bounds,problem_.gcopter_weights,flatmap_,
            cost,partial_times_,partial_coefficients_,PositionPenalty{problem_});
#else
        Native::constraintsFunctional(times_,coefficientsView(),geometry_.plane_indices,geometry_.planes,
            attractors_,attractor_distances_,problem_.smooth,problem_.integration,problem_.super_bounds,problem_.super_weights,
            flatmap_,cost,partial_times_,partial_coefficients_,penalty_log_,PositionPenalty{problem_});
#endif
        if constexpr(UseSpline) {
            kernel_.backward(partial_coefficients_,partial_times_,spline_gradients_);
            gradient_points_=spline_gradients_.inner_points.transpose(); gradient_times_=spline_gradients_.durations;
        } else kernel_.propogateGrad(partial_coefficients_,partial_times_,gradient_points_,gradient_times_);
        cost+=problem_.time_weight*times_.sum(); gradient_times_.array()+=problem_.time_weight;
        Eigen::Map<Eigen::VectorXd> gradient_tau(gradient.data(),problem_.segments);
        Eigen::Map<Eigen::VectorXd> gradient_xi(gradient.data()+problem_.segments,problem_.spatialDimension());
#if PLANNER_GCOPTER
        Native::backwardGradT(tau_,gradient_times_,gradient_tau);
#else
        Native::propagateGradientTToTau(tau_,gradient_times_,gradient_tau);
#endif
        if(problem_.usesPolytopeMap()) {
            Native::backwardGradP(xi_,geometry_.vertex_indices,geometry_.vertices,gradient_points_,gradient_xi);
            Native::normRetrictionLayer(xi_,geometry_.vertex_indices,geometry_.vertices,cost,gradient_xi);
        } else gradient_xi=Eigen::Map<const Eigen::VectorXd>(gradient_points_.data(),gradient_points_.size());
        return cost;
    }
    void accept(const Eigen::VectorXd &x) { build(x); }
    typename Spline::Polynomial snapshot() const {
        if constexpr(UseSpline) return kernel_.copyPolynomial();
        else {
            std::vector<double> knots(problem_.segments+1,0.0);
            for(int i=0;i<problem_.segments;++i) knots[i+1]=knots[i]+times_(i);
            return Spline::Polynomial::create(knots,coefficientsView(),coefficients);
        }
    }
};

/** @brief Named structured objective using the same point penalties and decision parameterization. */
class StructuredEvaluator {
    Optimizer optimizer_;
    struct Objective { DurationCost duration; IntegralCost integral; DomainCost decision; } objective_;
public:
    static constexpr const char *label="spline_optimizer";
    StructuredEvaluator(const ProblemData &p,const Geometry &geometry)
        :optimizer_(Parameterization{{},{&p,&geometry},{}}),objective_{{p.time_weight},{p,geometry},{p}} {
        SplineTrajectory::OptimizerOptions options;
        options.energy_weight=p.energy; options.integration_steps=p.integration;
        const auto prepared=optimizer_.prepare(splineProblem(p),options);
        if(!prepared) throw std::runtime_error(prepared.message);
    }
    double evaluate(const Eigen::VectorXd &x,Eigen::VectorXd &gradient) {
        return optimizer_.evaluate(x,gradient,objective_).cost;
    }
    void accept(const Eigen::VectorXd &x) {
        const auto status=optimizer_.updateAccepted(x);
        if(!status) throw std::runtime_error(status.message);
    }
    typename Spline::Polynomial snapshot() const { return optimizer_.copyPolynomial(); }
};
}
