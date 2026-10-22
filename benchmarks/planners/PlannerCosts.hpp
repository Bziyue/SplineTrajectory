/* Numerical adapters derived from GCOPTER (MIT) and SUPER (LGPL-3.0-or-later).
 * Copyright (c) 2021 Zhepei Wang and 2025 Yunfan REN.
 * See upstream snapshots and README.md for provenance, retained notices and exact adaptations. */
#pragma once
#include "ProblemData.hpp"
#include "SplineOptimizer.hpp"
#include <native.hpp>
#include <algorithm>
#include <limits>
#include <numeric>

namespace planner_benchmark {
using Native = upstream::Native;
#if PLANNER_GCOPTER
using Spline = SplineTrajectory::QuinticSplineND<3>;
inline constexpr const char *project = "gcopter";
#else
using Spline = SplineTrajectory::SepticSplineND<3>;
inline constexpr const char *project = "super";
#endif
using State = SplineTrajectory::IntegralState<3>;
using Partials = SplineTrajectory::IntegralGradient<3>;

/** @brief Read-only geometry in the native half-space and normalized-square vertex conventions. */
struct Geometry {
    upstream::PolyhedraH planes;
    upstream::PolyhedraV vertices;
    Eigen::VectorXi plane_indices, vertex_indices;
    explicit Geometry(const ProblemData &p) : plane_indices(p.segments), vertex_indices(p.segments-1) {
        for(int i=0;i<p.segments;++i) {
            Eigen::MatrixX4d h(p.usesPolytopeMap()?6:0,4);
            if(p.usesPolytopeMap()) for(int axis=0;axis<3;++axis) {
                h.row(2*axis).setZero(); h(2*axis,axis)=1; h(2*axis,3)=-p.corridor_boxes[i](axis+3);
                h.row(2*axis+1).setZero(); h(2*axis+1,axis)=-1; h(2*axis+1,3)=p.corridor_boxes[i](axis);
            }
            planes.push_back(h); plane_indices(i)=i;
        }
        for(int i=0;i<p.segments-1;++i) {
            Eigen::Matrix3Xd v(3,8);
            const auto &box=p.map_boxes[i];
            v.col(0)=box.head<3>()-box.tail<3>();
            for(int corner=1;corner<8;++corner) for(int axis=0;axis<3;++axis)
                v(axis,corner)=(corner&(1<<axis))?2*box(axis+3):0;
            vertices.push_back(v); vertex_indices(i)=i;
        }
    }
};

/** @brief Minimum signed sphere distances and their gradient, an exact ESDF for disjoint spheres.
 * @note The smallest-index sphere wins ties. Dense clearance checks are sampled, not continuous certificates. */
inline double distance(const ProblemData &p, const Vector &position, Vector *gradient=nullptr) {
    double closest=std::numeric_limits<double>::infinity();
    Vector direction=Vector::Zero();
    for(const auto &sphere:p.obstacles) {
        const Vector offset=position-sphere.head<3>();
        const double length=offset.norm(), value=length-sphere(3);
        if(value<closest) { closest=value; direction=length>1e-12?Vector(offset/length):Vector::Zero(); }
    }
    if(gradient) *gradient=direction;
    return closest;
}
struct PositionPenalty {
    const ProblemData &problem;
    double operator()(const Vector &position, Vector &gradient) const {
        if(problem.scenario!="esdf") return 0;
        Vector direction;
        const double violation=problem.clearance-distance(problem,position,&direction);
        double cost, derivative;
        if(!Native::smoothedL1(violation,problem.smooth,cost,derivative)) return 0;
        gradient-=problem.obstacle_weight*derivative*direction;
        return problem.obstacle_weight*cost;
    }
};

/** @brief Scalar form of the native positive-time map, matching its forward/inverse/pullback formulas. */
using TimeMap = SplineTrajectory::QuadInvTimeMap;

/** @brief Allocation-free native normalized-square box map, or Cartesian coordinates for ESDF. */
struct SpatialMap {
    const ProblemData *problem;
    const Geometry *geometry;
    int dimension(int) const { return problem->usesPolytopeMap()?8:3; }
    SplineTrajectory::StatusOr<Vector> forward(const Eigen::Ref<const Eigen::VectorXd> &x,int index) const {
        if(!problem->usesPolytopeMap()) return Vector(x);
        Eigen::Matrix<double,8,1> unit=x;
        unit.normalize();
        const Vector result = geometry->vertices[index-1].rightCols<7>()*unit.head<7>().cwiseProduct(unit.head<7>())+
                              geometry->vertices[index-1].col(0);
        if (!result.allFinite())
            return SplineTrajectory::Status{SplineTrajectory::ErrorCode::NumericalFailure, "Nonfinite spatial map output", index};
        return result;
    }
    SplineTrajectory::StatusOr<Eigen::VectorXd> inverse(const Vector &point,int index) const {
        if(!problem->usesPolytopeMap()) return Eigen::VectorXd(point);
        const auto &box=problem->map_boxes[index-1];
        const Vector ratio=((point-box.head<3>()).array()/box.tail<3>().array()+1)*0.5;
        Eigen::VectorXd result(8);
        for(int corner=0;corner<8;++corner) {
            double weight=1;
            for(int axis=0;axis<3;++axis) weight*=(corner&(1<<axis))?ratio(axis):1-ratio(axis);
            if(weight<0) return SplineTrajectory::Status{SplineTrajectory::ErrorCode::InvalidInput, "Initial waypoint outside map box", index};
            result(corner?corner-1:7)=std::sqrt(weight);
        }
        return Eigen::VectorXd(result);
    }
    SplineTrajectory::Status backward(const Eigen::Ref<const Eigen::VectorXd> &x,const Vector &gradient,
                      int index,Eigen::Ref<Eigen::VectorXd> output) const {
        if(!problem->usesPolytopeMap()) { output=gradient; return SplineTrajectory::Status::success(); }
        const double inverse=1/x.norm();
        const Eigen::Matrix<double,8,1> unit=x*inverse;
        output.head<7>()=(geometry->vertices[index-1].rightCols<7>().transpose()*gradient).array()*unit.head<7>().array()*2;
        output(7)=0;
        const double projection=unit.dot(output);
        output=(output-unit*projection)*inverse;

        return SplineTrajectory::Status::success();
    }
};
struct Parameterization { TimeMap time; SpatialMap space; SplineTrajectory::NoAuxiliaryMap auxiliary; };
using Optimizer = SplineTrajectory::SplineOptimizer<Spline,Parameterization>;

struct DurationCost {
    double weight;
    template<class Curve>
    SplineTrajectory::EvaluationResult operator()(const SplineTrajectory::ParameterView<Curve> &state,
        SplineTrajectory::ParameterGradientView<Curve> &output) const
    {
        const auto &times = state.durations;
        auto &gradient = output.durations;
        gradient.array()+=weight;
        return {weight*std::accumulate(times.begin(),times.end(),0.0), SplineTrajectory::Status::success()};
    }
};
/** @brief Add the native cubic penalty on mapped coordinate norms without overwriting physical gradients. */
struct DomainCost {
    const ProblemData &problem;
    SplineTrajectory::EvaluationResult operator()(const SplineTrajectory::DecisionView &view,Eigen::Ref<Eigen::VectorXd> gradient) const {
        if(!problem.usesPolytopeMap()) return {0, SplineTrajectory::Status::success()};
        double cost=0;
        for(const auto &variable:view.layout.waypoints) {
            const auto q=view.variables.segment(variable.offset,variable.dof);
            const double violation=q.squaredNorm()-1;
            if(violation>0) { cost+=violation*violation*violation; gradient.segment(variable.offset,variable.dof)+=6*violation*violation*q; }
        }
        return {cost, SplineTrajectory::Status::success()};
    }
};

/** @brief Native point penalties with structured state partials; quadrature remains the optimizer's responsibility.
 * @note Flatness scratch is exclusively owned by this serial objective. SUPER's original joint enable condition is retained. */
struct IntegralCost {
    static constexpr int kDerivativeOrder=3;
    const ProblemData &problem;
    const Geometry &geometry;
    flatness::FlatnessMap flatmap;
    IntegralCost(const ProblemData &p,const Geometry &g):problem(p),geometry(g) {
        flatmap.reset(p.physical[0],p.physical[1],p.physical[2],p.physical[3],p.physical[4],p.physical[5]);
    }
    SplineTrajectory::EvaluationResult operator()(const SplineTrajectory::IntegralPointInfo &info,const State &state,Partials &out) {
        double cost=0;
        Vector gp=Vector::Zero(),gv=Vector::Zero(),ga=Vector::Zero(),gj=Vector::Zero();
#if PLANNER_GCOPTER
        const auto &bounds=problem.gcopter_bounds; const auto &weights=problem.gcopter_weights;
#else
        const auto &bounds=problem.super_bounds; const auto &weights=problem.super_weights;
#endif
        const auto &planes=geometry.planes[info.segment_index];
        for(Eigen::Index k=0;k<planes.rows();++k) {
            const Vector normal=planes.row(k).head<3>().transpose();
            double value,derivative;
            if(Native::smoothedL1(normal.dot(state.p)+planes(k,3),problem.smooth,value,derivative)) {
                gp+=weights(0)*derivative*normal; cost+=weights(0)*value;
            }
        }
        cost+=PositionPenalty{problem}(state.p,gp);
        auto boundPenalty=[&](const Vector &value,double limit,double weight,Vector &gradient) {
            double penalty,derivative;
            if(weight>0 && Native::smoothedL1(value.squaredNorm()-limit*limit,problem.smooth,penalty,derivative)) {
                gradient+=weight*derivative*2*value; cost+=weight*penalty;
            }
        };
        boundPenalty(state.v,bounds(0),weights(1),gv);
#if PLANNER_GCOPTER
        constexpr int omega_index=1,thrust_index=3,omega_weight=2,thrust_weight=4;
        const bool use_flatness=true;
#else
        if(weights(4)>0) {
            const bool first=info.step_index==0 && info.segment_index!=0;
            const bool last=info.step_index==info.step_count && info.segment_index!=problem.segments-1;
            if(first || last) {
                const int waypoint=last?info.segment_index+1:info.segment_index;
                const Vector difference=state.p-problem.waypoints.row(waypoint).transpose();
                double penalty,derivative;
                if(Native::smoothedL1(difference.squaredNorm(),problem.smooth,penalty,derivative)) {
                    gp+=weights(4)*derivative*2*difference; cost+=weights(4)*penalty;
                }
            }
        }
        boundPenalty(state.a,bounds(1),weights(2),ga);
        boundPenalty(state.j,bounds(2),weights(3),gj);
        constexpr int omega_index=3,thrust_index=4,omega_weight=5,thrust_weight=6;
        const bool use_flatness=weights(5)>0 && weights(6)>0;
#endif
        if(use_flatness) {
            double thrust; Eigen::Vector4d quaternion; Vector omega;
            flatmap.forward(state.v,state.a,state.j,0.0,0.0,thrust,quaternion,omega);
            Vector gradient_omega=Vector::Zero();
            boundPenalty(omega,bounds(omega_index),weights(omega_weight),gradient_omega);
            double gradient_thrust=0;
            Eigen::Vector4d gradient_quaternion=Eigen::Vector4d::Zero();
            const double mean=0.5*(bounds(thrust_index)+bounds(thrust_index+1));
            const double radius=0.5*std::abs(bounds(thrust_index+1)-bounds(thrust_index));
            double penalty,derivative;
#if PLANNER_GCOPTER
            const double cosine=1-2*(quaternion(1)*quaternion(1)+quaternion(2)*quaternion(2));
            if(Native::smoothedL1(std::acos(cosine)-bounds(2),problem.smooth,penalty,derivative)) {
                gradient_quaternion+=weights(3)*derivative/std::sqrt(1-cosine*cosine)*4*
                                     Eigen::Vector4d(0,quaternion(1),quaternion(2),0);
                cost+=weights(3)*penalty;
            }
#endif
            if(Native::smoothedL1((thrust-mean)*(thrust-mean)-radius*radius,problem.smooth,penalty,derivative)) {
                gradient_thrust+=weights(thrust_weight)*derivative*2*(thrust-mean); cost+=weights(thrust_weight)*penalty;
            }
            double yaw_gradient=0,yaw_rate_gradient=0;
#if PLANNER_GCOPTER
            flatmap.backward(gp,gv,gradient_thrust,gradient_quaternion,gradient_omega,
                             out.p,out.v,out.a,out.j,yaw_gradient,yaw_rate_gradient);
#else
            flatmap.backward(gp,gv,ga,gj,gradient_thrust,gradient_quaternion,gradient_omega,
                             out.p,out.v,out.a,out.j,yaw_gradient,yaw_rate_gradient);
#endif
        } else { out.p=gp; out.v=gv; out.a=ga; out.j=gj; }
        return {cost, SplineTrajectory::Status::success()};
    }
};

inline SplineTrajectory::SplineInput<Spline::kDimension> splineProblem(const ProblemData &p) {
    SplineTrajectory::SplineInput<Spline::kDimension> problem;
    problem.durations=p.durations; problem.waypoints=p.waypoints;
    problem.boundary.start.v=p.start_derivatives.col(0); problem.boundary.end.v=p.end_derivatives.col(0);
    problem.boundary.start.a=p.start_derivatives.col(1); problem.boundary.end.a=p.end_derivatives.col(1);
    problem.boundary.start.j=p.start_derivatives.col(2); problem.boundary.end.j=p.end_derivatives.col(2);
    return problem;
}
}
