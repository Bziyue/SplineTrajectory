/**
* This file is part of SUPER
*
* Copyright 2025 Yunfan REN, MaRS Lab, University of Hong Kong, <mars.hku.hk>
* Developed by Yunfan REN <renyf at connect dot hku dot hk>
* for more information see <https://github.com/hku-mars/SUPER>.
* If you use this code, please cite the respective publications as
* listed on the above website.
*
* SUPER is free software: you can redistribute it and/or modify
* it under the terms of the GNU Lesser General Public License as published by
* the Free Software Foundation, either version 3 of the License, or
* (at your option) any later version.
*
* SUPER is distributed in the hope that it will be useful,
* but WITHOUT ANY WARRANTY; without even the implied warranty of
* MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
* GNU General Public License for more details.
*
* You should have received a copy of the GNU Lesser General Public License
* along with SUPER. If not, see <http://www.gnu.org/licenses/>.
*/

#include <traj_opt/exp_traj_optimizer_s4.h>
#include <utils/optimization/lbfgs.h>
#include <ros_interface/ros_interface.hpp>

#define POS_IDX 1
#define VEL_IDX 2
#define ACC_IDX 3
#define JER_IDX 4
#define ATT_IDX 5
#define OMG_IDX 6
#define THR_IDX 7

using namespace traj_opt;
using namespace color_text;
using namespace super_utils;
using namespace math_utils;
using namespace optimization_utils;

using Vec8f = Eigen::Matrix<double, 8, 1>;
using Mat83f = Eigen::Matrix<double, 8, 3>;

void ExpTrajOpt::constraintsFunctional(const VecDf &T,
                                       const MatD3f &coeffs,
                                       const VecDi &hIdx,
                                       const PolyhedraH &hPolys,
                                       const Mat3Df &waypoint_attractor,
                                       const VecDf &waypoint_attractor_dead_d,
                                       const double &smoothFactor,
                                       const int &integralResolution,
                                       const VecDf &magnitudeBounds,
                                       const VecDf &penaltyWeights,
                                       flatness::FlatnessMap &flatMap,
        // outputs
                                       double &cost,
                                       VecDf &gradT,
                                       MatD3f &gradC,
                                       VecDf &pena_log) {
    /* 1) define some varible alias*/
    const auto &vmax = magnitudeBounds[0];
    const auto &amax = magnitudeBounds[1];
    const auto &jmax = magnitudeBounds[2];
    const auto &omgmax = magnitudeBounds[3];
    const auto &accthrmin = magnitudeBounds[4];
    const auto &accthrmax = magnitudeBounds[5];

    const auto &vmaxSqr = vmax * vmax;
    const auto &amaxSqr = amax * amax;
    const auto &jmaxSqr = jmax * jmax;
    const auto &omgmaxSqr = omgmax * omgmax;

    const auto &thrustMean = 0.5 * (accthrmax + accthrmin);
    const auto &thrustRadi = 0.5 * std::abs(accthrmax - accthrmin);
    const auto &thrustSqrRadi = thrustRadi * thrustRadi;

    const auto &weightPos = penaltyWeights[0];
    const auto &weightVel = penaltyWeights[1];
    const auto &weightAcc = penaltyWeights[2];
    const auto &weightJer = penaltyWeights[3];
    const auto &weightAtt = penaltyWeights[4];
    const auto &weightOmg = penaltyWeights[5];
    const auto &weightAccThr = penaltyWeights[6];

    const auto &piece_num = T.size();

    const double integralFrac = 1.0 / integralResolution;
    VecDf max_pena(8);
    max_pena.setZero();

    /* 2) add integral cost */

    for (int i = 0; i < piece_num; i++) {
        const Mat83f &c = coeffs.block<8, 3>(i * 8, 0);
        const auto &step = T(i) * integralFrac;
        for (int j = 0; j <= integralResolution; j++) {
            double s1 = j * step;
            double s2 = s1 * s1;
            double s3 = s2 * s1;
            double s4 = s2 * s2;
            double s5 = s4 * s1;
            double s6 = s4 * s2;
            double s7 = s4 * s3;
            Vec8f beta0, beta1, beta2, beta3, beta4;
            beta0 << 1.0, s1, s2, s3, s4, s5, s6, s7;
            beta1 << 0.0, 1.0, 2.0 * s1, 3.0 * s2, 4.0 * s3, 5.0 * s4, 6.0 * s5, 7.0 * s6;
            beta2 << 0.0, 0.0, 2.0, 6.0 * s1, 12.0 * s2, 20.0 * s3, 30.0 * s4, 42.0 * s5;
            beta3 << 0.0, 0.0, 0.0, 6.0, 24.0 * s1, 60.0 * s2, 120.0 * s3, 210.0 * s4;
            beta4 << 0.0, 0.0, 0.0, 0.0, 24.0, 120.0 * s1, 360.0 * s2, 840.0 * s3;
            //beta5 << 0.0, 0.0, 0.0, 0., 0.0, 120.0, 720.0 * s1, 2520.0 * s2;

            const Vec3f pos = c.transpose() * beta0;
            const Vec3f vel = c.transpose() * beta1;
            const Vec3f acc = c.transpose() * beta2;
            const Vec3f jer = c.transpose() * beta3;
            const Vec3f sna = c.transpose() * beta4;

            double tmp_cost{0.0};
            Vec3f gradPos{0, 0, 0}, gradVel{0, 0, 0}, gradAcc{0, 0, 0}, gradJer{0, 0, 0};

            /* 2.1  For position cost */
            const auto &L = hIdx(i);
            const auto &K = hPolys[L].rows();
            if (weightPos > 0) {
                for (int k = 0; k < K; k++) {
                    const Vec3f outerNormal = hPolys[L].block<1, 3>(k, 0);
                    const double violaPos = outerNormal.dot(pos) + hPolys[L](k, 3);
                    if (violaPos > max_pena(POS_IDX)) max_pena(POS_IDX) = violaPos;
                    double violaPosPena, violaPosPenaD;
                    if (gcopter::smoothedL1(violaPos, smoothFactor, violaPosPena, violaPosPenaD)) {
                        gradPos += weightPos * violaPosPenaD * outerNormal;
                        tmp_cost += weightPos * violaPosPena;
                    }
                }
            }

            /* 2.2  For attract point cost  */
            if (weightAtt > 0.0) {
                const auto is_waypoint = (j == 0) && (i != 0);
                const auto is_end = ((j == integralResolution) && (i != piece_num - 1));
                const auto idx = is_end ? i : i - 1;

                if (is_waypoint || is_end) {
                    Vec3f p_a = pos - waypoint_attractor.col(idx);
                    const auto &violaAtt =
                            p_a.squaredNorm() - waypoint_attractor_dead_d(idx) * waypoint_attractor_dead_d(idx);
                    double violaAttPena, violaAttPenaD;
                    if (violaAtt > max_pena(ATT_IDX)) max_pena(ATT_IDX) = violaAtt;
                    if (gcopter::smoothedL1(violaAtt, smoothFactor, violaAttPena, violaAttPenaD)) {
                        gradPos += weightAtt * violaAttPenaD * 2.0 * p_a;
                        tmp_cost += weightAtt * violaAttPena;
                    }
                }
            }

            /* 2.3 For vel cost  */
            const auto &violaVel = vel.squaredNorm() - vmaxSqr;
            double violaVelPena, violaVelPenaD;
            if (weightVel > 0 && gcopter::smoothedL1(violaVel, smoothFactor, violaVelPena, violaVelPenaD)) {
                gradVel += weightVel * violaVelPenaD * 2.0 * vel;
                tmp_cost += weightVel * violaVelPena;
                if (violaVel > max_pena(VEL_IDX)) max_pena(VEL_IDX) = violaVel;
            }

            /* 2.4 For acc cost  */
            const auto &violaAcc = acc.squaredNorm() - amaxSqr;
            double violaAccPena, violaAccPenaD;
            if (weightAcc > 0 && gcopter::smoothedL1(violaAcc, smoothFactor, violaAccPena, violaAccPenaD)) {
                gradAcc += weightAcc * violaAccPenaD * 2.0 * acc;
                tmp_cost += weightAcc * violaAccPena;
                if (violaAcc > max_pena(ACC_IDX)) max_pena(ACC_IDX) = violaAcc;
            }

            /* 2.5 For acc cost  */
            const auto &violaJer = jer.squaredNorm() - jmaxSqr;
            double violaJerPena, violaJerPenaD;
            if (weightJer > 0 && gcopter::smoothedL1(violaJer, smoothFactor, violaJerPena, violaJerPenaD)) {
                gradJer += weightJer * violaJerPenaD * 2.0 * jer;
                tmp_cost += weightJer * violaJerPena;
                if (violaJer > max_pena(JER_IDX)) max_pena(JER_IDX) = violaJer;
            }

            Vec3f totalGradPos{0.0, 0.0, 0.0}, totalGradVel{0.0, 0.0, 0.0},
                    totalGradAcc{0.0, 0.0, 0.0}, totalGradJer{0.0, 0.0, 0.0};

            /* 2.6  For omg amd thr cost  */
            if (weightOmg > 0 && weightAccThr > 0) {
                double thr;
                Vec4f quat;
                Vec3f omg;
                flatMap.forward(vel, acc, jer, 0.0, 0.0, thr, quat, omg);
                const auto &violaOmg = omg.squaredNorm() - omgmaxSqr;
                const auto &violaThrust = (thr - thrustMean) * (thr - thrustMean) - thrustSqrRadi;

                /* 2.6.1  For omg cost  */
                double violaOmgPena, violaOmgPenaD;
                Vec3f gradOmg{0, 0, 0};
                if (weightOmg > 0 && gcopter::smoothedL1(violaOmg, smoothFactor, violaOmgPena, violaOmgPenaD)) {
                    gradOmg += weightOmg * violaOmgPenaD * 2.0 * omg;
                    tmp_cost += weightOmg * violaOmgPena;
                    if (violaOmg > max_pena(OMG_IDX)) max_pena(OMG_IDX) = violaOmg;
                }

                /* 2.6.2  For thr cost  */
                double violaThrustPena, violaThrustPenaD;
                double gradThr{0.0};
                if (weightAccThr > 0 &&
                    gcopter::smoothedL1(violaThrust, smoothFactor, violaThrustPena, violaThrustPenaD)) {
                    gradThr += weightAccThr * violaThrustPenaD * 2.0 * (thr - thrustMean);
                    tmp_cost += weightAccThr * violaThrustPena;
                    if (violaThrust > max_pena(THR_IDX)) max_pena(THR_IDX) = violaThrust;
                }
                double totalGradPsi{0.0}, totalGradPsiD{0.0};
                flatMap.backward(gradPos, gradVel, gradAcc, gradJer, gradThr, Vec4f(0, 0, 0, 0), gradOmg,
                                 totalGradPos, totalGradVel, totalGradAcc, totalGradJer,
                                 totalGradPsi, totalGradPsiD);
            } else {
                totalGradPos = gradPos;
                totalGradVel = gradVel;
                totalGradAcc = gradAcc;
                totalGradJer = gradJer;
            }

            const auto node = (j == 0 || j == integralResolution) ? 0.5 : 1.0;
            const double alpha = j * integralFrac;
            gradC.block<8, 3>(i * 8, 0) += (beta0 * totalGradPos.transpose() +
                                            beta1 * totalGradVel.transpose() +
                                            beta2 * totalGradAcc.transpose() +
                                            beta3 * totalGradJer.transpose()) *
                                           node * step;
            gradT(i) += (totalGradPos.dot(vel) +
                         totalGradVel.dot(acc) +
                         totalGradAcc.dot(jer) +
                         totalGradJer.dot(sna)) *
                        alpha * node * step +
                        node * integralFrac * tmp_cost;
            cost += node * step * tmp_cost;
        }
    }

    /* 3) log all violations */
    pena_log.tail(7) = max_pena.tail(7);
}


/*
 * @ brief: This is the callback function of the L-BFGS solver
 *
 */
double ExpTrajOpt::costFunctional(void *ptr,
                                  const VecDf &x,
                                  VecDf &g) {
    /* 1) Decode the pointer */
    OptimizationVariables &obj = *static_cast<OptimizationVariables *>(ptr);
    const auto &dimTau = obj.temporalDim;
    const auto &dimXi = obj.spatialDim;
    const auto &weightT = obj.rho;
    const auto &vPolyIdx = obj.vPolyIdx;
    const auto &vPolytopes = obj.vPolytopes;
    const auto &hPolyIdx = obj.hPolyIdx;
    const auto &hPolytopes = obj.hPolytopes;
    const auto &waypoint_attractor = obj.waypoint_attractor;
    const auto &waypoint_attractor_dead_d = obj.waypoint_attractor_dead_d;
    const auto &smooth_eps = obj.smooth_eps;
    const auto &integral_res = obj.integral_res;
    const auto &magnitudeBounds = obj.magnitudeBounds;
    const auto &penaltyWeights = obj.penaltyWeights;
    const auto &block_energy_cost = obj.block_energy_cost;

    auto &quadrotor_flatness = obj.quadrotor_flatness;

    obj.iter_num++;
    const auto &pos_constraint_type = obj.pos_constraint_type;

    const Eigen::Map<const VecDf> tau(x.data(), dimTau);
    const Eigen::Map<const VecDf> xi(x.data() + dimTau, dimXi);
    Eigen::Map<VecDf> gradTau(g.data(), dimTau);
    Eigen::Map<VecDf> gradXi(g.data() + dimTau, dimXi);

    /* 2) Reconstruct the optimization varibles */

    Mat3Df points;
    VecDf times;
    gcopter::forwardMapTauToT(tau, times);
    switch (pos_constraint_type) {
        case 1: {
            VecDf xi_e = xi;
            points = Eigen::Map<Eigen::Matrix<double, 3, Eigen::Dynamic>>(xi_e.data(), 3, xi_e.size() / 3);
            break;
        }
        default: {
            gcopter::forwardP(xi, vPolyIdx, vPolytopes, points);
            break;
        }
    }

    /* 3) Compute the energy const and gradient */
    double cost{0};
    obj.minco.setParameters(points, times);
    MatD3f partialGradByCoeffs(8 * times.size(), 3);
    VecDf partialGradByTimes(times.size());
    partialGradByCoeffs.setZero();
    partialGradByTimes.setZero();
    if (!block_energy_cost) {
        obj.minco.getEnergy(cost);
        obj.minco.getEnergyPartialGradByCoeffs(partialGradByCoeffs);
        obj.minco.getEnergyPartialGradByTimes(partialGradByTimes);
    }
    obj.penalty_log(0) = cost;

    /* 4) Compute the constrain cost and gradient  */
    constraintsFunctional(times, obj.minco.getCoeffs(),
                          hPolyIdx, hPolytopes,
                          waypoint_attractor, waypoint_attractor_dead_d,
                          smooth_eps, integral_res,
                          magnitudeBounds, penaltyWeights,
                          quadrotor_flatness,
                          cost, partialGradByTimes, partialGradByCoeffs, obj.penalty_log);

    /* 5) Propagate the gradient from CT to PT */
    Mat3Df gradByPoints;
    VecDf gradByTimes;
    obj.minco.propogateGrad(partialGradByCoeffs, partialGradByTimes,
                            gradByPoints, gradByTimes);
    cost += weightT * times.sum();
    gradByTimes.array() += weightT;

    /* 6) Propagate the gradient from PT to optimization varibles*/
    gcopter::propagateGradientTToTau(tau, gradByTimes, gradTau);
    switch (pos_constraint_type) {
        case 1: {
            MatDf gp = gradByPoints;
            gradXi = Eigen::Map<VecDf>(gp.data(), gp.size());
            break;
        }
        default: {
            gcopter::backwardGradP(xi, vPolyIdx, vPolytopes, gradByPoints, gradXi);
            gcopter::normRetrictionLayer(xi, vPolyIdx, vPolytopes, cost, gradXi);
            break;
        }
    }
    return cost;
}

static void truncateToSixDecimals(double &num) {
    num = std::trunc(num * 1e6) / 1e6; // 直接截断，无四舍五入
}

/*
 * @ brief: This function pre-process the corridor
 *
 */
bool ExpTrajOpt::processCorridor() {
    const long sizeCorridor = static_cast<long>(opt_vars.hPolytopes.size() - 1);

    opt_vars.vPolytopes.clear();
    opt_vars.vPolytopes.reserve(2 * sizeCorridor + 1);

    long nv;
    PolyhedronH curIH;
    PolyhedronV curIV, curIOB;
    opt_vars.waypoint_attractor.resize(3, sizeCorridor);
    opt_vars.waypoint_attractor_dead_d.resize(sizeCorridor);
    opt_vars.hOverlapPolytopes.resize(sizeCorridor);

    for (long i = 0; i < sizeCorridor; i++) {
        if (!geometry_utils::enumerateVs(opt_vars.hPolytopes[i], curIV)) {
            cout << YELLOW << " -- [SUPER] in [ GcopterExpS4::processCorridor]: Failed to enumerate corridor Vs." << RESET
                 << endl;
            return false;
        }
        nv = curIV.cols();
        curIOB.resize(3, nv);
        curIOB.col(0) = curIV.col(0);
        curIOB.rightCols(nv - 1) = curIV.rightCols(nv - 1).colwise() - curIV.col(0);
        opt_vars.vPolytopes.push_back(curIOB);
        curIH.resize(opt_vars.hPolytopes[i].rows() + opt_vars.hPolytopes[i + 1].rows(), 4);
        curIH.topRows(opt_vars.hPolytopes[i].rows()) = opt_vars.hPolytopes[i];
        curIH.bottomRows(opt_vars.hPolytopes[i + 1].rows()) = opt_vars.hPolytopes[i + 1];
        opt_vars.hOverlapPolytopes[i] = curIH;
        Vec3f interior;
        const double &dis = geometry_utils::findInteriorDist(curIH, interior) / 2;
        opt_vars.waypoint_attractor.col(i) = curIV.colwise().mean();
        opt_vars.waypoint_attractor_dead_d(i) = dis;
        nv = curIV.cols();
        curIOB.resize(3, nv);
        curIOB.col(0) = curIV.col(0);
        curIOB.rightCols(nv - 1) = curIV.rightCols(nv - 1).colwise() - curIV.col(0);
        opt_vars.vPolytopes.push_back(curIOB);
    }

    if (!geometry_utils::enumerateVs(opt_vars.hPolytopes.back(), curIV)) {
        cout << YELLOW << " -- [SUPER] in [ GcopterExpS4::processCorridor]: Failed to enumerate corridor Vs." <<
             RESET << endl;
        return false;
    }

    nv = curIV.cols();
    curIOB.resize(3, nv);
    curIOB.col(0) = curIV.col(0);
    curIOB.rightCols(nv - 1) = curIV.rightCols(nv - 1).colwise() - curIV.col(0);
    opt_vars.vPolytopes.push_back(curIOB);
    return true;
}

bool ExpTrajOpt::processCorridorWithGuideTraj() {
    // * 1) allocate memory for vertex
    const int sizeCorridor = static_cast<int>(opt_vars.hPolytopes.size() - 1);

    opt_vars.vPolytopes.clear();
    opt_vars.vPolytopes.reserve(2 * sizeCorridor + 1);

    long nv;
    PolyhedronH curIH;
    PolyhedronV curIV, curIOB;
    opt_vars.waypoint_attractor.resize(3, sizeCorridor);
    opt_vars.hOverlapPolytopes.resize(sizeCorridor);
    opt_vars.waypoint_attractor_dead_d.resize(sizeCorridor);
    // * 2) Process the corridor
    for (int i = 0; i < sizeCorridor; i++) {
        // * 2.1) Get current vertex
        if (!geometry_utils::enumerateVs(opt_vars.hPolytopes[i], curIV)) {
            cout << YELLOW << " -- [SUPER] in [ GcopterExpS4::processCorridor]: Failed to enumerate corridor Vs."
                 << RESET << endl;

            return false;
        }
        // * 2.3) Conver the vertex to the frame of the first point
        nv = curIV.cols();
        curIOB.resize(3, nv);
        // *    Save the position of the first point
        curIOB.col(0) = curIV.col(0);
        // *    Use the relative position of the rest vertex.
        curIOB.rightCols(nv - 1) = curIV.rightCols(nv - 1).colwise() - curIV.col(0);
        // *    save the i-th corridor's vertex
        opt_vars.vPolytopes.push_back(curIOB);

        // * 2.4) Find the overlap corridor
        curIH.resize(opt_vars.hPolytopes[i].rows() + opt_vars.hPolytopes[i + 1].rows(), 4);
        curIH.topRows(opt_vars.hPolytopes[i].rows()) = opt_vars.hPolytopes[i];
        curIH.bottomRows(opt_vars.hPolytopes[i + 1].rows()) = opt_vars.hPolytopes[i + 1];
        opt_vars.hOverlapPolytopes[i] = curIH;
        Vec3f interior;

        const double dis = geometry_utils::findInteriorDist(curIH, interior) / 2;
        if (dis < 0.0 || std::isinf(dis)) {

            cout << YELLOW << " -- [SUPER] in [ GcopterExpS4::processCorridor]: Failed findInteriorDist Vs." <<
                 RESET << endl;
            return false;
        }
        geometry_utils::enumerateVs(curIH, interior, curIV);
        const double test_sum = curIV.sum();
        if (std::isnan(test_sum) || std::isinf(test_sum)) {
            return false;
        }
        opt_vars.waypoint_attractor.col(i) = interior;
        opt_vars.waypoint_attractor_dead_d(i) = dis;
        nv = curIV.cols();
        curIOB.resize(3, nv);
        curIOB.col(0) = curIV.col(0);
        curIOB.rightCols(nv - 1) = curIV.rightCols(nv - 1).colwise() - curIV.col(0);
        opt_vars.vPolytopes.push_back(curIOB);
    }

    // * 3) Time and waypoint allocation for hot initialization
    VecDf min_dis(opt_vars.waypoint_attractor.cols());
    VecDi min_id(opt_vars.waypoint_attractor.cols());
    VecDf time_stamps(opt_vars.waypoint_attractor.cols() + 2);
    time_stamps(0) = 0.0;
    time_stamps(opt_vars.waypoint_attractor.cols() + 1) = opt_vars.guide_t.back();
    min_id.setConstant(0);
    min_dis.setConstant(std::numeric_limits<double>::max());
    for (int i = 0; i < opt_vars.guide_path.size(); i++) {
        for (int j = 0; j < opt_vars.waypoint_attractor.cols(); j++) {
            const double dis = (opt_vars.guide_path[i] - opt_vars.waypoint_attractor.col(j)).norm();
            if (dis < min_dis[j]) {
                min_dis[j] = dis;
                min_id[j] = i;
                opt_vars.points.col(j) = opt_vars.waypoint_attractor.col(j);//opt_vars.guide_path[i];
                time_stamps(j + 1) = opt_vars.guide_t[i];
            }
        }
    }

    for (int i = 1; i < time_stamps.size(); i++) {
        opt_vars.times(i - 1) = time_stamps(i) - time_stamps(i - 1);
        opt_vars.times(i - 1) = std::max(0.01, opt_vars.times(i - 1));
    }

    if (!geometry_utils::enumerateVs(opt_vars.hPolytopes.back(), curIV)) {
        return false;
    }
    nv = curIV.cols();
    curIOB.resize(3, nv);
    curIOB.col(0) = curIV.col(0);
    curIOB.rightCols(nv - 1) = curIV.rightCols(nv - 1).colwise() - curIV.col(0);
    opt_vars.vPolytopes.push_back(curIOB);
    return true;
}

void ExpTrajOpt::defaultInitialization() {
    const VecDf dis = (opt_vars.init_path.leftCols(opt_vars.piece_num) -
                       opt_vars.init_path.rightCols(opt_vars.piece_num)).colwise().norm();
    const double speed = cfg_.max_vel;
    opt_vars.times = dis / speed;
    opt_vars.points = opt_vars.waypoint_attractor;
}

bool ExpTrajOpt::setupProblemAndCheck() {
    // init internal variables size;
    opt_vars.piece_num = static_cast<int>(opt_vars.hPolytopes.size());
    opt_vars.times.resize(opt_vars.piece_num);
    opt_vars.points.resize(3, opt_vars.piece_num - 1);


    // Check corridor and init points
    if (opt_vars.default_init) {
        throw std::runtime_error("Not support default init in this version.");
        if (!processCorridor()) {
            return false;
        }
    } else {
        if (!processCorridorWithGuideTraj()) {
            return false;
        }
    }
    opt_vars.init_path.resize(3, opt_vars.piece_num + 1);
    for (long i = 0; i < opt_vars.piece_num - 1; i++) {
        opt_vars.init_path.col(i + 1) = opt_vars.waypoint_attractor.col(i);
    }
    opt_vars.init_path.col(0) = opt_vars.headPVAJ.col(0);
    opt_vars.init_path.rightCols(1) = opt_vars.tailPVAJ.col(0);
    if (opt_vars.default_init) {
        defaultInitialization();
    } else {
        opt_vars.times *= 0.8;
    }

    if (std::isnan(opt_vars.times.sum())) {
        cout << YELLOW << " -- [ExpOpt] Init times and point failed." << RESET << endl;
        return false;
    }

    const Mat3Df deltas = opt_vars.init_path.rightCols(opt_vars.piece_num)
                          - opt_vars.init_path.leftCols(opt_vars.piece_num);
    opt_vars.pieceIdx = (deltas.colwise().norm() / INFINITY).cast<int>().transpose();
    opt_vars.pieceIdx.array() += 1;

    opt_vars.temporalDim = opt_vars.piece_num;
    opt_vars.spatialDim = 0;
    opt_vars.vPolyIdx.resize(opt_vars.piece_num - 1);
    opt_vars.hPolyIdx.resize(opt_vars.piece_num);

    switch (cfg_.pos_constraint_type) {
        case 1: {
            for (int i = 0, j = 0, k; i < opt_vars.piece_num; i++) {
                k = opt_vars.pieceIdx(i);
                for (int l = 0; l < k; l++, j++) {
                    if (l < k - 1) {
                        opt_vars.vPolyIdx(j) = 2 * i;
                    } else if (i < opt_vars.piece_num - 1) {
                        opt_vars.vPolyIdx(j) = 2 * i + 1;
                    }
                    opt_vars.hPolyIdx(j) = i;
                }
            }
            opt_vars.spatialDim = 3 * (opt_vars.piece_num - 1);
            break;
        }
        default: {
            for (int i = 0, j = 0, k; i < opt_vars.piece_num; i++) {
                k = opt_vars.pieceIdx(i);
                for (int l = 0; l < k; l++, j++) {
                    if (l < k - 1) {
                        opt_vars.vPolyIdx(j) = 2 * i;
                        opt_vars.spatialDim += static_cast<int>(opt_vars.vPolytopes[2 * i].cols());
                    } else if (i < opt_vars.piece_num - 1) {
                        opt_vars.vPolyIdx(j) = 2 * i + 1;
                        opt_vars.spatialDim += static_cast<int>(opt_vars.vPolytopes[2 * i + 1].cols());
                    }
                    opt_vars.hPolyIdx(j) = i;
                }
            }
        }
    }

    opt_vars.minco.setConditions(opt_vars.headPVAJ, opt_vars.tailPVAJ, opt_vars.piece_num);
    opt_vars.gradByPoints.resize(3, opt_vars.piece_num - 1);
    opt_vars.gradByTimes.resize(opt_vars.piece_num);
    opt_vars.partialGradByCoeffs.resize(8 * opt_vars.piece_num, 3);
    opt_vars.partialGradByTimes.resize(opt_vars.piece_num);
    return true;
}

bool ExpTrajOpt::setInitPsAndTs(const vec_Vec3f &init_ps, const vector<double> &init_ts) {
    opt_vars.default_init = false;
    if (opt_vars.times.size() != init_ts.size()) {
        return false;
    }
    if (opt_vars.points.cols() != init_ps.size()) {
        return false;
    }

    for (long i = 0; i < opt_vars.points.cols(); i++) {
        opt_vars.times[i] = init_ts[i];
        opt_vars.points.col(i) = init_ps[i];
    }
    opt_vars.times[opt_vars.times.size() - 1] = init_ts.back();
    return true;
}

double ExpTrajOpt::optimize(Trajectory &traj, const double &relCostTol) {
    /* 1) allocate vector for optimization varibles */
    VecDf x(opt_vars.temporalDim + opt_vars.spatialDim);
    /*    creat map for the opt_var vector */
    Eigen::Map<VecDf> tau(x.data(), opt_vars.temporalDim);
    Eigen::Map<VecDf> xi(x.data() + opt_vars.temporalDim, opt_vars.spatialDim);

    opt_vars.penalty_log.resize(8);
    opt_vars.penalty_log.setZero();

    /* 2) check the initial value of the optimization varibles */
    if (opt_vars.times.minCoeff() < 1e-3) {
        cout << YELLOW << " -- [TrajOpt] Error, the init times have zero, force return." << RESET << endl;
        cout << " -- Head PVAJ: " << endl;
        cout << opt_vars.headPVAJ << endl;
        cout << " -- Head PVAJ: " << endl;
        cout << opt_vars.tailPVAJ << endl;
        cout << " -- Times: " << endl;
        cout << opt_vars.times.transpose() << endl;
        return INFINITY;
    }

    if (opt_vars.given_init_ts_and_ps) {
        opt_vars.times = opt_vars.init_ts;
        for (int i = 0; i < opt_vars.init_ps.size(); i++) {
            opt_vars.points.col(i) = opt_vars.init_ps[i];
        }
    }

    /* 3)  construct the initial guess of the optimization varibles*/
    gcopter::backwardMapTToTau(opt_vars.times, tau);
    switch (opt_vars.pos_constraint_type) {
        case 1: {
            MatDf p_e = opt_vars.points;
            xi = Eigen::Map<const VecDf>(p_e.data(), p_e.size());
            break;
        }
        default: {
            gcopter::backwardP(opt_vars.points, opt_vars.vPolyIdx, opt_vars.vPolytopes, xi);
            break;
        }
    }

    /* 4) setup the optimizer's parameters*/
    opt_vars.iter_num = 0;
    double minCostFunctional{0};
    lbfgs::lbfgs_parameter_t lbfgs_params;
    lbfgs_params.mem_size = 256;
    lbfgs_params.past = 3;
    lbfgs_params.min_step = 1.0e-32;
    lbfgs_params.g_epsilon = 0.0;
    lbfgs_params.delta = relCostTol;
    VecDf times_init = opt_vars.times;

    opt_vars.init_ts = opt_vars.times;
    opt_vars.init_ps.clear();
    for (int col = 0; col < opt_vars.points.cols(); col++) {
        opt_vars.init_ps.emplace_back(opt_vars.points.col(col));
    }

    // keep fixed accuracy for
    for (int i = 0; i < opt_vars.waypoint_attractor_dead_d.size(); i++) {
        truncateToSixDecimals(opt_vars.waypoint_attractor_dead_d(i));
        truncateToSixDecimals(opt_vars.waypoint_attractor(0, i));
        truncateToSixDecimals(opt_vars.waypoint_attractor(1, i));
        truncateToSixDecimals(opt_vars.waypoint_attractor(2, i));
    }

    cout << std::fixed << std::setprecision(15);
    auto x0 = x;
    // only for debug
//    cout << " -- [ExpOpt] Start optimization." << x.transpose() << endl;
//    cout << " -- [ExpOpt] minCostFunctional: " << minCostFunctional << endl;
//    cout << " -- [ExpOpt] relCostTol: " << relCostTol << endl;
//    cout << " -- [ExpOpt] weightAtt: " << opt_vars.penaltyWeights(4) << endl;
//    cout << " -- [ExpOpt] waypoint_attractor: " << opt_vars.waypoint_attractor << endl;
//    cout << " -- [ExpOpt] waypoint_attractor_dead_d: " << opt_vars.waypoint_attractor_dead_d.transpose() << endl;
    // TimeConsuming ttt(" -- [ExpTrajOpt]", false);
    opt_vars.iter_num = 0;
    int ret = lbfgs::lbfgs_optimize(x,
                                    minCostFunctional,
                                    &ExpTrajOpt::costFunctional,
                                    nullptr,
                                    nullptr,
                                    &this->opt_vars,
                                    lbfgs_params);
    // double dt = ttt.stop();
    gcopter::forwardMapTauToT(tau, opt_vars.times);
    if (cfg_.print_optimizer_log) {
        cout << " -- [ExpOpt] Opt finish, with iter num: " << opt_vars.iter_num << "\n";
        cout << "\tEnergy: " << opt_vars.penalty_log(0) << endl;
        cout << "\tPos: " << opt_vars.penalty_log(1) << endl;
        cout << "\tVel: " << opt_vars.penalty_log(2) << endl;
        cout << "\tAcc: " << opt_vars.penalty_log(3) << endl;
        cout << "\tJerk: " << opt_vars.penalty_log(4) << endl;
        cout << "\tAttract: " << opt_vars.penalty_log(5) << endl;
        cout << "\tOmg: " << opt_vars.penalty_log(6) << endl;
        cout << "\tThr: " << opt_vars.penalty_log(7) << endl;
        cout << "\tOptimized Time: " << opt_vars.times.transpose() << endl;
    }

    if ((cfg_.penna_pos > 0 && opt_vars.penalty_log(1) > 0.2) ||
        // (cfg_.penna_vel > 0 && opt_vars.penalty_log(2) > cfg_.max_vel * cfg_.penna_margin) ||
        (cfg_.penna_acc > 0 && opt_vars.penalty_log(3) > cfg_.max_acc * cfg_.penna_margin) ||
        (cfg_.penna_omg > 0 && opt_vars.penalty_log(6) > cfg_.max_omg * cfg_.penna_margin) ||
        (cfg_.penna_thr > 0 && opt_vars.penalty_log(7) > cfg_.max_acc * cfg_.penna_margin)) {
        if (cfg_.print_optimizer_log) {
            cout << " -- [ExpOpt] Opt finish, with iter num: " << opt_vars.iter_num << "\n";
            cout << "\tEnergy: " << opt_vars.penalty_log(0) << endl;
            cout << "\tPos: " << opt_vars.penalty_log(1) << endl;
            cout << "\tVel: " << opt_vars.penalty_log(2) << endl;
            cout << "\tAcc: " << opt_vars.penalty_log(3) << endl;
            cout << "\tJerk: " << opt_vars.penalty_log(4) << endl;
            cout << "\tAttract: " << opt_vars.penalty_log(5) << endl;
            cout << "\tOmg: " << opt_vars.penalty_log(6) << endl;
            cout << "\tThr: " << opt_vars.penalty_log(7) << endl;
            cout << "\tOptimized Time: " << opt_vars.times.transpose() << endl;
        }
        ros_ptr_->warn(" -- [ExpOpt] Opt failed, Omg or thr or Pos violation.");
        ret = -1;
    }

    if (ret >= 0) {
        gcopter::forwardMapTauToT(tau, opt_vars.times);
        switch (opt_vars.pos_constraint_type) {
            case 1: {
                VecDf xi_e = xi;
                opt_vars.points = Eigen::Map<Eigen::Matrix<double, 3, Eigen::Dynamic>>(xi_e.data(), 3, xi_e.size() / 3);
                break;
            }
            default: {
                gcopter::forwardP(xi, opt_vars.vPolyIdx,
                                  opt_vars.vPolytopes, opt_vars.points);
                break;
            }
        }
//        opt_vars.minco.setConditions(opt_vars.headPVAJ, opt_vars.tailPVAJ, opt_vars.temporalDim);
        opt_vars.minco.setParameters(opt_vars.points, opt_vars.times);
        opt_vars.minco.getTrajectory(traj);
    } else {
        traj.clear();
        minCostFunctional = INFINITY;
        cout << YELLOW << " -- [MINCO] TrajOpt failed, " << lbfgs::lbfgs_strerror(ret) << RESET << endl;
//        cout << "Init times: " << times_init.transpose() << endl;
    }
    return minCostFunctional + ret;
}

ExpTrajOpt::ExpTrajOpt(const traj_opt::Config &cfg, const ros_interface::RosInterface::Ptr &ros_ptr) :
        cfg_(cfg),
        ros_ptr_(ros_ptr) {
    /// Use time as log file name
    //    auto now = std::chrono::system_clock::now();
    //    std::time_t t = std::chrono::system_clock::to_time_t(now);
    //    std::tm tm = *std::localtime(&t);
    //    std::stringstream ss;
    //    ss << std::put_time(&tm, "%Y-%m-%d_%H-%M-%S");
    //    std::string filename = ss.str() + "_exp_opt_log.csv";
    if(cfg_.save_log_en){
        std::string filename = "exp_opt_log.csv";
        failed_traj_log.open(DEBUG_FILE_DIR(filename), std::ios::out | std::ios::trunc);
        penalty_log.open(DEBUG_FILE_DIR("exp_opt_penna.csv"), std::ios::out | std::ios::trunc);
    }

    opt_vars.magnitudeBounds.resize(6);
    opt_vars.penaltyWeights.resize(7);
    opt_vars.magnitudeBounds << cfg_.max_vel, cfg_.max_acc, cfg_.max_jerk,
            cfg_.max_omg, cfg_.min_acc_thr * cfg_.mass, cfg_.max_acc_thr * cfg_.mass;
    opt_vars.penaltyWeights << cfg_.penna_pos, cfg_.penna_vel,
            cfg_.penna_acc, cfg_.penna_jerk,
            cfg_.penna_attract, cfg_.penna_omg,
            cfg_.penna_thr;
    opt_vars.rho = cfg_.penna_t;
    opt_vars.pos_constraint_type = cfg_.pos_constraint_type;
    opt_vars.block_energy_cost = cfg_.block_energy_cost;
    opt_vars.smooth_eps = cfg_.smooth_eps;
    opt_vars.integral_res = cfg_.integral_reso;
    opt_vars.quadrotor_flatness = cfg_.quadrotot_flatness;
}

ExpTrajOpt::~ExpTrajOpt() {
    failed_traj_log.close();
    penalty_log.close();
}


//bool ExpTrajOpt::optimize(const StatePVAJ &headPVAJ, const StatePVAJ &tailPVAJ,
//                          PolytopeVec &sfcs,
//                          Trajectory &out_traj) {
//    /// Check if SFC is valid
//    if (sfcs.empty()) {
//        cout << YELLOW << " -- [TrajOpt] Error, the SFC is empty." << RESET << endl;
//        return false;
//    }
//
//    if (!SimplifySFC(headPVAJ.col(0), tailPVAJ.col(0), sfcs)) {
//        cout << YELLOW << " -- [TrajOpt] Cannot simplify sfcs." << RESET << endl;
//        //        VisualUtils::VisualizePoint(mkr_pub_, headPVAJ.col(0),Color::Pink(),"ill_start",0.5,1);
//        //        VisualUtils::VisualizePoint(mkr_pub_, tailPVAJ.col(0),Color::Pink(),"ill_end",0.5,2);
//        //        cout << "headPVAJ: " << headPVAJ.col(0).transpose() << endl;
//        //        cout << "tailPVAJ: " << tailPVAJ.col(0).transpose() << endl;
//        //        cout << YELLOW << "Killing the node." << RESET << endl;
//        //        exit(-1);
//        return false;
//    }
//
//    for (const auto &poly: sfcs) {
//        if (std::isnan(poly.GetPlanes().sum())) {
//            cout << YELLOW << " -- [TrajOpt] Error, the SFC containes NaN." << RESET << endl;
//            return false;
//        }
//    }
//
//    bool success{true};
//
//    /// Setup optimization problems
//    opt_vars.default_init = true;
//    opt_vars.given_init_ts_and_ps = false;
//    opt_vars.headPVAJ = headPVAJ;
//    opt_vars.tailPVAJ = tailPVAJ;
//    opt_vars.guide_path.clear();
//    opt_vars.guide_t.clear();
//    opt_vars.hPolytopes.resize(sfcs.size());
//    for (long i = 0; i < sfcs.size(); i++) {
//        opt_vars.hPolytopes[i] = sfcs[i].GetPlanes();
//    }
//
//    if (!setupProblemAndCheck()) {
//        cout << YELLOW << " -- [SUPER] Minco corridor preprocess error." << RESET << endl;
//        success = false;
//    }
//
//    if (success && std::isinf(optimize(out_traj, cfg_.opt_accuracy))) {
//        std::cout << YELLOW << " -- [SUPER] in [ExpTrajOpt::optimize]: Optimization failed." << RESET << std::endl;
//        success = false;
//    }
//
//    if(success){
//        out_traj.start_WT = ros_ptr_->getSimTime();
//    }
//
//    if (!success && cfg_.save_log_en) {
//        failed_traj_log << 990419 << endl;
//        failed_traj_log << headPVAJ << endl;
//        failed_traj_log << tailPVAJ << endl;
//        for (long i = 0; i < sfcs.size(); i++) {
//            failed_traj_log << i << endl;
//            failed_traj_log << sfcs[i].GetPlanes() << endl;
//        }
//    }
//
//    return success;
//}

bool ExpTrajOpt::optimize(const StatePVAJ &headPVAJ, const StatePVAJ &tailPVAJ,
                          const vec_E<Vec3f> &guide_path, const vector<double> &guide_t,
                          PolytopeVec &sfcs,
                          Trajectory &out_traj) {
    /// Check if hot init is valid
    if (guide_path.size() != guide_t.size()) {
        cout << YELLOW << " -- [TrajOpt] Error, the guide trajectory has wrong path and time stamp." << RESET
             << endl;
        return false;
    }
    /// Check if SFC is valid
    if (sfcs.empty()) {
        cout << YELLOW << " -- [TrajOpt] Error, the SFC is empty." << RESET << endl;
        return false;
    }

    if (!SimplifySFC(headPVAJ.col(0), tailPVAJ.col(0), sfcs)) {
        cout << YELLOW << " -- [TrajOpt] Cannot simplify sfcs." << RESET << endl;
        return false;
    }

    bool success{true};

    /// Setup optimization problems
    opt_vars.default_init = false;
    opt_vars.given_init_ts_and_ps = false;
    opt_vars.headPVAJ = headPVAJ;
    opt_vars.tailPVAJ = tailPVAJ;
    opt_vars.guide_path = guide_path;
    opt_vars.guide_t = guide_t;
    opt_vars.hPolytopes.resize(sfcs.size());

    for (long i = 0; i < sfcs.size(); i++) {
        opt_vars.hPolytopes[i] = sfcs[i].GetPlanes();
        const Eigen::ArrayXd norms = opt_vars.hPolytopes[i].leftCols<3>().rowwise().norm();
        opt_vars.hPolytopes[i].array().colwise() /= norms;
    }

    if (!setupProblemAndCheck()) {
        cout << YELLOW << " -- [SUPER] Minco corridor preprocess error." << RESET << endl;
        success = false;
    }

    out_traj.clear();


    if (success && std::isinf(optimize(out_traj, cfg_.opt_accuracy))) {
        cout << YELLOW << " -- [SUPER] Minco exp_traj opt failed." << RESET << endl;
        success = false;
    }

    penalty_log << opt_vars.penalty_log.transpose() << endl;

    if (success) {
        out_traj.start_WT = ros_ptr_->getSimTime();
    }

    if (!success && cfg_.save_log_en) {
        failed_traj_log << 990419 << endl;
        failed_traj_log << headPVAJ << endl;
        failed_traj_log << tailPVAJ << endl;
        for (double i: guide_t) {
            failed_traj_log << i << " ";
        }
        failed_traj_log << endl;
        for (const auto &i: guide_path) {
            failed_traj_log << i.transpose() << " ";
        }
        failed_traj_log << endl;
        for (long i = 0; i < sfcs.size(); i++) {
            failed_traj_log << i << endl;
            failed_traj_log << sfcs[i].GetPlanes() << endl;
        }
    }
    return success;
}

bool ExpTrajOpt::optimize(const StatePVAJ &headPVAJ, const StatePVAJ &tailPVAJ,
                          PolytopeVec &sfcs,
                          const vec_Vec3f &init_ps,
                          const VecDf &init_ts,
                          Trajectory &out_traj) {
    vec_Vec3f guide_path;
    guide_path.emplace_back(headPVAJ.col(0));
    for (const auto &i: init_ps) {
        guide_path.emplace_back(i);
    }
    guide_path.emplace_back(tailPVAJ.col(0));
    vector<double> guide_t;
    guide_t.emplace_back(0);
    double accumulate_t = 0;
    for (int i = 0; i < init_ts.size(); i++) {
        accumulate_t += init_ts[i];
        guide_t.emplace_back(accumulate_t);
    }
    /// Check if hot init is valid
    if (guide_path.size() != guide_t.size()) {
        cout << YELLOW << " -- [TrajOpt] Error, the guide trajectory has wrong path and time stamp." << RESET
             << endl;
        return false;
    }
    /// Check if SFC is valid
    if (sfcs.empty()) {
        cout << YELLOW << " -- [TrajOpt] Error, the SFC is empty." << RESET << endl;
        return false;
    }

    if (!SimplifySFC(headPVAJ.col(0), tailPVAJ.col(0), sfcs)) {
        cout << YELLOW << " -- [TrajOpt] Cannot simplify sfcs." << RESET << endl;
        return false;
    }

    bool success{true};

    /// Setup optimization problems
    opt_vars.default_init = false;
    opt_vars.given_init_ts_and_ps = true;
    opt_vars.init_ts = init_ts;
    opt_vars.init_ps = init_ps;
    opt_vars.headPVAJ = headPVAJ;
    opt_vars.tailPVAJ = tailPVAJ;
    opt_vars.guide_path = guide_path;
    opt_vars.guide_t = guide_t;
    opt_vars.hPolytopes.resize(sfcs.size());

    for (long i = 0; i < sfcs.size(); i++) {
        opt_vars.hPolytopes[i] = sfcs[i].GetPlanes();
        const Eigen::ArrayXd norms = opt_vars.hPolytopes[i].leftCols<3>().rowwise().norm();
        opt_vars.hPolytopes[i].array().colwise() /= norms;
    }

    if (!setupProblemAndCheck()) {
        cout << YELLOW << " -- [SUPER] Minco corridor preprocess error." << RESET << endl;
        success = false;
    }

    out_traj.clear();

    if (success && std::isinf(optimize(out_traj, cfg_.opt_accuracy))) {
        cout << YELLOW << " -- [SUPER] Minco exp_traj opt failed." << RESET << endl;
        success = false;
    }
    penalty_log << opt_vars.penalty_log.transpose() << endl;

    if (success) {
        out_traj.start_WT = ros_ptr_->getSimTime();
    }


    if (!success && cfg_.save_log_en) {
        failed_traj_log << 990419 << endl;
        failed_traj_log << headPVAJ << endl;
        failed_traj_log << tailPVAJ << endl;
        for (double i: guide_t) {
            failed_traj_log << i << " ";
        }
        failed_traj_log << endl;
        for (const auto &i: guide_path) {
            failed_traj_log << i.transpose() << " ";
        }
        failed_traj_log << endl;
        for (long i = 0; i < sfcs.size(); i++) {
            failed_traj_log << i << endl;
            failed_traj_log << sfcs[i].GetPlanes() << endl;
        }
    }
    return success;
}