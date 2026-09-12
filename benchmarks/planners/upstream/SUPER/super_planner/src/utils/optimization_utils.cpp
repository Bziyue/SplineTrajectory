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

#include <utils/optimization/optimization_utils.h>

using namespace optimization_utils;
using namespace math_utils;

template<typename EIGENVEC>
bool Gcopter<EIGENVEC>::smoothedL1(const double &x,
                                   const double &mu,
                                   double &f,
                                   double &df) {
    if (x < 0.0) {
        return false;
    } else if (x > mu) {
        f = x - 0.5 * mu;
        df = 1.0;
        return true;
    } else {
        const double xdmu = x / mu;
        const double sqrxdmu = xdmu * xdmu;
        const double mumxd2 = mu - 0.5 * x;
        f = mumxd2 * sqrxdmu * xdmu;
        df = sqrxdmu * ((-0.5) * xdmu + 3.0 * mumxd2 / mu);
        return true;
    }
}



template<typename EIGENVEC>
void Gcopter<EIGENVEC>::forwardMapTauToT(const Eigen::VectorXd &tau,
                                         Eigen::VectorXd &T) {
    const long sizeTau = tau.size();
    T.resize(sizeTau);
    for (long i = 0; i < sizeTau; i++) {
        T(i) = tau(i) > 0.0
               ? ((0.5 * tau(i) + 1.0) * tau(i) + 1.0)
               : 1.0 / ((0.5 * tau(i) - 1.0) * tau(i) + 1.0);
    }
}

template<typename EIGENVEC>
void Gcopter<EIGENVEC>::backwardMapTToTau(const Eigen::VectorXd &T, EIGENVEC &tau) {
    const long sizeT = T.size();
    tau.resize(sizeT);
    for (long i = 0; i < sizeT; i++) {
        tau(i) = T(i) > 1.0
                 ? (sqrt(2.0 * T(i) - 1.0) - 1.0)
                 : (1.0 - sqrt(2.0 / T(i) - 1.0));
    }
}

template<typename EIGENVEC>
void Gcopter<EIGENVEC>::propagateGradientTToTau(const Eigen::VectorXd &tau, const Eigen::VectorXd &gradT,
                                                EIGENVEC &gradTau) {
    const long sizeTau = tau.size();
    gradTau.resize(sizeTau);
    double denSqrt;
    for (long i = 0; i < sizeTau; i++) {
        if (tau(i) > 0) {
            gradTau(i) = gradT(i) * (tau(i) + 1.0);
        } else {
            denSqrt = (0.5 * tau(i) - 1.0) * tau(i) + 1.0;
            gradTau(i) = gradT(i) * (1.0 - tau(i)) / (denSqrt * denSqrt);
        }
    }
}

template<typename EIGENVEC>
void Gcopter<EIGENVEC>::mapIntervalToInf(const double &lower_bound, const double &upper_bound, const double &inter,
                                         double &inf) {
    inf = asin((inter - (lower_bound + upper_bound) / 2) / ((upper_bound - lower_bound) / 2));
//    inf = tan(2 * (inter - lower_bound)/ (upper_bound - lower_bound) - 1);
//    inf = atanh((inter - lower_bound)/(upper_bound - lower_bound));
}

template<typename EIGENVEC>
void Gcopter<EIGENVEC>::mapInfToInterval(const double &lower_bound, const double &upper_bound, const double &inf,
                                         double &inter) {
    inter = (upper_bound - lower_bound) / 2 * sin(inf) + (lower_bound + upper_bound) / 2;
//    inter = (atan(inf) + 1) * (upper_bound - lower_bound) / 2 + lower_bound;
//    inter = tanh(inf) * (upper_bound - lower_bound) + lower_bound;
}

template<typename EIGENVEC>
void Gcopter<EIGENVEC>::propagateGradIntervalToInf(const double &lower_bound, const double &upper_bound,
                                                   const double &inf, const double &grad_inter, double &grad_inf) {
    grad_inf = grad_inter * (upper_bound - lower_bound) / 2 * cos(inf);
//    grad_inf = (upper_bound - lower_bound) / 2.0 * (1.0 / (1 + inf * inf)) * grad_inter;
//    grad_inf = (1-tanh(inf)*tanh(inf)) * (upper_bound - lower_bound) * grad_inter;
}

template<typename EIGENVEC>
void Gcopter<EIGENVEC>::normRetrictionLayer(const Eigen::VectorXd &xi, const PolyhedronV &vPoly, double &cost,
                                            EIGENVEC &gradXi) {
    const long sizeP = xi.size() / vPoly.cols();
    gradXi.resize(xi.size());
    double sqrNormQ, sqrNormViolation, c, dc;
    Eigen::VectorXd q;
    for (long i = 0, j = 0, k; i < sizeP; i++, j += k) {
        k = vPoly.cols();
        q = xi.segment(j, k);
        sqrNormQ = q.squaredNorm();
        sqrNormViolation = sqrNormQ - 1.0;
        if (sqrNormViolation > 0.0) {
            c = sqrNormViolation * sqrNormViolation;
            dc = 3.0 * c;
            c *= sqrNormViolation;
            cost += c;
            gradXi.segment(j, k) += dc * 2.0 * q;
        }
    }
}

template<typename EIGENVEC>
void
Gcopter<EIGENVEC>::backwardGradP(const Eigen::VectorXd &xi, const PolyhedronV &vPoly, const Eigen::Matrix3Xd &gradP,
                                 EIGENVEC &gradXi) {
    const long sizeP = xi.size() / vPoly.cols();
    gradXi.resize(xi.size());

    double normInv;
    Eigen::VectorXd q, gradQ, unitQ;
    for (long i = 0, j = 0, k; i < sizeP; i++, j += k) {
        k = vPoly.cols();
        q = xi.segment(j, k);
        normInv = 1.0 / q.norm();
        unitQ = q * normInv;
        gradQ.resize(k);
        gradQ.head(k - 1) = (vPoly.rightCols(k - 1).transpose() * gradP.col(i)).array() *
                            unitQ.head(k - 1).array() * 2.0;
        gradQ(k - 1) = 0.0;
        gradXi.segment(j, k) = (gradQ - unitQ * unitQ.dot(gradQ)) * normInv;
    }


}


template<typename EIGENVEC>
void Gcopter<EIGENVEC>::backwardGradP(const Eigen::VectorXd &xi,
                                      const Eigen::VectorXi &vIdx,
                                      const PolyhedraV &vPolys,
                                      const Eigen::Matrix3Xd &gradP,
                                      EIGENVEC &gradXi) {
    const long sizeP = vIdx.size();
    gradXi.resize(xi.size());

    double normInv;
    Eigen::VectorXd q, gradQ, unitQ;
    for (long i = 0, j = 0, k, l; i < sizeP; i++, j += k) {
        l = vIdx(i);
        k = vPolys[l].cols();
        q = xi.segment(j, k);
        normInv = 1.0 / q.norm();
        unitQ = q * normInv;
        gradQ.resize(k);
        gradQ.head(k - 1) = (vPolys[l].rightCols(k - 1).transpose() * gradP.col(i)).array() *
                            unitQ.head(k - 1).array() * 2.0;
        gradQ(k - 1) = 0.0;
        gradXi.segment(j, k) = (gradQ - unitQ * unitQ.dot(gradQ)) * normInv;
    }

}

template<typename EIGENVEC>
void Gcopter<EIGENVEC>::backwardP(const Eigen::Matrix3Xd &points, const PolyhedronV &vPoly, EIGENVEC &xi) {
    // 所有需要优化的点大小
    const long sizeP = points.cols();
    // 最小的平方距离
    double minSqrD;
    lbfgs::lbfgs_parameter_t tiny_nls_params;
    tiny_nls_params.past = 0;
    tiny_nls_params.delta = 1.0e-5;
    tiny_nls_params.g_epsilon = FLT_EPSILON;
    tiny_nls_params.max_iterations = 128;

    Eigen::Matrix3Xd ovPoly;
    for (long i = 0, j = 0, k; i < sizeP; i++, j += k) {
        k = vPoly.cols();
        // ovPoly的尺寸为顶点数+1
        ovPoly.resize(3, k + 1);
        // 增加了一个该点本身
        ovPoly.col(0) = points.col(i);
        // 其余的都各自顶点
        ovPoly.rightCols(k) = vPoly;
        // 优化的xi维度等于顶点的个数
        Eigen::VectorXd x(k);
        // x设置为面分之一
        x.setConstant(sqrt(1.0 / static_cast<double>(k)));
        lbfgs::lbfgs_optimize(x,           // 优化变量
                              minSqrD,  // 代价
                              &Gcopter<EIGENVEC>::costTinyNLS, // 代价函数
                              nullptr,
                              nullptr,
                              &ovPoly,
                              tiny_nls_params);

        xi.segment(j, k) = x; // 优化出来的xi放进去
    }
}

template<typename EIGENVEC>
double Gcopter<EIGENVEC>::costTinyNLS(void *ptr, const Eigen::VectorXd &xi, Eigen::VectorXd &gradXi) {
    const long n = xi.size();
    const Eigen::Matrix3Xd &ovPoly = *(Eigen::Matrix3Xd *) ptr;

    const double sqrNormXi = xi.squaredNorm();
    const double invNormXi = 1.0 / sqrt(sqrNormXi);
    const Eigen::VectorXd unitXi = xi * invNormXi;
    const Eigen::VectorXd r = unitXi.head(n - 1);
    const Eigen::Vector3d delta = ovPoly.rightCols(n - 1) * r.cwiseProduct(r) +
                                  ovPoly.col(1) - ovPoly.col(0);
    double cost = delta.squaredNorm();
    gradXi.head(n - 1) = (ovPoly.rightCols(n - 1).transpose() * (2 * delta)).array() *
                         r.array() * 2.0;
    gradXi(n - 1) = 0.0;
    gradXi = (gradXi - unitXi.dot(gradXi) * unitXi).eval() * invNormXi;
    const double sqrNormViolation = sqrNormXi - 1.0;
    if (sqrNormViolation > 0.0) {
        double c = sqrNormViolation * sqrNormViolation;
        const double dc = 3.0 * c;
        c *= sqrNormViolation;
        cost += c;
        gradXi += dc * 2.0 * xi;
    }
    return cost;
}

template<typename EIGENVEC>
void Gcopter<EIGENVEC>::forwardP(const Eigen::VectorXd &xi, const Eigen::VectorXi &vIdx, const PolyhedraV &vPolys,
                                 Eigen::Matrix3Xd &P) {
    const long sizeP = vIdx.size();
    P.resize(3, sizeP);
    Eigen::VectorXd q;
    for (long i = 0, j = 0, k, l; i < sizeP; i++, j += k) {
        l = vIdx(i);
        k = vPolys[l].cols();
        q = xi.segment(j, k).normalized().head(k - 1);
        P.col(i) = vPolys[l].rightCols(k - 1) * q.cwiseProduct(q) +
                   vPolys[l].col(0);
    }
}

template<typename EIGENVEC>
void Gcopter<EIGENVEC>::forwardP(const Eigen::VectorXd &xi, const PolyhedronV &vPoly, Eigen::Matrix3Xd &P) {
    const long sizeP = xi.size() / vPoly.cols();
    P.resize(3, sizeP);
    Eigen::VectorXd q;
    for (long i = 0, j = 0, k; i < sizeP; i++, j += k) {
        k = vPoly.cols();
        q = xi.segment(j, k).normalized().head(k - 1);
        P.col(i) = vPoly.rightCols(k - 1) * q.cwiseProduct(q) +
                   vPoly.col(0);
    }

}

template<typename EIGENVEC>
void Gcopter<EIGENVEC>::backwardP(const Eigen::Matrix3Xd &P,
                                  const Eigen::VectorXi &vIdx,
                                  const PolyhedraV &vPolys,
                                  EIGENVEC &xi) {
    const long sizeP = P.cols();

    double minSqrD;
    lbfgs::lbfgs_parameter_t tiny_nls_params;
    tiny_nls_params.past = 0;
    tiny_nls_params.delta = 1.0e-5;
    tiny_nls_params.g_epsilon = FLT_EPSILON;
    tiny_nls_params.max_iterations = 128;

    Eigen::Matrix3Xd ovPoly;
    for (long i = 0, j = 0, k, l; i < sizeP; i++, j += k) {
        l = vIdx(i);
        k = vPolys[l].cols();

        ovPoly.resize(3, k + 1);
        ovPoly.col(0) = P.col(i);
        ovPoly.rightCols(k) = vPolys[l];
        Eigen::VectorXd x(k);
        x.setConstant(sqrt(1.0 / static_cast<double>(k)));
        lbfgs::lbfgs_optimize(x,
                              minSqrD,
                              &Gcopter<EIGENVEC>::costTinyNLS,
                              nullptr,
                              nullptr,
                              &ovPoly,
                              tiny_nls_params);

        xi.segment(j, k) = x;
    }
}

template<typename EIGENVEC>
void
Gcopter<EIGENVEC>::normRetrictionLayer(const Eigen::VectorXd &xi, const Eigen::VectorXi &vIdx, const PolyhedraV &vPolys,
                                       double &cost, EIGENVEC &gradXi) {
    const long sizeP = vIdx.size();
    gradXi.resize(xi.size());

    double sqrNormQ, sqrNormViolation, c, dc;
    Eigen::VectorXd q;
    for (long i = 0, j = 0, k; i < sizeP; i++, j += k) {
        k = vPolys[vIdx(i)].cols();

        q = xi.segment(j, k);
        sqrNormQ = q.squaredNorm();
        sqrNormViolation = sqrNormQ - 1.0;
        if (sqrNormViolation > 0.0) {
            c = sqrNormViolation * sqrNormViolation;
            dc = 3.0 * c;
            c *= sqrNormViolation;
            cost += c;
            gradXi.segment(j, k) += dc * 2.0 * q;
        }
    }
}

template
class optimization_utils::Gcopter<Eigen::Map<Eigen::VectorXd>>;