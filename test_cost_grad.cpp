#include <Eigen/Dense>
#include <iostream>
#include <vector>
#include <cmath>
#include <limits>
#include <iomanip>

#include "SplineTrajectory.hpp"

using std::cout;
using std::endl;

static inline double rel_err(double a, double b) {
    double denom = std::max(1.0, std::max(std::abs(a), std::abs(b)));
    return std::abs(a - b) / denom;
}

template <typename SplineType>
double loss_from_spline(const SplineType& sp, double alpha_time) {
    const auto& C = sp.getPPoly().getCoefficients();   // (N*coeff_num) x DIM
    double Lc = C.squaredNorm();
    double Lt = 0.0;
    for (double h : sp.getTimeSegments()) Lt += h * h;
    return Lc + alpha_time * Lt;
}

template <typename SplineType>
void make_partials(const SplineType& sp,
                   double alpha_time,
                   typename SplineType::MatrixType& dLdC,
                   Eigen::VectorXd& dLdT) {
    const auto& C = sp.getPPoly().getCoefficients();
    dLdC = 2.0 * C;

    const auto& T = sp.getTimeSegments();
    dLdT.resize((int)T.size());
    for (int i = 0; i < (int)T.size(); ++i) dLdT(i) = 2.0 * alpha_time * T[i];
}

// ===================== Cubic =====================
template <int DIM>
void test_cubic() {
    using Spline = SplineTrajectory::CubicSplineND<DIM>;
    using Vec = typename Spline::VectorType;
    using Mat = typename Spline::MatrixType;

    cout << "\n=== CubicSplineND<" << DIM << "> gradient check ===\n";

    const double alpha_time = 0.3;
    const double eps = 1e-6;

    // ---- build a random-ish problem (deterministic) ----
    const int n_seg = 4;          // => n_pts = n_seg + 1
    const int n_pts = n_seg + 1;

    std::vector<double> T(n_seg);
    T[0] = 0.8; T[1] = 1.1; T[2] = 0.9; T[3] = 1.3;

    Mat P(n_pts, DIM);
    for (int i = 0; i < n_pts; ++i) {
        Vec v; v.setZero();
        for (int d = 0; d < DIM; ++d) v(d) = 0.2 * (i + 1) + 0.1 * (d + 1) * (i % 2 ? -1.0 : 1.0);
        P.row(i) = v.transpose();
    }

    SplineTrajectory::BoundaryConditions<DIM> bc;
    bc.start_velocity.setConstant(0.15);
    bc.end_velocity.setConstant(-0.05);

    Spline sp(T, P, /*start_time*/ 0.0, bc);

    // ---- analytic gradient via propagateGrad ----
    Mat dLdC;
    Eigen::VectorXd dLdT;
    make_partials(sp, alpha_time, dLdC, dLdT);

    auto g = sp.propagateGrad(dLdC, dLdT);

    // g.inner_points is (n_pts-2) x DIM
    // g.times is n_seg
    // g.start.v, g.end.v are DIM
    // NOTE: Cubic boundary grads struct has p,v only; p is also returned.

    // ---- numeric checks ----
    auto eval_loss = [&](const std::vector<double>& TT,
                         const Mat& PP,
                         const SplineTrajectory::BoundaryConditions<DIM>& bcc) {
        Spline tmp(TT, PP, 0.0, bcc);
        return loss_from_spline(tmp, alpha_time);
    };

    cout << std::scientific << std::setprecision(6);

    // 1) inner points gradients
    cout << "\n[Check] dL/dP (inner points)\n";
    for (int k = 1; k <= n_pts - 2; ++k) {
        for (int d = 0; d < DIM; ++d) {
            auto Pp = P, Pm = P;
            Pp(k, d) += eps;
            Pm(k, d) -= eps;
            double fp = eval_loss(T, Pp, bc);
            double fm = eval_loss(T, Pm, bc);
            double num = (fp - fm) / (2.0 * eps);
            double ana = g.inner_points.row(k - 1)(d);
            cout << "P[" << k << "]." << d
                 << "  ana=" << ana << "  num=" << num
                 << "  rel=" << rel_err(ana, num) << "\n";
        }
    }

    // 2) time gradients
    cout << "\n[Check] dL/dT\n";
    for (int i = 0; i < n_seg; ++i) {
        auto Tp = T, Tm = T;
        Tp[i] += eps;
        Tm[i] -= eps;
        double fp = eval_loss(Tp, P, bc);
        double fm = eval_loss(Tm, P, bc);
        double num = (fp - fm) / (2.0 * eps);
        double ana = g.times(i);
        cout << "T[" << i << "]  ana=" << ana << "  num=" << num
             << "  rel=" << rel_err(ana, num) << "\n";
    }

    // 3) boundary velocity gradients
    cout << "\n[Check] dL/d(start_velocity), dL/d(end_velocity)\n";
    for (int d = 0; d < DIM; ++d) {
        auto bcp = bc, bcm = bc;
        bcp.start_velocity(d) += eps;
        bcm.start_velocity(d) -= eps;
        double fp = eval_loss(T, P, bcp);
        double fm = eval_loss(T, P, bcm);
        double num = (fp - fm) / (2.0 * eps);
        double ana = g.start.v(d);
        cout << "start_v." << d << "  ana=" << ana << "  num=" << num
             << "  rel=" << rel_err(ana, num) << "\n";
    }
    for (int d = 0; d < DIM; ++d) {
        auto bcp = bc, bcm = bc;
        bcp.end_velocity(d) += eps;
        bcm.end_velocity(d) -= eps;
        double fp = eval_loss(T, P, bcp);
        double fm = eval_loss(T, P, bcm);
        double num = (fp - fm) / (2.0 * eps);
        double ana = g.end.v(d);
        cout << "end_v." << d << "  ana=" << ana << "  num=" << num
             << "  rel=" << rel_err(ana, num) << "\n";
    }
}

// ===================== Quintic =====================
template <int DIM>
void test_quintic() {
    using Spline = SplineTrajectory::QuinticSplineND<DIM>;
    using Vec = typename Spline::VectorType;
    using Mat = typename Spline::MatrixType;

    cout << "\n=== QuinticSplineND<" << DIM << "> gradient check ===\n";

    const double alpha_time = 0.3;
    const double eps = 1e-6;

    const int n_seg = 4;
    const int n_pts = n_seg + 1;

    std::vector<double> T(n_seg);
    T[0] = 0.8; T[1] = 1.1; T[2] = 0.9; T[3] = 1.3;

    Mat Pvec(n_pts, DIM);
    for (int i = 0; i < n_pts; ++i) {
        Vec v; v.setZero();
        for (int d = 0; d < DIM; ++d) v(d) = 0.25 * (i + 1) + 0.07 * (d + 1) * (i % 2 ? -1.0 : 1.0);
        Pvec.row(i) = v.transpose();
    }

    SplineTrajectory::BoundaryConditions<DIM> bc;
    bc.start_velocity.setConstant(0.12);
    bc.end_velocity.setConstant(-0.08);
    bc.start_acceleration.setConstant(0.05);
    bc.end_acceleration.setConstant(-0.03);

    Spline sp(T, Pvec, 0.0, bc);

    Mat dLdC;
    Eigen::VectorXd dLdT;
    make_partials(sp, alpha_time, dLdC, dLdT);

    auto g = sp.propagateGrad(dLdC, dLdT);

    auto eval_loss = [&](const std::vector<double>& TT,
                         const Mat& PP,
                         const SplineTrajectory::BoundaryConditions<DIM>& bcc) {
        Spline tmp(TT, PP, 0.0, bcc);
        return loss_from_spline(tmp, alpha_time);
    };

    cout << std::scientific << std::setprecision(6);

    cout << "\n[Check] dL/dP (inner points)\n";
    for (int k = 1; k <= n_pts - 2; ++k) {
        for (int d = 0; d < DIM; ++d) {
            auto Pp = Pvec, Pm = Pvec;
            Pp(k, d) += eps;
            Pm(k, d) -= eps;
            double fp = eval_loss(T, Pp, bc);
            double fm = eval_loss(T, Pm, bc);
            double num = (fp - fm) / (2.0 * eps);
            double ana = g.inner_points.row(k - 1)(d);
            cout << "P[" << k << "]." << d
                 << "  ana=" << ana << "  num=" << num
                 << "  rel=" << rel_err(ana, num) << "\n";
        }
    }

    cout << "\n[Check] dL/dT\n";
    for (int i = 0; i < n_seg; ++i) {
        auto Tp = T, Tm = T;
        Tp[i] += eps;
        Tm[i] -= eps;
        double fp = eval_loss(Tp, Pvec, bc);
        double fm = eval_loss(Tm, Pvec, bc);
        double num = (fp - fm) / (2.0 * eps);
        double ana = g.times(i);
        cout << "T[" << i << "]  ana=" << ana << "  num=" << num
             << "  rel=" << rel_err(ana, num) << "\n";
    }

    cout << "\n[Check] boundary v/a\n";
    for (int d = 0; d < DIM; ++d) {
        auto bcp = bc, bcm = bc;
        bcp.start_velocity(d) += eps; bcm.start_velocity(d) -= eps;
        double num = (eval_loss(T, Pvec, bcp) - eval_loss(T, Pvec, bcm)) / (2.0 * eps);
        double ana = g.start.v(d);
        cout << "start_v." << d << "  ana=" << ana << "  num=" << num
             << "  rel=" << rel_err(ana, num) << "\n";
    }
    for (int d = 0; d < DIM; ++d) {
        auto bcp = bc, bcm = bc;
        bcp.end_velocity(d) += eps; bcm.end_velocity(d) -= eps;
        double num = (eval_loss(T, Pvec, bcp) - eval_loss(T, Pvec, bcm)) / (2.0 * eps);
        double ana = g.end.v(d);
        cout << "end_v." << d << "  ana=" << ana << "  num=" << num
             << "  rel=" << rel_err(ana, num) << "\n";
    }
    for (int d = 0; d < DIM; ++d) {
        auto bcp = bc, bcm = bc;
        bcp.start_acceleration(d) += eps; bcm.start_acceleration(d) -= eps;
        double num = (eval_loss(T, Pvec, bcp) - eval_loss(T, Pvec, bcm)) / (2.0 * eps);
        double ana = g.start.a(d);
        cout << "start_a." << d << "  ana=" << ana << "  num=" << num
             << "  rel=" << rel_err(ana, num) << "\n";
    }
    for (int d = 0; d < DIM; ++d) {
        auto bcp = bc, bcm = bc;
        bcp.end_acceleration(d) += eps; bcm.end_acceleration(d) -= eps;
        double num = (eval_loss(T, Pvec, bcp) - eval_loss(T, Pvec, bcm)) / (2.0 * eps);
        double ana = g.end.a(d);
        cout << "end_a." << d << "  ana=" << ana << "  num=" << num
             << "  rel=" << rel_err(ana, num) << "\n";
    }
}

// ===================== Septic =====================
template <int DIM>
void test_septic() {
    using Spline = SplineTrajectory::SepticSplineND<DIM>;
    using Vec = typename Spline::VectorType;
    using Mat = typename Spline::MatrixType;

    cout << "\n=== SepticSplineND<" << DIM << "> gradient check ===\n";

    const double alpha_time = 0.3;
    const double eps = 1e-6;

    const int n_seg = 4;
    const int n_pts = n_seg + 1;

    std::vector<double> T(n_seg);
    T[0] = 0.8; T[1] = 1.1; T[2] = 0.9; T[3] = 1.3;

    Mat Pvec(n_pts, DIM);
    for (int i = 0; i < n_pts; ++i) {
        Vec v; v.setZero();
        for (int d = 0; d < DIM; ++d) v(d) = 0.18 * (i + 1) + 0.09 * (d + 1) * (i % 2 ? -1.0 : 1.0);
        Pvec.row(i) = v.transpose();
    }

    SplineTrajectory::BoundaryConditions<DIM> bc;
    bc.start_velocity.setConstant(0.11);
    bc.end_velocity.setConstant(-0.07);
    bc.start_acceleration.setConstant(0.04);
    bc.end_acceleration.setConstant(-0.02);
    bc.start_jerk.setConstant(0.015);
    bc.end_jerk.setConstant(-0.01);

    Spline sp(T, Pvec, 0.0, bc);

    typename Spline::MatrixType dLdC;
    Eigen::VectorXd dLdT;
    make_partials(sp, alpha_time, dLdC, dLdT);

    auto g = sp.propagateGrad(dLdC, dLdT);

    auto eval_loss = [&](const std::vector<double>& TT,
                         const Mat& PP,
                         const SplineTrajectory::BoundaryConditions<DIM>& bcc) {
        Spline tmp(TT, PP, 0.0, bcc);
        return loss_from_spline(tmp, alpha_time);
    };

    cout << std::scientific << std::setprecision(6);

    cout << "\n[Check] dL/dP (inner points)\n";
    for (int k = 1; k <= n_pts - 2; ++k) {
        for (int d = 0; d < DIM; ++d) {
            auto Pp = Pvec, Pm = Pvec;
            Pp(k, d) += eps;
            Pm(k, d) -= eps;
            double num = (eval_loss(T, Pp, bc) - eval_loss(T, Pm, bc)) / (2.0 * eps);
            double ana = g.inner_points.row(k - 1)(d);
            cout << "P[" << k << "]." << d
                 << "  ana=" << ana << "  num=" << num
                 << "  rel=" << rel_err(ana, num) << "\n";
        }
    }

    cout << "\n[Check] dL/dT\n";
    for (int i = 0; i < n_seg; ++i) {
        auto Tp = T, Tm = T;
        Tp[i] += eps;
        Tm[i] -= eps;
        double num = (eval_loss(Tp, Pvec, bc) - eval_loss(Tm, Pvec, bc)) / (2.0 * eps);
        double ana = g.times(i);
        cout << "T[" << i << "]  ana=" << ana << "  num=" << num
             << "  rel=" << rel_err(ana, num) << "\n";
    }

    cout << "\n[Check] boundary v/a/j (show only v for brevity; you can extend similarly)\n";
    for (int d = 0; d < DIM; ++d) {
        auto bcp = bc, bcm = bc;
        bcp.start_velocity(d) += eps; bcm.start_velocity(d) -= eps;
        double num = (eval_loss(T, Pvec, bcp) - eval_loss(T, Pvec, bcm)) / (2.0 * eps);
        double ana = g.start.v(d);
        cout << "start_v." << d << "  ana=" << ana << "  num=" << num
             << "  rel=" << rel_err(ana, num) << "\n";
    }
}

int main() {
    // 你可以把 DIM 改成 1/2/3 等
    test_cubic<3>();
    test_quintic<3>();
    test_septic<3>();
    return 0;
}
