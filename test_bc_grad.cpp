#include <iostream>
#include <vector>
#include <cmath>
#include <iomanip>
#include <Eigen/Dense>

#include "SplineTrajectory.hpp" 

double relativeError(double analytical, double numerical) {
    if (std::abs(analytical) < 1e-9 && std::abs(numerical) < 1e-9) return 0.0;
    return std::abs(analytical - numerical) / std::max(1e-6, std::abs(analytical));
}

using namespace SplineTrajectory;

void testCubicGradients() {
    std::cout << "\n================ Testing CubicSplineND Gradients ================" << std::endl;

    const int DIM = 3;
    using SplineType = CubicSplineND<DIM>;

    std::vector<double> times = {1.0, 2.0, 1.5};
    SplineVector<Eigen::Matrix<double, DIM, 1>> points;
    for(size_t i=0; i<=times.size(); ++i) points.push_back(Eigen::Matrix<double, DIM, 1>::Random());

    BoundaryConditions<DIM> bc;
    bc.start_velocity = Eigen::Matrix<double, DIM, 1>::Random();
    bc.end_velocity = Eigen::Matrix<double, DIM, 1>::Random();

    SplineType spline(times, points, 0.0, bc);

    auto gdC = spline.getEnergyPartialGradByCoeffs();
    auto gdT = spline.getEnergyPartialGradByTimes();

    auto grads = spline.propagateGrad(gdC, gdT);
    auto bc_grads = spline.getEnergyGradBoundary();

    double eps = 1e-6;

    auto verify_boundary = [&](std::string name, double grad_propagate, double grad_boundary, double& target_val_ref) {
        double original = target_val_ref;

        // J(x + eps)
        target_val_ref = original + eps;
        spline.update(times, points, 0.0, bc);
        double cost_p = spline.getEnergy();

        // J(x - eps)
        target_val_ref = original - eps;
        spline.update(times, points, 0.0, bc);
        double cost_m = spline.getEnergy();

        // Central Difference
        double num_grad = (cost_p - cost_m) / (2.0 * eps);

        // Restore
        target_val_ref = original;
        spline.update(times, points, 0.0, bc);

        // Check consistency between propagateGrad and getEnergyGradBoundary
        double consistency_err = relativeError(grad_propagate, grad_boundary);
        double numerical_err = relativeError(grad_propagate, num_grad);

        std::cout << std::left << std::setw(25) << name
                  << " | Prop: " << std::setw(12) << grad_propagate
                  << " | Boundary: " << std::setw(12) << grad_boundary
                  << " | Num: " << std::setw(12) << num_grad
                  << " | CErr: " << consistency_err
                  << " | NErr: " << numerical_err;

        bool pass = consistency_err < 1e-10 && numerical_err < 1e-4;
        if (pass) std::cout << " [PASS]" << std::endl;
        else std::cout << " [FAIL] <<<<<<<<<<<<<<<" << std::endl;
    };

    // Test position gradients (start and end points)
    for (int d = 0; d < DIM; ++d) {
        verify_boundary("Start Pos [" + std::to_string(d) + "]",
                         grads.start.p(d), bc_grads.start.p(d), points[0](d));
    }

    for (int d = 0; d < DIM; ++d) {
        verify_boundary("End Pos [" + std::to_string(d) + "]",
                         grads.end.p(d), bc_grads.end.p(d), points[points.size()-1](d));
    }

    // Test velocity gradients
    for (int d = 0; d < DIM; ++d) {
        verify_boundary("Start Vel [" + std::to_string(d) + "]",
                         grads.start.v(d), bc_grads.start.v(d), bc.start_velocity(d));
    }

    for (int d = 0; d < DIM; ++d) {
        verify_boundary("End Vel [" + std::to_string(d) + "]",
                         grads.end.v(d), bc_grads.end.v(d), bc.end_velocity(d));
    }

    // --- PROOF SECTION: Cubic Spline (k=2) ---
    std::cout << "\n--- [Proof] Hierarchy: Pos->Jerk, Vel->Acc (Cubic k=2) ---" << std::endl;
    std::cout << "Formula: m = 2k-1-n, where n is boundary derivative order" << std::endl;

    auto& traj = spline.getTrajectory();

    // Start Boundary (t=0)
    // Vel (n=1) -> Acc (m=2): Grad_Vel / Acc = -2
    auto start_acc = traj.getAcc(0.0);
    auto start_jerk = traj.getJerk(0.0);

    std::cout << "\n[Start Boundary t=0]" << std::endl;
    std::cout << "Grad_V / Acc:  " << (grads.start.v.array() / start_acc.array()).transpose()
              << " (Expected: -2.0)" << std::endl;
    std::cout << "Grad_P / Jerk: " << (grads.start.p.array() / start_jerk.array()).transpose()
              << " (Expected: 2.0)" << std::endl;

    // End Boundary (t=T)
    double T = spline.getEndTime();
    auto end_acc = traj.getAcc(T);
    auto end_jerk = traj.getJerk(T);

    std::cout << "\n[End Boundary t=T]" << std::endl;
    std::cout << "Grad_V / Acc:  " << (grads.end.v.array() / end_acc.array()).transpose()
              << " (Expected: 2.0)" << std::endl;
    std::cout << "Grad_P / Jerk: " << (grads.end.p.array() / end_jerk.array()).transpose()
              << " (Expected: -2.0)" << std::endl;
}

void testQuinticGradients() {
    std::cout << "\n================ Testing QuinticSplineND Gradients ================" << std::endl;

    const int DIM = 3;
    using SplineType = QuinticSplineND<DIM>;

    std::vector<double> times = {1.0, 1.5, 0.8};

    SplineVector<Eigen::Matrix<double, DIM, 1>> points;

    for(size_t i=0; i<=times.size(); ++i) points.push_back(Eigen::Matrix<double, DIM, 1>::Random());

    BoundaryConditions<DIM> bc;
    bc.start_velocity = Eigen::Matrix<double, DIM, 1>::Random();
    bc.start_acceleration = Eigen::Matrix<double, DIM, 1>::Random();
    bc.end_velocity = Eigen::Matrix<double, DIM, 1>::Random();
    bc.end_acceleration = Eigen::Matrix<double, DIM, 1>::Random();

    SplineType spline(times, points, 0.0, bc);

    auto gdC = spline.getEnergyPartialGradByCoeffs();
    auto gdT = spline.getEnergyPartialGradByTimes();

    auto grads = spline.propagateGrad(gdC, gdT);
    auto bc_grads = spline.getEnergyGradBoundary();

    double eps = 1e-6;

    auto verify_boundary = [&](std::string name, double grad_propagate, double grad_boundary, double& target_val_ref) {
        double original = target_val_ref;

        // J(x + eps)
        target_val_ref = original + eps;
        spline.update(times, points, 0.0, bc);
        double cost_p = spline.getEnergy();

        // J(x - eps)
        target_val_ref = original - eps;
        spline.update(times, points, 0.0, bc);
        double cost_m = spline.getEnergy();

        // Central Difference
        double num_grad = (cost_p - cost_m) / (2.0 * eps);

        target_val_ref = original;
        spline.update(times, points, 0.0, bc);

        // Check consistency between propagateGrad and getEnergyGradBoundary
        double consistency_err = relativeError(grad_propagate, grad_boundary);
        double numerical_err = relativeError(grad_propagate, num_grad);

        std::cout << std::left << std::setw(25) << name
                  << " | Prop: " << std::setw(12) << grad_propagate
                  << " | Boundary: " << std::setw(12) << grad_boundary
                  << " | Num: " << std::setw(12) << num_grad
                  << " | CErr: " << consistency_err
                  << " | NErr: " << numerical_err;

        bool pass = consistency_err < 1e-10 && numerical_err < 1e-4;
        if (pass) std::cout << " [PASS]" << std::endl;
        else std::cout << " [FAIL] <<<<<<<<<<<<<<<" << std::endl;
    };

    // Test position gradients
    for (int d = 0; d < DIM; ++d) {
        verify_boundary("Start Pos [" + std::to_string(d) + "]",
                         grads.start.p(d), bc_grads.start.p(d), points[0](d));
    }

    for (int d = 0; d < DIM; ++d) {
        verify_boundary("End Pos [" + std::to_string(d) + "]",
                         grads.end.p(d), bc_grads.end.p(d), points[points.size()-1](d));
    }

    // Test velocity gradients
    for (int d = 0; d < DIM; ++d) {
        verify_boundary("Start Vel [" + std::to_string(d) + "]",
                         grads.start.v(d), bc_grads.start.v(d), bc.start_velocity(d));
        verify_boundary("Start Acc [" + std::to_string(d) + "]",
                         grads.start.a(d), bc_grads.start.a(d), bc.start_acceleration(d));
    }

    for (int d = 0; d < DIM; ++d) {
        verify_boundary("End Vel [" + std::to_string(d) + "]",
                         grads.end.v(d), bc_grads.end.v(d), bc.end_velocity(d));
        verify_boundary("End Acc [" + std::to_string(d) + "]",
                         grads.end.a(d), bc_grads.end.a(d), bc.end_acceleration(d));
    }

    // --- PROOF SECTION: Quintic Spline (k=3) ---
    std::cout << "\n--- [Proof] Hierarchy: Pos->Crackle, Vel->Snap, Acc->Jerk (Quintic k=3) ---" << std::endl;
    std::cout << "Formula: m = 2k-1-n = 5-n, where n is boundary derivative order" << std::endl;

    auto& traj = spline.getTrajectory();

    // Start Boundary (t=0)
    // Acc (n=2) -> Jerk (m=3): Grad_Acc / Jerk = -2
    // Vel (n=1) -> Snap (m=4): Grad_Vel / Snap = 2
    // Pos (n=0) -> Crackle (m=5): Grad_Pos / Crackle = -2
    auto start_jerk = traj.getJerk(0.0);
    auto start_snap = traj.getSnap(0.0);
    auto start_crackle = traj.evaluate(0.0, 5);  // 5th derivative

    std::cout << "\n[Start Boundary t=0]" << std::endl;
    std::cout << "Grad_A / Jerk:    " << (grads.start.a.array() / start_jerk.array()).transpose()
              << " (Expected: -2.0)" << std::endl;
    std::cout << "Grad_V / Snap:    " << (grads.start.v.array() / start_snap.array()).transpose()
              << " (Expected: 2.0)" << std::endl;
    std::cout << "Grad_P / Crackle: " << (grads.start.p.array() / start_crackle.array()).transpose()
              << " (Expected: -2.0)" << std::endl;

    // End Boundary (t=T)
    double T = spline.getEndTime();
    auto end_jerk = traj.getJerk(T);
    auto end_snap = traj.getSnap(T);
    auto end_crackle = traj.evaluate(T, 5);

    std::cout << "\n[End Boundary t=T]" << std::endl;
    std::cout << "Grad_A / Jerk:    " << (grads.end.a.array() / end_jerk.array()).transpose()
              << " (Expected: 2.0)" << std::endl;
    std::cout << "Grad_V / Snap:    " << (grads.end.v.array() / end_snap.array()).transpose()
              << " (Expected: -2.0)" << std::endl;
    std::cout << "Grad_P / Crackle: " << (grads.end.p.array() / end_crackle.array()).transpose()
              << " (Expected: 2.0)" << std::endl;
}

void testSepticGradients() {
    std::cout << "\n================ Testing SepticSplineND Gradients ================" << std::endl;
    
    const int DIM = 2; 
    using SplineType = SepticSplineND<DIM>;
    
    std::vector<double> times = {2.0, 1.0}; 
    
    SplineVector<Eigen::Matrix<double, DIM, 1>> points;
    
    for(size_t i=0; i<=times.size(); ++i) points.push_back(Eigen::Matrix<double, DIM, 1>::Random());

    BoundaryConditions<DIM> bc;
    bc.start_velocity = Eigen::Matrix<double, DIM, 1>::Random();
    bc.start_acceleration = Eigen::Matrix<double, DIM, 1>::Random();
    bc.start_jerk = Eigen::Matrix<double, DIM, 1>::Random();
    
    bc.end_velocity = Eigen::Matrix<double, DIM, 1>::Random();
    bc.end_acceleration = Eigen::Matrix<double, DIM, 1>::Random();
    bc.end_jerk = Eigen::Matrix<double, DIM, 1>::Random();

    SplineType spline(times, points, 0.0, bc);

    auto gdC = spline.getEnergyPartialGradByCoeffs();
    auto gdT = spline.getEnergyPartialGradByTimes();

    auto grads = spline.propagateGrad(gdC, gdT);
    auto bc_grads = spline.getEnergyGradBoundary();

    double eps = 1e-6;

    auto verify_boundary = [&](std::string name, double grad_propagate, double grad_boundary, double& target_val_ref) {
        double original = target_val_ref;
        target_val_ref = original + eps;
        spline.update(times, points, 0.0, bc);
        double cost_p = spline.getEnergy();

        target_val_ref = original - eps;
        spline.update(times, points, 0.0, bc);
        double cost_m = spline.getEnergy();

        double num_grad = (cost_p - cost_m) / (2.0 * eps);

        target_val_ref = original;
        spline.update(times, points, 0.0, bc);

        // Check consistency between propagateGrad and getEnergyGradBoundary
        double consistency_err = relativeError(grad_propagate, grad_boundary);
        double numerical_err = relativeError(grad_propagate, num_grad);

        std::cout << std::left << std::setw(25) << name
                  << " | Prop: " << std::setw(12) << grad_propagate
                  << " | Boundary: " << std::setw(12) << grad_boundary
                  << " | Num: " << std::setw(12) << num_grad
                  << " | CErr: " << consistency_err
                  << " | NErr: " << numerical_err;

        bool pass = consistency_err < 1e-10 && numerical_err < 1e-4;
        if (pass) std::cout << " [PASS]" << std::endl;
        else std::cout << " [FAIL] <<<<<<<<<<<<<<<" << std::endl;
    };

    // Test position gradients
    for (int d = 0; d < DIM; ++d) {
        verify_boundary("Start Pos [" + std::to_string(d) + "]",
                         grads.start.p(d), bc_grads.start.p(d), points[0](d));
    }

    for (int d = 0; d < DIM; ++d) {
        verify_boundary("End Pos [" + std::to_string(d) + "]",
                         grads.end.p(d), bc_grads.end.p(d), points[points.size()-1](d));
    }

    // Test velocity, acceleration, and jerk gradients
    for (int d = 0; d < DIM; ++d) verify_boundary("Start Vel [" + std::to_string(d) + "]", grads.start.v(d), bc_grads.start.v(d), bc.start_velocity(d));
    for (int d = 0; d < DIM; ++d) verify_boundary("Start Acc [" + std::to_string(d) + "]", grads.start.a(d), bc_grads.start.a(d), bc.start_acceleration(d));
    for (int d = 0; d < DIM; ++d) verify_boundary("Start Jerk [" + std::to_string(d) + "]", grads.start.j(d), bc_grads.start.j(d), bc.start_jerk(d));

    for (int d = 0; d < DIM; ++d) verify_boundary("End Vel [" + std::to_string(d) + "]", grads.end.v(d), bc_grads.end.v(d), bc.end_velocity(d));
    for (int d = 0; d < DIM; ++d) verify_boundary("End Acc [" + std::to_string(d) + "]", grads.end.a(d), bc_grads.end.a(d), bc.end_acceleration(d));
    for (int d = 0; d < DIM; ++d) verify_boundary("End Jerk [" + std::to_string(d) + "]", grads.end.j(d), bc_grads.end.j(d), bc.end_jerk(d));

    // --- PROOF SECTION: Septic Spline (k=4) ---
    std::cout << "\n--- [Proof] Hierarchy: Pos->Flick, Vel->Pop, Acc->Crackle, Jerk->Snap (Septic k=4) ---" << std::endl;
    std::cout << "Formula: m = 2k-1-n = 7-n, where n is boundary derivative order" << std::endl;

    auto& traj = spline.getTrajectory();

    // Start Boundary (t=0)
    // Jerk (n=3) -> Snap (m=4): Grad_Jerk / Snap = -2
    // Acc (n=2) -> Crackle (m=5): Grad_Acc / Crackle = 2
    // Vel (n=1) -> Pop (m=6): Grad_Vel / Pop = -2
    // Pos (n=0) -> Flick (m=7): Grad_Pos / Flick = 2
    auto start_snap = traj.getSnap(0.0);
    auto start_crackle = traj.evaluate(0.0, 5);   // 5th derivative
    auto start_pop = traj.evaluate(0.0, 6);      // 6th derivative
    auto start_flick = traj.evaluate(0.0, 7);     // 7th derivative

    std::cout << "\n[Start Boundary t=0]" << std::endl;
    std::cout << "Grad_J / Snap:     " << (grads.start.j.array() / start_snap.array()).transpose()
              << " (Expected: -2.0)" << std::endl;
    std::cout << "Grad_A / Crackle:  " << (grads.start.a.array() / start_crackle.array()).transpose()
              << " (Expected: 2.0)" << std::endl;
    std::cout << "Grad_V / Pop:      " << (grads.start.v.array() / start_pop.array()).transpose()
              << " (Expected: -2.0)" << std::endl;
    std::cout << "Grad_P / Flick:    " << (grads.start.p.array() / start_flick.array()).transpose()
              << " (Expected: 2.0)" << std::endl;

    // End Boundary (t=T)
    double T = spline.getEndTime();
    auto end_snap = traj.getSnap(T);
    auto end_crackle = traj.evaluate(T, 5);
    auto end_pop = traj.evaluate(T, 6);
    auto end_flick = traj.evaluate(T, 7);

    std::cout << "\n[End Boundary t=T]" << std::endl;
    std::cout << "Grad_J / Snap:     " << (grads.end.j.array() / end_snap.array()).transpose()
              << " (Expected: 2.0)" << std::endl;
    std::cout << "Grad_A / Crackle:  " << (grads.end.a.array() / end_crackle.array()).transpose()
              << " (Expected: -2.0)" << std::endl;
    std::cout << "Grad_V / Pop:      " << (grads.end.v.array() / end_pop.array()).transpose()
              << " (Expected: 2.0)" << std::endl;
    std::cout << "Grad_P / Flick:    " << (grads.end.p.array() / end_flick.array()).transpose()
              << " (Expected: -2.0)" << std::endl;
}

int main() {
    srand((unsigned int) time(0));
    testCubicGradients();
    testQuinticGradients();
    testSepticGradients();
    return 0;
}