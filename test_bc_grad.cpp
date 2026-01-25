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
}

void testQuinticGradients() {
    std::cout << "================ Testing QuinticSplineND Gradients ================" << std::endl;

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
        verify_boundary("Start Acc [" + std::to_string(d) + "]",
                         grads.start.a(d), bc_grads.start.a(d), bc.start_acceleration(d));
    }

    for (int d = 0; d < DIM; ++d) {
        verify_boundary("End Vel [" + std::to_string(d) + "]",
                         grads.end.v(d), bc_grads.end.v(d), bc.end_velocity(d));
        verify_boundary("End Acc [" + std::to_string(d) + "]",
                         grads.end.a(d), bc_grads.end.a(d), bc.end_acceleration(d));
    }
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

    // Test position gradients (start and end points)
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
}

int main() {
    srand((unsigned int) time(0));
    testCubicGradients();
    testQuinticGradients();
    testSepticGradients();
    return 0;
}