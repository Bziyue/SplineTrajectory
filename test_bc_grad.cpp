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

    typename SplineType::MatrixType gdC;
    spline.getEnergyPartialGradByCoeffs(gdC); 
    Eigen::VectorXd gdT;
    spline.getEnergyPartialGradByTimes(gdT); 

    auto grads = spline.propagateGrad(gdC, gdT, true); 

    double eps = 1e-6;

    auto verify_boundary = [&](std::string name, double analytical_val, double& target_val_ref) {
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

        double err = relativeError(analytical_val, num_grad);
        std::cout << std::left << std::setw(25) << name 
                  << " | Ana: " << std::setw(12) << analytical_val 
                  << " | Num: " << std::setw(12) << num_grad 
                  << " | Err: " << err;
        if (err < 1e-4) std::cout << " [PASS]" << std::endl;
        else std::cout << " [FAIL] <<<<<<<<<<<<<<<" << std::endl;
    };

    for (int d = 0; d < DIM; ++d) {
        verify_boundary("Start Vel [" + std::to_string(d) + "]", grads.grad_start_v(d), bc.start_velocity(d));
    }

    for (int d = 0; d < DIM; ++d) {
        verify_boundary("End Vel [" + std::to_string(d) + "]", grads.grad_end_v(d), bc.end_velocity(d));
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

    typename SplineType::MatrixType gdC;
    spline.getEnergyPartialGradByCoeffs(gdC); 
    Eigen::VectorXd gdT;
    spline.getEnergyPartialGradByTimes(gdT); 

    auto grads = spline.propagateGrad(gdC, gdT, true); 

    double eps = 1e-6;

    auto verify_boundary = [&](std::string name, double analytical_val, double& target_val_ref) {
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

        double err = relativeError(analytical_val, num_grad);
        std::cout << std::left << std::setw(25) << name 
                  << " | Ana: " << std::setw(12) << analytical_val 
                  << " | Num: " << std::setw(12) << num_grad 
                  << " | Err: " << err;
        if (err < 1e-4) std::cout << " [PASS]" << std::endl;
        else std::cout << " [FAIL] <<<<<<<<<<<<<<<" << std::endl;
    };

    for (int d = 0; d < DIM; ++d) {
        verify_boundary("Start Vel [" + std::to_string(d) + "]", grads.grad_start_va(0, d), bc.start_velocity(d));
        verify_boundary("Start Acc [" + std::to_string(d) + "]", grads.grad_start_va(1, d), bc.start_acceleration(d));
    }

    for (int d = 0; d < DIM; ++d) {
        verify_boundary("End Vel [" + std::to_string(d) + "]", grads.grad_end_va(0, d), bc.end_velocity(d));
        verify_boundary("End Acc [" + std::to_string(d) + "]", grads.grad_end_va(1, d), bc.end_acceleration(d));
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

    typename SplineType::MatrixType gdC;
    spline.getEnergyPartialGradByCoeffs(gdC);
    Eigen::VectorXd gdT;
    spline.getEnergyPartialGradByTimes(gdT);

    auto grads = spline.propagateGrad(gdC, gdT, true); 

    double eps = 1e-6;

    auto verify_boundary = [&](std::string name, double analytical_val, double& target_val_ref) {
        double original = target_val_ref;
        target_val_ref = original + eps;
        spline.update(times, points, 0.0, bc);
        double cost_p = spline.getEnergy();

        target_val_ref = original - eps;
        spline.update(times, points, 0.0, bc);
        double cost_m = spline.getEnergy();

        double num_grad = (cost_p - cost_m) / (2.0 * eps);

        target_val_ref = original;
        spline.update(times, points,0.0, bc);

        double err = relativeError(analytical_val, num_grad);
        std::cout << std::left << std::setw(25) << name 
                  << " | Ana: " << std::setw(12) << analytical_val 
                  << " | Num: " << std::setw(12) << num_grad 
                  << " | Err: " << err;
        if (err < 1e-4) std::cout << " [PASS]" << std::endl;
        else std::cout << " [FAIL] <<<<<<<<<<<<<<<" << std::endl;
    };

    for (int d = 0; d < DIM; ++d) verify_boundary("Start Vel [" + std::to_string(d) + "]", grads.grad_start_vaj(0, d), bc.start_velocity(d));
    for (int d = 0; d < DIM; ++d) verify_boundary("Start Acc [" + std::to_string(d) + "]", grads.grad_start_vaj(1, d), bc.start_acceleration(d));
    for (int d = 0; d < DIM; ++d) verify_boundary("Start Jerk [" + std::to_string(d) + "]", grads.grad_start_vaj(2, d), bc.start_jerk(d));

    for (int d = 0; d < DIM; ++d) verify_boundary("End Vel [" + std::to_string(d) + "]", grads.grad_end_vaj(0, d), bc.end_velocity(d));
    for (int d = 0; d < DIM; ++d) verify_boundary("End Acc [" + std::to_string(d) + "]", grads.grad_end_vaj(1, d), bc.end_acceleration(d));
    for (int d = 0; d < DIM; ++d) verify_boundary("End Jerk [" + std::to_string(d) + "]", grads.grad_end_vaj(2, d), bc.end_jerk(d));
}

int main() {
    srand((unsigned int) time(0));
    testCubicGradients();
    testQuinticGradients();
    testSepticGradients();
    return 0;
}