#include "SplineTrajectory.hpp"
#include <iostream>

int main()
{
    using Spline = SplineTrajectory::QuinticSplineND<3>;
    Spline::CoefficientMatrix points(3, 3);
    points << 0, 0, 0, 1, 0.5, 0, 2, 0, 0;
    const std::vector<double> durations{1.2, 1.5};
    Spline spline;
    if (const auto status = spline.update(durations, points, 0.0); !status)
    {
        std::cerr << status.message << '\n';
        return 1;
    }

    const auto &trajectory = spline.polynomial();
    auto sampler = trajectory.makeSampler<2>();
    for (int i = 0; i <= 10; ++i)
    {
        const double time = trajectory.duration() * i / 10.0;
        const auto state = sampler.evaluateDerivatives(time);
        std::cout << "t=" << time << " p=" << state[0].transpose()
                  << " v=" << state[1].transpose() << " a=" << state[2].transpose() << '\n';
    }
    const auto gradient = spline.energyGradient();
    std::cout << "energy=" << spline.energy()
              << " duration gradient=" << gradient.durations.transpose() << '\n';
    return 0;
}
