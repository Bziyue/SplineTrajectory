#include "SplineTrajectory.hpp"
#include <algorithm>
#include <chrono>
#include <iomanip>
#include <iostream>
#include <random>

using Poly = SplineTrajectory::PPolyND<3,8>;
using State = std::array<Eigen::Vector3d,4>;
using Clock = std::chrono::steady_clock;

/** @brief Alternative simultaneous Horner kernel, retained only for the algorithm comparison. */
State fused(const Poly &poly, double t)
{
    const auto &knots=poly.breakpoints();
    const int segment=std::clamp(int(std::upper_bound(knots.begin(),knots.end(),t)-knots.begin())-1,0,poly.numSegments()-1);
    const double dt=t-knots[segment];
    State state; for(auto &v:state)v.setZero();
    for(int k=7;k>=0;--k)
    {
        state[3]=state[3]*dt+3.0*state[2];
        state[2]=state[2]*dt+2.0*state[1];
        state[1]=state[1]*dt+state[0];
        state[0]=state[0]*dt+poly.coefficients().row(segment*8+k).transpose();
    }
    return state;
}

template <typename Evaluator>
void measure(const char *mode,int pieces,const char *order,const std::vector<double> &times,Evaluator &&evaluate)
{
    const auto start=Clock::now(); State first=evaluate(times.front()); const auto cold=Clock::now();
    std::vector<double> timings; double checksum=0;
    for(int batch=0;batch<21;++batch)
    {
        const auto begin=Clock::now();
        for(double t:times) {const auto state=evaluate(t); checksum+=state[0].sum()+state[3].sum();}
        timings.push_back(std::chrono::duration<double,std::nano>(Clock::now()-begin).count()/times.size());
    }
    std::sort(timings.begin(),timings.end());
    if(!first[0].allFinite()||!std::isfinite(checksum))throw std::runtime_error("invalid sample");
    std::cout<<mode<<','<<pieces<<','<<order<<','<<std::chrono::duration<double,std::micro>(cold-start).count()
             <<','<<timings[10]<<','<<timings[19]<<','<<checksum<<'\n';
}

int main()
{
    std::cout<<"mode,pieces,order,first_us,p50_ns,p95_ns,checksum\n"<<std::setprecision(17);
    for(int n:{2,8,32,128})
    {
        Poly::CoefficientMatrix coefficients(n*8,3);
        for(int i=0;i<coefficients.rows();++i)for(int j=0;j<3;++j)coefficients(i,j)=0.01*std::sin(i+0.3*j);
        std::vector<double> knots(n+1); for(int i=0;i<=n;++i)knots[i]=double(i);
        std::vector<double> times(20000); for(int i=0;i<int(times.size());++i)times[i]=double(n)*i/(times.size()-1);
        for(int shuffled=0;shuffled<2;++shuffled)
        {
            if(shuffled) {std::mt19937 random(20260912);std::shuffle(times.begin(),times.end(),random);}
            const char *order=shuffled?"random":"sequential";
            const auto polynomial = Poly::create(knots,coefficients,8);
            measure("separate",n,order,times,[&](double t){return State{polynomial.evaluate(t,0),polynomial.evaluate(t,1),polynomial.evaluate(t,2),polynomial.evaluate(t,3)};});
#ifdef BENCH_COMBINED
            measure("combined",n,order,times,[&](double t){return polynomial.evaluateDerivatives<3>(t);});
#endif
#ifdef BENCH_SAMPLER
            auto sampler=polynomial.makeSampler<3>();
            measure("sampler",n,order,times,[&](double t){return sampler.evaluateDerivatives(t);});
#endif
            measure("fused",n,order,times,[&](double t){return fused(polynomial,t);});
        }
    }
}
