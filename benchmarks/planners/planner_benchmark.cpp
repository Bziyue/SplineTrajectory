#define EIGEN_RUNTIME_NO_MALLOC
#include "PlannerEvaluators.hpp"
#include <chrono>
#include <iomanip>
#include <iostream>
#include <memory>
#include <set>

namespace pb=planner_benchmark;
using Clock=std::chrono::steady_clock;
using Polynomial=pb::Spline::Polynomial;
namespace {
struct Options {
    std::filesystem::path data,output,candidate;
    std::string case_id;
    int repeats=100,solve_repeats=5,limit=0;
    bool validate_only=false;
};
Options options(int argc,char **argv) {
    Options result;
    for(int i=1;i<argc;++i) {
        const std::string key=argv[i];
        if(key=="--validate-only") { result.validate_only=true; continue; }
        if(i+1==argc) throw std::runtime_error("Missing value for "+key);
        const std::string value=argv[++i];
        if(key=="--data") result.data=value;
        else if(key=="--output") result.output=value;
        else if(key=="--case") result.case_id=value;
        else if(key=="--candidate") result.candidate=value;
        else if(key=="--repeats") result.repeats=std::stoi(value);
        else if(key=="--solve-repeats") result.solve_repeats=std::stoi(value);
        else if(key=="--limit-per-file") result.limit=std::stoi(value);
        else throw std::runtime_error("Unknown argument "+key);
    }
    if(result.data.empty() || result.output.empty() || result.repeats<=0 || result.solve_repeats<=0 || result.limit<0)
        throw std::runtime_error("Required: --data DIR --output DIR; repetition counts must be positive");
    if(!result.candidate.empty() && (!result.validate_only || result.case_id.empty()))
        throw std::runtime_error("--candidate requires --case and --validate-only");
    return result;
}
/** @brief Replay an exported solver vector against its exact source problem. */
Eigen::VectorXd loadCandidate(const Options &opt,const pb::ProblemData &p) {
    std::ifstream stream(opt.candidate);
    std::string value;
    pb::expect(stream,"project"); stream>>value;
    if(value!=pb::project) throw std::runtime_error("Candidate project does not match executable");
    pb::expect(stream,"case"); stream>>value;
    if(value!=p.id) throw std::runtime_error("Candidate case does not match fixture");
    pb::expect(stream,"backend"); stream>>value;
    pb::expect(stream,"code"); int code=0;
    if(!(stream>>code)) throw std::runtime_error("Invalid candidate termination code");
    pb::expect(stream,"dimension"); int dimension=0;
    if(!(stream>>dimension) || dimension!=p.dimension()) throw std::runtime_error("Candidate dimension mismatch");
    pb::expect(stream,"x");
    return pb::vector(stream,dimension);
}
double elapsed(Clock::time_point start) { return std::chrono::duration<double,std::micro>(Clock::now()-start).count(); }
double relative(double a,double b) { return std::abs(a-b)/std::max({1.0,std::abs(a),std::abs(b)}); }
double maxabs(const Eigen::VectorXd &x) { return x.size()?x.cwiseAbs().maxCoeff():0; }

struct Files {
    std::ofstream checks,evaluations,solves,comparisons;
    std::filesystem::path solutions;
    explicit Files(const std::filesystem::path &path) {
        std::filesystem::create_directories(path);
        solutions=path/"solutions";
        std::filesystem::create_directories(solutions);
        checks.open(path/"checks.csv"); evaluations.open(path/"evaluations.csv");
        solves.open(path/"solves.csv"); comparisons.open(path/"comparisons.csv");
        if(!checks||!evaluations||!solves||!comparisons) throw std::runtime_error("Cannot open benchmark output");
        for(auto *file:{&checks,&evaluations,&solves,&comparisons}) *file<<std::setprecision(17);
        checks<<"project,case,scale,scenario,backend,candidate,cost_relative,gradient_relative,coefficient_relative,fd_relative\n";
        evaluations<<"project,case,scale,scenario,segments,dimension,backend,round,prepare_us,first_us,p50_us,p95_us\n";
        solves<<"project,case,scale,scenario,segments,backend,trial,solve_us,evaluations,iterations,line_search_evaluations,code,cost,gradient_inf,duration,min_esdf,corridor_violation\n";
        comparisons<<"project,case,scale,scenario,backend,trial,both_stopped,cost_relative,duration_relative,position_max,velocity_max,reference_cost_at_solution_relative,reference_gradient_at_solution_relative,reference_coefficients_at_solution_relative\n";
    }
};

/** @brief Check complete decision gradients and coefficients at identical candidates, including active norm penalties.
 * @return False for a failed saved-vector comparison. Normal corpus checks throw on failure. */
template<class Evaluator>
bool validate(const pb::ProblemData &problem,const pb::Geometry &geometry,const Eigen::VectorXd &initial,
              Files &files,bool perturb=true) {
    pb::DirectEvaluator<false> reference(problem,geometry);
    Evaluator actual(problem,geometry);
    bool passed=true;
    for(int candidate=0;candidate<(perturb?3:1);++candidate) {
        Eigen::VectorXd x=initial;
        if(candidate) for(Eigen::Index i=0;i<x.size();++i) x(i)+=(candidate==1?0.035:-0.065)*std::sin(0.7*(i+1));
        if(candidate==2) {
            x.head(problem.segments).array()-=1.0; // Activate dynamic penalties as well as mapped-domain terms.
            if(problem.usesPolytopeMap()) x.tail(problem.spatialDimension())*=1.2;
        }
        Eigen::VectorXd expected_gradient(x.size()),gradient(x.size());
        const double expected_cost=reference.evaluate(x,expected_gradient);
#ifndef NDEBUG
        if constexpr(std::is_same_v<Evaluator,pb::StructuredEvaluator>) Eigen::internal::set_is_malloc_allowed(false);
#endif
        const double cost=actual.evaluate(x,gradient);
#ifndef NDEBUG
        if constexpr(std::is_same_v<Evaluator,pb::StructuredEvaluator>) Eigen::internal::set_is_malloc_allowed(true);
#endif
        if(!std::isfinite(expected_cost)||!std::isfinite(cost)||!gradient.allFinite()||!expected_gradient.allFinite())
            throw std::runtime_error(problem.id+": nonfinite fixed-candidate evaluation");
        const auto expected_poly=reference.snapshot(),poly=actual.snapshot();
        const double cost_error=relative(cost,expected_cost);
        const double gradient_error=maxabs(gradient-expected_gradient)/std::max(1.0,maxabs(expected_gradient));
        const double coefficient_error=(poly.coefficients()-expected_poly.coefficients()).cwiseAbs().maxCoeff()/
                                      std::max(1.0,expected_poly.coefficients().cwiseAbs().maxCoeff());
        double fd_error=0;
        if(candidate!=0) {
            Eigen::VectorXd scratch(x.size()),perturbed=x,numerical(x.size());
            for(Eigen::Index i=0;i<x.size();++i) {
                const double step=2e-6*std::max(1.0,std::abs(x(i)));
                perturbed(i)=x(i)+step; const double plus=actual.evaluate(perturbed,scratch);
                perturbed(i)=x(i)-step; const double minus=actual.evaluate(perturbed,scratch);
                numerical(i)=(plus-minus)/(2*step); perturbed(i)=x(i);
            }
            if(!numerical.allFinite()) throw std::runtime_error("Nonfinite finite differences");
            fd_error=maxabs(numerical-gradient)/std::max(1.0,maxabs(gradient));
        }
        files.checks<<pb::project<<','<<problem.id<<','<<problem.scale<<','<<problem.scenario<<','<<Evaluator::label<<','
                    <<candidate<<','<<cost_error<<','<<gradient_error<<','<<coefficient_error<<','<<fd_error<<'\n';
        if(cost_error>1e-8||gradient_error>2e-7||coefficient_error>2e-8||fd_error>2e-5) {
            files.checks.flush();
            const std::string message=problem.id+" / "+Evaluator::label+": consistency failed: cost="+std::to_string(cost_error)+
                " gradient="+std::to_string(gradient_error)+" coefficients="+std::to_string(coefficient_error)+" fd="+std::to_string(fd_error);
            if(perturb) throw std::runtime_error(message);
            // A saved-vector replay reports every backend before returning failure.
            std::cerr<<message<<'\n'; passed=false;
        }
    }
    return passed;
}

template<class Evaluator>
void measure(const pb::ProblemData &p,const pb::Geometry &geometry,const Eigen::VectorXd &x,int repeats,int round,Files &files) {
    auto begin=Clock::now(); Evaluator evaluator(p,geometry); const double prepare_us=elapsed(begin);
    Eigen::VectorXd gradient(x.size());
    begin=Clock::now(); double sink=evaluator.evaluate(x,gradient); const double first_us=elapsed(begin);
    std::array<double,9> batches{};
    for(double &batch:batches) {
        begin=Clock::now();
        for(int i=0;i<repeats;++i) sink+=evaluator.evaluate(x,gradient);
        batch=elapsed(begin)/repeats;
    }
    if(!std::isfinite(sink)||!gradient.allFinite()) throw std::runtime_error("Nonfinite timing result");
    std::sort(batches.begin(),batches.end());
    files.evaluations<<pb::project<<','<<p.id<<','<<p.scale<<','<<p.scenario<<','<<p.segments<<','<<x.size()<<','<<Evaluator::label<<','
                     <<round<<','<<prepare_us<<','<<first_us<<','<<batches[4]<<','<<batches[8]<<'\n';
}

struct Solution { Polynomial poly; Eigen::VectorXd x,gradient_vector; double cost=0,gradient=0; int code=0; };

template<class Evaluator>
Solution solve(const pb::ProblemData &p,const pb::Geometry &geometry,const Eigen::VectorXd &initial,int trial,Files &files) {
    Evaluator evaluator(p,geometry);
    struct Call { Evaluator *evaluator; int evaluations=0,iterations=0,line_evaluations=0; } call{&evaluator};
    auto evaluate=[](void *ptr,const Eigen::VectorXd &x,Eigen::VectorXd &g) {
        auto &context=*static_cast<Call *>(ptr); ++context.evaluations;
        return context.evaluator->evaluate(x,g);
    };
    auto progress=[](void *ptr,const Eigen::VectorXd &,const Eigen::VectorXd &,double,double,int iteration,int lines) {
        auto &context=*static_cast<Call *>(ptr); context.iterations=iteration; context.line_evaluations+=lines; return 0;
    };
    pb::Solver::lbfgs_parameter_t parameters;
    parameters.mem_size=p.memory; parameters.past=p.past; parameters.delta=p.delta;
    parameters.g_epsilon=p.gradient_tolerance; parameters.max_iterations=p.max_iterations; parameters.min_step=p.min_step;
    Solution result; result.x=initial;
    const auto begin=Clock::now();
    result.code=pb::Solver::lbfgs_optimize(result.x,result.cost,evaluate,nullptr,progress,&call,parameters);
    const double solve_us=elapsed(begin);
    Eigen::VectorXd gradient(initial.size());
    result.cost=evaluator.evaluate(result.x,gradient); result.gradient=maxabs(gradient);
    result.gradient_vector=gradient;
    evaluator.accept(result.x); result.poly=evaluator.snapshot();
    if(!result.poly.isValid()||!std::isfinite(result.cost)||!gradient.allFinite()) throw std::runtime_error("Invalid returned solution: "+p.id);
    if(trial==0) {
        std::ofstream stream(files.solutions/(p.id+"_"+Evaluator::label+".txt"));
        if(!stream) throw std::runtime_error("Cannot save returned decision vector");
        stream<<std::setprecision(17)<<"project "<<pb::project<<"\ncase "<<p.id<<"\nbackend "<<Evaluator::label
              <<"\ncode "<<result.code<<"\ndimension "<<result.x.size()<<"\nx "<<result.x.transpose()
              <<"\nbreakpoints";
        for(double knot:result.poly.breakpoints()) stream<<' '<<knot;
        stream<<'\n';
    }
    double clearance=std::numeric_limits<double>::infinity(),corridor=0;
    for(int piece=0;piece<p.segments;++piece) {
        const auto segment=result.poly.at(piece);
        for(int k=0;k<=100;++k) {
            const auto position=segment.evaluateLocal(segment.duration()*k/100.0);
            clearance=std::min(clearance,pb::distance(p,position));
            if(p.usesPolytopeMap()) for(int axis=0;axis<3;++axis) {
                corridor=std::max(corridor,position(axis)-p.corridor_boxes[piece](axis+3));
                corridor=std::max(corridor,p.corridor_boxes[piece](axis)-position(axis));
            }
        }
    }
    files.solves<<pb::project<<','<<p.id<<','<<p.scale<<','<<p.scenario<<','<<p.segments<<','<<Evaluator::label<<','<<trial<<','
                <<solve_us<<','<<call.evaluations<<','<<call.iterations<<','<<call.line_evaluations<<','<<result.code<<','
                <<result.cost<<','<<result.gradient<<','<<result.poly.duration()<<','<<clearance<<','<<corridor<<'\n';
    return result;
}

void compare(const pb::ProblemData &p,const pb::Geometry &geometry,const Solution &reference,const Solution &actual,
             const char *label,int trial,Files &files) {
    double position=0,velocity=0;
    for(int k=0;k<=1000;++k) {
        const auto a=reference.poly.evaluateDerivatives<1>(reference.poly.duration()*k/1000.0);
        const auto b=actual.poly.evaluateDerivatives<1>(actual.poly.duration()*k/1000.0);
        position=std::max(position,(a[0]-b[0]).norm()); velocity=std::max(velocity,(a[1]-b[1]).norm());
    }
    pb::DirectEvaluator<false> native(p,geometry); Eigen::VectorXd gradient(actual.x.size());
    const double reference_at_actual=native.evaluate(actual.x,gradient);
    if(!std::isfinite(reference_at_actual)||!gradient.allFinite())
        throw std::runtime_error("Nonfinite reference evaluation at returned vector: "+p.id);
    const double recheck=relative(reference_at_actual,actual.cost);
    const double gradient_recheck=maxabs(gradient-actual.gradient_vector)/std::max(1.0,maxabs(gradient));
    const auto native_poly=native.snapshot();
    const double coefficient_recheck=(native_poly.coefficients()-actual.poly.coefficients()).cwiseAbs().maxCoeff()/
                                     std::max(1.0,native_poly.coefficients().cwiseAbs().maxCoeff());
    files.comparisons<<pb::project<<','<<p.id<<','<<p.scale<<','<<p.scenario<<','<<label<<','<<trial<<','
                     <<(reference.code>=0&&actual.code>=0)<<','<<relative(reference.cost,actual.cost)<<','
                     <<relative(reference.poly.duration(),actual.poly.duration())<<','<<position<<','<<velocity<<','<<recheck<<','
                     <<gradient_recheck<<','<<coefficient_recheck<<'\n';
    if(reference.code>=0 && actual.code>=0 && recheck>2e-7) throw std::runtime_error("Different objectives at final accepted vector: "+p.id);
}
}

int main(int argc,char **argv) try {
    const auto opt=options(argc,argv);
    Files files(opt.output);
    std::vector<std::filesystem::path> paths;
    for(const auto &entry:std::filesystem::directory_iterator(opt.data)) if(entry.path().extension()==".txt") paths.push_back(entry.path());
    std::sort(paths.begin(),paths.end());
    int completed=0;
    std::set<std::string> identifiers;
    for(const auto &path:paths) {
        auto problems=pb::loadProblems(path);
        int per_file=0;
        for(const auto &p:problems) {
            if(!identifiers.insert(p.id).second) throw std::runtime_error("Duplicate case identifier");
            if(!opt.case_id.empty() && opt.case_id!=p.id) continue;
            if(opt.limit && per_file>=opt.limit) break;
            ++per_file;
            const pb::Geometry geometry(p);
            const auto initial=opt.candidate.empty()?pb::initialGuess(p,geometry):loadCandidate(opt,p);
            bool correct=validate<pb::DirectEvaluator<false>>(p,geometry,initial,files,opt.candidate.empty());
            correct=validate<pb::DirectEvaluator<true>>(p,geometry,initial,files,opt.candidate.empty()) && correct;
            correct=validate<pb::StructuredEvaluator>(p,geometry,initial,files,opt.candidate.empty()) && correct;
            if(!correct) return 1;
            if(!opt.validate_only) {
                for(int round=0;round<3;++round) {
                    for(int offset=0;offset<3;++offset) {
                        switch((round+offset)%3) {
                        case 0: measure<pb::DirectEvaluator<false>>(p,geometry,initial,opt.repeats,round,files); break;
                        case 1: measure<pb::DirectEvaluator<true>>(p,geometry,initial,opt.repeats,round,files); break;
                        default: measure<pb::StructuredEvaluator>(p,geometry,initial,opt.repeats,round,files);
                        }
                    }
                }
                for(int trial=0;trial<opt.solve_repeats;++trial) {
                    Solution native,direct,structured;
                    for(int offset=0;offset<3;++offset) {
                        switch((trial+offset)%3) {
                        case 0: native=solve<pb::DirectEvaluator<false>>(p,geometry,initial,trial,files); break;
                        case 1: direct=solve<pb::DirectEvaluator<true>>(p,geometry,initial,trial,files); break;
                        default: structured=solve<pb::StructuredEvaluator>(p,geometry,initial,trial,files);
                        }
                    }
                    compare(p,geometry,native,direct,pb::DirectEvaluator<true>::label,trial,files);
                    compare(p,geometry,native,structured,pb::StructuredEvaluator::label,trial,files);
                }
            }
            ++completed;
            std::cout<<pb::project<<": "<<p.id<<" passed fixed-candidate checks"<<std::endl;
        }
    }
    if(!completed) throw std::runtime_error("No matching dataset cases");
    return 0;
} catch(const std::exception &error) {
    std::cerr<<pb::project<<": "<<error.what()<<'\n'; return 1;
}
