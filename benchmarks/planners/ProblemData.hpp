#pragma once
#include <Eigen/Dense>
#include <algorithm>
#include <array>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace planner_benchmark {
using Vector = Eigen::Vector3d;
using Matrix = Eigen::Matrix<double, Eigen::Dynamic, 3, Eigen::RowMajor>;
using Box = Eigen::Matrix<double, 6, 1>;
using Obstacle = Eigen::Vector4d;

/** @brief Fully specified, immutable numerical fixture loaded from a versioned text file.
 * @note Distances are metres and times seconds. No random input is generated at benchmark runtime. */
struct ProblemData {
    std::string id, scenario, scale;
    int segments = 0, integration = 0, memory = 0, past = 0, max_iterations = 0;
    double delta = 0, gradient_tolerance = 0, min_step = 0;
    double energy = 0, time_weight = 0, smooth = 0, clearance = 0, obstacle_weight = 0;
    std::array<double,6> physical{};
    Eigen::VectorXd gcopter_bounds, gcopter_weights, super_bounds, super_weights;
    std::vector<double> durations;
    Matrix waypoints;
    Eigen::Matrix3d start_derivatives, end_derivatives;
    std::vector<Box> map_boxes, corridor_boxes;
    std::vector<Obstacle> obstacles;
    bool usesPolytopeMap() const { return scenario == "corridor"; }
    int spatialDimension() const { return (segments - 1) * (usesPolytopeMap() ? 8 : 3); }
    int dimension() const { return segments + spatialDimension(); }
};

inline void expect(std::istream &stream, const std::string &expected) {
    std::string token;
    if (!(stream >> token) || token != expected)
        throw std::runtime_error("Expected dataset token: " + expected + "; got: " + token);
}
inline double number(std::istream &stream) {
    double value;
    if (!(stream >> value) || !std::isfinite(value)) throw std::runtime_error("Invalid dataset number");
    return value;
}
inline Eigen::VectorXd vector(std::istream &stream, int size) {
    Eigen::VectorXd values(size);
    for (int i = 0; i < size; ++i) values(i) = number(stream);
    return values;
}

/** @brief Load every fixture in a file, rejecting unsupported versions and malformed dimensions. */
inline std::vector<ProblemData> loadProblems(const std::filesystem::path &path) {
    std::ifstream stream(path);
    if (!stream) throw std::runtime_error("Cannot open " + path.string());
    expect(stream, "SPLINE_PLANNER_DATASET");
    int version, count;
    if (!(stream >> version) || version != 1) throw std::runtime_error("Unsupported dataset version");
    expect(stream, "cases");
    if (!(stream >> count) || count < 1 || count > 1000) throw std::runtime_error("Invalid case count");
    std::vector<ProblemData> result;
    for (int index = 0; index < count; ++index) {
        ProblemData p;
        expect(stream,"case"); stream >> p.id;
        if(p.id.empty() || p.id.size()>128 || !std::all_of(p.id.begin(),p.id.end(),[](char c) {
            return (c>='a' && c<='z') || (c>='A' && c<='Z') || (c>='0' && c<='9') || c=='_' || c=='-';
        })) throw std::runtime_error("Invalid case identifier");
        expect(stream,"scenario"); stream >> p.scenario;
        expect(stream,"scale"); stream >> p.scale;
        expect(stream,"segments"); stream >> p.segments;
        expect(stream,"integration"); stream >> p.integration;
        if (p.segments < 2 || p.segments > 256 || p.integration < 1 || p.integration > 1000 ||
            (p.scenario != "corridor" && p.scenario != "esdf")) throw std::runtime_error("Invalid problem shape/scenario");
        expect(stream,"solver"); stream >> p.memory >> p.past;
        p.delta=number(stream); p.gradient_tolerance=number(stream); stream >> p.max_iterations;
        p.min_step=number(stream);
        expect(stream,"energy"); p.energy=number(stream);
        expect(stream,"time_weight"); p.time_weight=number(stream);
        expect(stream,"smooth"); p.smooth=number(stream);
        expect(stream,"physical"); for (double &value:p.physical) value=number(stream);
        expect(stream,"gcopter_bounds"); p.gcopter_bounds=vector(stream,5);
        expect(stream,"gcopter_weights"); p.gcopter_weights=vector(stream,5);
        expect(stream,"super_bounds"); p.super_bounds=vector(stream,6);
        expect(stream,"super_weights"); p.super_weights=vector(stream,7);
        expect(stream,"esdf"); p.clearance=number(stream); p.obstacle_weight=number(stream);
        expect(stream,"start_derivatives");
        for (int column=0;column<3;++column) for (int row=0;row<3;++row) p.start_derivatives(row,column)=number(stream);
        expect(stream,"end_derivatives");
        for (int column=0;column<3;++column) for (int row=0;row<3;++row) p.end_derivatives(row,column)=number(stream);
        expect(stream,"durations"); p.durations.resize(p.segments);
        for (double &duration:p.durations) { duration=number(stream); if(duration<=0) throw std::runtime_error("Nonpositive duration"); }
        expect(stream,"waypoints"); p.waypoints.resize(p.segments+1,3);
        for (int row=0;row<=p.segments;++row) for(int column=0;column<3;++column) p.waypoints(row,column)=number(stream);
        expect(stream,"map_boxes"); p.map_boxes.resize(p.segments-1);
        for (auto &box:p.map_boxes) for(int i=0;i<6;++i) box(i)=number(stream);
        expect(stream,"corridors"); p.corridor_boxes.resize(p.segments);
        for (auto &box:p.corridor_boxes) for(int i=0;i<6;++i) box(i)=number(stream);
        expect(stream,"obstacles"); int obstacles=0;
        if(!(stream >> obstacles) || obstacles<1 || obstacles>1000) throw std::runtime_error("Invalid obstacle count");
        p.obstacles.resize(obstacles);
        for (auto &obstacle:p.obstacles) { for(int i=0;i<4;++i) obstacle(i)=number(stream); if(obstacle(3)<=0) throw std::runtime_error("Invalid sphere"); }
        expect(stream,"end");
        if (p.memory<3 || p.past<0 || p.delta<=0 || p.gradient_tolerance<0 || p.max_iterations<1 ||
            p.min_step<=0 || p.energy<0 || p.time_weight<=0 || p.smooth<=0 || p.clearance<0 || p.obstacle_weight<0)
            throw std::runtime_error("Invalid solver or cost settings");
        for(const auto &box:p.map_boxes) if((box.tail<3>().array()<=0).any()) throw std::runtime_error("Invalid map box");
        for(const auto &box:p.corridor_boxes) if((box.tail<3>().array()<=box.head<3>().array()).any()) throw std::runtime_error("Invalid corridor box");
        result.push_back(std::move(p));
    }
    std::string extra;
    if (stream >> extra) throw std::runtime_error("Unexpected trailing dataset content");
    return result;
}
}
