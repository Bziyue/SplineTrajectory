#!/usr/bin/env python3
"""Check independent headers and targeted diagnostics without linking test binaries."""
import pathlib
import subprocess
import sys
import tempfile

compiler, include, eigen = sys.argv[1:]
flags = [compiler, '-std=c++17', '-Wall', '-Wextra', '-Wpedantic', '-Werror',
         '-I' + include, '-isystem', eigen, '-fsyntax-only']
prefix = '''#include "SplineOptimizer.hpp"
using S = SplineTrajectory::QuinticSplineND<3>;
using O = SplineTrajectory::SplineOptimizer<S>;
'''
cases = []
for header in ('SplineTrajectory.hpp', 'SplineOptimizer.hpp', 'SplineConvexHull.hpp'):
    cases.append((header, '#include "' + header + '"\nint main() {}', None))
cases += [
    ('empty-objective', prefix + '''struct Objective {};
int main() { O o; Eigen::VectorXd x, g; Objective f; o.evaluate(x,g,f); }''',
     'Objective must declare a supported cost member'),
    ('duration-signature', prefix + '''struct Objective { int duration = 0; };
int main() { O o; Eigen::VectorXd x, g; Objective f; o.evaluate(x,g,f); }''',
     'Objective.duration must return double'),
    ('integral-signature', prefix + '''struct Objective { int integral = 0; };
int main() { O o; Eigen::VectorXd x, g; Objective f; o.evaluate(x,g,f); }''',
     'Objective.integral must return double'),
    ('derivative-order', prefix + '''struct Integral {
static constexpr int kDerivativeOrder = 5;
double operator()(const SplineTrajectory::IntegralPointInfo &,
const SplineTrajectory::SampleState<3> &, SplineTrajectory::SampleGradient<3> &) const { return 0; }
};
struct Objective { Integral integral; };
int main() { O o; Eigen::VectorXd x, g; Objective f; o.evaluate(x,g,f); }''',
     'kDerivativeOrder'),
    ('temporary-sampler', prefix + 'int main() { auto s = S::Polynomial{}.makeSampler<3>(); }', 'deleted'),
    ('temporary-polynomial', prefix + 'int main() { const auto &s = S{}.polynomial(); }', 'deleted'),
    ('temporary-reference', prefix + 'int main() { auto f = std::ref(SplineTrajectory::NoObjective{}); }', 'deleted'),
]
with tempfile.TemporaryDirectory(prefix='spline-compile-') as directory:
    for name, source, expected in cases:
        path = pathlib.Path(directory) / (name + '.cpp')
        path.write_text(source)
        result = subprocess.run(flags + [str(path)], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, universal_newlines=True)
        if expected is None:
            passed = result.returncode == 0
        else:
            passed = result.returncode != 0 and expected in result.stdout
        if not passed:
            sys.stderr.write(name + '\n' + result.stdout)
            sys.exit(1)
        print('{}: passed ({} diagnostic lines)'.format(name, len(result.stdout.splitlines())))
