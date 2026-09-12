#!/usr/bin/env python3
"""Compare frozen d504135 and current engines in Ubuntu 20.04, without changing either checkout.

The frozen harness is adapted only for degree/cost selection; its library headers remain immutable.
Build before measuring, so compiler activity cannot pollute CPU-pinned timings.
"""
import argparse
import concurrent.futures
import csv
import io
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--baseline', type=Path, required=True)
parser.add_argument('--output', type=Path, required=True)
parser.add_argument('--baseline-revision', default='unspecified; see header hashes')
parser.add_argument('--mode', choices=('build', 'run', 'all'), default='all')
parser.add_argument('--repeats', type=int, default=1000)
parser.add_argument('--solve-repeats', type=int, default=5)
parser.add_argument('--lbfgs-header', type=Path,
                    help='External GCOPTER lbfgs.hpp; required when full solves are enabled')
parser.add_argument('--jobs', type=int, default=3)
parser.add_argument('--build-labels', nargs='+', default=['minco','direct','old-direct','old','new'])
parser.add_argument('--degrees', type=int, nargs='+', default=[3, 5, 7])
parser.add_argument('--orders', type=int, nargs='+', default=[0, 3, 4])
args = parser.parse_args()
root = Path(__file__).resolve().parents[1]
base, out = args.baseline.resolve(), args.output.resolve()
lbfgs_header = (args.lbfgs_header or root.parent/'GCOPTER/gcopter/include/gcopter/lbfgs.hpp').resolve()
if args.repeats <= 0 or args.solve_repeats < 0 or args.jobs <= 0:
    parser.error('Repetitions and jobs must be positive; solve repetitions may be zero')
if args.solve_repeats and not lbfgs_header.is_file():
    parser.error('Specify --lbfgs-header, or --solve-repeats 0 to measure evaluation only')
out.mkdir(parents=True, exist_ok=True)
flags = ['-std=c++17', '-O3', '-DNDEBUG', '-fno-fast-math', '-isystem', '/usr/include/eigen3']
variants = [('minco', 1, root), ('direct', 2, root), ('old-direct', 2, base),
            ('old', 0, base), ('new', 0, root)]

# The same scalar objective (position through selected derivative) is used by all five paths.
def frozen_harness():
    source = (base/'benchmarks/benchmark_optimizer.cpp').read_text()
    source = source.replace('#ifndef BENCH_LABEL', '#ifndef BENCH_DERIVATIVE_ORDER\n#define BENCH_DERIVATIVE_ORDER 3\n#endif\n#ifndef BENCH_RECORD_SAMPLES\n#define BENCH_RECORD_SAMPLES 0\n#endif\n#ifndef BENCH_LABEL')
    source = source.replace('#else\nusing Spline = SplineTrajectory::QuinticSplineND<3>;', '#elif BENCH_DEGREE == 5\nusing Spline = SplineTrajectory::QuinticSplineND<3>;')
    source = source.replace('using Minco = minco::MINCO_S3NU;\n#endif', 'using Minco = minco::MINCO_S3NU;\n#else\nusing Spline = SplineTrajectory::CubicSplineND<3>;\nusing Minco = minco::MINCO_S2NU;\n#endif')
    source = source.replace('static constexpr int kDerivativeOrder = 3;', 'static constexpr int kDerivativeOrder = BENCH_DERIVATIVE_ORDER;')
    source = source.replace('const Eigen::Vector3d &j, const Eigen::Vector3d &,', 'const Eigen::Vector3d &j, const Eigen::Vector3d &s,')
    a = source.index('        gp = p;'); b = source.index('\n    }\n};', a)
    source = source[:a] + '''        gp = p; gv.setZero(); ga.setZero(); gj.setZero(); gs.setZero(); gt = 0;
        double cost = 0.5*p.squaredNorm();
        if constexpr (kDerivativeOrder >= 1) { gv = 0.3*v; cost += 0.15*v.squaredNorm(); }
        if constexpr (kDerivativeOrder >= 2) { ga = 0.02*a; cost += 0.01*a.squaredNorm(); }
        if constexpr (kDerivativeOrder >= 3) { gj = 0.002*j; cost += 0.001*j.squaredNorm(); }
        if constexpr (kDerivativeOrder >= 4) { gs = 0.0002*s; cost += 0.0001*s.squaredNorm(); }
        return cost;''' + source[b:]
    source = source.replace('if (!status) throw std::runtime_error("benchmark preparation failed");', 'if (!status) throw std::runtime_error("benchmark preparation failed");\n        optimizer.setRecordIntegralSamples(BENCH_RECORD_SAMPLES, context);')
    source = source.replace('bj.transpose()*gj.transpose())*common;', 'bj.transpose()*gj.transpose()+bs.transpose()*gs.transpose())*common;')
    source = source.replace('gj.dot(s))*alpha*common;', 'gj.dot(s)+gs.dot((bc*block).transpose()))*alpha*common;')
    return source

cases = [(degree, order, 0, variant) for degree in args.degrees for order in args.orders for variant in variants]
# Recording-only comparisons isolate the cost of the optional recording capability.
cases += [(degree, 3, 1, variant) for degree in args.degrees for variant in variants if variant[0] in ('old', 'new')]
def name(degree, order, record, variant):
    return '{}-d{}-r{}-record{}'.format(variant[0], degree, order, record)

if args.mode in ('build', 'all'):
    frozen = out/'frozen-benchmark.cpp'
    frozen.write_text(frozen_harness())
    def build(case):
        degree, order, record, variant = case
        label, backend, repo = variant
        binary = out/name(*case)
        source = frozen if repo == base else root/'benchmarks/benchmark_optimizer.cpp'
        command = ['g++'] + flags + ['-I'+str(repo/'include'), '-isystem', str(root/'include/gcopter'),
            '-DBENCH_BACKEND='+str(backend), '-DBENCH_DEGREE='+str(degree),
            '-DBENCH_DERIVATIVE_ORDER='+str(order), '-DBENCH_RECORD_SAMPLES='+str(record),
            '-DBENCH_LABEL="'+label+'"']
        if args.solve_repeats:
            command.append('-DBENCH_LBFGS_HEADER="'+str(lbfgs_header)+'"')
        command += [str(source), '-o', str(binary)]
        result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, universal_newlines=True)
        (out/(binary.name+'.build.log')).write_text(result.stdout)
        if result.returncode: raise RuntimeError(binary.name + ': ' + result.stdout[-3000:])
        return binary.name
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.jobs) as executor:
        for built in executor.map(build, [case for case in cases if case[3][0] in args.build_labels]): print('Built '+built, flush=True)
    for label, repo in [('old', base), ('new', root)]:
        for degree in args.degrees:
            for dimension in (1, 3):
                subprocess.run(['g++'] + flags + ['-I'+str(repo/'include'), '-DBENCH_DEGREE='+str(degree),
                    '-DBENCH_DIMENSION='+str(dimension)] + (['-DBENCH_OLD'] if label == 'old' else []) +
                    [str(root/'benchmarks/benchmark_kernel.cpp'), '-o', str(out/('kernel-{}-d{}-dim{}'.format(label, degree, dimension)))], check=True)
        subprocess.run(['g++'] + flags + ['-I'+str(repo/'include'), '-DBENCH_COMBINED', '-DBENCH_SAMPLER',
            str(repo/'benchmarks/benchmark_sampling.cpp'), '-o', str(out/('sampling-'+label))], check=True)

if args.mode in ('run', 'all'):
    allowed = sorted(os.sched_getaffinity(0))
    os.sched_setaffinity(0, {allowed[0]})
    metadata = dict(cpu=allowed[0], platform=platform.platform(), flags=flags,
        compiler=subprocess.check_output(['g++', '--version'], universal_newlines=True).splitlines()[0],
        baseline=args.baseline_revision,
        header_hashes={label: {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
            for p in (repo/'include').glob('Spline*.hpp')} for label, repo in [('old', base), ('new', root)]},
        repetitions=args.repeats, solve_repetitions=args.solve_repeats,
        lbfgs_header_sha256=hashlib.sha256(lbfgs_header.read_bytes()).hexdigest() if args.solve_repeats else None)
    (out/'environment.json').write_text(json.dumps(metadata, indent=2)+'\n')
    checks = {}
    for case in cases:
        binary = out/name(*case)
        result = subprocess.check_output([str(binary), str(args.repeats)], universal_newlines=True)
        (out/(binary.name+'.csv')).write_text(result)
        checks[tuple(case[:3]) + (case[3][0],)] = list(csv.DictReader(io.StringIO(result)))
    for degree in args.degrees:
        for order in args.orders:
            reference = checks[(degree, order, 0, 'minco')]
            for variant in variants:
                for expected, actual in zip(reference, checks[(degree, order, 0, variant[0])]):
                    for field in ('cost', 'gradient_checksum'):
                        a, b = float(expected[field]), float(actual[field])
                        if abs(a-b) > 1e-8*max(1, abs(a)): raise RuntimeError('Numeric mismatch: '+str((degree, order, variant[0], field, a, b)))
        if args.solve_repeats:
            for variant in variants:
                case = (degree, 3, 0, variant)
                binary = out/name(*case)
                result = subprocess.check_output([str(binary), str(args.solve_repeats), 'solve'], universal_newlines=True)
                (out/('solve-'+binary.name+'.csv')).write_text(result)
        print('Validated and measured degree '+str(degree), flush=True)
    for label in ('old', 'new'):
        for degree in args.degrees:
            for dimension in (1, 3):
                binary = out/('kernel-{}-d{}-dim{}'.format(label, degree, dimension))
                result = subprocess.check_output([str(binary), str(args.repeats * 5)], universal_newlines=True)
                (out/(binary.name+'.csv')).write_text(result)
        result = subprocess.check_output([str(out/('sampling-'+label))], universal_newlines=True)
        (out/('sampling-'+label+'.csv')).write_text(result)
