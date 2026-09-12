#!/usr/bin/env python3
"""Build/run immutable historical and current backends with identical numerical inputs.

Run inside the documented Ubuntu 20.04 container. Baselines must be existing detached
worktrees; this tool never checks out, resets or modifies a consumer repository.
"""
import argparse
import csv
import io
import json
import os
from pathlib import Path
import platform
import subprocess

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--workspace', type=Path, default=Path(__file__).resolve().parents[2])
parser.add_argument('--baselines', type=Path)
parser.add_argument('--output', type=Path)
parser.add_argument('--mode', choices=('build', 'run', 'all'), default='all')
parser.add_argument('--repeats', type=int, default=2000)
parser.add_argument('--solve-repeats', type=int, default=11)
args = parser.parse_args()
root = args.workspace.resolve()
base = (args.baselines or root / '.scratch/spline-refactor').resolve()
out = (args.output or base).resolve()
out.mkdir(parents=True, exist_ok=True)
flags = ['-std=c++17', '-O3', '-DNDEBUG', '-fno-fast-math', '-I/usr/include/eigen3']
variants = [('minco', 1, base/'gcopter-minco/gcopter/include', False),
            ('direct', 2, root/'SplineTrajectory/include', False),
            ('old-main', 0, base/'spline-main/include', True),
            ('new', 0, root/'SplineTrajectory/include', False),
            ('old-vendored', 0, base/'gcopter-current/gcopter/include/SplineTrajectory', True)]
if args.mode in ('build', 'all'):
    for degree in (5, 7):
        for label, backend, headers, legacy in variants:
            command = ['g++'] + flags + [f'-DBENCH_BACKEND={backend}', f'-DBENCH_DEGREE={degree}',
                f'-DBENCH_LABEL="{label}"', '-I'+str(headers),
                '-I'+str(base/'gcopter-current/gcopter/include'), '-I'+str(root/'SplineTrajectory/include'),
                f'-DBENCH_LBFGS_HEADER="{root}/GCOPTER/gcopter/include/gcopter/lbfgs.hpp"']
            if legacy:
                command.append('-DBENCH_LEGACY_CONTEXT')
            command += [str(root/'SplineTrajectory/benchmarks/benchmark_optimizer.cpp'),
                        '-o', str(out/f'bench-{label}-{degree}')]
            subprocess.run(command, check=True)
    for label, headers, macros in [
        ('old-main', base/'spline-main/include', []),
        ('old-diff', base/'diff-current/include/DiffSwarm/planning/backend/spline', ['BENCH_COMBINED']),
        ('new', root/'SplineTrajectory/include', ['BENCH_COMBINED', 'BENCH_SAMPLER'])]:
        subprocess.run(['g++'] + flags + ['-I'+str(headers)] + ['-D'+m for m in macros] +
            [str(root/'SplineTrajectory/benchmarks/benchmark_sampling.cpp'), '-o', str(out/f'sampling-{label}')], check=True)

if args.mode in ('run', 'all'):
    # One allowed CPU and serial execution keep comparisons on the same execution resource.
    allowed = sorted(os.sched_getaffinity(0))
    os.sched_setaffinity(0, {allowed[0]})
    metadata = dict(cpu=allowed[0], platform=platform.platform(), flags=flags,
                    compiler=subprocess.check_output(['g++', '--version'], text=True).splitlines()[0],
                    repetitions=args.repeats, solve_repetitions=args.solve_repeats)
    (out/'comparison-environment.json').write_text(json.dumps(metadata, indent=2)+'\n')
    for degree in (5, 7):
        checks = {}
        for label in ('old-vendored', 'new', 'minco', 'direct', 'old-main'):
            binary = out/f'bench-{label}-{degree}'
            result = subprocess.check_output([str(binary), str(args.repeats)], text=True)
            (out/f'eval-{label}-{degree}.csv').write_text(result)
            checks[label] = list(csv.DictReader(io.StringIO(result)))
            solved = subprocess.check_output([str(binary), str(args.solve_repeats), 'solve'], text=True)
            (out/f'solve-{label}-{degree}.csv').write_text(solved)
        for i, reference in enumerate(checks['minco']):
            for label, rows in checks.items():
                for field in ('cost', 'gradient_checksum'):
                    expected, actual = float(reference[field]), float(rows[i][field])
                    if abs(expected-actual) > 1e-8 * max(1, abs(expected)):
                        raise RuntimeError(f'{label} degree {degree}: {field} differs from MINCO')
        print(f'Validated and measured degree {degree}', flush=True)
    for label in ('old-main', 'old-diff', 'new'):
        result = subprocess.check_output([str(out/f'sampling-{label}')], text=True)
        (out/f'sampling-{label}.csv').write_text(result)
    print('Sampling measured; first-use costs are separate CSV columns.', flush=True)
