#!/usr/bin/env python3
"""Build C++17 planner comparisons and measure the checked-in corpus on one CPU.

All builds finish before either timing process starts. Native MINCO, direct Spline,
and SplineOptimizer use each project's own frozen L-BFGS within a comparison group.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--build', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--data', type=Path, default=Path(__file__).resolve().parent/'data')
    parser.add_argument('--mode', choices=['all','build','run'], default='all')
    parser.add_argument('--jobs', type=int, default=2)
    parser.add_argument('--repeats', type=int, default=100)
    parser.add_argument('--solve-repeats', type=int, default=5)
    parser.add_argument('--case')
    parser.add_argument('--validate-only', action='store_true')
    args = parser.parse_args()
    if min(args.jobs,args.repeats,args.solve_repeats) < 1: parser.error('Counts must be positive')
    build, output, data = args.build.resolve(), args.output.resolve(), args.data.resolve()
    output.mkdir(parents=True,exist_ok=True)
    if args.mode in ['all','build']:
        subprocess.run(['cmake','-S',str(root),'-B',str(build),'-DCMAKE_BUILD_TYPE=Release',
                        '-DCMAKE_CXX_STANDARD=17','-DSPLINE_BUILD_BENCHMARKS=ON'],check=True)
        subprocess.run(['cmake','--build',str(build),'--target','benchmark_gcopter_planner',
                        'benchmark_super_planner','-j'+str(args.jobs)],check=True)
    if args.mode == 'build': return
    cpu = min(os.sched_getaffinity(0))
    os.sched_setaffinity(0,{cpu})
    binaries={name:build/'benchmarks/planners'/('benchmark_'+name+'_planner') for name in ['gcopter','super']}
    # Record sources and binaries separately; --mode run intentionally never claims to rebuild stale binaries.
    metadata = {'started_utc':datetime.now(timezone.utc).isoformat(),
                'cpu':cpu,'platform':platform.platform(),'build_directory':str(build),
                'repeats':args.repeats,'solve_repeats':args.solve_repeats,'case':args.case,
                'validation_only':args.validate_only,
                'upstream':json.loads((root/'benchmarks/planners/upstream/manifest.json').read_text()),
                'dataset':{p.name:digest(p) for p in sorted(data.glob('*.txt'))},
                'public_headers':{p.name:digest(p) for p in sorted((root/'include').glob('Spline*.hpp'))},
                'binaries':{name:digest(path) for name,path in binaries.items()},
                'cmake_cache_sha256':digest(build/'CMakeCache.txt')}
    cache=(build/'CMakeCache.txt').read_text()
    metadata['toolchain_cache_entries']=[line for line in cache.splitlines()
        if line.startswith(('CMAKE_CXX_COMPILER:','CMAKE_CXX_FLAGS:','CMAKE_CXX_FLAGS_RELEASE:','CMAKE_CXX_STANDARD:','CMAKE_BUILD_TYPE:'))]
    compiler=next(line.split('=',1)[1] for line in cache.splitlines() if line.startswith('CMAKE_CXX_COMPILER:'))
    metadata['compiler_version']=subprocess.check_output([compiler,'--version'],universal_newlines=True).strip()
    metadata['cpu_model']=next((line.split(':',1)[1].strip() for line in Path('/proc/cpuinfo').read_text().splitlines()
        if line.startswith('model name')),platform.machine())
    metadata['compile_commands']=[entry for entry in json.loads((build/'compile_commands.json').read_text())
        if entry['file'].endswith(('planner_benchmark.cpp','/super/minco.cpp','/super/lbfgs.cpp','/super/banded_system.cpp'))]
    metadata['owned_sources']={str(p.relative_to(root)):digest(p) for p in sorted((root/'benchmarks/planners').glob('*')) if p.is_file()}
    (output/'environment.json').write_text(json.dumps(metadata,indent=2)+'\n')
    for name,binary in binaries.items():
        command=[str(binary),'--data',str(data),'--output',str(output/name),
                 '--repeats',str(args.repeats),'--solve-repeats',str(args.solve_repeats)]
        if args.case: command += ['--case',args.case]
        if args.validate_only: command.append('--validate-only')
        print('Running '+name,flush=True)
        with (output/(name+'.log')).open('w') as log:
            subprocess.run(command,stdout=log,stderr=subprocess.STDOUT,check=True)
        print('Finished '+name,flush=True)


if __name__ == '__main__': main()
