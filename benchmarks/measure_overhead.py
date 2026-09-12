#!/usr/bin/env python3
"""Repeat prebuilt comparisons in alternating order, or measure clean TU compilation.

First use compare_backends.py --mode build. Runtime modes perform no compilation;
compile mode measures wall time, compiler-child peak RSS and object size on Linux.
"""
import argparse
import csv
import io
import os
from pathlib import Path
import subprocess
import time


def write_rows(path, rows):
    with path.open('w') as stream:
        writer = csv.DictWriter(stream, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=('evaluate', 'sampling', 'compile'))
    parser.add_argument('--output', type=Path, required=True,
                        help='Existing output directory from compare_backends.py')
    parser.add_argument('--baseline', type=Path, help='Required for compile mode')
    parser.add_argument('--rounds', type=int,
                        help='Defaults: 5 evaluation, 7 sampling, 3 compilation')
    parser.add_argument('--repeats', type=int, default=500)
    parser.add_argument('--degrees', type=int, nargs='+', default=[3, 5, 7])
    parser.add_argument('--orders', type=int, nargs='+', default=[0, 3, 4])
    args = parser.parse_args()
    rounds = args.rounds if args.rounds is not None else dict(evaluate=5, sampling=7, compile=3)[args.mode]
    if rounds <= 0 or args.repeats <= 0:
        parser.error('Rounds and repetitions must be positive')
    if args.mode == 'compile' and args.baseline is None:
        parser.error('--baseline is required for compile mode')
    root = Path(__file__).resolve().parents[1]
    out = args.output.resolve()
    rows = []

    if args.mode != 'compile':
        os.sched_setaffinity(0, {min(os.sched_getaffinity(0))})
    if args.mode == 'evaluate':
        for degree in args.degrees:
            for order in args.orders:
                for trial in range(rounds):
                    labels = ['old', 'new', 'direct']
                    labels = labels[trial % 3:] + labels[:trial % 3]
                    for label in labels:
                        binary = out / '{}-d{}-r{}-record0'.format(label, degree, order)
                        data = subprocess.check_output([str(binary), str(args.repeats)], universal_newlines=True)
                        for row in csv.DictReader(io.StringIO(data)):
                            row.update(trial=trial, derivative_order=order)
                            rows.append(row)
                print('Paired degree {}, derivative {}'.format(degree, order), flush=True)
        write_rows(out / 'paired.csv', rows)
    elif args.mode == 'sampling':
        for trial in range(rounds):
            for label in (('old', 'new') if trial % 2 == 0 else ('new', 'old')):
                data = subprocess.check_output([str(out / ('sampling-' + label))], universal_newlines=True)
                for row in csv.DictReader(io.StringIO(data)):
                    row.update(label=label, trial=trial)
                    rows.append(row)
        write_rows(out / 'sampling-paired.csv', rows)
    else:
        compilation = out / 'compile-cost'
        compilation.mkdir(exist_ok=True)
        for trial in range(rounds):
            for label in ('old', 'new'):
                repo = args.baseline.resolve() if label == 'old' else root
                source = out / 'frozen-benchmark.cpp' if label == 'old' else root / 'benchmarks/benchmark_optimizer.cpp'
                obj = compilation / (label + '.o')
                command = ['g++', '-std=c++17', '-O3', '-DNDEBUG', '-fno-fast-math',
                           '-DBENCH_DEGREE=7', '-DBENCH_DERIVATIVE_ORDER=3', '-DBENCH_BACKEND=0',
                           '-I' + str(repo / 'include'), '-isystem', '/usr/include/eigen3',
                           '-isystem', str(root / 'include/gcopter'), '-c', str(source), '-o', str(obj)]
                start = time.perf_counter()
                with (compilation / (label + '.log')).open('w') as log:
                    process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT)
                    _, status, usage = os.wait4(process.pid, 0)
                    process.returncode = os.WEXITSTATUS(status) if os.WIFEXITED(status) else 1
                if process.returncode:
                    raise RuntimeError('Compilation failed: ' + label)
                rows.append(dict(label=label, repeat=trial, seconds=time.perf_counter() - start,
                                 max_rss_kib=usage.ru_maxrss, object_bytes=obj.stat().st_size))
        write_rows(compilation / 'results.csv', rows)


if __name__ == '__main__':
    main()
