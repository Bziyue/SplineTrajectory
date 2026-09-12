#!/usr/bin/env python3
"""Summarize paired case medians. Solver speed ratios exclude pairs with negative termination codes."""
import argparse
from collections import defaultdict
import csv
from pathlib import Path
import statistics


def read(path):
    with path.open() as stream: return list(csv.DictReader(stream))


def med(values): return statistics.median(values)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('results',type=Path)
    args=parser.parse_args()
    print('| Project | Problem | Scale | Cases | Eval MINCO/Direct/Optimizer µs | Eval ratio | Solve MINCO/Direct/Optimizer µs | Solve ratio (stopped pairs) | Stopped MINCO/Direct/Optimizer | Cost/trajectory agreement |')
    print('| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | --- | --- |')
    diagnostics=[]
    for project in ['gcopter','super']:
        base=args.results/project
        evaluations,solves,checks,comparisons=[read(base/(name+'.csv')) for name in ['evaluations','solves','checks','comparisons']]
        for scenario in ['corridor','esdf']:
            for scale in ['small','medium','large']:
                ev=[r for r in evaluations if r['scenario']==scenario and r['scale']==scale]
                so=[r for r in solves if r['scenario']==scenario and r['scale']==scale]
                co=[r for r in comparisons if r['scenario']==scenario and r['scale']==scale and r['backend']=='spline_optimizer']
                cases=sorted(set(r['case'] for r in ev))
                if not cases: continue
                timings=defaultdict(list); solving=defaultdict(list); codes=defaultdict(list); stopped=defaultdict(set)
                for r in ev: timings[r['case'],r['backend']].append(float(r['p50_us']))
                for r in so:
                    solving[r['case'],r['backend']].append(float(r['solve_us']))
                    codes[r['case'],r['backend']].append(int(r['code']))
                for (case,backend),values in codes.items():
                    if all(code>=0 for code in values): stopped[backend].add(case)
                pair=stopped['minco'] & stopped['spline_optimizer']
                ratios=[med(solving[c,'spline_optimizer'])/med(solving[c,'minco']) for c in cases if c in pair]
                cost_agrees={case for case in cases if all(r['both_stopped']=='1' and float(r['cost_relative'])<=1e-4 for r in co if r['case']==case)}
                trajectory_agrees={case for case in cases if all(r['both_stopped']=='1' and float(r['duration_relative'])<=2e-3 and float(r['position_max'])<=0.02 for r in co if r['case']==case)}
                e=[med(med(timings[c,b]) for c in cases) for b in ['minco','spline_direct','spline_optimizer']]
                s=[med(med(solving[c,b]) for c in cases) for b in ['minco','spline_direct','spline_optimizer']]
                ratio=med(med(timings[c,'spline_optimizer'])/med(timings[c,'minco']) for c in cases)
                print('| {} | {} | {} | {} | {:.3f}/{:.3f}/{:.3f} | {:.3f} | {:.2f}/{:.2f}/{:.2f} | {} | {}/{}/{} | {}/{} |'.format(
                    project,scenario,scale,len(cases),*e,ratio,*s,format(med(ratios),'.3f') if ratios else 'n/a',
                    len(stopped['minco']),len(stopped['spline_direct']),len(stopped['spline_optimizer']),len(cost_agrees),len(trajectory_agrees)))
        diagnostics.append('\n{} maximum fixed-candidate errors: cost {:.3g}, gradient {:.3g}, coefficients {:.3g}, FD {:.3g}.\n'.format(
            project,*[max(float(r[field]) for r in checks) for field in ['cost_relative','gradient_relative','coefficient_relative','fd_relative']]))
        for backend in ['spline_direct','spline_optimizer']:
            returned=[r for r in comparisons if r['backend']==backend]
            if returned:
                diagnostics.append('{} / {} maximum same-returned-vector errors: cost {:.3g}, gradient {:.3g}, coefficients {:.3g}.\n'.format(
                    project,backend,*[max(float(r[field]) for r in returned) for field in
                        ['reference_cost_at_solution_relative','reference_gradient_at_solution_relative','reference_coefficients_at_solution_relative']]))
        failures=sorted({(r['case'],r['backend'],r['code']) for r in solves if int(r['code'])<0})
        if failures: diagnostics.append('{} negative termination codes: {}.\n'.format(project,', '.join('/'.join(item) for item in failures)))
    print(''.join(diagnostics))


if __name__ == '__main__': main()
