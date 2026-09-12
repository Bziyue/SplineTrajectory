#!/usr/bin/env python3
"""Regenerate the checked-in deterministic fixture corpus; benchmarks only load these files."""
import math
from pathlib import Path


def numbers(values):
    return ' '.join(format(v, '.17g') for v in values)


def main():
    directory = Path(__file__).resolve().parent/'data'
    for scenario in ['corridor', 'esdf']:
        for scale, segments in [('small',4),('medium',12),('large',32)]:
            lines=['SPLINE_PLANNER_DATASET 1', 'cases 12']
            for case in range(12):
                phase=case*0.37
                amplitude=0.25+0.035*case
                lines += ['case {}_{}_{:02d}'.format(scenario,scale,case), 'scenario '+scenario,
                          'scale '+scale, 'segments '+str(segments), 'integration 16',
                          'solver 256 3 1e-7 0 1500 1e-32', 'energy 1',
                          'time_weight '+numbers([12+case]), 'smooth 0.01',
                          'physical 1 9.81 0.1 0.2 0.001 0.0001',
                          'gcopter_bounds 3 2 0.8 5 16', 'gcopter_weights 100 10 2 2 1',
                          'super_bounds 3 4 10 2 5 16', 'super_weights 100 10 2 1 0 2 1',
                          'esdf 0.2 200',
                          'start_derivatives 0.5 0.02 0 0 0 0 0 0 0',
                          'end_derivatives 0.5 -0.02 0 0 0 0 0 0 0',
                          'durations '+numbers([0.95+0.1*math.sin(i*0.71+phase)**2 for i in range(segments)]),
                          'waypoints']
                points=[]
                for i in range(segments+1):
                    u=i/segments
                    y=amplitude*math.sin(math.pi*u)*(1+0.3*math.sin(3*math.pi*u+phase))
                    z=0.08*math.sin(2*math.pi*u)*math.sin(math.pi*u)
                    point=[0.8*i,y,z]; points.append(point)
                    lines.append(numbers(point))
                lines.append('map_boxes')
                for point in points[1:-1]: lines.append(numbers(point+[0.24,0.9,0.45]))
                lines.append('corridors')
                for i in range(segments):
                    width=0.36+0.04*math.sin(i*0.3+phase)**2
                    lower=[i*0.8-0.25,-width,-0.35]
                    upper=[(i+1)*0.8+0.25,width,0.35]
                    lines.append(numbers(lower+upper))
                lines.append('obstacles 3')
                for k,fraction in enumerate([0.27,0.5,0.73]):
                    center=[0.8*segments*fraction,0.06*math.sin(phase+k),0]
                    lines.append(numbers(center+[0.30+0.025*((case+k)%4)]))
                lines.append('end')
            (directory/(scenario+'_'+scale+'.txt')).write_text('\n'.join(lines)+'\n')


if __name__ == '__main__': main()
