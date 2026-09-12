#!/usr/bin/env python3
"""Compare the three planner integrations in an already-built optimizer workspace.

Requires the documented baseline/current binaries and an isolated ROS master.
Raw logs and archives stay in --output; the directory must not contain a previous
DiffSwarm archive run. See docs/refactor_validation.md for setup and interpretation.
"""
import argparse
import socket
import xmlrpc.client
from pathlib import Path
import csv,os,re,subprocess
parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--workspace',type=Path,default=Path(__file__).resolve().parents[2])
parser.add_argument('--output',type=Path)
parser.add_argument('--master-uri',default='http://127.0.0.1:11417')
args=parser.parse_args()
root=args.workspace.resolve(); scratch=(args.output or root/'.scratch/spline-refactor').resolve()
os.sched_setaffinity(0,{min(os.sched_getaffinity(0))})
env=os.environ.copy(); env['ROS_MASTER_URI']=args.master_uri;env['ROS_IP']='127.0.0.1';env['ROS_HOSTNAME']='127.0.0.1'
socket.setdefaulttimeout(3)
if xmlrpc.client.ServerProxy(args.master_uri).getPid('/spline_benchmark')[0] != 1:
 raise RuntimeError('The isolated ROS master is unavailable')
for label in ('old','new','minco'):
 result=subprocess.run([str(scratch/f'gcopter-solve-{label}'),'21'],stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True)
 (scratch/f'gcopter-solve-{label}.csv').write_text(result.stdout)
 if result.returncode:raise RuntimeError(f'GCOPTER {label}: {result.returncode}: {result.stdout}')
print('GCOPTER full solves passed',flush=True)
with (scratch/'super-solves.csv').open('w') as output:
 csvout=csv.writer(output);csvout.writerow(['backend','perturbation','trial','solve_us','evaluations','duration','position_violation','omega_squared_violation'])
 for label,workspace in [('old','catkin-baseline'),('new','catkin-current')]:
  ws=scratch/workspace;local=env.copy()
  local['LD_LIBRARY_PATH']=str(ws/'devel/lib')+':/opt/ros/noetic/lib'
  local['ROS_PACKAGE_PATH']=str(ws/'src')+':/opt/ros/noetic/share'
  binary=ws/'devel/lib/super_planner/backend_equivalence'
  stability=subprocess.run(['python3',str(root/'SUPER/scripts/check_backend_stability.py'),str(binary),'5e-6'],env=local,text=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,timeout=60)
  (scratch/f'super-{label}-stability.log').write_text(stability.stdout)
  if stability.returncode:raise RuntimeError(stability.stdout)
  for perturbation in (-1,0,1):
   for trial in range(11):
    result=subprocess.run([str(binary),str(perturbation),'5e-6'],env=local,text=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,timeout=60)
    text=result.stdout;(scratch/f'super-{label}-{perturbation}-{trial}.log').write_text(text)
    if result.returncode:raise RuntimeError(text)
    def scalar(pattern):return float(re.search(pattern,text).group(1))
    csvout.writerow([label,perturbation,trial,scalar(r'optimize_us=([\d.eE+-]+)'),scalar(r'iter num: (\d+)'),scalar(r'\[EQ-RESULT\].*duration=([\d.eE+-]+)'),scalar(r'\bPos: ([\d.eE+-]+)'),scalar(r'\bOmg: ([\d.eE+-]+)')]);output.flush()
print('SUPER stability and full solves passed',flush=True)
for label,build in [('old','build-diff-baseline'),('new','build-diff-current')]:
 output=scratch/f'diff-{label}-benchmark-final'
 with (scratch/f'diff-{label}-benchmark.log').open('w') as log:
  subprocess.run([str(scratch/build/'optimization_dataset_benchmark'),str(output),'31'],stdout=log,stderr=subprocess.STDOUT,check=True,timeout=120)
print('DiffSwarm full solves and archives passed',flush=True)
for binary in ('super_planner/super_spline_map_regression','gcopter/gcopter_spline_map_regression','diffswarm_ros1/diffswarm_swarm_trajectory_codec_test'):
 with (scratch/(Path(binary).name+'.log')).open('w') as log:
  subprocess.run([str(scratch/'catkin-current/devel/lib'/binary)],env=env,stdout=log,stderr=subprocess.STDOUT,check=True,timeout=120)
print('Mapped gradients and ROS codec passed',flush=True)
