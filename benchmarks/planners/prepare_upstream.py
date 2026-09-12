#!/usr/bin/env python3
"""Verify frozen sources and generate standalone numerical bindings, without changing snapshots.

Only ROS-dependent trajectory export is omitted from SUPER's MINCO translation unit.
Solver/adjoint bodies and L-BFGS sources are retained. Native quadrature gains matrix
storage templates and an additive ESDF position callback; all other penalty math remains.
"""
import hashlib
import json
from pathlib import Path
import re
import shutil
import sys


def function(text, marker, start=None):
    """Return one complete function using a lexer that ignores comments and literals."""
    position = text.index(marker) if start is None else text.index(marker, start)
    opening = text.index('{', position)
    tokens = re.compile(r'//[^\n]*|/\*[\s\S]*?\*/|"(?:\\.|[^"\\])*"|\'(?:\\.|[^\'\\])*\'|[{}]')
    depth = 0
    for token in tokens.finditer(text, opening):
        if token.group() == '{': depth += 1
        elif token.group() == '}': depth -= 1
        if depth == 0:
            return position, token.end(), text[position:token.end()]
    raise ValueError('Unbalanced function: ' + marker)


def main():
    source, output = map(Path, sys.argv[1:])
    manifest = json.loads((source/'manifest.json').read_text())
    for project, record in manifest.items():
        for name, expected in record['files'].items():
            if hashlib.sha256((source/project/name).read_bytes()).hexdigest() != expected:
                raise ValueError('Frozen source hash mismatch: ' + project + '/' + name)
    output.mkdir(parents=True, exist_ok=True)
    gsource = source/'GCOPTER/gcopter/include/gcopter'
    gtarget = output/'gcopter/frozen_gcopter'
    gtarget.mkdir(parents=True, exist_ok=True)
    for name in ['minco.hpp','trajectory.hpp','root_finder.hpp','lbfgs.hpp','flatness.hpp']:
        content = (gsource/name).read_text().replace('"gcopter/', '"frozen_gcopter/')
        (gtarget/name).write_text(content)
    text = (gsource/'gcopter.hpp').read_text()
    methods = []
    for name in ['forwardT','backwardT','backwardGradT','forwardP','backwardGradP','normRetrictionLayer','smoothedL1']:
        start = text.rfind('        static inline ', 0, text.index(name+'('))
        _, _, method = function(text, text[start:text.index(name+'(',start)+len(name)+1])
        if name in ['backwardT','backwardGradT','backwardGradP','normRetrictionLayer']:
            method = 'template<typename EIGENVEC>\n' + method
        methods.append(method)
    _, _, penalty = function(text, '        static inline void attachPenaltyFunctional(')
    penalty = penalty.replace('const Eigen::MatrixX3d &coeffs', 'const Coefficients &coeffs')
    penalty = penalty.replace('Eigen::MatrixX3d &gradC)', 'CoefficientGradient &gradC, const PositionPenalty &position_penalty)')
    penalty = penalty.replace('coeffs.block<', 'coeffs.template block<').replace('gradC.block<','gradC.template block<')
    penalty = penalty.replace('                    if (smoothedL1(violaVel', '                    pena += position_penalty(pos, gradPos);\n\n                    if (smoothedL1(violaVel')
    methods.append('template<class Coefficients, class CoefficientGradient, class PositionPenalty>\n' + penalty)
    aliases = '''using PolyhedronV = Eigen::Matrix3Xd; using PolyhedronH = Eigen::MatrixX4d;
using PolyhedraV = std::vector<PolyhedronV>; using PolyhedraH = std::vector<PolyhedronH>;
'''
    license_text = text[:text.index('#ifndef')]
    (output/'gcopter/native.hpp').write_text(license_text + '#pragma once\n#include <frozen_gcopter/flatness.hpp>\n#include <vector>\nnamespace upstream {\n'+aliases+'struct Native {\n'+ '\n'.join(methods)+'\n};\n}\n')

    ssource = source/'SUPER/super_planner'
    starget = output/'super'
    (starget/'traj_opt').mkdir(parents=True, exist_ok=True)
    (starget/'utils/optimization').mkdir(parents=True, exist_ok=True)
    header = (ssource/'include/traj_opt/minco.h').read_text()
    header = header.replace('#include <data_structure/base/trajectory.h>', 'namespace geometry_utils { class Trajectory; using StatePVAJ = Eigen::Matrix<double, 3, 4>; using StatePVA = Eigen::Matrix3d; using Mat3Df = Eigen::Matrix3Xd; }')
    (starget/'traj_opt/minco.h').write_text(header)
    implementation = (ssource/'src/utils/minco.cpp').read_text()
    for order in [2,3,4]:
        start, end, _ = function(implementation, 'void MINCO_S{}NU::getTrajectory('.format(order))
        implementation = implementation[:start] + implementation[end:]
    (starget/'minco.cpp').write_text(implementation)
    header = (ssource/'include/utils/optimization/banded_system.h').read_text()
    header = header.replace('#include <utils/optimization/root_finder.h>', '#include <Eigen/Eigen>')
    (starget/'utils/optimization/banded_system.h').write_text(header)
    shutil.copyfile(ssource/'src/utils/banded_system.cpp', starget/'banded_system.cpp')
    shutil.copyfile(ssource/'include/utils/optimization/lbfgs.h', starget/'utils/optimization/lbfgs.h')
    shutil.copyfile(ssource/'src/utils/lbfgs.cpp', starget/'lbfgs.cpp')
    shutil.copyfile(ssource/'include/utils/geometry/quadrotor_flatness.hpp', starget/'flatness.hpp')
    text = (ssource/'src/utils/optimization_utils.cpp').read_text()
    methods=[]
    for name in ['smoothedL1','forwardMapTauToT','backwardMapTToTau','propagateGradientTToTau','forwardP','backwardGradP','normRetrictionLayer']:
        candidates = list(re.finditer(r'(?:bool|void)\s+Gcopter<EIGENVEC>::'+name+r'\(', text))
        selected = None
        for match in candidates:
            _, _, candidate = function(text, match.group(), match.start())
            if name in ['forwardP','backwardGradP','normRetrictionLayer'] and 'vIdx' not in candidate[:candidate.index('{')]:
                continue
            selected = candidate
            break
        if selected is None: raise ValueError('Missing SUPER helper '+name)
        selected = selected.replace('Gcopter<EIGENVEC>::','')
        if name in ['smoothedL1','forwardMapTauToT','forwardP']:
            selected = 'static ' + selected
        else:
            selected = 'template<typename EIGENVEC>\nstatic ' + selected
        methods.append(selected)
    text = (ssource/'src/traj_opt/exp_traj_optimizer_s4.cpp').read_text()
    license_text = text[:text.index('#include')]
    _,_,penalty=function(text,'void ExpTrajOpt::constraintsFunctional(')
    penalty=penalty.replace('void ExpTrajOpt::','static void ')
    penalty=penalty.replace('const MatD3f &coeffs','const Coefficients &coeffs')
    penalty=penalty.replace('MatD3f &gradC','CoefficientGradient &gradC')
    penalty=penalty.replace('VecDf &pena_log)', 'VecDf &pena_log, const PositionPenalty &position_penalty)')
    penalty=penalty.replace('coeffs.block<','coeffs.template block<').replace('gradC.block<','gradC.template block<')
    penalty=penalty.replace('            /* 2.2  For attract point cost', '            tmp_cost += position_penalty(pos, gradPos);\n\n            /* 2.2  For attract point cost')
    methods.append('template<class Coefficients, class CoefficientGradient, class PositionPenalty>\n'+penalty)
    extra='''using VecDf = Eigen::VectorXd; using VecDi = Eigen::VectorXi;
using Vec3f = Eigen::Vector3d; using Vec4f = Eigen::Vector4d;
using MatD3f = Eigen::MatrixX3d; using Mat3Df = Eigen::Matrix3Xd;
using Vec8f = Eigen::Matrix<double, 8, 1>; using Mat83f = Eigen::Matrix<double, 8, 3>;
'''
    macros='\n'.join('#define '+name+'_IDX '+str(i) for i,name in enumerate(['POS','VEL','ACC','JER','ATT','OMG','THR'],1))
    undefs='\n'.join('#undef '+name+'_IDX' for name in ['POS','VEL','ACC','JER','ATT','OMG','THR'])
    (starget/'native.hpp').write_text(license_text+'#pragma once\n#include "flatness.hpp"\n#include <vector>\n'+macros+'\nnamespace upstream {\n'+aliases+extra+'struct Native {\nusing gcopter = Native;\n'+'\n'.join(methods)+'\n};\n}\n'+undefs+'\n')
    (output/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')


if __name__ == '__main__': main()
