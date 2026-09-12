# Third-party notices

## MINVO basis solution data

The degree 0–7 MINVO conversion constants in `include/SplineConvexHull.hpp` are
derived from the solution data and ordering code in the MINVO project.

Copyright (c) 2020, Jesus Tordesillas Torres, MIT Aerospace Controls Laboratory
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice, this
   list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright notice,
   this list of conditions and the following disclaimer in the documentation
   and/or other materials provided with the distribution.

3. Neither the name of the copyright holder nor the names of its contributors
   may be used to endorse or promote products derived from this software
   without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

## Frozen planner benchmark sources

`benchmarks/planners/upstream/GCOPTER` contains unmodified numerical source snapshots from
GCOPTER commit `e0444f6d47b84f972ced91746b05feb36ce1fd4f`. Its MIT license and original
copyright notices are retained in that directory and in the files.

`benchmarks/planners/upstream/SUPER` contains unmodified numerical source snapshots from
SUPER commit `2ad3419c127a617c6d7df6925e81a14175a9c096`. Original MIT and/or
LGPL-3.0-or-later notices are retained as supplied. LGPL and GPL license texts accompany the
snapshot. The generated standalone bindings retain the corresponding notices. SUPER-derived
point-penalty adapters in `benchmarks/planners/PlannerCosts.hpp` are LGPL-3.0-or-later.

The GCOPTER L-BFGS snapshot is also used by the runnable examples; SUPER dependencies are benchmark-only. The authoritative source manifest is
[upstream/manifest.json](benchmarks/planners/upstream/manifest.json).
The [benchmark guide](benchmarks/planners/README.md) records the exact extraction and portability
adaptations, update process, and numerical verification. No original solver/adjoint algorithm
is reformatted or patched in the stored snapshots. The three public library headers retain
their existing MIT license.
