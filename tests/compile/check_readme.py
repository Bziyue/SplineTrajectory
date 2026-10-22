#!/usr/bin/env python3
"""Compile and run each complete C++ example in a README with the public headers."""
import pathlib
import re
import subprocess
import sys
import tempfile

compiler, include, eigen, readme = sys.argv[1:]
blocks = re.findall(r'```cpp\n(.*?)```', pathlib.Path(readme).read_text(), re.DOTALL)
if not blocks:
    raise SystemExit(f'No C++ examples found in {readme}')
with tempfile.TemporaryDirectory(prefix='spline-readme-') as temporary:
    root = pathlib.Path(temporary)
    for index, source in enumerate(blocks):
        path = root / f'example_{index}.cpp'
        executable = root / f'example_{index}'
        path.write_text(source)
        subprocess.run([compiler, '-std=c++17', '-O2', '-Wall', '-Wextra', '-Werror',
                        '-I' + include, '-isystem', eigen, str(path),
                        '-o', str(executable)], check=True)
        subprocess.run([str(executable)], check=True)
