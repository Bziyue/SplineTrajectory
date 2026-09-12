#!/usr/bin/env python3
"""Compile and run the README quick-start snippets with the public headers."""
import pathlib
import re
import subprocess
import sys
import tempfile

compiler, include, eigen, readme = sys.argv[1:]
blocks = re.findall(r'```cpp\n(.*?)```', pathlib.Path(readme).read_text(), re.DOTALL)
headers, body = [], []
for block in blocks:
    for line in block.splitlines():
        (headers if line.startswith('#include') else body).append(line)
source = '\n'.join(headers) + '\nint main() {\n' + '\n'.join(body)
source += '\nreturn std::isfinite(cost) && gradient.allFinite() && state[0].allFinite() ? 0 : 1;\n}\n'
with tempfile.TemporaryDirectory(prefix='spline-readme-') as temporary:
    root = pathlib.Path(temporary)
    (root / 'quickstart.cpp').write_text(source)
    subprocess.run([compiler, '-std=c++17', '-O2', '-Wall', '-Wextra', '-Werror',
                    '-I' + include, '-isystem', eigen, str(root / 'quickstart.cpp'),
                    '-o', str(root / 'quickstart')], check=True)
    subprocess.run([str(root / 'quickstart')], check=True)
