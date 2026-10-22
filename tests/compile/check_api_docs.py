#!/usr/bin/env python3
"""Check API-page links and compile/run its marked complete C++ programs."""
from html.parser import HTMLParser
from pathlib import Path
import subprocess
import sys
import tempfile
from urllib.parse import unquote, urljoin, urlsplit


class ApiPage(HTMLParser):
    def __init__(self, source):
        super().__init__(convert_charrefs=True)
        self.ids = set()
        self.links = []
        self.examples = {}
        self.example = None
        self.feed(source)

    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
        if 'id' in attrs:
            ident = attrs['id']
            if ident in self.ids:
                raise ValueError(f'Duplicate HTML id: {ident}')
            self.ids.add(ident)
        if 'data-search-entry' in attrs:
            is_api_entry = tag == 'section' and 'api-entry' in attrs.get('class', '').split()
            if 'id' not in attrs or (tag != 'tr' and not is_api_entry):
                raise ValueError('Search entries must be API sections or table rows with an id')
        for key in ('href', 'src', 'xlink:href'):
            if key in attrs:
                self.links.append(attrs[key])
        if tag == 'pre' and 'data-example' in attrs:
            self.example = attrs['data-example']
            if self.example in self.examples:
                raise ValueError(f'Duplicate example: {self.example}')
            self.examples[self.example] = ''

    def handle_data(self, data):
        if self.example is not None:
            self.examples[self.example] += data

    def handle_endtag(self, tag):
        if tag == 'pre':
            self.example = None


def check_links(page, html_path, repository):
    docs = repository / 'docs'
    base = 'https://bziyue.github.io/SplineTrajectory/'
    page_url = urljoin(base, html_path.relative_to(docs).as_posix())
    for link in page.links:
        url = urlsplit(urljoin(page_url, link))
        if url.netloc.lower() == 'bziyue.github.io':
            prefix = '/SplineTrajectory/'
            if not url.path.startswith(prefix):
                raise ValueError(f'Link escapes the deployed site: {link}')
            target = docs / unquote(url.path[len(prefix):])
            if target.is_dir():
                target /= 'index.html'
            if not target.is_file():
                raise ValueError(f'Missing site file: {link}')
            if url.fragment and target.suffix == '.html':
                target_ids = page.ids if target == html_path else ApiPage(target.read_text()).ids
                if unquote(url.fragment) not in target_ids:
                    raise ValueError(f'Missing anchor: {link}')
        elif url.netloc.lower() == 'github.com':
            parts = url.path.strip('/').split('/')
            if [part.lower() for part in parts[:2]] == ['bziyue', 'splinetrajectory']:
                if len(parts) >= 4 and parts[2] in ('blob', 'tree') and parts[3] == 'main':
                    target = repository.joinpath(*map(unquote, parts[4:]))
                    if not target.exists():
                        raise ValueError(f'Missing repository link target: {link}')


def main():
    compiler, include, eigen, html = sys.argv[1:]
    html_path = Path(html).resolve()
    repository = Path(include).resolve().parent
    page = ApiPage(html_path.read_text())
    check_links(page, html_path, repository)
    if not page.examples:
        raise ValueError('No complete API examples found')
    solver_include = repository / 'benchmarks/planners/upstream/GCOPTER/gcopter/include'
    with tempfile.TemporaryDirectory(prefix='api-docs-', dir=Path.cwd()) as temporary:
        root = Path(temporary)
        for index, (name, source) in enumerate(page.examples.items()):
            path = root / f'example_{index}.cpp'
            executable = root / f'example_{index}'
            path.write_text(source)
            print(f'Compile and run: {name}', flush=True)
            subprocess.run([compiler, '-std=c++17', '-O2', '-Wall', '-Wextra', '-Werror',
                            '-I' + include, '-isystem', eigen, '-isystem', str(solver_include),
                            str(path), '-o', str(executable)], check=True, timeout=120)
            subprocess.run([str(executable)], check=True, timeout=30)
    print(f'{len(page.examples)} programs passed; anchors, assets and repository links checked.')


if __name__ == '__main__':
    main()
