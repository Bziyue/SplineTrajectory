/* Offline documentation: language visibility, task-based search and two diagrams. */
(() => {
  'use strict';
  const $ = id => document.getElementById(id);
  const svgNS = 'http://www.w3.org/2000/svg';
  const colors = ['#ad6c20', '#345fb5', '#24866e'];
  const curveNames = ['Cubic', 'Quintic', 'Septic'];
  const knots = [0, 1, 2.4, 3.5];
  let language = 'zh';
  const t = (en, zh) => language === 'zh' ? zh : en;

  // Fixed example data exported from the C++ library. No spline solver runs here.
  // scalar: piece-major ascending local-time coefficients; bezier: two quintic pieces.
  const exampleData = {
    scalar: [
      [
        [0, 0, 2.734035615039982, -1.734035615039982],
        [1, 0.26596438496001806, -2.4680712300799637, 1.1352293994215392],
        [-0.35, 0.030513809334769836, 2.299892247490501, -1.4022800974693859],
      ],
      [
        [0, 0, 0, 4.7956606760351335, -5.322257256302417, 1.5265965802672832],
        [1, 0.7309359042321495, -2.280595707036268, -1.2274025465017058, 2.3107256450340037, -0.6344517408103503],
        [-0.35, 0.3041165935085457, 2.3290914154203963, -0.7225930541941672, -2.1304365406384576, 1.1662235122806566],
      ],
      [
        [0, 0, 0, 0, 7.489300314793759, -11.304078435056041, 5.873153567375017, -1.0583754471127365],
        [1, 1.2671023583557877, -2.233563340540039, -2.6636523928307683, 2.0230710011930624, 1.7089585798266242, -1.5354745624141366, 0.2998946943191871],
        [-0.35, 0.639323661651709, 2.6550672857714237, -1.7830695874548161, -2.3552846325813714, 1.1546378737256502, 1.403493441913894, -0.764697800209106],
      ],
    ],
    bezier: [
      [[0, 0], [0, 0], [0, 0], [0.24620890022675726, 0.5682681405895691], [0.6117104828042329, 1.0089368386243383], [1.0000000000000009, 1.1999999999999988]],
      [[1, 1.2], [1.5436053240740741, 1.467488425925926], [2.131875, 1.2457500000000001], [2.4999999999999996, 0.20000000000000034], [2.4999999999999982, 0.20000000000000195], [2.4999999999999942, 0.2000000000000055]],
    ]
  };

  /** Read one language without changing or flattening the authored rich-text DOM. */
  function localizedText(node, lang) {
    if (node.nodeType === Node.TEXT_NODE) return node.textContent;
    if (node.nodeType !== Node.ELEMENT_NODE) return '';
    if (node.dataset.lang && node.dataset.lang !== lang) return '';
    if (node.classList.contains('anchor')) return '';
    return [...node.childNodes].map(child => localizedText(child, lang)).join('');
  }

  const entries = [...document.querySelectorAll('[data-search-entry]')].map(row => {
    const title = row.cells[0];
    const section = row.closest('.subsection');
    const heading = section.querySelector('h3');
    return {
      id: row.id,
      title: {en: localizedText(title, 'en'), zh: localizedText(title, 'zh')},
      context: {en: localizedText(heading, 'en'), zh: localizedText(heading, 'zh')},
      search: (row.textContent + ' ' + heading.textContent).toLowerCase()
    };
  });

  /** Filter semantic API rows; results link directly to the matching contract table. */
  function search() {
    const query = $('search-input').value.trim().toLowerCase();
    const panel = $('search-panel');
    const list = $('search-results');
    list.replaceChildren();
    panel.hidden = !query;
    $('search-input').setAttribute('aria-expanded', String(Boolean(query)));
    if (!query) return;
    const words = query.split(/\s+/);
    const matches = entries.filter(entry => words.every(word => entry.search.includes(word)));
    const score = entry => Object.values(entry.title).some(title => title.toLowerCase().includes(query)) ? 0 : 1;
    matches.sort((a, b) => score(a) - score(b));
    $('search-count').textContent = matches.length ? t(`${matches.length} matching APIs`, `${matches.length} 个匹配接口`) :
      t('No matches. Try a function name or concept.', '没有匹配结果。可输入函数名或功能关键词。');
    for (const entry of matches) {
      const item = document.createElement('li');
      const link = document.createElement('a');
      link.href = '#' + entry.id;
      const title = document.createElement('strong');
      title.textContent = entry.title[language];
      const context = document.createElement('small');
      context.textContent = entry.context[language];
      link.append(title, context);
      link.addEventListener('click', () => { closeSearch(); highlight(entry.id); });
      item.append(link);
      list.append(item);
    }
  }

  function closeSearch() {
    $('search-panel').hidden = true;
    $('search-input').setAttribute('aria-expanded', 'false');
  }

  function highlight(id) {
    const target = $(id);
    if (!target) return;
    document.querySelectorAll('.search-hit').forEach(node => node.classList.remove('search-hit'));
    target.classList.add('search-hit');
    requestAnimationFrame(() => target.scrollIntoView({block: 'center'}));
    setTimeout(() => target.classList.remove('search-hit'), 2200);
  }

  /** Toggle complete language nodes via CSS, preserving code, emphasis and formulas. */
  function setLanguage(value) {
    language = value;
    document.documentElement.lang = value === 'zh' ? 'zh-CN' : 'en';
    document.title = t('SplineTrajectory · API reference', 'SplineTrajectory · API 文档');
    $('search-input').placeholder = t('Search APIs or concepts…', '搜索接口或功能关键词…');
    document.querySelectorAll('[data-language]').forEach(button => {
      button.setAttribute('aria-pressed', String(button.dataset.language === value));
    });
    try { localStorage.setItem('spline-api-language', value); } catch (_) { /* Storage can be disabled for file URLs. */ }
    const url = new URL(location.href);
    url.searchParams.set('lang', value);
    try { history.replaceState(null, '', url); } catch (_) { /* Optional file URL persistence. */ }
    if (!$('search-panel').hidden) search();
    document.querySelectorAll('.copy-button').forEach(button => { button.textContent = t('Copy', '复制'); });
    drawCurves();
    drawHull();
  }

  /** Horner evaluation of one physical derivative of a fixed example segment. */
  function evaluatePiece(coefficients, localTime, order) {
    let result = 0;
    for (let power = coefficients.length - 1; power >= order; --power) {
      let factor = 1;
      for (let j = 0; j < order; ++j) factor *= power - j;
      result = result * localTime + coefficients[power] * factor;
    }
    return result;
  }

  function svgElement(tag, attributes = {}, value) {
    const node = document.createElementNS(svgNS, tag);
    Object.entries(attributes).forEach(([key, val]) => node.setAttribute(key, val));
    if (value !== undefined) node.textContent = value;
    return node;
  }

  /** Responsive axes use CSS-pixel-sized text, including on a narrow phone. */
  function axes(svg, bounds, titleId, title, yLabel, equalScale = false) {
    const width = Math.max(280, svg.clientWidth);
    const height = width < 500 ? 265 : 300;
    const left = 48, right = width - 13, top = 25, bottom = height - 36;
    let [xMin, xMax, yMin, yMax] = bounds;
    if (equalScale) {
      const scale = Math.max((xMax - xMin) / (right - left), (yMax - yMin) / (bottom - top));
      const centerX = (xMin + xMax) / 2, centerY = (yMin + yMax) / 2;
      xMin = centerX - scale * (right - left) / 2;
      xMax = centerX + scale * (right - left) / 2;
      yMin = centerY - scale * (bottom - top) / 2;
      yMax = centerY + scale * (bottom - top) / 2;
    }
    const x = value => left + (value - xMin) / (xMax - xMin) * (right - left);
    const y = value => bottom - (value - yMin) / (yMax - yMin) * (bottom - top);
    svg.setAttribute('viewBox', `0 0 ${width} ${height}`);
    svg.replaceChildren(svgElement('title', {id: titleId}, title));
    for (let i = 0; i <= 4; ++i) {
      const value = yMin + (yMax - yMin) * i / 4;
      svg.append(svgElement('line', {x1: left, x2: right, y1: y(value), y2: y(value), stroke: '#e7ecf2'}));
      svg.append(svgElement('text', {x: left - 7, y: y(value) + 4, 'text-anchor': 'end'}, value.toFixed(1)));
    }
    const ticks = width < 500 ? 3 : 5;
    for (let i = 0; i <= ticks; ++i) {
      const value = xMin + (xMax - xMin) * i / ticks;
      svg.append(svgElement('text', {x: x(value), y: bottom + 20, 'text-anchor': 'middle'}, value.toFixed(1)));
    }
    svg.append(svgElement('text', {x: left, y: 12}, yLabel));
    svg.append(svgElement('text', {x: right, y: height - 2, 'text-anchor': 'end'}, equalScale ? 'x' : t('time (s)', '时间 (s)')));
    return {x, y, top, bottom, left, right, width};
  }

  function linePath(points, x, y) {
    return points.map((point, i) => `${i ? 'L' : 'M'}${x(point[0]).toFixed(3)},${y(point[1]).toFixed(3)}`).join(' ');
  }

  let plotGeometry;
  function drawCurves() {
    const order = Number($('derivative-order').value);
    const time = Number($('query-time').value) / 100;
    const segment = time < knots[1] ? 0 : time < knots[2] ? 1 : 2;
    const localTime = time - knots[segment];
    const curves = exampleData.scalar.map(pieces => pieces.map((c, i) => {
      const duration = knots[i + 1] - knots[i];
      return Array.from({length: 101}, (_, j) => {
        const local = duration * j / 100;
        return [knots[i] + local, evaluatePiece(c, local, order)];
      });
    }));
    const values = curves.flat(2).map(p => p[1]);
    const min = Math.min(...values), max = Math.max(...values);
    const pad = Math.max((max - min) * .1, .15);
    const svg = $('curve-plot');
    const yLabel = ['p', 'v = dp/dt', 'a = d²p/dt²', 'jerk = d³p/dt³', 'snap = d⁴p/dt⁴'][order];
    const geometry = axes(svg, [0, 3.5, min - pad, max + pad], 'curve-title', t('Cubic, quintic and septic derivative comparison', '三次、五次与七次样条导数对照'), yLabel);
    const {x, y, top, bottom} = geometry;
    plotGeometry = geometry;
    svg.append(svgElement('rect', {x: x(knots[segment]), y: top, width: x(knots[segment + 1]) - x(knots[segment]), height: bottom - top, fill: '#345fb5', 'fill-opacity': .035}));
    for (const knot of knots) svg.append(svgElement('line', {x1: x(knot), x2: x(knot), y1: top, y2: bottom, stroke: '#b4c1d3', 'stroke-dasharray': '3 4'}));
    curves.forEach((pieces, i) => pieces.forEach(points => {
      // Separate paths preserve jumps in higher derivatives at knots.
      svg.append(svgElement('path', {d: linePath(points, x, y), fill: 'none', stroke: colors[i], 'stroke-width': 1.9, 'stroke-linejoin': 'round'}));
    }));
    if (order === 0) [0, 1, -.35, .6].forEach((value, i) => {
      svg.append(svgElement('circle', {cx: x(knots[i]), cy: y(value), r: 3.5, fill: 'white', stroke: '#5a6d86', 'stroke-width': 1.4}));
    });
    svg.append(svgElement('line', {x1: x(time), x2: x(time), y1: top, y2: bottom, stroke: '#71819a', 'stroke-width': 1}));
    const readout = $('curve-readout');
    readout.replaceChildren();
    exampleData.scalar.forEach((pieces, i) => {
      const value = evaluatePiece(pieces[segment], localTime, order);
      svg.append(svgElement('circle', {cx: x(time), cy: y(value), r: 4, fill: colors[i], stroke: '#fff', 'stroke-width': 1.5}));
      const card = document.createElement('div');
      const name = document.createElement('span');
      name.style.color = colors[i]; name.textContent = curveNames[i];
      const output = document.createElement('strong');
      output.textContent = value.toFixed(4);
      card.append(name, output); readout.append(card);
    });
    $('time-readout').textContent = t(`t=${time.toFixed(2)} s · piece ${segment} · local=${localTime.toFixed(2)} s`, `t=${time.toFixed(2)} s · 段 ${segment} · 局部 ${localTime.toFixed(2)} s`);
    $('query-time').setAttribute('aria-valuetext', $('time-readout').textContent);
  }

  /** Split a Bezier piece at u=1/2, keeping both control polygons exactly. */
  function splitBezier(points) {
    let level = points.map(point => point.slice());
    const left = [level[0]], right = [level[level.length - 1]];
    while (level.length > 1) {
      level = level.slice(0, -1).map((p, i) => [(p[0] + level[i + 1][0]) / 2, (p[1] + level[i + 1][1]) / 2]);
      left.push(level[0]); right.push(level[level.length - 1]);
    }
    return [left, right.reverse()];
  }

  function evaluateBezier(points, u) {
    let level = points.map(point => point.slice());
    while (level.length > 1) level = level.slice(0, -1).map((p, i) => [
      (1 - u) * p[0] + u * level[i + 1][0], (1 - u) * p[1] + u * level[i + 1][1]
    ]);
    return level[0];
  }

  /** Convex hull is only a display polygon, not a trajectory collision checker. */
  function convexHull(points) {
    const sorted = points.slice().sort((a, b) => a[0] - b[0] || a[1] - b[1]);
    const cross = (o, a, b) => (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0]);
    function half(input) {
      const result = [];
      for (const point of input) {
        while (result.length >= 2 && cross(result[result.length - 2], result[result.length - 1], point) <= 0) result.pop();
        result.push(point);
      }
      return result;
    }
    const lower = half(sorted), upper = half(sorted.slice().reverse());
    return lower.slice(0, -1).concat(upper.slice(0, -1));
  }

  function drawHull() {
    const depth = Number($('hull-depth').value);
    const showControls = $('show-controls').checked;
    let pieces = exampleData.bezier;
    for (let i = 0; i < depth; ++i) pieces = pieces.flatMap(splitBezier);
    const all = exampleData.bezier.flat();
    const xs = all.map(p => p[0]), ys = all.map(p => p[1]);
    const bounds = [Math.min(...xs) - .15, Math.max(...xs) + .15, Math.min(...ys) - .12, Math.max(...ys) + .12];
    const svg = $('hull-plot');
    const {x, y} = axes(svg, bounds, 'hull-title', t('Bezier subdivisions and convex bounds', 'Bezier 细分与凸包包络'), 'y', true);
    pieces.forEach((points, i) => {
      const color = colors[i % colors.length];
      svg.append(svgElement('path', {d: linePath(convexHull(points), x, y) + ' Z', fill: color, 'fill-opacity': .09, stroke: color, 'stroke-opacity': .55, 'stroke-width': 1, class: 'hull-bound'}));
      if (showControls) {
        svg.append(svgElement('path', {d: linePath(points, x, y), fill: 'none', stroke: color, 'stroke-width': .8, 'stroke-dasharray': '3 3', class: 'control-polygon'}));
        points.forEach(p => svg.append(svgElement('circle', {cx: x(p[0]), cy: y(p[1]), r: 2.4, fill: 'white', stroke: color, 'stroke-width': 1, class: 'control-point'})));
      }
    });
    exampleData.bezier.forEach(points => {
      const curve = Array.from({length: 151}, (_, i) => evaluateBezier(points, i / 150));
      svg.append(svgElement('path', {d: linePath(curve, x, y), fill: 'none', stroke: '#27384f', 'stroke-width': 2.2, class: 'source-curve'}));
    });
    $('hull-stats').textContent = t(`${pieces.length} pieces · 6 controls/piece`, `${pieces.length} 个子段 · 每段 6 个控制点`);
  }

  document.querySelectorAll('[data-language]').forEach(button => button.addEventListener('click', () => setLanguage(button.dataset.language)));
  $('search-input').addEventListener('input', search);
  $('search-input').addEventListener('focus', search);
  document.addEventListener('click', event => { if (!event.target.closest('.search')) closeSearch(); });
  document.addEventListener('keydown', event => {
    if (event.key === '/' && !event.target.closest('input,select,textarea,[contenteditable]')) {
      event.preventDefault(); $('search-input').focus();
    }
    if (event.key === 'Escape') {
      closeSearch(); $('doc-nav').classList.remove('open'); $('menu-toggle').setAttribute('aria-expanded', 'false');
      if (event.target.closest('#search-results')) $('search-input').focus();
      closeSearch();
    }
    if (event.target.closest('.search') && !$('search-panel').hidden) {
      const links = [...$('search-results').querySelectorAll('a')];
      const current = links.indexOf(document.activeElement);
      if (event.key === 'ArrowDown' && links.length) { event.preventDefault(); links[Math.min(current + 1, links.length - 1)].focus(); }
      if (event.key === 'ArrowUp') { event.preventDefault(); if (current > 0) links[current - 1].focus(); else $('search-input').focus(); }
      if (event.key === 'Enter' && event.target === $('search-input') && links.length) { event.preventDefault(); links[0].click(); }
    }
  });
  $('menu-toggle').addEventListener('click', () => {
    const open = $('doc-nav').classList.toggle('open');
    $('menu-toggle').setAttribute('aria-expanded', String(open));
  });
  document.querySelectorAll('.sidebar a').forEach(link => link.addEventListener('click', () => {
    $('doc-nav').classList.remove('open'); $('menu-toggle').setAttribute('aria-expanded', 'false');
  }));
  for (const pre of document.querySelectorAll('.example-code')) {
    const button = document.createElement('button'); button.type = 'button'; button.className = 'copy-button';
    button.addEventListener('click', async () => {
      const code = pre.querySelector('code');
      try {
        if (!navigator.clipboard) throw new Error('Unavailable clipboard');
        await navigator.clipboard.writeText(code.textContent); button.textContent = t('Copied', '已复制');
      } catch (_) {
        const range = document.createRange(); range.selectNodeContents(code);
        const selection = window.getSelection(); selection.removeAllRanges(); selection.addRange(range);
        button.textContent = t('Selected: Ctrl/Cmd+C', '已选中：Ctrl/Cmd+C');
      }
      setTimeout(() => { button.textContent = t('Copy', '复制'); }, 1600);
    });
    pre.append(button);
  }
  $('derivative-order').addEventListener('change', drawCurves);
  $('query-time').addEventListener('input', drawCurves);
  $('hull-depth').addEventListener('change', drawHull);
  $('show-controls').addEventListener('change', drawHull);
  function pointQuery(event) {
    if (!plotGeometry) return;
    const rect = $('curve-plot').getBoundingClientRect();
    const pixel = (event.clientX - rect.left) * plotGeometry.width / rect.width;
    const time = Math.max(0, Math.min(3.5, (pixel - plotGeometry.left) / (plotGeometry.right - plotGeometry.left) * 3.5));
    $('query-time').value = Math.round(time * 100);
    drawCurves();
  }
  $('curve-plot').addEventListener('pointerdown', pointQuery);
  $('curve-plot').addEventListener('pointermove', event => { if (event.buttons === 1) pointQuery(event); });
  let lastWidth = 0;
  new ResizeObserver(entries => {
    const width = entries[0].contentRect.width;
    if (Math.abs(width - lastWidth) > 1) { lastWidth = width; drawCurves(); drawHull(); }
  }).observe($('main'));
  const navLinks = [...document.querySelectorAll('.sidebar a')];
  const observer = new IntersectionObserver(entries => {
    const current = entries.filter(entry => entry.isIntersecting).sort((a, b) => a.boundingClientRect.top - b.boundingClientRect.top)[0];
    if (!current) return;
    navLinks.forEach(link => {
      const active = link.hash === '#' + current.target.id;
      link.classList.toggle('active', active);
      if (active) link.setAttribute('aria-current', 'location'); else link.removeAttribute('aria-current');
    });
  }, {rootMargin: '-125px 0px -65% 0px', threshold: 0});
  document.querySelectorAll('.module, .subsection > h3').forEach(node => observer.observe(node));
  let saved;
  try { saved = localStorage.getItem('spline-api-language'); } catch (_) { /* Optional persistence. */ }
  const requested = new URLSearchParams(location.search).get('lang');
  setLanguage(['zh', 'en'].includes(requested) ? requested : ['zh', 'en'].includes(saved) ? saved : 'zh');
})();
