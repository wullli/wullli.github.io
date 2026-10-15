#!/usr/bin/env node
'use strict';

const fs   = require('fs');
const path = require('path');
const vm   = require('vm');
const { Marked } = require('marked');
const { markedHighlight } = require('marked-highlight');
const hljs = require('highlight.js');
const markedKatex = require('marked-katex-extension');

// Highlight fenced code at build time; colours come from --hl-* tokens in style.css
const marked = new Marked(markedHighlight({
  emptyLangClass: 'hljs',
  langPrefix: 'hljs language-',
  highlight(code, lang) {
    const language = hljs.getLanguage(lang) ? lang : 'plaintext';
    return hljs.highlight(code, { language }).value;
  }
}));
// ```python fold-imports: leading import lines go in a collapsed <details> above
// the rest; the copy button (code.js) still copies the whole block.
// Registered after markedHighlight, so this walkTokens runs before it.
const IMPORT_LINE = /^\s*(import\s|from\s+\S+\s+import\s|#|$)/;
marked.use({
  walkTokens(token) {
    if (token.type !== 'code' || !/(^|\s)fold-imports(\s|$)/.test(token.lang || '')) return;
    const lines = token.text.split('\n');
    let n = 0;
    while (n < lines.length && IMPORT_LINE.test(lines[n])) n++;
    while (n > 0 && !lines[n - 1].trim()) n--;  // trailing blank lines stay visible
    if (!n || n === lines.length) return;
    const lang = token.lang.match(/\S*/)[0];
    const language = hljs.getLanguage(lang) ? lang : 'plaintext';
    token.foldCount = n;
    token.foldHtml  = hljs.highlight(lines.slice(0, n).join('\n'), { language }).value;
    token.text      = lines.slice(n).join('\n').replace(/^\n+/, '');
  },
  renderer: {
    code(token) {
      if (!token.foldHtml) return false;
      const cls = 'hljs language-' + token.lang.match(/\S*/)[0];
      return '<div class="code-fold">' +
        '<details><summary>' + token.foldCount + ' import lines</summary>' +
        '<pre><code class="' + cls + '">' + token.foldHtml + '\n</code></pre></details>' +
        '<pre><code class="' + cls + '">' + token.text.replace(/\n$/, '') + '\n</code></pre>' +
        '</div>\n';
    }
  }
});
// LaTeX at build time: $inline$ and $$display$$; CSS/fonts copied to _site/katex/
marked.use(markedKatex({ throwOnError: false }));
const KATEX_CSS = 'katex/katex.min.css';
const hasMath = html => html.includes('class="katex');
// Adds the KaTeX stylesheet to <head>; prefix is the path back to the site root
function withKatexCss(html, prefix) {
  return html.replace('</head>', () => '  <link rel="stylesheet" href="' + prefix + KATEX_CSS + '">\n</head>');
}

const SITE_URL  = 'https://wullli.github.io';
const POSTS_DIR = 'posts';
// --drafts (used by serve.js) includes posts marked draft: true; deploys omit them
const SHOW_DRAFTS = process.argv.includes('--drafts');

// Load pubs.js globals into this context (pure string functions, no DOM calls)
vm.runInThisContext(fs.readFileSync('./pubs.js', 'utf8'));

const bibRaw  = fs.readFileSync('./publications.bib', 'utf8');
const yamlRaw = fs.readFileSync('./pub_links.yaml',   'utf8');

const entries = parseBibTeX(bibRaw);
const linkMap = parseSimpleYAML(yamlRaw);

function sortedEntries(ents) {
  return ents
    .map((e, i) => ({ e, i }))
    .sort((a, b) => {
      const diff = (parseInt(clean(b.e.entryTags.YEAR), 10) || 0)
                 - (parseInt(clean(a.e.entryTags.YEAR), 10) || 0);
      return diff !== 0 ? diff : a.i - b.i;
    })
    .map(x => x.e);
}

function renderYearGroup(year, ents, startNum) {
  return [
    '<div class="pub-year-group">',
    '  <div class="pub-year-label">' + esc(year) + '</div>',
    ents.map((e, i) => renderEntry(e, linkMap[e.citationKey], startNum + i)).join('\n'),
    '</div>'
  ].join('\n');
}

function renderAllPubs(ents) {
  const byYear = Object.create(null);
  ents.forEach(e => {
    const y = clean(e.entryTags.YEAR) || 'n.d.';
    (byYear[y] = byYear[y] || []).push(e);
  });
  const years = Object.keys(byYear).sort((a, b) => Number(b) - Number(a));
  let idx = 1;
  return years.map(y => {
    const html = renderYearGroup(y, byYear[y], idx);
    idx += byYear[y].length;
    return html;
  }).join('');
}

function renderRecentPubs(ents) {
  const top3 = ents.slice(0, 3);
  let abstractShown = false;
  return top3.map((e, i) => {
    const num = i + 1;
    if (!abstractShown) {
      const abs = clean(e.entryTags.ABSTRACT || '');
      if (abs) {
        abstractShown = true;
        const absTitle  = clean(e.entryTags.TITLE || '');
        const absUrl    = primaryUrl(e.entryTags);
        const absTitleLinked = absUrl
          ? '<a href="' + esc(absUrl) + '" target="_blank" rel="noopener">' + esc(absTitle) + '</a>'
          : esc(absTitle);
        const absHeading =
          '    <p class="pub-abstract-heading"><span class="pub-abstract-title">' + absTitleLinked +
          '</span> <span class="pub-abstract-label">&middot; Abstract</span></p>\n' +
          '    <p class="pub-abstract">' + esc(abs) + '</p>\n';
        let h = renderEntry(e, linkMap[e.citationKey], num);
        h = h.replace(/    <p class="pub-title">[\s\S]*?<\/p>\n/, absHeading);
        return h;
      }
    }
    return renderEntry(e, linkMap[e.citationKey], num);
  }).join('');
}

function inject(html, startMarker, endMarker, content) {
  const s = html.indexOf(startMarker);
  const e = html.indexOf(endMarker);
  if (s === -1 || e === -1) throw new Error('Marker not found: ' + startMarker);
  return html.slice(0, s + startMarker.length) + content + html.slice(e);
}

// --- Blog posts ---

// Parses a leading "---\nkey: value\n---" block. Returns { meta, body }.
function parseFrontMatter(src, file) {
  const m = src.match(/^---\r?\n([\s\S]*?)\r?\n---\r?\n?/);
  if (!m) throw new Error(file + ': missing front matter (--- title/date/summary ---)');
  const meta = {};
  m[1].split(/\r?\n/).forEach(line => {
    const kv = line.match(/^(\w+):\s*(.*)$/);
    if (kv) meta[kv[1]] = kv[2].replace(/^(["'])(.*)\1$/, '$2').trim();
  });
  return { meta, body: src.slice(m[0].length) };
}

// Reads one Markdown file. Missing date/tags fall back to `defaults` (a series,
// for its chapters).
function loadMarkdown(file, defaults = {}) {
  const { meta, body } = parseFrontMatter(fs.readFileSync(file, 'utf8'), file);
  if (!meta.title) throw new Error(file + ': front matter needs "title"');
  const date = meta.date || defaults.date;
  if (!/^\d{4}-\d{2}-\d{2}$/.test(date || '')) {
    throw new Error(file + ': front matter needs "date: YYYY-MM-DD"');
  }
  return {
    title:   meta.title,
    date,
    summary: meta.summary || '',
    tags:    meta.tags ? meta.tags.split(',').map(t => t.trim()).filter(Boolean) : (defaults.tags || []),
    draft:   meta.draft === 'true',
    html:    marked.parse(body)
  };
}

// posts/<slug>.md → posts/<slug>.html
function loadPost(f) {
  const slug = f.replace(/\.md$/, '');
  return { kind: 'post', slug, url: POSTS_DIR + '/' + slug + '.html', sortDate: null,
           ...loadMarkdown(path.join(POSTS_DIR, f)) };
}

// posts/<slug>/ holds _series.md (title, date, summary, tags, intro text) and one
// file per chapter, ordered by filename. Builds posts/<slug>.html (overview),
// posts/<slug>/<chapter>.html and posts/<slug>/all.html (every chapter on one page).
function loadSeries(dir) {
  const base = path.join(POSTS_DIR, dir);
  const indexFile = path.join(base, '_series.md');
  if (!fs.existsSync(indexFile)) throw new Error(base + ': series folder needs _series.md');
  const series = loadMarkdown(indexFile);
  const chapters = fs.readdirSync(base)
    .filter(f => f.endsWith('.md') && !f.startsWith('_'))
    .sort()
    .map(f => {
      if (f === 'all.md') throw new Error(base + ': "all" is reserved for the one-page view');
      const slug = f.replace(/\.md$/, '');
      return { slug, url: POSTS_DIR + '/' + dir + '/' + slug + '.html',
               ...loadMarkdown(path.join(base, f), series) };
    })
    .filter(c => SHOW_DRAFTS || !c.draft);
  if (!chapters.length) return null;
  chapters.forEach((c, i) => { c.num = i + 1; });
  return {
    kind: 'series', slug: dir, url: POSTS_DIR + '/' + dir + '.html',
    allUrl: POSTS_DIR + '/' + dir + '/all.html', ...series, chapters,
    // A new chapter moves the series back to the top of the blog
    sortDate: chapters.reduce((d, c) => c.date > d ? c.date : d, series.date)
  };
}

// Posts and series, newest first
function loadEntries() {
  if (!fs.existsSync(POSTS_DIR)) return [];
  return fs.readdirSync(POSTS_DIR, { withFileTypes: true })
    .filter(e => !e.name.startsWith('_') && !e.name.startsWith('.'))
    .map(e => e.isDirectory() ? loadSeries(e.name) : e.name.endsWith('.md') ? loadPost(e.name) : null)
    .filter(p => p && (SHOW_DRAFTS || !p.draft))
    .map(p => ({ ...p, sortDate: p.sortDate || p.date }))
    .sort((a, b) => b.sortDate.localeCompare(a.sortDate));
}

function formatDate(iso) {
  const [y, m, d] = iso.split('-').map(Number);
  return new Date(Date.UTC(y, m - 1, d))
    .toLocaleDateString('en-GB', { day: 'numeric', month: 'long', year: 'numeric', timeZone: 'UTC' });
}

function draftBadge(p) {
  return p.draft ? ' <span class="draft-badge">Draft</span>' : '';
}

function partsBadge(s) {
  return ' <span class="series-badge">' + s.chapters.length + ' part' +
    (s.chapters.length === 1 ? '' : 's') + '</span>';
}

function renderTags(tags) {
  if (!tags.length) return '';
  return '<div class="blog-tags">' +
    tags.map(t => '<span class="blog-tag">' + esc(t) + '</span>').join('') + '</div>';
}

function renderDate(iso) {
  return '<time class="post-date" datetime="' + iso + '">' + formatDate(iso) + '</time>';
}

// Numbered chapter list; `current` is highlighted, `summaries` adds each teaser.
// Links are root-relative like everything else (see renderPage).
function renderChapterList(s, { current = null, summaries = false, anchors = false } = {}) {
  return [
    '<ol class="series-chapters' + (summaries ? ' series-chapters-full' : '') + '">',
    s.chapters.map(c => [
      '  <li' + (c === current ? ' class="current"' : '') + '>',
      '    <a href="' + (anchors ? '#chapter-' + c.num : c.url) + '"' +
        (c === current ? ' aria-current="page"' : '') + '>' + esc(c.title) + '</a>' + draftBadge(c),
      summaries && c.summary ? '    <p class="series-summary">' + esc(c.summary) + '</p>' : '',
      '  </li>'
    ].filter(Boolean).join('\n')).join('\n'),
    '</ol>'
  ].join('\n');
}

function renderSeriesActions(s) {
  return '<p class="series-actions"><a href="' + s.chapters[0].url + '">Start reading &rarr;</a>' +
    ' &middot; <a href="' + s.allUrl + '">Read the whole series on one page</a></p>';
}

// Intro text, chapter list and reading options; shared by the overview page
// and the featured slot on blog.html.
function renderSeriesOverview(s) {
  return [
    s.html.trim() ? '<div class="post-body">\n' + s.html + '</div>' : '',
    '<h2 class="series-heading">Chapters</h2>',
    renderChapterList(s, { summaries: true }),
    renderSeriesActions(s)
  ].filter(Boolean).join('\n');
}

// Compact index of all posts; the one being shown in full is marked current.
function renderPostIndex(posts, current) {
  return [
    '        <aside class="post-index" aria-label="All posts">',
    '          <h2>All posts</h2>',
    '          <ul>',
    posts.map(p => [
      '            <li' + (p === current ? ' class="current"' : '') + '>',
      '              <a href="' + p.url + '"' + (p === current ? ' aria-current="page"' : '') + '>' +
        esc(p.title) + '</a>' + (p.kind === 'series' ? partsBadge(p) : '') + draftBadge(p),
      '              ' + renderDate(p.sortDate),
      '            </li>'
    ].join('\n')).join('\n'),
    '          </ul>',
    '        </aside>'
  ].join('\n');
}

// Blog page: newest post (or series overview) in full, with an index of all
// posts beside it (above it on narrow screens). Bodies use root-relative paths,
// so they work as-is on blog.html.
function renderBlogContent(posts) {
  if (!posts.length) return '\n      <p class="blog-intro">No posts yet.</p>\n      ';
  const latest = posts[0];
  const isSeries = latest.kind === 'series';
  return [
    '',
    '      <div class="blog-layout">',
    renderPostIndex(posts, latest),
    '        <article class="post post-featured">',
    isSeries ? '          <p class="series-kicker">Series' + partsBadge(latest) + '</p>' : '',
    '          ' + renderDate(latest.date),
    '          <h2 class="post-title"><a href="' + latest.url + '">' + esc(latest.title) + '</a>' +
      draftBadge(latest) + '</h2>',
    '          ' + renderTags(latest.tags),
    isSeries ? renderSeriesOverview(latest)
             : '          <div class="post-body">\n' + latest.html + '\n          </div>',
    '        </article>',
    '      </div>',
    '      '
  ].filter(Boolean).join('\n');
}

// Builds a page from blog.html: swaps head metadata and <main> content, and
// prefixes relative URLs with ../ per directory level below the site root.
function renderPage(template, page) {
  const desc = esc(page.desc);
  const url  = SITE_URL + '/' + page.url;
  // Function replacers so "$" in post content isn't treated as a pattern
  const setAttr = (h, prefix, value) =>
    h.replace(new RegExp('(' + prefix + ')[^"]*'), (_, pre) => pre + value);
  let html = template.replace(/<title>[\s\S]*?<\/title>/,
    () => '<title>' + esc(page.title) + ' &mdash; Pascal Wullschleger</title>');
  html = setAttr(html, '<meta name="description" content="', desc);
  html = setAttr(html, '<link rel="canonical" href="', url);
  html = setAttr(html, '<meta property="og:type" content="', 'article');
  html = setAttr(html, '<meta property="og:title" content="', esc(page.title));
  html = setAttr(html, '<meta property="og:description" content="', desc);
  html = setAttr(html, '<meta property="og:url" content="', url);
  html = html.replace(/<main>[\s\S]*<\/main>/, () => [
    '<main>',
    '    <div class="container">',
    '      <article class="post">',
    page.main,
    '      </article>',
    '    </div>',
    '  </main>'
  ].join('\n'));
  const prefix = '../'.repeat(page.url.split('/').length - 1);
  html = html.replace(/\b(href|src)="(?![a-z][a-z0-9+.-]*:|\/|#)([^"]*)"/gi,
    (_, attr, ref) => attr + '="' + prefix + ref + '"');
  return hasMath(page.main) ? withKatexCss(html, prefix) : html;
}

function renderPostPage(template, p) {
  return renderPage(template, { url: p.url, title: p.title, desc: p.summary || p.title, main: [
    '        <p class="post-back"><a href="blog.html">&larr; All posts</a></p>',
    '        <h1>' + esc(p.title) + draftBadge(p) + '</h1>',
    '        ' + renderDate(p.date),
    '        ' + renderTags(p.tags),
    '        <div class="post-body">',
    p.html,
    '        </div>'
  ].join('\n') });
}

function renderSeriesPage(template, s) {
  return renderPage(template, { url: s.url, title: s.title, desc: s.summary || s.title, main: [
    '        <p class="post-back"><a href="blog.html">&larr; All posts</a></p>',
    '        <p class="series-kicker">Series' + partsBadge(s) + '</p>',
    '        <h1>' + esc(s.title) + draftBadge(s) + '</h1>',
    '        ' + renderDate(s.date),
    '        ' + renderTags(s.tags),
    renderSeriesOverview(s)
  ].join('\n') });
}

function renderChapterPage(template, s, c) {
  const prev = s.chapters[c.num - 2];
  const next = s.chapters[c.num];
  const navLink = (ch, cls, label) => ch
    ? '  <a class="' + cls + '" href="' + ch.url + '"><span class="series-nav-label">' + label +
      '</span>' + esc(ch.title) + '</a>'
    : '';
  return renderPage(template, {
    url: c.url, title: c.title + ' · ' + s.title, desc: c.summary || s.summary || c.title,
    main: [
      '        <p class="post-back"><a href="blog.html">&larr; All posts</a></p>',
      '        <p class="series-kicker"><a href="' + s.url + '">' + esc(s.title) + '</a>' +
        ' &middot; Part ' + c.num + ' of ' + s.chapters.length + '</p>',
      '        <h1>' + esc(c.title) + draftBadge(c) + '</h1>',
      '        ' + renderDate(c.date),
      '        ' + renderTags(c.tags),
      '        <details class="series-box">',
      '          <summary>All chapters</summary>',
      renderChapterList(s, { current: c }),
      '          <p class="series-all"><a href="' + s.allUrl + '">Read the whole series on one page</a></p>',
      '        </details>',
      '        <div class="post-body">',
      c.html,
      '        </div>',
      '        <nav class="series-nav" aria-label="Chapter navigation">',
      navLink(prev, 'prev', '&larr; Previous'),
      next ? navLink(next, 'next', 'Next &rarr;')
           : '  <a class="next" href="' + s.url + '"><span class="series-nav-label">End of series</span>' +
             'Back to overview</a>',
      '        </nav>'
    ].filter(Boolean).join('\n')
  });
}

// One-page view: chapter titles become h2, so each chapter's headings move down a level.
function renderSeriesAllPage(template, s) {
  const demote = html => html.replace(/<(\/?)h([2-5])\b/g, (_, slash, n) => '<' + slash + 'h' + (+n + 1));
  return renderPage(template, {
    url: s.allUrl, title: s.title + ' (all chapters)', desc: s.summary || s.title,
    main: [
      '        <p class="post-back"><a href="' + s.url + '">&larr; Series overview</a></p>',
      '        <p class="series-kicker">Series' + partsBadge(s) + ' &middot; one page</p>',
      '        <h1>' + esc(s.title) + draftBadge(s) + '</h1>',
      '        ' + renderDate(s.date),
      '        ' + renderTags(s.tags),
      '        <nav class="series-box" aria-label="Contents">',
      renderChapterList(s, { anchors: true }),
      '        </nav>',
      s.chapters.map(c => [
        '        <section class="series-chapter" id="chapter-' + c.num + '">',
        '          <h2 class="chapter-title"><span class="chapter-num">Part ' + c.num + '</span>' +
          esc(c.title) + draftBadge(c) + '</h2>',
        '          <div class="post-body">',
        demote(c.html),
        '          </div>',
        '        </section>'
      ].join('\n')).join('\n')
    ].join('\n')
  });
}

const sorted = sortedEntries(entries);
const posts  = loadEntries();

// Copy source to _site/
const outDir = '_site';
if (fs.existsSync(outDir)) fs.rmSync(outDir, { recursive: true });
fs.mkdirSync(outDir);

const skip = new Set(['_site', 'node_modules', '.git', 'build.js', 'Makefile',
                      'package.json', 'package-lock.json', '.github', 'serve.js', POSTS_DIR]);
fs.readdirSync('.').forEach(f => {
  if (skip.has(f) || f.startsWith('.')) return;
  fs.cpSync(f, path.join(outDir, f), { recursive: true });
});

// Patch publications.html
let pubsHtml = fs.readFileSync(path.join(outDir, 'publications.html'), 'utf8');
pubsHtml = inject(pubsHtml, '<!-- PUBS_STATIC_START -->', '<!-- PUBS_STATIC_END -->', renderAllPubs(sorted));
fs.writeFileSync(path.join(outDir, 'publications.html'), pubsHtml);

// Patch index.html
let indexHtml = fs.readFileSync(path.join(outDir, 'index.html'), 'utf8');
indexHtml = inject(indexHtml, '<!-- PUBS_RECENT_START -->', '<!-- PUBS_RECENT_END -->', renderRecentPubs(sorted));
// Show more-link (build makes it immediately visible)
indexHtml = indexHtml.replace(
  /(<a href="publications\.html" class="more-link" id="more-link") style="display:none"/,
  '$1'
);
fs.writeFileSync(path.join(outDir, 'index.html'), indexHtml);

// KaTeX stylesheet + fonts (only linked from pages that contain math)
const katexDist = path.join(path.dirname(require.resolve('katex/package.json')), 'dist');
fs.mkdirSync(path.join(outDir, 'katex'));
fs.copyFileSync(path.join(katexDist, 'katex.min.css'), path.join(outDir, KATEX_CSS));
fs.cpSync(path.join(katexDist, 'fonts'), path.join(outDir, 'katex', 'fonts'), { recursive: true });

// Blog: post/series/chapter pages, post list, sitemap
const blogTemplate = fs.readFileSync('blog.html', 'utf8');
const pages = [];  // { url, html, lastmod }
posts.forEach(p => {
  if (p.kind === 'post') {
    pages.push({ url: p.url, html: renderPostPage(blogTemplate, p), lastmod: p.date });
    return;
  }
  pages.push({ url: p.url, html: renderSeriesPage(blogTemplate, p), lastmod: p.sortDate });
  p.chapters.forEach(c => {
    pages.push({ url: c.url, html: renderChapterPage(blogTemplate, p, c), lastmod: c.date });
  });
  pages.push({ url: p.allUrl, html: renderSeriesAllPage(blogTemplate, p), lastmod: p.sortDate });
});
pages.forEach(pg => {
  const file = path.join(outDir, pg.url);
  fs.mkdirSync(path.dirname(file), { recursive: true });
  fs.writeFileSync(file, pg.html);
});

let blogHtml = fs.readFileSync(path.join(outDir, 'blog.html'), 'utf8');
const blogContent = renderBlogContent(posts);
blogHtml = inject(blogHtml, '<!-- POSTS_START -->', '<!-- POSTS_END -->', blogContent);
if (hasMath(blogContent)) blogHtml = withKatexCss(blogHtml, '');
fs.writeFileSync(path.join(outDir, 'blog.html'), blogHtml);

let sitemap = fs.readFileSync(path.join(outDir, 'sitemap.xml'), 'utf8');
sitemap = sitemap.replace('</urlset>', () => pages.map(pg => [
  '  <url>',
  '    <loc>' + SITE_URL + '/' + pg.url + '</loc>',
  '    <lastmod>' + pg.lastmod + '</lastmod>',
  '    <priority>0.6</priority>',
  '  </url>\n'
].join('\n')).join('') + '</urlset>');
fs.writeFileSync(path.join(outDir, 'sitemap.xml'), sitemap);

console.log('Built ' + entries.length + ' publications, ' + posts.length + ' posts/series, ' +
            pages.length + ' blog pages → _site/');
