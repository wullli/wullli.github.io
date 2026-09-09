#!/usr/bin/env node
'use strict';

const fs   = require('fs');
const path = require('path');
const vm   = require('vm');

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

const sorted = sortedEntries(entries);

// Copy source to _site/
const outDir = '_site';
if (fs.existsSync(outDir)) fs.rmSync(outDir, { recursive: true });
fs.mkdirSync(outDir);

const skip = new Set(['_site', 'node_modules', '.git', 'build.js', 'Makefile',
                      'package.json', 'package-lock.json', '.github']);
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

console.log('Built ' + entries.length + ' publications → _site/');
