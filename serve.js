#!/usr/bin/env node
'use strict';

// Dev server: builds the site, serves _site/, rebuilds on source changes and
// live-reloads open pages. Includes draft posts. Usage: node serve.js [port]

const fs    = require('fs');
const http  = require('http');
const path  = require('path');
const { spawnSync } = require('child_process');

const PORT = Number(process.argv[2]) || 8000;
const ROOT = path.resolve('_site');
const IGNORE = ['_site', 'node_modules', '.git'];

const TYPES = {
  '.html': 'text/html; charset=utf-8', '.css': 'text/css', '.js': 'text/javascript',
  '.svg': 'image/svg+xml', '.png': 'image/png', '.jpg': 'image/jpeg', '.jpeg': 'image/jpeg',
  '.gif': 'image/gif', '.webp': 'image/webp', '.xml': 'application/xml', '.txt': 'text/plain',
  '.json': 'application/json', '.yaml': 'text/plain', '.bib': 'text/plain', '.pdf': 'application/pdf'
};

const RELOAD_SNIPPET =
  '<script>new EventSource("/__reload").onmessage = () => location.reload();</script>';

const clients = new Set();

function build() {
  const r = spawnSync(process.execPath, ['build.js', '--drafts'], { stdio: 'inherit' });
  return r.status === 0;
}

function serveFile(req, res) {
  let rel = decodeURIComponent(new URL(req.url, 'http://x').pathname);
  if (rel.endsWith('/')) rel += 'index.html';
  const file = path.join(ROOT, path.normalize(rel));
  if (!file.startsWith(ROOT + path.sep)) { res.writeHead(403).end(); return; }

  fs.readFile(file, (err, data) => {
    if (err) { res.writeHead(404, { 'Content-Type': 'text/plain' }).end('Not found'); return; }
    const ext = path.extname(file).toLowerCase();
    if (ext === '.html') data = data.toString().replace('</body>', RELOAD_SNIPPET + '</body>');
    res.writeHead(200, { 'Content-Type': TYPES[ext] || 'application/octet-stream',
                         'Cache-Control': 'no-store' });
    res.end(data);
  });
}

http.createServer((req, res) => {
  if (req.url === '/__reload') {
    res.writeHead(200, { 'Content-Type': 'text/event-stream', 'Cache-Control': 'no-store' });
    res.write(': connected\n\n');
    clients.add(res);
    req.on('close', () => clients.delete(res));
    return;
  }
  serveFile(req, res);
}).listen(PORT, '127.0.0.1', () => {
  console.log('Serving http://localhost:' + PORT + ' — watching for changes (Ctrl+C to stop)');
});

// Watch every source directory (not files): directory watches also catch editors
// that save by writing a temp file and renaming it over the original, which
// recursive fs.watch on Linux misses. Re-scanned after each build to pick up
// new directories.
const watchers = new Map();
let timer = null;

function ignored(name) {
  return IGNORE.includes(name) || name.startsWith('.') || name.endsWith('~');
}

function sourceDirs(dir = '.', out = []) {
  out.push(dir);
  fs.readdirSync(dir, { withFileTypes: true }).forEach(e => {
    if (e.isDirectory() && !ignored(e.name)) sourceDirs(path.join(dir, e.name), out);
  });
  return out;
}

function onChange(dir, name) {
  if (!name || ignored(name)) return;
  clearTimeout(timer);
  timer = setTimeout(() => {
    console.log('Changed: ' + path.join(dir, name) + ' — rebuilding');
    if (build()) clients.forEach(c => c.write('data: reload\n\n'));
    syncWatchers();
  }, 150);
}

function syncWatchers() {
  const dirs = new Set(sourceDirs());
  watchers.forEach((w, d) => { if (!dirs.has(d)) { w.close(); watchers.delete(d); } });
  dirs.forEach(d => {
    if (watchers.has(d)) return;
    const w = fs.watch(d, (_, name) => onChange(d, name));
    w.on('error', () => { w.close(); watchers.delete(d); });
    watchers.set(d, w);
  });
}

build();
syncWatchers();
