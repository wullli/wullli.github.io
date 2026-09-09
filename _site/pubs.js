/* Shared BibTeX parser + APA formatter */

/* ── BibTeX parser ── */
function parseBibTeX(raw) {
  var entries = [];
  var entryRe = /@(\w+)\s*\{\s*([^,\s]+)\s*,([\s\S]*?)(?=\n@|\s*$)/g;
  var m;
  while ((m = entryRe.exec(raw)) !== null) {
    entries.push({
      entryType:   m[1].toLowerCase(),
      citationKey: m[2].trim(),
      entryTags:   parseFields(m[3])
    });
  }
  return entries;
}

function parseFields(body) {
  var out = {};
  /* Handles: field={value}, field="value", field=bareword, field=number */
  var re = /(\w+)\s*=\s*(?:\{((?:[^{}]|\{[^{}]*\})*)\}|"([^"]*)"|(\w+))/g;
  var m;
  while ((m = re.exec(body)) !== null) {
    var key = m[1].toUpperCase();
    var val = m[2] !== undefined ? m[2] : (m[3] !== undefined ? m[3] : (m[4] || ''));
    out[key] = val;
  }
  return out;
}

/* ── Utilities ── */
function clean(s) {
  if (!s) return '';
  return String(s).replace(/^\{(.*)\}$/, '$1').replace(/[{}]/g, '').trim();
}

function esc(s) {
  return String(s)
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;');
}

/* ── APA author formatting ── */
function apaAuthorSingle(raw) {
  var name = raw.trim();
  if (!name || name.toLowerCase() === 'others') return null;

  var last, firstParts;
  var idx = name.indexOf(',');
  if (idx > -1) {
    last       = name.slice(0, idx).trim();
    firstParts = name.slice(idx + 1).trim().split(/\s+/).filter(Boolean);
  } else {
    var parts  = name.split(/\s+/);
    last       = parts[parts.length - 1];
    firstParts = parts.slice(0, -1);
  }

  var initials = firstParts
    .map(function(p) { return /^[A-Z]\.$/.test(p) ? p : p.charAt(0).toUpperCase() + '.'; })
    .join(' ');

  return {
    formatted: initials ? last + ', ' + initials : last,
    isPW: /Wullschleger/i.test(last)
  };
}

function formatAuthorsAPA(rawStr) {
  if (!rawStr) return '';
  var str     = clean(rawStr);
  var parts   = str.split(/\s+and\s+/i);
  var hasEtAl = parts[parts.length - 1].trim().toLowerCase() === 'others';
  var raw     = hasEtAl ? parts.slice(0, -1) : parts;
  var authors = raw.map(function(p) { return apaAuthorSingle(p); }).filter(Boolean);

  var rendered = authors.map(function(a, i) {
    var html = a.isPW ? '<strong>' + esc(a.formatted) + '</strong>' : esc(a.formatted);
    if (authors.length === 1)                    return html;
    if (!hasEtAl && i === authors.length - 1)    return '&amp; ' + html;
    return html;
  });

  return rendered.join(', ') + (hasEtAl ? ', et al.' : '');
}

/* ── APA venue formatting ── */
function formatVenueAPA(tags) {
  var journal   = clean(tags.JOURNAL);
  var booktitle = clean(tags.BOOKTITLE);
  var volume    = clean(tags.VOLUME);
  var number    = clean(tags.NUMBER);
  var pages     = clean(tags.PAGES);
  var note      = clean(tags.NOTE);

  if (journal) {
    var v = '<em>' + esc(journal) + '</em>';
    if (volume) v += ', <em>' + esc(volume) + '</em>';
    if (number) v += '(' + esc(number) + ')';
    if (pages)  v += ', ' + esc(pages.replace(/--/, '–'));
    return v + '.';
  }
  if (booktitle) {
    var b = 'In <em>' + esc(booktitle) + '</em>';
    if (pages) b += ' (pp. ' + esc(pages.replace(/--/, '–')) + ')';
    return b + '.';
  }
  if (note) return esc(note) + '.';
  return '';
}

/* ── URL / link label helper ── */
function primaryUrl(tags) {
  var u   = clean(tags.URL);
  var doi = clean(tags.DOI);
  var arxiv = clean(tags.ARXIV);
  if (u)     return u;
  if (doi)   return 'https://doi.org/' + doi;
  if (arxiv) return 'https://arxiv.org/abs/' + arxiv;
  return '';
}

function linkLabel(url) {
  if (!url) return '';
  if (/arxiv\.org/i.test(url))       return 'arXiv';
  if (/aclanthology\.org/i.test(url)) return 'ACL Anthology';
  return 'Full Text';
}

/* ── BibTeX syntax highlighter ── */
function highlightBib(entry) {
  var h = esc(toBibTeX(entry));
  /* @entrytype{citationkey, */
  h = h.replace(/^(@\w+)\{([^,\n]+),/m,
    '<span class="bib-t">$1</span>{<span class="bib-k">$2</span>,');
  /* field = {string value} — value may span lines, clean() ensures no nested braces */
  h = h.replace(/^( {2}\w[\w ]*?)(=) \{([^}]*)\}/gm,
    '<span class="bib-f">$1</span><span class="bib-eq">$2</span> {<span class="bib-v">$3</span>}');
  /* field = number */
  h = h.replace(/^( {2}\w[\w ]*?)(=) (\d+)/gm,
    '<span class="bib-f">$1</span><span class="bib-eq">$2</span> <span class="bib-n">$3</span>');
  return h;
}

/* ── Reconstruct BibTeX from parsed entry ── */
var BIB_SKIP = ['SCHOLAR', 'CITATIONS', 'CODE', 'ANTHOLOGY', 'PDF', 'SELECTED'];

function toBibTeX(entry) {
  var fields = Object.entries(entry.entryTags)
    .filter(function(kv) { return BIB_SKIP.indexOf(kv[0]) === -1; })
    .map(function(kv) {
      var val = kv[1];
      /* Numeric fields (year, volume, number) without braces */
      if (/^\d+$/.test(val)) return '  ' + kv[0].toLowerCase().padEnd(12) + '= ' + val;
      return '  ' + kv[0].toLowerCase().padEnd(12) + '= {' + clean(val) + '}';
    })
    .join(',\n');
  return '@' + entry.entryType + '{' + entry.citationKey + ',\n' + fields + '\n}';
}

/* ── Single entry renderer (shared by publications.html and index.html) ── */
function renderEntry(entry, extraLinks, num) {
  var t     = entry.entryTags;
  var title = clean(t.TITLE) || 'Untitled';
  var year  = clean(t.YEAR)  || 'n.d.';
  var code    = clean(t.CODE)  || (extraLinks && extraLinks.code)    || '';
  var website =                   (extraLinks && extraLinks.website) || '';
  var data    =                   (extraLinks && extraLinks.data)    || '';
  var scholar   = clean(t.SCHOLAR)   || '';
  var citations = clean(t.CITATIONS) || '';

  var url   = primaryUrl(t);
  var label = linkLabel(url);

  var titleHtml = url
    ? '<a href="' + esc(url) + '" target="_blank" rel="noopener">' + esc(title) + '</a>'
    : esc(title);

  var authors = formatAuthorsAPA(t.AUTHOR);
  var venue   = formatVenueAPA(t);
  var venueStr = '(' + esc(year) + ')' + (venue ? '. ' + venue : '.');

  var links = [];
  if (url)     links.push('<a href="' + esc(url) + '" class="pub-link" target="_blank" rel="noopener">' + esc(label) + '</a>');
  if (code)    links.push('<a href="' + esc(code) + '" class="pub-link" target="_blank" rel="noopener">Code</a>');
  if (website) links.push('<a href="' + esc(website) + '" class="pub-link" target="_blank" rel="noopener">Website</a>');
  if (data)    links.push('<a href="' + esc(data) + '" class="pub-link" target="_blank" rel="noopener">Data</a>');
  if (scholar) links.push('<a href="' + esc(scholar) + '" class="pub-link" target="_blank" rel="noopener">Scholar</a>');
  links.push('<button class="bibtex-toggle" onclick="toggleBibtex(this)" aria-expanded="false">BibTeX</button>');
  if (citations) links.push('<span class="cite-count">' + esc(citations) + ' citation' + (citations !== '1' ? 's' : '') + '</span>');

  var numCol = num != null ? '  <span class="pub-num">[' + num + ']</span>\n' : '';
  return [
    '<div class="pub-entry">',
    numCol + '  <div class="pub-body">',
    '    <p class="pub-title">' + titleHtml + '</p>',
    '    <p class="pub-authors">' + authors + '</p>',
    '    <p class="pub-venue">' + venueStr + '</p>',
    '    <div class="pub-meta">' + links.join('\n    ') + '</div>',
    '    <div class="bibtex-block"><pre>' + highlightBib(entry) + '</pre></div>',
    '  </div>',
    '</div>'
  ].join('\n');
}

/* ── BibTeX toggle ── */
function toggleBibtex(btn) {
  var block = btn.closest('.pub-entry').querySelector('.bibtex-block');
  var open  = block.classList.toggle('open');
  btn.textContent = open ? 'Close' : 'BibTeX';
  btn.setAttribute('aria-expanded', String(open));
}

/* ── Supplementary links YAML loader ── */
function parseSimpleYAML(text) {
  var result = {};
  var key = null;
  text.split('\n').forEach(function(line) {
    if (!line.trim() || line.trim()[0] === '#') return;
    if (/^\S/.test(line)) {
      key = line.trim().replace(/:$/, '').trim();
      result[key] = {};
    } else if (key) {
      var m = line.match(/^\s+(\w+)\s*:\s*(.+)$/);
      if (m) {
        var v = m[2].trim();
        if (v && v[0] !== '#') result[key][m[1]] = v.replace(/^['"]|['"]$/g, '');
      }
    }
  });
  return result;
}

function loadLinkMap(callback) {
  fetch('./pub_links.yaml')
    .then(function(r) { return r.ok ? r.text() : ''; })
    .then(function(t) { callback(parseSimpleYAML(t)); })
    .catch(function() { callback({}); });
}
