/* Init immediately in <head> to avoid flash */
(function () {
  var theme = (function () { try { return localStorage.getItem('theme') || 'light'; } catch (e) { return 'light'; } }());
  document.documentElement.setAttribute('data-theme', theme);

  var ACCENTS = {
    sapphire:   { dot: '#2a7fd4',
      light: { accent: '#1a6abf', 'accent-muted': '#90b8e8', 'accent-dark': '#0f4a8a', 'link-vis': '#0f4a8a' },
      dark:  { accent: '#5a9fe0', 'accent-muted': '#1a3a5a', 'accent-dark': '#78b8f0', 'link-vis': '#4a8fd0' } },
    terracotta: { dot: '#c05a38',
      light: { accent: '#c05a38', 'accent-muted': '#e09a80', 'accent-dark': '#8a3822', 'link-vis': '#8a3822' },
      dark:  { accent: '#e07858', 'accent-muted': '#5a2818', 'accent-dark': '#f09070', 'link-vis': '#c06040' } },
    plum:       { dot: '#9b59b6',
      light: { accent: '#7a2e87', 'accent-muted': '#c9a0d8', 'accent-dark': '#5c1f68', 'link-vis': '#5c1f68' },
      dark:  { accent: '#c97fda', 'accent-muted': '#5a2e6a', 'accent-dark': '#d89ae8', 'link-vis': '#b367cc' } }
  };

  window._ACCENTS = ACCENTS;
  window._currentAccent = (function () { try { return localStorage.getItem('accent') || 'sapphire'; } catch (e) { return 'sapphire'; } }());

  function applyAccentVars(name, themeOverride) {
    var dark = (themeOverride || document.documentElement.getAttribute('data-theme')) === 'dark';
    var vars = ACCENTS[name][dark ? 'dark' : 'light'];
    var root = document.documentElement;
    Object.keys(vars).forEach(function (k) { root.style.setProperty('--' + k, vars[k]); });
  }

  window._applyAccentVars = applyAccentVars;
  applyAccentVars(window._currentAccent, theme);
}());

function applyAccent(name) {
  window._currentAccent = name;
  window._applyAccentVars(name);
  document.querySelectorAll('.accent-dot').forEach(function (d) {
    d.classList.toggle('active', d.dataset.accent === name);
  });
  try { localStorage.setItem('accent', name); } catch (e) {}
}

function toggleTheme() {
  var next = document.documentElement.getAttribute('data-theme') === 'dark' ? 'light' : 'dark';
  document.documentElement.setAttribute('data-theme', next);
  try { localStorage.setItem('theme', next); } catch (e) {}
  window._applyAccentVars(window._currentAccent);
  syncToggleBtn();
}

function syncToggleBtn() {
  var btn = document.getElementById('theme-btn');
  if (!btn) return;
  var dark = document.documentElement.getAttribute('data-theme') === 'dark';
  btn.textContent = '●';
  btn.title = dark ? 'Switch to light mode' : 'Switch to dark mode';
}

document.addEventListener('DOMContentLoaded', function () {
  syncToggleBtn();

  var picker = document.createElement('div');
  picker.className = 'accent-picker';
  picker.setAttribute('role', 'group');
  picker.setAttribute('aria-label', 'Accent color');

  Object.keys(window._ACCENTS).forEach(function (name) {
    var dot = document.createElement('button');
    dot.className = 'accent-dot';
    dot.dataset.accent = name;
    dot.style.setProperty('--dot-color', window._ACCENTS[name].dot);
    dot.setAttribute('aria-label', name.charAt(0).toUpperCase() + name.slice(1));
    dot.title = name.charAt(0).toUpperCase() + name.slice(1);
    if (name === window._currentAccent) dot.classList.add('active');
    dot.addEventListener('click', function () { applyAccent(name); });
    picker.appendChild(dot);
  });

  document.body.appendChild(picker);
});
