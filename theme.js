/* Init immediately in <head> to avoid flash */
(function () {
  document.documentElement.setAttribute(
    'data-theme',
    (function () { try { return localStorage.getItem('theme') || 'light'; } catch (e) { return 'light'; } }())
  );
}());

function toggleTheme() {
  var next = document.documentElement.getAttribute('data-theme') === 'dark' ? 'light' : 'dark';
  document.documentElement.setAttribute('data-theme', next);
  try { localStorage.setItem('theme', next); } catch (e) {}
  syncToggleBtn();
}

function syncToggleBtn() {
  var btn = document.getElementById('theme-btn');
  if (!btn) return;
  var dark = document.documentElement.getAttribute('data-theme') === 'dark';
  btn.textContent = '●';
  btn.title = dark ? 'Switch to light mode' : 'Switch to dark mode';
}

document.addEventListener('DOMContentLoaded', syncToggleBtn);
