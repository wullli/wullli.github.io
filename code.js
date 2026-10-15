/* Copy buttons for post code blocks. A .code-fold block copies its folded
   imports too, since textContent includes closed <details> content. */
(function () {
  var SVG = '<svg viewBox="0 0 16 16" width="14" height="14" aria-hidden="true" fill="none" stroke="currentColor" stroke-width="1.4" stroke-linecap="round" stroke-linejoin="round">';
  var COPY_ICON = SVG + '<rect x="5.5" y="5.5" width="8" height="8" rx="1.5"/><path d="M10.5 3.5v-1a1 1 0 0 0-1-1h-6a1 1 0 0 0-1 1v6a1 1 0 0 0 1 1h1"/></svg>';
  var DONE_ICON = SVG + '<path d="M3 8.5l3 3 7-7"/></svg>';
  document.querySelectorAll('.post-body .code-fold, .post-body pre:not(.code-fold pre)').forEach(function (block) {
    var btn = document.createElement('button');
    btn.type = 'button';
    btn.className = 'code-copy';
    btn.setAttribute('aria-label', 'Copy code');
    btn.title = 'Copy code';
    btn.innerHTML = COPY_ICON;
    btn.addEventListener('click', function () {
      var text = Array.prototype.map.call(block.querySelectorAll('code'), function (c) {
        return c.textContent.replace(/\n$/, '');
      }).join('\n\n');
      navigator.clipboard.writeText(text + '\n').then(function () {
        btn.innerHTML = DONE_ICON;
        btn.setAttribute('aria-label', 'Copied');
        setTimeout(function () {
          btn.innerHTML = COPY_ICON;
          btn.setAttribute('aria-label', 'Copy code');
        }, 1500);
      });
    });
    var wrap = block;
    if (block.tagName === 'PRE') {
      wrap = document.createElement('div');
      wrap.className = 'code-wrap';
      block.parentNode.insertBefore(wrap, block);
      wrap.appendChild(block);
    }
    wrap.appendChild(btn);
  });
}());
