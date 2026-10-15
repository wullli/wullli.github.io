# wullli.github.io

## Blog posts

Add `posts/<slug>.md` (copy `posts/_template.md`) with front matter `title`, `date` (YYYY-MM-DD),
optional `summary`, `tags` (comma-separated), `draft: true`. `make serve` previews at
http://localhost:8000; pushing to `master` deploys. Files starting with `_` are ignored.
`make dev` serves with live reload and shows drafts (badged); deploys omit them.

Series: a folder `posts/<series>/` with `_series.md` (copy `posts/_series_template.md`; its
body is the overview intro) plus one file per chapter, ordered by filename (`01-intro.md`, ...).
Chapters need `title`; `date`/`tags` default to the series'. Builds an overview
(`posts/<series>.html`), one page per chapter with prev/next, and `posts/<series>/all.html`
with every chapter on one page. The series is a single entry on the blog, dated by its newest chapter.

Code blocks get a copy button; ```` ```python fold-imports ```` collapses leading import lines
(still included when copying).
