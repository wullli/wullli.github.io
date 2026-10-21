---
title: Series title
date: 2026-01-31
summary: One-sentence teaser shown on the blog and the series overview.
tags: nlp, methods
draft: false
---

A series is a folder: posts/<series-slug>/_series.md (this file) plus one
Markdown file per chapter, ordered by filename (01-intro.md, 02-models.md, ...).
Chapters need a `title`; `date` and `tags` default to the series' values, and
`summary` is listed on the overview. `draft: true` unlists a single chapter or,
here, the whole series. This text becomes the intro on the overview page.
