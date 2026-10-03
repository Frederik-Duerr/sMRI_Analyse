# LaTeX Master's Thesis / Scientific Report Template

This template is designed for Overleaf and for reusable scientific reports in biomedical engineering.

## Structure

- `main.tex` — main document and thesis/report metadata
- `config/preamble.tex` — page layout, fonts, spacing, figures, tables, references, etc.
- `frontmatter/` — title page, abstract, abbreviations, optional acknowledgements/declaration
- `chapters/` — one `.tex` file per chapter
- `references/references.bib` — bibliography database
- `figures/` — images, organized by chapter
- `tables/` — optional external table files
- `appendices/` — appendices

## Overleaf

1. Create a blank Overleaf project.
2. Upload the complete contents of this folder, preserving the directory structure.
3. Keep `main.tex` in the project root.
4. Open **Settings → Compiler** and make sure `main.tex` is selected as the Main document.
5. The default compiler `pdfLaTeX` is suitable for this template.
6. `biblatex` uses `biber` as the bibliography backend. Overleaf normally handles the required compilation steps automatically.

## Important

This is a general scientific template, not an official UCP thesis template. Before final submission,
compare the final formatting with the current requirements of your programme/faculty/supervisor.
University-specific cover-page, font, margin, line-spacing, reference-style, page-limit, and declaration
requirements override this template.

## Recommended image organisation

Keep images in `figures/`, grouped by chapter. For example:

```text
figures/
├── 01-introduction/
├── 02-methods/
├── 03-results/
└── 04-discussion/
```

Use descriptive filenames without spaces, e.g. `mri_preprocessing_pipeline.pdf` or `roc_curve_model_a.png`.
In the chapter file, use a path such as `\includegraphics[width=0.85\textwidth]{03-results/roc_curve_model_a}`. Overleaf supports `.png`, `.jpg`, and `.pdf` images with pdfLaTeX.

## Recommended citation workflow

Store all references in `references/references.bib`. Use `\cite{key}` in the text and let `biblatex` generate the reference list.
The example uses numeric citations in citation order (`[1]`, `[2]`, ...). Change the bibliography style only when your programme requires another style.
