## Structure

- `main.tex` — main document and thesis/report metadata
- `config/preamble.tex` — page layout, fonts, spacing, figures, tables, references, etc.
- `frontmatter/` — title page, abstract, abbreviations, optional acknowledgements/declaration
- `chapters/` — one `.tex` file per chapter
- `references/references.bib` — bibliography database
- `figures/` — images, organized by chapter
- `tables/` — optional external table files
- `appendices/` — appendices

# sMRI Analysis

This repository contains a machine learning pipeline for the analysis of structural brain MRI (sMRI) images for Alzheimer's disease stage classification.

The project compares feature representations extracted using **DINOv2**, a pretrained vision foundation model, with features obtained using the **Discrete Wavelet Transform (DWT2)**. The extracted features are used to train and evaluate machine learning classifiers across six diagnostic classes: Alzheimer's Disease (AD), Cognitively Normal (CN), Early Mild Cognitive Impairment (EMCI), Late Mild Cognitive Impairment (LMCI), Mild Cognitive Impairment (MCI), and Significant Memory Concern (SMC).

The workflow includes data loading, feature extraction, patient-level data splitting, cross-validation, model training, and performance evaluation.

**Note:** MRI image data are not included in this repository.
