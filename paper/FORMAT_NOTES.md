# Preparing the professor's final PDF

The requested working deliverable is `draft.md`. Markdown alone cannot establish compliance with the three-page limit; verify the actual PDF after applying `StudentsPack/Project-Template/template.tex`.

- Keep the supplied section order, use a compact title/author block, include contribution percentages, and avoid a cover page.
- Preserve the research question, two inspected data patterns, augmentation procedure, reproducible settings, model comparisons, per-class results and three concrete errors in the main paper.
- Refer to every table and figure in the text. Keep the full confusion matrix readable: use a full-width figure, or place it in the single permitted appendix page and refer to it from Results. Do not shrink its labels to fit a column.
- If space is tight, move the detailed metadata ablation table into that same appendix and retain its main conclusions in Discussion. The extended `experiment_details.md` is a working evidence file, not a submission-ready one-page appendix.
- Bibliography does not count toward the three main pages. Retain the pretrained-model references, software information and accurate LLM declaration; update the declaration to reflect the group's actual review and contributions.
- Check the final three-page main text and at-most-one-page appendix by opening the compiled PDF. The assignment warns that content beyond the main-page limit is not evaluated and that instruction violations cost three points.
- Use `package_submission.py` only after supplying the final PDF and actual group number. It preserves needed relative data paths and excludes virtual environments, caches and model weights. Submit the resulting `NUM.zip` via Fenix by 16 October 2026, 23:59.

All quoted errors and numeric claims in the draft must continue to match the saved run. Editing prose does not justify substituting results from the historical folders, changing the selected system after inspecting its test scores, or inventing clinical correctness beyond the supplied labels.
