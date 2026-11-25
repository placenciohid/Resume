# How to read the evidence

## Professional outcomes

The professional case studies draw on work experience described by Darío Placencio: a 365-day production forecasting pipeline with 21% lower error against its prior baseline; a planning-model rebuild identifying $746K in confirmed savings and a separate $7.3M over-commitment diagnostic; document search spanning 300,000+ documents and adopted by four teams; and delivered collections reporting and an hourly aging feed.

These employer outcomes are described in the articles. Internal datasets, private models, approval records, and customer information are not distributed in this repository. The runnable companions are independent implementations and do not reproduce those business figures.

## Maintained real-data analyses

Four `analyze.py` scripts were run for the portfolio update on **25 November 2026**. Their adjacent `analysis-results.json` files retain the data SHA-256, environment versions, evaluation protocol, and complete metrics. The wine data is sourced and licensed through UCI. The app-review and passenger analyses use the existing repository datasets with their original source context.

These results describe the stated holdouts and preprocessing. They are not production KPIs or competition leaderboard scores. Model selection for the wine network occurs within grouped training folds. Review duplicates, repeated wine chemistry, and passenger travel groups remain within a single train/test partition.

## Portable examples

Eight `demo.py` scripts use generated data, fictional records, simulation, or explicitly identified exercise statistics. Their `results.json` snapshots are regenerated from the scripts. They demonstrate implementation behavior and calculation controls.

## Historical material

The `legacy/` folders preserve the original work. Saved notebook outputs are identified as historical. Where an original source file is unavailable or an evaluation has a known issue, the article explains the practical consequence. Current results come from the maintained scripts rather than a pasted historical performance table.

## Regenerate

Run the four real-data scripts with `--write`, then `python tools/build_artifacts.py` to refresh the portable snapshots and article figures. The latter requires the optional analysis environment. Run `python tools/check_links.py` to check repository-local article and image targets, and the test commands in the root README to verify the implementations.
