# Event prediction and data preparation

**Problem:** turn operational snapshots and change history into data for predicting future events.

The original PySpark script selected away required fields, combined differently named event dates, used future history as a feature, and fitted preprocessing before splitting the data. It is retained in `legacy/` alongside the GEV and graph-neural-network research examples.

## Runnable reference

`python projects/event-prediction/demo.py`

`demo.py` is a repaired, single-machine reference implementation using standard Python and four synthetic records. Required record fields are `id`, `record_date` (ISO date), `operation_code`, and `numeric_feature`. Events use `id`, `date`, and `kind` (`event` or `resolve`). All fields needed downstream are retained.

Each snapshot uses only history available on its date. New events in the following seven days form the target; they never become input features. Event starts and resolutions share one date field. Scaling is fitted to training rows only, with a seven-day gap between training labels and validation. The demo prints the rows and fitted mean/scale so these decisions are inspectable. It prepares data; it does not claim a trained predictive model.

## Limits

This four-row fixture is for correctness, not accuracy measurement or distributed performance. Legacy Spark and deep-learning scripts are illustrative and depend on data and environments not supplied here. A production implementation would also require source completeness, event deduplication, explicit timezone policy, and a label-observation window. No historical accuracy is asserted by this example.

[Website article](https://placenciohid.github.io/2024/10/26/EventDataPipeline.html)
