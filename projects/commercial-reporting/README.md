# Receivables reporting and whole-remittance reconciliation

## Professional context and public companion

My professional work includes collections dashboards, hourly aging feeds, and reconciliation views. This independent implementation uses five fictional invoices and three fictional remittances. It demonstrates calculation and control logic; it is not an employer extract, policy, production reconciler, or cash-recovery claim.

## Reproduce

`python projects/commercial-reporting/demo.py`

Python 3.10+, standard library only. Monetary values use `Decimal`; dates are ISO dates. [Source](demo.py) · [Saved output](results.json).

## Data contract

Invoice fields: unique `id`, `customer`, `currency`, decimal `amount`, `due`, and `category` (`trade` or `sundry`). Remittance fields: `id`, `currency`, decimal `payment`, and a list of `invoice_ids`. A caller supplies a reporting date; rerunning a past reporting date gives the same aging buckets.

## Calculation and controls

- Age open invoices into current, 1–30, 31–60, 61–90, and 90+ days, grouping by currency and category.
- Resolve every invoice reference in a remittance; reject duplicate IDs or reuse across documents.
- Flag unknown references and currency mismatches before evaluating clearance.
- Compare the document payment with the sum of its referenced lines, using an explicit one-cent demonstration tolerance.
- Count only fully matched documents and their distinct lines; sum each document's payment once, by currency.

The synthetic result contains one fully matched document with two invoice lines and USD 2,000.00. A CAD 50 difference and an unknown invoice remain exceptions. These values are fixture outcomes, not real collections. The tolerance is a demo parameter, not an internal business rule.

## What this does not solve

The compact example does not allocate partial payments to individual lines, convert currencies, accept repeated installments, or claim that a line match proves bank settlement. It leaves a difference document open for human review. Sundry rows remain visible separately; the code does not alter an official KPI. Deployment would require source reconciliation, approved tolerances, user ownership, and a verified settlement feed.

Tests cover document-level totals, missing invoices, currency errors, duplicate claims, aging boundaries, and currency separation.
