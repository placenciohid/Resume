"""Aging and whole-remittance reconciliation using fictional invoice lines."""
from collections import defaultdict
from datetime import date
from decimal import Decimal
import json

INVOICES = [
    {'id':'A1','customer':'North','currency':'USD','amount':'1200.00','due':'2024-03-01','category':'trade'},
    {'id':'A2','customer':'North','currency':'USD','amount':'800.00','due':'2024-03-20','category':'trade'},
    {'id':'B1','customer':'West','currency':'USD','amount':'500.00','due':'2024-04-15','category':'trade'},
    {'id':'C1','customer':'Lake','currency':'CAD','amount':'900.00','due':'2024-01-01','category':'trade'},
    {'id':'S1','customer':'Other','currency':'USD','amount':'300.00','due':'2024-02-01','category':'sundry'},
]
REMITTANCES = [
    {'id':'R1','currency':'USD','payment':'2000.00','invoice_ids':['A1','A2']},
    {'id':'R2','currency':'CAD','payment':'850.00','invoice_ids':['C1']},
    {'id':'R3','currency':'USD','payment':'100.00','invoice_ids':['MISSING']},
]


def invoice_index(invoices):
    lookup = {i['id']:i for i in invoices}
    if len(lookup) != len(invoices): raise ValueError('Duplicate invoice identifiers')
    return lookup


def reconcile(invoices, remittances, tolerance=Decimal('0.01')):
    lookup = invoice_index(invoices)
    claimed = set()
    rows = []
    for remit in remittances:
        ids = remit['invoice_ids']
        if len(ids) != len(set(ids)) or any(i in claimed for i in ids):
            raise ValueError('An invoice cannot be claimed twice')
        claimed.update(ids)
        missing = [i for i in ids if i not in lookup]
        currencies = {lookup[i]['currency'] for i in ids if i in lookup}
        total = sum((Decimal(lookup[i]['amount']) for i in ids if i in lookup), Decimal('0'))
        difference = Decimal(remit['payment'])-total
        status = ('missing_invoice' if missing or not ids else
                  'currency_mismatch' if currencies != {remit['currency']} else
                  'cleared' if abs(difference) <= tolerance else 'payment_difference')
        rows.append({'remittance':remit['id'], 'currency':remit['currency'], 'invoice_count':len(ids),
                     'invoice_total':str(total), 'payment':remit['payment'], 'difference':str(difference),
                     'status':status, 'missing':missing})
    return rows


def age(invoices, as_of, cleared_ids=()):
    invoice_index(invoices)
    cutoff = date.fromisoformat(as_of)
    groups = defaultdict(lambda: {'lines':0,'amount':Decimal('0')})
    for invoice in invoices:
        if invoice['id'] in cleared_ids: continue
        days = (cutoff-date.fromisoformat(invoice['due'])).days
        bucket = 'current' if days <= 0 else '1-30' if days <= 30 else '31-60' if days <= 60 else '61-90' if days <= 90 else '90+'
        key = (invoice['currency'], invoice['category'], bucket)
        groups[key]['lines'] += 1
        groups[key]['amount'] += Decimal(invoice['amount'])
    return [{'currency':currency,'category':category,'bucket':bucket,
             'lines':v['lines'],'amount':str(v['amount'])} for (currency,category,bucket),v in sorted(groups.items())]


def run():
    rows = reconcile(INVOICES, REMITTANCES)
    cleared_docs = {r['remittance'] for r in rows if r['status']=='cleared'}
    cleared_ids = {i for r in REMITTANCES if r['id'] in cleared_docs for i in r['invoice_ids']}
    return {'dataset':'fictional records as of 2024-04-01; not actual collections',
            'reconciliation':rows, 'open_aging':age(INVOICES,'2024-04-01',cleared_ids),
            'fully_matched': {'documents':len(cleared_docs),'invoice_lines':len(cleared_ids),
                             'payments_by_currency':{currency:str(sum((Decimal(r['payment']) for r in rows if r['status']=='cleared' and r['currency']==currency),Decimal('0')))
                                                     for currency in sorted({r['currency'] for r in rows if r['status']=='cleared'})}}}


if __name__ == '__main__': print(json.dumps(run(),indent=2))
