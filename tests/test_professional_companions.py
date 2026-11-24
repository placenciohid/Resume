import importlib.util
from pathlib import Path
from decimal import Decimal
import unittest

ROOT = Path(__file__).resolve().parents[1]
def load(name):
    spec=importlib.util.spec_from_file_location(name,ROOT/'projects'/name/'demo.py')
    mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod);return mod
forecast=load('forecasting');search=load('knowledge-search');report=load('commercial-reporting')

class ForecastTests(unittest.TestCase):
    def test_backtest_is_forward_only(self):
        rows=forecast.sample_data();result=forecast.backtest(rows)
        self.assertEqual(result['forecasted_days'],91)
        self.assertTrue(all(f['trained_through']<f['predict_from'] for f in result['folds']))
        self.assertLess(result['trend_plus_weekday']['mae'],result['seasonal_naive']['mae'])
    def test_future_values_do_not_affect_fit(self):
        rows=forecast.sample_data();first=forecast.fit(rows[:84]);rows[-1]['value']=1000000
        self.assertEqual(first,forecast.fit(rows[:84]))
    def test_interval_orders(self):
        self.assertTrue(all(r['low']<=r['forecast']<=r['high'] for r in forecast.run()['next_seven_days']))

class SearchTests(unittest.TestCase):
    def test_queries_find_expected_documents(self):
        self.assertEqual(search.run()['top1_hits'],5)
    def test_evidence_is_source_text(self):
        hit=search.search('pump vibration')[0]
        self.assertEqual(hit['document_id'],'maintenance')
        self.assertIn(hit['text'],next(d['text'] for d in search.DOCUMENTS if d['id']==hit['document_id']))
    def test_unknown_and_empty_queries_abstain(self):
        self.assertEqual(search.search(''),[])
        self.assertEqual(search.search('qzxxxyy'),[])

class ReportingTests(unittest.TestCase):
    def test_document_level_throughput(self):
        self.assertEqual(report.run()['fully_matched'],{'documents':1,'invoice_lines':2,'payments_by_currency':{'USD':'2000.00'}})
    def test_missing_and_currency_mismatch(self):
        rows=report.reconcile(report.INVOICES,report.REMITTANCES)
        self.assertEqual([r['status'] for r in rows],['cleared','payment_difference','missing_invoice'])
        remit=[{'id':'X','currency':'USD','payment':'900.00','invoice_ids':['C1']}]
        self.assertEqual(report.reconcile(report.INVOICES,remit)[0]['status'],'currency_mismatch')
    def test_duplicate_claim_rejected(self):
        with self.assertRaises(ValueError): report.reconcile(report.INVOICES,report.REMITTANCES*2)
        with self.assertRaises(ValueError): report.reconcile(report.INVOICES+report.INVOICES[:1],report.REMITTANCES)
    def test_aging_due_today_and_currency_separation(self):
        rows=report.age(report.INVOICES,'2024-03-01')
        self.assertEqual(sum(Decimal(r['amount']) for r in rows if r['currency']=='USD'),Decimal('2800.00'))
        self.assertEqual(sum(Decimal(r['amount']) for r in rows if r['currency']=='CAD'),Decimal('900.00'))
        self.assertTrue(any(r['currency']=='USD' and r['category']=='trade' and r['bucket']=='current' and r['lines']==3 for r in rows))

if __name__=='__main__': unittest.main()
