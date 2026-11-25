import importlib.util
from pathlib import Path
import unittest
from decimal import Decimal

ROOT=Path(__file__).resolve().parents[1]
def load(slug):
    spec=importlib.util.spec_from_file_location(slug, ROOT/'projects'/slug/'demo.py')
    mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod);return mod

class EvaluationControls(unittest.TestCase):
    def test_multiweek_forecast_keeps_every_observation(self):
        forecast=load('forecasting')
        result=forecast.backtest(forecast.sample_data(),horizon=14)
        self.assertEqual(len(result['predictions']),result['forecasted_days'])
        self.assertEqual(result['forecasted_days'],84)
        with self.assertRaises(ValueError):forecast.metrics([1,2],[1])
        self.assertIsNone(forecast.metrics([0,0],[1,1])['wape'])
        rows=forecast.sample_data();del rows[3]
        with self.assertRaises(ValueError):forecast.backtest(rows)

    def test_remittance_identity_and_tolerance_controls(self):
        report=load('commercial-reporting')
        remits=[dict(report.REMITTANCES[0]),dict(report.REMITTANCES[1])]
        remits[1]['id']=remits[0]['id']
        with self.assertRaises(ValueError):report.reconcile(report.INVOICES,remits)
        with self.assertRaises(ValueError):report.reconcile(report.INVOICES,[],Decimal('-.01'))

    def test_incremental_effect_uses_incremental_comparator(self):
        causal=load('clustering-noncompliance')
        self.assertEqual(causal.run()['email_call_vs_email']['wald_ratio'],.1)
        with self.assertRaises(ValueError):causal.wald_comparison(.68,.4,.66,.4)

    def test_frame_accounting_and_bounded_queue(self):
        video=load('dynamic-zoom')
        for policy in ['fifo','latest']:
            result=video.simulate(policy)
            processed={r['frame_id'] for r in result['frames']}
            dropped=set(result['dropped_ids'])
            self.assertFalse(processed&dropped)
            self.assertEqual(processed|dropped,set(range(60)))
        latest=video.simulate('latest')
        self.assertLess(latest['mean_latency_ms'],video.simulate('fifo')['mean_latency_ms'])
        self.assertEqual(latest['peak_waiting_frames'],1)

if __name__=='__main__':unittest.main()
