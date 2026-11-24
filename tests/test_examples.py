import importlib.util
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[1]
def load(name, project):
    spec = importlib.util.spec_from_file_location(name, ROOT/'projects'/project/'demo.py')
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

event = load('event', 'event-prediction')
ml = load('ml', 'ml-from-scratch')
opt = load('opt', 'optimization')

class EventTests(unittest.TestCase):
    def test_as_of_resolution_and_future_label(self):
        records, events = event.sample_data()
        row = event.build_rows(records, events)[0]
        self.assertEqual(row['event_count'], 1)
        self.assertEqual(row['currently_blocked'], 0)
        self.assertEqual(row['last_resolve_date'], '2023-12-31')
        self.assertEqual(row['target_event_flag'], 1)
        self.assertEqual(row['operation_code'], 'code1')
        self.assertEqual(row['numeric_feature'], 10)

    def test_validation_values_cannot_change_fit(self):
        records, events = event.sample_data()
        rows = event.build_rows(records, events)
        first = event.split_and_scale(rows, '2024-01-15')[2]
        rows[-1]['numeric_feature'] = 1000000
        second = event.split_and_scale(rows, '2024-01-15')[2]
        self.assertEqual(first, second)
        self.assertEqual(first['mean'], 15)

    def test_boundary_and_purge(self):
        records, events = event.sample_data()
        events.append({'id':'B','date':'2024-01-10','kind':'event'})
        self.assertEqual(event.build_rows(records, events)[1]['target_event_flag'], 0)
        records[1]['record_date'] = '2024-01-10'
        with self.assertRaises(ValueError):
            event.split_and_scale(event.build_rows(records, events), '2024-01-15')

class ClassifierTests(unittest.TestCase):
    def test_known_boundary(self):
        xs, ys = [[-2],[-1],[1],[2]], [0,0,1,1]
        test = [[-1.5],[1.5]]
        self.assertEqual(ml.knn_predict(xs,ys,test,1), [0,1])
        self.assertEqual(ml.logistic_predict(ml.fit_logistic(xs,ys),test), [0,1])
        self.assertEqual(ml.stump_predict(ml.fit_stump(xs,ys),test), [0,1])

    def test_probability_is_finite(self):
        self.assertTrue(0 <= ml.sigmoid(-100000) <= 1)
        self.assertTrue(0 <= ml.sigmoid(100000) <= 1)

class OptimizationTests(unittest.TestCase):
    def test_feasible_and_matches_known_optimum(self):
        score, assignment, loads = opt.solve()
        self.assertEqual(score, 28)
        self.assertTrue(all(load <= 4 for load in loads))
        self.assertEqual(len(assignment), len(opt.PROCEDURES))
        self.assertGreaterEqual(score, opt.first_fit())

if __name__ == '__main__': unittest.main()

