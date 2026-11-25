"""Checks for the real-data evaluation contracts; optional analysis environment."""
import importlib.util
from pathlib import Path
import unittest

ROOT=Path(__file__).resolve().parents[1]
def load(slug):
    spec=importlib.util.spec_from_file_location(slug,ROOT/'projects'/slug/'analyze.py')
    mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod);return mod

class DataIsolation(unittest.TestCase):
    def test_identical_wine_inputs_cannot_cross_holdout(self):
        _,x,y,groups,train,test=load('wine-neural-networks').prepare()
        self.assertEqual(len(x),1599)
        self.assertEqual(x.shape[1],11)
        self.assertNotIn('quality',x)
        self.assertFalse(set(groups.iloc[train])&set(groups.iloc[test]))
        self.assertEqual(set(train)|set(test),set(range(len(x))))

    def test_passenger_groups_and_identifiers_are_isolated(self):
        _,x,y,groups,train,test=load('spaceship-titanic').prepare()
        self.assertFalse(set(groups.iloc[train])&set(groups.iloc[test]))
        self.assertNotIn('Transported',x)
        self.assertNotIn('PassengerId',x)
        self.assertNotIn('Name',x)
        self.assertEqual(len(x),8693)

    def test_duplicate_reviews_and_proxy_labels_are_isolated(self):
        _,data,y,groups,train,test,_=load('play-store-reviews').prepare()
        self.assertFalse(set(groups.iloc[train])&set(groups.iloc[test]))
        self.assertFalse(data['score'].eq(3).any())
        self.assertTrue(data['text'].str.len().gt(0).all())
        self.assertTrue(y.equals(data['score'].ge(4).astype(int)))

if __name__=='__main__':unittest.main()
