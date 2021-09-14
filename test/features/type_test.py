"""
Unit Tests for Feature Type and Learning categories. Some of the base needed classes.
(c) 2020 d373c7
"""
import unittest
import d373c7.features as ft


class LearningCategoriesTest(unittest.TestCase):
    def test_lists(self):
        self.assertNotIn(ft.LEARNING_CATEGORY_LABEL, ft.LEARNING_CATEGORIES_MODEL_INPUT, f'Label should not be input')
        self.assertNotIn(ft.LEARNING_CATEGORY_NONE, ft.LEARNING_CATEGORIES_MODEL_INPUT, f'None should not be in input')
        self.assertNotIn(ft.LEARNING_CATEGORY_NONE, ft.LEARNING_CATEGORIES_MODEL, f'None should not be ib model')

    def test_index(self):
        lc_key = [lc.key for lc in ft.LEARNING_CATEGORIES_MODEL]
        self.assertEqual(len(lc_key), len(set(lc_key)), 'Can not have duplicate indexes')
        lc_key.sort()
        lcs = ft.LEARNING_CATEGORIES_MODEL.copy()
        lcs.sort()
        self.assertListEqual(lc_key, [lc.key for lc in lcs], f'Sort not working we do not seem to be sorting on key')


# TODO include some FeatureType tests
class FeatureTypeTest(unittest.TestCase):
    def test_string(self):
        self.assertIsInstance(ft.FEATURE_TYPE_STRING.name, str, f'expected name to be of type string')
        self.assertIsInstance(ft.FEATURE_TYPE_STRING.key, int, f'expected type to be of type int')
        self.assertIsInstance(ft.FEATURE_TYPE_STRING.precision, int, f'expected precision to be type int')
        self.assertEqual(ft.FEATURE_TYPE_STRING.learning_category, ft.LEARNING_CATEGORY_NONE, f'LC to be None')


def main():
    unittest.main()


if __name__ == '__main__':
    main()
