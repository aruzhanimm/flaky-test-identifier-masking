import unittest
from sklearn.feature_extraction.text import TfidfVectorizer
from run_extension_v5_mechanism import CONDITIONS, count_metrics, path_differences


class MechanismTests(unittest.TestCase):
    def test_contingency_metric_counts_false_predictions_and_absent_class(self):
        got = count_metrics(['A', 'A', 'B', 'B'], ['A', 'B', 'B', 'C'], ['A', 'B', 'C'])
        self.assertEqual(got['confusion_matrix'], [[1, 1, 0], [0, 1, 1], [0, 0, 0]])
        self.assertAlmostEqual(got['pooled_macro_f1'], (2 / 3 + 1 / 2 + 0) / 3)
        self.assertEqual(got['accuracy'], .5)
        self.assertEqual(got['classification_report']['C']['support'], 0)

    def test_all_paths_reach_endpoint_without_dropping_interaction(self):
        scores = dict(zip(CONDITIONS, [.8, .7, .5, .6, .9, .3, .2, .4]))
        paths = path_differences(scores)
        self.assertEqual(len(paths), 6)
        for path in paths:
            self.assertEqual(path['steps'][0]['from'], 'OOO')
            self.assertEqual(path['steps'][-1]['to'], 'MMM')
            self.assertAlmostEqual(path['total_delta'], -.4)
        self.assertNotEqual(paths[0]['steps'][0]['delta_macro_f1'], paths[-1]['steps'][0]['delta_macro_f1'])

    def test_frozen_feature_mapping_does_not_learn_evaluation_tokens(self):
        vectorizer = TfidfVectorizer()
        vectorizer.fit(['exception classoriginal', 'failure classoriginal'])
        baseline = dict(vectorizer.vocabulary_)
        matrix = vectorizer.transform(['exception TEST_CLASS targetonlytoken'])
        self.assertEqual(vectorizer.vocabulary_, baseline)
        self.assertNotIn('targetonlytoken', vectorizer.vocabulary_)
        self.assertNotIn('test_class', vectorizer.vocabulary_)
        self.assertEqual(matrix.nnz, 1)


if __name__ == '__main__':
    unittest.main()
