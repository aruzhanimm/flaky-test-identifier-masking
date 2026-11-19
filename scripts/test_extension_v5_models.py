"""Checks for the metric denominator and fixed experiment design."""
import unittest
from run_extension_v5_models import metrics_from_counts, pairs, settings


class ModelExtensionChecks(unittest.TestCase):
    def test_present_label_f1_accounts_for_misclassification(self):
        value=metrics_from_counts(['ID','ID'],['ID','TD'],['ID'])
        self.assertAlmostEqual(value['macro_f1'],2/3)
        self.assertEqual(value['per_class']['ID']['fn'],1)

    def test_fixed_denominator_includes_absent_labels(self):
        value=metrics_from_counts(['ID','ID'],['ID','TD'],['ID','NIO','OD','TD'])
        self.assertAlmostEqual(value['macro_f1'],1/6)
        self.assertEqual(value['per_class']['TD']['fp'],1)
        self.assertEqual(value['confusion_matrix'][0], [1,0,0,1])

    def test_misaligned_input_is_rejected(self):
        with self.assertRaises(ValueError):metrics_from_counts(['ID'],[],['ID'])

    def test_fixed_grid_has_16_unique_conditions(self):
        conditions=[f'{r}_{m}_{v}' for r,m in pairs() for v in ('original','masked')]
        self.assertEqual(len(conditions),16)
        self.assertEqual(len(set(conditions)),16)

    def test_character_representation_does_not_inherit_word_tokenizer(self):
        result=settings({},'char')
        self.assertEqual(result['analyzer'],'char')
        self.assertEqual(result['ngram_range'],(3,5))
        self.assertEqual(result['max_features'],30000)
        self.assertNotIn('token_pattern',result)


if __name__=='__main__':unittest.main()
