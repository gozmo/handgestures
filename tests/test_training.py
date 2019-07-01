from unittest import TestCase
import unittest
from handsignals import main


class InteractionsTests(TestCase):
    @unittest.skip("skip")
    def test_training(self):
        main.train(epochs=1, learning_rate=0.0001, batch_size=8, resume=False)

    @unittest.skip("skip")
    def test_resume_training(self):
        main.train(epochs=1, learning_rate=0.0001, batch_size=8, resume=True)

    def test_active_learning(self):
        self._given_unlabeled_images()
        main.active_learning()

    def test_aided_annotation(self):
        self._given_unlabeled_images()
        main.set_aided_annotation_batch_size(4)
        main.aided_annotation()

    def _given_unlabeled_images(self):
        pass

    def test_results(self):
        main.results(62)

    @unittest.skip("skip")
    def test_annotate(self):
        self.assertTrue(False)


unittest.main()
