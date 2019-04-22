import math
import torch
from handsignals.constants import Labels

class PredictionResult:
    def __init__(self, prediction_result, true_label_dist):
        self.distribution = prediction_result.tolist()

        self.prediction_distribution = dict(zip(Labels.get_labels(), self.distribution))

        self.score = self.__get_score(prediction_result)
        self.label = self.__get_label(prediction_result)
        self.active_learning_score = self.__get_active_learning_score()
        self.true_label = self.__get_label(true_label_dist)

    def __get_score(self, prediction_result):
        score, prediction_idx = torch.max(prediction_result, 0)
        return score.item()

    def __get_label(self, prediction_result):
        _, prediction_idx = torch.max(prediction_result,0)
        prediction_idx = prediction_idx.item()
        label = Labels.int_to_label(prediction_idx)
        return label

    def __get_active_learning_score(self):
        nlogn = [n * math.log(n) for n in self.distribution]
        score = -sum(nlogn)
        return score

    def to_json(self):
        return {"score": self.score,
                "distribution": self.distribution,
                "prediction_distribution": self.prediction_distribution,
                "label": self.label,
                "active_learning_score": self.active_learning_score,
                "true_label": self.true_label}

