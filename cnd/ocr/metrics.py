import torch
from argus.metrics import Metric
from cnd.ocr.converter import strLabelConverter
from Levenshtein import distance, jaro


class StringAccuracy(Metric):
    name = "str_accuracy"
    better = "max"

    def __init__(self):
        self.alphabet = "ABEKMHOPCTYX" + "".join([str(i) for i in range(10)]) + "-"
        self.encoder = strLabelConverter(self.alphabet)

    def reset(self):
        self.correct = 0
        self.count = 0

    def compare_two_str(self, a, b):
        cnt = 0
        for pair in zip(a, b):
            if pair[0] == pair[1]:
                cnt += 1
        return cnt

    def update(self, step_output: dict):
        preds = step_output["prediction"]
        targets = step_output["target"]
        min_len = min(len(preds), len(targets))
        for i in range(min_len):
            self.correct += self.compare_two_str(preds[i], targets[i])
        self.count += sum([len(t) for t in targets][:min_len])

    def compute(self):
        if self.count == 0:
            return 0
        return self.correct / self.count


class LevenshteinDistance(Metric):
    name = "avg Levenshtein distance"
    better = "min"

    def __init__(self):
        self.alphabet = "ABEKMHOPCTYX" + "".join([str(i) for i in range(10)]) + "-"
        self.encoder = strLabelConverter(self.alphabet)

    def reset(self):
        self.levdist = 0

    def update(self, step_output: dict):
        preds = step_output["prediction"]
        targets = step_output["target"]
        min_len = min(len(preds), len(targets))
        for i in range(min_len):
            self.levdist += distance(preds[i], targets[i])
        self.levdist /= min_len

    def compute(self):
        return self.levdist


class JaroDistance(Metric):
    name = "avg Jaro distance"
    better = "max"

    def __init__(self):
        self.alphabet = "ABEKMHOPCTYX" + "".join([str(i) for i in range(10)]) + "-"
        self.encoder = strLabelConverter(self.alphabet)

    def reset(self):
        self.jarodist = 0

    def update(self, step_output: dict):
        preds = step_output["prediction"]
        targets = step_output["target"]
        min_len = min(len(preds), len(targets))
        for i in range(min_len):
            self.jarodist += jaro(preds[i], targets[i])
        self.jarodist /= min_len

    def compute(self):
        return self.jarodist

