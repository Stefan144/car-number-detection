import torch
from argus.model import load_model
from cnd.ocr.converter import strLabelConverter


class Predictor:
    def __init__(self, model_path, device="cpu"):
        self.model = load_model(model_path, device=device)
        self.alphabet = "ABEKMHOPCTYX" + "".join([str(i) for i in range(10)]) + "-"
        self.converter = strLabelConverter(self.alphabet)

    def predict(self, images):
        logits = self.model.predict(images)
        len_images = torch.IntTensor([logits.size(0)] * logits.size(1))

        _, preds = logits.max(2)
        preds = preds.transpose(1, 0).contiguous().view(-1)
        text = self.converter.decode(preds, len_images)
        return text
