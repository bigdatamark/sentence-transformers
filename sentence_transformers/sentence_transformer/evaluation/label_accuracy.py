# [MODULE DOC] Label accuracy evaluator: measures classification accuracy of a softmax classification head (such as the
# SoftmaxLoss module) on a labeled data loader. Optionally appends results to a CSV.
from __future__ import annotations

import csv
import logging
import os
from typing import TYPE_CHECKING

import torch
from torch.utils.data import DataLoader

from sentence_transformers.base.evaluation.evaluator import BaseEvaluator
from sentence_transformers.util import batch_to_device

if TYPE_CHECKING:
    from sentence_transformers.sentence_transformer.model import SentenceTransformer

logger = logging.getLogger(__name__)


class LabelAccuracyEvaluator(BaseEvaluator):
    # [CLASS DOC] Evaluator holding a DataLoader of labeled examples, a name, the softmax model (a module returning
    # (representations, logits) when called with features and labels None), write_csv flag, CSV file name and
    # headers. Primary metric is "accuracy".
    """
    Evaluate a model based on its accuracy on a labeled dataset

    This requires a model with LossFunction.SOFTMAX

    The results are written in a CSV. If a CSV already exists, then values are appended.
    """

    def __init__(self, dataloader: DataLoader, name: str = "", softmax_model=None, write_csv: bool = True):
        # [FUNCTION DOC] Store the data loader, name, softmax model and CSV settings; the CSV file name includes "_name" if a name is
        # given; set primary_metric to "accuracy".
        """
        Constructs an evaluator for the given dataset

        Args:
            dataloader (DataLoader): the data for the evaluation
        """
        super().__init__()
        self.dataloader = dataloader
        self.name = name
        self.softmax_model = softmax_model

        if name:
            name = "_" + name

        self.write_csv = write_csv
        self.csv_file = "accuracy_evaluation" + name + "_results.csv"
        self.csv_headers = ["epoch", "steps", "accuracy"]
        self.primary_metric = "accuracy"

    def __call__(
        self, model: SentenceTransformer, output_path: str | None = None, epoch: int = -1, steps: int = -1
    ) -> dict[str, float]:
        # [FUNCTION DOC] Put the model in eval mode, log a header, set the data loader's collate function to the model's legacy
        # smart batching collate, then for each batch move every feature dict and the labels to the model device,
        # run the softmax model without gradients and count predictions whose argmax equals the label. Accuracy is
        # correct divided by total (ZeroDivisionError on an empty loader). Logs it, optionally writes/appends a CSV
        # row (epoch, steps, accuracy), prefixes the metric with the name, stores it in the model card data and
        # returns the dict. Side effects: mutates the data loader's collate function and the model's train/eval mode.
        model.eval()
        total = 0
        correct = 0

        if epoch != -1:
            if steps == -1:
                out_txt = f" after epoch {epoch}:"
            else:
                out_txt = f" in epoch {epoch} after {steps} steps:"
        else:
            out_txt = ":"

        logger.info("Evaluation on the " + self.name + " dataset" + out_txt)
        self.dataloader.collate_fn = model.smart_batching_collate
        for step, batch in enumerate(self.dataloader):
            features, label_ids = batch
            for idx in range(len(features)):
                features[idx] = batch_to_device(features[idx], model.device)
            label_ids = label_ids.to(model.device)
            with torch.no_grad():
                _, prediction = self.softmax_model(features, labels=None)

            total += prediction.size(0)
            correct += torch.argmax(prediction, dim=1).eq(label_ids).sum().item()
        accuracy = correct / total

        logger.info(f"Accuracy: {accuracy:.4f} ({correct}/{total})\n")

        if output_path is not None and self.write_csv:
            os.makedirs(output_path, exist_ok=True)
            csv_path = os.path.join(output_path, self.csv_file)
            if not os.path.isfile(csv_path):
                with open(csv_path, newline="", mode="w", encoding="utf-8") as f:
                    writer = csv.writer(f)
                    writer.writerow(self.csv_headers)
                    writer.writerow([epoch, steps, accuracy])
            else:
                with open(csv_path, newline="", mode="a", encoding="utf-8") as f:
                    writer = csv.writer(f)
                    writer.writerow([epoch, steps, accuracy])

        metrics = {"accuracy": accuracy}
        metrics = self.prefix_name_to_metrics(metrics, self.name)
        self.store_metrics_in_model_card_data(model, metrics, epoch, steps)
        return metrics
