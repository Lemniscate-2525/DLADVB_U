import torch

from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_recall_fscore_support

from config.config import DEVICE


def evaluate(model, dataloader):

    model.eval()

    all_preds = []
    all_labels = []

    with torch.no_grad():

        for batch in dataloader:

            input_ids = batch["input_ids"].to(DEVICE)
            attention_mask = batch["attention_mask"].to(DEVICE)
            labels = batch["labels"].to(DEVICE)

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask
            )

            preds = torch.argmax(outputs.logits, dim=1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    precision, recall, f1, _ = precision_recall_fscore_support(
        all_labels,
        all_preds,
        average="binary",
        zero_division=0
    )

    accuracy = accuracy_score(
        all_labels,
        all_preds
    ) * 100

    return (
        precision,
        recall,
        f1,
        accuracy
    )