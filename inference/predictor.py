import time
import torch

from peft import PeftModel
from transformers import AutoTokenizer
from transformers import AutoModelForSequenceClassification

from config.config import (
    DEVICE,
    MODEL_NAME,
    OUTPUT_DIR,
    MAX_SEQ_LEN,
)


class LogPredictor:

    def __init__(self):

        tokenizer_path = OUTPUT_DIR if (OUTPUT_DIR / "tokenizer_config.json").exists() else MODEL_NAME
        self.tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_path
        )

        base_model = AutoModelForSequenceClassification.from_pretrained(
            MODEL_NAME,
            num_labels=2
        )

        if (OUTPUT_DIR / "adapter_config.json").exists():
            self.model = PeftModel.from_pretrained(
                base_model,
                OUTPUT_DIR
            )
        else:
            self.model = base_model

        self.model.to(DEVICE)
        self.model.eval()

    def predict(self, log):

        inputs = self.tokenizer(
            log,
            truncation=True,
            padding="max_length",
            max_length=MAX_SEQ_LEN,
            return_tensors="pt"
        )

        input_ids = inputs["input_ids"].to(DEVICE)
        attention_mask = inputs["attention_mask"].to(DEVICE)

        start = time.perf_counter()

        with torch.no_grad():

            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask
            )

        latency = (time.perf_counter() - start) * 1000

        probs = torch.softmax(
            outputs.logits,
            dim=1
        )[0]

        prediction = torch.argmax(
            probs
        ).item()

        confidence = probs[
            prediction
        ].item()

        return {
            "prediction": prediction,
            "label": "Anomaly" if prediction else "Normal",
            "confidence": round(confidence * 100, 2),
            "latency_ms": round(latency, 2)
        }

    def predict_batch(self, logs):

        if not logs:
            return []

        start = time.perf_counter()

        inputs = self.tokenizer(
            logs,
            truncation=True,
            padding="max_length",
            max_length=MAX_SEQ_LEN,
            return_tensors="pt"
        )

        input_ids = inputs["input_ids"].to(DEVICE)
        attention_mask = inputs["attention_mask"].to(DEVICE)

        with torch.no_grad():
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask
            )

        total_latency = (time.perf_counter() - start) * 1000
        per_item_latency = round(total_latency / len(logs), 2)

        probs = torch.softmax(outputs.logits, dim=1)
        predictions = torch.argmax(probs, dim=1).cpu().tolist()

        results = []
        for i, (log, pred) in enumerate(zip(logs, predictions)):
            conf = probs[i, pred].item()
            results.append({
                "log": log,
                "prediction": pred,
                "label": "Anomaly" if pred else "Normal",
                "confidence": round(conf * 100, 2),
                "latency_ms": per_item_latency
            })

        return results
