import torch._dynamo

torch._dynamo.disable()
torch._dynamo.config.suppress_errors = True

import os
import time

import torch
from torch.amp import autocast, GradScaler
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import get_linear_schedule_with_warmup

from config.config import (
    DEVICE,
    TRAIN_FILE,
    VAL_FILE,
    BATCH_SIZE,
    LEARNING_RATE,
    EPOCHS,
    OUTPUT_DIR,
    NUM_WORKERS,
    PIN_MEMORY,
)

from training.dataset import LogDataset
from training.evaluate import evaluate
from models.bert_classifier import build_model


def train():

    train_dataset = LogDataset(TRAIN_FILE)
    val_dataset = LogDataset(VAL_FILE)

    is_cuda = (DEVICE.type == "cuda")
    workers = NUM_WORKERS if is_cuda else 0
    pin_mem = PIN_MEMORY if is_cuda else False
    use_persistent = is_cuda and workers > 0

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=workers,
        pin_memory=pin_mem,
        persistent_workers=use_persistent,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=workers,
        pin_memory=pin_mem,
        persistent_workers=use_persistent,
    )

    model = build_model().to(DEVICE)

    scaler = GradScaler("cuda", enabled=is_cuda)

    optimizer = AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=LEARNING_RATE,
    )

    total_steps = len(train_loader) * EPOCHS

    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(0.1 * total_steps) if total_steps > 0 else 0,
        num_training_steps=max(total_steps, 1),
    )

    best_f1 = -1.0

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    total_start = time.time()

    for epoch in range(EPOCHS):

        print(f"\nEpoch {epoch + 1}/{EPOCHS}")

        model.train()

        epoch_loss = 0
        correct = 0
        total = 0

        start = time.time()

        for batch in train_loader:

            input_ids = batch["input_ids"].to(DEVICE)
            attention_mask = batch["attention_mask"].to(DEVICE)
            labels = batch["labels"].to(DEVICE)

            optimizer.zero_grad(set_to_none=True)

            if is_cuda:
                with autocast(device_type="cuda"):
                    outputs = model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        labels=labels,
                    )
                    loss = outputs.loss
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels,
                )
                loss = outputs.loss
                loss.backward()
                optimizer.step()

            scheduler.step()

            epoch_loss += loss.item()

            preds = torch.argmax(outputs.logits, dim=1)

            correct += (preds == labels).sum().item()

            total += labels.size(0)

        train_acc = 100 * correct / max(total, 1)
        avg_loss = epoch_loss / max(len(train_loader), 1)

        precision, recall, f1, val_acc = evaluate(
            model,
            val_loader,
        )

        print("=" * 60)
        print(f"Train Loss      : {avg_loss:.4f}")
        print(f"Train Accuracy  : {train_acc:.2f}%")
        print(f"Validation Acc  : {val_acc:.2f}%")
        print(f"Precision       : {precision:.4f}")
        print(f"Recall          : {recall:.4f}")
        print(f"F1 Score        : {f1:.4f}")
        print(f"Epoch Time      : {time.time() - start:.2f}s")

        if f1 >= best_f1:

            best_f1 = f1

            model.save_pretrained(OUTPUT_DIR)

            train_dataset.tokenizer.save_pretrained(
                OUTPUT_DIR
            )

            print("Best model saved.")

    print("=" * 60)
    print("Training Finished.")
    print(f"Total Time : {(time.time() - total_start) / 60:.2f} Minutes")


if __name__ == "__main__":
    train()
