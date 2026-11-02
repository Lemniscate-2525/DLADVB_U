from transformers import AutoModelForSequenceClassification

from peft import LoraConfig
from peft import TaskType
from peft import get_peft_model

from config.config import (
    MODEL_NAME,
    NUM_LABELS,
    LORA_RANK,
    LORA_ALPHA,
    LORA_DROPOUT,
    TARGET_MODULES,
)


def build_model():

    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=NUM_LABELS
    )

    peft_config = LoraConfig(
        task_type=TaskType.SEQ_CLS,
        inference_mode=False,
        r=LORA_RANK,
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        target_modules=TARGET_MODULES,
        bias="none"
    )

    model = get_peft_model(
        model,
        peft_config
    )

    trainable = sum(
        p.numel()
        for p in model.parameters()
        if p.requires_grad
    )

    total = sum(
        p.numel()
        for p in model.parameters()
    )

    print(f"\nTrainable Parameters : {trainable:,}")
    print(f"Total Parameters     : {total:,}")
    print(f"Update Ratio         : {100*trainable/total:.4f}%\n")

    model.print_trainable_parameters()

    return model
