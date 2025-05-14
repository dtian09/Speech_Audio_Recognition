import os
import pandas as pd
import torch
import torchaudio
import wandb
from datasets import Dataset, Audio
from transformers import (
    WhisperProcessor,
    WhisperForConditionalGeneration,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
    TrainerCallback
)
from torch.nn.utils.rnn import pad_sequence
from jiwer import wer

# -----------------------------------
# Load Dataset from CSV
# -----------------------------------
df = pd.read_csv("data/harvard_metadata.csv")
dataset = Dataset.from_pandas(df)
dataset = dataset.cast_column("file", Audio(sampling_rate=16000))
dataset = dataset.rename_column("file", "audio")

# -----------------------------------
# Split Dataset: Train (0–98), Valid (99–108), Held-out Eval (109)
# -----------------------------------
train_dataset = dataset.select(range(99))
valid_dataset = dataset.select(range(99, 109))
eval_example = dataset[109]  # One held-out for prediction display

# -----------------------------------
# Weights & Biases Init
# -----------------------------------
wandb.login()
wandb.init(
    project="whisper-finetune-harvard",
    config={
        "sampling_rate": 16000,
        "num_epochs": 5,
        "model_name": "openai/whisper-tiny"
    }
)

# -----------------------------------
# Preprocessor and Model
# -----------------------------------
processor = WhisperProcessor.from_pretrained(wandb.config.model_name)

def prepare_example(batch):
    audio = batch["audio"]
    inputs = processor(audio["array"], sampling_rate=wandb.config.sampling_rate)
    with processor.as_target_processor():
        labels = processor(batch["text"]).input_ids
    batch["input_features"] = inputs.input_features[0]
    batch["labels"] = labels
    return batch

# Apply preprocessing
train_dataset = train_dataset.map(prepare_example, remove_columns=train_dataset.column_names)
valid_dataset = valid_dataset.map(prepare_example, remove_columns=valid_dataset.column_names)
eval_example = prepare_example(eval_example)

# Load and configure model
model = WhisperForConditionalGeneration.from_pretrained(wandb.config.model_name)
model.config.forced_decoder_ids = None
model.gradient_checkpointing_enable()

# -----------------------------------
# Data Collator and Evaluation Metric
# -----------------------------------
def data_collator(batch):
    input_features = torch.stack([torch.tensor(e["input_features"]) for e in batch])
    labels = pad_sequence([torch.tensor(e["labels"]) for e in batch], batch_first=True, padding_value=-100)
    return {"input_features": input_features, "labels": labels}

def compute_metrics(pred):
    pred_ids = pred.predictions
    label_ids = pred.label_ids
    pred_str = processor.batch_decode(pred_ids, skip_special_tokens=True)
    label_str = processor.batch_decode(label_ids, skip_special_tokens=True)
    return {"wer": wer(label_str, pred_str)}

# -----------------------------------
# Callback to Print Prediction vs. Target
# -----------------------------------
class EvalCallback(TrainerCallback):
    def on_evaluate(self, args, state, control, **kwargs):
        model = kwargs["model"]
        model.eval()
        input_tensor = torch.tensor(eval_example["audio"]["array"]).unsqueeze(0)
        input_tensor = processor(input_tensor.squeeze().numpy(), sampling_rate=16000, return_tensors="pt").input_features
        input_tensor = input_tensor.to(model.device)
        with torch.no_grad():
            pred_ids = model.generate(input_tensor)
        pred = processor.batch_decode(pred_ids, skip_special_tokens=True)[0]
        print(f"\n>>> Epoch {state.epoch:.1f} Prediction vs Target")
        print("Prediction:", pred)
        print("Target:    ", eval_example["text"])

# -----------------------------------
# Training Setup
# -----------------------------------
training_args = Seq2SeqTrainingArguments(
    output_dir="./whisper-harvard",
    per_device_train_batch_size=2,
    gradient_accumulation_steps=1,
    learning_rate=1e-5,
    evaluation_strategy="epoch",
    save_total_limit=2,
    predict_with_generate=True,
    logging_steps=5,
    num_train_epochs=wandb.config.num_epochs,
    report_to="wandb",
    fp16=torch.cuda.is_available(),
)

trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=valid_dataset,
    tokenizer=processor,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
)

trainer.add_callback(EvalCallback())
trainer.train()

# -----------------------------------
# Save Model + Upload to WandB
# -----------------------------------
model.save_pretrained("whisper-harvard")
processor.save_pretrained("whisper-harvard")

artifact = wandb.Artifact("whisper-tiny-finetuned-harvard", type="model")
artifact.add_dir("whisper-harvard")
wandb.log_artifact(artifact)
wandb.finish()
