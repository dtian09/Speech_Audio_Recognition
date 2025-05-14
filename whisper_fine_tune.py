import os
from os.path import isdir
from datasets import load_dataset, load_from_disk, Audio
from transformers import (
    WhisperProcessor,
    WhisperForConditionalGeneration,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
    TrainerCallback
)
from torch.nn.utils.rnn import pad_sequence
import torch
from jiwer import wer
import wandb

# -----------------------------
# WANDB Setup
# -----------------------------
wandb.login()
wandb.init(
    project="whisper-finetune-360",
    entity="dtian",
    config={
        "sampling_rate": 16000,
        "num_epochs": 3,
        "model_name": "openai/whisper-tiny"
    }
)

sampling_rate = wandb.config.sampling_rate
num_epochs = wandb.config.num_epochs
model_name = wandb.config.model_name

# -----------------------------
# Load Processor First
# -----------------------------
processor = WhisperProcessor.from_pretrained(model_name)

# -----------------------------
# Prepare Preprocessing Function
# -----------------------------
def prepare_example(batch):
    audio = batch["audio"]
    inputs = processor(audio["array"], sampling_rate=sampling_rate)
    with processor.as_target_processor():
        labels = processor(batch["text"]).input_ids
    batch["input_features"] = inputs.input_features[0]
    batch["labels"] = labels
    return batch

# -----------------------------
# Load and Preprocess Train Data
# -----------------------------
preprocessed_path = f"librispeech_preprocessed_{sampling_rate}"
raw_path = "librispeech"

if isdir(preprocessed_path):
    librispeech = load_from_disk(preprocessed_path)
else:
    if isdir(raw_path):
        raw_dataset = load_from_disk(raw_path)
    else:
        raw_dataset = load_dataset(
                      "librispeech_asr",
                      "clean",
                      split="validation",
                      #split="train.360",
                      cache_dir="/content/datasets/"
                    )

        raw_dataset.save_to_disk(raw_path)

    raw_dataset = raw_dataset.cast_column("audio", Audio(sampling_rate=sampling_rate))
    librispeech = raw_dataset.map(prepare_example, remove_columns=raw_dataset.column_names)
    librispeech.save_to_disk(preprocessed_path)

# For debugging: use only 10 samples
librispeech = librispeech.select(range(10))

# -----------------------------
# Load and Preprocess Dev-clean for Validation
# -----------------------------
dev_clean = load_dataset("librispeech_asr", "clean", split="validation")
dev_clean = dev_clean.cast_column("audio", Audio(sampling_rate=sampling_rate))
dev_clean = dev_clean.map(prepare_example, remove_columns=dev_clean.column_names)

# -----------------------------
# Load Whisper Model
# -----------------------------
model = WhisperForConditionalGeneration.from_pretrained(model_name)
model.config.forced_decoder_ids = None
model.gradient_checkpointing_enable()

# -----------------------------
# Data Collator
# -----------------------------
def data_collator(batch):
    input_features = torch.stack([torch.tensor(e["input_features"]) for e in batch])
    labels = [torch.tensor(e["labels"]) for e in batch]
    labels = pad_sequence(labels, batch_first=True, padding_value=-100)
    return {"input_features": input_features, "labels": labels}

# -----------------------------
# WER Evaluation Metric
# -----------------------------
def compute_metrics(pred):
    pred_ids = pred.predictions
    label_ids = pred.label_ids
    pred_str = processor.batch_decode(pred_ids, skip_special_tokens=True)
    label_str = processor.batch_decode(label_ids, skip_special_tokens=True)
    return {"wer": wer(label_str, pred_str)}

# -----------------------------
# Training Args (initial)
# -----------------------------
training_args = Seq2SeqTrainingArguments(
    output_dir="./whisper-finetuned-360",
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,
    learning_rate=1e-5,
    warmup_steps=500,
    logging_steps=100,
    save_steps=1000,
    save_total_limit=2,
    evaluation_strategy="steps",
    predict_with_generate=True,
    fp16=torch.cuda.is_available(),
    report_to="wandb",
    logging_dir="./logs"
)

# -----------------------------
# Compute Dynamic Step Sizes
# -----------------------------
effective_batch_size = (
    training_args.per_device_train_batch_size * training_args.gradient_accumulation_steps
)
steps_per_epoch = len(librispeech) // effective_batch_size
total_steps = steps_per_epoch * num_epochs

training_args.eval_steps = steps_per_epoch
training_args.max_steps = total_steps

# -----------------------------
# Callback: Print Prediction Each Epoch
# -----------------------------
class PrintFirstExampleCallback(TrainerCallback):
    def __init__(self, processor, eval_dataset, device):
        self.processor = processor
        self.eval_dataset = eval_dataset
        self.device = device

    def on_evaluate(self, args, state, control, **kwargs):
        model = kwargs["model"]
        model.eval()
        example = self.eval_dataset[0]
        input_tensor = torch.tensor(example["input_features"]).unsqueeze(0).to(self.device)
        with torch.no_grad():
            pred_ids = model.generate(input_tensor)
        pred = self.processor.batch_decode(pred_ids, skip_special_tokens=True)[0]
        target = self.processor.batch_decode([example["labels"]], skip_special_tokens=True)[0]
        print(f"\n Epoch {state.epoch:.1f}")
        print("Prediction:", pred)
        print("Target:    ", target)

# -----------------------------
# Trainer Setup
# -----------------------------
trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=librispeech,
    eval_dataset=dev_clean,
    tokenizer=processor.feature_extractor,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
)

trainer.add_callback(PrintFirstExampleCallback(
    processor=processor,
    eval_dataset=dev_clean,
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
))

# -----------------------------
# Train the Model
# -----------------------------
trainer.train()

# -----------------------------
# Save Model and Processor
# -----------------------------
save_dir = "whisper-finetuned-360"
model.save_pretrained(save_dir)
processor.save_pretrained(save_dir)

# -----------------------------
# Upload to W&B
# -----------------------------
artifact = wandb.Artifact("whisper-tiny-finetuned-360", type="model", metadata={"epochs": num_epochs})
artifact.add_dir(save_dir)
wandb.log_artifact(artifact, aliases=["latest", f"epoch{num_epochs}"])
wandb.finish()