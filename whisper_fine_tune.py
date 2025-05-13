import os
from os.path import isdir
from datasets import load_dataset, load_from_disk, Audio
from transformers import WhisperProcessor, WhisperForConditionalGeneration
from transformers import Seq2SeqTrainingArguments, Seq2SeqTrainer
from torch.nn.utils.rnn import pad_sequence
import torch
from jiwer import wer
import wandb

# -------- WANDB Setup --------
wandb.login()  # Run this once with your API key

wandb.init(
        project="whisper-finetune-360", 
        entity="dtian",
        config={
          "sampling_rate": 16000,
          "num_epochs": 3
        }
)

sampling_rate = wandb.config.sampling_rate
num_epochs = wandb.config.num_epochs

# -------- Dataset Loading --------
if isdir("librispeech"):
    librispeech = load_from_disk("librispeech")
else:
    librispeech = load_dataset("librispeech_asr", "clean", split="train.360")
    librispeech.save_to_disk("librispeech")

# -------- Processor and Model --------
model_name = "openai/whisper-tiny"
processor = WhisperProcessor.from_pretrained(model_name)
model = WhisperForConditionalGeneration.from_pretrained(model_name)
model.config.forced_decoder_ids = None  # disable language token enforcement
model.gradient_checkpointing_enable()  # Save memory on T4

# -------- Audio Preprocessing --------

if isdir("librispeech_preprocessed_"+str(sampling_rate)):
    librispeech = load_from_disk("librispeech_preprocessed_"+str(sampling_rate))
else:
    librispeech = librispeech.cast_column("audio", Audio(sampling_rate=sampling_rate))
    librispeech.save_to_disk("librispeech_preprocessed_"+str(sampling_rate))

def prepare_example(batch):
    audio = batch["audio"]
    inputs = processor(audio["array"], sampling_rate=sampling_rate)
    with processor.as_target_processor():
        labels = processor(batch["text"]).input_ids
    batch["input_features"] = inputs.input_features[0]
    batch["labels"] = labels
    return batch

librispeech = librispeech.map(prepare_example, remove_columns=librispeech.column_names)

# -------- Data Collator --------
def data_collator(batch):
    input_features = torch.stack([torch.tensor(example["input_features"]) for example in batch])
    labels = [torch.tensor(example["labels"]) for example in batch]
    labels = pad_sequence(labels, batch_first=True, padding_value=-100)
    return {"input_features": input_features, "labels": labels}

# -------- Metric Computation --------
def compute_metrics(pred):
    pred_ids = pred.predictions
    label_ids = pred.label_ids
    pred_str = processor.batch_decode(pred_ids, skip_special_tokens=True)
    label_str = processor.batch_decode(label_ids, skip_special_tokens=True)
    return {"wer": wer(label_str, pred_str)}

# -------- Training Arguments --------
# train-clean-360 has ~104,000 samples → ~6,500 steps per epoch (with effective batch size 16)

steps_per_epoch = 6500

total_steps = steps_per_epoch * num_epochs  # ~19,500

training_args = Seq2SeqTrainingArguments(
    output_dir="./whisper-finetuned-360",
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,  # Effective batch size = 16
    learning_rate=1e-5,
    warmup_steps=500,
    max_steps=total_steps,
    logging_steps=100,
    save_steps=1000,
    eval_steps=1000,
    save_total_limit=2,
    evaluation_strategy="steps",
    predict_with_generate=True,
    fp16=torch.cuda.is_available(),
    report_to="wandb",  
    logging_dir="./logs",
)

# -------- Trainer Setup --------
trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=librispeech,
    tokenizer=processor.feature_extractor,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
)

# -------- Train --------
trainer.train()

# -------- Save Fine-tuned Model --------
model.save_pretrained("whisper-finetuned-360")
processor.save_pretrained("whisper-finetuned-360")

# Log as W&B artifact
save_dir = "whisper-finetuned-360"
artifact = wandb.Artifact("whisper-tiny-finetuned-360", type="model")
artifact.add_dir(save_dir)
wandb.log_artifact(artifact)

wandb.finish()
