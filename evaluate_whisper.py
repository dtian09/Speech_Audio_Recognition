# evaluate_whisper_wandb.py

import torch
from datasets import load_dataset, Audio
from transformers import WhisperProcessor, WhisperForConditionalGeneration
from jiwer import wer
from tqdm import tqdm
import wandb

# --------------------------
# CONFIG: Customize here
# --------------------------
PROJECT_NAME = "whisper-finetune-360"
ARTIFACT_NAME = "your-username/whisper-finetune-360/whisper-tiny-finetuned-360:latest"  # Update this
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 1  # Increase for speed if using larger GPU
# --------------------------

# Step 1: Load model from W&B
wandb.login()
run = wandb.init(project=PROJECT_NAME)
artifact = run.use_artifact(ARTIFACT_NAME, type="model")
artifact_dir = artifact.download()

processor = WhisperProcessor.from_pretrained(artifact_dir)
model = WhisperForConditionalGeneration.from_pretrained(artifact_dir)
model.to(DEVICE)
model.eval()
model.config.forced_decoder_ids = None

# Step 2: Load and preprocess test-clean
print("Loading LibriSpeech test-clean...")
test_dataset = load_dataset("librispeech_asr", "clean", split="test")
test_dataset = test_dataset.cast_column("audio", Audio(sampling_rate=16000))

def preprocess(example):
    audio = example["audio"]
    inputs = processor(audio["array"], sampling_rate=16000, return_tensors="pt")
    example["input_features"] = inputs.input_features[0]
    example["reference"] = example["text"]
    return example

test_dataset = test_dataset.map(preprocess)

# Step 3: Evaluate
preds, refs = [], []

print("Evaluating...")
for sample in tqdm(test_dataset, desc="Transcribing"):
    input_features = sample["input_features"].unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        predicted_ids = model.generate(input_features)
    pred = processor.batch_decode(predicted_ids, skip_special_tokens=True)[0].lower()
    ref = sample["reference"].lower()
    preds.append(pred)
    refs.append(ref)

# Step 4: Compute WER
error = wer(refs, preds)
print(f"\nWER on test-clean: {error:.3f}")

run.finish()
