import os
import gc
import torch
import pandas as pd
from PIL import Image
from tqdm import tqdm
from datasets import Dataset
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler, autocast
from sklearn.metrics import accuracy_score, classification_report
from transformers import Pix2StructProcessor, Pix2StructForConditionalGeneration


# --- Configuration ---
IMAGE_FOLDER = "/home/shossain/inhouse_project/HG_EGG/HG_Data/training_data"  # Replace with the actual path to your image folder
CSV_FILE = "/home/shossain/inhouse_project/HG_EGG/HG_Data/training_data_label.csv"      # Replace with the actual path to your CSV file
MODEL_NAME = "google/pix2struct-base"
BATCH_SIZE = 16
GRADIENT_ACCUMULATION_STEPS = 4  # Accumulate gradients over 8 micro-batches (effective batch size = 8)
LEARNING_RATE = 5e-5
NUM_EPOCHS = 100
OUTPUT_DIR = "/home/shossain/inhouse_project/HG_EGG/HG_Data/pix2struct_eggshell_digits"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

# --- 1. Load and Prepare Dataset ---
print("Loading and preparing dataset...")
df = pd.read_csv(CSV_FILE)

def load_image_and_text(row):
    image_path = os.path.join(IMAGE_FOLDER, row['Filename'])
    try:
        image = Image.open(image_path).convert("RGB")
    except FileNotFoundError:
        print(f"Error: Image not found at {image_path}")
        return None
    text = "0"+str(row['Label'])  # Ensure label is treated as a string
    return {"image": image, "text": text}

processed_data = df.apply(load_image_and_text, axis=1).dropna()
dataset = Dataset.from_list(processed_data.tolist())
dataset = dataset.train_test_split(test_size=0.2, shuffle=True, seed=42)
train_test_dataset = dataset["train"].train_test_split(test_size=0.1, shuffle=True, seed=42)
train_dataset = train_test_dataset["train"]
val_dataset = train_test_dataset["test"]
test_dataset = dataset["test"]

# --- 2. Load Processor and Model ---
print("Loading processor and model...")
processor = Pix2StructProcessor.from_pretrained(MODEL_NAME)
model = Pix2StructForConditionalGeneration.from_pretrained(MODEL_NAME).to(DEVICE)

def collate_fn(batch):
    images = [item["image"] for item in batch]
    texts = [item["text"] for item in batch]
    encoding = processor(images=images, return_tensors="pt", padding="max_length", max_length=2048, truncation=True) # Process images only

    processed_texts = processor(text=texts, return_tensors="pt", padding="max_length", max_length=7, truncation=True) # Process texts separately

    pixel_values = encoding.flattened_patches.to(DEVICE)
    attention_mask = encoding.attention_mask.to(DEVICE)
    labels = processed_texts.input_ids.to(DEVICE)
    decoder_attention_mask = processed_texts.attention_mask.to(DEVICE)

    # Replace padding token id (-100) with the pad token id of the processor
    labels[labels == processor.tokenizer.pad_token_id] = -100

    return {
        "pixel_values": pixel_values,
        "attention_mask": attention_mask,
        "labels": labels,
        "decoder_attention_mask": decoder_attention_mask,
    }


train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)
val_dataloader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn)
test_dataloader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn)

# --- 3. Training Loop ---
print("Starting training...")
optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)
scaler = GradScaler()  # Initialize GradScaler for AMP
accumulation_steps = GRADIENT_ACCUMULATION_STEPS

model.train()
optimizer.zero_grad()
best_val_loss = float('inf')  # Initialize best validation loss
patience = 10  # Number of epochs to wait for improvement
no_improve_count = 0

for epoch in range(NUM_EPOCHS):
    total_loss = 0
    for i, batch in enumerate(tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{NUM_EPOCHS}")):
        optimizer.zero_grad()
        with autocast(dtype=torch.bfloat16):  # Enable autocasting for mixed precision
            outputs = model(
                flattened_patches=batch["pixel_values"],
                attention_mask=batch["attention_mask"],
                labels=batch["labels"],
                decoder_attention_mask=batch["decoder_attention_mask"]
            )
            loss = outputs.loss / accumulation_steps

        scaler.scale(loss).backward()  # Scale the loss

        if (i + 1) % accumulation_steps == 0:
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()               # Update the scale for the next iteration

        total_loss += loss.item() * accumulation_steps

    avg_loss = total_loss / len(train_dataloader)

    # --- 3.1 Validation ---
    model.eval()
    val_loss = 0
    with torch.no_grad():
        for batch in val_dataloader:
            with autocast(dtype=torch.bfloat16):
                outputs = model(
                    flattened_patches=batch["pixel_values"],
                    attention_mask=batch["attention_mask"],
                    labels=batch["labels"],
                    decoder_attention_mask=batch["decoder_attention_mask"]
                )
                loss = outputs.loss
            val_loss += loss.item()
    avg_val_loss = val_loss / len(val_dataloader)
    print(f"Epoch {epoch+1}/{NUM_EPOCHS}, Average Validation Loss: {avg_val_loss:.4f}")
    model.train()

    # --- 3.2 Early Stopping ---
    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        no_improve_count = 0
        torch.save(model.state_dict(), os.path.join(OUTPUT_DIR, "best_model.pth"))  # Save the best model
        print("Saving best model...")
    else:
        no_improve_count += 1
        if no_improve_count >= patience:
            print("Early stopping triggered.")
            break

    torch.cuda.empty_cache()

# --- 4. Save Trained Model ---
print("Saving trained model...")
model.save_pretrained(OUTPUT_DIR)
processor.save_pretrained(OUTPUT_DIR)
print(f"Trained model saved to {OUTPUT_DIR}")

# --- 5. Evaluation ---
print("Starting evaluation...")
model.eval()
all_predictions = []
all_ground_truths = []

with torch.no_grad():
    for batch in tqdm(test_dataloader, desc="Evaluating"):
        flattened_patches = batch["pixel_values"].to(DEVICE)
        attention_mask = batch["attention_mask"].to(DEVICE)
        labels = batch["labels"].to(DEVICE)

        with autocast(): # Enable autocasting for inference as well (optional but recommended)
            generated_ids = model.generate(
                flattened_patches=flattened_patches,
                attention_mask=attention_mask,
                max_length=7
            )
        predictions = processor.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)

        labels_cpu = labels.cpu().numpy()
        batch_ground_truths = []
        for label_sequence in labels_cpu:
            try:
                decoded_label = processor.tokenizer.decode(label_sequence, skip_special_tokens=True)
                batch_ground_truths.append(decoded_label)
            except OverflowError as e:
                print(f"OverflowError during decoding: {e}")
                print(f"Problematic label sequence: {label_sequence}")
                batch_ground_truths.append("[ERROR]")
            except Exception as e:
                print(f"An unexpected error occurred during decoding: {e}")
                print(f"Problematic label sequence: {label_sequence}")
                batch_ground_truths.append("[ERROR]")

        all_predictions.extend(predictions)
        all_ground_truths.extend(batch_ground_truths)
        torch.cuda.empty_cache()

print(all_predictions)
print(all_ground_truths)

# --- 6. Calculate and Display Metrics ---
print("Calculating evaluation metrics...")
accuracy = accuracy_score(all_ground_truths, all_predictions)
print(f"Accuracy on the test set: {accuracy:.4f}")
