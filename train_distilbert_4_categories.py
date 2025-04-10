import os
import json
import torch
import warnings
import numpy as np
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification, Trainer, TrainingArguments
from datasets import Dataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from torch.nn import BCEWithLogitsLoss

# ============================
#  Suppress Warnings
# ============================
warnings.filterwarnings("ignore")
os.environ["TRANSFORMERS_NO_ADVISORY_WARNINGS"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# ============================
#  Load Dataset
# ============================
print("🔄 Loading dataset...")

with open("datasets/current_4_category_dataset.json", "r") as f:
    data = json.load(f)

nouns = list(data.keys())

# Convert labels into multi-hot encoded vectors
label_map = {"object": 0, "agent": 1, "surface": 2, "region": 3}
num_labels = len(label_map)

def encode_labels(label_list):
    label_vector = np.zeros(num_labels)
    for label in label_list:
        label_vector[label_map[label]] = 1
    return label_vector.tolist()

labels = [encode_labels(data[noun]) for noun in nouns]

# Split into train/test
train_nouns, test_nouns, train_labels, test_labels = train_test_split(nouns, labels, test_size=0.2, random_state=42)

print("✅ Dataset loaded! First few samples:")
print("📌 Nouns:", train_nouns[:5])
print("📌 Labels:", train_labels[:5])

# Load tokenizer
tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")

# Tokenization function
def tokenize_function(examples):
    return tokenizer(examples["text"], padding="max_length", truncation=True)

# Convert to Hugging Face dataset
train_dataset = Dataset.from_dict({"text": train_nouns, "label": train_labels}).map(tokenize_function, batched=True)
test_dataset = Dataset.from_dict({"text": test_nouns, "label": test_labels}).map(tokenize_function, batched=True)

print("✅ Dataset tokenized!")

# ============================
#  Load Model & Trainer
# ============================
print("🔄 Loading DistilBERT model...")

model = DistilBertForSequenceClassification.from_pretrained("distilbert-base-uncased", num_labels=num_labels, problem_type="multi_label_classification")

training_args = TrainingArguments(
    output_dir="./results",
    evaluation_strategy="epoch",
    save_strategy="no",
    learning_rate=5e-5,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=5,
    weight_decay=0.01,
    logging_dir="./logs",
)

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = (torch.sigmoid(torch.tensor(logits)) > 0.5).int().numpy()
    labels = np.array(labels)
    accuracy = accuracy_score(labels, predictions)
    report = classification_report(labels, predictions, target_names=list(label_map.keys()), zero_division=0)
    return {"accuracy": accuracy, "report": report}

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    compute_metrics=compute_metrics,
)

# ============================
#  Train Model
# ============================
print("🚀 Training DistilBERT model...")
trainer.train()
print("✅ Training complete!")

# Save model
model.save_pretrained("distilbert_four_category_classifier")
tokenizer.save_pretrained("distilbert_four_category_classifier")
print("✅ Model saved in 'distilbert_four_category_classifier/'")

# ============================
#  Run Inference on Test Set
# ============================
print("🔄 Running inference on test data...")

def classify_nouns(nouns):
    """Run inference on a list of nouns and classify into multiple categories."""
    model.to("cpu")  # ✅ Ensure model is on CPU
    inputs = tokenizer(nouns, padding=True, truncation=True, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs)
    predictions = (torch.sigmoid(outputs.logits) > 0.5).int().tolist()
    return predictions

predictions = classify_nouns(test_nouns)

# ============================
#  Evaluate Model Performance
# ============================
accuracy = accuracy_score(np.array(test_labels), np.array(predictions))
report = classification_report(np.array(test_labels), np.array(predictions), target_names=list(label_map.keys()), zero_division=0)

print("\n📊 **Evaluation Results:**")
print(f"✅ Accuracy: {accuracy:.4f}")
print(report)

# Separate classified categories
classified_categories = {"object": [], "agent": [], "surface": [], "region": []}
for i, noun in enumerate(test_nouns):
    for j, label in enumerate(predictions[i]):
        if label == 1:
            classified_categories[list(label_map.keys())[j]].append(noun)

print("\n🔹 Objects:", classified_categories["object"])
print("🔸 Agents:", classified_categories["agent"])
print("🟩 Surfaces:", classified_categories["surface"])
print("🌍 Regions:", classified_categories["region"])
