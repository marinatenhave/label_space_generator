import json
import torch
import numpy as np
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split

# ============================
#  Load Dataset (Test Set Only)
# ============================
print("🔄 Loading test dataset...")

with open("datasets/current_4_category_dataset.json", "r") as f:
    data = json.load(f)

nouns = list(data.keys())
label_map = {"object": 0, "agent": 1, "surface": 2, "region": 3}
num_labels = len(label_map)

def encode_labels(label_list):
    label_vector = np.zeros(num_labels)
    for label in label_list:
        label_vector[label_map[label]] = 1
    return label_vector.tolist()

labels = [encode_labels(data[noun]) for noun in nouns]

# Ensure the test set is completely separate from the training set
_, test_nouns, _, test_labels = train_test_split(nouns, labels, test_size=0.2, random_state=42)

print(f"✅ Test dataset loaded! {len(test_nouns)} samples.")

# ============================
#  Load Trained Model
# ============================
print("🔄 Loading trained DistilBERT model...")

model_path = "distilbert_four_category_classifier"
tokenizer = DistilBertTokenizer.from_pretrained(model_path)
model = DistilBertForSequenceClassification.from_pretrained(model_path)

model.to("cpu")  # ✅ Run inference on CPU

print("🚀 Model loaded successfully!")

# ============================
#  Run Inference on Test Set
# ============================

def classify_nouns(nouns):
    """Run inference on test nouns and ensure at least one category is assigned."""
    inputs = tokenizer(nouns, padding=True, truncation=True, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs)
    
    probabilities = torch.sigmoid(outputs.logits)  # Convert logits to probabilities
    predictions = (probabilities > 0.5).int().tolist()  # Apply thresholding

    # Ensure at least one category is assigned per noun
    for i, probs in enumerate(probabilities):
        if sum(predictions[i]) == 0:  # If no category selected
            max_index = torch.argmax(probs).item()  # Find the most confident category
            predictions[i][max_index] = 1  # Assign it

    return predictions


print("🔄 Running inference on test data...")
predictions = classify_nouns(test_nouns)

# ============================
#  Evaluate Model Performance
# ============================
accuracy = accuracy_score(np.array(test_labels), np.array(predictions))
report = classification_report(np.array(test_labels), np.array(predictions), target_names=list(label_map.keys()), zero_division=0)

print("\n📊 **Evaluation Results:**")
print(f"✅ Accuracy: {accuracy:.4f}")
print(report)

# Identify incorrect classifications
incorrect_predictions = []
for i, (true_label, pred_label) in enumerate(zip(test_labels, predictions)):
    if true_label != pred_label:
        incorrect_predictions.append((test_nouns[i], true_label, pred_label))

print("\n❌ Incorrect Predictions:")
for noun, true, pred in incorrect_predictions:
    true_labels = [label for label, idx in label_map.items() if true[idx] == 1]
    pred_labels = [label for label, idx in label_map.items() if pred[idx] == 1]
    print(f"{noun}: True {true_labels}, Predicted {pred_labels}")

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
