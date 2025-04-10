import os
import torch
from transformers import BlipProcessor, BlipForConditionalGeneration, AutoTokenizer, AutoModelForTokenClassification, DistilBertForSequenceClassification
from PIL import Image

# ============================
#  Load Models
# ============================
device = "cuda" if torch.cuda.is_available() else "cpu"

# Load BLIP for caption generation
print("🔄 Loading BLIP model...")
blip_processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-large")
blip_model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-large").to(device)
print("🚀 BLIP Model loaded!")

# Load BERT POS tagger for noun extraction
print("🔄 Loading BERT POS model...")
pos_tokenizer = AutoTokenizer.from_pretrained("vblagoje/bert-english-uncased-finetuned-pos")
pos_model = AutoModelForTokenClassification.from_pretrained("vblagoje/bert-english-uncased-finetuned-pos").to(device)
print("🚀 BERT POS Model loaded!")

# Load trained DistilBERT for classification
print("🔄 Loading trained DistilBERT classifier...")
classifier_tokenizer = AutoTokenizer.from_pretrained("distilbert_four_category_classifier")
classifier_model = DistilBertForSequenceClassification.from_pretrained("distilbert_four_category_classifier").to(device)
print("🚀 DistilBERT Model loaded!")

# ============================
#  Functions
# ============================

def generate_caption(image_filename):
    """Generate a caption from an image using BLIP."""
    image = Image.open(image_filename).convert("RGB")
    inputs = blip_processor(images=image, return_tensors="pt").to(device)
    out = blip_model.generate(**inputs, max_length=150, min_length=30)
    return blip_processor.decode(out[0], skip_special_tokens=True)

import re

def extract_nouns(caption):
    """Extract only nouns from the caption using a BERT POS tagger while ensuring plural words get both singular and plural versions."""
    STOPWORDS = {"the", "a", "an", "that", "this", "these", "those", "with", "on", "in", "of", "at", "by", "for", "to", "and", "is"}
    
    # Tokenize input caption
    inputs = pos_tokenizer(caption, return_tensors="pt", padding=True, truncation=True).to(device)
    
    with torch.no_grad():
        outputs = pos_model(**inputs).logits
    
    predictions = torch.argmax(outputs, dim=2)
    
    tokens = pos_tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])
    labels = [pos_model.config.id2label[pred.item()] for pred in predictions[0]]

    nouns = set()
    plural_nouns = set()
    current_noun = []

    for token, label in zip(tokens, labels):
        if token.startswith("##"):  
            if current_noun:
                current_noun[-1] += token[2:]  
        elif label == "NOUN" and token not in STOPWORDS:
            current_noun.append(token)
        else:
            if current_noun:
                noun_phrase = " ".join(current_noun).strip()
                nouns.add(noun_phrase)

                # Check if the noun is plural and should include its singular form
                if noun_phrase.endswith("s") and len(noun_phrase) > 3 and not noun_phrase.endswith(("ss", "us", "is", "grass", "glass", "species")):
                    singular_form = noun_phrase[:-1]
                    nouns.add(singular_form)
                    plural_nouns.add(noun_phrase)
                
                current_noun = []

    if current_noun:
        noun_phrase = " ".join(current_noun).strip()
        nouns.add(noun_phrase)

        # Check if the noun is plural and should include its singular form
        if noun_phrase.endswith("s") and len(noun_phrase) > 3 and not noun_phrase.endswith(("ss", "us", "is", "grass", "glass", "species")):
            singular_form = noun_phrase[:-1]
            nouns.add(singular_form)
            plural_nouns.add(noun_phrase)

    print("\n✅ Extracted Nouns:", sorted(nouns))
    return sorted(nouns)


def classify_nouns(nouns):
    """Predict multiple labels for nouns using trained DistilBERT."""
    inputs = classifier_tokenizer(nouns, truncation=True, padding=True, return_tensors="pt").to(device)
    
    with torch.no_grad():
        outputs = classifier_model(**inputs)
    
    predictions = (torch.sigmoid(outputs.logits) > 0.5).int().tolist()
    
    classified_categories = {"object": [], "agent": [], "surface": [], "region": []}
    for noun, label_vector in zip(nouns, predictions):
        for idx, label in enumerate(label_vector):
            if label == 1:
                classified_categories[list(classified_categories.keys())[idx]].append(noun)
    
    return classified_categories

# ============================
#  Run on Image
# ============================

# image_filename = "rgb_0000200.png"  # Set image filename here
image_filename = "rgb_0002709.png"  # Set image filename here
# image_filename = "rgb_0000712.png"  # Set image filename here
# image_filename = "img_3097.jpg"  # Set image filename here
# image_filename = "img_2077.jpg"  # Set image filename here
# image_filename = "img_1106.jpg"  # Set image filename here

print(f"📷 Processing image: {image_filename}")

# Generate caption
caption = generate_caption("images/" + image_filename)
print("\n📝 Generated Caption:", caption)

# Extract nouns
nouns = extract_nouns(caption)

# Classify nouns
classified_categories = classify_nouns(nouns)
# Display results
print("\n🔹 Objects:", classified_categories["object"])
print("🔸 Agents:", classified_categories["agent"])
print("🟩 Surfaces:", classified_categories["surface"])
print("🌍 Regions:", classified_categories["region"])
