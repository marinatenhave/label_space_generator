import torch
from transformers import BlipProcessor, BlipForConditionalGeneration, AutoTokenizer, AutoModelForTokenClassification, DistilBertForSequenceClassification
from huggingface_hub import snapshot_download
from PIL import Image

class LabelGenerator:
    def __init__(self, device=None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        print(f"🚀 Using device: {self.device}")

        # Load BLIP model
        print("🔄 Loading BLIP model...")
        self.blip_processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-large")
        self.blip_model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-large").to(self.device)
        print("🚀 BLIP Model loaded!")

        # Load BERT POS tagger
        print("🔄 Loading BERT POS model...")
        self.pos_tokenizer = AutoTokenizer.from_pretrained("vblagoje/bert-english-uncased-finetuned-pos")
        self.pos_model = AutoModelForTokenClassification.from_pretrained("vblagoje/bert-english-uncased-finetuned-pos").to(self.device)
        print("🚀 BERT POS Model loaded!")

        # Download and load DistilBERT classifier
        print("🔄 Downloading DistilBERT model from HuggingFace...")
        model_dir = snapshot_download(
            repo_id="mtenhave/label-space-generator-distilbert",
            local_dir="./distilbert_model",
            local_dir_use_symlinks=False
        )
        self.classifier_tokenizer = AutoTokenizer.from_pretrained(model_dir)
        self.classifier_model = DistilBertForSequenceClassification.from_pretrained(model_dir).to(self.device)
        print("🚀 DistilBERT Model loaded!")

        # Initialize label space
        self.label_space = {
            "object": [],
            "agent": [],
            "surface": [],
            "region": []
        }

    def step(self, image_path):
        print(f"\n📷 Processing image: {image_path}")

        # Generate caption
        caption = self._generate_caption(image_path)
        print("\n📝 Generated Caption:", caption)

        # Extract nouns
        nouns = self._extract_nouns(caption)

        # Classify nouns
        self.label_space = self._classify_nouns(nouns)

        # Print result
        print("\n🔹 Objects:", self.label_space["object"])
        print("🔸 Agents:", self.label_space["agent"])
        print("🟩 Surfaces:", self.label_space["surface"])
        print("🌍 Regions:", self.label_space["region"])

    def _generate_caption(self, image_path):
        image = Image.open(image_path).convert("RGB")
        inputs = self.blip_processor(images=image, return_tensors="pt").to(self.device)
        out = self.blip_model.generate(**inputs, max_length=150, min_length=30)
        return self.blip_processor.decode(out[0], skip_special_tokens=True)

    def _extract_nouns(self, caption):
        STOPWORDS = {"the", "a", "an", "that", "this", "these", "those", "with", "on", "in", "of", "at", "by", "for", "to", "and", "is"}

        inputs = self.pos_tokenizer(caption, return_tensors="pt", padding=True, truncation=True).to(self.device)

        with torch.no_grad():
            outputs = self.pos_model(**inputs).logits

        predictions = torch.argmax(outputs, dim=2)

        tokens = self.pos_tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])
        labels = [self.pos_model.config.id2label[pred.item()] for pred in predictions[0]]

        nouns = set()
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
                    if noun_phrase.endswith("s") and len(noun_phrase) > 3 and not noun_phrase.endswith(("ss", "us", "is", "grass", "glass", "species")):
                        nouns.add(noun_phrase[:-1])
                    current_noun = []

        if current_noun:
            noun_phrase = " ".join(current_noun).strip()
            nouns.add(noun_phrase)
            if noun_phrase.endswith("s") and len(noun_phrase) > 3 and not noun_phrase.endswith(("ss", "us", "is", "grass", "glass", "species")):
                nouns.add(noun_phrase[:-1])

        print("\n✅ Extracted Nouns:", sorted(nouns))
        return sorted(nouns)

    def _classify_nouns(self, nouns):
        inputs = self.classifier_tokenizer(nouns, truncation=True, padding=True, return_tensors="pt").to(self.device)

        with torch.no_grad():
            outputs = self.classifier_model(**inputs)

        predictions = (torch.sigmoid(outputs.logits) > 0.5).int().tolist()

        classified_categories = {"object": [], "agent": [], "surface": [], "region": []}
        for noun, label_vector in zip(nouns, predictions):
            for idx, label in enumerate(label_vector):
                if label == 1:
                    category = list(classified_categories.keys())[idx]
                    classified_categories[category].append(noun)

        return classified_categories
