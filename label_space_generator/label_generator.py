import torch
from transformers import (
    BlipProcessor,
    BlipForConditionalGeneration,
    AutoTokenizer,
    AutoModelForTokenClassification,
    DistilBertForSequenceClassification,
)
from huggingface_hub import snapshot_download
from PIL import Image
import json
import os
from pathlib import Path


class LabelGenerator:
    def __init__(self, device=None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        print(f"🚀 Using device: {self.device}")

        self._dataset_name = None
        self._scene_name = None
        self._hydra_fixed_labelspace_path = None  # New field to store Hydra label space path

        self._step_count = 0  # For autosave tracking

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

        # Initialize label spaces
        self.reset_scene_space()

    def set_run_info(self, dataset_name, scene_name):
        """Set dataset name and scene name for YAML label space saving."""
        self._dataset_name = dataset_name
        self._scene_name = scene_name
        print(f"📝 Run info set: dataset = {dataset_name}, scene = {scene_name}")
    
    def set_hydra_fixed_labelspace_path(self, labelspace_path):
        """Set the fixed Hydra label space YAML path to update."""
        self._hydra_fixed_labelspace_path = labelspace_path
        print(f"📝 Hydra fixed label space path set to: {labelspace_path}")

    def reset_scene_space(self):
        """Reset scene-specific label space."""
        self.scene_label_space = {
            "object": set(),
            "agent": set(),
            "surface": set(),
            "region": set()
        }
    
    def update_hydra_fixed_label_space_file(self, labelspace_path):
        """Update Hydra fixed label space YAML file with new labels from scene_label_space."""
        import yaml
        import pathlib

        labelspace_file = pathlib.Path(labelspace_path)
        if not labelspace_file.exists():
            print(f"❌ Hydra fixed label space file does not exist: {labelspace_file}")
            return

        with open(labelspace_file, 'r') as f:
            label_data = yaml.safe_load(f)

        # Build existing name → label index map
        existing_names = {entry['name']: entry['label'] for entry in label_data.get('label_names', [])}
        current_index = label_data.get('total_semantic_labels', len(existing_names))

        # Sanity init in case fields missing
        for field in ["object_labels", "surface_places_labels", "dynamic_labels", "label_names"]:
            if field not in label_data:
                label_data[field] = []

        # Go through current scene_label_space
        for category, label_set in self.scene_label_space.items():
            for label in sorted(label_set):
                norm_label = label.lower().replace(' ', '_')
                if norm_label not in existing_names:
                    # Add new label entry
                    label_data['label_names'].append({
                        "label": current_index,
                        "name": norm_label,
                        "name_descriptive": f"a {label}"
                    })

                    # Add to corresponding category index list
                    if category == "object" or category == "agent":
                        label_data["object_labels"].append(current_index)
                    elif category == "surface":
                        label_data["surface_places_labels"].append(current_index)
                    elif category == "region":
                        label_data["surface_places_labels"].append(current_index)  # Treat region as surface

                    # Add dynamic if agent
                    if category == "agent":
                        label_data["dynamic_labels"].append(current_index)

                    # Update map and counter
                    existing_names[norm_label] = current_index
                    current_index += 1

        # Update total
        label_data['total_semantic_labels'] = current_index

        # Save back to YAML
        with open(labelspace_file, 'w') as f:
            yaml.dump(label_data, f, sort_keys=False)

        print(f"✅ Hydra fixed label space updated at: {labelspace_file}")

    # def step(self, image_array):
    #     """Process a single image frame and update label spaces."""
    #     # 🆕 Reset instance label space at the start of each step
    #     self.instance_label_space = {
    #         "object": set(),
    #         "agent": set(),
    #         "surface": set(),
    #         "region": set()
    #     }

    #     # Generate caption
    #     caption = self._generate_caption(image_array)
    #     print("\n📝 Generated Caption:", caption)

    #     # Extract nouns
    #     nouns = self._extract_nouns(caption)

    #     # Classify nouns
    #     current_labels = self._classify_nouns(nouns)

    #     # Update instance label space (current frame only)
    #     for category in self.instance_label_space:
    #         self.instance_label_space[category].update(current_labels[category])

    #     # Update scene label space (accumulated over scene)
    #     for category in self.scene_label_space:
    #         self.scene_label_space[category].update(current_labels[category])

    #     # Print current step result
    #     print("\n📸 Instance label space (current image):")
    #     for category, labels in self.instance_label_space.items():
    #         print(f"{category.capitalize()}: {sorted(labels)}")

    #     print("\n🗺️ Scene label space (accumulated):")
    #     for category, labels in self.scene_label_space.items():
    #         print(f"{category.capitalize()}: {sorted(labels)}")

    #     # Increment step count
    #     self._step_count += 1

    #     # Autosave every 5 label steps (or however often you want)
    #     if self._step_count % 5 == 0:
    #         if hasattr(self, "_autosave_path") and self._autosave_path is not None:
    #             self.save_scene_label_space(self._autosave_path)

    def step(self, image_array):
        """Process a single image frame and update label spaces."""
        import os
        import yaml
        import pathlib

        # 🌟 Reset instance label space at the start of each step
        self.instance_label_space = {
            "object": set(),
            "agent": set(),
            "surface": set(),
            "region": set()
        }

        # Generate caption
        caption = self._generate_caption(image_array)
        print("\n📜 Generated Caption:", caption)

        # Extract nouns
        nouns = self._extract_nouns(caption)

        # Classify nouns
        current_labels = self._classify_nouns(nouns)

        # Update instance label space (current frame only)
        for category in self.instance_label_space:
            self.instance_label_space[category].update(current_labels[category])

        # Update scene label space (accumulated over scene)
        for category in self.scene_label_space:
            self.scene_label_space[category].update(current_labels[category])

        # Print current step result
        print("\n📸 Instance label space (current image):")
        for category, labels in self.instance_label_space.items():
            print(f"{category.capitalize()}: {sorted(labels)}")

        print("\n🗼 Scene label space (accumulated):")
        for category, labels in self.scene_label_space.items():
            print(f"{category.capitalize()}: {sorted(labels)}")

        # Increment step count
        self._step_count += 1

        # Autosave after every step
        if hasattr(self, "_autosave_path") and self._autosave_path is not None:
            self.save_scene_label_space(self._autosave_path)

        # Additionally update Hydra fixed label space YAML if path is set
        if hasattr(self, "_hydra_fixed_labelspace_path") and self._hydra_fixed_labelspace_path is not None:
            self.update_hydra_fixed_label_space_file(self._hydra_fixed_labelspace_path)

        # Additional logic to update YAML label space file
        repo_root = pathlib.Path(__file__).absolute().parent.parent
        labelspace_dir = repo_root / "label_spaces"
        labelspace_dir.mkdir(parents=True, exist_ok=True)

        labelspace_filename = f"{self._dataset_name}_{self._scene_name}_label_space_generated.yaml"
        labelspace_file = labelspace_dir / labelspace_filename

        if labelspace_file.exists():
            with open(labelspace_file, 'r') as f:
                label_data = yaml.safe_load(f)
        else:
            label_data = {
                "total_semantic_labels": 0,
                "dynamic_labels": [],
                "object_labels": [],
                "surface_places_labels": [],
                "label_names": []
            }

        existing_names = {entry['name']: entry['label'] for entry in label_data['label_names']}
        current_index = label_data['total_semantic_labels']

        for category, label_set in self.scene_label_space.items():
            for label in sorted(label_set):
                norm_label = label.lower().replace(' ', '_')
                if norm_label not in existing_names:
                    label_data['label_names'].append({
                        "label": current_index,
                        "name": norm_label,
                        "name_descriptive": f"a {label}"
                    })

                    if category == "object" or category == "agent":
                        label_data["object_labels"].append(current_index)
                    elif category == "surface":
                        label_data["surface_places_labels"].append(current_index)
                    elif category == "region":
                        label_data["surface_places_labels"].append(current_index)  # Treat as surface for now

                    if category == "agent":
                        label_data["dynamic_labels"].append(current_index)

                    existing_names[norm_label] = current_index
                    current_index += 1

        label_data['total_semantic_labels'] = current_index

        with open(labelspace_file, 'w') as f:
            yaml.dump(label_data, f, sort_keys=False)

        print(f"📂 Updated label space saved to: {labelspace_file}")

    def set_autosave_path(self, scene_output_path):
        """Set the output path for periodic autosaves."""
        self._autosave_path = scene_output_path


    def _generate_caption(self, image_array):
        """Generate a caption for the given image array."""
        image = Image.fromarray(image_array).convert("RGB")
        inputs = self.blip_processor(images=image, return_tensors="pt").to(self.device)
        out = self.blip_model.generate(**inputs, max_length=150, min_length=30)
        return self.blip_processor.decode(out[0], skip_special_tokens=True)

    def _extract_nouns(self, caption):
        """Extract noun phrases from the caption."""
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
                    # Basic plural cleanup
                    if noun_phrase.endswith("s") and len(noun_phrase) > 3 and not noun_phrase.endswith(("ss", "us", "is", "grass", "glass", "species")):
                        nouns.add(noun_phrase[:-1])
                    current_noun = []

        # Catch any remaining noun phrase
        if current_noun:
            noun_phrase = " ".join(current_noun).strip()
            nouns.add(noun_phrase)
            if noun_phrase.endswith("s") and len(noun_phrase) > 3 and not noun_phrase.endswith(("ss", "us", "is", "grass", "glass", "species")):
                nouns.add(noun_phrase[:-1])

        print("\n✅ Extracted Nouns:", sorted(nouns))
        return sorted(nouns)

    def _classify_nouns(self, nouns):
        """Classify nouns into categories."""
        if not nouns:
            return {"object": [], "agent": [], "surface": [], "region": []}

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
    
    def save_scene_label_space(self, scene_output_path):
        """
        Save the current scene label space to a JSON file under 
        scene_output_path/scene_label_spaces/scene_label_space.json.
        """
        label_space_dir = Path(scene_output_path) / "label_spaces"
        label_space_dir.mkdir(parents=True, exist_ok=True)

        output_file = label_space_dir / "scene_label_space.json"

        label_dict = {k: sorted(list(v)) for k, v in self.scene_label_space.items()}
        with open(output_file, "w") as f:
            json.dump(label_dict, f, indent=2)

        print(f"💾 Saved scene label space to {output_file}")
