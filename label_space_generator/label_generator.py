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
import yaml
import pathlib
from datetime import datetime


class LabelGenerator:
    def __init__(self, fixed_labelspace_path=None, dataset_name=None, device=None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        print(f"🚀 Using device: {self.device}")

        self._step_count = 0  # For autosave 
        self.dataset_name = dataset_name
        self.scene_name = None 
        self._set_output_paths()

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

        # ✅ Initialize scene label space path & variable (either from YAML or clean)
        if fixed_labelspace_path:
            self._fixed_labelspace_path = fixed_labelspace_path
            self.initialize_generated_yaml()
            self.load_scene_space_from_yaml(fixed_labelspace_path)
            self._fixed_scene_space = {
                k: set(v) for k, v in self.scene_label_space.items()
            }  # 🧠 Store fixed labelspace copy
        else:
            self.reset_scene_space()
    
    def _set_output_paths(self):
        """Set paths for saving YAML and JSON label spaces under: out/<dataset>/label_spaces/<timestamp>/."""
        out_root = Path(__file__).resolve().parent.parent / "out"
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        if self.dataset_name:
            dir_path = out_root / self.dataset_name / "label_spaces" / timestamp
        else:
            dir_path = out_root / "general" / "label_spaces" / timestamp

        dir_path.mkdir(parents=True, exist_ok=True)

        self._generated_yaml_path = dir_path / "scene_label_space.yaml"
        self._generated_json_path = dir_path / "scene_label_space.json"
    
    def initialize_generated_yaml(self):
        """Copy the original fixed YAML contents exactly to the generated YAML path."""

        if not hasattr(self, "_fixed_labelspace_path") or not self._fixed_labelspace_path:
            print("⚠️ No fixed labelspace path available to duplicate.")
            return

        src_path = Path(self._fixed_labelspace_path)
        dst_path = Path(self._generated_yaml_path)

        if not src_path.exists():
            print(f"❌ Fixed labelspace file not found: {src_path}")
            return

        with open(src_path, "r") as f:
            yaml_data = yaml.safe_load(f)

        # Dump exactly as loaded — preserves original field and label order
        with open(dst_path, "w") as f:
            yaml.dump(yaml_data, f, sort_keys=False)

        print(f"✅ Duplicated fixed YAML to generated one: {dst_path}")

    def load_scene_space_from_yaml(self, labelspace_path):
        """Initialize scene_label_space from a fixed Hydra YAML label space file."""

        labelspace_file = pathlib.Path(labelspace_path)
        if not labelspace_file.exists():
            print(f"⚠️ Label space YAML file not found at: {labelspace_file}")
            return

        with open(labelspace_file, "r") as f:
            label_data = yaml.safe_load(f)
        
        self.original_num_labels = label_data["total_semantic_labels"]

        label_id_to_name = {entry["label"]: entry["name"] for entry in label_data.get("label_names", [])}

        self.scene_label_space = {
            "object": set(),
            "agent": set(),
            "surface": set(),
            "region": set(),
        }

        for lid in label_data.get("object_labels", []):
            name = label_id_to_name.get(lid)
            if name:
                self.scene_label_space["object"].add(name)

        for lid in label_data.get("dynamic_labels", []):
            name = label_id_to_name.get(lid)
            if name:
                self.scene_label_space["agent"].add(name)

        for lid in label_data.get("surface_places_labels", []):
            name = label_id_to_name.get(lid)
            if name:
                # Surface and Region both map from this field
                self.scene_label_space["surface"].add(name)
                self.scene_label_space["region"].add(name)

        print(f"📥 Initialized scene_label_space from: {labelspace_file}")
        for k, v in self.scene_label_space.items():
            print(f"  🔸 {k}: {len(v)} labels")

    def reset_scene_space(self):
        """Reset scene-specific label space."""
        self.scene_label_space = {
            "object": set(),
            "agent": set(),
            "surface": set(),
            "region": set()
        }

    def _update_generated_yaml(self, newly_added_labels=None):
        """Append new labels to the YAML, preserving original label order."""
        import yaml
        from pathlib import Path

        labelspace_file = Path(self._generated_yaml_path)

        if not labelspace_file.exists():
            print(f"❌ Generated YAML path does not exist: {labelspace_file}")
            return

        with open(labelspace_file, "r") as f:
            label_data = yaml.safe_load(f) or {}

        # Ensure necessary fields exist
        for field in ["total_semantic_labels", "dynamic_labels", "object_labels",
                    "surface_places_labels", "label_names"]:
            if field not in label_data:
                label_data[field] = [] if field != "total_semantic_labels" else 0

        existing_names = {entry['name'] for entry in label_data['label_names']}
        current_index = label_data["total_semantic_labels"]

        if not newly_added_labels:
            print("ℹ️ No new labels to add to YAML.")
        else:
            for norm_label, category in newly_added_labels:
                if norm_label in existing_names:
                    continue  # Already added before

                print(f"💡 Writing new label to YAML: {norm_label} ({category})")

                # Append to label_names list (preserving order)
                label_data["label_names"].append({
                    "label": current_index,
                    "name": norm_label,
                    "name_descriptive": f"a {norm_label.replace('_', ' ')}"
                })

                # Append to category lists
                if category in ["object", "agent"]:
                    label_data["object_labels"].append(current_index)
                if category in ["surface", "region"]:
                    label_data["surface_places_labels"].append(current_index)
                if category == "agent":
                    label_data["dynamic_labels"].append(current_index)

                current_index += 1

            label_data["total_semantic_labels"] = current_index

            with open(labelspace_file, "w") as f:
                yaml.dump(label_data, f, sort_keys=False)

            print(f"✅ Label space YAML saved to: {labelspace_file}")
            print(f"🛠️ Final YAML now contains {len(label_data['label_names'])} labels")

   
    def step(self, image_array):
        """Process a single image frame and update label spaces."""

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

        newly_added_labels = []

        for category in self.instance_label_space:
            for label in current_labels[category]:
                norm = label.lower().replace(" ", "_")
                already_seen = any(
                    norm == existing.lower().replace(" ", "_")
                    for existing in self.scene_label_space[category]
                )
                if already_seen:
                    continue

                # ✅ New label → add to scene and mark for logging
                self.scene_label_space[category].add(label)
                newly_added_labels.append((norm, category))

            # Store detected labels for this step
            self.instance_label_space[category].update(current_labels[category])

        # 🔍 Log newly added labels
        if newly_added_labels:
            print("\n🆕 New labels to be added to YAML files:")
            for norm, category in newly_added_labels:
                print(f"  ➕ {norm} → {category}")
        else:
            print("\n✅ No new labels detected this step.")

        # 🔄 Print full state
        print("\n📸 Instance label space (current image):")
        for category, labels in self.instance_label_space.items():
            print(f"{category.capitalize()}: {sorted(labels)}")

        print("\n🗼 Scene label space (accumulated):")
        for category, labels in self.scene_label_space.items():
            print(f"{category.capitalize()}: {sorted(labels)}")

        # ➕ Increment step count
        self._step_count += 1

        # 📝 Write label space JSON and YAML
        self._update_generated_json()
        self._update_generated_yaml(newly_added_labels)

        return newly_added_labels


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
    
    def _update_generated_json(self):
        """Update the scene label space JSON file in the same location as the YAML."""
        label_dict = {k: sorted(list(v)) for k, v in self.scene_label_space.items()}
        with open(self._generated_json_path, "w") as f:
            json.dump(label_dict, f, indent=2)
        print(f"✅ Scene label space JSON updated at: {self._generated_json_path}")

    
    def assign_missing_categories(self):
        """Ensure all labels in the fixed labelspace YAML have a category assignment."""

        if not hasattr(self, "_fixed_labelspace_path") or self._fixed_labelspace_path is None:
            print("⚠️ No fixed labelspace path set. Skipping category assignment.")
            return

        labelspace_file = pathlib.Path(self._fixed_labelspace_path)
        if not labelspace_file.exists():
            print(f"❌ Fixed labelspace file does not exist: {labelspace_file}")
            return

        with open(labelspace_file, 'r') as f:
            label_data = yaml.safe_load(f)

        if label_data is None or "label_names" not in label_data:
            print("❌ Malformed or empty labelspace YAML.")
            return

        # Ensure required fields exist
        for key in ["object_labels", "surface_places_labels", "dynamic_labels"]:
            if key not in label_data:
                label_data[key] = []

        existing_assignments = set(label_data["object_labels"]) | set(label_data["surface_places_labels"]) | set(label_data["dynamic_labels"])
        name_to_label = {entry["name"]: entry["label"] for entry in label_data["label_names"]}

        labels_to_classify = []
        label_ids_to_names = {}

        for entry in label_data["label_names"]:
            label = entry["label"]
            name = entry["name"]

            if label not in existing_assignments and name != "void":
                labels_to_classify.append(name)
                label_ids_to_names[name] = label

        if not labels_to_classify:
            print("✅ No missing category assignments.")
            return

        print(f"🔍 Assigning categories to {len(labels_to_classify)} unassigned labels...")

        predictions = self._classify_nouns(labels_to_classify)

        for category, name_set in predictions.items():
            for name in name_set:
                label_id = label_ids_to_names[name]
                if category in ["object", "agent"]:
                    label_data["object_labels"].append(label_id)
                if category in ["surface", "region"]:
                    label_data["surface_places_labels"].append(label_id)
                if category == "agent":
                    label_data["dynamic_labels"].append(label_id)
                print(f"🧠 Assigned {name} → {category}")

        # Write back to YAML
        with open(labelspace_file, 'w') as f:
            yaml.dump(label_data, f, sort_keys=False)

        print(f"✅ Fixed labelspace categories updated at: {labelspace_file}")

