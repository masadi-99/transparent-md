from pathlib import Path
import json
from typing import Dict, List, Optional
from pydantic import BaseModel

class DiReCTObservation(BaseModel):
    observation: str
    rationale: str
    diagnosis: str

class DiReCTSample(BaseModel):
    clinical_note: str
    observations: List[DiReCTObservation]
    knowledge_graph: Optional[Dict] = None

class DiReCTIntegration:
    def __init__(self, samples_dir: Path, kg_dir: Optional[Path] = None):
        self.samples_dir = Path(samples_dir)
        self.kg_dir = Path(kg_dir) if kg_dir else None
        
    def load_sample(self, sample_id: str) -> DiReCTSample:
        """Load a DiReCT sample with its associated knowledge graph."""
        sample_path = self.samples_dir / sample_id
        if not sample_path.exists():
            raise FileNotFoundError(f"Sample {sample_id} not found in {self.samples_dir}")
            
        # Handle different file formats
        if sample_path.suffix == '.json':
            with open(sample_path, 'r') as f:
                sample_data = json.load(f)
        elif sample_path.suffix == '.txt':
            with open(sample_path, 'r') as f:
                clinical_note = f.read()
                # For txt files, create a basic structure
                sample_data = {
                    "clinical_note": clinical_note,
                    "observations": [
                        {
                            "observation": "Extracted from clinical note",
                            "rationale": "Based on clinical presentation",
                            "diagnosis": "To be determined"
                        }
                    ]
                }
        else:
            raise ValueError(f"Unsupported file format: {sample_path.suffix}")
            
        # Load knowledge graph if available
        kg_data = None
        if self.kg_dir:
            # Try to find matching knowledge graph
            kg_name = sample_path.stem
            kg_path = self.kg_dir / f"{kg_name}_kg.json"
            if not kg_path.exists():
                # Try parent directory name as condition
                condition = sample_path.parent.name
                kg_path = self.kg_dir / f"{condition}_kg.json"
            
            if kg_path.exists():
                with open(kg_path, 'r') as f:
                    kg_data = json.load(f)
                    
        return DiReCTSample(
            clinical_note=sample_data["clinical_note"],
            observations=[
                DiReCTObservation(**obs) for obs in sample_data["observations"]
            ],
            knowledge_graph=kg_data
        )
    
    def convert_to_vignette(self, sample: DiReCTSample) -> Dict:
        """Convert a DiReCT sample to our clinical vignette format."""
        return {
            "patient_id": sample.clinical_note[:30],  # Use first 30 chars as ID
            "clinical_info": sample.clinical_note,
            "observations": [obs.observation for obs in sample.observations],
            "diagnoses": [obs.diagnosis for obs in sample.observations],
            "knowledge_graph": sample.knowledge_graph
        }
    
    def evaluate_against_sample(self, 
                              predictions: List[Dict], 
                              sample: DiReCTSample) -> Dict:
        """Evaluate our model's predictions against the DiReCT sample."""
        # Extract ground truth observations and diagnoses
        ground_truth = {
            "observations": [obs.observation for obs in sample.observations],
            "diagnoses": [obs.diagnosis for obs in sample.observations],
            "rationales": [obs.rationale for obs in sample.observations]
        }
        
        # Compare predictions with ground truth
        evaluation = {
            "observation_matches": [],
            "diagnosis_matches": [],
            "rationale_quality": []
        }
        
        for pred in predictions:
            # Match observations
            for gt_obs in ground_truth["observations"]:
                if pred["observation"] == gt_obs:
                    evaluation["observation_matches"].append(True)
                    break
            else:
                evaluation["observation_matches"].append(False)
                
            # Match diagnoses
            for gt_dx in ground_truth["diagnoses"]:
                if pred["diagnosis"] == gt_dx:
                    evaluation["diagnosis_matches"].append(True)
                    break
            else:
                evaluation["diagnosis_matches"].append(False)
                
            # Evaluate rationale quality (placeholder for more sophisticated evaluation)
            evaluation["rationale_quality"].append(0.5)  # Placeholder score
            
        return evaluation 