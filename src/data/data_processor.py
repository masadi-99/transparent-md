import json
from pathlib import Path
from typing import Dict, List, Optional
from pydantic import BaseModel

from src.core.clinical_reasoning import ClinicalVignette

class DataProcessor:
    """Handles data processing for clinical vignettes and guidelines."""
    
    def __init__(self, data_dir: str):
        self.data_dir = Path(data_dir)
        self.raw_dir = self.data_dir / "raw"
        self.processed_dir = self.data_dir / "processed"
        self.guidelines_dir = self.data_dir / "guidelines"
        
        # Create directories if they don't exist
        self.raw_dir.mkdir(parents=True, exist_ok=True)
        self.processed_dir.mkdir(parents=True, exist_ok=True)
        self.guidelines_dir.mkdir(parents=True, exist_ok=True)
    
    def load_vignette(self, vignette_id: str) -> ClinicalVignette:
        """Load a clinical vignette from JSON file."""
        vignette_path = self.raw_dir / f"{vignette_id}.json"
        if not vignette_path.exists():
            raise FileNotFoundError(f"Vignette {vignette_id} not found")
        
        with open(vignette_path, 'r') as f:
            data = json.load(f)
        return ClinicalVignette(**data)
    
    def save_vignette(self, vignette: ClinicalVignette) -> None:
        """Save a clinical vignette to JSON file."""
        vignette_path = self.raw_dir / f"{vignette.patient_id}.json"
        with open(vignette_path, 'w') as f:
            json.dump(vignette.dict(), f, indent=2)
    
    def load_guidelines(self, specialty: Optional[str] = None) -> List[str]:
        """Load medical guidelines for a specific specialty."""
        if specialty:
            guideline_path = self.guidelines_dir / f"{specialty}.json"
        else:
            guideline_path = self.guidelines_dir / "general.json"
            
        if not guideline_path.exists():
            raise FileNotFoundError(f"Guidelines for {specialty or 'general'} not found")
        
        with open(guideline_path, 'r') as f:
            return json.load(f)
    
    def preprocess_vignette(self, vignette: ClinicalVignette) -> Dict:
        """Preprocess a clinical vignette for LLM input."""
        return {
            "patient_info": f"Patient ID: {vignette.patient_id}\n"
                          f"Age: {vignette.age}\n"
                          f"Gender: {vignette.gender}",
            "clinical_info": f"Chief Complaint: {vignette.chief_complaint}\n"
                           f"History of Present Illness: {vignette.history_of_present_illness}",
            "findings": {
                "physical_exam": vignette.physical_examination,
                "lab_findings": vignette.laboratory_findings,
                "imaging_findings": vignette.imaging_findings or {}
            },
            "additional_notes": vignette.additional_notes or ""
        }
    
    def save_processed_data(self, data: Dict, filename: str) -> None:
        """Save processed data to JSON file."""
        output_path = self.processed_dir / filename
        with open(output_path, 'w') as f:
            json.dump(data, f, indent=2)
    
    def load_processed_data(self, filename: str) -> Dict:
        """Load processed data from JSON file."""
        input_path = self.processed_dir / filename
        if not input_path.exists():
            raise FileNotFoundError(f"Processed data file {filename} not found")
        
        with open(input_path, 'r') as f:
            return json.load(f) 