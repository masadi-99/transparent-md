from abc import ABC, abstractmethod
from typing import Dict, List, Optional
import json
from pathlib import Path
from pydantic import BaseModel

class ClinicalVignette(BaseModel):
    """Represents a clinical case with patient information and findings."""
    patient_id: str
    age: int
    gender: str
    chief_complaint: str
    history_of_present_illness: str
    physical_examination: Dict[str, str]
    laboratory_findings: Dict[str, str]
    imaging_findings: Optional[Dict[str, str]] = None
    additional_notes: Optional[str] = None
    clinical_info: str
    observations: List[str]
    diagnoses: List[str]
    knowledge_graph: Optional[Dict] = None

class DiagnosticStep(BaseModel):
    """Represents a single step in the diagnostic reasoning process."""
    step_number: int
    reasoning: str
    supporting_evidence: List[str]
    guideline_reference: Optional[str] = None
    confidence_score: float
    observation: str
    diagnosis: str
    guideline_references: List[str]

class ClinicalReasoningEngine(ABC):
    """Abstract base class for clinical reasoning engines."""
    
    @abstractmethod
    def process_vignette(self, vignette: ClinicalVignette) -> List[DiagnosticStep]:
        """Process a clinical vignette and return step-by-step diagnostic reasoning."""
        pass
    
    @abstractmethod
    def get_guideline_references(self, diagnostic_step: DiagnosticStep) -> List[str]:
        """Retrieve relevant medical guideline references for a diagnostic step."""
        pass
    
    @abstractmethod
    def evaluate_confidence(self, diagnostic_step: DiagnosticStep) -> float:
        """Evaluate the confidence level of a diagnostic step."""
        pass

class TransparentReasoningEngine(ClinicalReasoningEngine):
    """Implementation of clinical reasoning engine with transparency features."""
    
    def __init__(self, guideline_path: str):
        self.guideline_path = Path(guideline_path)
        self.guidelines = self._load_guidelines()
    
    def _load_guidelines(self) -> Dict:
        """Load guidelines from the knowledge graph directory."""
        guidelines = {}
        
        # If path is a directory, load all JSON files
        if self.guideline_path.is_dir():
            for kg_file in self.guideline_path.glob("*.json"):
                try:
                    with open(kg_file, 'r') as f:
                        kg_data = json.load(f)
                        # Use filename without extension as key
                        guidelines[kg_file.stem] = kg_data
                except Exception as e:
                    print(f"Warning: Could not load knowledge graph {kg_file}: {str(e)}")
        else:
            # If path is a file, load single file
            try:
                with open(self.guideline_path, 'r') as f:
                    guidelines[self.guideline_path.stem] = json.load(f)
            except Exception as e:
                print(f"Warning: Could not load knowledge graph {self.guideline_path}: {str(e)}")
                
        return guidelines
    
    def process_vignette(self, vignette: ClinicalVignette) -> List[DiagnosticStep]:
        """Process a clinical vignette and generate diagnostic steps."""
        steps = []
        
        # Use vignette's knowledge graph if available, otherwise use loaded guidelines
        kg_data = vignette.knowledge_graph if vignette.knowledge_graph else self.guidelines
        
        for i, (obs, dx) in enumerate(zip(vignette.observations, vignette.diagnoses), 1):
            step = DiagnosticStep(
                step_number=i,
                observation=obs,
                reasoning=self._generate_reasoning(obs, dx, kg_data),
                diagnosis=dx,
                confidence_score=0.0,  # Will be updated
                guideline_references=[]
            )
            
            # Update step with references and confidence
            step.guideline_references = self.get_guideline_references(step)
            step.confidence_score = self.evaluate_confidence(step)
            
            steps.append(step)
            
        return steps
    
    def _generate_reasoning(self, observation: str, diagnosis: str, kg_data: Dict) -> str:
        """Generate reasoning for a diagnostic step using knowledge graph data."""
        # This is a placeholder - in practice, you would use the LLM to generate
        # detailed reasoning based on the knowledge graph
        return f"Based on the observation '{observation}', the diagnosis '{diagnosis}' is considered."
    
    def get_guideline_references(self, diagnostic_step: DiagnosticStep) -> List[str]:
        """Retrieve relevant medical guideline references for a diagnostic step."""
        references = []
        for kg_name, kg_data in self.guidelines.items():
            # Simple matching - in practice, you would use more sophisticated matching
            if diagnostic_step.diagnosis.lower() in str(kg_data).lower():
                references.append(f"Reference from {kg_name}")
        return references
    
    def evaluate_confidence(self, diagnostic_step: DiagnosticStep) -> float:
        """Evaluate the confidence level of a diagnostic step."""
        # Simple confidence scoring based on number of references
        base_confidence = 0.5
        reference_bonus = 0.1 * len(diagnostic_step.guideline_references)
        return min(base_confidence + reference_bonus, 1.0) 