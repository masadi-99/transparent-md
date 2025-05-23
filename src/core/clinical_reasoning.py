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

class DiagnosticStep(BaseModel):
    """Represents a single step in the diagnostic reasoning process."""
    step_number: int
    reasoning: str
    supporting_evidence: List[str]
    guideline_reference: Optional[str] = None
    confidence_score: float

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
        """Load and parse medical guidelines."""
        if not self.guideline_path.exists():
            raise FileNotFoundError(f"Guidelines file not found: {self.guideline_path}")
        
        with open(self.guideline_path, 'r') as f:
            return json.load(f)
    
    def _match_guidelines(self, vignette: ClinicalVignette) -> List[Dict]:
        """Match clinical vignette with relevant guidelines."""
        matched_guidelines = []
        
        for guideline in self.guidelines.get("guidelines", []):
            # Check if patient meets criteria
            criteria_met = 0
            total_criteria = len(guideline.get("criteria", []))
            
            for criterion in guideline.get("criteria", []):
                # Simple keyword matching - in a real system, this would be more sophisticated
                if criterion.lower() in vignette.chief_complaint.lower() or \
                   criterion.lower() in vignette.history_of_present_illness.lower():
                    criteria_met += 1
            
            # Check risk factors
            risk_factors_met = 0
            total_risk_factors = len(guideline.get("risk_factors", []))
            
            for risk_factor in guideline.get("risk_factors", []):
                if risk_factor.lower() in str(vignette.__dict__).lower():
                    risk_factors_met += 1
            
            # Calculate match score
            if total_criteria > 0 and total_risk_factors > 0:
                match_score = (criteria_met / total_criteria + risk_factors_met / total_risk_factors) / 2
                if match_score > 0.5:  # Threshold for considering a guideline relevant
                    matched_guidelines.append({
                        "guideline": guideline,
                        "match_score": match_score
                    })
        
        return sorted(matched_guidelines, key=lambda x: x["match_score"], reverse=True)
    
    def process_vignette(self, vignette: ClinicalVignette) -> List[DiagnosticStep]:
        """Process a clinical vignette and return step-by-step diagnostic reasoning."""
        matched_guidelines = self._match_guidelines(vignette)
        diagnostic_steps = []
        
        # Step 1: Initial assessment
        diagnostic_steps.append(DiagnosticStep(
            step_number=1,
            reasoning=f"Initial assessment based on chief complaint: {vignette.chief_complaint}",
            supporting_evidence=[vignette.chief_complaint],
            confidence_score=0.9
        ))
        
        # Step 2: History analysis
        diagnostic_steps.append(DiagnosticStep(
            step_number=2,
            reasoning=f"Analysis of history of present illness: {vignette.history_of_present_illness}",
            supporting_evidence=[vignette.history_of_present_illness],
            confidence_score=0.85
        ))
        
        # Step 3: Physical examination findings
        diagnostic_steps.append(DiagnosticStep(
            step_number=3,
            reasoning="Physical examination findings analysis",
            supporting_evidence=[f"{k}: {v}" for k, v in vignette.physical_examination.items()],
            confidence_score=0.8
        ))
        
        # Step 4: Laboratory and imaging findings
        findings_evidence = []
        if vignette.laboratory_findings:
            findings_evidence.extend([f"{k}: {v}" for k, v in vignette.laboratory_findings.items()])
        if vignette.imaging_findings:
            findings_evidence.extend([f"{k}: {v}" for k, v in vignette.imaging_findings.items()])
        
        diagnostic_steps.append(DiagnosticStep(
            step_number=4,
            reasoning="Analysis of laboratory and imaging findings",
            supporting_evidence=findings_evidence,
            confidence_score=0.85
        ))
        
        # Step 5: Guideline-based assessment
        if matched_guidelines:
            top_guideline = matched_guidelines[0]["guideline"]
            diagnostic_steps.append(DiagnosticStep(
                step_number=5,
                reasoning=f"Assessment based on {top_guideline['title']}",
                supporting_evidence=top_guideline["criteria"],
                guideline_reference=top_guideline["id"],
                confidence_score=matched_guidelines[0]["match_score"]
            ))
        
        return diagnostic_steps
    
    def get_guideline_references(self, diagnostic_step: DiagnosticStep) -> List[str]:
        """Retrieve relevant medical guideline references for a diagnostic step."""
        if not diagnostic_step.guideline_reference:
            return []
        
        for guideline in self.guidelines.get("guidelines", []):
            if guideline["id"] == diagnostic_step.guideline_reference:
                return guideline["criteria"]
        
        return []
    
    def evaluate_confidence(self, diagnostic_step: DiagnosticStep) -> float:
        """Evaluate the confidence level of a diagnostic step."""
        # Base confidence on number of supporting evidence items and guideline reference
        base_confidence = min(1.0, len(diagnostic_step.supporting_evidence) / 5)
        
        if diagnostic_step.guideline_reference:
            # If there's a guideline reference, increase confidence
            base_confidence = min(1.0, base_confidence + 0.2)
        
        return base_confidence 