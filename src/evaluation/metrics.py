from typing import Dict, List, Optional
from pydantic import BaseModel
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer

class EvaluationMetrics:
    """Evaluation metrics for clinical reasoning quality."""
    
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)
    
    def calculate_semantic_similarity(self, text1: str, text2: str) -> float:
        """Calculate semantic similarity between two texts using sentence embeddings."""
        embeddings1 = self.model.encode([text1])
        embeddings2 = self.model.encode([text2])
        return float(cosine_similarity(embeddings1, embeddings2)[0][0])
    
    def evaluate_guideline_adherence(
        self,
        reasoning: str,
        guidelines: List[str]
    ) -> Dict[str, float]:
        """Evaluate how well the reasoning adheres to medical guidelines."""
        guideline_embeddings = self.model.encode(guidelines)
        reasoning_embedding = self.model.encode([reasoning])[0]
        
        similarities = cosine_similarity([reasoning_embedding], guideline_embeddings)[0]
        
        return {
            "max_guideline_similarity": float(np.max(similarities)),
            "mean_guideline_similarity": float(np.mean(similarities)),
            "guideline_coverage": float(np.mean(similarities > 0.5))  # Threshold for considering a guideline covered
        }
    
    def evaluate_reasoning_structure(
        self,
        reasoning_steps: List[str]
    ) -> Dict[str, float]:
        """Evaluate the structure and coherence of the reasoning steps."""
        if not reasoning_steps:
            return {
                "step_coherence": 0.0,
                "reasoning_flow": 0.0
            }
        
        # Calculate coherence between consecutive steps
        step_embeddings = self.model.encode(reasoning_steps)
        consecutive_similarities = [
            cosine_similarity([step_embeddings[i]], [step_embeddings[i+1]])[0][0]
            for i in range(len(reasoning_steps)-1)
        ]
        
        # Calculate overall flow by comparing first and last steps
        flow_score = cosine_similarity([step_embeddings[0]], [step_embeddings[-1]])[0][0]
        
        return {
            "step_coherence": float(np.mean(consecutive_similarities)),
            "reasoning_flow": float(flow_score)
        }
    
    def evaluate_completeness(
        self,
        reasoning: str,
        gold_standard: str
    ) -> Dict[str, float]:
        """Evaluate the completeness of the reasoning compared to a gold standard."""
        # Split into sentences for more granular comparison
        reasoning_sentences = reasoning.split('.')
        gold_sentences = gold_standard.split('.')
        
        # Calculate coverage of gold standard concepts
        gold_embeddings = self.model.encode(gold_sentences)
        reasoning_embedding = self.model.encode([reasoning])[0]
        
        similarities = cosine_similarity([reasoning_embedding], gold_embeddings)[0]
        
        return {
            "concept_coverage": float(np.mean(similarities > 0.5)),
            "detail_level": float(np.mean(similarities))
        }
    
    def calculate_overall_score(
        self,
        metrics: Dict[str, Dict[str, float]],
        weights: Optional[Dict[str, float]] = None
    ) -> float:
        """Calculate overall evaluation score from individual metrics."""
        if weights is None:
            weights = {
                "guideline_adherence": 0.4,
                "reasoning_structure": 0.3,
                "completeness": 0.3
            }
        
        score = 0.0
        for category, category_metrics in metrics.items():
            category_score = np.mean(list(category_metrics.values()))
            score += category_score * weights[category]
        
        return float(score) 