import argparse
import json
from pathlib import Path
from typing import Dict, Optional

from src.core.clinical_reasoning import ClinicalVignette, TransparentReasoningEngine
from src.data.data_processor import DataProcessor
from src.llm.llm_interface import LLMInterface, LLMConfig
from src.evaluation.metrics import EvaluationMetrics

def parse_args():
    parser = argparse.ArgumentParser(description="Clinical Reasoning System")
    parser.add_argument("--input", required=True, help="Path to input clinical vignette JSON")
    parser.add_argument("--output", required=True, help="Path to output directory")
    parser.add_argument("--guidelines", help="Path to medical guidelines JSON")
    parser.add_argument("--config", help="Path to configuration JSON")
    return parser.parse_args()

def load_config(config_path: Optional[str] = None) -> Dict:
    """Load configuration from file or use defaults."""
    if config_path and Path(config_path).exists():
        with open(config_path, 'r') as f:
            return json.load(f)
    
    return {
        "llm": {
            "model_name": "gpt-4",
            "temperature": 0.1,
            "max_tokens": 2000
        },
        "evaluation": {
            "model_name": "all-MiniLM-L6-v2",
            "weights": {
                "guideline_adherence": 0.4,
                "reasoning_structure": 0.3,
                "completeness": 0.3
            }
        }
    }

def main():
    args = parse_args()
    config = load_config(args.config)
    
    # Get the project root directory (where src/ is located)
    project_root = Path(__file__).parent.parent
    
    # Initialize components with correct paths
    data_processor = DataProcessor(project_root / "data")
    llm_interface = LLMInterface(LLMConfig(**config["llm"]))
    evaluation_metrics = EvaluationMetrics(config["evaluation"]["model_name"])
    
    # Load and process input
    input_path = Path(args.input)
    if not input_path.is_absolute():
        input_path = project_root / input_path
    
    vignette = data_processor.load_vignette(input_path.stem)
    processed_vignette = data_processor.preprocess_vignette(vignette)
    
    # Load guidelines
    guidelines = []
    if args.guidelines:
        guidelines_path = Path(args.guidelines)
        if not guidelines_path.is_absolute():
            guidelines_path = project_root / guidelines_path
        guidelines = data_processor.load_guidelines(guidelines_path.stem)
    
    # Generate clinical reasoning
    reasoning_engine = TransparentReasoningEngine(args.guidelines)
    diagnostic_steps = reasoning_engine.process_vignette(vignette)
    
    # Evaluate reasoning
    evaluation_results = {
        "guideline_adherence": evaluation_metrics.evaluate_guideline_adherence(
            "\n".join(step.reasoning for step in diagnostic_steps),
            guidelines
        ),
        "reasoning_structure": evaluation_metrics.evaluate_reasoning_structure(
            [step.reasoning for step in diagnostic_steps]
        ),
        "completeness": evaluation_metrics.evaluate_completeness(
            "\n".join(step.reasoning for step in diagnostic_steps),
            processed_vignette["clinical_info"]  # Using clinical info as reference
        )
    }
    
    # Calculate overall score
    overall_score = evaluation_metrics.calculate_overall_score(
        evaluation_results,
        config["evaluation"]["weights"]
    )
    
    # Save results
    output_dir = Path(args.output)
    if not output_dir.is_absolute():
        output_dir = project_root / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    
    results = {
        "vignette_id": vignette.patient_id,
        "diagnostic_steps": [step.model_dump() for step in diagnostic_steps],
        "evaluation": evaluation_results,
        "overall_score": overall_score
    }
    
    with open(output_dir / f"{vignette.patient_id}_results.json", 'w') as f:
        json.dump(results, f, indent=2)

if __name__ == "__main__":
    main() 