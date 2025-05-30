import argparse
from pathlib import Path
import json
from typing import Dict, List

from src.core.clinical_reasoning import TransparentReasoningEngine, ClinicalVignette
from src.integration.direct_integration import DiReCTIntegration
from src.llm.llm_interface import LLMInterface, LLMConfig

def parse_args():
    parser = argparse.ArgumentParser(description="Run model on DiReCT samples")
    parser.add_argument("--samples_dir", required=True, help="Path to DiReCT samples directory")
    parser.add_argument("--kg_dir", help="Path to DiReCT knowledge graphs directory")
    parser.add_argument("--output_dir", required=True, help="Path to output directory")
    parser.add_argument("--config", help="Path to configuration JSON")
    return parser.parse_args()

def load_config(config_path: str = None) -> Dict:
    """Load configuration from file or use defaults."""
    if config_path and Path(config_path).exists():
        with open(config_path, 'r') as f:
            return json.load(f)
    
    return {
        "llm": {
            "model_name": "gpt-4",
            "temperature": 0.1,
            "max_tokens": 2000
        }
    }

def find_sample_files(samples_dir: Path) -> List[Path]:
    """Find all sample files recursively in the samples directory."""
    sample_files = []
    for ext in ['.json', '.txt']:
        sample_files.extend(samples_dir.rglob(f"*{ext}"))
    return sample_files

def main():
    args = parse_args()
    config = load_config(args.config)
    
    # Initialize components
    direct_integration = DiReCTIntegration(
        samples_dir=Path(args.samples_dir),
        kg_dir=Path(args.kg_dir) if args.kg_dir else None
    )
    
    llm_interface = LLMInterface(LLMConfig(**config["llm"]))
    reasoning_engine = TransparentReasoningEngine(args.kg_dir)
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Find all sample files
    sample_files = find_sample_files(Path(args.samples_dir))
    print(f"Found {len(sample_files)} sample files")
    
    # Process each sample
    results = {}
    for sample_file in sample_files:
        # Use relative path from samples_dir as sample_id
        sample_id = str(sample_file.relative_to(Path(args.samples_dir)))
        print(f"Processing sample: {sample_id}")
        
        try:
            # Load and convert sample
            sample = direct_integration.load_sample(sample_id)
            vignette_data = direct_integration.convert_to_vignette(sample)
            
            # Create ClinicalVignette object
            vignette = ClinicalVignette(
                patient_id=vignette_data["patient_id"],
                clinical_info=vignette_data["clinical_info"],
                observations=vignette_data["observations"],
                diagnoses=vignette_data["diagnoses"],
                knowledge_graph=vignette_data["knowledge_graph"]
            )
            
            # Generate predictions
            diagnostic_steps = reasoning_engine.process_vignette(vignette)
            
            # Evaluate against ground truth
            evaluation = direct_integration.evaluate_against_sample(
                [step.model_dump() for step in diagnostic_steps],
                sample
            )
            
            # Save results
            results[sample_id] = {
                "predictions": [step.model_dump() for step in diagnostic_steps],
                "evaluation": evaluation
            }
            
        except Exception as e:
            print(f"Error processing sample {sample_id}: {str(e)}")
            results[sample_id] = {"error": str(e)}
    
    # Save all results
    with open(output_dir / "direct_evaluation_results.json", 'w') as f:
        json.dump(results, f, indent=2)
    
    # Print summary
    total_samples = len(results)
    successful_samples = sum(1 for r in results.values() if "error" not in r)
    print(f"\nEvaluation complete:")
    print(f"Total samples: {total_samples}")
    print(f"Successfully processed: {successful_samples}")
    print(f"Results saved to: {output_dir / 'direct_evaluation_results.json'}")

if __name__ == "__main__":
    main() 