# Transparent Clinical Reasoning with LLMs

A modular Python framework for exploring transparent and guideline-grounded clinical reasoning using large language models (LLMs).

## Project Overview

This project aims to develop a prototype framework that takes clinical vignettes (patient symptoms and findings) and produces step-by-step diagnostic reasoning, grounded in formal medical guidelines. The framework emphasizes transparency, reproducibility, and adherence to medical best practices.

## Project Structure

```
transparent-md/
├── src/
│   ├── core/                 # Core clinical reasoning components
│   ├── data/                 # Data handling and preprocessing
│   ├── llm/                  # LLM integration and prompting
│   ├── evaluation/           # Evaluation metrics and analysis
│   └── utils/               # Utility functions and helpers
├── tests/                   # Test suite
├── config/                  # Configuration files
├── data/                    # Data storage
│   ├── raw/                # Raw clinical vignettes
│   ├── processed/          # Processed data
│   └── guidelines/         # Medical guidelines
├── notebooks/              # Jupyter notebooks for analysis
└── docs/                   # Documentation
```

## Setup

1. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Set up environment variables:
```bash
cp .env.example .env
# Edit .env with your configuration
```

## Usage

1. Prepare your clinical vignettes in the `data/raw` directory
2. Configure your LLM settings in `config/llm_config.yaml`
3. Run the clinical reasoning pipeline:
```bash
python src/main.py --input data/raw/vignette.json --output results/
```

## Development

- Run tests: `pytest tests/`
- Format code: `black src/ tests/`
- Type checking: `mypy src/`

## Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## License

MIT License - see LICENSE file for details
