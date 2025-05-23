from typing import Dict, List, Optional
from pydantic import BaseModel
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.output_parsers import PydanticOutputParser

class LLMConfig(BaseModel):
    """Configuration for LLM interactions."""
    model_name: str = "gpt-4"
    temperature: float = 0.1
    max_tokens: int = 2000
    top_p: float = 1.0
    frequency_penalty: float = 0.0
    presence_penalty: float = 0.0

class LLMInterface:
    """Interface for interacting with language models."""
    
    def __init__(self, config: LLMConfig):
        self.config = config
        self.llm = ChatOpenAI(
            model_name=config.model_name,
            temperature=config.temperature,
            max_tokens=config.max_tokens,
            top_p=config.top_p,
            frequency_penalty=config.frequency_penalty,
            presence_penalty=config.presence_penalty
        )
    
    def generate_clinical_reasoning(
        self,
        vignette: str,
        guidelines: List[str],
        output_parser: Optional[PydanticOutputParser] = None
    ) -> Dict:
        """Generate clinical reasoning based on vignette and guidelines."""
        prompt = ChatPromptTemplate.from_messages([
            ("system", "You are a clinical reasoning assistant that provides step-by-step diagnostic reasoning based on medical guidelines."),
            ("user", """
            Please analyze the following clinical vignette and provide step-by-step diagnostic reasoning.
            Ground your reasoning in the provided medical guidelines.
            
            Clinical Vignette:
            {vignette}
            
            Relevant Guidelines:
            {guidelines}
            
            {format_instructions}
            """)
        ])
        
        chain = prompt | self.llm
        if output_parser:
            chain = chain | output_parser
            
        return chain.invoke({
            "vignette": vignette,
            "guidelines": "\n".join(guidelines),
            "format_instructions": output_parser.get_format_instructions() if output_parser else ""
        })
    
    def evaluate_reasoning_quality(
        self,
        reasoning: str,
        gold_standard: str
    ) -> Dict[str, float]:
        """Evaluate the quality of generated reasoning against a gold standard."""
        prompt = ChatPromptTemplate.from_messages([
            ("system", "You are an expert evaluator of clinical reasoning quality."),
            ("user", """
            Please evaluate the following clinical reasoning against the gold standard.
            Provide scores for accuracy, completeness, and guideline adherence.
            
            Generated Reasoning:
            {reasoning}
            
            Gold Standard:
            {gold_standard}
            """)
        ])
        
        chain = prompt | self.llm
        return chain.invoke({
            "reasoning": reasoning,
            "gold_standard": gold_standard
        }) 