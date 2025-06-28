#!/usr/bin/env python3
"""
LLM Comparison CLI Tool

A comprehensive CLI tool for comparing Base, Instruct, and Fine-tuned models
from different providers (OpenAI, Gemini, Hugging Face).
"""

import os
import json
import time
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from enum import Enum

import click
import torch
from transformers import (
    AutoTokenizer, AutoModelForCausalLM, AutoModelForSeq2SeqLM,
    pipeline
)
from colorama import init, Fore, Style
from tabulate import tabulate
import tiktoken

# Initialize colorama for cross-platform colored output
init()

class ModelType(Enum):
    BASE = "base"
    INSTRUCT = "instruct"
    FINE_TUNED = "fine-tuned"

class Provider(Enum):
    HUGGINGFACE = "huggingface"
    OPENAI = "openai"
    GEMINI = "gemini"

@dataclass
class ModelConfig:
    name: str
    provider: Provider
    model_type: ModelType
    max_tokens: int
    context_size: int
    description: str
    hf_model_name: Optional[str] = None
    
@dataclass
class ModelResponse:
    model_name: str
    model_type: ModelType
    provider: Provider
    response: str
    tokens_used: int
    max_context: int
    response_time: float
    characteristics: str

class HuggingFaceHandler:
    """Handler for Hugging Face models"""
    
    def __init__(self):
        self.tokenizers = {}
        self.models = {}
        self.pipelines = {}
    
    def load_model(self, model_name: str, model_type: ModelType) -> bool:
        """Load a Hugging Face model"""
        try:
            print(f"{Fore.YELLOW}Loading {model_name}...{Style.RESET_ALL}")
            
            # Load tokenizer
            tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
            
            # Create a simple text generation pipeline without device_map
            try:
                pipe = pipeline(
                    "text-generation",
                    model=model_name,
                    tokenizer=tokenizer,
                    torch_dtype=torch.float32,  # Use float32 for better compatibility
                    trust_remote_code=True
                )
                self.pipelines[model_name] = pipe
                self.tokenizers[model_name] = tokenizer
                return True
            except Exception as e:
                print(f"{Fore.RED}Error creating pipeline: {e}{Style.RESET_ALL}")
                # Fallback: try with minimal settings
                pipe = pipeline(
                    "text-generation",
                    model=model_name,
                    tokenizer=tokenizer
                )
                self.pipelines[model_name] = pipe
                self.tokenizers[model_name] = tokenizer
                return True
                
        except Exception as e:
            print(f"{Fore.RED}Failed to load model {model_name}: {e}{Style.RESET_ALL}")
            return False
    
    def generate_response(self, model_name: str, prompt: str, max_tokens: int = 150) -> Tuple[str, int, float]:
        """Generate response using Hugging Face model"""
        if model_name not in self.pipelines:
            raise ValueError(f"Model {model_name} not loaded")
        
        start_time = time.time()
        
        try:
            pipe = self.pipelines[model_name]
            tokenizer = self.tokenizers[model_name]
            
            # Count input tokens
            input_tokens = len(tokenizer.encode(prompt))
            
            # Generate response
            outputs = pipe(
                prompt,
                max_new_tokens=max_tokens,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
                pad_token_id=tokenizer.eos_token_id,
                return_full_text=False
            )
            
            response = outputs[0]['generated_text'].strip()
            
            # Count output tokens
            output_tokens = len(tokenizer.encode(response))
            total_tokens = input_tokens + output_tokens
            
            response_time = time.time() - start_time
            
            return response, total_tokens, response_time
            
        except Exception as e:
            return f"Error generating response: {e}", 0, time.time() - start_time

class OpenAIHandler:
    """Handler for OpenAI models"""
    
    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.getenv('OPENAI_API_KEY')
        if not self.api_key:
            print(f"{Fore.YELLOW}Warning: OpenAI API key not found. Set OPENAI_API_KEY environment variable.{Style.RESET_ALL}")
            self.client = None
            return
        
        try:
            import openai
            self.client = openai.OpenAI(api_key=self.api_key)
        except ImportError:
            print(f"{Fore.RED}OpenAI library not installed. Install with: pip install openai{Style.RESET_ALL}")
            self.client = None
    
    def generate_response(self, model_name: str, prompt: str, max_tokens: int = 150) -> Tuple[str, int, float]:
        """Generate response using OpenAI model"""
        if not self.client or not self.api_key:
            return "OpenAI API key not configured", 0, 0.0
        
        start_time = time.time()
        
        try:
            response = self.client.chat.completions.create(
                model=model_name,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=max_tokens,
                temperature=0.7
            )
            
            response_time = time.time() - start_time
            content = response.choices[0].message.content
            tokens_used = response.usage.total_tokens
            
            return content, tokens_used, response_time
            
        except Exception as e:
            return f"Error with OpenAI API: {e}", 0, time.time() - start_time

class GeminiHandler:
    """Handler for Google Gemini models"""
    
    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.getenv('GOOGLE_API_KEY')
        if not self.api_key:
            print(f"{Fore.YELLOW}Warning: Google API key not found. Set GOOGLE_API_KEY environment variable.{Style.RESET_ALL}")
            self.genai = None
            return
        
        try:
            import google.generativeai as genai
            genai.configure(api_key=self.api_key)
            self.genai = genai
        except ImportError:
            print(f"{Fore.RED}Google Generative AI library not installed. Install with: pip install google-generativeai{Style.RESET_ALL}")
            self.genai = None
    
    def generate_response(self, model_name: str, prompt: str, max_tokens: int = 150) -> Tuple[str, int, float]:
        """Generate response using Gemini model"""
        if not self.genai or not self.api_key:
            return "Google API key not configured", 0, 0.0
        
        start_time = time.time()
        
        try:
            model = self.genai.GenerativeModel(model_name)
            response = model.generate_content(prompt)
            
            response_time = time.time() - start_time
            content = response.text
            
            # Estimate token count (Gemini doesn't provide exact counts in free tier)
            estimated_tokens = len(content.split()) * 1.3  # Rough estimation
            
            return content, int(estimated_tokens), response_time
            
        except Exception as e:
            return f"Error with Gemini API: {e}", 0, time.time() - start_time

class LLMComparator:
    """Main class for comparing different LLM models"""
    
    def __init__(self):
        self.hf_handler = HuggingFaceHandler()
        self.openai_handler = OpenAIHandler()
        self.gemini_handler = GeminiHandler()
        
        self.model_configs = {
            # Hugging Face Base Models
            "gpt2": ModelConfig(
                name="GPT-2 Base",
                provider=Provider.HUGGINGFACE,
                model_type=ModelType.BASE,
                max_tokens=1024,
                context_size=1024,
                description="Base transformer model, not instruction-tuned",
                hf_model_name="gpt2"
            ),
            "distilgpt2": ModelConfig(
                name="DistilGPT-2 Base",
                provider=Provider.HUGGINGFACE,
                model_type=ModelType.BASE,
                max_tokens=1024,
                context_size=1024,
                description="Smaller, faster version of GPT-2",
                hf_model_name="distilgpt2"
            ),
            
            # Hugging Face Instruct Models
            "microsoft/DialoGPT-medium": ModelConfig(
                name="DialoGPT Instruct",
                provider=Provider.HUGGINGFACE,
                model_type=ModelType.INSTRUCT,
                max_tokens=1024,
                context_size=1024,
                description="Conversational model trained on dialog data",
                hf_model_name="microsoft/DialoGPT-medium"
            ),
            "google/flan-t5-small": ModelConfig(
                name="FLAN-T5 Instruct",
                provider=Provider.HUGGINGFACE,
                model_type=ModelType.INSTRUCT,
                max_tokens=512,
                context_size=512,
                description="Instruction-tuned T5 model",
                hf_model_name="google/flan-t5-small"
            ),
            
            # Hugging Face Fine-tuned Models
            "microsoft/CodeBERT-base": ModelConfig(
                name="CodeBERT Fine-tuned",
                provider=Provider.HUGGINGFACE,
                model_type=ModelType.FINE_TUNED,
                max_tokens=512,
                context_size=512,
                description="BERT fine-tuned for code understanding",
                hf_model_name="microsoft/codebert-base"
            ),
            
            # OpenAI Models
            "gpt-3.5-turbo": ModelConfig(
                name="GPT-3.5 Turbo",
                provider=Provider.OPENAI,
                model_type=ModelType.INSTRUCT,
                max_tokens=4096,
                context_size=16385,
                description="Instruction-following model optimized for chat"
            ),
            "gpt-4": ModelConfig(
                name="GPT-4",
                provider=Provider.OPENAI,
                model_type=ModelType.INSTRUCT,
                max_tokens=8192,
                context_size=32768,
                description="Advanced reasoning and instruction-following capabilities"
            ),
            
            # Gemini Models
            "gemini-pro": ModelConfig(
                name="Gemini Pro",
                provider=Provider.GEMINI,
                model_type=ModelType.INSTRUCT,
                max_tokens=2048,
                context_size=32768,
                description="Google's instruction-tuned multimodal model"
            )
        }
    
    def list_available_models(self) -> None:
        """Display all available models grouped by type"""
        print(f"\n{Fore.CYAN}Available Models:{Style.RESET_ALL}\n")
        
        for model_type in ModelType:
            print(f"{Fore.GREEN}{model_type.value.upper()} MODELS:{Style.RESET_ALL}")
            models = [
                (key, config) for key, config in self.model_configs.items()
                if config.model_type == model_type
            ]
            
            if models:
                table_data = []
                for key, config in models:
                    table_data.append([
                        key,
                        config.name,
                        config.provider.value,
                        f"{config.context_size:,}",
                        config.description[:50] + ("..." if len(config.description) > 50 else "")
                    ])
                
                print(tabulate(
                    table_data,
                    headers=["Key", "Name", "Provider", "Context Size", "Description"],
                    tablefmt="grid"
                ))
            else:
                print("  No models available")
            print()
    
    def compare_models(self, query: str, model_keys: List[str], max_tokens: int = 150) -> List[ModelResponse]:
        """Compare responses from multiple models"""
        responses = []
        
        print(f"\n{Fore.CYAN}Generating responses for query: {Fore.WHITE}{query}{Style.RESET_ALL}\n")
        
        for model_key in model_keys:
            if model_key not in self.model_configs:
                print(f"{Fore.RED}Model '{model_key}' not found{Style.RESET_ALL}")
                continue
            
            config = self.model_configs[model_key]
            print(f"{Fore.YELLOW}Querying {config.name} ({config.provider.value})...{Style.RESET_ALL}")
            
            try:
                if config.provider == Provider.HUGGINGFACE:
                    # Load model if not already loaded
                    if config.hf_model_name not in self.hf_handler.pipelines:
                        if not self.hf_handler.load_model(config.hf_model_name, config.model_type):
                            continue
                    
                    response, tokens, response_time = self.hf_handler.generate_response(
                        config.hf_model_name, query, max_tokens
                    )
                
                elif config.provider == Provider.OPENAI:
                    response, tokens, response_time = self.openai_handler.generate_response(
                        model_key, query, max_tokens
                    )
                
                elif config.provider == Provider.GEMINI:
                    response, tokens, response_time = self.gemini_handler.generate_response(
                        model_key, query, max_tokens
                    )
                
                # Generate characteristics description
                characteristics = self._get_model_characteristics(config)
                
                model_response = ModelResponse(
                    model_name=config.name,
                    model_type=config.model_type,
                    provider=config.provider,
                    response=response,
                    tokens_used=tokens,
                    max_context=config.context_size,
                    response_time=response_time,
                    characteristics=characteristics
                )
                
                responses.append(model_response)
                
            except Exception as e:
                print(f"{Fore.RED}Error with {config.name}: {e}{Style.RESET_ALL}")
        
        return responses
    
    def _get_model_characteristics(self, config: ModelConfig) -> str:
        """Generate characteristics description for a model"""
        characteristics = []
        
        # Model type characteristics
        if config.model_type == ModelType.BASE:
            characteristics.append("Pre-trained on large text corpus without task-specific fine-tuning")
            characteristics.append("May require careful prompting for specific tasks")
        elif config.model_type == ModelType.INSTRUCT:
            characteristics.append("Fine-tuned to follow instructions and engage in conversations")
            characteristics.append("Better at understanding and responding to user requests")
        elif config.model_type == ModelType.FINE_TUNED:
            characteristics.append("Specialized for specific tasks or domains")
            characteristics.append("May excel in particular areas but less general-purpose")
        
        # Provider-specific characteristics
        if config.provider == Provider.HUGGINGFACE:
            characteristics.append("Open-source model that can be run locally")
            characteristics.append("Full control over inference and customization")
        elif config.provider == Provider.OPENAI:
            characteristics.append("Commercial API with state-of-the-art performance")
            characteristics.append("Optimized for safety and reliability")
        elif config.provider == Provider.GEMINI:
            characteristics.append("Google's multimodal AI with strong reasoning capabilities")
            characteristics.append("Integrated with Google's ecosystem")
        
        return "; ".join(characteristics)

def display_results(responses: List[ModelResponse], query: str) -> None:
    """Display comparison results in a formatted way"""
    print(f"\n{Fore.CYAN}{'='*80}{Style.RESET_ALL}")
    print(f"{Fore.CYAN}COMPARISON RESULTS{Style.RESET_ALL}")
    print(f"{Fore.CYAN}{'='*80}{Style.RESET_ALL}")
    print(f"{Fore.WHITE}Query: {query}{Style.RESET_ALL}\n")
    
    for i, response in enumerate(responses, 1):
        print(f"{Fore.GREEN}[{i}] {response.model_name} ({response.model_type.value}){Style.RESET_ALL}")
        print(f"{Fore.YELLOW}Provider:{Style.RESET_ALL} {response.provider.value}")
        print(f"{Fore.YELLOW}Response:{Style.RESET_ALL}")
        print(f"{response.response}\n")
        print(f"{Fore.YELLOW}Metrics:{Style.RESET_ALL}")
        print(f"  • Tokens Used: {response.tokens_used}")
        print(f"  • Max Context: {response.max_context:,}")
        print(f"  • Response Time: {response.response_time:.2f}s")
        print(f"{Fore.YELLOW}Characteristics:{Style.RESET_ALL}")
        print(f"  {response.characteristics}")
        print(f"{Fore.CYAN}{'-'*80}{Style.RESET_ALL}")

def save_summary(responses: List[ModelResponse], query: str) -> None:
    """Save detailed summary to summary.md"""
    with open("summary.md", "w", encoding="utf-8") as f:
        f.write("# LLM Comparison Summary\n\n")
        f.write(f"**Query:** {query}\n\n")
        f.write(f"**Generated on:** {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        f.write("## Model Responses\n\n")
        
        for i, response in enumerate(responses, 1):
            f.write(f"### {i}. {response.model_name}\n\n")
            f.write(f"**Type:** {response.model_type.value}\n")
            f.write(f"**Provider:** {response.provider.value}\n\n")
            f.write(f"**Response:**\n```\n{response.response}\n```\n\n")
            f.write(f"**Metrics:**\n")
            f.write(f"- Tokens Used: {response.tokens_used}\n")
            f.write(f"- Max Context Size: {response.max_context:,}\n")
            f.write(f"- Response Time: {response.response_time:.2f} seconds\n\n")
            f.write(f"**Model Characteristics:**\n{response.characteristics}\n\n")
            f.write("---\n\n")
        
        # Add comparison table
        f.write("## Comparison Table\n\n")
        table_data = []
        for response in responses:
            table_data.append([
                response.model_name,
                response.model_type.value,
                response.provider.value,
                response.tokens_used,
                f"{response.response_time:.2f}s",
                response.max_context
            ])
        
        f.write(tabulate(
            table_data,
            headers=["Model", "Type", "Provider", "Tokens", "Time", "Max Context"],
            tablefmt="pipe"
        ))
        f.write("\n\n")
        
        # Add analysis section
        f.write("## Analysis\n\n")
        f.write("### Model Type Comparison\n\n")
        f.write("**Base Models:** Pre-trained models without specific instruction tuning. ")
        f.write("They may require more careful prompting but offer more creative freedom.\n\n")
        f.write("**Instruct Models:** Fine-tuned to follow instructions and engage in conversations. ")
        f.write("Generally more reliable for question-answering and task completion.\n\n")
        f.write("**Fine-tuned Models:** Specialized for specific domains or tasks. ")
        f.write("May excel in particular areas but might be less versatile.\n\n")
        
        f.write("### Provider Comparison\n\n")
        f.write("**Hugging Face:** Open-source models that can be run locally with full control.\n\n")
        f.write("**OpenAI:** Commercial API with state-of-the-art performance and safety measures.\n\n")
        f.write("**Gemini:** Google's multimodal AI with strong reasoning and integration capabilities.\n\n")

# CLI Interface
@click.group()
def cli():
    """LLM Comparison CLI Tool
    
    Compare responses from Base, Instruct, and Fine-tuned models across different providers.
    """
    pass

@cli.command()
def list_models():
    """List all available models"""
    comparator = LLMComparator()
    comparator.list_available_models()

@cli.command()
@click.argument('query')
@click.option('--models', '-m', multiple=True, help='Model keys to compare (can be specified multiple times)')
@click.option('--max-tokens', default=150, help='Maximum tokens to generate')
@click.option('--all-types', is_flag=True, help='Compare one model from each type')
def compare(query, models, max_tokens, all_types):
    """Compare model responses to a query"""
    comparator = LLMComparator()
    
    if all_types:
        # Select one model from each type
        selected_models = []
        for model_type in ModelType:
            type_models = [
                key for key, config in comparator.model_configs.items()
                if config.model_type == model_type
            ]
            if type_models:
                selected_models.append(type_models[0])  # Take the first one of each type
        models = selected_models
    
    if not models:
        print(f"{Fore.RED}Please specify models to compare using --models or --all-types{Style.RESET_ALL}")
        print("Use 'llm-cli list-models' to see available models")
        return
    
    responses = comparator.compare_models(query, list(models), max_tokens)
    
    if responses:
        display_results(responses, query)
        save_summary(responses, query)
        print(f"\n{Fore.GREEN}Summary saved to summary.md{Style.RESET_ALL}")
    else:
        print(f"{Fore.RED}No successful responses generated{Style.RESET_ALL}")

@cli.command()
@click.argument('query')
@click.argument('model_key')
@click.option('--max-tokens', default=150, help='Maximum tokens to generate')
def single(query, model_key, max_tokens):
    """Get response from a single model"""
    comparator = LLMComparator()
    responses = comparator.compare_models(query, [model_key], max_tokens)
    
    if responses:
        response = responses[0]
        print(f"\n{Fore.GREEN}{response.model_name} Response:{Style.RESET_ALL}")
        print(f"{response.response}")
        print(f"\n{Fore.YELLOW}Tokens: {response.tokens_used}, Time: {response.response_time:.2f}s{Style.RESET_ALL}")
    else:
        print(f"{Fore.RED}Failed to generate response{Style.RESET_ALL}")

if __name__ == '__main__':
    cli() 