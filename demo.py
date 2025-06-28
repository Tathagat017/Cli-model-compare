#!/usr/bin/env python3
"""
Demo script for the LLM Comparison CLI Tool

This script demonstrates the functionality of the LLM comparison tool
with various examples and use cases.
"""

import subprocess
import sys
import time
from colorama import init, Fore, Style

# Initialize colorama
init()

def run_command(command, description):
    """Run a command and display the output"""
    print(f"\n{Fore.CYAN}{'='*60}{Style.RESET_ALL}")
    print(f"{Fore.CYAN}{description}{Style.RESET_ALL}")
    print(f"{Fore.CYAN}{'='*60}{Style.RESET_ALL}")
    print(f"{Fore.YELLOW}Command: {command}{Style.RESET_ALL}\n")
    
    try:
        result = subprocess.run(command, shell=True, capture_output=True, text=True)
        if result.stdout:
            print(result.stdout)
        if result.stderr:
            print(f"{Fore.RED}Errors:{Style.RESET_ALL}")
            print(result.stderr)
        return result.returncode == 0
    except Exception as e:
        print(f"{Fore.RED}Error running command: {e}{Style.RESET_ALL}")
        return False

def main():
    """Run the demo"""
    print(f"{Fore.GREEN}{'='*80}{Style.RESET_ALL}")
    print(f"{Fore.GREEN}LLM COMPARISON TOOL DEMONSTRATION{Style.RESET_ALL}")
    print(f"{Fore.GREEN}{'='*80}{Style.RESET_ALL}")
    
    # Demo 1: List available models
    run_command(
        "python llm_cli.py list-models",
        "Demo 1: Listing All Available Models"
    )
    
    # Demo 2: Simple query with a Hugging Face model
    run_command(
        'python llm_cli.py single "The future of AI is" distilgpt2 --max-tokens 20',
        "Demo 2: Single Model Query (DistilGPT-2)"
    )
    
    # Demo 3: Compare multiple models (if API keys are available)
    print(f"\n{Fore.YELLOW}Note: The following demos require API keys for full functionality{Style.RESET_ALL}")
    print(f"{Fore.YELLOW}Set OPENAI_API_KEY and GOOGLE_API_KEY environment variables for complete testing{Style.RESET_ALL}")
    
    # Demo 4: Show help
    run_command(
        "python llm_cli.py --help",
        "Demo 4: CLI Help Information"
    )
    
    print(f"\n{Fore.GREEN}{'='*80}{Style.RESET_ALL}")
    print(f"{Fore.GREEN}DEMO COMPLETED{Style.RESET_ALL}")
    print(f"{Fore.GREEN}{'='*80}{Style.RESET_ALL}")
    
    print(f"\n{Fore.CYAN}To test with API models, set your API keys:{Style.RESET_ALL}")
    print(f"{Fore.YELLOW}export OPENAI_API_KEY='your-openai-key'{Style.RESET_ALL}")
    print(f"{Fore.YELLOW}export GOOGLE_API_KEY='your-google-key'{Style.RESET_ALL}")
    
    print(f"\n{Fore.CYAN}Then try these commands:{Style.RESET_ALL}")
    print(f'{Fore.YELLOW}python llm_cli.py compare "What is machine learning?" --all-types{Style.RESET_ALL}')
    print(f'{Fore.YELLOW}python llm_cli.py compare "Write a Python function" -m gpt2 -m gpt-3.5-turbo{Style.RESET_ALL}')

if __name__ == "__main__":
    main() 