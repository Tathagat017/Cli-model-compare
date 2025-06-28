# LLM Comparison CLI Tool

A comprehensive command-line tool for comparing different types of Large Language Models (LLMs) across multiple providers including OpenAI, Google Gemini, and Hugging Face.

## Features

- **Multi-Provider Support**: Compare models from OpenAI, Google Gemini, and Hugging Face
- **Model Type Classification**: Compare Base, Instruct, and Fine-tuned models
- **Local Execution**: Run Hugging Face models locally with GPU acceleration
- **Token Usage Tracking**: Monitor token consumption and response times
- **Detailed Analysis**: Generate comprehensive summaries with model characteristics
- **Flexible CLI**: Easy-to-use command-line interface with multiple options

## Installation

1. **Clone/Create the project directory**

```bash
mkdir llm-comparison-tool
cd llm-comparison-tool
```

2. **Create and activate virtual environment**

```bash
python -m venv venv

# On Windows
venv\Scripts\activate

# On macOS/Linux
source venv/bin/activate
```

3. **Install dependencies**

```bash
pip install -r requirements.txt
```

## Configuration

### API Keys (Optional but recommended)

For full functionality, set up API keys for external providers:

```bash
# OpenAI API Key
export OPENAI_API_KEY="your-openai-api-key"

# Google Gemini API Key
export GOOGLE_API_KEY="your-google-api-key"
```

**Note**: Hugging Face models run locally and don't require API keys.

## Usage

### List Available Models

```bash
python llm_cli.py list-models
```

This will show all available models grouped by type (Base, Instruct, Fine-tuned).

### Compare Multiple Models

```bash
# Compare specific models
python llm_cli.py compare "What is machine learning?" -m gpt2 -m gpt-3.5-turbo -m gemini-pro

# Compare one model from each type
python llm_cli.py compare "Explain quantum computing" --all-types

# Specify maximum tokens
python llm_cli.py compare "Write a poem about AI" -m gpt2 -m gpt-3.5-turbo --max-tokens 200
```

### Single Model Query

```bash
python llm_cli.py single "What is the capital of France?" gpt2
```

## Model Types

### Base Models

- **Description**: Pre-trained on large text corpora without task-specific fine-tuning
- **Characteristics**: Raw language understanding, may require careful prompting
- **Examples**: GPT-2, DistilGPT-2

### Instruct Models

- **Description**: Fine-tuned to follow instructions and engage in conversations
- **Characteristics**: Better at understanding user requests, more reliable responses
- **Examples**: GPT-3.5 Turbo, GPT-4, Gemini Pro, FLAN-T5

### Fine-tuned Models

- **Description**: Specialized for specific tasks or domains
- **Characteristics**: Excel in particular areas, may be less general-purpose
- **Examples**: CodeBERT (code understanding), DialoGPT (conversations)

## Available Models

### Hugging Face Models

- **gpt2**: GPT-2 Base model
- **distilgpt2**: Smaller, faster GPT-2 variant
- **microsoft/DialoGPT-medium**: Conversational model
- **google/flan-t5-small**: Instruction-tuned T5
- **microsoft/codebert-base**: Code understanding model

### OpenAI Models

- **gpt-3.5-turbo**: Instruction-following chat model
- **gpt-4**: Advanced reasoning capabilities

### Gemini Models

- **gemini-pro**: Google's multimodal instruction model

## Output

The tool generates:

1. **Console Output**: Formatted comparison results with colors
2. **summary.md**: Detailed markdown report with:
   - Model responses
   - Performance metrics
   - Model characteristics
   - Comparative analysis

## Examples

### Basic Comparison

```bash
python llm_cli.py compare "Explain photosynthesis" --all-types
```

### Code-Related Query

```bash
python llm_cli.py compare "Write a Python function to sort a list" -m gpt2 -m microsoft/codebert-base -m gpt-3.5-turbo
```

### Creative Writing

```bash
python llm_cli.py compare "Write a short story about time travel" -m gpt2 -m gpt-4 --max-tokens 300
```

## System Requirements

- **Python**: 3.8 or higher
- **Memory**: 8GB+ RAM recommended for larger Hugging Face models
- **GPU**: Optional but recommended for faster Hugging Face model inference
- **Storage**: 2-10GB depending on which models you use

## Troubleshooting

### Common Issues

1. **CUDA Out of Memory**: Reduce model size or use CPU-only mode
2. **API Key Errors**: Ensure environment variables are set correctly
3. **Model Loading Issues**: Some Hugging Face models may require authentication

### Performance Tips

- Use smaller models (distilgpt2, flan-t5-small) for faster responses
- Enable GPU acceleration for Hugging Face models
- Set reasonable max_tokens limits to control response length

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request

## License

This project is open-source and available under the MIT License.

## Acknowledgments

- Hugging Face for the transformers library
- OpenAI for their API
- Google for Gemini API
- The open-source community for various model implementations
