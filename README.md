# SWE-Bench Prompting, Debugging, and Innovation Project

## Project Overview

This project is based on SWE-Bench-Verified benchmark and focuses on prompt design, inference, and evaluation for software engineering tasks. The project provides a complete RAG (Retrieval-Augmented Generation) workflow, supports multiple large language models, and includes an innovative LangGraph workflow.

### Key Features

- **Curated Subset**: Deep research based on 10 carefully selected instances from the astropy project
- **RAG Enhancement**: Integrated BM25 retrieval and context-enhanced prompt generation
- **Multi-Model Support**: Supports OpenAI GPT, Anthropic Claude, xAI Grok, and other models
- **Self-Repair Mechanism**: Implements Self-Repair prompt strategies to improve code fix quality
- **Complete Evaluation**: Docker-based automated testing and evaluation framework
- **LangGraph Workflow**: Optional LangGraph-driven intelligent workflow

### Project Architecture

```
SWE-bench_Test/
├── subset_10_swebench/          # Core working directory
│   ├── subset_10/               # Dataset with 10 instances
│   ├── subset_2/                # Simplified dataset with 2 instances
│   ├── retrieval/               # BM25 retrieval results
│   ├── text_ds/                # Generated text datasets
│   │   ├── cot/                 # Chain-of-Thought prompts
│   │   └── self_repair/         # Self-Repair prompts
│   ├── preds/                   # Model prediction results
│   └── generate_prompt.py       # Prompt generation script
├── swebench/                    # SWE-bench core library
│   ├── inference/               # Inference module
│   │   ├── run_api.py          # API inference script
│   │   └── langgraph_patch_flow.py  # LangGraph workflow
│   └── harness/                 # Evaluation framework
└── astropy/                     # Test astropy project
```

## Dataset Details

### Subset 10 Instances (astropy project)
- `astropy__astropy-12907` 
- `astropy__astropy-13033` 
- `astropy__astropy-13236` 
- `astropy__astropy-13398` 
- `astropy__astropy-13453` 
- `astropy__astropy-13579` 
- `astropy__astropy-13977` 
- `astropy__astropy-14096` 
- `astropy__astropy-14182` 
- `astropy__astropy-14309` 

### Core Files Description
- **Added Files**:
  - `subset_10_swebench/generate_prompt.py` - Custom prompt generator
  - `subset_10_swebench/save_dataset.py` - Dataset saving utility
  - `swebench/inference/langgraph_patch_flow.py` - LangGraph workflow implementation
- **Modified Files**:
  - `swebench/inference/run_api.py` - Enhanced API inference script

### Workflow Description
- **Parts 1 & 2**: Use `run_api.py` directly for inference and evaluation
- **Part 3**: Set `USE_LANGGRAPH=1` to enable LangGraph workflow

## Installation and Environment Setup

### 1. Clone the Project

```bash
git clone <repository-url>
cd SWE-bench_Test
```

### 2. Install Dependencies

#### Basic Installation
```bash
# Install core dependencies
pip install -e .

# Install inference-related dependencies
pip install -e ".[inference]"

# Install dataset processing dependencies
pip install -e ".[datasets]"
```

#### Optional Dependencies
```bash
# LangGraph workflow support
pip install langgraph langchain-openai

# Development tools
pip install -e ".[test,docs]"
```

### 3. API Key Configuration

#### OpenAI API
```bash
export OPENAI_API_KEY=sk-your-openai-key-here
```

#### OpenRouter API (Recommended, supports more models)
```bash
export OPENROUTER_API_KEY=or-your-openrouter-key-here
```

#### Anthropic API
```bash
export ANTHROPIC_API_KEY=sk-ant-your-anthropic-key-here
```

### 4. LangGraph Configuration

If using LangGraph workflow:

```bash
export USE_LANGGRAPH=1
export LANGGRAPH_STRICT_MODE=both
export LANGGRAPH_MAX_RETRIES=4
export LANGCHAIN_TRACING_V2=1
export LANGCHAIN_PROJECT=my-swebench-run
# export LANGCHAIN_API_KEY=<your_langsmith_api_key>  # Optional, for tracing
```

## Core Components

### 1. Prompt Generator (`generate_prompt.py`)

Supports two prompt strategies:

Chain-of-Thought (CoT) Prompts and Self-Repair Prompts

### 2. Inference Engine (`run_api.py`)

#### Supported Models
- **OpenAI**: GPT-5
- **Anthropic**: Claude-Sonnet-4.5 

### 3. LangGraph Workflow (`langgraph_patch_flow.py`)

#### Workflow Features
- **State Management**: Maintains inference state and intermediate results
- **Conditional Branching**: Selects different paths based on result quality
- **Iterative Optimization**: Supports multi-round inference and optimization
- **Error Handling**: Intelligent retry and fallback strategies

### 4. Evaluation Framework (`harness/`)

#### Dockerized Evaluation
- Isolated testing environment
- Reproducible evaluation results
- Support for multiple programming languages and frameworks

## Usage Guide

### Subset 10: Complete RAG Workflow

Data is located in `subset_10_swebench/subset_10`. To rebuild retrieval outputs and text datasets, follow these steps:

#### Step 1: BM25 Retrieval

Build retrieval index for the codebase to enhance context:

```bash
python -m swebench.inference.make_datasets.bm25_retrieval \
  --dataset_name_or_path ./subset_10_swebench/subset_10 \
  --splits test \
  --output_dir ./subset_10_swebench/retrieval
```

**Parameter Description**:
- `--dataset_name_or_path`: Dataset path
- `--splits`: Split to process (test)
- `--output_dir`: Retrieval results output directory

#### Step 2: Generate Prompt Datasets

`generate_prompt.py` generates both CoT and Self-Repair prompt strategy datasets:

```bash
# Generate non-consistent prompts (recommended for research)
python ./subset_10_swebench/generate_prompt.py --subset_type 10 --consistent false

# Generate consistent prompts (recommended for production)
python ./subset_10_swebench/generate_prompt.py --subset_type 10 --consistent true
```

**Output Locations**:
- CoT prompts: `./subset_10_swebench/text_ds/cot/`
- Self-Repair prompts: `./subset_10_swebench/text_ds/self_repair/`

**Default Configuration** (k=8, mcc=100000, tokenizer=cl100k):
```
./subset_10_swebench/text_ds/cot/.__subset_10_swebench__subset_10__style-3-cot__fs-bm25__k-8__mcc-100000-cl100k
```

#### Step 3: Model Inference

Supports multiple models and allows free switching:

```bash
# GPT-5 inference
python -m swebench.inference.run_api \
  --dataset_name_or_path ./subset_10_swebench/text_ds/cot/.__subset_10_swebench__subset_10__style-3-cot__fs-bm25__k-8__mcc-100000-cl100k \
  --split test \
  --model_name_or_path openai/gpt-5 \
  --output_dir ./subset_10_swebench/preds

# Grok Code Fast inference
python -m swebench.inference.run_api \
  --dataset_name_or_path ./subset_10_swebench/text_ds/cot/.__subset_10_swebench__subset_10__style-3-cot__fs-bm25__k-8__mcc-100000-cl100k \
  --split test \
  --model_name_or_path x-ai/grok-code-fast-1 \
  --output_dir ./subset_10_swebench/preds

# Claude Sonnet 4.5 inference
python -m swebench.inference.run_api \
  --dataset_name_or_path ./subset_10_swebench/text_ds/cot/.__subset_10_swebench__subset_10__style-3-cot__fs-bm25__k-8__mcc-100000-cl100k \
  --split test \
  --model_name_or_path anthropic/claude-sonnet-4.5 \
  --output_dir ./subset_10_swebench/preds
```

**Advanced Parameters**:
```bash
# Set maximum cost limit
python -m swebench.inference.run_api \
  --dataset_name_or_path <dataset_path> \
  --split test \
  --model_name_or_path openai/gpt-5 \
  --output_dir ./subset_10_swebench/preds \
  --max_cost 50.0

# Use sharding for large datasets
python -m swebench.inference.run_api \
  --dataset_name_or_path <dataset_path> \
  --split test \
  --model_name_or_path openai/gpt-5 \
  --output_dir ./subset_10_swebench/preds \
  --shard_id 0 \
  --num_shards 4
```

#### Step 4: Evaluate Results

Use Docker environment for automated evaluation:

```bash
python -m swebench.harness.run_evaluation \
  --dataset_name ./subset_10_swebench/subset_10 \
  --split test \
  --predictions_path ./subset_10_swebench/preds/gpt-5__.__subset_10_swebench__subset_10__style-3-cot__fs-bm25__k-8__mcc-100000-cl100k__test.jsonl \
  --run_id openrouter-gpt5-sub10 \
  --max_workers 1
```

### Subset 2: Quick Iteration Workflow

> Suitable for quick iteration and debugging on 1-2 instances. The steps mirror subset 10: prepare subset, run BM25 retrieval, create text dataset, inference, then evaluation.

#### Step 1: Create subset_2

```bash
python ./subset_10_swebench/save_dataset.py --subset_type 2
```

**Included Instances**:
- `astropy__astropy-13236` - Time handling improvements
- `astropy__astropy-14182` - Algorithm improvements

#### Step 2: BM25 Retrieval

```bash
python -m swebench.inference.make_datasets.bm25_retrieval \
  --dataset_name_or_path ./subset_10_swebench/subset_2 \
  --splits test \
  --output_dir ./subset_10_swebench/retrieval
```

#### Step 3: Generate CoT Text Dataset

```bash
python ./subset_10_swebench/generate_prompt.py --subset_type 2 --consistent true
```

#### Step 4: Model Inference

```bash
# GPT-5 inference
python -m swebench.inference.run_api \
  --dataset_name_or_path ./subset_10_swebench/text_ds/cot/.__subset_10_swebench__subset_2__style-3-cot__fs-bm25__k-8__mcc-100000-cl100k \
  --split test \
  --model_name_or_path openai/gpt-5 \
  --output_dir ./subset_10_swebench/preds

# Claude Sonnet 4.5 inference
python -m swebench.inference.run_api \
  --dataset_name_or_path ./subset_10_swebench/text_ds/cot/.__subset_10_swebench__subset_2__style-3-cot__fs-bm25__k-8__mcc-100000-cl100k \
  --split test \
  --model_name_or_path anthropic/claude-sonnet-4.5 \
  --output_dir ./subset_10_swebench/preds
```

#### Step 5: Evaluate Results

```bash
# GPT-5 evaluation
python -m swebench.harness.run_evaluation \
  --dataset_name ./subset_10_swebench/subset_2 \
  --split test \
  --predictions_path ./subset_10_swebench/preds/gpt-5__.__subset_10_swebench__subset_2__style-3-cot__fs-bm25__k-8__mcc-100000-cl100k__test.jsonl \
  --run_id openrouter-gpt5-sub2 \
  --max_workers 1

# Claude Sonnet 4.5 evaluation
python -m swebench.harness.run_evaluation \
  --dataset_name ./subset_10_swebench/subset_2 \
  --split test \
  --predictions_path ./subset_10_swebench/preds/claude-sonnet-4.5__.__subset_10_swebench__subset_2__style-3-cot__fs-bm25__k-8__mcc-100000-cl100k__test.jsonl \
  --run_id openrouter-cl-sub2 \
  --max_workers 1
```

### LangGraph Workflow (Optional)

LangGraph provides a more intelligent inference workflow with state management and conditional branching.

#### Enable LangGraph

After setting environment variables, `run_api` will automatically switch to LangGraph mode:

```bash
# Set LangGraph environment variables
export USE_LANGGRAPH=1
export LANGGRAPH_STRICT_MODE=both
export LANGGRAPH_MAX_RETRIES=4
export LANGCHAIN_TRACING_V2=1
export LANGCHAIN_PROJECT=my-swebench-run

# Run LangGraph inference
python -m swebench.inference.run_api \
  --dataset_name_or_path ./subset_10_swebench/text_ds/cot/.__subset_10_swebench__subset_2__style-3-cot__fs-bm25__k-8__mcc-100000-cl100k \
  --split test \
  --model_name_or_path anthropic/claude-sonnet-4.5 \
  --output_dir ./subset_10_swebench/preds/lang_preds

# Evaluate LangGraph results
python -m swebench.harness.run_evaluation \
  --dataset_name ./subset_10_swebench/subset_2 \
  --split test \
  --predictions_path ./subset_10_swebench/preds/lang_preds/claude-sonnet-4.5__.__subset_10_swebench__subset_2__style-3-cot__fs-bm25__k-8__mcc-100000-cl100k__test.jsonl \
  --run_id openrouter-cl-sub2 \
  --max_workers 1
```