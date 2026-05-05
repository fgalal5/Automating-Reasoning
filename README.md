# ECE 219 Project 3: Automating Reasoning and Data Mining with Small Language Models

This project builds an automated data mining pipeline that combines small language model fine-tuning, ReAct-style agentic CSV analysis, and regression modeling to improve reasoning, structured tool use, and predictive performance on real datasets.

## Project Overview

The project is divided into three main parts:

### Part A: Teaching a Small Model to Reason

We fine-tuned a small instruction-tuned language model on math reasoning tasks using LoRA, a parameter-efficient fine-tuning method. The goal was to improve the model’s ability to solve GSM8K-style grade-school math problems without training a large model from scratch.

Key components:
- Baseline evaluation of `Qwen2.5-1.5B-Instruct`
- LoRA fine-tuning on GSM8K
- Few-shot prompting experiments
- Analysis of reasoning failures and accuracy improvements

### Part B: Agentic Data Mining with ReAct

We built a ReAct-style data analysis agent that answers questions about CSV files by planning, writing Python code, executing it, and using structured observations to continue reasoning.

Key components:
- Planner agent with structured JSON output
- Code generation agent
- Python execution tool
- Observation summarizer
- Evaluation on selected InfiAgent-DABench tasks

### Part C: Regression Analysis

We performed exploratory data analysis, feature engineering, and regression modeling on a synthetic diamonds dataset to predict diamond prices.

Key components:
- Correlation, distribution, and categorical analysis
- Encoding and standardization of features
- Feature selection using mutual information and F-scores
- Linear regression, Lasso, and Ridge regression
- 10-fold cross-validation using RMSE

## Tech Stack

- Python
- PyTorch
- Hugging Face Transformers
- Hugging Face PEFT
- LoRA fine-tuning
- Outlines
- Pydantic
- Pandas
- NumPy
- Scikit-learn
- Matplotlib
- Google Colab

## Models and Datasets

### Models
- `Qwen2.5-1.5B-Instruct`
- `Qwen3-4B-Instruct-2507`

### Datasets
- GSM8K
- InfiAgent-DABench / DAEval
- Synthetic Diamonds Dataset

## Repository Structure

```text
.
├── Part_A_Reasoning/
│   ├── baseline_evaluation.ipynb
│   ├── lora_finetuning.ipynb
│   └── few_shot_experiments.ipynb
│
├── Part_B_ReAct_Agent/
│   ├── react_agent.ipynb
│   ├── planner.py
│   ├── coder.py
│   ├── executor.py
│   └── observer.py
│
├── Part_C_Regression/
│   ├── diamonds_analysis.ipynb
│   ├── diamonds_standardized.csv
│   └── diamonds_selected.csv
│
├── report.pdf
└── README.md
