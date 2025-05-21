# 🛡️ LLM Safety Evaluation Framework

A comprehensive framework of sensitive knowledge processing, evaluating and detecting potential safety issues in Large Language Models (LLMs).


## 📋 Overview

This framework provides a multi-faceted approach to evaluate LLM safety through:
- 🔍 Jailbreak detection
- 🚫 Harmful content filtering
- 🔄 Counterfactual evaluation
- ✅ Policy compliance assessment

## 🧩 Core Components

### 🔧 Evaluation Tools
- `choice.py`: Multiple-choice data processing
- `judge.py`: Judgment-based data processing
- `open.py`: Open-ended response processing
- `counterfactual_eval.py`: Counterfactual evaluation framework
- `jailbreak_detection.py`: Multi-mode detection system supporting:
  - 🤖 GPT-based judgment
  - 📊 Meta policy scoring
  - 🔗 Combined detection approach

### 🛡️ Filtering System
- `filter_judge_enhanced.py`: Enhanced filtering for judgment-based content
- `filter_choice_enhanced.py`: Enhanced filtering for multiple-choice content
- `filter_raw_enhanced.py`: Raw content filtering with improved accuracy

## 📁 Directory Structure
- `mask/`: 🎭 Masking utilities for sensitive content
- `judge/`: ⚖️ Judgment-based evaluation components
- `choice/`: ✅ Multiple-choice evaluation components
- `dataset/`: 📊 Sample datasets (partial release for review)

## 🚀 Usage

### 📝 Open-Ended Evaluation
```bash
python open.py \
    --target_model "gpt-4o" \
    --eval_model "gpt-4o" \
    --temperature 0.7 \
    --reasoning \
    --sample 200 \
    --dataset_path "dataset/VENOM_dataset/fraud/fraud_final_translated.json" \
    --output_dir "evaluation_results" \
    --log_dir "logs"
```

### ✅ Multiple-Choice Evaluation
```bash
python choice.py \
    --target_model "gpt-4o" \
    --temperature 0.0 \
    --reasoning \
    --sample 200 \
    --dataset_path "dataset/VENOM_dataset/fraud/fraud_choice_translated.json"
```

### ⚖️ Judgment-Based Evaluation
```bash
python judge.py \
    --target_model "gpt-4o" \
    --temperature 0.7 \
    --reasoning \
    --sample 200 \
    --dataset_path "dataset/VENOM_dataset/fraud/fraud_judge_translated.json" \
    --output_dir "evaluation_results" \
    --log_dir "logs"
```

### 🔄 Counterfactual Evaluation
```bash
python counterfactual_eval.py \
    --input_file "dataset/VENOM_dataset/counterfactual_plan/counterfactual_planning.json" \
    --sample 10 \
    --target_model "gpt-4o"
```

### 🛡️ Jailbreak Detection
```bash
python jailbreak_detection.py \
    --input_file "dataset/VENOM_dataset/counterfactual_judge/mixed_harmful_qa_0.50_100.json" \
    --model "gpt-4o" \
    --mode "meta_policy"
```

## ✨ Features

1. **🔍 Multi-Mode Detection**
   - 🤖 GPT-based judgment
   - 📊 Policy-based scoring
   - 🔗 Combined approach

2. **🛡️ Enhanced Filtering**
   - 📈 Improved accuracy
   - 🔄 Multiple content types support
   - ⚙️ Configurable thresholds

3. **📊 Comprehensive Evaluation**
   - 🔄 Counterfactual analysis
   - ⚖️ Balanced judgment
   - ✅ Results verification

## 📝 LLM-as-a-Judge Sample Generation Methodology

> **Note**: This section details the implementation of the generation method of adversarial samples for LLM-as-a-Judge testing described in our paper. A more comprehensive version will be included in the paper's appendix.

### 1. 🎭 Sensitive Information Recognition and Masking (mask.py)

`mask.py` implements an iterative sensitive information extraction and masking system with the following workflow:

1. **🔄 Multi-round LLM Calls**
   - Performs multiple (n_iter) LLM calls for each QA pair
   - Accumulates entity recognition, mask suggestions, and sensitive terms
   - Conducts extraction based on previous round's mask results, achieving "layer-by-layer purification"

2. **🎯 Mask Management**
   - Uses `MaskManager` class to manage sensitive word masks
   - Generates unique mask codes for each sensitive term
   - Provides harmless replacement text
   - Supports bidirectional mapping between sensitive terms and masks/harmless text

3. **💾 Result Saving**
   - Writes results to .jsonl files in real-time
   - Automatically backs up old output files
   - Supports checkpoint continuation

### 2. 🎮 Controlled Harmful Content Generation (joint.py)

`joint.py` implements a probability-based harmful content control generation system:

1. **🎲 Probability-controlled Replacement**
   - Supports setting probability for retaining harmful content (keep_harmful_prob)
   - Controls error range through tolerance parameter
   - Prioritizes longer sensitive terms to ensure replacement accuracy

2. **📋 Data Format Processing**
   - Supports both standard JSON and JSONL formats
   - Automatically fixes format issues
   - Performs data validation and cleaning

3. **🎯 Sample Generation**
   - Supports specified sample count
   - Prevents duplicate generation
   - Ensures harmful content ratio stays within target range

### 🛠️ Implementation Details

The implementation follows the methodology described in our paper's LLM-as-a-Judge approach, with the following key features:

- **🔄 Iterative Refinement**: Multiple rounds of LLM-based judgment to ensure comprehensive coverage
- **🎮 Controlled Generation**: Precise control over harmful content ratio for balanced evaluation
- **✅ Quality Assurance**: Multiple validation steps to ensure data quality and consistency

### 🚀 Usage Examples

```bash
# 🎭 Sensitive Information Recognition and Masking
python mask.py \
    --input ./data/sensitive_data.json \
    --output ./data/masked_data.jsonl \
    --n_iter 3

# 🎮 Controlled Harmful Content Generation
python joint.py \
    --input ./data/masked_data.jsonl \
    --output ./data/generated_data.json \
    --keep_harmful_prob 0.5 \
    --sample_count 100 \
    --tolerance 0.05
```

## 🤝 Research Ethics

This framework is developed for research purposes to:
- 🛡️ Improve LLM safety mechanisms
- 🔍 Understand model vulnerabilities
- 🌟 Contribute to responsible AI development

## 📚 Citation

If you use this framework in your research, please cite our paper (citation to be added upon acceptance).

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 📧 Contact

For questions regarding the framework or dataset access, please contact the corresponding author.

---
<div align="center">
  <img src="https://img.shields.io/badge/LLM-Safety-blue" alt="LLM Safety">
  <img src="https://img.shields.io/badge/Evaluation-Framework-green" alt="Evaluation Framework">
  <img src="https://img.shields.io/badge/Research-Tool-orange" alt="Research Tool">
  <img src="https://img.shields.io/badge/MIT-License-yellow" alt="MIT License">
</div>


> **Note**: This framework integrates with [easy-dataset](https://github.com/ConardLi/easy-dataset) for knowledge chunk generation and processing.
