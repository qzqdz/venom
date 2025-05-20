# LLM Safety Evaluation Framework

A comprehensive framework for evaluating and detecting potential safety issues in Large Language Models (LLMs), including jailbreak attempts and harmful content generation.

## Overview

This framework provides a multi-faceted approach to evaluate LLM safety through:
- Jailbreak detection
- Harmful content filtering
- Counterfactual evaluation
- Policy compliance assessment

## Core Components

### Evaluation Tools
- `choice.py`: Multiple-choice data processing
- `judge.py`: Judgment-based data processing
- `open.py`: Open-ended response processing
- `counterfactual_eval.py`: Counterfactual evaluation framework
- `jailbreak_detection.py`: Multi-mode detection system supporting:
  - GPT-based judgment
  - Meta policy scoring
  - Combined detection approach

### Filtering System
- `filter_judge_enhanced.py`: Enhanced filtering for judgment-based content
- `filter_choice_enhanced.py`: Enhanced filtering for multiple-choice content
- `filter_raw_enhanced.py`: Raw content filtering with improved accuracy

## Directory Structure
- `mask/`: Masking utilities for sensitive content
- `judge/`: Judgment-based evaluation components
- `choice/`: Multiple-choice evaluation components
- `dataset/`: Sample datasets (partial release for review)

## Usage

### Open-Ended Evaluation
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

### Multiple-Choice Evaluation
```bash
python choice.py \
    --target_model "gpt-4o" \
    --temperature 0.0 \
    --reasoning \
    --sample 200 \
    --dataset_path "dataset/VENOM_dataset/fraud/fraud_choice_translated.json"
```

### Judgment-Based Evaluation
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

### Counterfactual Evaluation
```bash
python counterfactual_eval.py \
    --input_file "dataset/VENOM_dataset/counterfactual_plan/counterfactual_planning.json" \
    --sample 10 \
    --target_model "Pro/Qwen/Qwen2.5-7B-Instruct"
```

## Features

1. **Multi-Mode Detection**
   - GPT-based judgment
   - Policy-based scoring
   - Combined approach

2. **Enhanced Filtering**
   - Improved accuracy
   - Multiple content types support
   - Configurable thresholds

3. **Comprehensive Evaluation**
   - Counterfactual analysis
   - Balanced judgment
   - Results verification

## Research Ethics

This framework is developed for research purposes to:
- Improve LLM safety mechanisms
- Understand model vulnerabilities
- Contribute to responsible AI development

## Citation

If you use this framework in your research, please cite our paper (citation to be added upon acceptance).

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Contact

For questions regarding the framework or dataset access, please contact the corresponding author.