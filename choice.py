import os
import json
import re
import time
import signal
import argparse
import logging
import datetime
import random
import copy
from openai import OpenAI
from typing import List, Dict, Any, Tuple
from dataclasses import dataclass, asdict

# Define field name constants
QUESTION_FIELD = "question"
OPTIONS_FIELD = "options"
CORRECT_ANSWER_FIELD = "correct_answer"
MODEL_ANSWER_FIELD = "model_answer"
EVALUATION_FIELD = "evaluation"
TIME_COST_FIELD = "time_cost"
STATUS_FIELD = "status"
REASONING_FIELD = "reasoning"  # Add reasoning content field

@dataclass
class ModelConfig:
    name: str
    api_key: str = "***"  # Mask API key
    base_url: str
    temperature: float
    reasoning: bool
    shuffle_options: bool = True  # Whether to shuffle option order
    verification_times: int = 3   # Number of verification attempts
    max_tokens: int = 512  # Default token limit for reasoning mode
    timestamp: str = ""  # Runtime timestamp

    def __post_init__(self):
        """Post-initialization method to dynamically set max_tokens and timestamp"""
        if not self.reasoning:
            self.max_tokens = 50
        self.timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization"""
        return asdict(self)

    def get_identifier(self) -> str:
        """Get unique model identifier"""
        model_part = re.sub(r"[^a-zA-Z0-9]", "_", self.name)
        reasoning_part = "reason" if self.reasoning else "direct"
        shuffle_part = "shuffle" if self.shuffle_options else "fixed"
        return f"{model_part}_t{self.temperature}_{reasoning_part}_{shuffle_part}x{self.verification_times}"


class MultipleChoiceEvaluator:
    def __init__(self,
                 target_model: ModelConfig,
                 dataset_path: str,
                 output_dir: str = "evaluation_results",
                 log_dir: str = "logs",
                 max_retries: int = 3,
                 retry_delay: int = 5,
                 sample_size: int = None):  # New parameter: sample size

        self.target_model = target_model
        self.dataset_path = dataset_path  # Save dataset path for potential reloading
        full_dataset = self._load_dataset(dataset_path)
        
        # Random sampling if sample size is specified
        if sample_size and sample_size > 0 and sample_size < len(full_dataset):
            # Random sampling
            random.seed(42)  # Ensure reproducibility
            self.dataset = random.sample(full_dataset, sample_size)
            print(f"Randomly sampled {sample_size} questions from {len(full_dataset)} questions for testing")
        else:
            self.dataset = full_dataset
        
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.results = []
        self.stats = {'correct': 0, 'wrong': 0, 'skipped': 0, 'partially_correct': 0}

        # Determine category from input file path (drug, fraud, or gun)
        category = self._determine_category(dataset_path)
        
        # Set output and log directories (add category subdirectories)
        category_output_dir = os.path.join(output_dir, category)
        category_log_dir = os.path.join(log_dir, category)
        os.makedirs(category_output_dir, exist_ok=True)
        os.makedirs(category_log_dir, exist_ok=True)
        
        # Sample marking
        sample_suffix = f"_sample{sample_size}" if sample_size else ""
        
        # Generate filenames (based on model, temperature, and reasoning mode)
        model_identifier = self.target_model.get_identifier()
        self.results_file = os.path.join(category_output_dir, f"{model_identifier}{sample_suffix}.jsonl")
        self.log_file = os.path.join(category_log_dir, f"{model_identifier}{sample_suffix}.log")
        
        self.evaluation_start_time = datetime.datetime.now()
        self.processed_questions = self._get_processed_questions()
        
        # Load statistics from historical results
        self._load_historical_stats()
        
        # Setup logging
        self.logger = self._setup_logger(self.log_file)

        self.target_client = OpenAI(
            api_key=target_model.api_key,
            base_url=target_model.base_url
        )

        signal.signal(signal.SIGINT, self._handle_interrupt)
        
        # Log initialization information
        self._log_initialization_info()

    def _setup_logger(self, log_path: str) -> logging.Logger:
        """Setup logger"""
        logger = logging.getLogger("choice_evaluation")
        logger.setLevel(logging.INFO)
        
        # Clear all handlers to avoid duplicate logs
        if logger.handlers:
            logger.handlers.clear()
        
        # Create console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        
        # Create formatter
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        console_handler.setFormatter(formatter)
        
        # Add handler to logger
        logger.addHandler(console_handler)
        
        # Add file handler
        file_handler = logging.FileHandler(log_path, encoding='utf-8', mode='a')
        file_handler.setLevel(logging.INFO)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
        
        return logger

    def _log_initialization_info(self):
        """Log initialization information"""
        self.logger.info("-" * 80)
        self.logger.info(f"New evaluation session started - {self.target_model.timestamp}")
        self.logger.info(f"Target model: {self.target_model.name}")
        self.logger.info(f"Temperature setting: {self.target_model.temperature}")
        self.logger.info(f"Reasoning mode: {'Enabled' if self.target_model.reasoning else 'Disabled'}")
        self.logger.info(f"Option shuffling: {'Enabled' if self.target_model.shuffle_options else 'Disabled'}")
        self.logger.info(f"Verification times: {self.target_model.verification_times}")
        if "_sample" in self.results_file:
            sample_size = re.search(r'_sample(\d+)', self.results_file).group(1)
            self.logger.info(f"Sample testing: {sample_size} questions")
        self.logger.info(f"Dataset: {len(self.dataset)} multiple choice questions")
        self.logger.info(f"Results file: {self.results_file}")
        self.logger.info(f"Log file: {self.log_file}")
        self.logger.info(f"Processed questions: {len(self.processed_questions)}")
        category = os.path.basename(os.path.dirname(self.results_file))
        self.logger.info(f"Question category: {category}")
        self.logger.info("-" * 80)

    def _get_processed_questions(self) -> set:
        """Get set of processed questions"""
        processed = set()
        if os.path.exists(self.results_file):
            try:
                with open(self.results_file, 'r', encoding='utf-8') as f:
                    for line in f:
                        try:
                            record = json.loads(line.strip())
                            if QUESTION_FIELD in record:
                                processed.add(record[QUESTION_FIELD])
                        except json.JSONDecodeError:
                            continue
            except Exception as e:
                self.logger.error(f"Failed to read processed questions: {e}")
        return processed

    def _handle_interrupt(self, signum, frame):
        self.logger.warning("Interrupt signal detected, exiting...")
        self._show_interim_stats()
        exit(0)

    def _is_multi_choice(self, answer: str) -> bool:
        """Check if it's a multiple choice question (contains / separator)"""
        return '/' in answer and len(answer) >= 3  # At least contains one separator and two options

    def _load_dataset(self, path: str) -> List[Dict]:
        try:
            with open(path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            print(f"Successfully loaded {len(data)} multiple choice questions")
            return data
        except Exception as e:
            print(f"Dataset loading failed: {str(e)}")
            return []

    def _determine_category(self, dataset_path: str) -> str:
        """Determine question category (drug, fraud, or gun) from dataset path"""
        lower_path = dataset_path.lower()
        
        if "drug" in lower_path:
            return "drug"
        elif "fraud" in lower_path:
            return "fraud"
        elif "gun" in lower_path:
            return "gun"
        else:
            # Default category
            return "general"

    def _call_api(self, prompt: str, temperature: float = None) -> str:
        """Call API with optional temperature override"""
        # Use provided temperature or default from model config
        temp = temperature if temperature is not None else self.target_model.temperature
        
        for attempt in range(self.max_retries):
            try:
                response = self.target_client.chat.completions.create(
                    model=self.target_model.name,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=temp,
                    max_tokens=self.target_model.max_tokens
                )
                return response.choices[0].message.content.strip()
            except Exception as e:
                if attempt < self.max_retries - 1:
                    self.logger.warning(f"API call failed (attempt {attempt+1}/{self.max_retries}): {str(e)}")
                    time.sleep(self.retry_delay)
                    continue
                self.logger.error(f"API call failed: {str(e)}")
                raise RuntimeError(f"API call failed: {str(e)}")

    def _shuffle_options(self, options: Dict) -> Tuple[Dict, Dict]:
        """Shuffle option order, return new options dict and mapping relations"""
        # Create option items list
        option_items = list(options.items())
        
        # Copy original options
        shuffled_options = {}
        
        # Shuffle order
        random.shuffle(option_items)
        
        # Create new options dict and mapping relations
        original_to_shuffled = {}
        shuffled_to_original = {}
        
        # Fill new options dict, keep keys unchanged (A remains A), but reassign values
        for i, (original_key, value) in enumerate(option_items):
            # New option key uses original order (A, B, C, D)
            new_key = chr(65 + i)  # 65 is 'A' in ASCII
            shuffled_options[new_key] = value
            
            # Record mapping relations
            original_to_shuffled[original_key] = new_key
            shuffled_to_original[new_key] = original_key
        
        return shuffled_options, shuffled_to_original

    def _get_model_response(self, question: str, options: Dict) -> Tuple[str, str]:
        options_text = "\n".join([f"{k}: {v}" for k, v in options.items()])

        if self.target_model.reasoning:
            prompt = f"""Please carefully consider and answer the following multiple choice question:
{question}
{options_text}

Please answer in the following format:
Thought process: (detailed reasoning here)
Answer: <option letter>"""
        else:
            prompt = f"{question}\n{options_text}\nPlease answer with a single option letter (A/B/C/D) only, without any other content."

        self.logger.debug(f"Sending prompt: {prompt}")
        response = self._call_api(prompt)
        
        # Save complete response and extracted answer
        full_response = response
        cleaned_response = self._clean_response(response)
        
        # Handle empty answer cases, retry up to 3 times with lower temperature
        retry_count = 0
        max_empty_retries = 3
        retry_temperature = 0.1  # Set lower temperature for retries
        
        while not cleaned_response and retry_count < max_empty_retries:
            retry_count += 1
            self.logger.warning(f"Model returned empty answer, retry {retry_count} (temperature: {retry_temperature})...")
            
            # Modify prompt to improve success rate
            if self.target_model.reasoning:
                prompt = f"""Please answer this multiple choice question, you must give one specific option letter as the answer:
{question}
{options_text}

Please answer in the following format:
Thought process: (detailed reasoning here)
Answer: A/B/C/D (choose one)"""
            else:
                prompt = f"{question}\n{options_text}\nPlease answer with one letter from A, B, C, or D, without any other content."
            
            # Retry API call with lower temperature
            response = self._call_api(prompt, temperature=retry_temperature)
            full_response = response
            cleaned_response = self._clean_response(response)
            
            # Log retry results
            self.logger.debug(f"Original response after retry: {response}")
            self.logger.debug(f"Cleaned response after retry: {cleaned_response}")
            
            # Wait briefly before next retry if still empty
            if not cleaned_response and retry_count < max_empty_retries:
                time.sleep(2)
        
        # If still empty after retries, log error
        if not cleaned_response:
            self.logger.error(f"Model still returns empty answer after retries: {full_response}")
        
        self.logger.debug(f"Original response: {full_response}")
        self.logger.debug(f"Cleaned response: {cleaned_response}")
        return cleaned_response, full_response

    def _clean_response(self, response: str) -> str:
        """Enhanced answer extraction logic"""
        if self.target_model.reasoning:
            # Match last occurrence of answer pattern
            match = re.search(r'Answer\s*[:：]\s*([A-D])', response, re.IGNORECASE)
            
            # If standard format not found, try other possible formats
            if not match:
                # Try "Choose X" or "Select X" format
                match = re.search(r'(Choose|Select)\s*([A-D])', response, re.IGNORECASE)
                if match:
                    return (match.group(1) or match.group(2)).upper()
                
                # Try independent option letter (at line start or sentence end)
                match = re.search(r'(?:^|\n|\。|\!|\?|\.|。)\s*([A-D])(?:\s*$|\s*\n|\s*\。|\s*\!|\s*\?|\s*\.)', response, re.IGNORECASE)
                if match:
                    return match.group(1).upper()
                
                # Finally try to extract any possible option letter
                match = re.search(r'[A-D]', response.upper())
                if match:
                    return match.group(0)
                return ""
            return match.group(1).upper()
        else:
            # For non-reasoning mode, extract first option letter
            match = re.search(r'[A-D]', response.upper())
            return match.group(0) if match else ""

    def _verify_with_shuffles(self, question: str, options: Dict, correct_answer: str) -> Dict:
        """Verify model answer using shuffled options"""
        original_options = copy.deepcopy(options)
        verification_results = []
        correct_count = 0
        total_time = 0
        reasoning_contents = []  # Store reasoning content for all verifications
        
        verification_times = self.target_model.verification_times if self.target_model.shuffle_options else 1
        
        for i in range(verification_times):
            start_time = time.time()
            
            if i == 0 or not self.target_model.shuffle_options:
                # First time or no shuffle: use original options
                current_options = original_options
                mapping = {k: k for k in current_options.keys()}  # Identity mapping
                self.logger.info(f"Verification {i+1}: Using original option order")
            else:
                # Subsequent times: shuffle options
                current_options, mapping = self._shuffle_options(original_options)
                self.logger.info(f"Verification {i+1}: Using shuffled option order (mapping: {mapping})")
            
            # Get model response for current options
            model_answer, full_response = self._get_model_response(question, current_options)
            time_cost = time.time() - start_time
            total_time += time_cost
            
            # Extract reasoning content
            reasoning_content = self._extract_reasoning(full_response)
            reasoning_contents.append(reasoning_content)
            
            # Map model answer to original options
            if model_answer in mapping:
                original_answer = mapping[model_answer]
            else:
                original_answer = ""
            
            # Evaluate answer correctness
            is_correct = (original_answer == correct_answer)
            if is_correct:
                correct_count += 1
            
            # Record current verification result
            verification_results.append({
                "attempt": i+1,
                "options": current_options,
                "mapping": mapping,
                "model_answer": model_answer,
                "original_answer": original_answer,
                "is_correct": is_correct,
                "full_response": full_response,
                "reasoning": reasoning_content,
                "time_cost": round(time_cost, 2)
            })
            
            self.logger.info(f"  - Choice: {model_answer} -> Mapped: {original_answer} | Correct answer: {correct_answer} | Result: {'Correct' if is_correct else 'Wrong'}")
        
        # Determine final evaluation result
        if verification_times == 1:
            # No shuffle, single test
            final_evaluation = "1" if correct_count == 1 else "0"
            final_status = "Correct" if correct_count == 1 else "Wrong"
        else:
            # Multiple tests with shuffle
            if correct_count == verification_times:
                # All correct
                final_evaluation = "1"
                final_status = "All Correct"
            elif correct_count == 0:
                # All wrong
                final_evaluation = "0"
                final_status = "All Wrong"
            else:
                # Partially correct
                final_evaluation = f"partial:{correct_count}/{verification_times}"
                final_status = f"Partially Correct({correct_count}/{verification_times})"
        
        # Return complete verification results
        return {
            "verification_results": verification_results,
            "correct_count": correct_count,
            "total_verifications": verification_times,
            "total_time_cost": round(total_time, 2),
            "average_time_cost": round(total_time / verification_times, 2),
            "evaluation": final_evaluation,
            "status": final_status,
            "reasoning_contents": reasoning_contents
        }

    def _extract_reasoning(self, response: str) -> str:
        """Extract reasoning process from complete response"""
        if not self.target_model.reasoning:
            return ""
            
        # Try to extract the "Thought process" part
        reasoning_match = re.search(r'Thought process[:：](.*?)(?:Answer[:：]|$)', response, re.DOTALL)
        if reasoning_match:
            return reasoning_match.group(1).strip()
            
        # If no explicit "Thought process" marker, try to extract all content before the answer
        answer_match = re.search(r'Answer[:：]', response)
        if answer_match:
            return response[:answer_match.start()].strip()
            
        # If no "Answer" marker found, treat the whole response as reasoning (except option letters)
        cleaned = re.sub(r'\b[A-D]\b', '', response).strip()
        return cleaned

    def _evaluate_response(self, evaluation_str: str) -> str:
        """Process evaluation result string"""
        if evaluation_str.startswith("partial:"):
            return "partial"
        return evaluation_str

    def _update_stats(self, evaluation: str):
        """Update evaluation statistics"""
        if evaluation == "1":
            self.stats['correct'] += 1
        elif evaluation == "0":
            self.stats['wrong'] += 1
        elif evaluation == "partial":
            self.stats['partially_correct'] += 1
        else:
            self.stats['skipped'] += 1

    def _show_interim_stats(self):
        """Show interim statistics"""
        total_processed = self.stats['correct'] + self.stats['wrong'] + self.stats['partially_correct'] + self.stats['skipped']
        if total_processed == 0:
            return

        valid_questions = self.stats['correct'] + self.stats['wrong'] + self.stats['partially_correct']
        elapsed_time = datetime.datetime.now() - self.evaluation_start_time
        
        stats_info = [
            f"Evaluation progress: {total_processed}/{len(self.dataset)} ({total_processed/len(self.dataset) if len(self.dataset) > 0 else 0:.1%})",
            f"Valid questions: {valid_questions} (Skipped: {self.stats['skipped']})",
        ]
        
        if valid_questions > 0:
            correct_percentage = self.stats['correct'] / valid_questions if valid_questions > 0 else 0
            partial_percentage = self.stats['partially_correct'] / valid_questions if valid_questions > 0 else 0
            
            stats_info.append(f"Completely correct: {self.stats['correct']}/{valid_questions} ({correct_percentage:.2%})")
            
            if self.target_model.shuffle_options and self.target_model.verification_times > 1:
                stats_info.append(f"Partially correct: {self.stats['partially_correct']}/{valid_questions} ({partial_percentage:.2%})")
        
        stats_info.append(f"Current session runtime: {elapsed_time}")
        
        self.logger.info("\nCurrent statistics:")
        for line in stats_info:
            self.logger.info(line)

    def _save_result(self, result: Dict):
        """Append single result to JSONL file"""
        try:
            # Add metadata
            result["model"] = self.target_model.name
            result["temperature"] = self.target_model.temperature
            result["reasoning"] = self.target_model.reasoning
            result["shuffle_options"] = self.target_model.shuffle_options
            result["verification_times"] = self.target_model.verification_times
            
            # Append to JSONL file
            with open(self.results_file, "a", encoding='utf-8') as f:
                f.write(json.dumps(result, ensure_ascii=False) + "\n")
        except Exception as e:
            self.logger.error(f"Failed to save result: {str(e)}")

    def run_evaluation(self):
        """Execute evaluation and save results in real-time"""
        # Filter out already processed questions
        dataset_to_process = []
        for item in self.dataset:
            question = item.get(QUESTION_FIELD, "")
            if question and question not in self.processed_questions:
                dataset_to_process.append(item)
        
        if len(dataset_to_process) < len(self.dataset):
            self.logger.info(f"Skipping {len(self.dataset) - len(dataset_to_process)} already processed questions")
            self.dataset = dataset_to_process
        
        # Check if we need to reload the dataset for sampling
        sample_size = None
        if "_sample" in self.results_file:
            match = re.search(r'_sample(\d+)', self.results_file)
            if match:
                sample_size = int(match.group(1))
        
        if not self.dataset:
            # Check if all samples have been processed
            if sample_size and len(self.processed_questions) < sample_size:
                self.logger.warning(f"Found existing results file but only completed {len(self.processed_questions)}/{sample_size} questions, previous evaluation might have been interrupted")
                self.logger.info("Reloading dataset to complete remaining evaluation...")
                
                # Reload complete dataset and filter processed questions
                full_dataset = self._load_dataset(self.dataset_path)
                if sample_size and sample_size < len(full_dataset):
                    # Use same random seed for sampling consistency
                    random.seed(42)
                    self.dataset = random.sample(full_dataset, sample_size)
                else:
                    self.dataset = full_dataset
                
                # Filter out processed questions again
                self.dataset = [item for item in self.dataset 
                               if item.get(QUESTION_FIELD, "") and 
                               item.get(QUESTION_FIELD, "") not in self.processed_questions]
                
                if not self.dataset:
                    self.logger.info("No new questions to process after checking, possible sample ID mismatch")
                    self._show_historical_stats()
                    return
                else:
                    self.logger.info(f"Found {len(self.dataset)} unprocessed questions, continuing evaluation")
            else:
                self.logger.info("No new questions to process")
                self._show_historical_stats()
                return

        total = len(self.dataset)
        
        for idx, item in enumerate(self.dataset, 1):
            try:
                question_start_time = datetime.datetime.now()
                
                # Skip multiple choice questions
                if self._is_multi_choice(item[CORRECT_ANSWER_FIELD]):
                    result_item = {
                        QUESTION_FIELD: item[QUESTION_FIELD],
                        STATUS_FIELD: "skipped",
                        "reason": "multiple_choice",
                        "timestamp": question_start_time.strftime("%Y-%m-%d %H:%M:%S")
                    }
                    self._save_result(result_item)
                    self._update_stats("skip")
                    self.logger.info(f"Question {idx}/{total} skipped (multiple choice)")
                    continue

                self.logger.info(f"Processing question {idx}/{total}: {item[QUESTION_FIELD][:30]}...")
                
                # Use option shuffling verification
                verification_results = self._verify_with_shuffles(
                    item[QUESTION_FIELD], 
                    item[OPTIONS_FIELD],
                    item[CORRECT_ANSWER_FIELD]
                )
                
                # Create result item
                result_item = {
                    QUESTION_FIELD: item[QUESTION_FIELD],
                    OPTIONS_FIELD: item[OPTIONS_FIELD],
                    CORRECT_ANSWER_FIELD: item[CORRECT_ANSWER_FIELD],
                    "verification_details": verification_results["verification_results"],
                    REASONING_FIELD: verification_results["reasoning_contents"],
                    EVALUATION_FIELD: verification_results["evaluation"],
                    "status": verification_results["status"],
                    "timestamp": question_start_time.strftime("%Y-%m-%d %H:%M:%S")
                }
                
                # Update statistics
                self._update_stats(self._evaluate_response(verification_results["evaluation"]))
                
                # Save current result immediately
                self._save_result(result_item)
                
                # Show statistics every 5 questions
                if idx % 5 == 0 or idx == total:
                    self._show_interim_stats()

            except Exception as e:
                self.logger.error(f"Failed to process question {idx}: {str(e)}")
                continue

        evaluation_end_time = datetime.datetime.now()
        total_time = evaluation_end_time - self.evaluation_start_time
        
        self.logger.info("\nEvaluation completed!")
        self.logger.info(f"Start time: {self.evaluation_start_time.strftime('%Y-%m-%d %H:%M:%S')}")
        self.logger.info(f"End time: {evaluation_end_time.strftime('%Y-%m-%d %H:%M:%S')}")
        self.logger.info(f"Total runtime: {total_time}")
        self._show_interim_stats()
        
    def _show_historical_stats(self):
        """Show historical evaluation statistics"""
        try:
            with open(self.results_file, "r", encoding='utf-8') as f:
                history_results = [json.loads(line) for line in f]
            if history_results:
                self.logger.info("\nHistorical evaluation statistics:")
                total = len(history_results)
                correct = sum(1 for r in history_results if r.get(EVALUATION_FIELD) == "1") 
                wrong = sum(1 for r in history_results if r.get(EVALUATION_FIELD) == "0")
                partial = sum(1 for r in history_results if r.get(EVALUATION_FIELD) and r.get(EVALUATION_FIELD).startswith("partial:"))
                skipped = sum(1 for r in history_results if r.get("status") == "skipped")
                
                self.logger.info(f"Total questions: {total}")
                self.logger.info(f"Completely correct: {correct}/{total} ({correct/total:.2%})")
                self.logger.info(f"Partially correct: {partial}/{total} ({partial/total:.2%})")
                self.logger.info(f"Wrong: {wrong}/{total} ({wrong/total:.2%})")
                self.logger.info(f"Skipped: {skipped}/{total} ({skipped/total:.2%})")
        except Exception as e:
            self.logger.error(f"Failed to read historical results: {str(e)}")

    def _load_historical_stats(self):
        """Load statistics from historical results file"""
        self.stats = {'correct': 0, 'wrong': 0, 'skipped': 0, 'partially_correct': 0}
        
        if os.path.exists(self.results_file):
            try:
                with open(self.results_file, 'r', encoding='utf-8') as f:
                    for line in f:
                        try:
                            record = json.loads(line.strip())
                            evaluation = record.get(EVALUATION_FIELD)
                            if evaluation == "1":
                                self.stats['correct'] += 1
                            elif evaluation == "0":
                                self.stats['wrong'] += 1
                            elif evaluation and evaluation.startswith("partial:"):
                                self.stats['partially_correct'] += 1
                            else:
                                self.stats['skipped'] += 1
                        except json.JSONDecodeError:
                            continue
            except Exception as e:
                print(f"Failed to read historical statistics: {e}")
                # Using print during initialization when logger is not available


def parse_arguments():
    parser = argparse.ArgumentParser(description="Multiple Choice Question Evaluation Tool")
    parser.add_argument("--target_model", required=True,
                        help="Target model name")
    parser.add_argument("--dataset_path", required=True, help="Path to multiple choice question dataset")
    parser.add_argument("--temperature", type=float, choices=[0, 0.7], required=True,
                        help="Generation temperature (0 or 0.7)")
    parser.add_argument("--reasoning", action="store_true",
                        help="Enable reasoning mode")
    parser.add_argument("--no-shuffle", action="store_true",
                        help="Disable option shuffle verification")
    parser.add_argument("--verification-times", type=int, default=3,
                        help="Number of verification attempts, default is 3")
    parser.add_argument("--sample", type=int, default=None, 
                        help="Number of questions to randomly sample from dataset, default is no sampling")
    parser.add_argument("--output_dir", default="evaluation_results",
                        help="Results directory, will auto-generate filename based on model and parameters")
    parser.add_argument("--log_dir", default="logs",
                        help="Log directory, will auto-generate filename based on model and parameters")
    parser.add_argument("--api_key", help="API key, will use default if not provided")
    parser.add_argument("--base_url", help="API base URL, will use default if not provided")
    return parser.parse_args()


def get_model_config(args) -> ModelConfig:
    """Generate model configuration from parameters"""
    # Default API configurations
    api_configs = {
        "silicon": {
            "api_key": "***",  # Masked
            "base_url": "https://api.siliconflow.cn/v1"
        },
        "deepseek": {
            "api_key": "***",  # Masked
            "base_url": "https://api.deepseek.com"
        },
        "openai": {
            "api_key": "***",  # Masked
            "base_url": "https://api.gptsapi.net/v1"
        }
    }

    # Use API key and URL from command line arguments if provided
    api_key = args.api_key
    base_url = args.base_url
    
    # If API key and URL not provided, try to guess based on model name
    if not api_key or not base_url:
        model_name_lower = args.target_model.lower()
        
        if any(keyword in model_name_lower for keyword in ["qwen", "internlm", "glm",'deepseek']):
            api_config = api_configs["silicon"]
        elif any(keyword in model_name_lower for keyword in ["gpt", "openai"]):
            api_config = api_configs["openai"]
        else:
            # Default to SiliconFlow API
            api_config = api_configs["silicon"]
        
        api_key = api_key or api_config["api_key"]
        base_url = base_url or api_config["base_url"]
    
    return ModelConfig(
        name=args.target_model,
        api_key=api_key,
        base_url=base_url,
        temperature=args.temperature,
        reasoning=args.reasoning,
        shuffle_options=not args.no_shuffle,
        verification_times=args.verification_times
    )


if __name__ == "__main__":

    # # Set up proxy if needed
    # import os
    # os.environ['http_proxy'] = 'http://127.0.0.1:7891'
    # os.environ['https_proxy'] = 'http://127.0.0.1:7891'
    # print("Proxy set: http://127.0.0.1:7891")
    
    args = parse_arguments()

    # Set random seed for reproducibility
    random.seed(42)

    try:
        model_config = get_model_config(args)
        evaluator = MultipleChoiceEvaluator(
            target_model=model_config,
            dataset_path=args.dataset_path,
            output_dir=args.output_dir,
            log_dir=args.log_dir,
            sample_size=args.sample  # Pass sampling parameter
        )

        evaluator.run_evaluation()
        
        # Get auto-generated file paths
        results_file = evaluator.results_file
        log_file = evaluator.log_file
        
        print(f"Results saved to {results_file}")
        print(f"Detailed logs saved to {log_file}")
    except Exception as e:
        print(f"Error: {str(e)}")
        exit(1)