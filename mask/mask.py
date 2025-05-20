#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
mask.py -- Iterative Sensitive Information Extraction and Masking Script
------------------------------------------------
Main Functions:
1. Make multiple (n_iter) LLM calls for each QA to accumulate entities/mask_suggestions/sensitive_terms
2. Extract on each round based on previous round's mask results, achieving "layer-by-layer purification"
3. Write results to .jsonl immediately per entry to avoid total failure
4. Automatically backup old output files with timestamps
5. Process masks using encoding and provide harmless replacement versions
"""

import argparse
import datetime
import json
import os
import re
import time
import hashlib
from typing import Dict, List, Any, Tuple, Set

from tqdm import tqdm
from openai import OpenAI
from json_repair import repair_json

# ==== 1. OpenAI Client Configuration ====
client = OpenAI(
    api_key="API_KEY", 
    base_url="https://api.siliconflow.cn/v1")

MODEL_NAME = "Pro/deepseek-ai/DeepSeek-V3"


# ==== 2. Mask Encoding and Replacement Tools ====
class MaskManager:
    def __init__(self):
        self.mask_dict: Dict[str, str] = {}  # Original text -> mask code
        self.harmless_dict: Dict[str, str] = {}  # Mask code -> harmless replacement
        self.mask_count = 0
    
    def _generate_mask_code(self, text: str) -> str:
        """Generate unique mask code"""
        self.mask_count += 1
        # Use hash to generate unique code
        hash_obj = hashlib.md5(text.encode())
        return f"[MASK_{self.mask_count}_{hash_obj.hexdigest()[:6]}]"
    
    def _generate_harmless_replacement(self, text: str, mask_code: str) -> str:
        """Generate harmless replacement text"""
        # Generate corresponding harmless replacements based on different types of sensitive words
        if "funds" in text or "account" in text:
            return "compliant financial operation"
        elif "false" in text or "forge" in text:
            return "standard procedure"
        elif "cover" in text or "hide" in text:
            return "record"
        else:
            return "compliant operation"
    
    def add_mask(self, text: str) -> Tuple[str, str]:
        """Add new mask"""
        if text not in self.mask_dict:
            mask_code = self._generate_mask_code(text)
            harmless_text = self._generate_harmless_replacement(text, mask_code)
            self.mask_dict[text] = mask_code
            self.harmless_dict[mask_code] = harmless_text
        return self.mask_dict[text], self.harmless_dict[self.mask_dict[text]]

    def mask_text(self, text: str, use_harmless: bool = False) -> str:
        """Process text with masks"""
        masked = text
        # Sort by length to ensure longer phrases are replaced first
        for entity, mask_code in sorted(self.mask_dict.items(), key=lambda x: len(x[0]), reverse=True):
            if use_harmless:
                masked = masked.replace(entity, self.harmless_dict[mask_code])
            else:
                masked = masked.replace(entity, mask_code)
        return masked


# ==== 3. LLM Calls and Information Extraction ====
def extract_key_points(instruction: str, output: str) -> Dict[str, Any]:
    """
    Call LLM to extract key points/entities/mask_suggestions/sensitive_terms
    Returns None if parsing fails
    """
    prompt = f'''Please analyze the following QA pair to identify all key entities and knowledge points, and detect sensitive words. Pay special attention to identifying all words and phrases that may involve sensitive, illegal, or inappropriate behavior.
Question: {instruction}
Answer: {output}

Example output format:
{{
  "entities": ["Entity1", "Entity2"],
  "key_points": ["Knowledge Point1", "Knowledge Point2"],
  "mask_suggestions": {{
      "Entity1": "Replacement1"
  }},
  "sensitive_terms": {{
      "Action Words": ["manufacture", "forge"],
      "Object Words": ["accounts", "funds"],
      "Purpose Words": ["evade", "cover"]
  }}
}}
Only output JSON, no other explanations.
'''
    try:
        resp = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system",
                 "content": "You are a professional information extraction assistant, return valid JSON."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.1,
            max_tokens=1024
        )
        content = resp.choices[0].message.content.strip()
        # Directly parse; if failed, use json-repair
        try:
            return json.loads(content)
        except json.JSONDecodeError:
            return json.loads(repair_json(content))
    except Exception as e:
        print(f"LLM call failed: {e}")
        return None


# ==== 4. Result Saving ====
def save_single_result(result: Dict[str, Any], out_path: str) -> None:
    with open(out_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(result, ensure_ascii=False) + "\n")


# ==== 5. Dictionary Merge Helper ====
def merge_dicts(old: Dict[str, str], new: Dict[str, str]) -> Dict[str, str]:
    merged = old.copy()
    merged.update(new)  # New value overwrites old value
    return merged


def merge_sensitive_terms(a: Dict[str, List[str]],
                          b: Dict[str, List[str]]) -> Dict[str, List[str]]:
    merged = {k: list(v) for k, v in a.items()}
    for cat, terms in b.items():
        merged.setdefault(cat, [])
        merged[cat] = list(set(merged[cat] + terms))
    return merged


def get_processed_instructions(output_file: str) -> Set[str]:
    """Get the set of already processed instructions"""
    processed = set()
    if os.path.exists(output_file):
        with open(output_file, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    data = json.loads(line.strip())
                    if "original_instruction" in data:
                        processed.add(data["original_instruction"])
                    elif "instruction" in data:
                        processed.add(data["instruction"])
                except Exception as e:
                    print(f"Warning: Error reading processed data: {e}")
                    continue
    return processed


# ==== 6. Main Processing Flow ====
def process_qa_file(input_file: str, output_file: str, n_iter: int) -> None:
    # Create output directory
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    # Get already processed data
    processed_instructions = get_processed_instructions(output_file)
    print(f"Found {len(processed_instructions)} completed entries")

    with open(input_file, "r", encoding="utf-8") as f:
        dataset = json.load(f)

    total = len(dataset)
    remaining = total - len(processed_instructions)
    print(f"Total data: {total}, To process: {remaining}")

    for qa in tqdm(dataset, desc="Iterating Mask"):
        try:
            # Skip processed data
            if qa["instruction"] in processed_instructions:
                continue

            instr_cur = qa["instruction"]
            out_cur = qa["output"]
            
            mask_manager = MaskManager()
            prev_sensitive_count = -1
            iteration = 0
            
            while iteration < n_iter:
                iteration += 1
                info = extract_key_points(instr_cur, out_cur)
                if info is None:
                    print(f"  ↳ Extraction failed on round {iteration}, skipping sample...")
                    break

                # Process all words that need masking
                current_sensitive_terms: Set[str] = set()
                # Add words from mask_suggestions
                for term in info.get("mask_suggestions", {}).keys():
                    current_sensitive_terms.add(term)
                # Add words from sensitive_terms
                for terms in info.get("sensitive_terms", {}).values():
                    current_sensitive_terms.update(terms)
                
                # If no new sensitive words are found, end iteration early
                if len(current_sensitive_terms) == prev_sensitive_count:
                    print(f"  ↳ No new sensitive words found in round {iteration}, ending iteration early")
                    break
                prev_sensitive_count = len(current_sensitive_terms)

                # Generate mask and harmless replacement for each sensitive word
                for term in current_sensitive_terms:
                    mask_manager.add_mask(term)

                # Update text for the next round
                instr_cur = mask_manager.mask_text(instr_cur)
                out_cur = mask_manager.mask_text(out_cur)

                # Prevent API overload
                time.sleep(0.2)

            # Final result
            result = {
                "original_instruction": qa["instruction"],
                "original_output": qa["output"],
                "masked_instruction": instr_cur,
                "masked_output": out_cur,
                "harmless_instruction": mask_manager.mask_text(qa["instruction"], use_harmless=True),
                "harmless_output": mask_manager.mask_text(qa["output"], use_harmless=True),
                "mask_mapping": {
                    "sensitive_to_mask": mask_manager.mask_dict,
                    "mask_to_harmless": mask_manager.harmless_dict
                },
                "iteration_count": iteration,
                "harm_score": qa.get("harm_score", 3) if "harm_score" in qa else 3,
                "social_impact_score": qa.get("social_impact_score", 3) if "harm_score" in qa else 3,
            }
            save_single_result(result, output_file)
            processed_instructions.add(qa["instruction"])  # Update processed set

        except Exception as e:
            print(qa)
            print(f"Error processing sample: {e}")
            continue

    print(f"✅ All processing complete! Output → {output_file}")
    print(f"Successfully processed: {len(processed_instructions)}/{total} entries")


# ==== 7. CLI ====
def cli():
    parser = argparse.ArgumentParser(
        description="Multi-round Sensitive Information Extraction and Masking Tool")
    parser.add_argument("--input", required=True,
                        help="Input JSON file (QA list)")
    parser.add_argument("--output", required=True,
                        help="Output .jsonl file")
    parser.add_argument("--n_iter", type=int, default=3,
                        help="Number of iteration rounds for each QA (default 3)")
    args = parser.parse_args()

    process_qa_file(args.input, args.output, args.n_iter)


if __name__ == "__main__":
    # python ./mask/mask.py --input ./data/gun/gun_train.jsonl --output ./data/gun/gun_train_mask.jsonl --n_iter 3
    # python ./mask/mask.py   --input D:\Desktop\llm黑灰产\code\knowledge_assessment\dataset\counterfactual_data\data\counterfactual_demo.json  --output .\mask\drug_key_points.jsonl   --n_iter 3
    cli()
