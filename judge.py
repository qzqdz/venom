import json
import re
import logging
from typing import Dict, Any
from pathlib import Path

# Constants
JUDGMENT_FIELD = "judgment"
REASONING_FIELD = "reasoning"
STATEMENT_FIELD = "statement"

class KnowledgeEvaluator:
    def __init__(self, log_path: str):
        self.logger = self._setup_logger(log_path)
        self.processed_statements = self._get_processed_statements()

    def _setup_logger(self, log_path: str) -> logging.Logger:
        """Set up the logger"""
        logger = logging.getLogger("knowledge_evaluation")
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

        # Add handlers to the logger
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
        self.logger.info("Initializing evaluation session...")
        self.logger.info(f"Number of processed statements: {len(self.processed_statements)}")
        self.logger.info("-" * 80)

    def _get_processed_statements(self) -> set:
        """Get the set of processed statements"""
        processed = set()
        results_file = Path("evaluation_results/processed_statements.jsonl")
        if results_file.exists():
            try:
                with open(results_file, 'r', encoding='utf-8') as f:
                    for line in f:
                        try:
                            record = json.loads(line.strip())
                            if STATEMENT_FIELD in record:
                                processed.add(record[STATEMENT_FIELD])
                        except json.JSONDecodeError:
                            continue
            except Exception as e:
                self.logger.warning(f"Failed to read processed statements: {e}")
        return processed

    def _clean_response(self, response: str) -> Dict[str, Any]:
        """Clean and validate the API response"""
        cleaned = {}
        try:
            data = json.loads(response)
            if JUDGMENT_FIELD in data:
                cleaned[JUDGMENT_FIELD] = data[JUDGMENT_FIELD]
            if REASONING_FIELD in data:
                cleaned[REASONING_FIELD] = data[REASONING_FIELD]
        except json.JSONDecodeError:
            pass
        return cleaned

    def _parse_json_response(self, response: str) -> Dict[str, Any]:
        """Attempt to parse JSON and validate structure"""
        try:
            data = json.loads(response)
            if JUDGMENT_FIELD in data:
                return data
        except json.JSONDecodeError:
            pass

        # Try extracting JSON blocks using regex
        json_pattern = r'\{(?:[^{}]|(?:\{[^{}]*\}))*\}'
        matches = re.findall(json_pattern, response, re.DOTALL)
        for match in matches:
            try:
                data = json.loads(match)
                if JUDGMENT_FIELD in data:
                    return data
            except json.JSONDecodeError:
                continue

        # Fallback: Manual parsing
        judgment_match = re.search(r'["\']?judgment["\']?\s*[:：]\s*["\']?(正确|错误)["\']?', response)
        if judgment_match:
            cleaned = {JUDGMENT_FIELD: judgment_match.group(1)}
            reasoning_match = re.search(r'["\']?reasoning["\']?\s*[:：]\s*["\']?(.*?)["\']?', response)
            if reasoning_match:
                cleaned[REASONING_FIELD] = reasoning_match.group(1)
            return cleaned

        return {}

    def _evaluate_statement(self, statement: str, api_key: str = "REDACTED_API_KEY", base_url: str = "REDACTED_BASE_URL") -> Dict[str, Any]:
        """Evaluate a single statement"""
        prompt = f"""Please evaluate the following statement and strictly return the result in JSON format:
{{"{REASONING_FIELD}": "Your detailed analysis (at least 50 words)", "{JUDGMENT_FIELD}": "Correct or Incorrect"}}
Do not include any other content. Analyze the following statement: {statement}"""

        response = self._call_api(prompt, api_key=api_key, base_url=base_url)
        cleaned = self._parse_json_response(response)

        # Retry if the response is invalid
        retry_count = 0
        max_invalid_retries = 3
        while not cleaned.get(JUDGMENT_FIELD) and retry_count < max_invalid_retries:
            retry_count += 1
            self.logger.warning(f"Invalid response format, retrying ({retry_count}/{max_invalid_retries})...")
            response = self._call_api(prompt, api_key=api_key, base_url=base_url)
            cleaned = self._parse_json_response(response)

        if not cleaned.get(JUDGMENT_FIELD):
            self.logger.error(f"Failed to get valid response after multiple retries: {response}")
        return cleaned

    def _call_api(self, prompt: str, api_key: str, base_url: str, temperature: float = 0.1) -> str:
        """Call OpenAI API with retry logic"""
        from openai import OpenAI
        client = OpenAI(api_key=api_key, base_url=base_url)
        try:
            response = client.chat.completions.create(
                model="Pro/Qwen/Qwen2.5-7B-Instruct",
                messages=[{"role": "user", "content": prompt}],
                temperature=temperature
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            self.logger.error(f"API call failed: {str(e)}")
            return ""

    def run_evaluation(self, statements: list):
        """Run the full evaluation process"""
        for statement in statements:
            if statement in self.processed_statements:
                self.logger.info(f"Skipping already processed statement: {statement[:50]}...")
                continue

            self.logger.info(f"Evaluating statement: {statement[:50]}...")
            result = self._evaluate_statement(statement)
            result[STATEMENT_FIELD] = statement

            # Save results immediately
            try:
                with open("evaluation_results/processed_statements.jsonl", "a", encoding='utf-8') as f:
                    f.write(json.dumps(result, ensure_ascii=False) + "\n")
            except Exception as e:
                self.logger.error(f"Failed to save result: {str(e)}")

            # Update processed statements
            self.processed_statements.add(statement)

        self.logger.info("Evaluation completed!")

def main():
    # Initialize evaluator
    evaluator = KnowledgeEvaluator(log_path="logs/evaluation.log")
    evaluator._log_initialization_info()

    # Load statements to evaluate
    statements = [
        "Statement 1: Example content...",
        "Statement 2: Another example...",
        # Add more statements here
    ]

    # Run evaluation
    evaluator.run_evaluation(statements)

if __name__ == "__main__":
    main()