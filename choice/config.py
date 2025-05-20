# OpenAI API Configuration
OPENAI_API_KEY = "API_KEY"  # Replace with your actual API key
OPENAI_MODEL = "Pro/Qwen/Qwen2.5-7B-Instruct"  # or Qwen/Qwen2.5-7B-Instruct Qwen/Qwen2.5-32B-Instruct
OPENAI_TEMPERATURE = 0.1

# Question Generation Configuration
GENERATION_PROMPT = '''Please generate a true/false question based on the following QA pair:

QA Pair:
Q: {question}
A: {answer}

Requirements:
1. Maintain the specific details and parameters from the original QA pair
2. Avoid using vague expressions like "this method", "this substance"
3. Use clear substance names, avoid vague descriptions like "silvery substance"

Please respond in the following format:
Statement: [True/False statement]
Correct Answer: [True/False]
'''

# Confusability Analysis Configuration
CONFUSABILITY_PROMPT = '''Please quickly evaluate the following true/false question:

Question: {statement}
Answer: {correct_answer}

Please check:
1. Whether the context is complete (avoid using vague expressions like "this method")
2. Whether it's related to black/grey market/criminal knowledge
3. Whether the expression is clear (avoid using vague descriptions like "silvery substance")

Please respond in the following format:
Context Completeness: [Complete/Incomplete]
Relevance: [Related/Unrelated]
Expression Clarity: [Clear/Unclear]
Difficulty Level: [0-1]
'''

# Question Quality Review Configuration
REVIEW_PROMPT = '''Please quickly review the following true/false question:

Question: {statement}
Answer: {correct_answer}

Please check:
1. Whether vague expressions like "this method", "this substance" are used
2. Whether vague descriptions like "silvery substance" are used
3. Whether it's related to black/grey market/criminal knowledge

Please respond in the following format:
Evaluation Result: [Pass/Fail]
Reason: [Brief explanation]
'''