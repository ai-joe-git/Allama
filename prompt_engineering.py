from typing import List

def few_shot_prompt(task: str, examples: List[tuple], query: str) -> str:
    prompt = f"Task: {task}\n\n"
    for i, (input_text, output_text) in enumerate(examples, 1):
        prompt += f"Example {i}:\nInput: {input_text}\nOutput: {output_text}\n\n"
    prompt += f"Now, please complete the following:\nInput: {query}\nOutput:"
    return prompt

def chain_of_thought_prompt(question: str) -> str:
    return f"""Question: {question}
Let's approach this step-by-step:
1)
2)
3)
Therefore, the final answer is:"""

def self_consistency_prompt(question: str, num_solutions: int = 3) -> str:
    prompt = f"Question: {question}\n\n"
    for i in range(1, num_solutions + 1):
        prompt += f"Solution {i}:\n1)\n2)\n3)\nTherefore, the answer for Solution {i} is:\n\n"
    prompt += "Now, considering all solutions, the most consistent answer is:"
    return prompt
