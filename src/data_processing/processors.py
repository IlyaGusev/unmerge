from typing import List, Dict, Any, Optional

from datasets import load_dataset


class PythonInstructionsAlpacaProcessor(BaseDatasetProcessor):
    def __init__(self, max_samples: int = 2000):
        super().__init__("iamtarun/python_code_instructions_18k_alpaca", "python_instructions_alpaca", max_samples)

    def get_system_prompt(self) -> str:
        return "You are an expert Python programmer. Write clean, efficient Python code that follows best practices and includes proper documentation."

    def process_example(self, example: Dict[str, Any]) -> Optional[List[Dict[str, str]]]:
        try:
            instruction = example.get("instruction", "").strip()
            input_text = example.get("input", "").strip()
            output = example.get("output", "").strip()

            if instruction and output:
                user_content = instruction
                if input_text:
                    user_content += f"\n\nInput: {input_text}"

                return [
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": user_content},
                    {"role": "assistant", "content": output}
                ]
        except Exception:
            return None
        return None


class ArjunPythonQAProcessor(BaseDatasetProcessor):
    def __init__(self, max_samples: int = 2000):
        super().__init__("Arjun-G-Ravi/Python-codes", "arjun_python_qa", max_samples)

    def get_system_prompt(self) -> str:
        return "You are a Python programming expert. Answer questions about Python and provide working code examples."

    def process_example(self, example: Dict[str, Any]) -> Optional[List[Dict[str, str]]]:
        try:
            question = example.get("question", "").strip()
            code = example.get("code", "").strip()

            if question and code:
                return [
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": question},
                    {"role": "assistant", "content": code}
                ]
        except Exception:
            return None
        return None


class GSM8KProcessor(BaseDatasetProcessor):
    def __init__(self, max_samples: int = 2000):
        super().__init__("openai/gsm8k", "gsm8k_math", max_samples)
        self.config_name = "main"

    def get_system_prompt(self) -> str:
        return "You are an expert at solving grade school math problems. Show your work step by step and provide the final numerical answer."

    def load_raw_dataset(self):
        dataset = load_dataset(self.dataset_name, self.config_name, split="train")
        return dataset

    def process_example(self, example: Dict[str, Any]) -> Optional[List[Dict[str, str]]]:
        try:
            question = example.get("question", "").strip()
            answer = example.get("answer", "").strip()

            if question and answer:
                return [
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": question},
                    {"role": "assistant", "content": answer}
                ]
        except Exception:
            return None
        return None


class SQuADProcessor(BaseDatasetProcessor):
    def __init__(self, max_samples: int = 2000):
        super().__init__("squad", "squad_qa", max_samples)

    def get_system_prompt(self) -> str:
        return "You are an expert at reading comprehension. Answer questions based on the given context accurately and concisely."

    def process_example(self, example: Dict[str, Any]) -> Optional[List[Dict[str, str]]]:
        try:
            context = example.get("context", "").strip()
            question = example.get("question", "").strip()
            answers = example.get("answers", {})

            if context and question and answers and "text" in answers and len(answers["text"]) > 0:
                answer = answers["text"][0].strip()

                if answer:
                    user_content = f"Context: {context}\n\nQuestion: {question}"
                    return [
                        {"role": "system", "content": self.system_prompt},
                        {"role": "user", "content": user_content},
                        {"role": "assistant", "content": answer}
                    ]
        except Exception:
            return None
        return None


class AlpacaProcessor(BaseDatasetProcessor):
    def __init__(self, max_samples: int = 2000):
        super().__init__("tatsu-lab/alpaca", "alpaca_instructions", max_samples)

    def get_system_prompt(self) -> str:
        return "You are a helpful assistant. Follow instructions carefully and provide accurate, helpful responses."

    def process_example(self, example: Dict[str, Any]) -> Optional[List[Dict[str, str]]]:
        try:
            instruction = example.get("instruction", "").strip()
            input_text = example.get("input", "").strip()
            output = example.get("output", "").strip()

            if instruction and output:
                user_content = instruction
                if input_text:
                    user_content += f"\n\nInput: {input_text}"

                return [
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": user_content},
                    {"role": "assistant", "content": output}
                ]
        except Exception:
            return None
        return None
