from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch


class SummarizerAgent:
    def __init__(self, model_name):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

    def run(self, query, docs):
        context = "\n\n".join([doc["text"] for doc in docs])

        prompt = f"""
Summarize the following PDF context clearly.

Include:
1. Main idea
2. Important points
3. Short conclusion

Context:
{context}

Summary:
"""

        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            max_length=1024,
            truncation=True
        )

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=300,
                do_sample=False
            )

        summary = self.tokenizer.decode(
            outputs[0],
            skip_special_tokens=True
        )

        return summary