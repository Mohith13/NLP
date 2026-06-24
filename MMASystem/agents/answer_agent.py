from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch


class QAAgent:
    def __init__(self, model_name="google/flan-t5-small"):
        print(f"Loading QA model: {model_name}")

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

        self.model = AutoModelForSeq2SeqLM.from_pretrained(
            model_name,
            low_cpu_mem_usage=True
        )

    def run(self, query, docs):
        # Combine retrieved documents
        context = "\n\n".join([doc["text"] for doc in docs])

        # Better structured prompt
        prompt = f"""
You are an academic assistant.

Answer the question using ONLY the context below.
If the answer is not found, reply exactly:
"I could not find this information in the uploaded PDFs."

Context:
{context}

Question:
{query}

Provide a clear and concise answer:
"""

        # Tokenize safely
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            max_length=512,   # reduced from 1024
            truncation=True
        )

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=150,
                do_sample=False,
                num_beams=4,
                early_stopping=True
            )

        answer = self.tokenizer.decode(
            outputs[0],
            skip_special_tokens=True
        )

        return answer.strip()