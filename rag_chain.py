from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
import torch
import numpy as np

model_name = "google/flan-t5-base"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
device = -1

pipe = pipeline(
    "text2text-generation",
    model=model,
    tokenizer=tokenizer,
    device=device,
)

def retrieve_chunks(query, embed_model, chunks, index, k=3):
    query_vec = embed_model.encode([query], convert_to_numpy=True)
    query_vec = np.atleast_2d(query_vec)
    assert query_vec.shape[1] == index.d, "embedd dimension mistmatch"
    _, I = index.search(query_vec, k)
    return [chunks[i] for i in I[0]]

def build_prompt(context, query):
    return f"answer this question using the context:\n\n{context}\n\nquestion: {query}"

def generate_answer(prompt):
    output = pipe(prompt, max_new_tokens=128, do_sample=False)[0]['generated_text']
    return output.strip()
