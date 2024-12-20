from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

# モデル名
model_name = "Salesforce/instructblip-flan-t5-xxl"

save_dir = "./flan_t5_xxl"

model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

model.save_pretrained(save_dir)
tokenizer.save_pretrained(save_dir)

print(f"Model and tokenizer have been saved to {save_dir}")

