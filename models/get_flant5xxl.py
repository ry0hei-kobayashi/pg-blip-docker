from transformers import InstructBlipForConditionalGeneration, InstructBlipProcessor
import os 


model_name = "Salesforce/instructblip-flan-t5-xxl"

save_dir = "/models/flan_t5_xxl"

#model = InstructBlipForConditionalGeneration.from_pretrained(model_name, cache_dir=save_dir)
model = InstructBlipForConditionalGeneration.from_pretrained(model_name)
model.save_pretrained(save_dir)

processor = InstructBlipProcessor.from_pretrained(model_name)
processor.save_pretrained(save_dir)

print(f"Model and tokenizer have been saved to {save_dir}")

