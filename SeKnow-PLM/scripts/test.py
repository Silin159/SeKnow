from model import AuGPTModel, AuGPTConfig, AuGPTTokenizer
model_name = "jkulhanek/augpt-bigdata"

config = AuGPTConfig.from_pretrained(model_name)
tokenizer = AuGPTTokenizer.from_pretrained(model_name)
model = AuGPTModel.from_pretrained(model_name, config=config)
print(tokenizer.convert_tokens_to_ids(['Document', '<|eob|>', '<|eokb|>', '<|eod|>']))
