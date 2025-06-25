import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer

class GPT2Responder:
    def __init__(self, model_name='gpt2'):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.tokenizer = GPT2Tokenizer.from_pretrained(model_name)
        self.model = GPT2LMHeadModel.from_pretrained(model_name).to(self.device)

    def generate_response(self, prompt: str, max_length=50) -> str:
        inputs = self.tokenizer.encode(prompt, return_tensors='pt').to(self.device)
        outputs = self.model.generate(inputs, max_length=max_length, pad_token_id=self.tokenizer.eos_token_id)
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)
