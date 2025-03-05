import torch
import os
from model import SatyaGPT
from tokenizer import train_tokenizer

class TextGenerator:
    def __init__(self, model_path, tokenizer_path, vocab_size=5000, embed_size=256, num_layers=4, heads=8, forward_expansion=4, dropout=0.1, max_length=512, device="cuda" if torch.cuda.is_available() else "cpu"):
        self.device = device
        self.tokenizer = train_tokenizer(tokenizer_path)
        
        self.model = SatyaGPT(
            vocab_size=vocab_size, 
            embed_size=embed_size, 
            num_layers=num_layers, 
            heads=heads, 
            forward_expansion=forward_expansion, 
            dropout=dropout, 
            max_length=max_length
        ).to(device)

        self.model.load_state_dict(torch.load(model_path, map_location=device))
        self.model.eval()

    def generate_text(self, input_text, max_new_tokens=50):
        """Generates text based on input prompt using autoregressive generation."""
        
        # Convert input text into token IDs
        input_ids = torch.tensor(self.tokenizer.encode(input_text).ids, dtype=torch.long).unsqueeze(0).to(self.device)

        self.model.eval()
        with torch.no_grad():
            for _ in range(max_new_tokens):
                logits = self.model(input_ids)  # Forward pass
                next_token = torch.argmax(logits[:, -1, :], dim=-1).unsqueeze(0)  # Get next token
                input_ids = torch.cat([input_ids, next_token], dim=1)  # Append to input
                
                # Stop if the model generates an end token (if applicable)
                if next_token.item() == self.tokenizer.token_to_id("[EOS]"):  
                    break  

        return self.tokenizer.decode(input_ids[0].tolist(), skip_special_tokens=True)


if __name__ == "__main__":
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(BASE_DIR, "../models/trained_model_final.pth")
    tokenizer_path = os.path.join(BASE_DIR, "../data/cleaned_constitution.txt")

    generator = TextGenerator(model_path, tokenizer_path)

    while True:
        user_input = input("\nEnter your text (or type 'exit' to quit): ")
        if user_input.lower() == "exit":
            break
        response = generator.generate_text(user_input)
        print("\nGenerated Response:", response)