import torch
import time
import nltk
import bert_score
import sacrebleu
from torch.nn import functional as F
from rouge_score import rouge_scorer
from nltk.translate.meteor_score import meteor_score
from nltk.translate.bleu_score import sentence_bleu

nltk.download("wordnet")


class ModelEvaluator:
    def __init__(self, model, tokenizer, device="cuda" if torch.cuda.is_available() else "cpu"):
        self.model = model.to(device)
        self.tokenizer = tokenizer
        self.device = device

    def compute_perplexity(self, dataset):
        """Computes perplexity on a given dataset."""
        self.model.eval()
        total_loss = 0
        num_tokens = 0
        criterion = torch.nn.CrossEntropyLoss(ignore_index=self.tokenizer.pad_token_id)

        with torch.no_grad():
            for inputs, targets in dataset:
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                outputs = self.model(inputs)
                loss = criterion(outputs.view(-1, outputs.size(-1)), targets.view(-1))
                total_loss += loss.item() * targets.numel()
                num_tokens += targets.numel()

        perplexity = torch.exp(torch.tensor(total_loss / num_tokens))
        return perplexity.item()

    def compute_bleu(self, references, predictions):
        """Computes BLEU score."""
        bleu_scores = [sentence_bleu([ref.split()], pred.split()) for ref, pred in zip(references, predictions)]
        return sum(bleu_scores) / len(bleu_scores)

    def compute_rouge(self, references, predictions):
        """Computes ROUGE-1, ROUGE-2, and ROUGE-L scores."""
        scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
        scores = [scorer.score(ref, pred) for ref, pred in zip(references, predictions)]
        avg_scores = {k: sum(d[k].fmeasure for d in scores) / len(scores) for k in scores[0]}
        return avg_scores

    def compute_meteor(self, references, predictions):
        """Computes METEOR score."""
        meteor_scores = [meteor_score([ref], pred) for ref, pred in zip(references, predictions)]
        return sum(meteor_scores) / len(meteor_scores)

    def compute_bert_score(self, references, predictions):
        """Computes BERTScore (Precision, Recall, F1)."""
        P, R, F1 = bert_score.score(predictions, references, lang="en", rescale_with_baseline=True)
        return {"Precision": P.mean().item(), "Recall": R.mean().item(), "F1": F1.mean().item()}

    def measure_latency(self, input_text, num_trials=10):
        """Measures the inference latency of the model."""
        input_ids = self.tokenizer.encode(input_text, return_tensors="pt").to(self.device)
        times = []
        with torch.no_grad():
            for _ in range(num_trials):
                start = time.time()
                _ = self.model(input_ids)
                times.append(time.time() - start)
        return sum(times) / len(times)  # Average latency in seconds

    def evaluate(self, dataset, references, predictions, sample_text):
        """Runs all evaluations and prints results."""
        print("\n--- Model Evaluation ---")
        print(f"Perplexity: {self.compute_perplexity(dataset):.4f}")
        print(f"BLEU Score: {self.compute_bleu(references, predictions):.4f}")
        print(f"ROUGE Scores: {self.compute_rouge(references, predictions)}")
        print(f"METEOR Score: {self.compute_meteor(references, predictions):.4f}")
        print(f"BERTScore: {self.compute_bert_score(references, predictions)}")
        print(f"Latency: {self.measure_latency(sample_text):.4f} seconds per inference")


# Example Usage
if __name__ == "__main__":
    from model import SatyaGPT
    from tokenizer import train_tokenizer
    from torch.utils.data import DataLoader
    from train import ConstitutionDataset, DATA_PATH  # Assuming the dataset is the same

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load tokenizer and dataset
    tokenizer = train_tokenizer(DATA_PATH)
    dataset = ConstitutionDataset(DATA_PATH, tokenizer)
    dataloader = DataLoader(dataset, batch_size=16, shuffle=False)

    # Load trained model
    model = SatyaGPT(
        vocab_size=5000, embed_size=256, num_layers=4,
        heads=8, forward_expansion=4, dropout=0.1, max_length=512
    ).to(device)

    model.load_state_dict(torch.load("../models/trained_model_final.pth", map_location=device))
    model.eval()

    # Example reference and generated predictions
    references = ["The Indian constitution is the supreme law of India."]
    predictions = ["India's constitution is the highest law of the country."]
    sample_text = "The Indian constitution grants fundamental rights."

    evaluator = ModelEvaluator(model, tokenizer, device)
    evaluator.evaluate(dataloader, references, predictions, sample_text)