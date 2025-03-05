import fitz
import re

def extract_text_from_pdf(pdf_path):
    """Extracts text from a given PDF."""
    text = ""
    with fitz.open(pdf_path) as doc:
        for page in doc:
            text += page.get_text("text") + "\n"
    return text

def clean_text(text):
    """Removes unwanted characters and formats the text."""
    text = re.sub(r"\n+", " ", text)
    text = re.sub(r"[^a-zA-Z0-9\s,.]", "", text)
    return text.strip()

def save_clean_text(text, output_path):
    """Saves cleaned text to a file."""
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(text)

if __name__ == "__main__":
    raw_text = extract_text_from_pdf("data/constitution.pdf")
    cleaned_text = clean_text(raw_text)
    save_clean_text(cleaned_text, "data/cleaned_constitution.txt")
    print("Text extraction and cleaning completed.")