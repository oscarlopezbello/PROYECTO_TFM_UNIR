import re
import unicodedata

def sanitize_text(text: str) -> str:
    if not text:
        return ""

    # 1. Lowercase
    text = text.lower()

    # 2. Remove accents (ñ → n, á → a, etc.)
    text = unicodedata.normalize("NFKD", text)
    text = "".join(c for c in text if not unicodedata.combining(c))

    # 3. Remove emails
    text = re.sub(r"\b[\w\.-]+@[\w\.-]+\.\w+\b", " ", text)

    # 4. Remove phone numbers / long numbers
    text = re.sub(r"\b\d{7,}\b", " ", text)

    # 5. Replace separators with spaces
    text = re.sub(r"[,\|\;/•\-]+", " ", text)

    # 6. Remove non-text symbols (keep letters and numbers)
    text = re.sub(r"[^a-z0-9\s]", " ", text)

    # 7. Normalize whitespace
    text = re.sub(r"\s+", " ", text).strip()

    return text
