from PIL import Image
import pytesseract


def extract_text_from_image(file_path: str) -> str:
    image = Image.open(file_path)
    text = pytesseract.image_to_string(image)

    return text.strip()
