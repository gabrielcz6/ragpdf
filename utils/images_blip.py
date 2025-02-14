from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration

class ImageCaptioner:
    def __init__(self, model_name="Salesforce/blip-image-captioning-large"):
        """Inicializa el procesador y el modelo para la generación de captions."""
        self.processor = BlipProcessor.from_pretrained(model_name)
        self.model = BlipForConditionalGeneration.from_pretrained(model_name)
    
    def generate_caption(self, image, prompt="a image of"):
        """Genera una descripción de la imagen proporcionada."""
        raw_image = image.convert("RGB")
        inputs = self.processor(raw_image, prompt, return_tensors="pt")
        output = self.model.generate(**inputs)
        
        return self.processor.decode(output[0], skip_special_tokens=True)