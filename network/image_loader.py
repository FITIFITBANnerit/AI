import requests
from PIL import Image
from io import BytesIO
from config import IMAGE_URL

def load_image_from_url(url):
    """
    Load an image from a URL and convert it to RGB format.
    
    Args:
        url (str): The URL of the image.
        
    Returns:
        PIL.Image: The loaded image in RGB format.
    """
    
    
    response = requests.get(url)
    if response.status_code == 200:
        image = Image.open(BytesIO(response.content)).convert("RGB")
        return image
    else:
        raise Exception(f"Failed to load image from URL: {response.status_code}")
        
    """return Image.open(url).convert("RGB")"""
    