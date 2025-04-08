import requests
from PIL import Image
from io import BytesIO
from config import IMAGE_URL
import pillow_heif

pillow_heif.register_heif_opener()

def load_image_from_url(url):
    print(f"🔍 Trying to load image from: {url}")
    response = requests.get(url)
    if response.status_code == 200:
        try: 
            image = Image.open(BytesIO(response.content)).convert("RGB")
            print(f"✅ Image loaded successfully from: {url}")
            return image
        except Exception as e:
            print(f"❌ Failed to load image from {url}: {e}")
            raise e
    else:
        print(f"❌ Failed to load image from {url}: {response.status_code}")
        raise Exception(f"Failed to load image from URL: {response.status_code}")
    