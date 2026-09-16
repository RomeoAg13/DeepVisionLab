from PIL import Image
import numpy as np
import torch 

def load__rgb_image(path):
    image = Image.open(path)
    image_rgb = image.convert("RGB")
    return image_rgb


def resize(image):
    image_resize = image.resize((224,224))
    return image_resize


def normalized(image): 
    img_convert = np.array(image)
    normalized = img_convert / 255
    return normalized


def tenseur(image):
    image2 = torch.from_numpy(image)
    modif = image2.permute(2,0,1)
    return modif


def main(path):
    image = load__rgb_image(path)
    img_resized = resize(image)
    img_normalized = normalized(img_resized)
    img_tenseur = tenseur(img_normalized)
    return img_tenseur


image1 = main("./data/raw/image.png")
image2 = main("./data/raw/image2.png")
image3 = main("./data/raw/image3.png")


batch = torch.stack((image1,image2,image3))
print(batch.shape)