import PIL.Image as Image
import os
from torchvision import transforms as transforms

dir_path = 'data/original/'
img_list = os.listdir(dir_path)

save_dir_path = "data/stain_augmentation"
if not os.path.exists(save_dir_path):
    os.mkdir(save_dir_path)

# ColorJitter randomly changes the brightness, contrast, saturation, and hue of an image.
for img_name in img_list:
    full_img_path = os.path.join(dir_path, img_name)
    
    # Open image and convert to RGB
    image = Image.open(full_img_path).convert("RGB")
    
    # Apply ColorJitter transformation
    jittered_image = transforms.ColorJitter(
        brightness=0.35, contrast=0.5, saturation=0.5, hue=0.5
    )(image)
    
    # Save transformed image
    save_img_path = os.path.join(save_dir_path, img_name)
    jittered_image.save(save_img_path)
    print(f"Saved {save_img_path}")
