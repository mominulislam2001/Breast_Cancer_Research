import numpy as np
import cv2
import os

input_dir = "data/original/"
input_image_list = os.listdir(input_dir)

output_dir = "data/stain_normalization/"
os.makedirs(output_dir, exist_ok=True)

def get_mean_and_std(x):
    x_mean, x_std = cv2.meanStdDev(x)
    x_mean = np.hstack(np.around(x_mean, 2))
    x_std = np.hstack(np.around(x_std, 2))
    return x_mean, x_std

template_img_path = os.path.join(input_dir, '1.png')
template_img = cv2.imread(template_img_path)
template_img = cv2.cvtColor(template_img, cv2.COLOR_BGR2LAB)
template_mean, template_std = get_mean_and_std(template_img)

for img_name in input_image_list:
    print(f'Processing {img_name}...')
    input_img_path = os.path.join(input_dir, img_name)
    input_img = cv2.imread(input_img_path)
    if input_img is None:
        print(f'Failed to read {input_img_path}. Skipping...')
        continue

    input_img = cv2.cvtColor(input_img, cv2.COLOR_BGR2LAB)
    img_mean, img_std = get_mean_and_std(input_img)

    height, width, channel = input_img.shape
    for i in range(height):
        for j in range(width):
            for k in range(channel):
                x = input_img[i, j, k]
                x = ((x - img_mean[k]) * (template_std[k] / img_std[k])) + template_mean[k]
                x = round(x)
                x = max(0, min(255, x))
                input_img[i, j, k] = x

    input_img = cv2.cvtColor(input_img, cv2.COLOR_LAB2BGR)
    output_img_path = os.path.join(output_dir, f'modified_{img_name}')
    cv2.imwrite(output_img_path, input_img)
