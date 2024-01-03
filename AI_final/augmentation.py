import os
import numpy as np
from PIL import Image


import albumentations as A

def CLAHE_augmentation(image_path):

    clahe_transform = A.Compose([
        A.CLAHE(clip_limit=1.0, tile_grid_size=(8, 8), p=1.0)
    ])

    image = Image.open(image_path)
    image_np = np.array(image)
    augmented_image = clahe_transform(image=image_np)['image']
    return Image.fromarray(augmented_image)



def augment_data_in_folder(input_folder, output_folder):
    # 创建输出文件夹
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    twoclass = ['ill', 'normal']
    # ./train/ill和./train/normal 这两个文件夹
    for class_folder in twoclass:
        class_input_folder = os.path.join(input_folder, class_folder)
        class_output_folder = os.path.join(output_folder, class_folder)
        if not os.path.exists(class_output_folder):
            os.makedirs(class_output_folder)

        # 遍历每个类别文件夹中的图像文件
        for filename in os.listdir(class_input_folder):
            if filename.endswith(".png"): 
                image_path = os.path.join(class_input_folder, filename)
                save_path = os.path.join(class_output_folder, filename)
                # print(save_path)

                augmented_image_pil = CLAHE_augmentation(image_path)
                augmented_image_pil.save(save_path)



if __name__ == '__main__':
    input_folder = './train'
    output_folder = "./augmented_train"
    augment_data_in_folder(input_folder, output_folder)
    print("finish train")

    input_folder = './val'
    output_folder = "./augmented_val"
    augment_data_in_folder(input_folder, output_folder)
    print("finish val")

    input_folder = './test'
    output_folder = "./augmented_test"
    augment_data_in_folder(input_folder, output_folder)
    print("finish test")


