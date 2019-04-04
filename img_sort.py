from PIL import Image
import glob
from math import ceil
import os

INPUT_FOLDER = 'D:/Projects/Python/tensorflow_object_detection/extra/'
OUTPUT_FOLDER = 'D:/Projects/Python/tensorflow_object_detection/images/'

img_num = 0
max_train_img_num = ceil(len(os.listdir(INPUT_FOLDER)) * .7)

for file in glob.glob(INPUT_FOLDER + '/*'):
    try:
        image = Image.open(file).convert('RGB')
        if img_num < max_train_img_num:
            image.save(OUTPUT_FOLDER + 'train/' + str(img_num) + '.jpg', "JPEG")
        else:
            image.save(OUTPUT_FOLDER + 'test/' + str(img_num) + '.jpg', "JPEG")
        img_num += 1
    except (IOError, AttributeError) as e:
        print(str(e))
        print("Failed to edit: %s.jpg" % img_num)
