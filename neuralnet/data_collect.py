import os
import numpy as np
from PIL import Image

img_size = 56

def load_data(path):
    data1, lu_data1, l_data1 = dir_load_data(path+'apple\\', 1)
    data2, lu_data2, l_data2 = dir_load_data(path+'napple\\', 0)
    data1.extend(data2)
    lu_data1.extend(lu_data2)
    l_data1.extend(l_data2)

    return zip(data1, l_data1), zip(lu_data1, l_data1)


def dir_load_data(path, isApple):
    data = []
    lu_data = []
    label_data = []

    for filename in os.listdir(path):
        img = Image.open(path + filename)
        width, height = img.size
        if width > 3024 or height > 3024:
            img = img.crop((width / 2 - 1512, height / 2 - 1512, width / 2 + 1512, height / 2 + 1512))
        img = img.resize((img_size, img_size))
        img = img.rotate(270)
        pixel_data = img.load()

        temp_data = np.ndarray((img_size*img_size*3, 1), dtype=float)
        temp_lu_data = np.ndarray((img_size*img_size, 1), dtype=float)
        for x in range(img_size):
            for y in range(img_size):
                temp_data[x*img_size*3 + y*3] = pixel_data[x, y][0] / 255.0
                temp_data[x*img_size*3 + y*3 + 1] = pixel_data[x, y][1] / 255.0
                temp_data[x*img_size*3 + y*3 + 2] = pixel_data[x, y][2] / 255.0
                temp_lu_data[x*img_size + y] = \
                    (0.2126*pixel_data[x, y][0] + 0.7152*pixel_data[x, y][1] + 0.0722*pixel_data[x, y][2])/255.0
        data.append(temp_data)
        lu_data.append(temp_lu_data)
        label = np.zeros((1, 1))
        if isApple == 1:
            label[0] = 1
            label_data.append(label)
        elif isApple == 0:
            label[0] = 0
            label_data.append(label)
        elif isApple == 2:
            label_data.append(filename)
    return data, lu_data, label_data
