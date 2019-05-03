import matplotlib.pyplot as plt
from PIL import Image
import os


def plot_images(file_path, row, col):
    fig = plt.figure(figsize=(8, 8))
    i = 1
    for image_name in os.listdir(file_path):
        if i >= row*col + 1:
            break
        if image_name.endswith('.png'):
            img = Image.open(os.path.join(file_path, image_name))
            fig.add_subplot(row, col, i)
            plt.imshow(img)
            i += 1
    plt.show()


if __name__ == '__main__':
    plot_images('./Train/0', 4, 5)
    plot_images('./Train_processed', 4, 5)
    plot_images('./Train_processed_extend', 4, 5)