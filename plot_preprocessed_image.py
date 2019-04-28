import matplotlib.pyplot as plt
import cv2
import os


def plot_images(file_path, row, col):
    fig = plt.figure(figsize=(8, 8))
    i = 1
    for image_name in os.listdir(file_path):
        if i >= row*col + 1:
            break
        if image_name.endswith('.png'):
            img = cv2.imread(os.path.join(file_path, image_name))
            fig.add_subplot(row, col, i)
            plt.imshow(img)
            i += 1
    plt.savefig('./image_after_extend.png')
    plt.show()


if __name__ == '__main__':
    plot_images('./Train_processed_extend', 4, 5)