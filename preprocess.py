import os
import cv2
import numpy as np
import shutil
import pandas as pd
import PIL.Image as Image
import random
import time


def crop_and_resize_image(csv_file_from_path, csv_file_to_path, dir_to_path, width, height):
    # if cvs_file_path does exist, throw exception
    if not os.path.exists(csv_file_from_path):
        raise ImportError

    # if dir_to_path does not exist, then create one; if exist, remove and recreate
    if not os.path.exists(dir_to_path):
        os.makedirs(dir_to_path)
    else:
        shutil.rmtree(dir_to_path)
        os.makedirs(dir_to_path)

    # read csv file
    print("Using: " + csv_file_from_path)
    annotations = pd.read_csv(csv_file_from_path, header=0)
    index = annotations.index.tolist()
    annotations = annotations.set_index([index])
    print(annotations)

    # record new file path and class save to new .csv file
    new_file_name_collection = []
    new_class_name_collection = []

    # iterate the image listed in .csv file
    for i in range(len(annotations)):
        # read image path and make new path
        image_file_path = annotations.at[i, 'Path']
        image_file_path = "./" + image_file_path
        image_new_file_path = os.path.join(dir_to_path, image_file_path.split('/')[-1])
        print("Current processing: " + image_new_file_path)

        # read ROI region
        x1 = annotations.at[i, 'Roi.X1']
        y1 = annotations.at[i, 'Roi.Y1']
        x2 = annotations.at[i, 'Roi.X2']
        y2 = annotations.at[i, 'Roi.Y2']

        # crop and resize region
        image = Image.open(image_file_path)
        image = image.crop((x1, y1, x2, y2))
        image = image.resize((width, height), resample=Image.BILINEAR)
        image.save(image_new_file_path)

        # equalize and normalize image
        image = cv2.imread(image_new_file_path)
        gray_img = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        cv2.imwrite(image_new_file_path, gray_img)
        equalize_normalize_images(image_new_file_path, width, height)

        # record class id
        classId = annotations.at[i, 'ClassId']
        new_file_name_collection.append(image_new_file_path)
        new_class_name_collection.append(classId)

    # save new image file path and new class id to new .csv file
    df = pd.DataFrame({'file': new_file_name_collection,
                       'class': new_class_name_collection})
    df.to_csv(path_or_buf=csv_file_to_path, index=False)


def equalize_normalize_images(image_path, width, height):
    image = cv2.imread(image_path, 0)

    # Gaussian filtering
    blur_img = cv2.GaussianBlur(image, (5, 5), 0)

    # equalize the histogram
    equalized_img = cv2.equalizeHist(blur_img)

    # normalize the image
    normalized_img = np.zeros((width, height))
    normalized_img = cv2.normalize(equalized_img, normalized_img, 0, 255, cv2.NORM_MINMAX)

    cv2.imwrite(image_path, normalized_img)


def crop_and_resize_image_extend(csv_file_from_path, csv_file_to_path, dir_to_path, width, height):
    # if cvs_file_path does exist, throw exception
    if not os.path.exists(csv_file_from_path):
        raise ImportError

    # if dir_to_path does not exist, then create one; if exist, remove and recreate
    if not os.path.exists(dir_to_path):
        os.makedirs(dir_to_path)
    else:
        shutil.rmtree(dir_to_path)
        os.makedirs(dir_to_path)

    # read csv file
    print("Using: " + csv_file_from_path)
    annotations = pd.read_csv(csv_file_from_path, header=0)
    index = annotations.index.tolist()
    annotations = annotations.set_index([index])
    print(annotations)

    # record new file path and class save to new .csv file
    new_file_name_collection = []
    new_class_name_collection = []

    # iterate the image listed in .csv file
    for i in range(len(annotations)):
        # read image path and make new path
        image_file_path = annotations.at[i, 'Path']
        image_sub_file_path = image_file_path.split('/')[-1]
        image_name = image_sub_file_path.split('.')[0]
        image_new_name_1 = image_name + "_01.png"
        image_new_name_2 = image_name + "_02.png"
        image_new_name_3 = image_name + "_03.png"
        image_new_file_path_1 = os.path.join(dir_to_path, image_new_name_1)
        image_new_file_path_2 = os.path.join(dir_to_path, image_new_name_2)
        image_new_file_path_3 = os.path.join(dir_to_path, image_new_name_3)

        # read ROI region
        x1 = annotations.at[i, 'Roi.X1']
        y1 = annotations.at[i, 'Roi.Y1']
        x2 = annotations.at[i, 'Roi.X2']
        y2 = annotations.at[i, 'Roi.Y2']

        # crop and resize region
        random_shift(image_file_path, image_new_file_path_1, width, height, x1, y1, x2, y2)
        random_scale(image_file_path, image_new_file_path_2, width, height, x1, y1, x2, y2)
        random_rotate(image_file_path, image_new_file_path_3, width, height, x1, y1, x2, y2)

        # equalize and normalize image
        equalize_normalize_images(image_new_file_path_1, width, height)
        equalize_normalize_images(image_new_file_path_2, width, height)
        equalize_normalize_images(image_new_file_path_3, width, height)

        # record class id
        classId = annotations.at[i, 'ClassId']
        new_file_name_collection.append(image_new_file_path_1)
        new_class_name_collection.append(classId)
        new_file_name_collection.append(image_new_file_path_2)
        new_class_name_collection.append(classId)
        new_file_name_collection.append(image_new_file_path_3)
        new_class_name_collection.append(classId)

    # save new image file path and new class id to new .csv file
    df = pd.DataFrame({'file': new_file_name_collection,
                       'class': new_class_name_collection})
    df.to_csv(path_or_buf=csv_file_to_path, index=False)


def random_rotate(image_file_path, image_new_file_path, width, height, x1, y1, x2, y2):
    # set current time to random seed
    random.seed(time.time())

    # generate random parameters
    delta_rotation = random.randint(-10, 11)

    img = cv2.imread(image_file_path)
    image = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    M = cv2.getRotationMatrix2D((width / 2, height / 2), delta_rotation, 1.0)
    rotated_img = cv2.warpAffine(image, M, (width, height))
    rotated_img_height, rotated_img_width = rotated_img.shape

    i = 0
    while not (0 <= x1 <= rotated_img_width and 0 <= x2 <= rotated_img_width and \
               0 <= y1 <= rotated_img_height and 0 <= y2 <= rotated_img_height):
        delta_rotation = random.randint(-10, 11)
        M = cv2.getRotationMatrix2D((width / 2, height / 2), delta_rotation, 1.0)
        rotated_img = cv2.warpAffine(image, M, (width, height))
        rotated_img_height, rotated_img_width = rotated_img.shape

        # avoid infinite loop
        i += 1
        if i > 100:
            cropped_img = img[y1: y2][x1: x2]
            resized_img = cv2.resize(cropped_img, (width, height))
            cv2.imwrite(image_new_file_path, resized_img)
            return

    cropped_img = rotated_img[y1: y2][x1: x2]
    resized_img = cv2.resize(cropped_img, (width, height))
    cv2.imwrite(image_new_file_path, resized_img)


def random_shift(image_file_path, image_new_file_path, width, height, x1, y1, x2, y2):
    # set current time to random seed
    random.seed(time.time())

    # generate random parameters
    delta_x = random.randint(-2, 3)
    delta_y = random.randint(-2, 3)

    img = cv2.imread(image_file_path)
    image = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    image_height, image_width = image.shape

    i = 0
    while not (0 <= x1 + delta_x <= image_width and 0 <= x2 + delta_x <= image_width and \
               0 <= y1 + delta_y <= image_height and 0 <= y2 + delta_y <= image_height):
        delta_x = random.randint(-2, 3)
        delta_y = random.randint(-2, 3)

        # avoid infinite loop
        i += 1
        if i > 100:
            cropped_img = img[y1: y2][x1: x2]
            resized_img = cv2.resize(cropped_img, (width, height))
            cv2.imwrite(image_new_file_path, resized_img)
            return

    new_x1 = x1 + delta_x
    new_y1 = y1 + delta_y
    new_x2 = x2 + delta_x
    new_y2 = y2 + delta_y

    cropped_img = image[new_y1: new_y2][new_x1: new_x2]
    resized_img = cv2.resize(cropped_img, (width, height))
    cv2.imwrite(image_new_file_path, resized_img)


def random_scale(image_file_path, image_new_file_path, width, height, x1, y1, x2, y2):
    # set current time to random seed
    random.seed(time.time())

    # generate random parameters
    delta_scale = random.uniform(0.9, 1.1)

    img = cv2.imread(image_file_path)
    image = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    scaled_img = cv2.resize(image, (0, 0), fx=delta_scale, fy=delta_scale)
    scaled_img_height, scaled_img_width = scaled_img.shape

    i = 0
    while not (0 <= x1 <= scaled_img_width and 0 <= x2 <= scaled_img_width and \
               0 <= y1 <= scaled_img_height and 0 <= y2 <= scaled_img_height):
        delta_scale = random.uniform(0.9, 1.1)
        scaled_img = cv2.resize(image, (0, 0), fx=delta_scale, fy=delta_scale)
        scaled_img_height, scaled_img_width = scaled_img.shape

        # avoid infinite loop
        i += 1
        if i > 100:
            cropped_img = img[y1: y2][x1: x2]
            resized_img = cv2.resize(cropped_img, (width, height))
            cv2.imwrite(image_new_file_path, resized_img)
            return

    cropped_img = scaled_img[y1: y2][x1: x2]
    resized_img = cv2.resize(cropped_img, (width, height))
    cv2.imwrite(image_new_file_path, resized_img)


if __name__ == '__main__':
    # crop_and_resize_image('./Train.csv', './Train_processed.csv', './Train_processed', 48, 48)
    crop_and_resize_image_extend('./Train.csv', './Train_processed_extend.csv', './Train_processed_extend', 48, 48)