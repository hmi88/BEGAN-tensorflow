import os
import cv2

def make_project_dir(project_dir):
    if not os.path.exists(project_dir):
        os.makedirs(project_dir)
        os.makedirs(os.path.join(project_dir, 'models'))
        os.makedirs(os.path.join(project_dir, 'result'))


def get_image(img_path):
    img = cv2.imread(img_path)/255. - 0.5
    return img


def inverse_image(img):
    img = (img + 0.5) * 255.
    img[img > 255] = 255
    img[img < 0] = 0
    return img

