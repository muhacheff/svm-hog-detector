from sklearn.svm import SVC
import albumentations as alb
import numpy as np
import pickle
import cv2
import os


def augmentation(raw_image, aug_shape):
    width_aug = aug_shape[0]
    height_aug = aug_shape[1]
    transforms = [
        alb.HorizontalFlip(p=1),
        alb.Rotate(limit=60, p=1),
        alb.Rotate(limit=15, p=1),]
    flipped = transforms[0](image=raw_image)["image"]
    augmented = [trans(image=img)["image"] for img in [raw_image, flipped] for trans in transforms]
    fixed_size_transform = alb.Compose([
        alb.PadIfNeeded(
            min_height=height_aug,
            min_width=width_aug,
            border_mode=cv2.BORDER_REFLECT,
            position='top_left'),
        alb.Resize(height=height_aug, width=width_aug, interpolation=cv2.INTER_AREA)
    ])

    return [fixed_size_transform(image=img)["image"] for img in augmented]


def create_hog_features(images_folder, class_label):
    size_window = (176, 256)
    features = []
    labels = []
    hog = cv2.HOGDescriptor(
        _winSize=size_window,
        _blockSize=(16, 16),
        _blockStride=(8, 8),
        _cellSize=(8, 8),
        _nbins=9)

    for image in os.listdir(images_folder):
        img = cv2.imread(os.path.join(images_folder, image), cv2.IMREAD_COLOR)
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        for raw_image in augmentation(img_gray, size_window):
            hogs_features = hog.compute(raw_image)
            features.append(hogs_features)
            labels.append(class_label)
    return np.array(features), np.array(labels)


def fit_model():
    base_dir = "dataset"
    positive_dir = os.path.join(base_dir, "faces_human")
    negative_dir = os.path.join(base_dir, "negative_new")
    positive_label = 1
    negative_label = 0

    x_positive, y_positive = create_hog_features(positive_dir, positive_label)
    x_negative, y_negative = create_hog_features(negative_dir, negative_label)
    x_data = np.vstack((x_positive, x_negative))
    y_data = np.hstack((y_positive, y_negative))

    svm = SVC(kernel='poly', C=0.1, gamma='scale', probability=True)
    svm.fit(x_data, y_data)

    with open('face_detector.pkl', 'wb') as f:
        pickle.dump(svm, f)


if __name__ == '__main__':
    fit_model()
