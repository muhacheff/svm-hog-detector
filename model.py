import numpy as np
import cv2
from skimage.feature import hog
from skimage import io, color
import albumentations as A
import os


base_dir = "dataset"
positive_dir = os.path.join(base_dir, "positive")
negative_dir = os.path.join(base_dir, "negative")
positive_label = 1
negative_label = 0


def augmentation(raw_image):
    transforms = [
        A.HorizontalFlip(p=1),
    ]
    flipped = transforms[0](image=raw_image)["image"]
    augmented = [trans(image=img)["image"] for img in [raw_image, flipped] for trans in transforms]
    fixed_size_transform = A.Compose([
        A.PadIfNeeded(
            min_height=250,
            min_width=800,
            border_mode=cv2.BORDER_REFLECT,
            position='top_left'
        ),
        A.Resize(height=250, width=800, interpolation=cv2.INTER_AREA)
    ])

    return [fixed_size_transform(image=img)["image"] for img in augmented]


def create_hog_features(images_folder, class_label):
    features = []
    labels = []
    for image in os.listdir(images_folder):
        img = io.imread(os.path.join(images_folder, image))
        img_gray = color.rgb2gray(img)
        for raw_image in augmentation(img_gray):
            hogs = hog(
                raw_image,
                orientations=9,
                pixels_per_cell=(8, 8),
                cells_per_block=(2, 2),
                visualize=False
                    )
            features.append(hogs)
            labels.append(class_label)
    return np.array(features), np.array(labels)


X_positive, Y_positive = create_hog_features(positive_dir, positive_label)
X_negative, Y_negative = create_hog_features(negative_dir, negative_label)

X = np.vstack((X_positive, X_negative))
Y = np.hstack((Y_positive, Y_negative))
print(X.shape)