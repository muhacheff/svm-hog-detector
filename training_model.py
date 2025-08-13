import numpy as np
import cv2
from skimage.feature import hog
from skimage import io, color
import albumentations as A
import os
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import pickle


def augmentation(raw_image):
    transforms = [
        A.HorizontalFlip(p=1),
    ]
    flipped = transforms[0](image=raw_image)["image"]
    augmented = [trans(image=img)["image"] for img in [raw_image, flipped] for trans in transforms]
    fixed_size_transform = A.Compose([
        A.PadIfNeeded(
            min_height=64,
            min_width=200,
            border_mode=cv2.BORDER_REFLECT,
            position='top_left'
        ),
        A.Resize(height=64, width=200, interpolation=cv2.INTER_AREA)
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



def fit_model():
    base_dir = "dataset"
    positive_dir = os.path.join(base_dir, "positive")
    negative_dir = os.path.join(base_dir, "negative")
    positive_label = 1
    negative_label = 0

    X_positive, y_positive = create_hog_features(positive_dir, positive_label)
    X_negative, y_negative = create_hog_features(negative_dir, negative_label)

    X = np.vstack((X_positive, X_negative))
    Y = np.hstack((y_positive, y_negative))

    svm = SVC(kernel='linear', C=5.0, probability=True)
    svm.fit(X, Y)

    #X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.3)
    #y_pred = svm.predict(X_test)
    #print(classification_report(y_test, y_pred))

    with open('model6.pkl', 'wb') as f:
        pickle.dump(svm, f)
    print('DONE!')


if __name__ == '__main__':
    fit_model()

