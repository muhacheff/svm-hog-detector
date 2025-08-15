from sklearn.svm import SVC
import albumentations as alb
import numpy as np
import pickle
import cv2
import os


def augmentation(raw_image: np.ndarray, aug_shape: tuple) -> list:
    """
    Applying image augmentations and resize to target shape.

    Args:
        raw_image: Input image to augment
        aug_shape: Target (width, height) for output images

    Returns:
        List of augmented images with fixed size
    """
    # Extracting target dimensions
    width_aug = aug_shape[0]
    height_aug = aug_shape[1]

    # Defining augmentation transforms
    transforms = [
        alb.HorizontalFlip(p=1),
        alb.Rotate(limit=60, p=1),
        alb.Rotate(limit=15, p=1),
                 ]
    flipped = transforms[0](image=raw_image)["image"]

    # Creating augmented images: apply all transforms to original and flipped image
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


def create_hog_features(images_folder: str, class_label: int) -> tuple:
    """
    Extracting HOG features from all images in a folder.

    Args:
        images_folder: Path to folder containing images
        class_label: Label to assign to these images (0 or 1)

    Returns:
        features: Array of HOG features
        labels: Array of corresponding labels
    """
    features = []
    labels = []

    # Defining HOG parameters and initializing HOG descriptor
    size_window = (176, 256)
    hog = cv2.HOGDescriptor(
        _winSize=size_window,
        _blockSize=(16, 16),
        _blockStride=(8, 8),
        _cellSize=(8, 8),
        _nbins=9
    )

    # Processing each image in folder
    for image in os.listdir(images_folder):
        img = cv2.imread(os.path.join(images_folder, image), cv2.IMREAD_COLOR)
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        for raw_image in augmentation(img_gray, size_window):
            hogs_features = hog.compute(raw_image)
            features.append(hogs_features)
            labels.append(class_label)

    # Convert to numpy arrays before returning
    return np.array(features), np.array(labels)


def train_model() -> None:
    """
    Training SVM classifier on HOG features and save the model.
    """
    # Defining dataset paths and class labels
    base_dir = "dataset"
    positive_dir = os.path.join(base_dir, "positive_img")
    negative_dir = os.path.join(base_dir, "negative_img")
    positive_label = 1
    negative_label = 0

    # Extracting features for both classes
    x_positive, y_positive = create_hog_features(positive_dir, positive_label)
    x_negative, y_negative = create_hog_features(negative_dir, negative_label)

    # Combining positive and negative samples
    x_data = np.vstack((x_positive, x_negative))
    y_data = np.hstack((y_positive, y_negative))

    # Initializing SVM classifier and training model
    svm = SVC(kernel='poly',
              C=0.1,
              gamma='scale',
              probability=True
              )
    svm.fit(x_data, y_data)

    # Saving trained model to file
    with open('model.pkl', 'wb') as f:
        pickle.dump(svm, f)


if __name__ == '__main__':
    train_model()

