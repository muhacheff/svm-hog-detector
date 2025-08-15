import numpy as np
import cv2


def sliding_window(img_slide: np.ndarray, step_size: int, window_size: tuple) -> tuple:
    """
    Generating sliding window coordinates across an image.

    Args:
        img_slide: Input image
        step_size: Number of pixels between consecutive windows
        window_size: (width, height) of the sliding window

    Yields:
        tuple: (x, y, window) where:
               - x, y: top-left coordinates of the window
               - window: image patch of size window_size
    """
    for y in range(0, img_slide.shape[0] - window_size[1], step_size):
        for x in range(0, img_slide.shape[1] - window_size[0], step_size):
            yield x, y, img_slide[y:y + window_size[1], x:x + window_size[0]]


def image_pyramid(img_pyramid: np.ndarray, scales_list: list) -> tuple:
    """
    Generate scale versions of an image.

    Args:
        img_pyramid: Input image
        scales_list: List of scale factors

    Yields:
        tuple: (resized_image, scale_factor) for each scale level
    """
    for scale in scales_list:
        resized_img = cv2.resize(img_pyramid, (int(img_pyramid.shape[1]*scale), int(img_pyramid.shape[0]*scale)))
        yield resized_img, scale


def non_max_suppression(boxes: np.ndarray, scores: np.ndarray, iou_threshold: float) -> np.ndarray:
    """
    Apply Non-Maximum Suppression (NMS) to eliminate overlapping bounding boxes.

    Args:
        boxes: Array of bounding boxes in [x1,y1,x2,y2] format
        scores: Confidence-probability scores for each box
        iou_threshold: IoU threshold for suppression

    Returns:
        numpy.ndarray: Selected boxes after NMS
    """
    boxes = np.array(boxes)
    scores = np.array(scores)

    # Sorting indices by descending score
    sorted_indices = np.argsort(scores)[::-1]
    selected_indices = []

    while len(sorted_indices) > 0:
        # Selecting the box with the highest score
        current_index = sorted_indices[0]
        selected_indices.append(current_index)
        current_box = boxes[current_index]

        # List of remaining boxes
        other_boxes = boxes[sorted_indices[1:]]

        # Calculating intersection coordinates
        inter_x1 = np.maximum(current_box[0], other_boxes[:, 0])
        inter_y1 = np.maximum(current_box[1], other_boxes[:, 1])
        inter_x2 = np.minimum(current_box[2], other_boxes[:, 2])
        inter_y2 = np.minimum(current_box[3], other_boxes[:, 3])

        # Calculating intersection area
        inter_width = np.maximum(0, inter_x2 - inter_x1)
        inter_height = np.maximum(0, inter_y2 - inter_y1)
        intersection_area = inter_width * inter_height

        # Calculating individual areas
        current_area = (current_box[2] - current_box[0]) * (current_box[3] - current_box[1])
        other_areas = (other_boxes[:, 2] - other_boxes[:, 0]) * (other_boxes[:, 3] - other_boxes[:, 1])

        # Calculating IoU
        union_area = current_area + other_areas - intersection_area
        iou = intersection_area / union_area

        # Removing boxes with IoU > threshold
        sorted_indices = sorted_indices[1:][iou <= iou_threshold]

    return boxes[selected_indices]


def detect_objects(image_detection: np.ndarray, model: 'sklearn.svm.SVC') -> list:
    """
    Detect objects in an image using HOG features and SVM classifier.

    Args:
        image_detection: Input BGR image
        model: Trained SVM classifier

    Returns:
        list: Detected bounding boxes in [x1,y1,x2,y2] format
    """
    # Algorithm parameters
    scales_mult = [0.8]   # Image pyramid scale factors
    size_windows = (160, 256)   # Detection window size (width, height)
    slide_step = 16   # Sliding window step size
    threshold = 0.1   # NMS IoU threshold
    min_confidence = 0.99   # Minimum detection confidence

    confidences = []
    found_boxes = []

    gray_image = cv2.cvtColor(image_detection, cv2.COLOR_BGR2GRAY)

    # Initializing HOG descriptor
    hog = cv2.HOGDescriptor(
        _winSize=size_windows,
        _blockSize=(16, 16),
        _blockStride=(8, 8),
        _cellSize=(8, 8),
        _nbins=9)

    # Processing each scale in image pyramid
    for scaled_img, scale in image_pyramid(gray_image, scales_mult):
        # Processing each window position
        for (x, y, window) in sliding_window(scaled_img, slide_step, size_windows):
            # Skip windows that don't match our target size
            if window.shape[0] != size_windows[1] or window.shape[1] != size_windows[0]:
                continue

            # Compute HOG features for current window
            hog_features = hog.compute(window)

            # Get classification probabilities
            predicted_prob = model.predict_proba([hog_features])[0]

            # If positive detection with sufficient confidence
            if predicted_prob[1] > predicted_prob[0] and predicted_prob[1] > min_confidence:
                # Scale coordinates back to original image
                x_1, y_1 = int(x / scale), int(y / scale)
                x_2, y_2 = int((x + size_windows[0]) / scale), int((y + size_windows[1]) / scale)

                # Store detection
                found_boxes.append([x_1, y_1, x_2, y_2])
                confidences.append(float(predicted_prob[1]))

    if len(found_boxes) > 0:
        found_boxes = non_max_suppression(np.array(found_boxes), np.array(confidences), threshold)

    return found_boxes
