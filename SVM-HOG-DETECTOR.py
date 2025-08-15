import numpy as np
import pickle
import cv2


with open('faces_detector.pkl', 'rb') as f:
    model = pickle.load(f)


def sliding_window(img_slide, step_size, window_size):
    for y in range(0, img_slide.shape[0] - window_size[1], step_size):
        for x in range(0, img_slide.shape[1] - window_size[0], step_size):
            yield x, y, img_slide[y:y + window_size[1], x:x + window_size[0]]


def image_pyramid(img_pyramid, scales_list):
    for scale in scales_list:
        resized_img = cv2.resize(img_pyramid, (int(img_pyramid.shape[1]*scale), int(img_pyramid.shape[0]*scale)))
        yield resized_img, scale


def non_max_suppression(boxes, scores, iou_threshold):
    boxes = np.array(boxes)
    scores = np.array(scores)
    sorted_indices = np.argsort(scores)[::-1]
    selected_indices = []

    while len(sorted_indices) > 0:
        current_index = sorted_indices[0]
        selected_indices.append(current_index)
        current_box = boxes[current_index]
        other_boxes = boxes[sorted_indices[1:]]

        inter_x1 = np.maximum(current_box[0], other_boxes[:, 0])
        inter_y1 = np.maximum(current_box[1], other_boxes[:, 1])
        inter_x2 = np.minimum(current_box[2], other_boxes[:, 2])
        inter_y2 = np.minimum(current_box[3], other_boxes[:, 3])

        inter_width = np.maximum(0, inter_x2 - inter_x1)
        inter_height = np.maximum(0, inter_y2 - inter_y1)
        intersection_area = inter_width * inter_height

        current_area = (current_box[2] - current_box[0]) * (current_box[3] - current_box[1])
        other_areas = (other_boxes[:, 2] - other_boxes[:, 0]) * (other_boxes[:, 3] - other_boxes[:, 1])

        union_area = current_area + other_areas - intersection_area
        iou = intersection_area / union_area
        sorted_indices = sorted_indices[1:][iou <= iou_threshold]

    return boxes[selected_indices]


def detect_objects(image_detection):
    scales_mult = [0.8]
    size_windows = (176, 256)
    slide_step = 16
    threshold = 0.1
    min_confidence = 0.99
    confidences = []
    found_boxes = []
    gray_image = cv2.cvtColor(image_detection, cv2.COLOR_BGR2GRAY)
    hog = cv2.HOGDescriptor(
        _winSize=size_windows,
        _blockSize=(16, 16),
        _blockStride=(8, 8),
        _cellSize=(8, 8),
        _nbins=9)

    for scaled_img, scale in image_pyramid(gray_image, scales_mult):
        for (x, y, window) in sliding_window(scaled_img, slide_step, size_windows):
            if window.shape[0] != size_windows[1] or window.shape[1] != size_windows[0]:
                continue

            hog_features = hog.compute(window)
            predicted_prob = model.predict_proba([hog_features])[0]
            if predicted_prob[1] > predicted_prob[0] and predicted_prob[1] > min_confidence:
                x_1, y_1 = int(x / scale), int(y / scale)
                x_2, y_2 = int((x + size_windows[0]) / scale), int((y + size_windows[1]) / scale)
                found_boxes.append([x_1, y_1, x_2, y_2])
                confidences.append(float(predicted_prob[1]))

    if len(found_boxes) > 0:
        found_boxes = non_max_suppression(np.array(found_boxes), np.array(confidences), threshold)

    return found_boxes
