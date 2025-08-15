## SVM-HOG-DETECTOR
![Deal With It](https://media1.tenor.com/m/XMifq_hu7uEAAAAd/deal-with.gif?width=500&height=300)
## Main description
The project implements a face classifier based on SVM (Support Vector Machine) and HOG (Histogram of Oriented Gradients).The model is capable of detecting a wide range of objects; this project showcases its application as an image-based face detector.

##  List of used libraries

- xml.etree, os, pickle
- cv2, numpy, cv2 
- albumentations, sklearn

## Usage
```python
import svm-detector

with open('model.pkl', 'rb') as f:
    model = pickle.load(f)

image = cv2.imread('image.jpg')
detections = detect_objects(image, model)

for (x1, y1, x2, y2) in detections:
    cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)

cv2.imshow('Detection', image)
cv2.waitKey(0)
cv2.destroyAllWindows()
```
