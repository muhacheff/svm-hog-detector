import os
import cv2
import xml.etree.ElementTree as pars


def extract_objects_from_img(img_folder, xml_folder, output_folder):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    for xml_file in os.listdir(xml_folder):
        if xml_file.endswith('.xml'):
            tree = pars.parse(os.path.join(xml_folder, xml_file))
            root = tree.getroot()
            img_path = os.path.join(img_folder, root.find('filename').text)
            img = cv2.imread(img_path)
            for i, obj in enumerate(root.findall('object')):
                bbox = obj.find('bndbox')
                xmin = int(bbox.find('xmin').text)
                ymin = int(bbox.find('ymin').text)
                xmax = int(bbox.find('xmax').text)
                ymax = int(bbox.find('ymax').text)
                cropped = img[ymin:ymax, xmin:xmax]
                cv2.imwrite(f"{output_folder}/{xml_file[:-4]}_{i}.jpg", cropped)


if __name__ == '__main__':
    base_dir = "dataset"
    img_folder = os.path.join(base_dir, "train_images_face")
    xml_folder = os.path.join(base_dir, "annotations")
    output_folder = os.path.join(base_dir, "positive")
    extract_objects_from_img(img_folder, xml_folder, output_folder)