import xml.etree.ElementTree as pars
import os
import cv2


def extract_objects_from_img(images_folder: str, xml_dir: str, output_dir: str) -> None:
    """
    Extracts objects from images based on bounding box annotations in XML files.

    Args:
        images_folder (str): Path to directory containing source images
        xml_dir (str): Path to directory containing XML annotation files
        output_dir (str): Path to directory where cropped objects will be saved

    Process:
        1. Creates output directory if it doesn't exist
        2. Processes each XML file in the annotations directory
        3. For each object in XML, crops the corresponding region from image
        4. Saves each cropped object as separate image
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    for xml_file in os.listdir(xml_dir):
        if xml_file.endswith('.xml'):
            tree = pars.parse(os.path.join(xml_dir, xml_file))
            root = tree.getroot()
            img_path = os.path.join(images_folder, root.find('filename').text)
            img = cv2.imread(img_path)
            for i, obj in enumerate(root.findall('object')):
                bbox = obj.find('bndbox')
                x_min = int(bbox.find('xmin').text)
                y_min = int(bbox.find('ymin').text)
                x_max = int(bbox.find('xmax').text)
                y_max = int(bbox.find('ymax').text)
                cropped = img[y_min:y_max, x_min:x_max]
                cv2.imwrite(f"{output_dir}/{xml_file[:-4]}_{i}.jpg", cropped)


if __name__ == '__main__':
    # Configuration paths
    base_dir = "dataset"   # Root dataset directory
    img_folder = os.path.join(base_dir, "train_images_face")   # Source images
    xml_folder = os.path.join(base_dir, "annotations")   # XML annotations
    output_folder = os.path.join(base_dir, "positive")   # Output for cropped objects

    # Run extraction process
    extract_objects_from_img(img_folder, xml_folder, output_folder)

