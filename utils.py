import os
import os.path
from natsort import natsorted
from xml.etree.ElementTree import parse

import torch
from torch.utils.data import Dataset

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import matplotlib.patches as patches
%matplotlib inline


def process_xml_annotation(xml_file_name):
    """Extract information from annotation xml file of the photo.
    We don't use {folder, segmented, pose, occluded, truncated, difficult, depth} features.
    Check exploratory_data_analysis.py for more info.
    
    Keyword arguments:
    xml_file_name -- path to the xml file
    """    
    doc = parse(xml_file_name)
    
    # Get meta parameters from annotation
    filename_arg = doc.findtext("filename")
    
    # Get size dimensions of the picture
    size_obj = doc.find("size")
    width_arg = int(size_obj.findtext("width"))
    height_arg = int(size_obj.findtext("height"))
    
    # Iterate through all objects
    objects = []
    for obj in doc.findall("object"):
        name_arg = obj.findtext("name")
        
        # Get the coordinates of the box
        bndbox_obj = obj.find("bndbox")
        xmin_arg = int(bndbox_obj.findtext("xmin"))
        xmax_arg = int(bndbox_obj.findtext("xmax"))
        ymin_arg = int(bndbox_obj.findtext("ymin"))
        ymax_arg = int(bndbox_obj.findtext("ymax"))
        
        objects.append((name_arg, (xmin_arg, ymin_arg), (xmax_arg, ymax_arg)))
    return (filename_arg, (width_arg, height_arg), objects)


def show_image(image_path, annot_path, title=None, with_boxes=True):
    """Plot image that is located at image_path
    
    Keyword arguments:
    image_path -- path to the image
    """
    label_to_color = {"with_mask": "g", "mask_weared_incorrect": "y", "without_mask": "r"}
    fig, ax = plt.subplots(1)  # create figure and axes

    img = mpimg.imread(image_path)  # extract image using path
    ax.imshow(img)  # display image
    if title:
        ax.set_title(title)
    
    # Get box coordinates from xml annotation
    if with_boxes:
        _, _, boxes = process_xml_annotation(annot_path)
        for box in boxes:  # add boxes for faces and masks
            label = box[0]
            xmin, ymin = box[5]
            xmax, ymax = box[6]
            rect = patches.Rectangle((xmin, ymin), xmax - xmin, ymax - ymin, linewidth=2,
                                     edgecolor=label_to_color[label], facecolor="none")
            ax.add_patch(rect)  # add box to the image
    
    plt.show()  # show final result

    
class FaceMaskDataset(Dataset):
    """PyTorch Dataset with face-mask images and xml annotations.
    """
    
    def __init__(self, data_folder, transform=None):
        self.transform = transform
        
        image_folder = os.path.join(data_folder, "images")
        annot_folder = os.path.join(data_folder, "annotations")
        
        assert (os.path.isdir(image_folder)), "there is no 'images' folder in given data folder"
        assert (os.path.isdir(annot_folder)), "there is no 'annotations' folder in given data folder"
      
        self.image_paths = natsorted([os.path.join(image_folder, image_name) 
                                      for image_name in os.listdir(image_folder)])
        self.annot_paths = natsorted([os.path.join(annot_folder, annot_name)
                                      for annot_name in os.listdir(annot_folder)])
        
        assert (len(image_paths) == len(annot_paths)), "num of images != num of annotations"
        
    def __len__(self):
        return len(self.image_paths)
