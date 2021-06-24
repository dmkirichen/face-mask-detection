import os
import os.path
import random
from natsort import natsorted
from xml.etree.ElementTree import parse

import torch
import torchvision
from torch.utils.data import Dataset

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import matplotlib.patches as patches


def get_labels(xml_file_name):
    """Extract bounding boxes information from annotation xml file of the photo.
    Check exploratory_data_analysis.py for more info on annotation.
    
    Keyword arguments:
    xml_file_name -- path to the xml file
    """    
    doc = parse(xml_file_name)
    
    # Iterate through all objects
    labels = []
    for obj in doc.findall("object"):
        name_arg = obj.findtext("name")
        
        # Get the coordinates of the box
        bndbox_obj = obj.find("bndbox")
        xmin_arg = int(bndbox_obj.findtext("xmin"))
        xmax_arg = int(bndbox_obj.findtext("xmax"))
        ymin_arg = int(bndbox_obj.findtext("ymin"))
        ymax_arg = int(bndbox_obj.findtext("ymax"))
        
        labels.append((name_arg, (xmin_arg, ymin_arg, xmax_arg, ymax_arg)))
    return labels


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
        labels = get_labels(annot_path)
        for box in labels:  # add boxes for faces and masks
            label = box[0]
            xmin, ymin, xmax, ymax = box[1]
            rect = patches.Rectangle((xmin, ymin), xmax - xmin, ymax - ymin, linewidth=2,
                                     edgecolor=label_to_color[label], facecolor="none")
            ax.add_patch(rect)  # add box to the image
    
    plt.show()  # show final result

    
class FaceMaskDataset(Dataset):
    """PyTorch Dataset with face-mask images and xml annotations.
    """
    
    def __init__(self, data_folder, idxs=None, transform=None):
        self.transform = transform
        
        # Get main data folders
        image_folder = os.path.join(data_folder, "images")
        annot_folder = os.path.join(data_folder, "annotations")
        
        assert (os.path.isdir(image_folder)), "there is no 'images' folder in given data folder"
        assert (os.path.isdir(annot_folder)), "there is no 'annotations' folder in given data folder"
      
        # Get all paths to files from data folders
        image_paths = natsorted([os.path.join(image_folder, image_name) 
                                 for image_name in os.listdir(image_folder)])
        annot_paths = natsorted([os.path.join(annot_folder, annot_name)
                                 for annot_name in os.listdir(annot_folder)])
        
        assert (len(image_paths) == len(annot_paths)), "num of images != num of annotations"
        
        if idxs:  # if we want to get only specific indices from all files
            
            assert (hasattr(idxs, '__iter__')), "idxs is not iterable"
            assert (max(idxs) < len(image_paths)), "index value out of range (more than len)"
            assert (min(idxs) >= 0), "index value out of range (less than 0)"
            
            self.image_paths_ = [image_paths[idx] for idx in idxs]
            self.annot_paths_ = [annot_paths[idx] for idx in idxs]
        
        else:  # if we want to get all the files
            self.image_paths_ = image_paths
            self.annot_paths_ = annot_paths
            
        self.features = []
        self.labels = []
        
        for image_path, annot_path in zip(self.image_paths_, self.annot_paths_):
            self.features.append(torchvision.io.read_image(image_path))
            self.labels.append(get_labels(annot_path))
        
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        return (self.features[idx], self.labels[idx])
