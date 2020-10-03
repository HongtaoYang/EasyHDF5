from typing import Any, Dict
from pathlib import Path
from PIL import Image


import h5py
import numpy as np



class HDFConnection:
    def __init__(self, filename: str, **kwargs):
        self.filename = filename
        self.connection = h5py.File(self.filename, mode="a", **kwargs)

    
    def insert_node(self, node_name: str, data: np.ndarray = None, attrs: Dict = None) -> None:
        """
        Insert a new node into the hdf5 file. The node can either be a `Group` or a `Dataset`.
        Args:
            node_name: Full path of the node.
            data: numpy array to be stored in the node.
            attrs: attributes associated with the node as key-value pairs.
        Return:
            None
        """
        if node_name in self.connection:
            raise ValueError(f"Node {node_name} already exist.")

        if data is not None:
            dst = self.connection.create_dataset(node_name, data=data, shuffle=True, fletcher32=True, compression="gzip")
            if attrs is not None:
                dst.attrs.update(attrs)
        else:
            grp = self.connection.create_group(node_name)
            if attrs is not None:
                dst.attrs.update(attrs)

    def insert_directory(self, image_root: str, insert_to: str) -> None:
        all_image_paths = Path(image_root).glob("**/*.jpg")
        for img_path in all_image_paths:
            img_arr = np.array(Image.open(img_path))

            hdf_node_name = Path(insert_to).joinpath(img_path.relative_to(image_root).with_suffix(""))
            self.insert_node(str(hdf_node_name), data=img_arr)

        
    def close(self) -> None:
        self.connection.close()
