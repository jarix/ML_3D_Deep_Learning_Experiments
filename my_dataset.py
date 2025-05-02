
import numpy as np
import pickle

import open3d as o3d

from itertools import product

from scipy.spatial import KDTree

import torch
from torch.utils.data import Dataset

class CustomDataset(Dataset):
    def __init__(self, files, features_used, has_ground_truth=True, is_training=True, num_points=4096, label_map_file="label_mapping.pkl", compute_normals=False):
        """
        Initialize the dataset.
        Args:
            files (list): List of file paths to load data from.
            features_used (list): List of features to extract (e.g., ["xyz", "rgb", "i"]).
            has_ground_truth (bool): Whether the data has ground truth labels.
            is_training (bool): Whether the dataset is for training or testing.
            num_point (int): The number of points per tile.
            label_map_file (str): Path to save/load the label mapping file.
            compute_normals (bool): Whether to compute normals for the point cloud. Default is False.
        """

        self.data, self.labels = self.load_data_and_labels(files,features_used, has_ground_truth, num_points, compute_normals)
        
        self.inputs = [self.preprocess(cloud_data)for cloud_data in self.data]

        if has_ground_truth:
            if is_training:
                all_labels = torch.cat(self.labels)
                unique_labels = np.unique(all_labels)
                self.label_mapping = {label: i for i, label in enumerate(unique_labels)}
                with open(label_map_file,"wb")as f:
                    pickle.dump(self.label_mapping,f)

            else: # Not training, load the label mapping from file
                with open(label_map_file,"rb")as f:
                    self.label_mapping = pickle.load(f)
            self.num_classes = len(self.label_mapping.keys())
            for i in range(0,len(self.labels)):
                self.labels[i] = torch.tensor([self.label_mapping[label.item()]for label in self.labels[i]])
        
        else: # No ground truth labels
            self.labels=[]
            for i in range(0,len(self.inputs)):
                self.labels.append(torch.empty_like(self.inputs[0]))
            self.num_classes=0

    def __len__(self):
        """
        Get the number of items in the dataset.
        Returns:
            int: Number of items in the dataset.
        """
        return len(self.inputs)


    def __getitem__(self, index):
        """
        Get a single item from the dataset.
        Args:
            index (int): Index of the item to retrieve.
        Returns:
            tuple: A tuple containing the input data, labels, and index.
    """
        return self.inputs[index],self.labels[index],index
    

    def get_data(self, index):
        """
        Get the input data at the specified index.
        Args:
            index (int): Index of the item to retrieve.
        Returns:
            torch.Tensor: The input data at the specified index.
        """
        return self.data[index]


    def cloud_loader(self, pcd_name, features_used, compute_normals=False):
        """
        Load point cloud data from a file and extract features based on the provided feature list.
        Args:
            pcd_name (str): Path to the point cloud file.
            features_used (list): List of features to extract (e.g., ["xyz", "rgb", "i"]).
            compute_normals (bool): Whether to compute normals for the point cloud. Default is False.
        Returns:
            tuple: A tuple containing the features and ground truth labels.
        """

        # Load point cloud data from a file
        cloud_data = np.loadtxt(pcd_name).transpose()
        
        # Extract features based on the provided feature list
        features = []
        if "xyz" in features_used:
            n_coords = cloud_data[:3]
            features.append(n_coords)
        
        if "rgb" in features_used:
            colors = cloud_data[3:6] / 255.0
            features.append(colors)

        if "i" in features_used:
            IRQ = np.quantile(cloud_data[-2], 0.75) - np.quantile(cloud_data[-2], 0.25)
            n_intensity = ((cloud_data[-2] - np.median(cloud_data[-2])) / IRQ)
            features.append(n_intensity)

        if compute_normals:
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(cloud_data[:3].transpose())
            pcd.estimate_normals()
            normals = np.asarray(pcd.normals).transpose()
            features.append(normals)

        # Get ground truth labels  
        gt = cloud_data[-1]
        
        return np.vstack(features), gt
    



    def tile_point_cloud(self, data, labels, num_points):
        """
        Tile the point cloud data into smaller sections based on the specified number of points per tile.
        Args:
            data (numpy.ndarray): The point cloud data (shape: [3, N]).
            labels (numpy.ndarray): The ground truth labels (shape: [N, ]).
            num_points (int): The number of points per tile.
        Returns:
            tuple: A tuple containing the tiled point cloud data and their corresponding labels.
        """
        
        tree = KDTree(data[:2,:].T)
        total_points = data.shape[1]
        total_area = (data[0,:].max()-data[0,:].min())*(data[1,:].max()-data[1,:].min())
        tile_area = total_area * num_points / total_points
        tile_size = np.sqrt(tile_area)

        # x, y ranges
        x_min, y_min = data[:2,:].min(axis=1)
        x_max, y_max = data[:2,:].max(axis=1)

        # x, y coordinates ranges
        x_coords = np.arange(x_min, x_max, tile_size)
        y_coords = np.arange(y_min, y_max, tile_size)

        tiles = []
        tile_labels = []

        for x,y in product(x_coords, y_coords):
            tile_points_indices = tree.query_ball_point([x,y], tile_size)
            if tile_points_indices:
                n_indeces = len(tile_points_indices)
                if n_indeces != num_points:
                    # Randomly sample points if the number of points is not equal to num_points
                    replace = n_indeces < num_points
                    tile_points_indices = np.random.choice(tile_points_indices, num_points, replace=replace)
                # Get the points and labels for the tile
                tile_points = data[:, tile_points_indices]
                tiles.append(tile_points)
                tile_labels.append(labels[tile_points_indices])
                tile_label = labels[tile_points_indices] if labels is not None else np.zeros(num_points)
                tile_labels.append(tile_label)

        return tiles, tile_labels



    def load_data_and_labels(self, files, features_used, has_ground_truth, num_points, compute_normals=False):
        """
        Load data and labels from the specified files.
        Args:
            files (list): List of file paths to load data from.
            features_used (list): List of features to extract (e.g., ["xyz", "rgb", "i"]).
            has_ground_truth (bool): Whether the data has ground truth labels.
            num_point (int): The number of points per tile.
            compute_normals (bool): Whether to compute normals for the point cloud. Default is False.
        Returns:
            tuple: A tuple containing the preprocessed data and labels.
        """
        data_list = []
        labels_list = []

        for file in files:
            cloud_data, gt = self.cloud_loader(file, features_used, compute_normals)
            tiles, tile_labels = self.tile_point_cloud(cloud_data, gt if has_ground_truth else None, num_points)

            tiles = [torch.tensor(tile).float() for tile in tiles]
            tile_labels = [torch.tensor(label).float() for label in tile_labels]    

            data_list.extend(tiles)
            labels_list.extend(tile_labels)

        return data_list, labels_list
    

    def preprocess(self, cloud_data):
        """
        Preprocess the point cloud data by centering it around the origin.
        Args:
            cloud_data (torch.Tensor): The point cloud data (shape: [3, N]).
        Returns:
            torch.Tensor: The preprocessed point cloud data.
        """
        cloud_data = cloud_data.clone()
        min_f = torch.min(cloud_data,dim=1).values
        mean_f = torch.mean(cloud_data,dim=1)
        correction = torch.tensor([mean_f[0],mean_f[1],min_f[2]])[:,None]
        cloud_data[0:3] -= correction

        return cloud_data