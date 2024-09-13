import csv
import sys

sys.path.append("../")
import pyedflib
import utils
from data.data_utils import *
from constants import INCLUDED_CHANNELS, FREQUENCY
from utils import StandardScaler
from torch.utils.data import Dataset, DataLoader
import torch
import math
import h5py
import numpy as np
import os
import pickle
import scipy
import scipy.signal
from pathlib import Path
import config


def computeSliceMatrix(
    h5_fn, edf_fn, clip_idx, time_step_size=1, clip_len=60, is_fft=False
):
    """
    Comvert entire EEG sequence into clips of length clip_len
    Args:
        h5_fn: file name of resampled signal h5 file (full path)
        clip_idx: index of current clip/sliding window
        time_step_size: length of each time_step_size, in seconds, int
        clip_len: sliding window size or EEG clip length, in seconds, int
        is_fft: whether to perform FFT on raw EEG data
    Returns:
        slices: list of EEG clips, each having shape (clip_len*freq, num_channels, time_step_size*freq)
        seizure_labels: list of seizure labels for each clip, 1 for seizure, 0 for no seizure
    """
    with h5py.File(h5_fn, "r") as f:
        signal_array = f["resampled_signal"][()]
        resampled_freq = f["resample_freq"][()]
    assert resampled_freq == FREQUENCY

    # get seizure times
    seizure_times = getSeizureTimes(edf_fn.split(".edf")[0])

    # Iterating through signal
    physical_clip_len = int(FREQUENCY * clip_len)
    physical_time_step_size = int(FREQUENCY * time_step_size)

    start_window = clip_idx * physical_clip_len
    end_window = start_window + physical_clip_len
    # (num_channels, physical_clip_len)
    curr_slc = signal_array[:, start_window:end_window]

    start_time_step = 0
    time_steps = []
    while start_time_step <= curr_slc.shape[1] - physical_time_step_size:
        end_time_step = start_time_step + physical_time_step_size
        # (num_channels, physical_time_step_size)
        curr_time_step = curr_slc[:, start_time_step:end_time_step]
        if is_fft:
            curr_time_step, _ = computeFFT(curr_time_step, n=physical_time_step_size)

        time_steps.append(curr_time_step)
        start_time_step = end_time_step

    eeg_clip = np.stack(time_steps, axis=0)

    # determine if there's seizure in current clip
    is_seizure = 0
    for t in seizure_times:
        start_t = int(t[0] * FREQUENCY)
        end_t = int(t[1] * FREQUENCY)
        if not ((end_window < start_t) or (start_window > end_t)):
            is_seizure = 1
            break

    return eeg_clip, is_seizure


def parseTxtFiles(split_type, seizure_file, cv_seed=123, scale_ratio=1):

    np.random.seed(cv_seed)

    combined_str = []

    with open(seizure_file, "r") as file:
        reader = csv.reader(file)
        next(reader)  # 跳过第一行
        for row in reader:
            combined_str.append(row[0] + "," + row[1])

    np.random.shuffle(combined_str)

    combined_tuples = []
    for i in range(len(combined_str)):
        tup = combined_str[i].strip("\n").split(",")
        tup[1] = int(tup[1])
        combined_tuples.append(tup)

    print_str = "Number of clips in " + split_type + ": " + str(len(combined_tuples))
    print(print_str)

    return combined_tuples


class SeizureDataset(Dataset):
    def __init__(
        self,
        time_step_size=1,
        max_seq_len=12,
        standardize=False,
        scaler=None,
        split="train",
        data_augment=False,
        adj_mat_dir=None,
        graph_type=None,
        top_k=None,
        filter_type="laplacian",
        sampling_ratio=1,
        seed=123,
        use_fft=False,
        preproc_dir=None,
    ):
        if standardize and (scaler is None):
            raise ValueError("To standardize, please provide scaler.")
        if (graph_type == "individual") and (top_k is None):
            raise ValueError("Please specify top_k for individual graph.")

        self.time_step_size = time_step_size
        self.max_seq_len = max_seq_len
        self.standardize = standardize
        self.scaler = scaler
        self.split = split
        self.data_augment = data_augment
        self.adj_mat_dir = adj_mat_dir
        self.graph_type = graph_type
        self.top_k = top_k
        self.filter_type = filter_type
        self.use_fft = use_fft
        self.preproc_dir = preproc_dir

        seizure_file = os.path.join(config.csv_data_path, split + "_data.csv")

        self.file_tuples = parseTxtFiles(
            split,
            seizure_file,
            cv_seed=seed,
            scale_ratio=sampling_ratio,
        )

        self.size = len(self.file_tuples)

        # Get sensor ids
        self.sensor_ids = [x.split(" ")[-1] for x in INCLUDED_CHANNELS]

        targets = []
        for i in range(len(self.file_tuples)):
            if self.file_tuples[i][-1] == 0:
                targets.append(0)
            else:
                targets.append(1)
        self._targets = targets

    def __len__(self):
        return self.size

    def targets(self):
        return self._targets

    def _random_reflect(self, EEG_seq):
        swap_pairs = get_swap_pairs(INCLUDED_CHANNELS)
        EEG_seq_reflect = EEG_seq.copy()
        if np.random.choice([True, False]):
            for pair in swap_pairs:
                EEG_seq_reflect[:, [pair[0], pair[1]], :] = EEG_seq[
                    :, [pair[1], pair[0]], :
                ]
        else:
            swap_pairs = None
        return EEG_seq_reflect, swap_pairs

    def _random_scale(self, EEG_seq):
        scale_factor = np.random.uniform(0.8, 1.2)
        if self.use_fft:
            EEG_seq += np.log(scale_factor)
        else:
            EEG_seq *= scale_factor
        return EEG_seq

    def _get_fixed_graph(self, swap_nodes=None):
        with open(self.adj_mat_dir, "rb") as pf:
            adj_mat = pickle.load(pf)
            adj_mat = adj_mat[-1]

        adj_mat_new = adj_mat.copy()
        if swap_nodes is not None:
            for node_pair in swap_nodes:
                for i in range(adj_mat.shape[0]):
                    adj_mat_new[node_pair[0], i] = adj_mat[node_pair[1], i]
                    adj_mat_new[node_pair[1], i] = adj_mat[node_pair[0], i]
                    adj_mat_new[i, node_pair[0]] = adj_mat[i, node_pair[1]]
                    adj_mat_new[i, node_pair[1]] = adj_mat[i, node_pair[0]]
                    adj_mat_new[i, i] = 1
                adj_mat_new[node_pair[0], node_pair[1]] = adj_mat[
                    node_pair[1], node_pair[0]
                ]
                adj_mat_new[node_pair[1], node_pair[0]] = adj_mat[
                    node_pair[0], node_pair[1]
                ]

        return adj_mat_new

    def _compute_supports(self, adj_mat):
        """
        Comput supports
        """
        supports = []
        supports_mat = []
        if self.filter_type == "laplacian":  # ChebNet graph conv
            supports_mat.append(
                utils.calculate_scaled_laplacian(adj_mat, lambda_max=None)
            )
        elif self.filter_type == "random_walk":  # Forward random walk
            supports_mat.append(utils.calculate_random_walk_matrix(adj_mat).T)
        elif self.filter_type == "dual_random_walk":  # Bidirectional random walk
            supports_mat.append(utils.calculate_random_walk_matrix(adj_mat).T)
            supports_mat.append(utils.calculate_random_walk_matrix(adj_mat.T).T)
        else:
            supports_mat.append(utils.calculate_scaled_laplacian(adj_mat))
        for support in supports_mat:
            supports.append(torch.FloatTensor(support.toarray()))
        return supports

    def _compute_dynamic_graph(self, eeg_clip, threshold=0.3):
        def calculate_correlation_matrix(data):
            # # 计算Pearson相关系数矩阵
            # corr_matrix = np.corrcoef(data)
            # return corr_matrix
            # 将频道和时间轴交换位置，使得频道成为最后一个轴 (12, 128, 18)
            fft_data = np.transpose(data, (0, 2, 1))

            # 初始化一个空的皮尔森相关系数矩阵
            pearson_matrix = np.zeros((18, 18))

            # 计算每对频道之间的皮尔森相关系数
            for i in range(18):
                for j in range(18):
                    # 获取第 i 和第 j 个频道的数据
                    channel_i = fft_data[:, :, i].flatten()
                    channel_j = fft_data[:, :, j].flatten()

                    # 计算皮尔森相关系数
                    corr_coef = np.corrcoef(channel_i, channel_j)[0, 1]

                    # 将结果放入相关系数矩阵中
                    pearson_matrix[i, j] = corr_coef
            return pearson_matrix

        def retain_high_correlations(matrix, threshold):
            retained_matrix = np.zeros_like(matrix)
            for i in range(matrix.shape[0]):
                for j in range(matrix.shape[1]):
                    if i != j and abs(matrix[i, j]) > threshold:
                        retained_matrix[i, j] = matrix[i, j]
            return retained_matrix

        def map_and_invert(data, old_min, old_max):
            normalized_data = (data - old_min) / (old_max - old_min)
            inverted_data = 1 - normalized_data
            return inverted_data

        def invert_high_correlations(matrix, threshold):
            inverted_matrix = np.zeros_like(matrix)
            for i in range(matrix.shape[0]):
                for j in range(matrix.shape[1]):
                    if i != j and abs(matrix[i, j]) > threshold:
                        inverted_matrix[i, j] = map_and_invert(
                            matrix[i, j], old_min=threshold, old_max=1.0
                        )
            return inverted_matrix

        corr_matrix = calculate_correlation_matrix(eeg_clip)

        average_corr_matrix = np.abs(corr_matrix)

        retained_matrix = retain_high_correlations(average_corr_matrix, threshold=0.7)

        inverted_matrix = invert_high_correlations(retained_matrix, threshold=0.7)

        return inverted_matrix

    def __getitem__(self, idx):

        clip_name, seizure_class = self.file_tuples[idx]
        with h5py.File(
            os.path.join(self.preproc_dir, clip_name),
            "r",
        ) as hf:
            eeg_clip = hf["clip"][()]

            curr_feature = eeg_clip.copy()

        # standardize wrt train mean and std
        if self.standardize:
            curr_feature = self.scaler.transform(curr_feature)

        # padding
        curr_len = curr_feature.shape[0]
        seq_len = np.minimum(curr_len, self.max_seq_len)
        if curr_len < self.max_seq_len:
            len_pad = self.max_seq_len - curr_len
            padded_feature = (
                np.ones((len_pad, curr_feature.shape[1], curr_feature.shape[2]))
                * self.padding_val
            )
            padded_feature = np.concatenate((curr_feature, padded_feature), axis=0)
        else:
            padded_feature = curr_feature.copy()

        if np.any(np.isnan(padded_feature)):
            raise ValueError("Nan found in x!")

        # convert to tensors
        # (max_seq_len, num_nodes, input_dim)
        x = torch.FloatTensor(padded_feature)
        y = torch.LongTensor([seizure_class])
        seq_len = torch.LongTensor([seq_len])
        writeout_fn = clip_name

        global_adj_mat = self._get_fixed_graph()
        global_supports = self._compute_supports(global_adj_mat)

        dynamic_adj_mat = self._compute_dynamic_graph(eeg_clip)
        dynamic_supports = self._compute_supports(dynamic_adj_mat)

        return (x, y, seq_len, global_supports, dynamic_supports, writeout_fn)


def load_dataset_detection(
    train_batch_size,
    test_batch_size=None,
    time_step_size=1,
    max_seq_len=60,
    standardize=False,
    num_workers=8,
    augmentation=False,
    adj_mat_dir=None,
    filter_type="laplacian",
    use_fft=False,
    seed=123,
    preproc_dir=None,
):

    if standardize:
        means_dir = os.path.join(
            config.csv_data_path,
            "means_seq2seq_fft_" + str(max_seq_len) + "s_szdetect_single.pkl",
        )
        stds_dir = os.path.join(
            config.csv_data_path,
            "stds_seq2seq_fft_" + str(max_seq_len) + "s_szdetect_single.pkl",
        )
        with open(means_dir, "rb") as f:
            means = pickle.load(f)
        with open(stds_dir, "rb") as f:
            stds = pickle.load(f)

        scaler = StandardScaler(mean=means, std=stds)
    else:
        scaler = None

    dataloaders = {}
    datasets = {}
    for split in ["train", "dev", "test"]:
        if split == "train":
            data_augment = augmentation
        else:
            data_augment = False  # never do augmentation on dev/test sets

        dataset = SeizureDataset(
            time_step_size=time_step_size,
            max_seq_len=max_seq_len,
            standardize=standardize,
            scaler=scaler,
            split=split,
            data_augment=data_augment,
            adj_mat_dir=adj_mat_dir,
            filter_type=filter_type,
            sampling_ratio=1.0,
            seed=seed,
            use_fft=use_fft,
            preproc_dir=preproc_dir,
        )

        if split == "train":
            shuffle = True
            batch_size = train_batch_size
        else:
            shuffle = False
            batch_size = test_batch_size

        loader = DataLoader(
            dataset=dataset,
            shuffle=shuffle,
            batch_size=batch_size,
            num_workers=num_workers,
        )
        dataloaders[split] = loader
        datasets[split] = dataset

    return dataloaders, datasets, scaler
