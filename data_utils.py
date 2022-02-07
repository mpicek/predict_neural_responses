#%%
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader, Subset
from rich import print
from zipfile import ZipFile
from pathlib import Path
import pytorch_lightning as pl
from neuralpredictors.measures.np_functions import (
    oracle_corr_jackknife,
    oracle_corr_conservative,
    fev,
)
from neuralpredictors.measures.modules import Corr
from tqdm import tqdm


class Antolik2016Dataset(Dataset):
    """
    Create custom map-style torch.nn.utils.data.Dataset.

    Map-style mouse V1 dataset of stimulus (images) and neural response
    (average number of spikes) as prensented in Antolik et al. 2016:
    Model Constrained by Visual Hierachy Improves Prediction of
    Neural Responses to Natural Scenes.
    Contains data for three regions in two animals (mice):\n
    training_images: n images (31x31) as npy file\n
    training_set: m neural responses as npy file\n
    validation_images: 50 images (31x31) as npy file\n
    validation_set: m neural responses as npy file
    """

    def __init__(
        self, region: str = "region1", dataset_type="training",
    ):
        """Instantiate Dataset

        Args:
            region (str, optional): Dataset region, options are 'region1',
                'region2' and 'region3'. Defaults to "region1".
            image_preprocessing (Callable, optional): Image-preprocessing
                function. Defaults to None.
            response_preprocessing (Callable, optional): Response-preprocessing
                function. Defaults to None.
        """

        super().__init__()
        self.data_path = Path.cwd() / "data" / "external" / "antolik2016"
        self.data_path.mkdir(parents=True, exist_ok=True)
        self.URL = "https://doi.org/10.1371/journal.pcbi.1004927.s001"
        self.region = region

        # download if not yet downloaded
        if (self.data_path / "Data" / "README").exists() is False:
            self.download()
        self.data_path = self.data_path / "Data"

        # load responses
        self.responses = torch.from_numpy(
            np.load(self.data_path / self.region / f"{dataset_type}_set.npy")
        )
        self.neuron_count = self.responses.shape[1]

        # load images
        self.images = torch.from_numpy(
            np.load(self.data_path / self.region / f"{dataset_type}_inputs.npy")
        )
        self.image_count = self.images.shape[0]
        self.channels = 1
        self.height = 31
        self.width = 31

        # reshape to image representation (n, c, h, w)
        self.images = self.images.reshape(
            self.image_count, self.channels, self.height, self.width
        ).float()

        # normalize images
        mean01 = 0.0001247079
        std01 = 7.9366705904e-05
        self.images = (self.images - mean01) / std01

    def download(self):
        print(f"Downloading dataset from {self.URL}")
        r = requests.get(self.URL)
        zip_file = "data.zip"
        with open(zip_file, "wb") as f:
            f.write(r.content)
        with ZipFile(zip_file, "r") as zip_ref:
            zip_ref.extractall(self.data_path)
        zip_file = Path.cwd() / zip_file
        zip_file.unlink()
        print("Finished downloading and extracting.")

    def __len__(self):
        return self.image_count

    def __getitem__(self, idx: int):
        """
        Enable indexing of Dataset object.

        Args:
            idx (Int): index of item to get.

        Returns:
            (Tuple): tuple of image and response tensor.
        """
        image = self.images[idx, :, :, :]
        response = self.responses[idx, :]
        return image.float(), response.float()


class Antolik2016Datamodule(pl.LightningDataModule):
    def __init__(
        self, region="region1", batch_size=32, with_test_dataset=False,
    ):
        self.data_path = Path.cwd() / "data" / "external" / "antolik2016" / "Data"
        self.region = region
        self.training_parent_dataset = Antolik2016Dataset(
            region=region, dataset_type="training",
        )
        self.validation_parent_dataset = Antolik2016Dataset(
            region=region, dataset_type="validation"
        )
        self.batch_size = batch_size
        self.with_test_dataset = with_test_dataset

    def get_mean_and_std(self):
        train_img = self.training_parent_dataset.images
        val_img = self.validation_parent_dataset.images
        img = torch.vstack([train_img, val_img])
        mean = torch.mean(img)
        std = torch.std(img)
        return mean, std

    def setup(self, stage=None):
        self.train_dataset = self.training_parent_dataset
        if self.with_test_dataset == False:
            self.val_dataset = self.validation_parent_dataset
        if self.with_test_dataset == True:
            idx = np.arange(50)
            self.val_dataset = Subset(self.validation_parent_dataset, idx[:25])
            self.test_dataset = Subset(self.validation_parent_dataset, idx[25:])

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size)

    def load_STA(self, path):
        pass

    def compute_STA(self):
        train_img, train_resp = self.train_dataset[:]
        num_spikes = num_spikes = train_resp.sum(axis=0)
        filters = torch.einsum(
            "in, ixy -> nxy", train_resp, train_img.squeeze()
        ) / num_spikes.reshape(-1, 1, 1)
        return filters

    def compute_STA_ridge_regr(self, ridge_regr_coeff=0):
        """ TODO not sure it works properly

        Args:
            ridge_regr_coeff (int, optional): [description]. Defaults to 0.

        Returns:
            [type]: [description]
        """
        train_img, train_resp = self.train_dataset[:]
        train_img = train_img.reshape(train_img.shape[0], -1)
        # compute coefficient
        num_img = len(train_img)
        num_spikes = train_resp.sum(axis=0)
        coeff = 1 / num_spikes
        # compute stim_cov_matrix
        # stim_cov_matrix = np.cov(train_img)
        stim_cov_matrix = np.einsum("ip, iq-> pq", train_img, train_img) / num_img
        # add regularization
        stim_cov_matrix_reg = stim_cov_matrix + ridge_regr_coeff * np.eye(
            stim_cov_matrix.shape[0]
        )
        # compute inverse
        stim_cov_matrix_inv = np.linalg.inv(stim_cov_matrix_reg)
        # compute unregularized unwhithened and unnormalized STA
        filters = torch.einsum(
            "in, ip -> np", train_resp, train_img.squeeze()
        ) / num_spikes.unsqueeze(-1)
        # compute STA
        sta = np.einsum("n, pq, nq->np", coeff, stim_cov_matrix_inv.T, filters)
        sta = sta.reshape(
            self.training_parent_dataset.neuron_count,
            self.training_parent_dataset.height,
            self.training_parent_dataset.width,
        )
        return sta

    def Laplacian_regularized_kernels(self, alpha):
        def __laplaceBias(sizex, sizey):
            S = np.zeros((sizex * sizey, sizex * sizey))
            for x in range(0, sizex):
                for y in range(0, sizey):
                    norm = np.mat(np.zeros((sizex, sizey)))
                    norm[x, y] = 4
                    if x > 0:
                        norm[x - 1, y] = -1
                    if x < sizex - 1:
                        norm[x + 1, y] = -1
                    if y > 0:
                        norm[x, y - 1] = -1
                    if y < sizey - 1:
                        norm[x, y + 1] = -1
                    S[x * sizex + y, :] = norm.flatten()
            S = np.mat(S)
            return S

        train_img, train_resp = self.train_dataset[:]
        num_img, size_x, size_y = train_img.squeeze().shape
        train_img = train_img.reshape(train_img.shape[0], -1)
        A = np.einsum("ip, iq-> pq", train_img, train_img)
        A_reg = A + alpha * __laplaceBias(size_x, size_y)
        A_reg_inv = np.linalg.pinv(A_reg)
        B = torch.einsum("in, ip -> np", train_resp, train_img.squeeze())
        k = np.einsum("pq, nq->np", A_reg_inv, B)
        k = k.reshape(
            self.training_parent_dataset.neuron_count,
            self.training_parent_dataset.height,
            self.training_parent_dataset.width,
        )
        # find corr
        images, responses = self.val_dataset[:]
        model_prediction = torch.einsum("icxy, nxy-> in", images, torch.Tensor(k))
        c = Corr()
        corr = c(responses, model_prediction).detach().numpy()
        return k, corr.squeeze()

    def compute_best_Laplacian_regularized_kernals(
        self, alpha_min, alpha_max, nalpha=100,
    ):
        best_k = 0
        best_corr = 0
        best_avg_corr = 0
        best_alpha = 0
        alpha_min = np.log10(alpha_min)
        alpha_max = np.log10(alpha_max)
        corr_values = []
        alphas = np.logspace(alpha_min, alpha_max, num=nalpha)
        for alpha in tqdm(alphas, desc="finding best regularization parameter"):
            k, corr = self.Laplacian_regularized_kernels(alpha)
            corr_values.append(np.mean(corr))
            avg_corr = np.mean(corr)
            if avg_corr > best_avg_corr:
                best_corr = corr
                best_avg_corr = avg_corr
                best_k = k
                best_alpha = alpha
        results = {
            "k": best_k,
            "avg_corr": best_avg_corr,
            "corr": best_corr.squeeze(),
            "best_alpha": best_alpha,
            "alphas_corr": (alphas, corr_values),
        }
        return results

    def model_performaces(self, model=None):
        images, responses = self.val_dataset[:]
        multitrial_responses = np.load(
            self.data_path / self.region / "raw_validation_set.npy"
        ).swapaxes(0, 1)
        oracle_j = oracle_corr_jackknife(multitrial_responses)
        oracle_c = oracle_corr_conservative(multitrial_responses)
        model_prediction = model(images)
        repeated_model_prediction = np.repeat(
            np.array(model_prediction.unsqueeze(1).detach().numpy()),
            multitrial_responses.shape[1],
            axis=1,
        )
        corr = model.corr(responses, model_prediction).detach().numpy()
        fraction_oracle_j = corr / oracle_j
        fraction_oracle_c = corr / oracle_c
        FEV, FEVe = fev(
            multitrial_responses, repeated_model_prediction, return_exp_var=True
        )
        performances = {
            "avg_corr": np.mean(np.squeeze(corr)),
            "corr": np.squeeze(corr),
            "fraction_oracle_j": np.squeeze(fraction_oracle_j),
            "fraction_oracle_c": np.squeeze(fraction_oracle_c),
            "FEV": FEV.squeeze(),
            "FEVe": FEVe.squeeze(),
        }

        return performances

    def STA_performance(self):
        images, responses = self.val_dataset[:]

        multitrial_responses = np.load(
            self.data_path / self.region / "raw_validation_set.npy"
        ).swapaxes(0, 1)
        oracle_j = oracle_corr_jackknife(multitrial_responses)
        oracle_c = oracle_corr_conservative(multitrial_responses)
        sta = self.compute_STA()
        model_prediction = torch.einsum("icxy, nxy-> in", images, sta)

        repeated_model_prediction = np.repeat(
            np.array(model_prediction.unsqueeze(1).detach().numpy()),
            multitrial_responses.shape[1],
            axis=1,
        )
        c = Corr()
        corr = c(responses, model_prediction).detach().numpy()

        fraction_oracle_j = corr / oracle_j
        fraction_oracle_c = corr / oracle_c
        FEV, FEVe = fev(
            multitrial_responses, repeated_model_prediction, return_exp_var=True
        )
        return (
            corr.squeeze(),
            fraction_oracle_j.squeeze(),
            fraction_oracle_c.squeeze(),
            FEV.squeeze(),
            FEVe.squeeze(),
        )
