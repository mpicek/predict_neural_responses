from custom_layers import dog_layer
import numpy as np
import dnn_blocks as bl
import torch.nn as nn
import pytorch_lightning as pl
import torch
from neuralpredictors.measures.modules import Corr, PoissonLoss
from neuralpredictors.regularizers import laplace, laplace5x5, laplace7x7
import neuralpredictors.layers.cores as cores
import neuralpredictors.layers.readouts as readouts


class encoding_model(pl.LightningModule):
    """Parent class for system identification enconding models, keeps track of useful metrics"""

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.loss = PoissonLoss()
        self.corr = Corr()
        self.save_hyperparameters()

    def regularization(self):
        return 0

    def training_step(self, batch, batch_idx):
        img, resp = batch
        prediction = self.forward(img)
        loss = self.loss(prediction, resp)
        reg_term = self.regularization()
        regularized_loss = loss + reg_term
        self.log("train/unregularized_loss", loss)
        self.log("train/regularization", reg_term)
        self.log("train/regularized_loss", regularized_loss)
        return regularized_loss

    def validation_step(self, batch, batch_idx):
        img, resp = batch
        prediction = self.forward(img)
        loss = self.loss(prediction, resp)
        corr = self.corr(prediction, resp)
        self.log("val/loss", loss)
        self.log("val/corr", corr)

    def test_step(self, batch, batch_idx):
        img, resp = batch
        prediction = self.forward(img)
        loss = self.loss(prediction, resp)
        corr = self.corr(prediction, resp)
        self.log("test/loss", loss)
        self.log("test/corr", corr)

    def configure_optimizers(self):
        opt = torch.optim.Adam(self.parameters(), lr=self.config["lr"])
        return opt


class CNN_SxF(encoding_model):
    """model inspired by Cadena 2019, based on neuralpredictors conv core and factorized readout"""

    def __init__(self, config):
        super().__init__(config)
        self.core = cores.Stacked2dCore(
            input_channels=self.config["input_channels"],
            hidden_channels=self.config["core_hidden_channels"],
            input_kern=self.config["core_input_kern"],
            hidden_kern=self.config["core_hidden_kern"],
            layers=self.config["core_layers"],
            input_regularizer="LaplaceL2norm",
            gamma_input=config["core_gamma_input"],
            gamma_hidden=config["core_gamma_hidden"],
            stack=-1,
            depth_separable=False,
        )
        self.readout = readouts.FullFactorized2d(
            in_shape=(
                self.config["core_hidden_channels"],
                self.config["input_size_x"],
                self.config["input_size_y"],
            ),
            outdims=self.config["num_neurons"],
            bias=self.config["readout_bias"],
            mean_activity=self.config["mean_activity"],
            gamma_readout=self.config["readout_gamma"],
        )
        self.nonlin = bl.act_func()["softplus"]

    def forward(self, x):
        x = self.core(x)
        x = self.nonlin(self.readout(x))
        return x

    def regularization(self):
        readout_reg = self.readout.regularizer(reduction="mean")
        core_reg = self.core.regularizer()
        reg_term = readout_reg + core_reg
        self.log("reg/core reg", core_reg)
        self.log("reg/readout_reg", readout_reg)
        return reg_term


class AntolikHouska_HSM_model(encoding_model):
    """Model based on Antolik 2016"""

    def __init__(self, config):
        super().__init__(config)
        self.config = config
        self.loss = PoissonLoss()
        self.corr = Corr()

        self.num_dog = config["num_dog"]
        self.h_size_per = config["h_size_perc"]
        self.h_size = int(np.ceil(config["h_size_perc"] * config["num_neurons"]))
        self.output_size = config["num_neurons"]
        self.DOG = dog_layer(
            self.num_dog, self.config["input_size_x"], self.config["input_size_y"],
        )
        self.h = bl.FC_block(self.num_dog, self.h_size, activation="softplus")
        self.readout = bl.FC_block(self.h_size, self.output_size, activation="softplus")

    def forward(self, x):
        x = self.DOG(x)
        x = self.h(x)
        x = self.readout(x)
        return x

    def L2_on_readout(self):
        weights = self.readout[0].weight
        w2 = torch.mean(weights * weights)
        reg_term = self.config["reg_L2_on_readout"] * w2
        return reg_term

    def regularization(self):
        reg_term = 0
        if "reg_L2_on_readout" in self.config.keys():
            reg_term = self.L2_on_readout()
        return reg_term


class GLM(encoding_model):
    """
    Generalized linear model (LN model)

    To add regularization weights add entries to config dictiorary with keys matching name
    of regularization and value matching regularization coeff (i.e config['reg_L1"]=0.1)
    """

    def __init__(self, config):
        super().__init__(config)
        self.config = config
        self.input_size = config["input_size_x"] * config["input_size_y"]
        self.output_size = config["num_neurons"]
        self.activation = config["activation"]
        self.FC = bl.FC_block(
            self.input_size, self.output_size, activation=self.activation
        )
        assert "loss" in config.keys()
        if self.config["loss"] == "mse":
            self.loss = nn.MSELoss()
        if self.config["loss"] == "poisson":
            self.loss = PoissonLoss()
        self.corr = Corr()
        if self.config["Laplace"] == 3:
            self.laplace = torch.Tensor(laplace())
        if self.config["Laplace"] == 5:
            self.laplace = torch.Tensor(laplace5x5())
        if self.config["Laplace"] == 7:
            self.laplace = torch.Tensor(laplace7x7())

    def forward(self, x):
        x = self.FC(x.reshape(x.shape[0], -1))
        return x

    def reg_L1(self):
        weights = self.FC[0].weight
        reg_term = torch.sum(torch.abs(weights))
        return reg_term

    def reg_L2(self):
        weights = self.FC[0].weight
        reg_term = torch.sum(weights * weights)
        return reg_term

    def reg_Laplacian(self):
        weights = self.FC[0].weight
        reg_term = torch.sqrt(
            torch.sum(
                torch.pow(
                    nn.functional.conv2d(
                        weights.reshape(
                            -1,
                            1,
                            self.config["input_size_x"],
                            self.config["input_size_y"],
                        ),
                        self.laplace,
                        padding="same",
                    ),
                    2,
                )
            )
        )
        return reg_term

    def reg_group_sparsity(self):
        weights = self.FC[0].weight
        reg_term = torch.sum(torch.sqrt(torch.sum(torch.pow(weights, 2), dim=-1)), 0)
        return reg_term

    def regularization(self):
        reg_term = 0
        for k in self.config.keys():
            if k.startswith("reg_"):
                kreg = getattr(self, k)()
                self.log("regularization/" + k[4:], kreg)
                coeff_times_kreg = self.config[k] * kreg
                self.log("regularization/" + k[4:] + "_times_coeff", coeff_times_kreg)
                reg_term = reg_term + coeff_times_kreg
        return reg_term

