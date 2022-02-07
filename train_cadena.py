#%%
import wandb
from data_utils import Antolik2016Datamodule
from models import *
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import EarlyStopping
import pytorch_lightning as pl

# Set up your default hyperparameters
config = {
    "seed": 42,
    "data_name": "Antolik2016",
    "region": "region1",
    "batch_size": 128,
    "lr": 0.00001,
    "max_epochs": 3000,
    "core_hidden_channels": 64,
    "core_layers": 4,
    "core_input_kern": 9,
    "core_hidden_kern": 7,
    "readout_bias": True,
    "core_gamma_input": 0.01,
    "core_gamma_hidden": 0.01,
    "readout_gamma": 1,
}

# set config seed for everything
pl.seed_everything(config["seed"], workers=True)

# init wandb run
run = wandb.init(
    config=config,
    # name=name,
    project="CNN_SxF_on_Antolik2016",
    entity="lucabaroni",
)

# Access all hyperparameter values through wandb.config
config = dict(wandb.config)

# # setup datamodule
dm = Antolik2016Datamodule(
    region=config["region"], batch_size=config["batch_size"], with_test_dataset=False
)
dm.setup()

# update config for initialization of model (<- certain config parameters depend on data)
config.update(
    {
        "input_channels": dm.train_dataset[:][0].shape[1],
        "input_size_x": dm.train_dataset[:][0].shape[2],
        "input_size_y": dm.train_dataset[:][0].shape[3],
        "num_neurons": dm.train_dataset[:][1].shape[1],
        "mean_activity": dm.train_dataset[:][1].mean(dim=0),
    }
)
# Set up model
model = CNN_SxF(config)

# setup wandb logger
wandb_logger = WandbLogger(log_model=True)
wandb_logger.watch(model, log="all", log_freq=250)

# define callbacks for train
early_stop = EarlyStopping(monitor="val/corr", patience=30, mode="max")

# define trainer
trainer = pl.Trainer(
    callbacks=[early_stop],
    max_epochs=config["max_epochs"],
    gpus=[0],
    logger=wandb_logger,
    log_every_n_steps=42,
    deterministic=True,
    enable_checkpointing=True,
)

trainer.fit(
    model, train_dataloaders=dm.train_dataloader(), val_dataloaders=dm.val_dataloader(),
)
performances = dm.model_performaces(model)
run.summary["corr"] = performances["corr"].mean()
run.summary["fraction oracle jackknife"] = performances["fraction_oracle_j"].mean()
run.summary["fraction oracle conservative"] = performances["fraction_oracle_c"].mean()
run.summary["FEVe"] = performances["FEVe"][performances["FEV"] > 0.3].mean()

