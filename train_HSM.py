#%%
import wandb
from data_utils import Antolik2016Datamodule
from models import *
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import EarlyStopping
import pytorch_lightning as pl
from models import AntolikHouska_HSM_model

#%%
# Set up your default hyperparameters
config = {
    "seed": 44,
    "region": "region1",
    "num_dog": 9,
    "h_size_perc": 0.2,
    "input_size_x": 31,
    "input_size_y": 31,
    "num_neurons": 103,
    "lr": 0.001,
    "batch_size": 32,
    "max_epochs": 3000,
    "reg_L2_on_readout": 0,
}
#%%
pl.seed_everything(config["seed"], workers=True)
# # Pass your defaults to wandb.init
name = f"baseline_seed={config['seed']}"
run = wandb.init(
    config=config, name=name, project="HSM_on_Antolik2016_data", entity="lucabaroni"
)
# Access all hyperparameter values through wandb.config
config = dict(wandb.config)
# Set up model
model = AntolikHouska_HSM_model(config)


wandb_logger = WandbLogger(log_model=True)
wandb_logger.watch(model, log="all", log_freq=250)

# model.DOG.plot_all_dogs()
dm = Antolik2016Datamodule(
    region="region1", batch_size=config["batch_size"], with_test_dataset=False
)
dm.setup()

#%%
# define logger for trainer
early_stop = (EarlyStopping(monitor="val/corr", patience=600, mode="max"),)

trainer = pl.Trainer(
    # callbacks=[early_stop],
    max_epochs=config["max_epochs"],
    # gpus=0,
    logger=wandb_logger,
    log_every_n_steps=42,
    deterministic=True,
    enable_checkpointing=True,
)
trainer.fit(
    model, train_dataloaders=dm.train_dataloader(), val_dataloaders=dm.val_dataloader(),
)
