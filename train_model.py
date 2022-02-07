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
    "seed": 43,
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
config = wandb.config
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


# #%%
# import numpy as np

# run = wandb.init()

# # best model HSM
# artifact = run.use_artifact(
#     "lucabaroni/HSM_on_Antolik2016_data/model-a76b6w1v:v0", type="model"
# )
# artifact_dir = artifact.download()

# # %%
# import torch

# path = "artifacts/model-a76b6w1v:v0/model.ckpt"
# x = torch.load(path)
# # %%

# from models import AntolikHouska_HSM_model


# config = {
#     "seed": 42,
#     "region": "region1",
#     "num_dog": 9,
#     "h_size_perc": 0.2,
#     "input_size_x": 31,
#     "input_size_y": 31,
#     "num_neurons": 103,
#     "lr": 0.001,
#     "batch_size": 32,
#     "max_epochs": 3000,
#     "reg_L2_on_readout": 0,
# }
# model = AntolikHouska_HSM_model(config)
# #%%
# model.DOG.plot_all_dogs()
# x = torch.load("/home/baroni/DOG/artifacts/model-1qtksh1e:v0/model.ckpt")
# sd = x["state_dict"]
# del sd["DOG.X"]
# del sd["DOG.Y"]
# # print(x["state_dict"].keys())

# new_model = model.load_state_dict(sd)
# model.DOG.plot_all_dogs()
# # print(new_model)

# #%%
# import matplotlib.pyplot as plt

# dm = Antolik2016Datamodule(
#     region="region1", batch_size=config["batch_size"], with_test_dataset=False
# )
# dm.setup()

# rrc = 1e3
# sta = dm.compute_STA_ridge_regr(ridge_regr_coeff=rrc)
# # sta = dm.compute_STA()
# plt.imshow(sta[0])
# plt.colorbar()
# plt.show()
# #%%
# val_img = dm.val_dataset[:][0]
# val_model_prediction = model(val_img)
# multitrial_resp = np.load(
#     "./data/external/antolik2016/Data/region1/raw_validation_set.npy"
# )
# multitrial_resp = multitrial_resp.swapaxes(0, 1)
# val_model_prediction = np.repeat(
#     np.expand_dims(val_model_prediction.detach().numpy(), 1),
#     multitrial_resp.shape[1],
#     axis=1,
# )
# FEV, FEVe = fev(multitrial_resp, val_model_prediction, return_exp_var=True)


# print(feve.astype(int))
# print(FEV)
# print(feve.min())
# print(feve.max())
# np.mean(feve[np.abs(feve) < 1])

# print(FEV)
# #%%

# corr = Corr()
# c = corr(dm.val_dataset[:][1], model(val_img))
# print(c.mean())
# o_j = oracle_corr_jackknife(multitrial_resp)  # underestimate it
# print(o_j.mean())
# o_c = oracle_corr_conservative(multitrial_resp)  # overestimate it
# print(o_c.mean())
# print(torch.mean(c / o_j))
# print(torch.mean(c / o_c))

# %%
