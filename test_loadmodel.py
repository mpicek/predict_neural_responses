from models import *
import os
import wandb


# you can get the artifact string from the section artifact in the wandb run (api)
run = wandb.init()
artifact = run.use_artifact(
    "lucabaroni/CNN_SxF_on_Antolik2016/model-2wt2r4cn:v0", type="model"
)
# this download the artifact model
artifact_dir = artifact.download()
wandb.finish()
ckpt = os.path.join(artifact_dir, "model.ckpt")

# you can load the model from checkpoint like this (<-information about the training is still here)
model = CNN_SxF.load_from_checkpoint(ckpt)
print(model)

