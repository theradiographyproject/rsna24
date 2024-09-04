import lightning as L
import subprocess as sp
from datasetTrain import DataModule 
from modelEmbedding import EndToEnd
from lightning.pytorch.loggers import CSVLogger
from torch.utils.data import DataLoader
from lightning.pytorch.cli import LightningCLI

def get_gpu_memory():
    command = "nvidia-smi --query-gpu=memory.free --format=csv"
    memory_free_info = sp.check_output(command.split()).decode('ascii').split('\n')[:-1][1:]
    memory_free_values = [int(x.split()[0]) for i, x in enumerate(memory_free_info)]
    return memory_free_values

def main():
    batch_size = 204
    print(get_gpu_memory())
    gpu_factor = get_gpu_memory()[0]//400
    gradients_to_accumulate = (batch_size//gpu_factor)
    print(gpu_factor)

    #model = EndToEnd.load_from_checkpoint("/cbica/home/gangarav/rsna24/logs/reset/test/checkpoints/epoch=3-step=68.ckpt")

    model = EndToEnd(
      dim=768,
      patch_size=(1,16,16),
      depth=12,
      heads=12,
      pos_head_dropout=0.25,
      mim_depth=4,
      mim_head_layer_dropout=0.1,
      learning_rate=1e-4
    )
    
    data = DataModule(
      mini_batch_size=gpu_factor,
      log_batch_size=(batch_size//gpu_factor)*gpu_factor,
      n_workers=4,
      patch_size=(1,16,16),
      support_prism_prob=0.5,
      compare_different_series_prob=0.5
    )
  
    logger = CSVLogger(
        save_dir='/cbica/home/gangarav/rsna24/logs',
        name='reset',
        version="test"
    )
    trainer = L.Trainer(accelerator="gpu", logger=logger, max_epochs=100000000, accumulate_grad_batches=gradients_to_accumulate)
  
    trainer.fit(model, datamodule=data) #, ckpt_path="/cbica/home/gangarav/rsna24/logs/reset/test/checkpoints/epoch=3-step=68.ckpt")
