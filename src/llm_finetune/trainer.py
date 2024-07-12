import time
from transformers import TrainerCallback
# from transformers.trainer_utils import PREFIX_CHECKPOINT_DIR


class EpochTimingCallback(TrainerCallback):
    """A callback to measure the time taken for each epoch."""

    def __init__(self):
        self.epoch_start_time = None

    def on_epoch_begin(self, args, state, control, **kwargs):
        self.epoch_start_time = time.time()

    def on_epoch_end(self, args, state, control, **kwargs):
        epoch_time = time.time() - self.epoch_start_time
        print(f"Epoch {state.epoch} took {epoch_time:.2f} seconds")
