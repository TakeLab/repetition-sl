from transformers.trainer import Trainer
from .discrete_gauss_probs import get_probs

import math
import random


class TrainerSSPFB(Trainer):
    def __init__(
        self,
        num_hidden_layers,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)

        train_bs = self.args.per_device_train_batch_size
        dlen = len(self.train_dataset)
        probs = get_probs(int(num_hidden_layers))
        sizes = math.ceil(dlen / train_bs)
        self.unmask_confs = []
        for z in range(sizes):
            unmask = [True if random.random() < p else False for p in probs]
            self.unmask_confs.append(unmask)

        # print(dlen, sizes, train_bs)
        # print(self.unmask_confs[0])
        self._current_batch_in_epoch = 0  # To track batch index within an epoch
        self.sizes = sizes

    def compute_loss(
        self, model, inputs, return_outputs=False, num_items_in_batch=None
    ):
        # Add batch_number to inputs
        batch_number = self._current_batch_in_epoch
        if model.training:
            self._current_batch_in_epoch += 1
            self._current_batch_in_epoch %= self.sizes
            # print(batch_number)
        inputs["unmask_conf"] = self.unmask_confs[batch_number]
        # if "return_dict" in inputs:
        #    del inputs["return_dict"]
        # Forward pass
        outputs = model(**inputs)
        loss = outputs["loss"]
        return (loss, outputs) if return_outputs else loss
