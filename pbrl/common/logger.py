import logging
import os

import numpy as np
from tensorboardX import SummaryWriter


def update_dict(d1: dict, d2: dict, prefix=''):
    for key, value in d2.items():
        if key == 'timestep':
            continue
        key = prefix + key
        if key not in d1:
            d1[key] = []
        if isinstance(value, list):
            d1[key] += value
        else:
            d1[key].append(value)


class Logger:
    def __init__(self, filename: str = None):
        self.writer = None
        self.logger = logging.getLogger('pbrl')
        self.logger.setLevel(logging.INFO)
        formatter = logging.Formatter('%(asctime)s: %(message)s')
        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(formatter)
        self.logger.addHandler(stream_handler)

        if filename is not None:
            self.writer = SummaryWriter(filename, flush_secs=10)
            file_hander = logging.FileHandler(os.path.join(filename, 'log'))
            file_hander.setFormatter(formatter)
            self.logger.addHandler(file_hander)

    def log(self, global_step: int, d: dict):
        s = '{}'.format(global_step)
        for key, value in d.items():
            if 'info' in key:
                value.clear()
                continue
            if len(value) == 0:
                continue

            if key.split('/')[-1] in ('episode',):
                scalar_value = np.sum(value)
                s += ', {}: {}'.format(key, scalar_value)
            else:
                if len(value) > 1:
                    scalar_value = np.mean(value)
                    scalar_value_std = np.std(value)
                    s += ', {}: {:.2f}+{:.2f}'.format(key, scalar_value, scalar_value_std)
                else:
                    scalar_value = value[0]
                    s += ', {}: {:.2f}'.format(key, scalar_value)

            if self.writer is not None:
                self.writer.add_scalar(
                    tag=key,
                    scalar_value=scalar_value,
                    global_step=global_step
                )
            value.clear()
        self.logger.info(s)

    def __del__(self):
        if self.writer is not None:
            self.writer.close()
