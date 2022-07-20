import logging

import numpy as np
from tensorboardX import SummaryWriter

logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(filename)s[line:%(lineno)d] - %(levelname)s: %(message)s'
)


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
        if filename is not None:
            self.writer = SummaryWriter(filename, flush_secs=10)

    def log(self, global_step: int, d: dict):
        s = '{}'.format(global_step)
        for key, value in d.items():
            if 'info' in key:
                continue
            if len(value) == 0:
                continue

            if key.split('/')[-1] in ('episode',):
                scalar_value = np.sum(value)
                s += ', {}: {}'.format(key, scalar_value)
                if self.writer is not None:
                    self.writer.add_scalar(
                        tag=key,
                        scalar_value=scalar_value,
                        global_step=global_step
                    )
            else:
                if len(value) == 1:
                    scalar_value = value[0]
                    s += ', {}: {}'.format(key, scalar_value)
                    if self.writer is not None:
                        self.writer.add_scalar(
                            tag=key,
                            scalar_value=scalar_value,
                            global_step=global_step
                        )
                else:
                    scalar_value1 = np.mean(value)
                    scalar_value2 = np.std(value)
                    s += ', {}: {:.2f}+{:.2f}'.format(key, scalar_value1, scalar_value2)
                    if self.writer is not None:
                        self.writer.add_scalar(
                            tag=key,
                            scalar_value=scalar_value1,
                            global_step=global_step
                        )
                        self.writer.add_scalar(
                            tag=key + '/std',
                            scalar_value=scalar_value2,
                            global_step=global_step
                        )
            value.clear()
        logging.info(s)

    def __del__(self):
        if self.writer is not None:
            self.writer.close()
