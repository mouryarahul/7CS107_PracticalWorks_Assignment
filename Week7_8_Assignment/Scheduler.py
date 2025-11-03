from abc import ABC, abstractmethod
import warnings
from Optimizers import BaseOptimizer
import numpy as np 

class BaseScheduler(ABC):
    def __init__(self, optimizer: BaseOptimizer, last_epoch: int =-1, verbose: bool =False) -> None:
        super().__init__()
        self.optimizer      = optimizer
        self.last_epoch     = last_epoch
        self._base_lr       = optimizer.lr
        self._last_lr       = self._base_lr
        self._step_count    = 0
        self.verbose        = verbose

        self.step()
    
    @abstractmethod
    def get_lr(self):
        raise NotImplemented

    @abstractmethod
    def _get_closed_form_lr(self):
        raise NotImplemented

    def get_last_lr(self):
        return self._last_lr

    def print_lr(self, is_verbose, lr, epoch=None):
        """Display the current learning rate.
        """
        if is_verbose:
            if epoch is None:
                print('Adjusting learning rate'
                      ' to {:.4e}.'.format(lr))
            else:
                epoch_str = ("%.2f" if isinstance(epoch, float) else
                             "%.5d") % epoch
                print('Epoch {}: adjusting learning rate'
                      ' to {:.4e}.'.format(epoch_str, lr))

    def step(self, epoch=None):
        if self._step_count == 1:
            if self.optimizer.counter < 1:
                warnings.warn("Detected call of `lr_scheduler.step()` before `optimizer.step()`."
                "You should call `optimizer.step()` before `lr_scheduler.step()`. Failure to do this "
                "will result in skipping the first value of the learning rate schedule.",
                UserWarning)

        self._step_count += 1
        
        if epoch is None:
            self.last_epoch +=1
            lr = self.get_lr()
        else:
            self.last_epoch = epoch
            if hasattr(self, "_get_closed_form_lr"):
                lr = self._get_closed_form_lr()
            else:
                lr = self.get_lr()

        # set lr of optimizer
        self.optimizer.lr = lr
        self.print_lr(self.verbose, lr, epoch)

        self._last_lr = self.optimizer.lr
        


class StepLR(BaseScheduler):
    """Decays the learning rate of optimizer by gamma every
    step_size epochs. When last_epoch=-1, sets initial lr as lr.

    Args:
        optimizer (Optimizer): Wrapped optimizer.
        step_size (int): Period of learning rate decay.
        gamma (float): Multiplicative factor of learning rate decay.
            Default: 0.1.
        last_epoch (int): The index of last epoch. Default: -1.
        verbose (bool): If ``True``, prints a message to stdout for
            each update. Default: ``False``.

    Example:
        >>> # Assuming optimizer uses lr = 0.05 for all groups
        >>> # lr = 0.05     if epoch < 30
        >>> # lr = 0.005    if 30 <= epoch < 60
        >>> # lr = 0.0005   if 60 <= epoch < 90
        >>> # ...
        >>> scheduler = StepLR(optimizer, step_size=30, gamma=0.1)
        >>> for epoch in range(100):
        >>>     train(...)
        >>>     validate(...)
        >>>     scheduler.step()
    """

    def __init__(self, optimizer: BaseOptimizer, step_size: int, gamma: float = 0.1, last_epoch: int =-1, verbose: bool = False) -> None:
        self.step_size = step_size
        self.gamma = gamma
        super(StepLR, self).__init__(optimizer, last_epoch, verbose)

    def get_lr(self):
        if (self.last_epoch == 0) or (self.last_epoch % self.step_size != 0):
            return self.optimizer.lr
        return self.optimizer.lr * self.gamma

    def _get_closed_form_lr(self):
        return self._base_lr * self.gamma ** (self.last_epoch // self.step_size)
        