from abc import ABC, abstractmethod
import numpy as np
from Utils import is_binary, is_stochastic, indices_to_one_hot

class BaseLoss(ABC):
    def __init__(self) -> None:
        super().__init__()
        self.input: np.ndarray = None
        self.target: np.ndarray = None

    @abstractmethod
    def forward(self, input: np.ndarray, target: np.ndarray) -> float:
        raise NotImplemented

    @abstractmethod
    def backward(self) -> np.ndarray:
        raise NotImplemented


class MSELoss(BaseLoss):
    """
    Creates a criterion that measures the mean squared error (squared L2 norm) between each element
    in the input (of dims [N, d]) and target (of dims [N, d]), where N is batch size and d is length
    of each sample (row vector). 
    """
    def __init__(self) -> None:
        super().__init__()

    def __str__(self):
        return "Mean-Square-Error Loss"
        
    def forward(self, input: np.ndarray, target: np.ndarray) -> float:
        N, d = input.shape
        M, e = target.shape
        assert N == M and d == e, "Shape of input and target should be equal!"
        
        self.input = input
        self.target = target
        total_mse_loss = np.sum(np.square(np.linalg.norm((self.input - self.target), axis=-1))) / (N*d)
        return total_mse_loss

    def backward(self) -> np.ndarray:
        N, d = self.input.shape
        dL = 2 * (self.input - self.target) / (N*d)
        return dL

class L1Loss(BaseLoss):
    """
    Creates a criterion that measures the mean absolute error (MAE) between each element 
    in the input (of dims [N, d]) and target (of dims [N, d]), where N is batch size and d is length
    of each sample (row vector).
    """
    def __init__(self) -> None:
        super().__init__()

    def __str__(self):
        return "L1-Loss"

    def forward(self, input: np.ndarray, target: np.ndarray) -> float:
        N, d = input.shape
        M, e = target.shape
        assert N == M and d == e, "Shape of input and target should be equal!"

        self.input = input
        self.target = target
        total_mae_loss = np.sum(np.linalg.norm((self.input - self.target), ord=1, axis=-1)) / (N*d)
        return total_mae_loss

    def backward(self) -> np.ndarray:
        N, d = self.input.shape
        diff = self.input - self.target
        dL = np.zeros_like(self.input)
        dL[diff > 0] = 1.0 / (N*d)
        dL[diff < 0] = -1.0 / (N*d)
        return dL

class BCELoss(BaseLoss):
    """
    Creates a criterion that measures the Binary Cross Entropy between the target (of dims [N, 1])
    and the input (of dims [N, 1]) probabilities:
    l(x,y) = -(y * log(x) + (1-y) * log(1-x))
    """
    def __init__(self) -> None:
        super().__init__()

    def __str__(self):
        return "Binary Cross-Entropy Loss"

    def forward(self, input: np.ndarray, target: np.ndarray) -> float:
        N, d = input.shape
        M, e = target.shape
        assert N == M and d == e, "Shape of input and target should be equal!"

        self.input = input
        self.target = target

        # prevent taking the log of 0
        eps = np.finfo(float).eps

        total_bce_loss = (np.sum(-(self.target * np.log(self.input + eps) + (1 - self.target) * np.log(1 - self.input + eps)))) / N

        return total_bce_loss

    def backward(self) -> np.ndarray:
        N, d = self.input.shape
        # prevent diving by 0
        eps = np.finfo(float).eps
        dL = -(np.divide(self.target, self.input+eps) - np.divide(1 - self.target, 1 - self.input - eps)) / N
        return dL


class CrossEntropyLoss(BaseLoss):
    """
    This criterion computes the cross entropy loss between input and target.
    It is useful when training a classification problem with C classes. For
    a one-hot target vector y of size d and predicted class probabilities x, 
    the cross-entropy loss is defined as:
    L(x, y) = - sum_{i=1}^{T} {y_i log(x_i)}
    """
    def __init__(self) -> None:
        super().__init__()

    def __str__(self):
        return "Cross-Entropy Loss"

    def forward(self, predictions, targets):
        """
        Calculates the cross-entropy loss.
        
        Parameters:
        - predictions: N x C array (softmax outputs), where N is the batch size and C is the number of classes.
        - targets: N x 1 array of class indices.
        
        Returns:
        - Cross-entropy loss (scalar).
        """
        # Clip predictions to avoid log(0)
        predictions = np.clip(predictions, 1e-12, 1 - 1e-12)
        self.predictions = predictions

        targets = targets.astype(int)
        self.targets = targets.flatten()  # Ensure targets is a 1D array for easy indexing

        # Number of samples
        N = predictions.shape[0]

        # Gather the log probabilities of the correct classes
        correct_class_log_probs = -np.log(predictions[np.arange(N), self.targets])

        # Calculate mean cross-entropy loss
        loss = np.mean(correct_class_log_probs)
        return loss

    def backward(self):
        """
        Calculates the gradient of the cross-entropy loss with respect to predictions.
        
        Returns:
        - Gradient of the loss with respect to predictions (N x C array).
        """
        # Number of samples
        N = self.predictions.shape[0]

        # Create a gradient matrix with predictions as base
        dL = self.predictions.copy()
        
        # Subtract 1 from the correct class positions
        dL[np.arange(N), self.targets] -= 1
        
        # Normalize by the number of samples
        dL /= N
        return dL
