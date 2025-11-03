from abc import ABC, abstractmethod
import numpy as np
from Models import BaseModel
import copy


class BaseOptimizer(ABC):
    def __init__(self, model: BaseModel, lr: float) -> None:
        self._model = model
        self._lr = lr
        self._step_counter = 0
        super().__init__()

    @property
    def counter(self) -> int:
        return self._step_counter

    @property
    def lr(self) -> float:
        return self._lr

    @lr.setter
    def lr(self, lr: float) -> None:
        if lr < 0.0:
            raise ValueError("Learning rate must be non-negative.")
        self._lr = lr

    @abstractmethod
    def step(self) -> None:
        raise NotImplementedError("The 'step' method must be implemented by subclasses.")


class SGD(BaseOptimizer):
    """
    Stochastic Gradient Descent algorithm to update model parameters.
    """

    def __init__(self, model: BaseModel, lr: float, momentum: float = 0.9, dampening: float = 0.0, 
                 weight_decay: float = 0.0, nesterov: bool = False, maximize: bool = False) -> None:
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if momentum < 0.0:
            raise ValueError(f"Invalid momentum value: {momentum}")
        if weight_decay < 0.0:
            raise ValueError(f"Invalid weight decay value: {weight_decay}")
        if nesterov and (momentum <= 0 or dampening != 0):
            raise ValueError("Nesterov momentum requires a positive momentum and zero dampening.")

        super().__init__(model, lr)
        self.momentum = momentum
        self.dampening = dampening
        self.weight_decay = weight_decay
        self.nesterov = nesterov
        self.maximize = maximize
        self._velocity = None

    def __str__(self) -> str:
        return "SGD optimizer"

    def step(self) -> None:
        params = self._model.get_params()
        grads = self._model.get_grads()

        # Apply weight decay
        if self.weight_decay != 0:
            for key, param in params.items():
                grads[key]["dW"] += self.weight_decay * param["W"]
                if param["b"] is not None:
                    grads[key]["db"] += self.weight_decay * param["b"]

        # Momentum update
        if self.momentum != 0:
            if self._velocity is None:
                self._velocity = copy.deepcopy(grads)
            else:
                for key, grad in grads.items():
                    self._velocity[key]["dW"] = (
                        self.momentum * self._velocity[key]["dW"] + (1 - self.dampening) * grad["dW"]
                    )
                    if grad["db"] is not None:
                        self._velocity[key]["db"] = (
                            self.momentum * self._velocity[key]["db"] + (1 - self.dampening) * grad["db"]
                        )

            # Nesterov momentum update
            if self.nesterov:
                for key in grads:
                    grads[key]["dW"] += self.momentum * self._velocity[key]["dW"]
                    if grads[key]["db"] is not None:
                        grads[key]["db"] += self.momentum * self._velocity[key]["db"]
            else:
                grads = copy.deepcopy(self._velocity)

        # Update parameters
        for key, grad in grads.items():
            params[key]["W"] += self._lr * grad["dW"] if self.maximize else -self._lr * grad["dW"]
            if grad["db"] is not None:
                params[key]["b"] += self._lr * grad["db"] if self.maximize else -self._lr * grad["db"]

        self._model.set_params(params)
        self._step_counter += 1


class Adam(BaseOptimizer):
    """
    Adam optimizer implementation with AMSGrad variant, based on Kingma and Ba (2014).
    """
    
    def __init__(self, model: BaseModel, lr: float = 0.001, beta1: float = 0.9, beta2: float = 0.999,
                 epsilon: float = 1e-8, weight_decay: float = 0.0, amsgrad: bool = False) -> None:
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if not (0.0 <= beta1 < 1.0):
            raise ValueError(f"Invalid beta1 value: {beta1}")
        if not (0.0 <= beta2 < 1.0):
            raise ValueError(f"Invalid beta2 value: {beta2}")
        if weight_decay < 0.0:
            raise ValueError(f"Invalid weight decay value: {weight_decay}")
        if epsilon <= 0.0:
            raise ValueError(f"Invalid epsilon value: {epsilon}")

        super().__init__(model, lr)
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.weight_decay = weight_decay
        self.amsgrad = amsgrad
        self._m = None  # First moment vector
        self._v = None  # Second moment vector
        self._v_hat = None if not amsgrad else {}  # Maximum second moment vector for AMSGrad

    def __str__(self) -> str:
        return "Adam optimizer with AMSGrad" if self.amsgrad else "Adam optimizer"

    def step(self) -> None:
        params = self._model.get_params()
        grads = self._model.get_grads()

        if self._m is None or self._v is None:
            # Initialize first and second moment vectors
            self._m = {k: {'dW': np.zeros_like(v['dW']), 'db': np.zeros_like(v['db']) if v['db'] is not None else None} for k, v in grads.items()}
            self._v = {k: {'dW': np.zeros_like(v['dW']), 'db': np.zeros_like(v['db']) if v['db'] is not None else None} for k, v in grads.items()}
            if self.amsgrad:
                # Initialize max second moment for AMSGrad
                self._v_hat = {k: {'dW': np.zeros_like(v['dW']), 'db': np.zeros_like(v['db']) if v['db'] is not None else None} for k, v in grads.items()}

        # Increment step counter for bias correction
        self._step_counter += 1
        beta1, beta2 = self.beta1, self.beta2

        # Update parameters with Adam / AMSGrad
        for key in params:
            grad_w, grad_b = grads[key]["dW"], grads[key].get("db")
            m_w, v_w = self._m[key]["dW"], self._v[key]["dW"]

            # Weight decay (L2 regularization)
            if self.weight_decay != 0:
                grad_w += self.weight_decay * params[key]["W"]
                if grad_b is not None:
                    grad_b += self.weight_decay * params[key]["b"]

            # Update biased first and second moment estimates for weights
            m_w[:] = beta1 * m_w + (1 - beta1) * grad_w
            v_w[:] = beta2 * v_w + (1 - beta2) * (grad_w ** 2)

            if self.amsgrad:
                # Use maximum of past and current second moment for weights
                v_hat_w = self._v_hat[key]["dW"]
                v_hat_w[:] = np.maximum(v_hat_w, v_w)
                weight_update = m_w / (np.sqrt(v_hat_w) + self.epsilon)
            else:
                # Standard Adam update
                weight_update = m_w / (np.sqrt(v_w) + self.epsilon)

            # Correct bias in moment estimates
            m_w_hat = m_w / (1 - beta1 ** self._step_counter)
            if self.amsgrad:
                weight_update = m_w_hat / (np.sqrt(v_hat_w) + self.epsilon)
            else:
                v_w_hat = v_w / (1 - beta2 ** self._step_counter)
                weight_update = m_w_hat / (np.sqrt(v_w_hat) + self.epsilon)

            # Update weights
            params[key]["W"] -= self._lr * weight_update

            # Bias correction and update for biases (if applicable)
            if grad_b is not None:
                m_b, v_b = self._m[key]["db"], self._v[key]["db"]
                m_b[:] = beta1 * m_b + (1 - beta1) * grad_b
                v_b[:] = beta2 * v_b + (1 - beta2) * (grad_b ** 2)

                if self.amsgrad:
                    v_hat_b = self._v_hat[key]["db"]
                    v_hat_b[:] = np.maximum(v_hat_b, v_b)
                    bias_update = m_b / (np.sqrt(v_hat_b) + self.epsilon)
                else:
                    m_b_hat = m_b / (1 - beta1 ** self._step_counter)
                    v_b_hat = v_b / (1 - beta2 ** self._step_counter)
                    bias_update = m_b_hat / (np.sqrt(v_b_hat) + self.epsilon)

                params[key]["b"] -= self._lr * bias_update

        # Update model with new parameters
        self._model.set_params(params)


"""
def clip_grad_norm(model: BaseModel, max_norm: float, norm_type: float = 2.0, error_if_nonfinite: bool = True) -> float:
    max_norm = float(max_norm)
    norm_type = float(norm_type)

    # Get model's current gradients
    grads = model.get_grads()

    if len(grads) == 0:
        return 0

    # Calculate norm of the all gradients together, as if they were concatenated into a single vector.
    norms = []
    for key, value in grads.items():
        if norm_type == np.inf:
            norms.append(np.max(np.abs(value["dW"].flatten())))
            if value["db"] is not None:
                norms.append(np.max(np.abs(value["db"].flatten())))
        else:
            norms.append(np.linalg.norm(value["dW"].flatten(), norm_type))
            if value["db"] is not None:
                norms.append(np.linalg.norm(value["db"].flatten(), norm_type))
        
    # Calculate total norm
    if norm_type == np.inf:
        total_norm = norms[0] if len(norms) == 1 else np.max(np.stack(norms))
    else:
        total_norm = np.linalg.norm(np.stack(norms), norm_type)

    if error_if_nonfinite and not np.isfinite(total_norm):
        raise RuntimeError(
            f"The total norm of order {norm_type} for gradients from parameters is non-finite, so it cannot be clipped!")
    
    clip_coef = max_norm / (total_norm + 1e-6)
    clip_coef_clamped = np.maximum(clip_coef, 1.0)

    # clip the gradients
    for key,value in grads.items():
        grads[key]["dW"] *= clip_coef
        if value["db"] is not None:
            grads[key]["db"] *= clip_coef
    
    # set model grads to new clipped grads
    model.set_grads(grads)

    return total_norm
"""

def clip_grad_norm(model: BaseModel, max_norm: float, norm_type: float = 2.0, error_if_nonfinite: bool = True) -> float:
    max_norm = float(max_norm)
    norm_type = float(norm_type)

    # Get model's gradients
    grads = model.get_grads()
    if not grads:
        return 0.0

    # Calculate norms of all gradients
    norms = [
        np.linalg.norm(value["dW"].flatten(), norm_type) if norm_type != np.inf else np.max(np.abs(value["dW"].flatten()))
        for value in grads.values()
    ] + [
        np.linalg.norm(value["db"].flatten(), norm_type) if value["db"] is not None and norm_type != np.inf else np.max(np.abs(value["db"].flatten()))
        for value in grads.values() if value["db"] is not None
    ]

    # Calculate total norm
    total_norm = max(norms) if norm_type == np.inf else np.linalg.norm(norms, norm_type)

    if error_if_nonfinite and not np.isfinite(total_norm):
        raise RuntimeError(f"Non-finite total norm {total_norm} for gradients with norm type {norm_type}. Clipping cannot proceed.")

    # Compute clipping coefficient, clamping it to a maximum of 1.0
    clip_coef = min(max_norm / (total_norm + 1e-6), 1.0)

    # Apply clipping to gradients
    for grad in grads.values():
        grad["dW"] *= clip_coef
        if grad["db"] is not None:
            grad["db"] *= clip_coef

    # Update model with clipped gradients
    model.set_grads(grads)
    return total_norm

