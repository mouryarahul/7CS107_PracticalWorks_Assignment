from abc import ABC, abstractmethod
from typing import Callable
import numpy as np

class BaseLayer(ABC):
    def __init__(self) -> None:
        self._params = None
        self._grads = None
        self._cache = None
        super().__init__()
    
    @property
    def params(self):
        return self._params

    @params.setter
    def params(self, params):
        self._params = params 

    @property
    def grads(self):
        return self._grads

    @grads.setter
    def grads(self, grads):
        self._grads = grads

    @abstractmethod
    def forward(self, x: np.ndarray, **kwargs) -> np.ndarray:
        """Perform a forward pass through the layer"""
        raise NotImplementedError

    @abstractmethod
    def backward(self, dL: np.ndarray, **kwargs) -> np.ndarray:
        """Perform a backward pass through the layer"""
        raise NotImplementedError

    @abstractmethod
    def zero_grad(self) -> None:
        """Zero-out gradient if there is one"""
        pass
    
class LinearLayer(BaseLayer):
    """
    Applies a linear transformation to the incoming data: z = x @ W + b
    """
    def __init__(self, in_features:int, out_features:int, bias:bool=True) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.bias = bias
        self._init_params()

    def __str__(self):
        return "Linear with parameters: W of shape [{:d} {:d}] and b of shape [{:d} {:d}]".format(self.in_features, self.out_features, 1, self.out_features) 

    def _init_params(self) -> None:
        """
        Initializes the parameters: W and b from Uniform(-sqrt(k), sqrt(k)), where k = 1 / in_features
        as suggested in Pytorch Documentation
        """
        k = np.sqrt(1/self.in_features)
        W = np.random.uniform(low=-k, high=k, size=(self.in_features, self.out_features)).astype('float32')
        dW = np.zeros_like(W)
        if self.bias:
            b = np.random.uniform(low=-k, high=k, size=(1,self.out_features)).astype('float32')
            db = np.zeros_like(b)
        else: 
            b = None
            db = None
        
        self._params = {"W": W, "b": b}
        self._grads = {"dW": dW, "db": db}

    def forward(self, X:np.ndarray) -> np.ndarray:
        Z = X @ self._params["W"]
        if self.bias:
            Z += self._params["b"]
        self._cache = {"X": X, "Z": Z}
        return Z

    def backward(self, dL_dZ: np.ndarray) -> np.ndarray:
        """
        Calculates the gradients of Loss w.r.t X, W, and b.
        
        Input:
        dL_dZ:  Gradient of Loss w.r.t output Z coming from the layer behind. 
                It should have dimension = [N, out_features], where N is batch size.
        
        Gradients w.r.t to parameters (W and b): vector-Jacobian products 
        dL/dW = dL/dZ @ dZ/dW
        dL/db = dL/dZ @ dZ/db

        Output:
        dL_dX: dL/dZ @ dZ/dX
        """
        X = self._cache["X"]
        W = self._params["W"]
        b = self._params["b"]

        dW  = X.T @ dL_dZ
        if self.bias:
            db = dL_dZ.sum(axis=0, keepdims=True)
        else:
            db = None
        self._grads = {"dW": dW, "db": db}
    
        dL_dx = dL_dZ @ W.T
        return dL_dx

    def zero_grad(self) -> None:
        self._grads["dW"] = np.zeros_like(self._grads["dW"])
        if self.bias:
            self._grads["db"] = np.zeros_like(self._grads["db"])
        else:
            self._grads["db"] = None


class ReLu(BaseLayer):
    """
    Applies the rectified linear unit function element-wise:
    Z = Maximum(0, X)
    """
    def __init__(self) -> None:
        super().__init__()

    def __str__(self):
        return "ReLu"
        
    def forward(self, X: np.ndarray) -> np.ndarray:
        """
        Arg:
        X: it should have dimension [N x m] where N is batch-size and m is feature's length

        Return:
        Z: output with dimension [N x m]
        """
        Z = np.maximum(0, X)
        self._cache = {"X": X, "Z": Z}
        return Z

    def backward(self, dL_dZ: np.ndarray) -> np.ndarray:
        """
        Arg:
        dL_dZ:  Gradient of Loss w.r.t output Z coming from the layer behind. 
                It should have dimension = [N, out_features], where N is batch size.
        
        Return: Vector-Jacobian product: dL_dX: dL/dZ @ dZ/dX
        """
        
        X = self._cache["X"]
        dL_dX = dL_dZ.copy()
        dL_dX[X <= 0.0] = 0.0
        return dL_dX
    
    def zero_grad(self) -> None:
        return super().zero_grad()

class LeakyReLu(BaseLayer):
    """
    Applies the element-wise function:
    LeakyReLu(X) = maximum(0, X) + alpha * minimum(0, X)
    """
    def __init__(self, alpha=0.01) -> None:
        super().__init__()
        self.alpha = alpha

    def __str__(self):
        return "Leaky-ReLu with negative-slope = {:0.4f}".format(self.alpha)

    def forward(self, X: np.ndarray) -> np.ndarray:
        """
        Arg:
        X: it should have dimention [N x m]

        Return:
        Z: it should have dimention [N x m]
        """
        Z = X.copy()
        Z[X < 0] = Z[X < 0] * self.alpha
        self._cache = {"X": X, "Z": Z}
        return Z

    def backward(self, dL_dZ: np.ndarray) -> np.ndarray:
        """
        Input:
        dL_dZ:  Gradient of Loss w.r.t output Z coming from the layer behind. 
                It should have dimension = [N, out_features], where N is batch size.
        
        Output: Vector-Jacobian product
        dL_dX: dL/dZ @ dZ/dX
        """
        X = self._cache["X"]
        dL_dX = dL_dZ.copy()
        dL_dX[X < 0] = dL_dX[X < 0] * self.alpha
        return dL_dX

    def zero_grad(self) -> None:
        return super().zero_grad()

class Sigmoid(BaseLayer):
    """
    Applies the element-wise Sigmoid function:
    Z = 1 / (1 + exp(-X))
    """
    def __init__(self) -> None:
        super().__init__()

    def __str__(self):
        return "Sigmoid"

    def forward(self, X: np.ndarray) -> np.ndarray:
        Z = 1/(1+np.exp(-X))
        self._cache = {"X": X, "Z": Z}
        return Z

    def backward(self, dL_dZ: np.ndarray) -> np.ndarray:
        """
        Input:
        dL_dZ:  Gradient of Loss w.r.t output Z coming from the layer behind. 
                It should have dimension = [N, out_features], where N is batch size.
        
        Output: Vector-Jacobian product
        dL_dx: dL/dZ @ dZ/dX
        """
        Z = self._cache["Z"]
        return dL_dZ * (Z * (1 - Z))
    
    def zero_grad(self) -> None:
        return super().zero_grad()

class Tanh(BaseLayer):
    """
    Applies the element-wise Tanh function:
    Z = Tanh(X)
    """
    def __init__(self) -> None:
        super().__init__()

    def __str__(self):
        return "Tanh"
    
    def forward(self, X: np.ndarray) -> np.ndarray:
        Z = np.tanh(X)
        self._cache = {"X": X, "Z": Z}
        return Z

    def backward(self, dL_dZ: np.ndarray) -> np.ndarray:
        """
        Input:
        dL_dZ:  Gradient of Loss w.r.t output Z coming from the layer behind. 
                It should have dimension = [N, out_features], where N is batch size.
        
        Output: Vector-Jacobian product
        dL_dx: dL/dZ @ dZ/dX
        """
        Z = self._cache["Z"]
        return dL_dZ * (1 - np.square(Z))

    def zero_grad(self) -> None:
        return super().zero_grad()


class   Softmax(BaseLayer):
    """
    Applies the Softmax function to an n-dimensional input Tensor rescaling them so that
    the elements of the n-dimensional output Tensor lie in the range [0,1] and sum to 1.
    Softmax is defined as:

    Softmax(X_i) = exp(X_i) / sum_j{exp(X_j)}
    """
    def __init__(self, axis=-1) -> None:
        super().__init__()
        self.axis = axis

    def __str__(self):
        return "Softmax"

    def forward(self, X: np.ndarray) -> np.ndarray:
        # center the data to avoid overflow
        e_X = np.exp(X - np.max(X, axis=self.axis, keepdims=True))
        Z = e_X / e_X.sum(axis=self.axis, keepdims=True)
        self._cache = {"X": X, "Z": Z}
        return Z

    def backward(self, dL_dZ: np.ndarray) -> np.ndarray:
        """
        Input:
        dL_dZ:  Gradient of Loss w.r.t output Z coming from the layer behind. 
                It should have dimension = [N, out_features], where N is batch size.

        Jacobian dZ_dX[i,j] = 
                            softmax(X_i) * (1 - softmax(X_j))   if i == j
                            -softmax(X_i) * softmax(X_j)        if i != j 
        
        Output: Vector-Jacobian product
        dL_dx: dL/dZ @ dZ/dX
        """
        Z = self._cache["Z"]
        N = dL_dZ.shape[0]
        dL_dX = np.zeros_like(dL_dZ)
        for i in range(N):
            dL_dX[i] = dL_dZ[i] @ (np.diagflat(Z[i]) - Z[i].T @ Z[i])

        return dL_dX
    
    def zero_grad(self) -> None:
        return super().zero_grad()
    
    
        

