from abc import ABC, abstractmethod
from collections import OrderedDict
from typing import Callable
import pickle
import numpy as np
from Layers import LinearLayer, ReLu, LeakyReLu, Sigmoid, Tanh, Softmax

class BaseModel(ABC):
    def __init__(self) -> None:
        super().__init__()
        self.layers = OrderedDict()

    def __len__(self) -> int:
        return len(self.layers)

    def __str__(self) -> str:
        result = "Layers: \n"
        for i, (key, value) in enumerate(self.layers.items()):
            result += "{}: {}: {} \n".format(i+1, key, value)
        return result
    
    @abstractmethod
    def forward(self, x: np.ndarray) -> np.ndarray:
         """Perform a forward pass through the layers"""
         raise NotImplemented
    
    @abstractmethod
    def backward(self, dL: np.ndarray) -> None:
        """Perform a backward pass through the layer"""
        raise NotImplemented

    def zero_grad(self) -> None:
        """
        Zero-out the gradients w.r.t the parameters.
        """
        for key, value in self.layers.items():
            if value.grads:
                value.zero_grad()

    def get_params(self) -> OrderedDict:
        """ Returns an OrderedDict of Learnable Parameters"""
        result = OrderedDict()
        for key, value in self.layers.items():
            if value.params is not None:
                result[key] = value.params
        return result

    def set_params(self, params: OrderedDict) -> None:
        """ Set the Learnable Parameters of the model."""
        for key, value in params.items():
            self.layers[key].params = value

    def get_grads(self) -> OrderedDict:
        """ Returns an OrderedDict of gradient of Loss w.r.t. Learnable Parameters """
        result = OrderedDict()
        for key, value in self.layers.items():
            if value.grads is not None:
                result[key] = value.grads
        return result
    
    def set_grads(self, grads: OrderedDict) -> None:
        """Set the gradients of the Learnable parameters in the model. """
        for key, value in grads.items():
            self.layers[key].grads = value

    def save_params(self, filename: str):
        """ Saves an OrderedDict of Learnable parameters of the model as a pickle binary file."""
        params = self.get_params()
        with open(filename, 'wb') as f:
            pickle.dump(params, f, protocol=pickle.HIGHEST_PROTOCOL)
       
    def load_params(self, filename: str):
        """ Loads the OrderedDict of Learnable parameter from a pickle file and set it into the model."""
        with open(filename, 'rb') as f:
            params = pickle.load(f)
        self.set_params(params)


class BinaryClassifier(BaseModel):
    def __init__(self, in_features:int, out_features:int) -> None:
        super().__init__()
        self.layers = OrderedDict([
                                    ('ll1', LinearLayer(in_features, 3, bias=True)),
                                    ('nl1', ReLu()),
                                    #('ll2', LinearLayer(5, 5, bias=True)),
                                    #('nl2', ReLu()),
                                    #('ll3', LinearLayer(5, 5, bias=True)),
                                    #('nl3', ReLu()),
                                    ('ll4', LinearLayer(3, out_features, bias=True)),
                                    ('nl4', Sigmoid())
                                ])
    
    def forward(self, X: np.ndarray) -> np.ndarray:
        """
        Performs Forward-pass calculations through the layers.

        Arg:
        X: should have dimension [N x d] where N is batch-size and d is feature's length

        Return:
        Z: should have dimension [N x 1]
        """
        for key, value in self.layers.items():
            X = value.forward(X)
        Z = X
        return Z
    
    def predict(self, X: np.array) -> np.ndarray:
        '''
        Performs Forward-pass calculations through the layers and then convert the last layer's output as final class indices.

        Arg: 
        X: should have dimension [N x d] where N is batch-size and d is input feature's length

        Return:
        class_indices: final class indices that should have dimension [N x 1]
        '''
        Z = self.forward(X)
        # Convert the Z value to 0 or 1 using threshold 0.5.
        class_indices = Z
        class_indices[class_indices < 0.5] = 0
        class_indices[class_indices >= 0.5] = 1
        return class_indices


    def backward(self, dL: np.ndarray) -> None:
        """
            Backpropagates the gradient of Loss (dL/dY) and calculates gradients w.r.t 
            the parameters in different layers.
            
            Arg:
            dL: gradient of Loss w.r.t final output Y from Forward-pass
            
            Return: 
            None
        """
        for key, value in reversed(self.layers.items()):
            dL = value.backward(dL)


class LinearRegressor(BaseModel):
    def __init__(self, in_features:int, out_features:int) -> None:
        super().__init__()
        self.layers = OrderedDict([
                                    ('ll1', LinearLayer(in_features, 6, bias=True)),
                                    ('nl1', ReLu()),
                                    #('ll2', LinearLayer(20, 20, bias=True)),
                                    #('nl2', ReLu()),
                                    #('ll3', LinearLayer(5, 5, bias=True)),
                                    #('nl3', ReLu()),
                                    ('ll4', LinearLayer(6, out_features, bias=True))
                                ])
    
    def forward(self, X: np.ndarray) -> np.ndarray:
        """
        Performs Forward-pass calculations through the layers.

        Arg:
        X: should have dimension [N x d] where N is batch-size and d is feature's length

        Return:
        Z: should have dimension [N x 1]
        """
        for key, value in self.layers.items():
            X = value.forward(X)
        Z = X
        return Z

    def backward(self, dL: np.ndarray) -> None:
        """
            Backpropagates the gradient of Loss (dL/dY) and calculates gradients w.r.t 
            the parameters in different layers.
            
            Arg:
            dL: gradient of Loss w.r.t final output Y from Forward-pass
            
            Return: 
            None
        """
        for key, value in reversed(self.layers.items()):
            dL = value.backward(dL)


class MultiClassClassifier(BaseModel):
    def __init__(self, in_features:int, out_features:int) -> None:
        super().__init__()
        self.layers = OrderedDict([
                                    ('ll1', LinearLayer(in_features, 7, bias=True)),
                                    ('nl1', ReLu()),
                                    #('ll2', LinearLayer(5, 5, bias=True)),
                                    #('nl2', ReLu()),
                                    #('ll3', LinearLayer(5, 5, bias=True)),
                                    #('nl3', ReLu()),
                                    ('ll4', LinearLayer(7, out_features, bias=True)),
                                    ('nl4', Softmax())
                                ])
    
    def forward(self, X: np.ndarray) -> np.ndarray:
        """
        Performs Forward-pass calculations through the layers.

        Arg:
        X: should have dimension [N x d] where N is batch-size and d is input feature's length

        Return:
        Z: should have dimension [N x C] where C is number of possible classes
        """
        for key, value in self.layers.items():
            X = value.forward(X)
        Z = X
        return Z
    
    def predict(self, X: np.array) -> np.ndarray:
        '''
        Performs Forward-pass calculations through the layers and then convert the last layer's output as final class index.

        Arg: 
        X: should have dimension [N x d] where N is batch-size and d is input feature's length

        Return:
        Z: final class indices that should have dimension [N x 1]
        '''
        Z = self.forward(X)
        # Use np.argmax to select the class with the highest probability for each sample
        class_indices = np.argmax(Z, axis=1)
        return class_indices

    def backward(self, dL: np.ndarray) -> None:
        """
            Backpropagates the gradient of Loss (dL/dY) and calculates gradients w.r.t 
            the parameters in different layers.
            
            Arg:
            dL: gradient of Loss w.r.t final output Y from Forward-pass
            
            Return: 
            None
        """
        for key, value in reversed(self.layers.items()):
            dL = value.backward(dL)


        

