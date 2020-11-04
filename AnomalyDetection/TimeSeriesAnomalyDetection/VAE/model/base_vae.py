from typing import List, Any
import torch
from torch import tensor
from torch import nn
from abc import abstractmethod

class BaseVAE(nn.Module):
    """
    Base VAE class, only the abstract class, you should implement it't functions yourself
    """
    def __init__(self)->None:
        super(BaseVAE, self).__init__()

    def encode(self, input:tensor)->List(tensor):
        """the encoder of VAE

        Args:
            input (torch.tensor): inputs

        Returns:
            list of torch.tensor
        """
        raise NotImplementedError

    def decode(self, input:tensor)->Any:
        """the decoder of VAE

        Args:
            input (tensor): inputs

        Returns:
            Any: the type based this functions' implemention
        """
        raise NotImplementedError

     def sample(self, batch_size:int, current_device: int, **kwargs) -> tensor:
         """sample from the dataset

         Args:
             batch_size (int): batch size
             current_device (int): the index of device which using train

         Raises:
             RuntimeWarning: runtime warning

         Returns:
             tensor: return type is torch.tensor
         """
        raise RuntimeWarning()

    def generate(self, x: tensor, **kwargs) -> tensor:
        """[summary]

        Args:
            x (Tensor): [description]

        Raises:
            NotImplementedError: [description]

        Returns:
            tensor: return type is torch.tensor
        """
        raise NotImplementedError

    @abstractmethod
    def forward(self, *inputs: tensor) -> tensor:
        """forward broadcast

        Returns:
            tensor: return type is torch.tensor
        """
        pass

    @abstractmethod
    def loss_function(self, *inputs: Any, **kwargs) -> tensor:
        """the cost function used for backward broadcast

        Returns:
            tensor: return type is torch.tensor
        """
        pass


    