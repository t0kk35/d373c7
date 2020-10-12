"""
Imports for Pytorch Training functions
(c) 2020 d373c7
"""


class PyTorchTrainException(Exception):
    """Standard exception raised during training"""
    def __init__(self, message: str):
        super().__init__('Error in PyTorch Training: ' + message)
