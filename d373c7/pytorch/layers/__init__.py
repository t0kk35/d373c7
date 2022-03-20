"""
Imports for Pytorch custom layers
(c) 2020 d373c7
"""
from .common import PyTorchLayerException, Layer
from .base import Embedding, TensorDefinitionHead, Attention
from .base import AttentionLastEntry
from .base import ConvolutionalBodyBase1d, ConvolutionalBodyBaseTranspose1d
from .output import CategoricalLogSoftmax1d, CategoricalLogSoftmax2d, TailBinary
from .classifiers import LSTMBody, GRUBody, ConvolutionalBody1d, TransformerBody #, TailBinary
from .encoders import LinearEncoder, LinearDecoder, LinearVAEEncoder, LinearVAEDecoder, LinearVAEOut
from .encoders import ConvolutionalEncoder, ConvolutionalDecoder
from .ntp import ConvolutionalNtpBody, LSTMNtpBody

