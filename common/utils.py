import random
import string
from abc import ABCMeta, abstractmethod
from typing import Optional, Callable, Mapping, Any, Dict

_LENGTH_OF_RANDOM_TOKEN = 64


def random_token(length: Optional[int] = None) -> str:
    """
    Overview:
        Generate random hex token
    Arguments:
        - length (:obj:`Optional[int]`): Length of the random token (`None` means `64`)
    Returns:
        - token (:obj:`str`): Generated random token
    Example:
        >>> random_token()  # '4eAbd5218e3d0da5e7AAFcBF48Ea0Df2dadED1bdDF0B8724FdE1569AA78F24A7'
        >>> random_token(24)  # 'Cd1CdD98caAb8602ac6501aC'
    """
    return ''.join([random.choice(string.hexdigits) for _ in range(length or _LENGTH_OF_RANDOM_TOKEN)])