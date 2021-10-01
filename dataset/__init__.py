from .cars import Cars
from .cub import CUBirds
from .SOP import SOP
from .import utils
from .base import BaseDataset
from .logo2k import Logo2K, Logo2K_super


_type = {
    'cars': Cars,
    'cub': CUBirds,
    'SOP': SOP,
    'logo2k': Logo2K,
    'logo2k_super100': Logo2K_super,
    'logo2k_super500': Logo2K_super,
}

def load(name, root, mode, transform = None):
    return _type[name](root = root, mode = mode, transform = transform)
    
