"""
Quantum Molecular Computing (QMC) Core Library
Single entry point for all transition matrix and molecular graphing tools.
"""


import os
import sys

# Import classes from their respective modules
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
if CURRENT_DIR not in sys.path:
    sys.path.insert(0, CURRENT_DIR)

import numpy as np
from scipy import sparse
import math

from IPython.display import display, Latex, SVG

import schemdraw
import schemdraw.logic as logic
import schemdraw.elements as elm

from rdkit import Chem
from rdkit.Chem.Draw import rdMolDraw2D


from transition_matrix import *
from carbon_molecule import *
from utilities import *

