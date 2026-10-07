"""
Quantum Molecular Computing (QMC) Core Library
Single entry point for all transition matrix and molecular graphing tools.
"""

import os
import sys

import numpy as np
from scipy import sparse
import math

from IPython.display import display, Latex, SVG

import schemdraw
import schemdraw.logic as logic
import schemdraw.elements as elm

from rdkit import Chem
from rdkit.Chem.Draw import rdMolDraw2D

# Import classes from their respective modules
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
if CURRENT_DIR not in sys.path:
    sys.path.insert(0, CURRENT_DIR)

from transition_matrix import *
from carbon_molecule import *


#
# Utilities
#
def display_inline(*elements):
    """
    Print matrices and symbols/texts on one line.
    """
    parts = []
    for elem in elements:
        # For TransitionMatrix
        if hasattr(elem, "_repr_latex_"):
            # We extract the LaTeX code without the '$$'
            parts.append(elem._repr_latex_().strip("$"))
        else:
            # If it's text or a symbol (ex: r"\cdot", "=", r"\otimes")
            parts.append(str(elem))
            
    # We assemble everything inside a bloc '$$ ... $$'
    full_latex = "$$ " + " ".join(parts) + " $$"
    display(Latex(full_latex))

def quote_latex(text: str) -> str:
    """
    Wraps a string with LaTeX typographical opening (``) and closing ('') quotes.
    Suitable for use inside math mode, e.g. with display_inline().
    
    Example:
        quote_latex("01") -> r"\text{``01''}"
    """
    # Escapes backslashes if present and formats as LaTeX text with curly quotes
    return rf"\text{{``{text}''}}"