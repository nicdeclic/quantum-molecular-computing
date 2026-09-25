import numpy as np
from scipy import sparse
import math

from IPython.display import display, Latex, SVG

import schemdraw
import schemdraw.logic as logic
import schemdraw.elements as elm

from rdkit import Chem
from rdkit.Chem.Draw import rdMolDraw2D


#
# Utilities
#
def display_inline(*elements):
    """
    Affiche des matrices et des symboles/textes côte à côte sur une seule ligne.
    """
    parts = []
    for elem in elements:
        # Si c'est une de vos matrices TransitionMatrix
        if hasattr(elem, "_repr_latex_"):
            # On récupère le code LaTeX sans les '$$'
            parts.append(elem._repr_latex_().strip("$"))
        else:
            # Si c'est du texte ou un symbole (ex: r"\cdot", "=", r"\otimes")
            parts.append(str(elem))
            
    # On assemble tout à l'intérieur d'un seul bloc '$$ ... $$'
    full_latex = "$$ " + " ".join(parts) + " $$"
    display(Latex(full_latex))

#
# Classes
#

# Dyadic Transition matrix
class TransitionMatrix:
    def __init__(self, data=[0]):
        # Convert input to a CSR sparse matrix
        if isinstance(data, sparse.spmatrix):
            sp_matrix = data.tocsr()
        else:
            sp_matrix = sparse.csr_matrix(data, dtype=float)
        
        # Validate dimensions are powers of 2
        self._check_dyadic_dim(sp_matrix.shape[0], "rows")
        self._check_dyadic_dim(sp_matrix.shape[1], "columns")
        
        self.matrix = sp_matrix

    def _check_dyadic_dim(self, n, dim_name):
        """Checks if n is a power of 2 using bitwise logic."""
        if n == 0 or (n & (n - 1) != 0):
            raise ValueError(f"Invalid dimension: {dim_name} ({n}) must be a power of 2.")

    @classmethod
    def bitstring(cls, bitstr):
        """Converts a bitstring to a sparse row vector."""
        n = len(bitstr)
        size = 2**n
        index = int(bitstr, 2)
        
        row, col, data = np.array([0]), np.array([index]), np.array([1.0])
        sp_matrix = sparse.csr_matrix((data, (row, col)), shape=(1, size))
        return cls(sp_matrix)

    def normalize(self):
        """Ensures row sums are 0 or 1."""
        row_sums = np.array(self.matrix.sum(axis=1)).flatten()
        
        # Safely calculate 1/row_sums only where row_sums is not 0
        inv_row_sums = np.divide(
            1.0, 
            row_sums, 
            out=np.zeros_like(row_sums), 
            where=row_sums != 0
        )
        
        m = sparse.diags(inv_row_sums)
        return TransitionMatrix(m @ self.matrix)

    def reverse(self):
        """Bayesian Inverse. Transposes the matrix and normalizes the result."""
        # Transpose then Normalize
        return TransitionMatrix(self.matrix.T).normalize()
    
    def measure(self):
        """
        Collapses a state vector (1x2^n matrix) into a single bitstring 
        based on the probabilities in the matrix.
        """
        # 1. Ensure it is a one-row matrix (state vector)
        if self.matrix.shape[0] != 1:
            raise ValueError(f"Measurement requires a state vector (1 row). "
                             f"Current matrix has {self.matrix.shape[0]} rows.")

        # 2. Extract probabilities and handle sparse format
        # We convert to a dense 1D array for np.random.choice
        probs = self.matrix.toarray().flatten()

        # 3. Handle floating point precision
        # np.random.choice requires probabilities to sum EXACTLY to 1.0
        total_prob = np.sum(probs)
        if total_prob == 0:
            raise ValueError("Cannot measure a zero vector (no possible outcomes).")
        
        # Re-normalize locally to fix tiny floating-point errors
        probs = probs / total_prob

        # 4. Pick an index based on the distribution
        indices = np.arange(len(probs))
        chosen_index = np.random.choice(indices, p=probs)

        # 5. Convert index back to bitstring
        num_bits = int(math.log2(self.matrix.shape[1]))
        return bin(chosen_index)[2:].zfill(num_bits)

    def __matmul__(self, other):
        """Dot Product (A @ B)"""
        return TransitionMatrix(self.matrix @ other.matrix)

    def __xor__(self, other):
        """Kronecker Product (A ^ B)"""
        return TransitionMatrix(sparse.kron(self.matrix, other.matrix))

    def __pow__(self, n):
        """Kronecker Power (A ** n)"""
        if not isinstance(n, int) or n < 1:
            raise ValueError("Power must be a positive integer.")
        res = self.matrix
        for _ in range(n - 1):
            res = sparse.kron(res, self.matrix)
        return TransitionMatrix(res)

    def __add__(self, other):
        """Matrix addition"""
        return TransitionMatrix(self.matrix + other.matrix)

    def __mul__(self, other):
        """Hadamard Product. Element-wise multiplication"""
        return TransitionMatrix(self.matrix.multiply(other.matrix))

    def toarray(self):
        return self.matrix.toarray()

    def __repr__(self):
        return f"TransitionMatrix({self.matrix.shape[0]}x{self.matrix.shape[1]})\n{self.toarray()}"
    
    def to_katex(self, precision=2, show_labels=False, show_dims=False, 
                     h_lines=[], v_lines=[], corner_label="", 
                     show_all=True, row_labels=None, col_labels=None):
            from IPython.display import Latex
            import math
    
            # 1. Determine which rows and columns to display
            rows_n, cols_n = self.matrix.shape
            if show_all:
                active_rows = list(range(rows_n))
                active_cols = list(range(cols_n))
            else:
                coo = self.matrix.tocoo()
                active_rows = sorted(list(set(coo.row)))
                active_cols = sorted(list(set(coo.col)))
    
            if not active_rows or not active_cols:
                return Latex("$$ \\text{Empty Matrix} $$")
    
            # 2. Label helper logic
            row_bits = int(math.log2(rows_n))
            col_bits = int(math.log2(cols_n))
            
            def get_label(idx, bits, custom_list):
                if custom_list and idx < len(custom_list):
                    return f"\\text{{{custom_list[idx]}}}"
                # Default to LaTeX curly quotes bitstring
                return f"\\text{{``{bin(idx)[2:].zfill(bits)}''}}"
    
            # 3. Build Column Alignment String (The Box)
            # r for labels, | border, c for data, | border
            align = "r" if show_labels else "" 
            align += "|"
            for i, c_idx in enumerate(active_cols):
                align += "c"
                # Draw internal vertical separator if specified
                if c_idx in v_lines and i < len(active_cols) - 1:
                    align += "|"
            align += "|"
            
            # 4. Build Table Header
            tex = f"\\begin{{array}}{{{align}}} "
            if show_labels:
                headers = [f"\\text{{{corner_label}}}"] 
                for c in active_cols:
                    col_label = "~" if cols_n == 1 else get_label(c, col_bits, col_labels)
                    headers.append(col_label)
                tex += " & ".join(headers) + " \\\\ "
    
            tex += " \\hline " # Top box border
    
            # 5. Build Rows
            dense = self.matrix.toarray()
            for i, r_idx in enumerate(active_rows):
                # Internal horizontal separator
                if r_idx in h_lines and i > 0:
                    tex += " \\hline "
                
                row_cells = []
                if show_labels:
                    row_label = "~" if rows_n == 1 else get_label(r_idx, row_bits, row_labels)
                    row_cells.append(row_label)
                
                for c_idx in active_cols:
                    val = dense[r_idx, c_idx]
                    if val == 0: row_cells.append("0") 
                    elif val == 1: row_cells.append("1")
                    else: row_cells.append(f"{val:.{precision}g}")
                
                tex += " & ".join(row_cells) + " \\\\ "
    
            tex += " \\hline " # Bottom box border
            tex += "\\end{array}"
    
            # 6. Dimension Footer
            if show_dims:
                dim_str = f"({rows_n} \\times {cols_n})"
                tex = f"\\begin{{array}}{{r}} {tex} \\\\ \\scriptstyle {dim_str} \\end{{array}}"
    
            return Latex(f"$$ {tex} $$")
        
    def _repr_latex_(self):
        """Internal hook for Jupyter to render LaTeX automatically."""
        return self.to_katex().data
    
    def to_qubit(self):
        """Convert the 1X8 transition matrix into a qubit ket"""
        matrix = self.matrix
        if matrix.shape[0] != 1 or matrix.shape[1] != 8:
            raise ValueError(f"Invalid dimension: matrix must be 1 X 8.")
        
        # Replace all double indices with single-step indices
        gamma = np.sqrt(np.abs(matrix[0,0] - matrix[0,2]) + 
                        np.abs(matrix[0,1] - matrix[0,3]) + 
                        np.abs(matrix[0,4] - matrix[0,6]) + 
                        np.abs(matrix[0,5] - matrix[0,7]))
        
        if gamma <= 0:
            raise ValueError(f"Invalid transition matrix describing a zero probability qubit.")
        
        # (Please keep in mind that the qubit matrices are defined for use in left-to-right notations.
        #  They will appear like transposed matrices compared to Dirac notation)
        qubit = np.array([
            [complex(np.sqrt(matrix[0,0]) - np.sqrt(matrix[0,2]), 
                    np.sqrt(matrix[0,1]) - np.sqrt(matrix[0,3])) / gamma,
            complex(np.sqrt(matrix[0,4]) - np.sqrt(matrix[0,6]), 
                    np.sqrt(matrix[0,5]) - np.sqrt(matrix[0,7])) / gamma]
        ])
        
        return qubit
    
    def from_qubit(self, qubit):
        """Convert the qubit ket into a 1X8 transition matrix"""
        if qubit.shape[0] != 1 or qubit.shape[1] != 2:
            raise ValueError(f"Invalid dimension: matrix must be 1 X 2.")
        
        x=qubit[0,0].real
        if x>=0:
            b0=x*x
            b2=0.
        else:
            b0=0.
            b2=x*x

        x=qubit[0,0].imag
        if x>=0:
            b1=x*x
            b3=0.
        else:
            b1=0.
            b3=x*x

        x=qubit[0,1].real
        if x>=0:
            b4=x*x
            b6=0.
        else:
            b4=0.
            b6=x*x

        x=qubit[0,1].imag
        if x>=0:
            b5=x*x
            b7=0.
        else:
            b5=0.
            b7=x*x

        self.matrix = sparse.csr_matrix([b0,b1,b2,b3,b4,b5,b6,b7], dtype=float)      

        return self


#
# Functions
#

def custom_swap(n, bit_i, bit_j):
    """
    Returns a 2^n x 2^n TransitionMatrix that swaps the values of bit_i and bit_j.
    
    Validation:
    - n, bit_i, bit_j must be integers.
    - n must be >= 1.
    - bit_i and bit_j must be in the range [0, n-1].
    """
    
    # 1. Validate Types
    if not all(isinstance(x, int) for x in [n, bit_i, bit_j]):
        raise TypeError(f"All parameters must be integers. Received: n={type(n)}, "
                        f"bit_i={type(bit_i)}, bit_j={type(bit_j)}")

    # 2. Validate n
    if n < 1:
        raise ValueError(f"The number of bits (n) must be at least 1. Received: {n}")

    # 3. Validate Bit Indices
    for label, idx in [("bit_i", bit_i), ("bit_j", bit_j)]:
        if not (0 <= idx < n):
            raise ValueError(f"Index {label} ({idx}) is out of range for a {n}-bit system. "
                             f"Indices must be between 0 and {n-1}.")

    # --- Logical Implementation ---

    if bit_i == bit_j:
        # Swapping a bit with itself is mathematically the Identity matrix
        return TransitionMatrix(sparse.eye(2**n))

    size = 2**n
    rows = np.arange(size)
    cols = np.zeros(size, dtype=int)

    # Shift logic: bit 0 is the leftmost (most significant)
    shift_i = n - 1 - bit_i
    shift_j = n - 1 - bit_j

    for k in range(size):
        # Extract the bit values at the two positions
        val_i = (k >> shift_i) & 1
        val_j = (k >> shift_j) & 1

        if val_i == val_j:
            # No change if bits are identical (both 0 or both 1)
            cols[k] = k
        else:
            # Flip both bits to perform the swap
            # XOR with (1 << pos) flips the bit at that position
            new_k = k ^ (1 << shift_i) ^ (1 << shift_j)
            cols[k] = new_k

    # Create the sparse permutation matrix
    data = np.ones(size)
    sp_matrix = sparse.csr_matrix((data, (rows, cols)), shape=(size, size))
    
    return TransitionMatrix(sp_matrix)

def permutation(n, mapping):
    """
    Returns a 2^n x 2^n TransitionMatrix that rearranges bits according to the mapping.
    
    mapping: A list of length n where mapping[i] is the destination index for bit i.
    Example: n=3, mapping=[2, 0, 1] 
             Bit 0 -> Pos 2
             Bit 1 -> Pos 0
             Bit 2 -> Pos 1
    """
    
    # 1. Validation
    if not isinstance(n, int) or n < 1:
        raise ValueError("n must be a positive integer.")
    
    if not isinstance(mapping, (list, tuple, np.ndarray)):
        raise TypeError("mapping must be a list or array-like.")
        
    if len(mapping) != n:
        raise ValueError(f"Mapping length ({len(mapping)}) must match n ({n}).")
        
    # Check if mapping is a valid permutation (contains all indices from 0 to n-1)
    if sorted(mapping) != list(range(n)):
        raise ValueError("mapping must be a valid permutation of indices from 0 to n-1 (no duplicates/missing indices).")

    # 2. Logic
    size = 2**n
    rows = np.arange(size)
    cols = np.zeros(size, dtype=int)

    # Pre-calculate the shifts to avoid doing (n - 1 - i) inside the loop
    # source_shifts[i] is the shift needed to extract bit i
    # target_shifts[i] is the shift needed to place bit i at mapping[i]
    source_shifts = [n - 1 - i for i in range(n)]
    target_shifts = [n - 1 - mapping[i] for i in range(n)]

    for k in range(size):
        new_k = 0
        for i in range(n):
            # Extract bit at source position i
            bit_val = (k >> source_shifts[i]) & 1
            # Place it at the target position defined by mapping[i]
            new_k |= (bit_val << target_shifts[i])
        cols[k] = new_k

    # 3. Create Sparse Matrix
    data = np.ones(size)
    sp_matrix = sparse.csr_matrix((data, (rows, cols)), shape=(size, size))
    
    return TransitionMatrix(sp_matrix)

def random_qubit():
    """Generate a random normalized qubit state (left-to-right notation)."""
    # Generate two random complex numbers (real and imaginary parts)
    z0 = np.random.randn() + 1j * np.random.randn()
    z1 = np.random.randn() + 1j * np.random.randn()

    # Combine into a vector
    psi = np.array([[z0, z1]])

    # Normalize
    psi = psi / np.linalg.norm(psi)
    
    return psi

#
# Common Transition matrices definitions
#

# Molecular elements

zero_out = TransitionMatrix([
    [1, 0]
])

zero_in = zero_out.reverse()

one_out = TransitionMatrix([
    [0, 1]
])

one_in = one_out.reverse()

random = TransitionMatrix([
    [1/2, 1/2]
])

end = TransitionMatrix([
    [1],
    [1]
])

identity = TransitionMatrix([
    [1, 0],
    [0, 1]
])

negate = TransitionMatrix([
    [0, 1],
    [1, 0]
])

fork = TransitionMatrix([
    [0, 1/2, 1/2, 0],
    [1, 0  , 0  , 0]
])

merge = fork.reverse()

swap = TransitionMatrix([
    [1, 0, 0, 0],
    [0, 0, 1, 0],
    [0, 1, 0, 0],
    [0, 0, 0, 1]
])

terminate = TransitionMatrix([
    [0],
    [1],
    [1],
    [0]
])

coupler = TransitionMatrix([
    [1, 0, 0, 0],
    [0, 0, 0, 0],
    [0, 0, 0, 0],
    [0, 0, 0, 1]
])

contravariant = terminate.reverse()

jumper = TransitionMatrix([
    [1],
    [0],
    [0],
    [1]
])

entangle = jumper.reverse()

river = TransitionMatrix([
    [1/2, 0, 0, 1/2],
    [0  , 0, 1, 0],
    [0  , 1, 0, 0],
    [1  , 0, 0, 0]
])

bridge = TransitionMatrix([
    [1, 0  , 0  , 0],
    [0, 1/2, 1/2, 0],
    [0, 1/2, 1/2, 0],
    [0, 0  , 0  , 0]
])

converge = TransitionMatrix([
    [0], 
    [1],
    [1], 
    [0], 
    [1], 
    [0], 
    [0], 
    [0]
])

diverge = converge.reverse()

cut = TransitionMatrix([
    [1/2, 1/2],
    [1/2, 1/2]
])

join = TransitionMatrix([
    [1, 0],
    [0, 0],
    [0, 0],
    [0, 1]
])

split = join.reverse()


# Logic gates

not_q = TransitionMatrix([
    [0, 1],
    [1, 0]
])

and_q = TransitionMatrix([
    [1, 0],
    [1, 0],
    [1, 0],
    [0, 1]
])

nand_q = and_q @ not_q

or_q = TransitionMatrix([
    [1, 0],
    [0, 1],
    [0, 1],
    [0, 1]
])

nor_q = or_q @ not_q

xor_q = TransitionMatrix([
    [1, 0],
    [0, 1],
    [0, 1],
    [1, 0]
])

triple_xor_gate = (xor_q ^ identity) @ xor_q

# This is the easy way to define XNOR
#xnor_q = xor_q @ not_q

# This is the hard way to define NXOR. We do it here only to prove that the molecule theory works
xnor_q = ((fork ^ fork) @ (identity ^ jumper ^ identity) @ merge).normalize()

# Reversed logic gates

not_rev_q = not_q

and_rev_q = and_q.reverse()
nand_rev_q = nand_q.reverse()

or_rev_q = or_q.reverse()
nor_rev_q = nor_q.reverse()

xor_rev_q = xor_q.reverse()
xnor_rev_q = xnor_q.reverse()

# Quaternions

i_multiply = (split ^ identity ^ split) @ (identity ^ triple_xor_gate ^ negate)
j_multiply = (split ^ identity ^ identity) @ (negate ^ xor_q ^ identity)
k_multiply = (identity ^ identity ^ split) @ (negate ^ xor_q ^ negate)
neg_multiply = (identity ^ negate ^ identity)
i_ident_multiply = (identity ^ identity ^ split) @ (identity ^ xor_q ^ negate)

# Qubits