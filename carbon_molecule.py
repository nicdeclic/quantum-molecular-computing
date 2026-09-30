"""
CarbonMolecule: A framework for modeling, manipulating, and rendering 
conjugated carbon molecular networks with custom isotopes, bonds, and couplers.
"""

from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Dict
import math
import numpy as np

import schemdraw
import schemdraw.elements as elm


@dataclass
class Atom:
    """Represents a carbon atom node."""
    x: float
    y: float
    visible: bool = True
    isotope: str = "C12"  # 'C12' or 'C13'
    label: str = ""

    def pos(self) -> Tuple[float, float]:
        return (self.x, self.y)


@dataclass
class Bond:
    """
    Represents a chemical bond (protobit) between two atoms.
    Order: 'single' (0) or 'double' (1)
    State type: 'floating' (resonating) or 'fixed'
    Role: 'input', 'output', or 'intermediate'
    """
    atom1: Atom
    atom2: Atom
    ctrl1: Optional[Tuple[float, float]] = None  # Bézier control point 1
    ctrl2: Optional[Tuple[float, float]] = None  # Bézier control point 2
    order: str = "single"                        # 'single' or 'double'
    state_type: str = "floating"                 # 'floating' or 'fixed'
    role: str = "intermediate"                   # 'input', 'output', 'intermediate'
    label: str = ""
    label_pos: Optional[Tuple[float, float]] = None

    def get_midpoint(self) -> Tuple[float, float]:
        """Calculates default midpoint between the two connected atoms."""
        return ((self.atom1.x + self.atom2.x) / 2.0, (self.atom1.y + self.atom2.y) / 2.0)


@dataclass
class Coupler:
    """Represents an interaction or coupling between two bonds."""
    bond1: Bond
    bond2: Bond
    label: str = ""


class CarbonMolecule:
    def __init__(self, name: str = "CarbonNetwork"):
        self.name = name
        self.atoms: List[Atom] = []
        self.bonds: List[Bond] = []
        self.couplers: List[Coupler] = []

    # ==========================================
    # ATOM MANAGEMENT
    # ==========================================
    def add_atom(self, x: float, y: float, isotope: str = "C12", 
                 visible: bool = True, label: str = "") -> Atom:
        """Creates and appends a new carbon atom."""
        if isotope not in ("C12", "C13"):
            raise ValueError("Isotope must be 'C12' or 'C13'.")
        atom = Atom(x=x, y=y, visible=visible, isotope=isotope, label=label)
        self.atoms.append(atom)
        return atom

    def remove_atom(self, atom: Atom) -> None:
        """Removes an atom and any cascading connected bonds and couplers."""
        if atom in self.atoms:
            # Remove associated bonds
            connected_bonds = [b for b in self.bonds if b.atom1 == atom or b.atom2 == atom]
            for b in connected_bonds:
                self.remove_bond(b)
            self.atoms.remove(atom)

    def find_atom(self, label: str) -> Optional[Atom]:
        """Finds the first atom matching the given label."""
        for a in self.atoms:
            if a.label == label:
                return a
        return None

    # ==========================================
    # BOND MANAGEMENT
    # ==========================================
    def add_bond(self, atom1: Atom, atom2: Atom, order: str = "single",
                 state_type: str = "floating", role: str = "intermediate",
                 ctrl1: Optional[Tuple[float, float]] = None,
                 ctrl2: Optional[Tuple[float, float]] = None,
                 label: str = "", label_pos: Optional[Tuple[float, float]] = None) -> Bond:
        """Connects two atoms with a bond."""
        if atom1 not in self.atoms or atom2 not in self.atoms:
            raise ValueError("Both atoms must belong to this molecule.")
        if order not in ("single", "double"):
            raise ValueError("Order must be 'single' or 'double'.")
        if role not in ("input", "output", "intermediate"):
            raise ValueError("Role must be 'input', 'output', or 'intermediate'.")

        bond = Bond(
            atom1=atom1, atom2=atom2, ctrl1=ctrl1, ctrl2=ctrl2,
            order=order, state_type=state_type, role=role,
            label=label, label_pos=label_pos
        )
        self.bonds.append(bond)
        return bond

    def remove_bond(self, bond: Bond) -> None:
        """Removes a bond and any couplers attached to it."""
        if bond in self.bonds:
            connected_couplers = [c for c in self.couplers if c.bond1 == bond or c.bond2 == bond]
            for c in connected_couplers:
                self.remove_coupler(c)
            self.bonds.remove(bond)

    def find_bond(self, label: str) -> Optional[Bond]:
        """Finds the first bond matching the given label."""
        for b in self.bonds:
            if b.label == label:
                return b
        return None

    # ==========================================
    # COUPLER MANAGEMENT
    # ==========================================
    def add_coupler(self, bond1: Bond, bond2: Bond, label: str = "") -> Coupler:
        """Creates an interaction link between two bonds."""
        if bond1 not in self.bonds or bond2 not in self.bonds:
            raise ValueError("Both bonds must exist in this molecule.")
        coupler = Coupler(bond1=bond1, bond2=bond2, label=label)
        self.couplers.append(coupler)
        return coupler

    def remove_coupler(self, coupler: Coupler) -> None:
        """Removes a coupler."""
        if coupler in self.couplers:
            self.couplers.remove(coupler)

    def find_coupler(self, label: str) -> Optional[Coupler]:
        """Finds the first coupler matching the given label."""
        for c in self.couplers:
            if c.label == label:
                return c
        return None

    # ==========================================
    # CLONING & OFFSET MERGE
    # ==========================================
    def copy_from(self, other: "CarbonMolecule", offset: Tuple[float, float] = (0.0, 0.0)) -> None:
        """
        Copies all atoms, bonds, and couplers from another molecule into this one,
        applying a 2D translational offset.
        """
        dx, dy = offset
        atom_map: Dict[Atom, Atom] = {}

        # 1. Replicate atoms with offset
        for a in other.atoms:
            new_atom = self.add_atom(
                x=a.x + dx, y=a.y + dy,
                isotope=a.isotope, visible=a.visible, label=a.label
            )
            atom_map[a] = new_atom

        # 2. Replicate bonds
        bond_map: Dict[Bond, Bond] = {}
        for b in other.bonds:
            c1 = (b.ctrl1[0] + dx, b.ctrl1[1] + dy) if b.ctrl1 else None
            c2 = (b.ctrl2[0] + dx, b.ctrl2[1] + dy) if b.ctrl2 else None
            lpos = (b.label_pos[0] + dx, b.label_pos[1] + dy) if b.label_pos else None

            new_bond = self.add_bond(
                atom1=atom_map[b.atom1], atom2=atom_map[b.atom2],
                order=b.order, state_type=b.state_type, role=b.role,
                ctrl1=c1, ctrl2=c2, label=b.label, label_pos=lpos
            )
            bond_map[b] = new_bond

        # 3. Replicate couplers
        for c in other.couplers:
            self.add_coupler(
                bond1=bond_map[c.bond1],
                bond2=bond_map[c.bond2],
                label=c.label
            )

    # ==========================================
    # AUTOMATIC LAYOUT (OPTIONAL HELPER)
    # ==========================================
    def auto_layout(self, scale: float = 2.5) -> None:
        """
        Calculates aesthetic 2D positions for atoms using a force-directed layout.
        Requires networkx if used, with an analytic geometric fallback.
        """
        try:
            import networkx as nx
            G = nx.Graph()
            for idx, a in enumerate(self.atoms):
                G.add_node(idx)
            for b in self.bonds:
                G.add_edge(self.atoms.index(b.atom1), self.atoms.index(b.atom2))

            positions = nx.spring_layout(G, scale=scale, seed=42)
            for idx, a in enumerate(self.atoms):
                a.x, a.y = positions[idx][0], positions[idx][1]
        except ImportError:
            # Fallback ring placement if networkx is absent
            n = len(self.atoms)
            if n == 0:
                return
            for i, a in enumerate(self.atoms):
                angle = 2 * math.pi * i / n
                a.x = scale * math.cos(angle)
                a.y = scale * math.sin(angle)

    # ==========================================
    # SCHEMDRAW VISUALIZATION
    # ==========================================
    def draw(self, auto_layout: bool = False, scale: float = 1.0, margin: float = 0.12, ax=None):
        """
        Renders the molecule in compact, crisp vector graphics using Matplotlib.
        - C12: Small solid black circle with white centered label.
        - C13: Small Bullseye (Black outer, white ring, black center).
        - Single bond: Thin black line.
        - Double bond: Thick black line.
        - Couplers: Centered dashed blue line between the two bonds.
        """
        import matplotlib.pyplot as plt
        import matplotlib.patches as patches
        from matplotlib.path import Path

        if auto_layout:
            self.auto_layout()

         # Calculate the bounding box of the molecule
        if self.atoms:
            xs = [a.x for a in self.atoms]
            ys = [a.y for a in self.atoms]
            min_x, max_x = min(xs) - margin, max(xs) + margin
            min_y, max_y = min(ys) - margin, max(ys) + margin
        else:
            min_x, max_x, min_y, max_y = -1, 1, -1, 1

        width_units = max_x - min_x
        height_units = max_y - min_y

        # Automatically compute canvas size (figsize) to keep atoms constant in size
        # 1 unit in coordinate space = 'scale' inches on screen
        figsize = (width_units * scale, height_units * scale)

        if ax is None:
            fig, ax = plt.subplots(figsize=figsize)
            should_show = True
        else:
            should_show = False

        ax.set_aspect("equal")
        ax.axis("off")

        # Set strict limits based on the computed box
        ax.set_xlim(min_x, max_x)
        ax.set_ylim(min_y, max_y)

        # 1. Draw Bonds
        for b in self.bonds:
            lw = 1.5 if b.order == "single" else 6.0
            p1 = (b.atom1.x, b.atom1.y)
            p2 = (b.atom2.x, b.atom2.y)

            if b.ctrl1 and b.ctrl2:
                # Cubic Bézier curve
                verts = [p1, b.ctrl1, b.ctrl2, p2]
                codes = [Path.MOVETO, Path.CURVE4, Path.CURVE4, Path.CURVE4]
                path = Path(verts, codes)
                patch = patches.PathPatch(path, facecolor="none", edgecolor="black", lw=lw, zorder=1)
                ax.add_patch(patch)
            else:
                # Straight line bond
                ax.plot([p1[0], p2[0]], [p1[1], p2[1]], color="black", lw=lw, zorder=1)

            # Bond label
            if b.label:
                l_pos = b.label_pos if b.label_pos else b.get_midpoint()
                ax.text(l_pos[0], l_pos[1], b.label, color="black", fontsize=10,
                        ha="center", va="center", zorder=4,
                        bbox=dict(boxstyle="round,pad=0.15", facecolor="white", edgecolor="none", alpha=0.8))

        # 2. Draw Couplers (Blue dashed line between bond midpoints)
        for c in self.couplers:
            m1 = np.array(c.bond1.get_midpoint())
            m2 = np.array(c.bond2.get_midpoint())

            center = (m1 + m2) / 2.0
            diff = m2 - m1
            dist = np.linalg.norm(diff)

            if dist > 1e-4:
                normal = np.array([-diff[1], diff[0]]) / dist
            else:
                normal = np.array([0.0, 1.0])

            p_start = center - 0.25 * normal
            p_end = center + 0.25 * normal

            ax.plot([p_start[0], p_end[0]], [p_start[1], p_end[1]],
                    color="blue", lw=1.5, linestyle="--", zorder=2)

            if c.label:
                lbl_pos = center + 0.15 * normal
                ax.text(lbl_pos[0], lbl_pos[1], c.label, color="blue", fontsize=7,
                        ha="center", va="center", zorder=4)

        # 3. Draw Atoms
        atom_radius = 0.1
        for a in self.atoms:
            if not a.visible:
                continue

            if a.isotope == "C12":
                # Solid black circle
                circle = patches.Circle((a.x, a.y), atom_radius, facecolor="black", edgecolor="black", zorder=3)
                ax.add_patch(circle)
            else:
                # C13: Bullseye (Outer black, white ring, inner black core)
                c_outer = patches.Circle((a.x, a.y), atom_radius, facecolor="black", edgecolor="black", zorder=3)
                c_mid   = patches.Circle((a.x, a.y), atom_radius * 0.85, facecolor="white", edgecolor="white", zorder=3)
                c_inner = patches.Circle((a.x, a.y), atom_radius * 0.70, facecolor="black", edgecolor="black", zorder=3)
                ax.add_patch(c_outer)
                ax.add_patch(c_mid)
                ax.add_patch(c_inner)

            # Centered white label
            if a.label:
                ax.text(a.x, a.y, a.label, color="white", fontsize=10,
                        ha="center", va="center", weight="bold", zorder=4)

        # 3. Final display without distorting the bounding box
        fig = ax.figure
        fig.subplots_adjust(left=0, right=1, bottom=0, top=1)

        plt.show()