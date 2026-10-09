from IPython.display import display, Latex, SVG

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