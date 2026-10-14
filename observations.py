"""Convert only explicit molar MIC observations to log10 micromolar space."""
import math
import re
from dataclasses import dataclass


@dataclass(frozen=True)
class Observation:
    lower: float
    upper: float
    operator: str


def parse_mic(text):
    normalized = text.strip().replace('μ','u').replace('µ','u').replace('≥','>=').replace('≤','<=')
    match = re.fullmatch(r'(>=|<=|>|<|=)?\s*([0-9]+(?:\.[0-9]+)?)\s*(nM|uM|mM|M)',normalized)
    if not match: raise ValueError('Only explicit positive molar observations are supported')
    operator,raw,unit = match.groups(); operator = operator or '='
    value = float(raw)*{'nM':.001,'uM':1.,'mM':1000.,'M':1e6}[unit]
    if value <= 0: raise ValueError('MIC must be positive')
    threshold = math.log10(value)
    return Observation(threshold if operator not in ['<','<='] else -math.inf,
                       threshold if operator not in ['>','>='] else math.inf,operator)
