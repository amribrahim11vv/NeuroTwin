# -*- coding: utf-8 -*-
"""Scan and fix all non-ASCII characters in .py files."""
import glob
import sys

# Force stdout to utf-8
sys.stdout.reconfigure(encoding='utf-8')

REPLACEMENTS = {
    '\u2714': '[OK]',    # heavy check mark
    '\u2705': '[OK]',    # white heavy check mark
    '\u26a0': '[!]',     # warning sign
    '\u25b6': '>',       # right-pointing triangle
    '\u2014': ' - ',     # em dash
    '\u2013': '-',       # en dash
    '\u2500': '-',       # box drawing horizontal
    '\u2192': '->',      # rightwards arrow
    '\u2190': '<-',      # leftwards arrow
    '\u2198': 'v',       # south east arrow
    '\u03b1': 'alpha',   # greek alpha
    '\u03b2': 'beta',    # greek beta
    '\u03b3': 'gamma',   # greek gamma
    '\u03b4': 'delta',   # greek delta (lowercase)
    '\u0394': 'Delta',   # greek Delta (uppercase)
    '\u03b5': 'epsilon', # greek epsilon
    '\u03b6': 'zeta',    # greek zeta
    '\u03b7': 'eta',     # greek eta
    '\u03b8': 'theta',   # greek theta
    '\u03bb': 'lambda_', # greek lambda
    '\u03bc': 'mu',      # greek mu
    '\u03c0': 'pi',      # greek pi
    '\u03c3': 'sigma',   # greek sigma (lowercase)
    '\u03a3': 'Sigma',   # greek Sigma (uppercase)
    '\u03c4': 'tau',     # greek tau
    '\u03c6': 'phi',     # greek phi
    '\u03c8': 'psi',     # greek psi
    '\u00b1': '+/-',     # plus-minus sign
    '\u2264': '<=',      # less-than or equal to
    '\u2265': '>=',      # greater-than or equal to
    '\u2260': '!=',      # not equal to
    '\u2208': 'in',      # element of
    '\u2209': 'not in',  # not element of
    '\u221e': 'inf',     # infinity
    '\u2211': 'sum',     # summation
    '\u220f': 'prod',    # product
    '\u2248': '~=',      # almost equal to
    '\u00d7': '*',       # multiplication sign
    '\u2591': '#',       # light shade
    '\u25ae': '|',       # black vertical rectangle
    '\u2026': '...',     # horizontal ellipsis
    '\u201c': '"',       # left double quotation mark
    '\u201d': '"',       # right double quotation mark
    '\u2018': "'",       # left single quotation mark
    '\u2019': "'",       # right single quotation mark
    '\u00b2': '^2',      # superscript two
    '\u00b3': '^3',      # superscript three
    '\u2070': '^0',      # superscript zero
    '\u00b9': '^1',      # superscript one
    '\u2074': '^4',      # superscript four
    '\u2022': '*',       # bullet
    '\u211d': 'R',       # double-struck capital R (real numbers)
    '\u00b7': '*',       # middle dot (multiplication)
    '\u2212': '-',       # minus sign
}

files = glob.glob('*.py')
total_fixed = 0

for f in files:
    if f == 'fix_unicode.py':
        continue
    with open(f, 'r', encoding='utf-8') as fh:
        content = fh.read()
    
    original = content
    for old, new in REPLACEMENTS.items():
        content = content.replace(old, new)
    
    if content != original:
        # Find which lines changed
        orig_lines = original.split('\n')
        new_lines = content.split('\n')
        for i, (ol, nl) in enumerate(zip(orig_lines, new_lines)):
            if ol != nl:
                print(f"  {f}:{i+1}: FIXED")
                total_fixed += 1
        
        with open(f, 'w', encoding='utf-8') as fh:
            fh.write(content)
        print(f"  -> {f} saved")

# Second pass: check for any remaining non-ASCII
print(f"\n--- Second pass: checking for remaining non-ASCII ---")
remaining = 0
for f in files:
    if f == 'fix_unicode.py':
        continue
    with open(f, 'r', encoding='utf-8') as fh:
        for i, line in enumerate(fh.readlines()):
            non_ascii = [c for c in line if ord(c) > 127]
            if non_ascii:
                chars = ', '.join(f'U+{ord(c):04X}' for c in non_ascii)
                print(f"  {f}:{i+1}: remaining non-ASCII: {chars}")
                remaining += 1

print(f"\nTotal lines fixed: {total_fixed}")
print(f"Remaining non-ASCII lines: {remaining}")
