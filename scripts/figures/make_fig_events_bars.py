"""Reproduce fig 13 from the public frozen source values.

The shared implementation keeps numerical values and reported statistics fixed.
The original paper export is retained separately in figures/paper.
"""
from public_figures import render
if __name__ == '__main__':
    print(render('fig_13'))
