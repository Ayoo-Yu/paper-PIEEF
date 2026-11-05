"""Export the final editable schematic source as standard SVG.

The supplied EMF/PNG paper exports remain in figures/paper. SVG export requires
only the Python standard library; optional raster conversion is not required.
"""
from pathlib import Path
import xml.etree.ElementTree as ET
ROOT=Path(__file__).resolve().parents[2]
def render(item):
 source=ROOT/'figures/source'/f'{item}.svg'
 ET.parse(source)
 dest=ROOT/'figures/reproduced'/f'{item}.svg';dest.parent.mkdir(parents=True,exist_ok=True)
 dest.write_bytes(source.read_bytes())
 return [str(dest.relative_to(ROOT))]
if __name__=='__main__':
 import argparse
 p=argparse.ArgumentParser();p.add_argument('--item',required=True);print(render(p.parse_args().item))
