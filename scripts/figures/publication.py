"""Export publication figures with editable text and measured label bounds."""
from pathlib import Path
import argparse
import json
import xml.etree.ElementTree as ET

from matplotlib.text import Text
from matplotlib.transforms import Bbox
from PIL import Image


def export_publication(fig, default_output, name):
    parser = argparse.ArgumentParser(description=f'Render {name} from its source data.')
    parser.add_argument('--output', type=Path, default=default_output)
    parser.add_argument('--check-dir', type=Path)
    args = parser.parse_args()
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=True)
    fig.set_dpi(600)
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    mm = 25.4 / fig.dpi
    boxes, margins = [], []
    for item in fig.findobj(Text):
        if not item.get_visible() or not item.get_text().strip():
            continue
        item.set_clip_on(False)
        box = Text.get_window_extent(item, renderer)
        if not box.width or not box.height:
            continue
        assert item.get_fontsize() >= 10.5 - 1e-8, item.get_text()
        margin = min(box.x0, box.y0, fig.bbox.width - box.x1, fig.bbox.height - box.y1) * mm
        assert margin >= 2.9, (item.get_text(), margin)
        margins.append(margin)
        boxes.append((item, box))
    overlaps = []
    for i, (first, a) in enumerate(boxes):
        for second, b in boxes[i + 1:]:
            intersection = Bbox.intersection(a, b)
            if intersection is not None and intersection.width * mm > .35 and intersection.height * mm > .35:
                overlaps.append([first.get_text(), second.get_text()])
    assert not overlaps, overlaps
    fig.savefig(out / (name + '.png'), dpi=600)
    fig.savefig(out / (name + '.svg'), metadata={'Date': None, 'Creator': 'Python/matplotlib'})
    fig.savefig(out / (name + '.pdf'), metadata={'Title': name, 'Creator': 'Python/matplotlib'})
    with Image.open(out / (name + '.png')) as image:
        pixels = list(image.size)
        image.convert('RGB').save(out / (name + '.tif'), compression='tiff_lzw', dpi=(600, 600))
    svg = ET.parse(out / (name + '.svg')).getroot()
    editable = len(svg.findall('.//{http://www.w3.org/2000/svg}text'))
    assert editable > 30
    report = dict(figure=name, canvas_mm=(fig.get_size_inches() * 25.4).round(6).tolist(),
                  png_pixels=pixels, dpi=600, minimum_font_pt=10.5,
                  minimum_text_margin_mm=min(margins), editable_svg_text_elements=editable,
                  text_overlap_candidates=overlaps)
    if args.check_dir:
        import resvg_py
        args.check_dir.mkdir(parents=True, exist_ok=True)
        (args.check_dir / (name + '.json')).write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
        (args.check_dir / (name + '_svg.png')).write_bytes(resvg_py.svg_to_bytes(
            svg_path=str(out / (name + '.svg')), width=1600, dpi=96))
    print(name + ': PNG, SVG, PDF and TIFF exported', flush=True)
