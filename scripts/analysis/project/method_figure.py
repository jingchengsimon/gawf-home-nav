"""Extract supplied PDF artwork and annotate EmbCLIP Figure 2 for the web page.

Existing cell/architecture paths are imported from the PDFs, never redrawn.
Only Model labels, the input photograph and the GaWF feedback arrow are edited.
"""
import argparse
from copy import deepcopy
import hashlib
from io import BytesIO
import json
from pathlib import Path
import subprocess

import numpy as np
from PIL import Image
from pypdf import PdfReader, PdfWriter, Transformation
from pypdf.generic import RectangleObject
from reportlab.pdfgen import canvas
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont

ROOT = Path(__file__).resolve().parents[3]
DIST = ROOT / 'project-page/dist'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def import_crop(target, source, bounds, x, y, scale):
    left, top, right, bottom = bounds
    page = deepcopy(source)
    lower = float(page.mediabox.height) - bottom
    upper = float(page.mediabox.height) - top
    page.cropbox = RectangleObject((left, lower, right, upper))
    transform = Transformation().translate(-left, -lower).scale(scale).translate(x, y)
    target.merge_transformed_page(page, transform)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cell-pdf', type=Path, required=True)
    parser.add_argument('--paper-pdf', type=Path, required=True)
    parser.add_argument('--frame', type=Path, required=True)
    parser.add_argument('--work', type=Path, required=True)
    args = parser.parse_args()
    args.work.mkdir(parents=True, exist_ok=True)
    frame_source = ROOT / 'results/nav-robothor-embclip-visible-gradcam-selection-local-v2-20261023'
    frame_record = next(r for r in json.loads((frame_source / 'summary.json').read_text())['records']
                        if r['target'] == 'Laptop')['models']['gawf1']
    raw_frame = frame_source / 'Laptop_gawf1_rgb.npy'
    assert sha(raw_frame) == frame_record['rgb_sha256']
    assert np.array_equal(np.asarray(Image.open(args.frame)),
                          np.round(np.load(raw_frame) * 255).astype('uint8'))
    font = DIST / 'assets/fonts/noto-sans-400.ttf'
    pdfmetrics.registerFont(TTFont('NotoSans', str(font)))
    pdfmetrics.registerFont(TTFont('NotoSansBold', str(DIST / 'assets/fonts/noto-sans-700.ttf')))
    cell = PdfReader(args.cell_pdf).pages[0]
    architecture = PdfReader(args.paper_pdf).pages[2]
    writer = PdfWriter()
    output = writer.add_blank_page(width=1000, height=350)
    cell_bounds = [5, 95, 164, 310]
    architecture_bounds = [49, 76, 287, 184]
    cell_scale = 1.36
    architecture_scale = 365 / 238
    architecture_y = 170 - (184 - 76) * architecture_scale / 2
    import_crop(output, cell, cell_bounds, 4, 20, cell_scale)
    for x in (255, 635):
        import_crop(output, architecture, architecture_bounds, x, architecture_y, architecture_scale)

    buffer = BytesIO()
    overlay = canvas.Canvas(buffer, pagesize=(1000, 350))
    overlay.setFillColorRGB(0, 0, 0)
    overlay.setFont('NotoSans', 15)
    for x, label in ((112, 'GaWF cell'), (437.5, 'EmbCLIP'), (817.5, 'EmbCLIP-GaWF')):
        overlay.setFont('NotoSans' if label == 'GaWF cell' else 'NotoSansBold', 15)
        overlay.drawCentredString(x, 330, label)
    for x, label in ((255, 'GRU'), (635, 'GaWF')):
        overlay.saveState()
        overlay.translate(x, architecture_y)
        overlay.scale(architecture_scale, architecture_scale)
        overlay.translate(-49, -(792 - 184))
        # Cover the original Model word with its exact PDF rectangle color.
        overlay.setFillColorRGB(.839, .863, .898)
        overlay.rect(204.5, 792 - 145, 31, 15, fill=1, stroke=0)
        overlay.setFillColorRGB(0, 0, 0)
        overlay.setFont('NotoSans', 10.2)
        overlay.drawCentredString(220.0177, 792 - 141.3, label)
        # Use the same actual RoboTHOR Laptop frame for both architectures.
        overlay.drawImage(str(args.frame), 57.5539884, 792 - 142.7561532,
                          width=41.7775116, height=41.7775116)
        if label == 'GaWF':
            # Actor output at t-1 is detached and fed to the recurrent gates.
            start, end, height = 259.31456, 241.67429, 792 - 121.5
            overlay.setStrokeColorRGB(.57, .26, .60)
            overlay.setFillColorRGB(.57, .26, .60)
            overlay.setLineWidth(1.1)
            overlay.line(start, height, end + 2.4, height)
            arrow = overlay.beginPath()
            arrow.moveTo(end, height)
            arrow.lineTo(end + 3, height + 1.9)
            arrow.lineTo(end + 3, height - 1.9)
            arrow.close()
            overlay.drawPath(arrow, fill=1, stroke=0)
            overlay.setFont('NotoSans', 6.8)
            overlay.drawCentredString((start + end) / 2, height + 3.6, 't-1')
        overlay.restoreState()
    overlay.save()
    output.merge_page(PdfReader(buffer).pages[0])
    pdf = args.work / 'method-overview.pdf'
    with pdf.open('wb') as stream:
        writer.write(stream)
    image = DIST / 'assets/method-overview.png'
    subprocess.run(['pdftoppm', '-singlefile', '-r', '240', '-png', str(pdf),
                    str(image.with_suffix(''))], check=True)
    receipt = dict(
        panels=['Unchanged extracted GaWF cell', 'EmbCLIP Figure 2 with GRU label',
                'EmbCLIP Figure 2 with GaWF label and delayed policy feedback'],
        source_cell=dict(filename=args.cell_pdf.name, page=1, crop_top_origin_pt=cell_bounds,
                         sha256=sha(args.cell_pdf), internal_structure_edited=False),
        source_architecture=dict(url='https://arxiv.org/pdf/2111.09888', figure=2, page=3,
                                 crop_top_origin_pt=architecture_bounds, sha256=sha(args.paper_pdf)),
        frame=dict(target='Laptop', source='Laptop_gawf1_rgb.npy', png_sha256=sha(args.frame),
                   source_array_sha256=sha(raw_frame), episode_id=frame_record['episode_id'],
                   identical_in_both_architecture_panels=True),
        edits=['Model to GRU or GaWF', 'Original input photo to recorded RoboTHOR RGB',
               'GaWF-only arrow from actor output toward recurrent gates, marked t-1'],
        feedback='Previous detached raw policy logits, not sampled actions or privileged observations.',
        model_parameters_modified=False, image_sha256=sha(image),
    )
    for folder in (ROOT / 'project-page/sources', DIST / 'assets/sources'):
        (folder / 'method_figure.json').write_text(json.dumps(receipt, indent=2) + '\n')
    print(image)


if __name__ == '__main__':
    main()
