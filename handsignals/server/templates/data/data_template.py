import os
from PIL import Image
from flask import (
    Flask,
    Response,
    request,
    abort,
    render_template_string,
    send_from_directory,
    render_template,
)
from io import StringIO

from handsignals.dataset import file_utils
from handsignals.constants import Labels
from handsignals.constants import Directories
from handsignals.constants import TemplateFiles
from handsignals.core.types.source_image import SourceImage

WIDTH = 640
HEIGHT = 400

def render_annotation(request):
    images = []

    unlabeled_files = os.listdir(Directories.UNLABEL)
    unlabeled_files = sorted(unlabeled_files)
    first_file = unlabeled_files[0]
    image_path = os.path.join(Directories.UNLABEL, first_file)

    im = Image.open(image_path)
    w, h = im.size
    aspect = 1.0 * w / h
    if aspect > 1.0 * WIDTH / HEIGHT:
        width = min(w, WIDTH)
        height = width / aspect
    else:
        height = min(h, HEIGHT)
        width = height * aspect
    image = SourceImage(image_path, int(width), int(height))

    return render_template(TemplateFiles.ANNOTATE, image=image)
