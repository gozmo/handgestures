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

def render_annotate(request):
    if request.method == "POST":
        items = request.form.to_dict().items()
        (image_path, label) = list(items)[0]
        file_utils.move_image_to_label(Directories.UNLABEL + image_path, label)

    images = []

    for root, dirs, files in os.walk(Directories.UNLABEL):
        files = sorted(files)
        files = files[0:1]
        for filename, name in [(os.path.join(root, name), name) for name in files]:
            if not filename.endswith(".jpg"):
                continue
            im = Image.open(filename)
            w, h = im.size
            aspect = 1.0 * w / h
            if aspect > 1.0 * WIDTH / HEIGHT:
                width = min(w, WIDTH)
                height = width / aspect
            else:
                height = min(h, HEIGHT)
                width = height * aspect
            image = SourceImage(name, int(width), int(height))
            images.append(image)

    return render_template(TemplateFiles.ANNOTATE, images=images, labels=Labels.get_labels())

def render_object_annotation(request):
    images = []

    for root, dirs, files in os.walk(Directories.UNLABEL):
        files = sorted(files)
        files = files[0:1]
        for filename, name in [(os.path.join(root, name), name) for name in files]:
            if not filename.endswith(".jpg"):
                continue
            im = Image.open(filename)
            w, h = im.size
            aspect = 1.0 * w / h
            if aspect > 1.0 * WIDTH / HEIGHT:
                width = min(w, WIDTH)
                height = width / aspect
            else:
                height = min(h, HEIGHT)
                width = height * aspect
            image = SourceImage(name, int(width), int(height))
            images.append(image)

    return render_template(TemplateFiles.ANNOTATE_OBJECT, image=images[0], labels=Labels.get_labels())
