import os
from PIL import Image
from flask import Flask, Response, request, abort, render_template_string, send_from_directory,render_template
from io import StringIO

from handsignals.dataset import file_utils
WIDTH = 640
HEIGHT = 400

LABELS = ["none", "metal", "ok", "victory"]
def render_annotate(request):
    print("render_annotate")
    if request.method == "POST":
        items = request.form.to_dict().items()
        (image_path, label) = list(items)[0]
        file_utils.move_image_to_label(image_path, label)

    images = []
    path = "dataset/unlabeled"

    for root, dirs, files in os.walk(path):
        for filename, name in [(os.path.join(root, name), name) for name in files]:
            if not filename.endswith('.jpg'):
                continue
            im = Image.open(filename)
            w, h = im.size
            aspect = 1.0*w/h
            if aspect > 1.0*WIDTH/HEIGHT:
                width = min(w, WIDTH)
                height = width/aspect
            else:
                height = min(h, HEIGHT)
                width = height*aspect
            print("render_annotate", filename)
            images.append({
                'width': int(width),
                'height': int(height),
                'src': name
            })

    return render_template("data/annotate.html",
                                  images=images,
                                  labels=LABELS)
