from flask import Flask, Response, request, abort, render_template_string, send_from_directory,render_template
from PIL import Image
import os

def capture(frames_to_capture):
    pass

def read_images(filepath, request):
    print("read_images:", filepath)

    try:
        im = Image.open(filepath)
        im.thumbnail((w, h), Image.ANTIALIAS)
        io = StringIO.StringIO()
        im.save(io, format='JPEG')
        return Response(io.getvalue(), mimetype='image/jpeg')

    except IOError:
        abort(404)

    filename = os.path.basename(filepath)
    folder_path = os.dirname(filepath)

    return send_from_directory(folder_path, filename)
