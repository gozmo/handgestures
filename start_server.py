from flask import Flask
app = Flask(__name__)
#!/bin/python

import os
from flask import Flask, Response, request, abort, render_template_string, send_from_directory
from PIL import Image
from io import StringIO

from handsignals.dataset import file_utils


WIDTH = 640 
HEIGHT = 400

LABELS = ["none", "metal", "ok", "victory"]


TEMPLATE = '''
<!DOCTYPE html>
<html>
<head>
    <title></title>
    <meta charset="utf-8" />
    <style>
body {
    margin: 0;
    background-color: #333;
}
.image {
    display: block;
    margin: 2em auto;
    background-color: #444;
    box-shadow: 0 0 10px rgba(0,0,0,0.3);
}
img {
    display: block;
}
    </style>
</head>
<body>
    {% for image in images %}
        <a class="image" href="{{image.src}}" style="width: {{ image.width }}px; height: {{ image.height }}px">
            <img src={{image.src}} width="{{ image.width }}" height="{{ image.height }}" />
        </a>

	<form action="" method="POST" onsubmit="" name={{image.src}}>
	{% for label in labels %}
            <p>
	    {% if label == "none" %}
		<input type="radio" name={{image.src}} id={{label}} value={{label}} checked> {{label}} 
	    {% else %}
		<input type="radio" name={{image.src}} id={{label}} value={{label}}>         {{label}}
	    {% endif %}
            </p>
	{% endfor %}
        <p><input type=submit value=Next></p>
	</form>
    {% endfor %}
</body>
'''

@app.route('/<path:filename>')
def image(filename):
    try:
        w = int(request.args['w'])
        h = int(request.args['h'])
    except (KeyError, ValueError):
        return send_from_directory('.', filename)

    try:
        im = Image.open(filename)
        im.thumbnail((w, h), Image.ANTIALIAS)
        io = StringIO.StringIO()
        im.save(io, format='JPEG')
        return Response(io.getvalue(), mimetype='image/jpeg')

    except IOError:
        abort(404)

    return send_from_directory('.', filename)

@app.route('/annotate', methods=["GET", "POST"])
def annotate():
    if request.method == "POST":
        items = request.form.to_dict().items()
        (image_path, label) = list(items)[0]
        file_utils.move_image_to_label(image_path, label)

    images = []
    for root, dirs, files in os.walk('dataset/unlabeled'):
        for filename in [os.path.join(root, name) for name in files]:
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
            images.append({
                'width': int(width),
                'height': int(height),
                'src': filename
            })

    return render_template_string(TEMPLATE,
                                  images=images,
                                  labels=LABELS)
    
@app.route('/record', methods=["GET", "POST"])
def record():
    if request.method == "POST":
        #setup a record html
        pass

app.run(debug=True, host='::')
