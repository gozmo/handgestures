#!/bin/python
from flask import Flask, request, render_template

from handsignals.dataset import file_utils
from handsignals import main
from handsignals.server.templates.data import data_template

app = Flask(__name__, template_folder="handsignals/server/templates", static_url_path="")

@app.route('/<path:filename>')
def image(filename):
    print("route_image", filename)
    return main.read_images(filename, request)

@app.route('/unlabeled/<string:filename>')
def unlabeled(filename):
    filepath = f"dataset/unlabeled/{filename}"
    print("unlabel")
    return main.read_images(filepath, request)

@app.route('/capture', methods=["GET", "POST"])
def capture():
    if request.method == "POST":
        post_dict= request.form.to_dict()
        frames_to_capture= post_dict["frames_to_capture"]
        main.capture(frames_to_capture)

    return render_template("capture/capture.html")

@app.route("/data")
@app.route("/data/<task>", methods=["GET", "POST"])
def data(task=None):
    print(f"data task: {task}")
    if task is None:
        return render_template(f"data/base.html")
    elif task == "annotate":
        return data_template.render_annotate(request)
    else:
        return render_template(f"data/{task}.html")

@app.route("/index")
def index():
    return render_template("base.html")

@app.route("/status")
def status():
    return "status: ok"

@app.route('/models', methods=["GET", "POST"])
def models():
    if request.method == "POST":
        #post_dict= request.form.to_dict()
        #frames_to_capture= post_dict["frames_to_capture"]
        main.train()

    return render_template("models/base.html")
app.run(debug=True, host='::')
