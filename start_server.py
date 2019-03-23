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
    elif task == "aided_annotation":
        if request.method == "POST":
            post_dict= request.form.to_dict()
            if "batch_size" in post_dict:
                batch_size = int(post_dict["batch_size"])
                main.set_aided_annotation_batch_size(batch_size)
            else:
                main.annotate(post_dict)
        aided_annotations, all_labels = main.aided_annotation()
        return render_template("data/aided_annotation.html", aided_annotations=aided_annotations, all_labels=all_labels)
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

@app.route('/live', methods=["GET", "POST"])
def live():
    if request.method == "POST":
        main.start_live_view()
    image_path, classification = main.live_view()

    return render_template("live_view/live_view.html", filename=image_path, classification=classification)

app.run(debug=True, host='::')
