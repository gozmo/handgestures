#!/bin/python
from flask import Flask, request, render_template

from handsignals.dataset import file_utils
from handsignals import main
from handsignals.server.templates.data import data_template

app = Flask(__name__, template_folder="handsignals/server/templates", static_url_path="")

@app.route('/<path:filename>')
def image(filename):
    print("route_image", filename)
    return main.read_images(filename)

@app.route('/unlabeled/<string:filename>')
def unlabeled(filename):
    filepath = f"dataset/unlabeled/{filename}"
    return main.read_images(filepath)

@app.route('/evaluations/<string:training_run_id>/<string:filename>')
def evaluation(training_run_id, filename):
    filepath = f"evaluations/{training_run_id}/{filename}"
    return main.read_images(filepath)

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
    elif task == "active_learning":
        if request.method == "POST":
            post_dict= request.form.to_dict()
            if "batch_size" in post_dict:
                batch_size = int(post_dict["batch_size"])
                main.set_active_learning_batch_size(batch_size)
            else:
                main.annotate(post_dict)
        active_learning_query, all_labels = main.active_learning()

        return render_template("data/active_learning.html", active_learning_query=active_learning_query, all_labels=all_labels)

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

@app.route("/models")
def models():
    return render_template("models/base.html")

@app.route('/models/train', methods=["GET", "POST"])
def train():
    if request.method == "POST":
        post_dict= request.form.to_dict()
        learning_rate = float(post_dict["learning_rate"])
        batch_size = int(post_dict["batch_size"])
        epochs = int(post_dict["epochs"])


        if post_dict["action"] == "train":
            resume=False
        else:
            resume=True
        main.train(learning_rate=learning_rate,
                    batch_size=batch_size,
                    epochs=epochs,
                    resume=resume)

    return render_template("models/train.html")

@app.route("/models/results", methods=["GET", "POST"])
def results():
    selected_training_run_id = None

    if request.method == "POST":
        post_dict= request.form.to_dict()
        selected_training_run_id = post_dict["training_run_id"]


    training_run_id, \
    training_run_ids, \
    parameters, \
    label_order, \
    confusion_matrix, \
    dataset_stats = main.results(selected_training_run_id)

    return render_template("models/results.html",
                           parameters=parameters,
                           training_run_id=training_run_id,
                           label_order=label_order,
                           dataset_stats=dataset_stats,
                           training_run_ids=training_run_ids,
                           confusion_matrix=confusion_matrix)

@app.route('/live', methods=["GET", "POST"])
def live():
    if request.method == "POST":
        main.start_live_view()
    image_path, classification = main.live_view()

    return render_template("live_view/live_view.html", filename=image_path, classification=classification)

app.run(debug=True, host='::')
