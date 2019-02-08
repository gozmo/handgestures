from flask import Flask
app = Flask(__name__, template_folder="handsignals/server/templates")
#!/bin/python
from flask import Flask, Response, request, abort, render_template_string, send_from_directory,render_template
from io import StringIO

from handsignals.dataset import file_utils
from handsignals import main
from handsignals.server.templates.data import data_template





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

@app.route('/record', methods=["GET", "POST"])
def record():
    if request.method == "POST":
        post_dict= request.form.to_dict()
        frames_to_capture= post_dict["frames_to_capture"]
        main.record(frames_to_capture)

    return render_template("record/record.html")

@app.route("/data")
@app.route("/data/<task>", methods=["GET", "POST"])
def data(task=None):
    print(task)
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

app.run(debug=True, host='::')
