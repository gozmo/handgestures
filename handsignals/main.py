from flask import Flask, Response, request, abort, render_template_string, send_from_directory,render_template
from PIL import Image
from io import StringIO
from handsignals.networks import trainer
import os

WIDTH = 640
HEIGHT = 400
def capture(frames_to_capture):
    pass

def read_images(filepath, request):
    filename = os.path.basename(filepath)
    folder_path = os.path.dirname(filepath)

    return send_from_directory(folder_path, filename)

def train():
    trainer.train()
