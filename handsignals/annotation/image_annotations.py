from handsignals.constants import ImageAnnotation
from handsignals.dataset import file_utils
class ImageAnnotations:
    def __init__(self, image_filename):
        self.__image_filename = image_filename
        self.__annotations = []
        self.__read_annotations()

    def __read_annotations(self):
        if file_utils.annotation_file_exists(self.__image_filename):
            self.__annotations = file_utils.read_annotations(self.__image_filename)

    def add_annotation(self, x, y, height, width, label):
        annotation = {ImageAnnotation.X: x,
                      ImageAnnotation.Y: y,
                      ImageAnnotation.HEIGHT: height,
                      ImageAnnotation.WIDTH: width,
                      ImageAnnotation.LABEL: label}
        self.__annotations.append(annotation)


    def write_annotations(self):
        file_utils.write_annotations(self.__image_filename,
                                     self.__annotations)

def add_annotation(image_filename, x, y, height, width, label):
    image_annotation = ImageAnnotation(image_filename)
    image_annotation.add_annotation(x, y, height, width, label)

