from handsignals.constants import JsonAnnotation
from handsignals.dataset import file_utils
class ImageAnnotations:
    def __init__(self, image_filename):
        self.__image_filename = image_filename
        self.__annotations = []
        self.__read_annotations()

    def __read_annotations(self):

        if file_utils.annotation_file_exists(self.__image_filename):
            self.__annotations = file_utils.read_annotations(self.__image_filename)
        else:
            print("file not found", self.__image_filename)

    def add_annotation(self, x, y, height, width, label):
        annotation = {JsonAnnotation.X: x,
                      JsonAnnotation.Y: y,
                      JsonAnnotation.HEIGHT: height,
                      JsonAnnotation.WIDTH: width,
                      JsonAnnotation.LABEL: label}
        self.__annotations.append(annotation)


    def write_annotations(self):
        file_utils.write_annotations(self.__image_filename,
                                     self.__annotations)

def add_annotation(image_filename, x, y, height, width, label):
    image_annotation = JsonAnnotation(image_filename)
    image_annotation.add_annotation(x, y, height, width, label)

