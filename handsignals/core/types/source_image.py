class SourceImage:
    def __init__(self, source, width, height):
        self.SOURCE = source
        self.WIDTH = width
        self.HEIGHT = height

    def __str__(self):
        return f"{self.SOURCE}, {self.WIDTH}, {self.HEIGHT}"
