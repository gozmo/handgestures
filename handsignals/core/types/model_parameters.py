class ModelParameters:
    learning_rate: float
    epochs: int
    batch_size: int
    resume: bool

    def __init__(self,
                 learning_rate,
                 epochs,
                 batch_size,
                 resume):
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.batch_size = batch_size
        self.resume = resume

    def to_dict(self):
        return {"learning_rate": self.learning_rate,
                "epochs": self.epochs,
                "batch_size": self.batch_size,
                "resume": self.resume}
