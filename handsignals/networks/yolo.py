class YoloV3:
    def __init__(self, num_output_classes):
        self.__num_output_classes = num_output_classes

    def train(self, training_dataset, holdout_dataset, model_parameters):
        pass

    def classify_dataset(self, dataset):
        return []

    def save(self, path):
        pass

class YoloV3Model(nn.Module):
    def __init__(self, num_classes=4):
        super(YoloV3Model, self).__init__()

        #conv 32 filters, 3x3
        #conv 64 filters, 3x3 /2

        # block 1
        # conv 32 filters 1x1
        # conv 64 filters 3x3
        # Residual

        #conv 128 filters, 3x3/2

        # block 2
        # conv 64 filters 1x1
        # conv 128 filters 3x3
        # Residual

        #conv 256 filters, 3x3/2

        # block 3
        # conv 128 filters 1x1
        # conv 256 filters 3x3
        # Residual

        #conv 512 filters, 3x3/2

        # block 4
        # conv 256 filters 1x1
        # conv 512 filters 3x3
        # Residual

        #conv 1024 filters, 3x3/2

        # block 5
        # conv 512 filters 1x1
        # conv 1024 filters 3x3
        # Residual

        # Avgpool global
        # Connected 1000
        # Softmax

    def __yolo_block(self, filters_1, filters_2):
        #convddk
