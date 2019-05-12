class EvaluationResults:
    def __init__(self, training_run_id):
        self.training_run_id = training_run_id
        self.dicts = dict()
        self.matrices = dict()
        self.images = dict()

    def add_matrix(self, name, matrix):
        self.matrices[name] = matrix

    def add_image(self, name, image_filename):
        self.images[name] = image_filename

    def add_dictionary(self, name, dictionary, header_key_name, header_value_name):
        elem = {"dict": dictionary,
                "header_key_name": header_key_name,
                "header_value_name": header_value_name}
        self.dicts[name] = elem
        print(self.dicts)

    def add_label_order(self, label_order):
        self.label_order = label_order

    def add_parameters(self, parameters):
        self.parameters = parameters

    def add_dataset_stats(self, data_stats):
        self.dataset_stats = data_stats

