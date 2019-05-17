from handsignals.dataset import file_utils
class GlobalState:
    def __init__(self):
        self.__refresh_active_learning = True
        self.__refresh_aided_annotation = True
        self.__training_run_id = None

    def new_training_run(self):
        self.__refresh_active_learning = True
        self.__refresh_aided_annotation = True
        self.__training_run_id = file_utils.get_training_run_id()

    def get_training_run_id(self):
        return self.__training_run_id

    def refresh_active_learning(self):
        return self.__refresh_active_learning

    def refresh_aided_annotation(self):
        return self.__refresh_aided_annotation

    def set_active_learning_refreshed(self):
        self.__refresh_active_learning = False

    def set_aided_annotation_refreshed(self):
        self.__refresh_aided_annotation = False

    def training_run_folder(self):
        return f"evaluations/{self.__training_run_id}"

    def get_model(self):
        return

__global_state = GlobalState()

def get_global_state():
    global __global_state
    return __global_state


@property
def state():
    global __global_state
    return __global_state
