from collections import defaultdict
class EventHandler:
    def __init__(self):
        self.__handlers = defaultdict(list)

    def register_callback(self, event, callback):
        self.__handlers[event].append(callback)

    def register_event(self, event):
        for callback in self.__handlers[event]:
            callback()

__event_handler = EventHandler()

def register_callback(event, callback):
    global __event_handler
    __event_handler.register_callback(event, callback)

def register_event(event):
    global __event_handler
    __event_handler.register_event(event)
