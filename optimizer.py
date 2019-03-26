import threading

class Optimizer(threading.Thread):
    stop_signal = False

    def __init__(self, brain):
        threading.Thread.__init__(self)
        self.brain = brain

    def run(self):
        while not (self.stop_signal or self.brain.stop_signal):
            self.brain.optimize()

    def stop(self):
        self.stop_signal = True
