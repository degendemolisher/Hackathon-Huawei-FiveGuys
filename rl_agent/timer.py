

import datetime

class Timer():
    def __init__(self):
        self.reset()

    def time(self, str):
        self.tick = self.tock
        self.tock = datetime.datetime.now()
        elapsed = self.tock - self.tick
        if elapsed.total_seconds() > 0.0:
            print(f"{elapsed.total_seconds() * 1000:.1f}ms ({str})")  
    def reset(self):
        self.start_time = datetime.datetime.now()
        self.tick = self.start_time
        self.tock = self.start_time