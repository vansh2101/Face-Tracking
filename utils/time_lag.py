import time

class TimeLag:
    def __init__(self):
        self.time_diff = []
        self.start_time = 0

    def register_time(self):
        if self.start_time == 0:
            self.start_time = time.time()

        else:
            self.calculate_time_lag(time.time())

    def calculate_time_lag(self, time):
        self.time_diff.append(time - self.start_time)
        self.start_time = time

    def get_time_lag(self):
        if len(self.time_diff) == 0:
            return 0
        
        return sum(self.time_diff) / len(self.time_diff)