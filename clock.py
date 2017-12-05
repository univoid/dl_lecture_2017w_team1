import time

class Clock():
    def __init__(self):
        self.start = 0
        
    def start(self):
        self.start = time.time()
        
    def check(self):
        now = time.time()
        dif = now - self.start
        return dif
        
    def check_HMS(self):
        dif = self.check()
        hour = dif // 3600
        dif = dif % 3600
        minute = dif // 60
        second = dif % 60
        print "elapsed time: %2d:%2d:%2d" % (hour, minute, second)
        
    def check_MS(self):
        dif = self.check()
        minute = dif // 60
        second = dif % 60
        print "elapsed time: %2d m %2d s" % (minute, second)