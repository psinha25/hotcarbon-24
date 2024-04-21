from collections import deque

def log(filename, string):
    with open(filename, "w") as h:
        h.write(string)