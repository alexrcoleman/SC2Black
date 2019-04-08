import os
import sys
import time
import random
import signal
import subprocess

def main(argv):
    for x in range(50):
        y = random.uniform(1e-3, 1e-6)
        command = 'main.py --learning_rate=' + str(y) + ' --tensorboard_dir=LR' + str(y)
        pro = subprocess.Popen(command,stdout=subprocess.PIPE, shell=True, creationflags=subprocess.CREATE_NEW_PROCESS_GROUP)
        time.sleep(int(sys.argv[1]))
        subprocess.call(['taskkill', '/F', '/T', '/PID', str(pro.pid)])

if __name__ == "__main__":
    main(sys.argv[1:])
