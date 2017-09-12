import os
import time
#os.system('nvidia-smi')
def excute():
    t = os.popen('nvidia-smi')
    for i, line in enumerate(t):
        if i == 8:break
    usage = eval(line.split('|')[2].split('/')[0].split('MiB')[0].lstrip(' '))
    print usage
    if usage < 5000:
        #os.system('python cnnlstm_joint_loss/cnnlstm_joint_changeble_S.py 1')
        os.system('ls')
while(True):
    excute()
    time.sleep(10)
