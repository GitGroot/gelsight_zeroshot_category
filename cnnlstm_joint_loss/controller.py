import os
import time
#os.system('nvidia-smi')
# def excute():
#     t = os.popen('nvidia-smi')
#     for i, line in enumerate(t):
#         if i == 8:break
#     usage = eval(line.split('|')[2].split('/')[0].split('MiB')[0].lstrip(' '))
#     print usage
#     if usage < 5000:
#         os.system('python cnnlstm_joint_changeble_S.py 1')
# while(True):
#     excute()
#     time.sleep(10)
#os.system('python cnnlstm_joint_changeble_S.py 1 057')
os.system('python cnnlstm_joint_dot.py 0.1 057')
os.system('python cnnlstm_joint_dot.py 1 057')


os.system('python cnnlstm_joint_changeble_S.py 0 057')
os.system('python cnnlstm_joint_changeble_S.py 10 057')


os.system('python cnnlstm_joint_dot.py 0 057')
os.system('python cnnlstm_joint_dot.py 10 057')


