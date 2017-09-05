def log(epoch_num, iter_num, cost, name):
    if epoch_num == 0 and iter_num == 0:
        f = file(name, 'w+')
    else:
        f = file(name, 'a+')
    if iter_num == 0:
        print('epoch:%d'%epoch_num)
        f.write('epoch:%d'%epoch_num)
    print('Iteration:%-7d\tCost:%-2.5f'%(iter_num, cost))
    f.write('Iteration:%-7d\tCost:%-2.5f\n'%(iter_num, cost))
    f.close()

def log2(epoch_num, iter_num, cost, acc_val, name):
    if epoch_num == 0 and iter_num == 0:
        f = file(name, 'w+')
    else:
        f = file(name, 'a+')
    if iter_num == 0:
        print('epoch:%d'%epoch_num)
        f.write('epoch:%d'%epoch_num)
    print('Iteration:%-7d\tCost:%-2.5f\tAccuracy:%2.5f'%(iter_num, cost, acc_val))
    f.write('Iteration:%-7d\tCost:%-2.5f\tAccuracy:%2.5f\n'%(iter_num, cost, acc_val))
    f.close()
