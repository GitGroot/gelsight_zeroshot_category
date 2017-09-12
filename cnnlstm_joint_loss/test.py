def get_train_test_labels(a):
    test = []
    train = []
    for i in a:
        test.append(eval(i))
    for i in range(8):
        if i not in test:
            train.append(i)
    return train, test

print get_train_test_labels('057')
