def data_load(filename):
    with open(filename, 'r', encoding='UTF8') as f:
        data = [line.split('\t') for line in f.read().splitlines()]
        data = data[1:]
    return data

train_data = data_load('ratings_train.txt')
test_data = data_load('ratings_test.txt')

print("전체 데이터 갯수 : {}".format(len(train_data)))
print("테스트용 데이터 갯수 : {}".format(len(test_data)))




