import json
import os
from konlpy.tag import Okt
import nltk
import numpy as np

from tensorflow.keras import models
from tensorflow.keras import layers
from tensorflow.keras import optimizers
from tensorflow.keras import losses
from tensorflow.keras import metrics

okt = Okt()

def tokenize(doc):
    return ['/'.join(t) for t in okt.pos(doc, norm=True, stem=True)]


def data_load(filename):
    with open(filename, 'r', encoding='UTF8') as f:
        data = [line.split('\t') for line in f.read().splitlines()]
        data = data[1:]
    return data


def term_frequency(doc):
    return [doc.count(word) for word in selected_words]


train_data = data_load('ratings_train_min.txt')
test_data = data_load('ratings_test_min.txt')

print("전체 데이터 갯수 : {}".format(len(train_data)))
print("테스트용 데이터 갯수 : {}".format(len(test_data)))

if os.path.isfile('train_docs.json'):
    with open('train_docs.json', encoding='UTF8') as f:
        train_docs = json.load(f)
    with open('test_docs.json', encoding='UTF8') as f:
        test_docs = json.load(f)
else:
    train_docs = [(tokenize(row[1]), row[2]) for row in train_data]
    test_docs = [(tokenize(row[1]), row[2]) for row in test_data]
    with open('train_docs.json', 'w', encoding='UTF8') as make_file:
        json.dump(train_docs, make_file, ensure_ascii=False, indent="\t")
    with open('test_docs.json', 'w', encoding='UTF8') as make_file:
        json.dump(test_docs, make_file, ensure_ascii=False, indent="\t")

print("문장 태깅 결과 (5개만 보기) : ")
for index in range(5):
    print(train_docs[index])

tokens = [t for d in train_docs for t in d[0]]
print("토큰 갯수 : ")
print(len(tokens))

text = nltk.Text(tokens, name='NMSC')

print("전체 토큰 : {}".format(len(text.tokens)))
print("중복 제외 토큰 : {}".format(len(set(text.tokens))))
print("출현빈도 상위 TOP 10 토큰 : {}".format(text.vocab().most_common(10)))

selected_words = [f[0] for f in text.vocab().most_common(1000)]

train_x = [term_frequency(d) for d, _ in train_docs]
test_x = [term_frequency(d) for d, _ in test_docs]
train_y = [c for _, c in train_docs]
test_y = [c for _, c in test_docs]

x_train = np.asarray(train_x).astype('float32')
x_test = np.asarray(test_x).astype('float32')

y_train = np.asarray(train_y).astype('float32')
y_test = np.asarray(test_y).astype('float32')

model = models.Sequential()
model.add(layers.Dense(64, activation='relu', input_shape=(1000,)))
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(1, activation='sigmoid'))

model.compile(optimizer=optimizers.RMSprop(lr=0.001), loss=losses.binary_crossentropy, metrics=[metrics.binary_accuracy])

model.fit(x_train, y_train, epochs=10, batch_size=512)
results = model.evaluate(x_test, y_test)

print("정확도 : {}".format(results))

## 학습 모델 결과 저장
model.save('./result_model.mod')

## selected_words 리스트 저장
with open('selected_words.list', 'w') as file:
    for index in range(0, len(selected_words)):
        file.writelines(selected_words[index])
        file.writelines('\n')
