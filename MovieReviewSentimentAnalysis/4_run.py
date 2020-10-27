import numpy as np
from tensorflow.keras.models import load_model
from konlpy.tag import Okt

model = load_model('./result_model.mod')
okt = Okt()

selected_words = []
with open('selected_words.list', 'r') as file:
    selected_words = file.readlines()
for index in range(0, len(selected_words)):
    selected_words[index] = selected_words[index].rstrip('\n')


def tokenize(doc):
    return ['/'.join(t) for t in okt.pos(doc, norm=True, stem=True)]


def term_frequency(doc):
    return [doc.count(word) for word in selected_words]


def run_review(review):
    token = tokenize(review)
    tf = term_frequency(token)
    data = np.expand_dims(np.asarray(tf).astype('float32'), axis=0)
    score = float(model.predict(data))
    if(score > 0.5):
        print("{:.2f}% 확률, 긍정 리뷰\n".format(score * 100))
    else:
        print("{:.2f}% 확률, 부정 리뷰\n".format((1 - score) * 100))


while True:
    txt = input('감정 분석을 위한 문장을 입력 하세요: ')
    if txt == '':
        break
    run_review(txt)