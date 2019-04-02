from gensim import models
#import preprocessing as pre
import Project.preprocessing as pre
from sklearn.manifold import TSNE
import matplotlib
matplotlib.use('Agg')
from matplotlib import font_manager, rc, pyplot
from collections import Counter

def apply_word2vec(file_list) :
    source_dir ='./data/'
    # 전체 문장을 담는 리스트 선언
    total_sentences = list()

    for file in file_list:
        with open(source_dir + file, 'r', encoding='UTF-8') as f:
            load_data = [line.split('\t') for line in f.read().splitlines()]
            for data in load_data :
                total_sentences.append(data[0].split())

    # word2vec로 단어 벡터로 변경 및 모델 저장
    model = models.Word2Vec(total_sentences, min_count=3, window=5, sg=1, size=100, workers=4, iter=50)
    model.save(source_dir + '3_word2vec_nsmc.w2v')
    model.wv.save_word2vec_format(source_dir + '3_word2vec_nsmc_format.w2v', binary=False)

def word2vec_test(file_list, w2v_name) :
    # 단어를 담을 리스트 선언
    total_word_list = list()

    source_dir ='./data/'
    fig_file = '3_word2vec_tsne.png'
    font_name = '/usr/share/fonts/truetype/nanum/NanumBarunGothic.ttf'

    # word2vec 모델 로드
    model = models.Word2Vec.load(source_dir + w2v_name)

    # 품사 태깅 된 데이터 추출 및 리스트 저장
    data_list = list()
    data1 = pre.konlpy_pos_tag('배우')
    data_list.append(data1)
    data2 = pre.konlpy_pos_tag('엄마')
    data_list.append(data2)
    data3 = pre.konlpy_pos_tag('여자')
    data_list.append(data3)
    data4 = pre.konlpy_pos_tag('남자')
    data_list.append(data4)

    # 모델에 적용하여 결과 출력
    # model.doesnt_match, model.most_similar의 method는 4.0.0 버전에서 deprecated
    print(model[data1])
    print(model.wv.doesnt_match(data_list))
    print(model.wv.most_similar(positive=[data1], topn=10))
    print(model.wv.most_similar(positive=[data2, data4], negative=[data3], topn=1))
    print(model.wv.similarity(data1, data2))
    print(model.wv.similarity(data1, data3))

    for file in file_list:
        with open(source_dir + file,'r', encoding='UTF-8') as f:
            load_data = [line.split('\t') for line in f.read().splitlines()]
            for data in load_data :
                total_word_list += data[0].split()

    # 단어 리스트 중 가장 많이 사용된 100개 단어 추출
    counter = Counter(total_word_list).most_common(100)
    word_list = [word[0] for word in counter]
    print(word_list)

    # 설정 가능한 폰트 리스트 출력
    font_list = font_manager.get_fontconfig_fonts()
    print([font for font in font_list if 'nanum' in font])

    # 폰트 설정
    rc('font', family=font_manager.FontProperties(fname=font_name).get_name())

    # 단어에 대한 벡터 리스트
    vector_list = model[word_list]

    # 2차원으로 차원 축소
    transformed = TSNE(n_components=2).fit_transform(vector_list)
    print(transformed)

    # 2차원의 데이터를 x, y 축으로 저장
    x_plot = transformed[:, 0]
    y_plot = transformed[:, 1]

    # 이미지의 사이즈 셋팅
    pyplot.figure(figsize=(10, 10))

    # x, y 축을 점 및 텍스트 표시
    pyplot.scatter(x_plot, y_plot)
    for i in range(len(x_plot)):
        pyplot.annotate(word_list[i], xy=(x_plot[i], y_plot[i]))

    # 이미지로 저장
    pyplot.savefig(source_dir + fig_file)

if __name__ == '__main__':
    # Konlpy 사용하여 품사 부착된 파일 리스트
    file_list = ['2_ratings_train.txt', '2_ratings_test.txt']
    apply_word2vec(file_list)
    word2vec_test(file_list, '3_word2vec_nsmc.w2v')