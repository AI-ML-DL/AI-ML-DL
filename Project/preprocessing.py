import requests, json,  os, html
from konlpy.tag import Okt
import numpy as np
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from collections import Counter

def nsmc_data_download(file_list) :
    # 데이터 다운로드 주소
    nsmc_url = 'https://raw.githubusercontent.com/e9t/nsmc/master/'
    source_dir ='./data/'

    # 데이터 다운로드 폴더 생성
    if not(os.path.isdir(source_dir)) :
        os.makedirs(os.path.join(source_dir))

    # 바이너리 데이터를 파일로 쓰기
    for file in file_list :
        response = requests.get(nsmc_url + file)
        print('file name : ' + file)
        print('status code : ' + str(response.status_code))
        with open(source_dir + file,'wb') as f:
            # 바이너리 형태로 데이터 추출
            f.write(response.content)
        f.close()

def naver_spell_cheker(input) :
    source_dir ='./data/'
    # 네이버 맞춤법 검사기 주소
    spell_checker_url = 'https://m.search.naver.com/p/csearch/ocontent/util/SpellerProxy'

    def spell_cheker(object) :
        # request parameter 셋팅
        req_params = {'_callback': 'SpellChecker', 'q': object, 'color_blindness': 0}
        while True :
            response = requests.get(spell_checker_url, params=req_params)
            status  = response.status_code

            # 응답코드가 200일 때까지 반복
            if status == 200 :
                # 텍스트 형태로 데이터 추출
                response = response.text
                break

        # json 포멧으로 변경하기 위한 불필요 문자 제거
        response = response.replace(req_params.get('_callback')+'(', '')
        response = response.replace(');', '')

        data = json.loads(response)
        # json 포멧에서 필요 결과 값만 가져오기
        object = data['message']['result']['notag_html']
        object = html.unescape(object)

        return object

    if type(input) is str :
        return spell_cheker(input)

    elif type(input) is list :
        for file in input:
            spell_check_data = ''
            with open(source_dir+file,'r', encoding='UTF-8') as f1:
                # 파일에서의 header 부분 삭제
                load_data = [line.split('\t') for line in f1.read().splitlines()][1:]
                print('file length : ' + file + ' - ' + str(len(load_data)))
            f1.close()

            # 맞춤법 검사 진행 결과를 파일 내 문서 포멧으로 변환
            for data in load_data:
                data[1] = spell_cheker(data[1])
                spell_check_data += data[1] + '\t' + data[2] + '\n'

            # 새로운 파일로 맞춤법 검사 결과를 파일로 저장
            with open(source_dir + '1_' + file , mode='wb') as f2:
                spell_check_data = spell_check_data.encode(encoding='utf-8')
                f2.write(spell_check_data.strip())
            f2.close()

def konlpy_pos_tag(input) :
    source_dir ='./data/'
    fig_file = '2_wordcloud.png'

    def pos_tagging(object) :
        # 형태소 분석 및 품사 태깅(정규화, 어간추출, 품사합치기)
        pos = Okt().pos(object, norm=True, stem=True, join=True)
        # 명사 추출
        noun = Okt().nouns(object)

        return pos, noun

    if type(input) is str :
        return ' '.join(pos_tagging(input)[0])

    elif type(input) is list :
        word_cloud_data = list()

        for file in input:
            pos_tag_data = ''

            with open(source_dir + file,'r', encoding='UTF-8') as f1:
                load_data = [line.split('\t') for line in f1.read().splitlines()]
            f1.close()

            # 품사 추출 결과를 파일 내 문서 포멧으로 변환
            # wordcloud 생성을 위한 텍스트 배열 생성
            for data in load_data:
                result_pos, result_noun = pos_tagging(data[0])
                pos_tag_data += ' '.join(result_pos) + '\t' + data[1] + '\n'
                word_cloud_data += result_noun

            # 텍스트 배열 내 단어들에 대한 빈도 계산
            counter = Counter(word_cloud_data)
            print(counter.most_common(20))

            with open(source_dir + file.replace('1_', '2_'), mode='wb') as f2:
                pos_tag_data = pos_tag_data.encode(encoding='utf-8')
                f2.write(pos_tag_data.strip())
            f2.close()

        #Windows 폰트 위치
        #font = 'C:/Windows/Fonts/malgun.ttf'
        #Linux 폰트 위치
        font = '/usr/share/fonts/truetype/nanum/NanumBarunGothic.ttf'

        # 명사만 추출된 리스트를 통해 wordcloud 생성

        wc = WordCloud(font_path=font, width=800, height=800, background_color="white")
        plt.imshow(wc.generate_from_frequencies(counter))
        plt.axis("off")
        plt.savefig(source_dir + fig_file)

def max_word_length_word2vec(w2v_model, embedding_dim , max_word_length, word_list):
    # 문장 내 단어 제한 x 벡터 차원 수
    data_arrays = np.zeros((max_word_length, embedding_dim))

    # string 문장으로 들어오는 경우 split 처리
    if type(word_list) is str :
        word_list = word_list.split()

    # 단어를 벡터로 변경
    if len(word_list) > 0 :
        word_length = max_word_length if max_word_length < len(word_list) else len(word_list)

        for i in range(word_length):
            try :
                data_arrays[i] = w2v_model[word_list[i]]
            except KeyError :
                pass
    return data_arrays

if __name__ == '__main__':
    file_list = ['ratings_train.txt', 'ratings_test.txt']
    nsmc_data_download(file_list)
    naver_spell_cheker(file_list)
    #naver_spell_cheker('테스트입니다.')
    konlpy_pos_tag(['1_' + file for file in file_list])
    #print(konlpy_pos_tag('항상 대단한 감독... 다큐멘터리인데 재미있었어욬ㅋㅋㅋㅋㅋㅋ'))
