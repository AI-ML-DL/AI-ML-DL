from gensim.models import Word2Vec
import tensorflow as tf
import numpy as np
import Project.preprocessing as pre
import sys, os

# py 파일이 실행되는 폴더 위치
base_dir = os.path.dirname(os.path.realpath(__file__)) + '/'
source_dir ='./data/'
# word2vec 파일 이름
w2v_file_name = '3_word2vec_nsmc.w2v'
# 알고리즘 학습  모델 저장 경로
cnn_model_dir = './cnn_model'
lstm_model_dir = './lstm_model'

# 사용자 입력 문장에 대해 단어 벡터 변환
def data_setting(w2v_model, embedding_dim, max_word_length, evaluation_text):
    eval_arrays = np.zeros((1, max_word_length, embedding_dim))
    # 네이버 맞춤법 검사 적용
    eval_spell_chcker = pre.naver_spell_cheker(evaluation_text)
    # 품사 부착 진행
    eval_pos_tag = pre.konlpy_pos_tag(eval_spell_chcker)
    #문장 내 단어 벡터 변환
    eval_arrays[0] = pre.max_word_length_word2vec(w2v_model, embedding_dim, max_word_length, eval_pos_tag)

    return eval_arrays

def evaluation(params) :
    # 각 알고리즘 별 최대 단어 개수 지정
    # 모델 학습 시 사용했던 값 사용
    cnn_max_sentence_length = 50
    lstm_max_sentence_length = 100
    evaluation_text = params[2]

    w2v_model = Word2Vec.load(base_dir + source_dir + w2v_file_name)
    embedding_dim = w2v_model.vector_size

    # 알고리즘 별 데이터 셋팅 및 모델의 마지막 저장된 checkpoint 파일 이름 검색
    if params[1] == 'CNN' :
        x_eval = data_setting(w2v_model, embedding_dim, cnn_max_sentence_length, evaluation_text)
        checkpoint_file = tf.train.latest_checkpoint(base_dir + cnn_model_dir)

    elif params[1] == 'LSTM' :
        x_eval = data_setting(w2v_model, embedding_dim, lstm_max_sentence_length, evaluation_text)
        checkpoint_file = tf.train.latest_checkpoint(base_dir + lstm_model_dir)

    graph = tf.Graph()
    with graph.as_default():
        sess_config = tf.ConfigProto()
        sess_config.gpu_options.allow_growth = True
        sess = tf.Session(config=sess_config)

        with sess.as_default():
            # 저장된 그래프를 재생성하여 모델을 불러옴
            # Tensorflow graph를 저장 하게 된다. 즉 all variables, operations, collections 등을 저장 한다. .meta로 확장자를 가진다.
            saver = tf.train.import_meta_graph("{}.meta".format(checkpoint_file))
            saver.restore(sess, checkpoint_file)

            # 그래프 내 operation 리스트 확인
            #for op in graph.get_operations():
            #    print(op.name)

            # 그래프에서의 Operation 불러오기
            input_x = graph.get_operation_by_name("input_x").outputs[0]
            dropout_keep_prob = graph.get_operation_by_name("dropout_keep_prob").outputs[0]
            predictions = graph.get_operation_by_name("output/predictions").outputs[0]
            result = graph.get_operation_by_name("output/result").outputs[0]

            if params[1] == 'CNN' :
                feed_dict = {input_x: x_eval, dropout_keep_prob: 1.0}
            elif params[1] == 'LSTM' :
                batch_size = graph.get_operation_by_name("batch_size").outputs[0]
                feed_dict = {input_x: x_eval, batch_size: 1, dropout_keep_prob: 1.0}

            eval_pred, eval_result = sess.run([predictions, result], feed_dict)

            # 예측된 결과에 대해 긍정/부정으로 나누고 Softmax를 통해 나온 값을 통해 확률 계산
            result_pred = '긍정' if(eval_pred == 1) else '부정'
            result_score = eval_result[0][1] if(eval_pred == 1) else eval_result[0][0]

            print('입력된 [' + evaluation_text + ']는 ')
            print('[' + str('{:.2f}'.format(result_score * 100)) + ']%의 확률로 [' + result_pred + ']으로 예측됩니다.')

if __name__ == "__main__":
    default_param = [sys.argv[0], 'CNN', '다시 찾아 보고 싶은 영화입니다.']

    if len(sys.argv) == 1 :
        evaluation(default_param)

    elif len(sys.argv) == 3 :
        evaluation(sys.argv)

    else :
        print("[USAGE] python(3) nsmc_evaluation.py 'CNN|LSTM' '테스트입니다.'")