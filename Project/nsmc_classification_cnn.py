import tensorflow as tf
import numpy as np
import Project.preprocessing as pre
import sys, os
from Project.cnn_model import cnn_model
from gensim.models import Word2Vec
import matplotlib.pyplot as plt

source_dir ='./data/'
file_list = ['2_ratings_train.txt', '2_ratings_test.txt']
w2v_file_name = '3_word2vec_nsmc.w2v'
fig_file_name = '4_cnn_pyplot.png'

# 파일을 읽어 각 문장을 탭으로 구분
def load_data(txtFilePath):
    #with open(txtFilePath,'r', encoding='UTF-8') as data_file:
    with open(txtFilePath,'r') as data_file:
        return [line.split('\t') for line in data_file.read().splitlines()]

# 긍/부정에 대한 one-hot encoding
def label_value(code, size):
    code_arrays = np.zeros((size))
    # 부정인 경우 [1, 0]
    if code == 0:
        code_arrays[0] = 1
    # 긍정인 경우 [0, 1]
    elif code == 1:
        code_arrays[1] = 1

    return code_arrays

def data_setting(w2v_model, embedding_dim, class_sizes, max_word_length):
    # 데이터 불러와서 문장의 총 개수 셋팅
    train_data = load_data(source_dir + file_list[0])
    train_size = len(train_data)
    #print('train_size : ' + str(train_size))

    test_data = load_data(source_dir + file_list[1])
    test_size = len(test_data)
    #print('dev_size : ' + str(dev_size))

    # 데이터 구조 : 전체 문장 x 문장 내 단어 제한 수 x 벡터의 차원
    train_arrays = np.zeros((train_size, max_word_length, embedding_dim))
    test_arrays = np.zeros((test_size, max_word_length, embedding_dim))
    # 정답의 구조 : 전체 문장 x 구분 수(긍정/부정)
    train_labels = np.zeros((train_size, class_sizes))
    test_labels = np.zeros((test_size, class_sizes))

    for train in range(len(train_data)) :
        # 각 문장의 단어를 벡터화 하고 문장 구성
        train_arrays[train] = pre.max_word_length_word2vec(w2v_model, embedding_dim, max_word_length, train_data[train][0])
        # 각 문장이 정답을 one-hot encoding으로 변경
        train_labels[train] = label_value(int(train_data[train][1]), class_sizes)

    for dev in range(len(test_data)) :
        test_arrays[dev] = pre.max_word_length_word2vec(w2v_model, embedding_dim, max_word_length, test_data[dev][0])
        test_labels[dev] = label_value(int(test_data[dev][1]), class_sizes)

    return train_arrays, train_labels, test_arrays, test_labels

def make_batch(list_data, batch_size):
    num_batches = int(len(list_data)/batch_size)
    batches = list()

    for i in range(num_batches):
        start = int(i * batch_size)
        end = int(start + batch_size)
        batches.append(list_data[start:end])

    return batches

def run_cnn(params) :
    class_sizes = 2
    max_sentence_length = int(params[1])
    filter_sizes = np.array(params[2].split(','), dtype=int)
    num_filters = int(params[3])
    dropout_keep_prob = float(params[4])
    num_epochs = int(params[5])
    batch_size = int(params[6])
    evaluate_every = int(params[7])
    learn_rate = float(params[8])

    model = Word2Vec.load(source_dir + w2v_file_name)
    # word2vec 파일에서의 벡터 차원 수 계산
    embedding_dim = model.vector_size

    print('-----------------------------------')
    print(' ** parameter ** ')
    print('embedding_dim :',embedding_dim)
    print('class_sizes :',class_sizes)
    print('max_sentence_length :',max_sentence_length)
    print('filter_sizes :',filter_sizes)
    print('num_filters :',num_filters)
    print('dropout_keep_prob :',dropout_keep_prob)
    print('num_epochs :',num_epochs)
    print('batch_size :',batch_size)
    print('evaluate_every :',evaluate_every)
    print('learn_rate :',learn_rate)
    print('-----------------------------------')

    x_train, y_train, x_dev, y_dev = data_setting(model, embedding_dim, class_sizes, max_sentence_length)

    # 학습/검증 정확도, 비용값 저장 리스트 선언
    train_x_plot = list()
    train_y_accracy = list()
    train_y_cost = list()
    valid_x_plot = list()
    valid_y_accuracy = list()
    valid_y_cost = list()

    # 모델 저장할 폴더 생성
    if not(os.path.isdir('./cnn_model')) :
        os.makedirs(os.path.join('./cnn_model'))

    with tf.Graph().as_default():
        #sess_config = tf.ConfigProto(device_count = {'GPU': 0})
        sess_config = tf.ConfigProto()
        sess_config.gpu_options.allow_growth = True
        sess = tf.Session(config=sess_config)

        with sess.as_default():
            cnn = cnn_model(
                sequence_length=x_train.shape[1],
                num_classes=y_train.shape[1],
                embedding_size=embedding_dim,
                filter_sizes=filter_sizes,
                num_filters=num_filters)

            global_step = tf.Variable(0, name="global_step", trainable=False)
            # 비용함수의 값이 최소가 되도록 하는 최적화 함수 선언
            optimizer = tf.train.AdamOptimizer(learn_rate)
            train_op = optimizer.minimize(cnn.cost, global_step=global_step)
            #grads_and_vars = optimizer.compute_gradients(cnn.cost)
            #train_op = optimizer.apply_gradients(grads_and_vars, global_step=global_step)

            saver = tf.train.Saver()
            sess.run(tf.global_variables_initializer())

            def train_step(x_batch, y_batch):
                feed_dict = {
                  cnn.input_x: x_batch,
                  cnn.input_y: y_batch,
                  cnn.dropout_keep_prob: dropout_keep_prob
                }
                _, step, cost, accuracy = sess.run([train_op, global_step, cnn.cost, cnn.accuracy], feed_dict)
                train_x_plot.append(step)
                train_y_accracy.append(accuracy * 100)
                train_y_cost.append(cost)
                #print("Train step {}, cost {:g}, accuracy {:g}".format(step, cost, accuracy))


            def dev_step(x_batch, y_batch, epoch):
                feed_dict = {
                  cnn.input_x: x_batch,
                  cnn.input_y: y_batch,
                  cnn.dropout_keep_prob: 1.0
                }
                step, cost, accuracy, dev_pred = sess.run([global_step, cnn.cost, cnn.accuracy, cnn.predictions], feed_dict)
                valid_x_plot.append(step)
                valid_y_accuracy.append(accuracy * 100)
                valid_y_cost.append(cost)
                print("Valid step, epoch {}, step {}, cost {:g}, accuracy {:g}".format((epoch+1), step, cost, accuracy))

            # 배치 데이터 생성
            train_x_batches = make_batch(x_train, batch_size)
            train_y_batches = make_batch(y_train, batch_size)

            # 배치별 트레이닝, 검증
            for epoch in range(num_epochs):
                for len_batch in range(len(train_x_batches)):
                    train_step(train_x_batches[len_batch], train_y_batches[len_batch])
                    current_step = tf.train.global_step(sess, global_step)
                    if current_step % evaluate_every == 0:
                        dev_step(x_dev, y_dev, epoch)
            # 모델 저장
            saver.save(sess, "./cnn_model/model.ckpt")

    # 학습 / 검증에서의 정확도와 비용 시각화
    plt.subplot(2,1,1)
    plt.plot(train_x_plot, train_y_accracy, linewidth = 2, label = 'Training')
    plt.plot(valid_x_plot, valid_y_accuracy, linewidth = 2, label = 'Validation')
    plt.title("Train and Validation Accuracy / Cost Result")
    plt.ylabel('accuracy')
    plt.legend()

    plt.subplot(2,1,2)
    plt.plot(train_x_plot, train_y_cost, linewidth = 2, label = 'Training')
    plt.plot(valid_x_plot, valid_y_cost, linewidth = 2, label = 'Validation')
    plt.xlabel('step')
    plt.ylabel('cost')
    plt.legend()

    # 이미지로 저장
    plt.savefig(source_dir + fig_file_name)

if __name__ == "__main__":
    default_param = [sys.argv[0], 50, '2,3,4', 50, 0.5, 20, 1000, 150, 0.001]

    if len(sys.argv)==1:
        run_cnn(default_param)
    else :
        print(sys.argv)
        run_cnn(sys.argv)