import tensorflow as tf

class lstm_model(object):
    def __init__(self, sequence_length, num_classes, embedding_size, hidden_unit, num_layer):
        # 학습 데이터가 들어갈 플레이스 홀더 선언
        self.input_x = tf.placeholder(tf.float32, shape=[None, sequence_length, embedding_size], name='input_x')
        self.input_y = tf.placeholder(tf.float32, shape=[None, num_classes], name='input_y')
        self.dropout_keep_prob = tf.placeholder(tf.float32, name='dropout_keep_prob')
        self.batch_size = tf.placeholder(tf.int32, [], name="batch_size")

        # LSTM Layer
        with tf.name_scope("lstm"):
            def lstm_cell():
                #tf.nn.rnn_cell.(Basic)LSTMCell / tf.nn.rnn_cell.(Basic)RNNCell / tf.nn.rnn_cell.GRUCell
                # LSTM Cell 및 DropOut 설정
                lstm = tf.nn.rnn_cell.LSTMCell(num_units=hidden_unit, forget_bias=1.0, state_is_tuple=True)
                return tf.nn.rnn_cell.DropoutWrapper(cell=lstm, output_keep_prob=self.dropout_keep_prob)

            # RNN Cell을 여러 층 쌓기
            lstm_cell = tf.nn.rnn_cell.MultiRNNCell([lstm_cell() for _ in range(num_layer)])
            # 초기 state 값을 0으로 초기화
            self.initial_state = lstm_cell.zero_state(self.batch_size, tf.float32)
            # outputs : [batch_size, sequence_length, hidden_unit]
            outputs, state = tf.nn.dynamic_rnn(lstm_cell, self.input_x, initial_state=self.initial_state , dtype=tf.float32)
            # output : [sequence_length, batch_sie, hidden_unit)
            output = tf.transpose(outputs, [1, 0, 2])
            # 마지막 출력만 사용
            output = tf.gather(output, int(output.get_shape()[0]) - 1)

        # Output Layer
        with tf.name_scope("output"):
            W = tf.get_variable("W", shape=[hidden_unit, num_classes], initializer=tf.contrib.layers.xavier_initializer())
            b = tf.Variable(tf.constant(0.1, shape=[num_classes]), name="b")
            self.scores = tf.nn.xw_plus_b(output, W, b, name="logits")
            self.predictions = tf.argmax(self.scores, 1, name="predictions")
            self.result = tf.nn.softmax(logits=self.scores, name="result")

        # 비용 함수(오차, 손실함수) 선언
        with tf.name_scope("loss"):
            self.cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=self.scores, labels=self.input_y))

        # 정확도 계산
        with tf.name_scope("accuracy"):
            correct_predictions = tf.equal(self.predictions, tf.argmax(self.input_y, axis=1))
            self.accuracy = tf.reduce_mean(tf.cast(correct_predictions, tf.float32), name="accuracy")