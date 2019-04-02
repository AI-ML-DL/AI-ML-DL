# add 연산

# Tensorflow 라이브러리 가져오기
import tensorflow as tf

# 입력 데이터 정의
inputData = [2,4,6,8]
# x : 입력데이터가 들어갈 데이터 자료형(Placeholder) 선언
x = tf.placeholder(dtype=tf.float32,name='x')
# W : 입력데이터와 연산을 할 상수형(Constant) 데이터형 선언
W = tf.constant([2],dtype=tf.float32, name='Weight')

# 연산식 정의
graph_function = tf.add(x,W)

# operation을 위해 변수 초기화
op = tf.global_variables_initializer()

# 정의된 연산식을 이용하여 그래프를 실행
with tf.Session() as sess :
    # 초기화 실행
    sess.run(op)
    # 연산 결과 출력
    print(sess.run(graph_function,feed_dict={x : inputData}))
