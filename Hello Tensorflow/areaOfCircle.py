# 원의 넓이 및 총합 구하기
# Tensorflow 라이브러리 가져오기
import tensorflow as tf

# 반지름 입력 데이터 선언
inputData = [2., 3., 4., 5.]
# pi값 지정(소수점이하 생략)
ValuePI = tf.constant([3.], dtype=tf.float32)
# 반지름 데이터가 들어갈 데이터 자료형(Placeholder) 선언
radius = tf.placeholder(dtype=tf.float32)

# 원의 넓이, 넓이의 총합 공식 선언
area = tf.pow(radius, 2) * ValuePI
resultSum = tf.reduce_sum(area)

# operation을 위해 변수 초기화
op = tf.global_variables_initializer()

with tf.Session() as sess :
    # 초기화 실행
    sess.run(op)
    # fetch 방법으로 2개의 결과를 가져오고 feed 방법으로 실행시 반지름 데이터 입력
    valueArea, valueSum = sess.run([area, resultSum], feed_dict={radius: inputData})
    # 결과 출력
    print ("Circle area : ", valueArea)
    print ("Total area sum : ", valueSum)
