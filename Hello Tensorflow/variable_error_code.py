# Tensorflow 라이브러리 가져오기
import tensorflow as tf

# 변수형 텐서 선언
var_1 = tf.Variable(3)
var_2 = tf.Variable(10)
# 텐서 덧셈 연산
result_sum = var_1 + var_2
# 세션 선언
sess = tf.Session()
print(sess.run(result_sum))
