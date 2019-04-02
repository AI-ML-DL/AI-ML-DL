# Tensorflow 라이브러리 가져오기
import tensorflow as tf

A = [1, 3, 5, 7, 9]
B = [ 2, 4, 6, 8, 10]

# 플레이스 홀더 선언
ph_A = tf.placeholder(dtype=tf.float32)
ph_B = tf.placeholder(dtype=tf.float32)
# 플레이스홀더를 이용한 덧셈 연산
result_sum = ph_A + ph_B

# 세션 선언
sess = tf.Session()
result = sess.run(result_sum, feed_dict = {ph_A:A, ph_B:B})
print(result)
