# Tensorflow 라이브러리 가져오기
import tensorflow as tf

var_1 = 15
var_2 = 8
# 플레이스 홀더 선언
p_holder1 = tf.placeholder(dtype=tf.float32)
p_holder2 = tf.placeholder(dtype=tf.float32)
# 플레이스홀더를 이용한 덧셈 연산
p_holder_sum = p_holder1 + p_holder2

# 세션 선언
sess = tf.Session()
# 플레이스 홀더에 값 입력
result = sess.run(p_holder_sum, feed_dict = {p_holder1: var_1, p_holder2: var_2})
print(result)
