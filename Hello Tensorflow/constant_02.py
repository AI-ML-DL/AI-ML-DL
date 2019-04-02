# Tensorflow 라이브러리 가져오기
import tensorflow as tf

# 상수형 텐서 선언
x = tf.constant(3)
# 세션 선언
sess = tf.Session()
result = sess.run(x)
print(result)
