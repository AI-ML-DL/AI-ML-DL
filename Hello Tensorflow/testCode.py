# Tensorflow 라이브러리 가져오기
import tensorflow as tf

result = tf.constant("Tensorflow is easy!!")
sess = tf.Session()

print(str(sess.run(result),encoding='utf8'))