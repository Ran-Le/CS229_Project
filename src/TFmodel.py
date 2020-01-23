import tensorflow as tf
import util
import numpy as np


def validation(model, input_col, label_col, profile_test):
    """
    :param input_col: list of X name
    :param label_col: list of y name
    :param profile_test: the list of test profile ID
    :return: n/a
    """
    test_path = 'profile_data/Profile_' + str(profile_test) + '.csv'
    x_test, y_test = util.load_dataset(test_path, input_col, label_col)
    x_test = tf.convert_to_tensor(x_test, dtype=tf.float32)
    y_test = tf.convert_to_tensor(y_test, dtype=tf.float32)

    model.evaluate(x_test, y_test, batch_size=100, steps=1)

    return


def main(profile, input_col, label_col, profile_test):
    """
    :param profile: training profile ID
    :param input_col: list of X name
    :param label_col: list of y name
    :param profile_test: the list of test profile ID
    :return: n/a
    """
    train_path = 'profile_data/Profile_' + str(profile) + '.csv'
    x_train, y_train = util.load_dataset(train_path, input_col, label_col)
    x_train = tf.convert_to_tensor(x_train, dtype=tf.float32)
    y_train = tf.convert_to_tensor(y_train, dtype=tf.float32)

    input_dim = len(input_col)
    output_dim = len(label_col)

    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Dense(
        32, activation='relu', input_shape=[input_dim]))
    model.add(tf.keras.layers.Dense(32, activation='relu'))
    model.add(tf.keras.layers.Dense(output_dim, activation=None))

    model.compile(optimizer=tf.train.AdamOptimizer(
        0.01), loss='mse', metrics=['mae'])

    model.summary()

    model.fit(x_train, y_train, epochs=1, batch_size=200, steps_per_epoch=100)

    for profile in profile_test:
        validation(model, input_col, label_col, profile)

    return


if __name__ == '__main__':
    input_col = ["ambient", "coolant", "motor_speed", "i_d"]
    label_col = ["pm", "stator_yoke", "stator_tooth", "stator_winding"]
    main(profile=4,
         input_col=input_col,
         label_col=label_col,
         profile_test=[6, 10, 20])
