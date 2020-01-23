


def main()


if __name__ == '__main__':
    input_col = ["ambient", "coolant", "motor_speed", "i_d", "i", "u"]
    label_col = ["pm", "stator_yoke", "stator_tooth", "stator_winding"]
    main(profile=4,
        input_col=input_col,
        label_col=label_col,
        ransac=False,
        cross = False,  # cross test: True
        profile_test=6) # if cross test, profile used for test