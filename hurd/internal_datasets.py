import os, pickle

from hurd.dataset import Dataset


def load_c13k_data(fb_filter="only_fb"):

    fb_filter = str(fb_filter)

    if fb_filter in ["only_fb", "no_fb", "None"]:
        fn = "data_c13k_no_amb_" + fb_filter
    elif fb_filter in ["amb_no_fb_EV_B_highest_target", "EV_B_highest_target"]:
        fn = "data_c13k_" + fb_filter
    else:
        raise ValueError("fb_filter must be 'only_fb', 'no_fb', 'EV_B_highest_target', or None")

    real_path = os.path.realpath(__file__)
    dir_path = os.path.dirname(real_path)

    load_path = os.path.join(dir_path, "datasets", fn)

    data = pickle.load(open(load_path,"rb"))

    return data
