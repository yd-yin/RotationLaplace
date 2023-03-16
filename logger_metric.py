import numpy as np


def get_errors_helper(error_list, class_list, class_enum):
    errors = {}  # list per class
    for c in class_enum:
        errors[str(c)] = []
    for e, c in zip(error_list, class_list):
        errors[str(class_enum(c))].append(e)
    per_class_stats = {}
    ang30s = []
    ang20s = []
    ang15s = []
    ang10s = []
    ang7_5s = []
    ang5s = []
    ang3s = []
    medians_list = []
    mean_list = []
    for classname, values in errors.items():
        if len(values) > 0:
            values = np.array(values)
            ang30 = np.mean(values < 30)
            ang20 = np.mean(values < 20)
            ang15 = np.mean(values < 15)
            ang10 = np.mean(values < 10)
            ang7_5 = np.mean(values < 7.5)
            ang5 = np.mean(values < 5)
            ang3 = np.mean(values < 3)
            median = np.median(values)
            mean = np.mean(values)

            medians_list.append(median)
            mean_list.append(mean)
            ang30s.append(ang30)
            ang20s.append(ang20)
            ang15s.append(ang15)
            ang10s.append(ang10)
            ang7_5s.append(ang7_5)
            ang5s.append(ang5)
            ang3s.append(ang3)
            per_class_stats[classname] = (median, mean, ang30, ang20, ang15, ang10, ang7_5, ang5, ang3, values)
    return [np.mean(medians_list), np.mean(mean_list), np.mean(ang30s), np.mean(ang20s), np.mean(ang15s), np.mean(ang10s), np.mean(ang7_5s), np.mean(ang5s), np.mean(ang3s), error_list], per_class_stats


def get_errors(error_list, class_list, hard_list, class_enum):
    easy_error_list = []
    easy_class_list = []
    for e, c, h in zip(error_list, class_list, hard_list):
        if not h:
            easy_error_list.append(e)
            easy_class_list.append(c)
    easy_stats = get_errors_helper(easy_error_list, easy_class_list, class_enum)
    stats = get_errors_helper(error_list, class_list, class_enum)
    return easy_stats, stats


def print_stats(stats, verbose=False):
    print('mean over median: {}'.format(stats[0][0]))
    print('angle 30: {}'.format(stats[0][1]))
    if verbose:
        print('angle 15: {}'.format(stats[0][2]))
        print('angle 7.5: {}'.format(stats[0][3]))
        for name, median in stats[1].items():
            print('{} median: {}'.format(name, median))
