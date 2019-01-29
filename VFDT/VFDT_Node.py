from itertools import combinations
import numpy as np


# VFDT node class
class VfdtNode:
    def __init__(self, possible_split_features):
        """
        nijk: statistics of feature i, value j, class
        :list possible_split_features: features
        """
        self.parent = None
        self.left_child = None
        self.right_child = None
        self.split_feature = None
        self.split_value = None  # both continuous and discrete value
        self.new_examples_seen = 0
        self.total_examples_seen = 0
        self.class_frequency = {}
        self.nijk = {f: {} for f in possible_split_features}
        self.possible_split_features = possible_split_features

    def add_children(self, split_feature, split_value, left, right):
        self.split_feature = split_feature
        self.split_value = split_value
        self.left_child = left
        self.right_child = right
        left.parent = self
        right.parent = self

        self.nijk.clear()  # reset stats
        if isinstance(split_value, list):
            left_value = split_value[0]
            right_value = split_value[1]
            # discrete split value list's length = 1, stop splitting
            if len(left_value) <= 1:
                new_features = [None if f == split_feature else f for f in left.possible_split_features]
                left.possible_split_features = new_features
            if len(right_value) <= 1:
                new_features = [None if f == split_feature else f for f in right.possible_split_features]
                right.possible_split_features = new_features

    def is_leaf(self):
        return self.left_child is None and self.right_child is None

    # recursively trace down the tree to distribute data examples to corresponding leaves
    def sort_example(self, x):
        if self.is_leaf():
            return self
        else:
            index = self.possible_split_features.index(self.split_feature)
            value = x[index]
            split_value = self.split_value

            if isinstance(split_value, list):  # discrete value
                if value in split_value[0]:
                    return self.left_child.sort_example(x)
                else:
                    return self.right_child.sort_example(x)
            else:  # continuous value
                if value <= split_value:
                    return self.left_child.sort_example(x)
                else:
                    return self.right_child.sort_example(x)

    # the most frequent class
    def most_frequent(self):
        try:
            prediction = max(self.class_frequency, key=self.class_frequency.get)
        except ValueError:
            # if self.class_frequency dict is empty, go back to parent
            class_frequency = self.parent.class_frequency
            prediction = max(class_frequency, key=class_frequency.get)
        return prediction

    # update leaf stats in order to calculate infomation gain
    def update_stats(self, x, y):
        feats = self.possible_split_features
        nijk = self.nijk
        iterator = [f for f in feats if f is not None]
        for i in iterator:
            value = x[feats.index(i)]
            if value not in nijk[i]:
                nijk[i][value] = {y: 1}
            else:
                try:
                    nijk[i][value][y] += 1
                except KeyError:
                    nijk[i][value][y] = 1

        self.total_examples_seen += 1
        self.new_examples_seen += 1
        class_frequency = self.class_frequency
        try:
            class_frequency[y] += 1
        except KeyError:
            class_frequency[y] = 1

    def check_not_splitting(self):
        # compute gini index for not splitting
        X0 = 1
        class_frequency = self.class_frequency
        n = sum(class_frequency.values())
        for j, k in class_frequency.items():
            X0 -= (k / n) ** 2
        return X0

    # use Hoeffding tree model to test node split, return the split feature
    def attempt_split(self, delta, nmin, tau):
        if self.new_examples_seen < nmin:
            return None
        class_frequency = self.class_frequency
        if len(class_frequency) == 1:
            return None

        self.new_examples_seen = 0  # reset
        nijk = self.nijk
        min = 1
        second_min = 1
        Xa = ''
        split_value = None
        for feature in self.possible_split_features:
            if feature is not None:
                njk = nijk[feature]
                gini, value = self.gini(njk, class_frequency)
                if gini < min:
                    min = gini
                    Xa = feature
                    split_value = value
                elif min < gini < second_min:
                    second_min = gini

        epsilon = self.hoeffding_bound(delta)
        g_X0 = self.check_not_splitting()
        if min < g_X0:
            # print(second_min - min, epsilon)
            if second_min - min > epsilon:
                # print('1 node split')
                return [Xa, split_value]
            elif second_min - min < epsilon < tau:
                # print('2 node split')
                return [Xa, split_value]
            else:
                return None
        return None

    def hoeffding_bound(self, delta):
        n = self.total_examples_seen
        R = np.log(len(self.class_frequency))
        return np.sqrt(R * R * np.log(1 / delta) / (2 * n))

    def gini(self, njk, class_frequency):
        # gini(D) = 1 - Sum(pi^2)
        # gini(D, F=f) = |D1|/|D|*gini(D1) + |D2|/|D|*gini(D2)

        D = self.total_examples_seen
        m1 = 1  # minimum gini
        # m2 = 1  # second minimum gini
        Xa_value = None
        feature_values = list(njk.keys())  # list() is essential
        if not isinstance(feature_values[0], str):  # numeric  feature values
            sort = np.array(sorted(feature_values))
            split = (sort[0:-1] + sort[1:]) / 2  # vectorized computation, like in R

            D1_class_frequency = {j: 0 for j in class_frequency.keys()}
            for index in range(len(split)):
                nk = njk[sort[index]]
                for j in nk:
                    D1_class_frequency[j] += nk[j]
                D1 = sum(D1_class_frequency.values())
                D2 = D - D1
                g_d1 = 1
                g_d2 = 1

                D2_class_frequency = {}
                for key, value in class_frequency.items():
                    if key in D1_class_frequency:
                        D2_class_frequency[key] = value - D1_class_frequency[key]
                    else:
                        D2_class_frequency[key] = value

                for key, v in D1_class_frequency.items():
                    g_d1 -= (v / D1) ** 2
                for key, v in D2_class_frequency.items():
                    if not D2 == 0:
                        g_d2 -= (v / D2) ** 2
                g = g_d1 * D1 / D + g_d2 * D2 / D
                if g < m1:
                    m1 = g
                    Xa_value = split[index]
                # elif m1 < g < m2:
                # m2 = g
            return [m1, Xa_value]

        else:  # discrete feature_values
            length = len(njk)
            if length > 10:  # too many discrete feature values, estimate
                for j, k in njk.items():
                    D1 = sum(k.values())
                    D2 = D - D1
                    g_d1 = 1
                    g_d2 = 1

                    D2_class_frequency = {}
                    for key, value in class_frequency.items():
                        if key in k:
                            D2_class_frequency[key] = value - k[key]
                        else:
                            D2_class_frequency[key] = value
                    for key, v in k.items():
                        g_d1 -= (v / D1) ** 2

                    if D2 != 0:
                        for key, v in D2_class_frequency.items():
                            g_d2 -= (v / D2) ** 2
                    g = g_d1 * D1 / D + g_d2 * D2 / D
                    if g < m1:
                        m1 = g
                        Xa_value = j
                    # elif m1 < g < m2:
                    # m2 = g
                right = list(np.setdiff1d(feature_values, Xa_value))

            else:  # fewer discrete feature values, get combinations
                comb = self.select_combinations(feature_values)
                for i in comb:
                    left = list(i)
                    D1_class_frequency = {key: 0 for key in class_frequency.keys()}
                    D2_class_frequency = {key: 0 for key in class_frequency.keys()}
                    for j, k in njk.items():
                        for key, value in class_frequency.items():
                            if j in left:
                                if key in k:
                                    D1_class_frequency[key] += k[key]
                            else:
                                if key in k:
                                    D2_class_frequency[key] += k[key]
                    g_d1 = 1
                    g_d2 = 1
                    D1 = sum(D1_class_frequency.values())
                    D2 = D - D1
                    for key, v in D1_class_frequency.items():
                        g_d1 -= (v / D1) ** 2
                    for key, v in D2_class_frequency.items():
                        g_d2 -= (v / D2) ** 2
                    g = g_d1 * D1 / D + g_d2 * D2 / D
                    if g < m1:
                        m1 = g
                        Xa_value = left
                    # elif m1 < g < m2:
                    # m2 = g
                right = list(np.setdiff1d(feature_values, Xa_value))
            return [m1, [Xa_value, right]]

    # divide values into two groups, return the combination of left groups
    def select_combinations(self, feature_values):
        combination = []
        e = len(feature_values)
        if e % 2 == 0:
            end = int(e / 2)
            for i in range(1, end + 1):
                if i == end:
                    cmb = list(combinations(feature_values, i))
                    enough = int(len(cmb) / 2)
                    combination.extend(cmb[:enough])
                else:
                    combination.extend(combinations(feature_values, i))
        else:
            end = int((e - 1) / 2)
            for i in range(1, end + 1):
                combination.extend(combinations(feature_values, i))

        return combination
