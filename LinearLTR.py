"""
EALTR: Learning to Rank Method
Using a composite differential evolution algorithm, with a linear model as the base and severity@20% as the selection criterion for the solution vector.

Key parameters for the composite differential evolution algorithm:
(1) Number of solutions in the population: np
(2) Dimensionality of the solution vector: d
(3) Number of algorithm iterations: generation
(4) Scaling factor F for mutation
(5) Crossover selection probability CR
(6) Upper and lower bounds for each element of the solution vector: U (upper) and L (lower)

Linear model: f = wx

Severity@20%: Severity of code smells in the top 20% of modules.
"""
import random
import numpy as np
import math

from PerformanceMeasure import PerformanceMeasure


class LTR:

    def __init__(self,
                 NP=100,
                 F_CR=[(1.0, 0.1), (1.0, 0.9), (0.8, 0.2)],
                 generation=100,
                 value_up_range=20.0,
                 value_down_range=-20.0,
                 x=None,
                 y=None,
                 optimized_metric="serverity@20%",
                 data_type=None
                 ):
        """
        Initialization method for the EALTR model.
        :param NP: Total number of solutions.
        :param F_CR: Set of scaling factor and crossover probability.
        :param generation: Number of iterations.
        :param value_up_range: Upper bound for each element of the solution.
        :param value_down_range: Lower bound for each element of the solution.
        :param x: Training set features.
        :param y: Training set labels.
        :param optimized_metric: Metric to optimize.
        :param data_type: Determines the dimensionality (len_x) based on data_type.
        """
        self.NP = NP
        self.F_CR = F_CR
        self.generation = generation
        # self.len_x is the dimensionality of the solution, which is determined after feature selection and one-hot encoding.
        if data_type == "GodClass":
            self.len_x = 31
        elif data_type == "DataClass":
            self.len_x = 30
        elif data_type == "LongMethod":
            self.len_x = 41
        elif data_type == "FeatureEnvy":
            self.len_x = 38
        self.value_up_range = value_up_range
        self.value_down_range = value_down_range
        self.np_list = self.initialization()
        self.training_data_x = x
        self.training_data_y = y
        self.optimized_metric = optimized_metric

    def initialization(self):
        """
        Initialization operation for the differential evolution algorithm:
        Randomly generate NP solutions (chromosomes), where each solution contains len_x elements (genes) that are randomly selected within the upper and lower bounds.
        :return: np_list
        """
        np_list = []
        for i in range(0, self.NP):
            x_list = []
            for j in range(0, self.len_x):
                x_list.append(self.value_down_range + random.random() *
                              (self.value_up_range - self.value_down_range))
            np_list.append(x_list)

        return np_list

    def substract(self, a_list, b_list):
        return [a - b for (a, b) in zip(a_list, b_list)]

    def add(self, a_list, b_list):
        return [a + b for (a, b) in zip(a_list, b_list)]

    def multiply(self, a, b_list):
        return [a * b for b in b_list]

    def random_distinct_integers(self, number, index=None):
        res = set()
        while len(res) != int(number):
            if index is not None:
                t = random.randint(0, self.NP - 1)
                if t != index:
                    res.add(t)
            else:
                res.add(random.randint(0, self.NP - 1))
        return list(res)

    def mutation_crossover_one(self, np_list):
        """
        Mutation crossover method 1:
        :param np_list: Current solution vectors.
        :return: Intermediate individuals u_list.
        """
        F_CR = random.choice(self.F_CR)
        F = F_CR[0]
        CR = F_CR[1]

        v_list = []
        for i in range(0, self.NP):
            r123 = self.random_distinct_integers(3, i)
            r1 = r123[0]
            r2 = r123[1]
            r3 = r123[2]

            sub = self.substract(np_list[r2], np_list[r3])
            mul = self.multiply(F, sub)
            add = self.add(np_list[r1], mul)

            for i in range(self.len_x):
                if add[i] > self.value_up_range or add[i] < self.value_down_range:
                    add[i] = self.value_down_range + random.random() * (
                            self.value_up_range - self.value_down_range)

            v_list.append(add)

        # crossover
        u_list = self.crossover(np_list, v_list, CR)
        return u_list

    def mutation_crossover_two(self, np_list):
        """
        Mutation crossover method 2:
        :param np_list: Current solution vectors.
        """
        F_CR = random.choice(self.F_CR)
        F = F_CR[0]
        CR = F_CR[1]
        F1 = random.random()

        v_list = []
        for i in range(0, self.NP):
            r12345 = self.random_distinct_integers(5)
            r1 = r12345[0]
            r2 = r12345[1]
            r3 = r12345[2]
            r4 = r12345[3]
            r5 = r12345[4]

            sub1 = self.substract(np_list[r2], np_list[r3])
            sub2 = self.substract(np_list[r4], np_list[r5])
            mul1 = self.multiply(F1, sub1)
            mul2 = self.multiply(F, sub2)
            add1 = self.add(np_list[r1], mul1)
            add2 = self.add(add1, mul2)

            for i in range(self.len_x):
                if add2[i] > self.value_up_range or add2[i] < self.value_down_range:
                    add2[i] = self.value_down_range + random.random() * (
                            self.value_up_range - self.value_down_range)
            v_list.append(add2)

        u_list = self.crossover(np_list, v_list, CR)
        return u_list

    def mutation_crossover_three(self, np_list):
        """
        Mutation crossover method 3 (Note: Direct scaling and addition without crossover).
        :param np_list:
        :return:
        """
        F_CR = random.choice(self.F_CR)
        F = F_CR[0]

        v_list = []
        for i in range(0, self.NP):
            r123 = self.random_distinct_integers(3)
            r1 = r123[0]
            r2 = r123[1]
            r3 = r123[2]
            sub1 = self.substract(np_list[r2], np_list[r3])
            sub2 = self.substract(np_list[r1], np_list[i])
            mul1 = self.multiply(F, sub1)
            mul2 = self.multiply(random.random(), sub2)
            add1 = self.add(mul1, mul2)
            add2 = self.add(add1, np_list[i])

            for i in range(self.len_x):
                if add2[i] > self.value_up_range or add2[i] < self.value_down_range:
                    add2[i] = self.value_down_range + random.random() * (
                            self.value_up_range - self.value_down_range)
            v_list.append(add2)

        return v_list


    def crossover(self, np_list, v_list, CR):
        """
        cross operation
        :param np_list:
        :param v_list:
        :param CR:
        :return:
        """
        u_list = []
        for i in range(0, self.NP):
            vv_list = []
            jrand = random.randint(0, self.len_x - 1)
            for j in range(0, self.len_x):
                if (random.random() <= CR) or (j == jrand):
                    vv_list.append(v_list[i][j])
                else:
                    vv_list.append(np_list[i][j])
            u_list.append(vv_list)
        return u_list

    def selection(self, u_list1, u_list2, u_list3, np_list):
        """
        Selection operation: Composite differential evolution with 3 intermediate individuals for selection.
        :param u_list1: Intermediate individual 1.
        :param u_list2: Intermediate individual 2.
        :param u_list3: Intermediate individual 3.
        :param np_list: Original individuals.
        :return: New individuals.
        """
        for i in range(0, self.NP):
            fpa1 = self.Objfunction(u_list1[i])
            fpa2 = self.Objfunction(u_list2[i])
            fpa3 = self.Objfunction(u_list3[i])
            fpa4 = self.Objfunction(np_list[i])
            max_fpa = max(fpa1, fpa2, fpa3, fpa4)
            if max_fpa == fpa1:
                np_list[i] = u_list1[i]
            elif max_fpa == fpa2:
                np_list[i] = u_list2[i]
            elif max_fpa == fpa3:
                np_list[i] = u_list3[i]
            else:
                np_list[i] = np_list[i]
        return np_list

    def Objfunction(self, Param):
        """
        Get the performance of the solution vector on the corresponding metric.
        :param Param: Solution vector.
        :return: Performance on a specific metric.
        """
        pred_y = []
        for test_x in self.training_data_x:
            pred_y.append(float(np.dot(test_x, Param)))

        if self.optimized_metric == "serverity@20%":
            performance_value = PerformanceMeasure(self.training_data_y, pred_y).severity_percentile_front(0.2)
        elif self.optimized_metric == "clc":
            performance_value = PerformanceMeasure(self.training_data_y, pred_y).CLC()
        elif self.optimized_metric == "relserverity@20%":
            performance_value = PerformanceMeasure(self.training_data_y, pred_y).relative_severity_percentile_front(0.2)
        return performance_value

    def process(self):
        np_list = self.np_list
        max_x = []
        max_f = []
        xx = []
        for i in range(0, self.NP):
            xx.append(self.Objfunction(np_list[i]))

        max_f.append(max(xx))
        max_x.append(np_list[xx.index(max(xx))])

        for i in range(0, self.generation):
            u_list1 = self.mutation_crossover_one(np_list)
            u_list2 = self.mutation_crossover_two(np_list)
            u_list3 = self.mutation_crossover_three(np_list)

            np_list = self.selection(u_list1, u_list2, u_list3, np_list)
            xx = []
            for i in range(0, self.NP):
                xx.append(self.Objfunction(np_list[i]))
            max_f.append(max(xx))
            max_x.append(np_list[xx.index(max(xx))])

        max_ff = max(max_f)

        max_xx = max_x[max_f.index(max_ff)]
        print('the maximum point x =', max_xx)
        print('the maximum value y =', max_ff)

        return max_xx

    def predict(self, testing_data_x, Param):
        pred_y = []
        for test_x in testing_data_x:
            pred_y.append(float(np.dot(test_x, Param)))

        return pred_y
