from Neuron import *

class NeuronNet:
    def __init__(self):

        self.n = []

        for i in range(3):
            self.n.append(Neuron(2))

    def activate(self, inputs):
        return self.n[2].activate(np.array([self.n[0].activate(inputs), self.n[1].activate(inputs)]))

    def learn(self, count, inputs, answers):
        
        rate = 0.1
        count_learn = 10000

        for o in range(count_learn):
            for inputt, answer in zip(inputs, answers):

                sum_n1 = self.n[0].w[0] * inputt[0] + self.n[0].w[1] * inputt[1] + self.n[0].b
                n1 = sig(sum_n1)

                sum_n2 = self.n[1].w[0] * inputt[0] + self.n[1].w[1] * inputt[1] + self.n[1].b
                n2 = sig(sum_n2)

                sum_n3 = self.n[2].w[0] * n1 + self.n[2].w[1] * n2 + self.n[2].b
                n3 = sig(sum_n3)
                out_res = n3

                err = -2 * (answer - out_res)

                err_rate = rate * err

                deriv_sig_n1 = deriv_sig(sum_n1)
                deriv_sig_n2 = deriv_sig(sum_n2)
                deriv_sig_n3 = deriv_sig(sum_n3)

                self.n[0].w[0] -= err_rate * self.n[2].w[0] * deriv_sig_n3 * inputt[0] * deriv_sig_n1
                self.n[0].w[1] -= err_rate * self.n[2].w[0] * deriv_sig_n3 * inputt[1] * deriv_sig_n1
                self.n[0].b -= err_rate * self.n[2].w[0] * deriv_sig_n3 * deriv_sig_n1

                self.n[1].w[0] -= err_rate * self.n[2].w[1] * deriv_sig_n3 * inputt[0] * deriv_sig_n2
                self.n[1].w[1] -= err_rate * self.n[2].w[1] * deriv_sig_n3 * inputt[1] * deriv_sig_n2
                self.n[1].b -= err_rate * self.n[2].w[1] * deriv_sig_n3 * deriv_sig_n2

                self.n[2].w[0] -= err_rate * n1 * deriv_sig_n3
                self.n[2].w[1] -= err_rate * n2 * deriv_sig_n3
                self.n[2].b -= err_rate * deriv_sig_n3
