
import random
import math
import timeit


def einsDurchEhoch(nodeInput):
    return 1.0 / (1.0 + math.exp(-nodeInput))


class N (object):

    def __init__(self, topologieVector, learnRate, actMethod):
        self.top = topologieVector
        self.learnRate = learnRate
        self.actMethod = eval(actMethod)

        self.nod = [[1.0 for n in range(i + 1)] for i in self.top]
        #print(self.nod)
        #self.err = [[0.0 for n in range(self.top[i])    ] for i in range(len(self.top))]
        self.err = [[0.0 for n in range(i)    ] for i in self.top]
        #print(self.err)
        self.wij = [[[random.uniform(-1.0, 1.0) for j in range(self.top[i + 1])] for n in range(self.top[i] + 1)] for i in range(len(self.top) - 1)]
        #print(self.wij)

        self.tru = [0.0 for t in range(self.top[-1])]
        #print(self.tru)
        self.Nlay = len(self.top)
        print("Neuronal Network is up and ready")

    def calc(self):

        #forwardcalculation
        for nlay in range(1, self.Nlay):
            for n in range(self.top[nlay]):

                self.nod[nlay][n] = 0.0
                for nprev in range (self.top[nlay - 1] + 1):
                    self.nod[nlay][n] += self.nod[nlay - 1][nprev] * self.wij[nlay - 1][nprev][n]
                self.nod[nlay][n] = self.actMethod(self.nod[nlay][n]) #eval("eins_durch_ehoch(self.nod[nlay][n])") #self.actMethod(self.nod[nlay][n]) #1.0 / (1.0 + math.exp(-self.nod[nlay][n]))


        #backpropagation
        for n in range(self.top[self.Nlay - 1]):  # err of the output layer
            self.err[self.Nlay - 1][n] = (self.tru[n] - self.nod[self.Nlay - 1][n]) * self.nod[self.Nlay - 1][n] * (1.0 - self.nod[self.Nlay - 1][n])

        for nlay in range(self.Nlay - 2, 0, -1):  # err of the hidden layers
            for n in range(self.top[nlay]):
                self.err[nlay][n] = 0.0
                for nnext in range(self.top[nlay + 1]):
                    self.err[nlay][n] += self.err[nlay + 1][nnext] * self.wij[nlay][n][nnext]
                self.err[nlay][n] *= self.nod[nlay][n] * (1.0 - self.nod[nlay][n])

        for nlay in range(1, self.Nlay):  # wij adjustments
            for n in range(self.top[nlay]):
                for nprev in range(self.top[nlay - 1] + 1):
                    self.wij[nlay - 1][nprev][n] += self.learnRate * self.nod[nlay - 1][nprev] * self.err[nlay][n]

        return self.nod[-1] # self.Nlay - 1] #returns the output array [:-1]


n = N([2,3,1], 0.9, 'einsDurchEhoch')

#print (n.nod)
#print (n.err)
#print (n.wij)
#print (n.tru)

start = timeit.default_timer()

for it in range(1000):
    n.nod[0][0] = 0.0
    n.nod[0][1] = 0.0
    n.tru[0] = 0.0
    n.calc()
    # print(n.nod[-1][0])
    # n.calc()
    # print(n.nod[-1][0])
    # n.calc()
    # print(n.nod[-1][0])
    n.nod[0][0] = 1.0
    n.nod[0][1] = 0.0
    n.tru[0] = 1.0
    n.calc()
    n.nod[0][0] = 0.0
    n.nod[0][1] = 1.0
    n.tru[0] = 1.0
    n.calc()
    n.nod[0][0] = 1.0
    n.nod[0][1] = 1.0
    n.tru[0] = 0.0
    n.calc()

end = timeit.default_timer()

if True:
    n.nod[0][0] = 0.0
    n.nod[0][1] = 0.0
    n.tru[0] = 0.0
    print("return from calc = " + str((n.calc())[0]))
    n.nod[0][0] = 1.0
    n.nod[0][1] = 0.0
    n.tru[0] = 1.0
    print("return from calc = " + str((n.calc())[0]))
    n.nod[0][0] = 0.0
    n.nod[0][1] = 1.0
    n.tru[0] = 1.0
    print("return from calc = " + str((n.calc())[0]))
    n.nod[0][0] = 1.0
    n.nod[0][1] = 1.0
    n.tru[0] = 0.0
    print("return from calc = " + str((n.calc())[0]))

print(end-start)

