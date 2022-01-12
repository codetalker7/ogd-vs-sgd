from GD import GD
import numpy as np
import time

class SGD(GD):
    def train(self, T, lam):
        '''
            takes parameters T and lam
            T: number of time steps
            lam: hyperparameter to control regularization
        '''
        #set starting time
        starttime = time.time()

        #theta is the current parameter vector
        #theta_final is the final parameter vector
        theta = self.zeros 
        theta_final = self.zeros

        d, n = self.dimension, self.size

        for t in range(1, T + 1):
            #update theta_final
            theta_final = theta_final + theta

            #pick a random point
            i = np.random.randint(0, n)

            #get the ith data point
            ai = np.transpose(self.data[i:i + 1, :])
            
            #get the ith label
            #class1 -> -1, class2 -> +1
            if (self.labels[i] == self.class1):
                bi = -1
            else:
                bi = 1

            #compute gradient
            hinge_gradient = self.hingeLossGradient(theta, ai, bi)
            nabla_t = (lam)*hinge_gradient + theta

            #compute x_{t + 1}
            theta = theta - (2/(t + 1))*nabla_t

        #return the average of thetas
        self.theta = theta_final/T

        #set and print trainingTime
        self.trainingTime = time.time() - starttime