import numpy as np
import matplotlib.pyplot as plt

class IWO():
    def __init__(self, func, dim, pop_init, pop_max=50, max_iter=500
                 Smin=2, Smax=5, std_init=3, std_final=0.001, 
                 n=3, 
                 sep=False, 
                 distribution='Gaussian', 
                 truncated=False, 
                 Xmin=0., # float or np.ndarray & shape=[dim]
                 Xmax=1., # float or np.ndarray
                 init_array=None
                ):
        #--------------------------------------------input
        self.func = func
        self.dim = dim
        self.iter = max_iter
        self.pop = pop_max #最大种群数量
        self.Smin = Smin #子代最小种子数量
        self.Smax = Smax #子代最大种子数量
        self.std_init = std_init #初始标准差
        self.std_final = std_final #结束标准差
        self.n = n #调和指数
        self.sep = sep #分段标准差
        self.distribution = distribution
        if pop_init==None:
            self.pop_init = 10 * pop_max
        else:
            self.pop_init = pop_init
        self.truncated = truncated
        self.Xmin = Xmin
        self.Xmax = Xmax
        self.init_array = init_array
        #--------------------------------------------save
        self.X = np.zeros([self.pop, self.dim])
        self.Y = np.zeros([self.pop])
        self.gbest_X = np.zeros([self.max_iter, self.dim])
        self.gbest_Y = np.zeros([self.max_iter])
        self.Nseeds = np.zeros([self.pop])
    def cal_y(self, X_cal):
        Y_cal = np.zeros([X_cal.shape[0]])
        for i_cal in X_cal:
            Y_cal[i_cal] = self.func(i_cal)
        return Y_cal
    
    def init_pop(self):
        if self.init_array != None:
            self.X = self.init_array
            self.Y = self.cal_y(self.X)
        elif self.truncated: 
            # 截断分布
            if type(self.Xmin) is float and type(self.Xmax) is float:
                X_init = np.random.uniform(self.Xmin, self.Xmax, [self.pop_init, self.dim])
                Y_init = self.cal_y(X_init)
                index = np.argsort(Y_init)
                self.X = X_init[index[:self.pop]]
                self.Y = Y_init[index[:self.pop]]
            if type(self.Xmin) is np.ndarray and type(self.Xmax) is np.ndarray:
                range_ = self.Xmax - self.Xmin
                X_init = self.Xmin[np.newaxis,:] + range_[np.newaxis,:] *\
                            np.random.uniform(0, 1, [self.pop_init, self.dim])
                Y_init = self.cal_y(X_init)
                index = np.argsort(Y_init)
                self.X = X_init[index[:self.pop]]
                self.Y = Y_init[index[:self.pop]]
        else:
            # 非截断分布
            X_init = np.random.normal(0., 2*self.std_init, [self.pop_init, self.dim])
            Y_init = self.cal_y(X_init)
            index = np.argsort(Y_init)
            self.X = X_init[index[:self.pop]]
            self.Y = Y_init[index[:self.pop]]
        
    def fit(self):
        if self.truncated:
            if type(self.Xmin) is float and type(self.Xmax) is float:
                if self.sep:
                    if self.distribution=='Gaussian':
                        for i in range(self.iter): # gaussian sep truncated(float)
                            if i < self.iter/2:
                                std = self.std_final + 1.5 * (self.std_init - self.std_final) * \
                                ((self.iter - i)/float(self.iter))**self.n
                            else:
                                std = self.std_final + 0.5 * (self.std_init - self.std_final) * \
                                ((self.iter - i)/float(self.iter))**self.n
                            Nseeds = (self.Y - self.Y[0]) * (self.Smax - self.Smin)/(self.Y[-1] - self.Y[0]) + self.Smin
                            self.Nseeds = Nseeds.astype(int)
                            X = np.zeros([self.pop + self.Nseeds.sum(), self.dim])
                            X[:Ln] = self.X
                            cnt = 0
                            for ii in range(self.pop):
                                X[Ln+cnt:Ln+cnt+self.Nseeds[ii]] = np.random.normal(self.X[ii][np.newaxis,:] ,std ,
                                                                                    [self.Nseeds[ii],self.dim])
                                cnt = cnt + self.Nseeds[ii] 
                            X = np.maximum(self.Xmin, np.minimum(self.Xmax, X))
                            Y = self.cal_y(X)
                            index = np.argsort(Y)
                            self.X = X[index[:self.pop]]
                            self.Y = Y[index[:self.pop]]
                            self.gbest_X[i] = self.X[0]
                            self.gbest_Y[i] = self.Y[0]
                    elif self.distribution=='Cauchy':
                        for i in range(self.iter): # cauchy sep truncated(float)
                            if i < self.iter/2:
                                std = self.std_final + 1.5 * (self.std_init - self.std_final) * \
                                ((self.iter - i)/float(self.iter))**self.n
                            else:
                                std = self.std_final + 0.5 * (self.std_init - self.std_final) * \
                                ((self.iter - i)/float(self.iter))**self.n
                            Nseeds = (self.Y - self.Y[0]) * (self.Smax - self.Smin)/(self.Y[-1] - self.Y[0]) + self.Smin
                            self.Nseeds = Nseeds.astype(int)
                            X = np.zeros([self.pop + self.Nseeds.sum(), self.dim])
                            X[:Ln] = self.X
                            cnt = 0
                            for ii in range(self.pop):
                                X[Ln+cnt:Ln+cnt+self.Nseeds[ii]] = np.random.standard_cauchy([self.Nseeds[ii],self.dim]) * std \
                                     + self.X[ii][np.newaxis,:]
                                cnt = cnt + self.Nseeds[ii] 
                            X = np.maximum(self.Xmin, np.minimum(self.Xmax, X))
                            Y = self.cal_y(X)
                            index = np.argsort(Y)
                            self.X = X[index[:self.pop]]
                            self.Y = Y[index[:self.pop]]
                            self.gbest_X[i] = self.X[0]
                            self.gbest_Y[i] = self.Y[0]
                    else:
                        print('wrong distribution name')
                else: # not sep
                    if self.distribution=='Gaussian':
                        for i in range(self.iter): # gaussian !sep truncated(float)
                            std = self.std_final + (self.std_init - self.std_final) * \
                            ((self.iter - i)/float(self.iter))**self.n
                            Nseeds = (self.Y - self.Y[0]) * (self.Smax - self.Smin)/(self.Y[-1] - self.Y[0]) + self.Smin
                            self.Nseeds = Nseeds.astype(int)
                            X = np.zeros([self.pop + self.Nseeds.sum(), self.dim])
                            X[:Ln] = self.X
                            cnt = 0
                            for ii in range(self.pop):
                                X[Ln+cnt:Ln+cnt+self.Nseeds[ii]] = np.random.normal(self.X[ii][np.newaxis,:] ,std ,
                                                                                    [self.Nseeds[ii],self.dim])
                                cnt = cnt + self.Nseeds[ii] 
                            X = np.maximum(self.Xmin, np.minimum(self.Xmax, X))
                            Y = self.cal_y(X)
                            index = np.argsort(Y)
                            self.X = X[index[:self.pop]]
                            self.Y = Y[index[:self.pop]]
                            self.gbest_X[i] = self.X[0]
                            self.gbest_Y[i] = self.Y[0]
                    elif self.distribution=='Cauchy':
                        for i in range(self.iter): # cauchy !sep truncated(float)
                            std = self.std_final + (self.std_init - self.std_final) * \
                            ((self.iter - i)/float(self.iter))**self.n
                            Nseeds = (self.Y - self.Y[0]) * (self.Smax - self.Smin)/(self.Y[-1] - self.Y[0]) + self.Smin
                            self.Nseeds = Nseeds.astype(int)
                            X = np.zeros([self.pop + self.Nseeds.sum(), self.dim])
                            X[:Ln] = self.X
                            cnt = 0
                            for ii in range(self.pop):
                                X[Ln+cnt:Ln+cnt+self.Nseeds[ii]] = np.random.standard_cauchy([self.Nseeds[ii],self.dim]) * std \
                                     + self.X[ii][np.newaxis,:]
                                cnt = cnt + self.Nseeds[ii] 
                            X = np.maximum(self.Xmin, np.minimum(self.Xmax, X))
                            Y = self.cal_y(X)
                            index = np.argsort(Y)
                            self.X = X[index[:self.pop]]
                            self.Y = Y[index[:self.pop]]
                            self.gbest_X[i] = self.X[0]
                            self.gbest_Y[i] = self.Y[0]
                    else:
                        print('wrong distribution name')
            elif type(self.Xmin) is np.ndarray and type(self.Xmax) is np.ndarray:
                if self.sep:
                    if self.distribution=='Gaussian':
                        for i in range(self.iter): # gaussian sep truncated(array)
                            if i < self.iter/2:
                                std = self.std_final + 1.5 * (self.std_init - self.std_final) * \
                                ((self.iter - i)/float(self.iter))**self.n
                            else:
                                std = self.std_final + 0.5 * (self.std_init - self.std_final) * \
                                ((self.iter - i)/float(self.iter))**self.n
                            Nseeds = (self.Y - self.Y[0]) * (self.Smax - self.Smin)/(self.Y[-1] - self.Y[0]) + self.Smin
                            self.Nseeds = Nseeds.astype(int)
                            X = np.zeros([self.pop + self.Nseeds.sum(), self.dim])
                            X[:Ln] = self.X
                            cnt = 0
                            for ii in range(self.pop):
                                X[Ln+cnt:Ln+cnt+self.Nseeds[ii]] = np.random.normal(self.X[ii][np.newaxis,:] ,std ,
                                                                                    [self.Nseeds[ii],self.dim])
                                cnt = cnt + self.Nseeds[ii] 
                            X = np.maximum(self.Xmin[np.newaxis,:], np.minimum(self.Xmax[np.newaxis,:], X))
                            Y = self.cal_y(X)
                            index = np.argsort(Y)
                            self.X = X[index[:self.pop]]
                            self.Y = Y[index[:self.pop]]
                            self.gbest_X[i] = self.X[0]
                            self.gbest_Y[i] = self.Y[0]
                    elif self.distribution=='Cauchy':
                        for i in range(self.iter): # cauchy sep truncated(array)
                            if i < self.iter/2:
                                std = self.std_final + 1.5 * (self.std_init - self.std_final) * \
                                ((self.iter - i)/float(self.iter))**self.n
                            else:
                                std = self.std_final + 0.5 * (self.std_init - self.std_final) * \
                                ((self.iter - i)/float(self.iter))**self.n
                            Nseeds = (self.Y - self.Y[0]) * (self.Smax - self.Smin)/(self.Y[-1] - self.Y[0]) + self.Smin
                            self.Nseeds = Nseeds.astype(int)
                            X = np.zeros([self.pop + self.Nseeds.sum(), self.dim])
                            X[:Ln] = self.X
                            cnt = 0
                            for ii in range(self.pop):
                                X[Ln+cnt:Ln+cnt+self.Nseeds[ii]] = np.random.standard_cauchy([self.Nseeds[ii],self.dim]) * std \
                                     + self.X[ii][np.newaxis,:]
                                cnt = cnt + self.Nseeds[ii] 
                            X = np.maximum(self.Xmin[np.newaxis,:], np.minimum(self.Xmax[np.newaxis,:], X))
                            Y = self.cal_y(X)
                            index = np.argsort(Y)
                            self.X = X[index[:self.pop]]
                            self.Y = Y[index[:self.pop]]
                            self.gbest_X[i] = self.X[0]
                            self.gbest_Y[i] = self.Y[0]
                    else:
                        print('wrong distribution name')
                else:
                    if self.distribution=='Gaussian':
                        for i in range(self.iter):# gaussian !sep truncated(array)
                            std = self.std_final + (self.std_init - self.std_final) * \
                                ((self.iter - i)/float(self.iter))**self.n
                            Nseeds = (self.Y - self.Y[0]) * (self.Smax - self.Smin)/(self.Y[-1] - self.Y[0]) + self.Smin
                            self.Nseeds = Nseeds.astype(int)
                            X = np.zeros([self.pop + self.Nseeds.sum(), self.dim])
                            X[:Ln] = self.X
                            cnt = 0
                            for ii in range(self.pop):
                                X[Ln+cnt:Ln+cnt+self.Nseeds[ii]] = np.random.normal(self.X[ii][np.newaxis,:] ,std ,
                                                                                    [self.Nseeds[ii],self.dim])
                                cnt = cnt + self.Nseeds[ii] 
                            X = np.maximum(self.Xmin[np.newaxis,:], np.minimum(self.Xmax[np.newaxis,:], X))
                            Y = self.cal_y(X)
                            index = np.argsort(Y)
                            self.X = X[index[:self.pop]]
                            self.Y = Y[index[:self.pop]]
                            self.gbest_X[i] = self.X[0]
                            self.gbest_Y[i] = self.Y[0]
                    elif self.distribution=='Cauchy':
                        for i in range(self.iter):# cauchy !sep truncated(array)
                            std = self.std_final + (self.std_init - self.std_final) * \
                                ((self.iter - i)/float(self.iter))**self.n
                            Nseeds = (self.Y - self.Y[0]) * (self.Smax - self.Smin)/(self.Y[-1] - self.Y[0]) + self.Smin
                            self.Nseeds = Nseeds.astype(int)
                            X = np.zeros([self.pop + self.Nseeds.sum(), self.dim])
                            X[:Ln] = self.X
                            cnt = 0
                            for ii in range(self.pop):
                                X[Ln+cnt:Ln+cnt+self.Nseeds[ii]] = np.random.standard_cauchy([self.Nseeds[ii],self.dim]) * std \
                                     + self.X[ii][np.newaxis,:]
                                cnt = cnt + self.Nseeds[ii] 
                            X = np.maximum(self.Xmin[np.newaxis,:], np.minimum(self.Xmax[np.newaxis,:], X))
                            Y = self.cal_y(X)
                            index = np.argsort(Y)
                            self.X = X[index[:self.pop]]
                            self.Y = Y[index[:self.pop]]
                            self.gbest_X[i] = self.X[0]
                            self.gbest_Y[i] = self.Y[0]
                    else:
                        print('wrong distribution name')
            else:
                print('wrong type: Xmin / Xmax')
        else:
            if self.sep:
                if self.distribution=='Gaussian':
                    for i in range(self.iter): # gaussian sep !truncated
                        if i < self.iter/2:
                            std = self.std_final + 1.5 * (self.std_init - self.std_final) * \
                            ((self.iter - i)/float(self.iter))**self.n
                        else:
                            std = self.std_final + 0.5 * (self.std_init - self.std_final) * \
                            ((self.iter - i)/float(self.iter))**self.n
                        Nseeds = (self.Y - self.Y[0]) * (self.Smax - self.Smin)/(self.Y[-1] - self.Y[0]) + self.Smin
                        self.Nseeds = Nseeds.astype(int)
                        X = np.zeros([self.pop + self.Nseeds.sum(), self.dim])
                        X[:Ln] = self.X
                        cnt = 0
                        for ii in range(self.pop):
                            X[Ln+cnt:Ln+cnt+self.Nseeds[ii]] = np.random.normal(self.X[ii][np.newaxis,:] ,std ,
                                                                                [self.Nseeds[ii],self.dim])
                            cnt = cnt + self.Nseeds[ii] 
                        Y = self.cal_y(X)
                        index = np.argsort(Y)
                        self.X = X[index[:self.pop]]
                        self.Y = Y[index[:self.pop]]
                        self.gbest_X[i] = self.X[0]
                        self.gbest_Y[i] = self.Y[0]
                elif self.distribution=='Cauchy':
                    for i in range(self.iter): # cauchy sep !truncated
                        if i < self.iter/2:
                            std = self.std_final + 1.5 * (self.std_init - self.std_final) * \
                            ((self.iter - i)/float(self.iter))**self.n
                        else:
                            std = self.std_final + 0.5 * (self.std_init - self.std_final) * \
                            ((self.iter - i)/float(self.iter))**self.n
                        Nseeds = (self.Y - self.Y[0]) * (self.Smax - self.Smin)/(self.Y[-1] - self.Y[0]) + self.Smin
                        self.Nseeds = Nseeds.astype(int)
                        X = np.zeros([self.pop + self.Nseeds.sum(), self.dim])
                        X[:Ln] = self.X
                        cnt = 0
                        for ii in range(self.pop):
                            X[Ln+cnt:Ln+cnt+self.Nseeds[ii]] = np.random.standard_cauchy([self.Nseeds[ii],self.dim]) * std \
                                 + self.X[ii][np.newaxis,:]
                            cnt = cnt + self.Nseeds[ii] 
                        Y = self.cal_y(X)
                        index = np.argsort(Y)
                        self.X = X[index[:self.pop]]
                        self.Y = Y[index[:self.pop]]
                        self.gbest_X[i] = self.X[0]
                        self.gbest_Y[i] = self.Y[0]
            else: # not sep
                if self.distribution=='Gaussian':
                    for i in range(self.iter): # gaussian !sep !truncated
                        std = self.std_final + (self.std_init - self.std_final) * \
                            ((self.iter - i)/float(self.iter))**self.n
                        Nseeds = (self.Y - self.Y[0]) * (self.Smax - self.Smin)/(self.Y[-1] - self.Y[0]) + self.Smin
                        self.Nseeds = Nseeds.astype(int)
                        X = np.zeros([self.pop + self.Nseeds.sum(), self.dim])
                        X[:Ln] = self.X
                        cnt = 0
                        for ii in range(self.pop):
                            X[Ln+cnt:Ln+cnt+self.Nseeds[ii]] = np.random.normal(self.X[ii][np.newaxis,:] ,std ,
                                                                                [self.Nseeds[ii],self.dim])
                            cnt = cnt + self.Nseeds[ii] 
                        Y = self.cal_y(X)
                        index = np.argsort(Y)
                        self.X = X[index[:self.pop]]
                        self.Y = Y[index[:self.pop]]
                        self.gbest_X[i] = self.X[0]
                        self.gbest_Y[i] = self.Y[0]
                if self.distribution=='Cauchy':
                    for i in range(self.iter): # cauchy !sep !truncated
                        std = self.std_final + (self.std_init - self.std_final) * \
                            ((self.iter - i)/float(self.iter))**self.n
                        Nseeds = (self.Y - self.Y[0]) * (self.Smax - self.Smin)/(self.Y[-1] - self.Y[0]) + self.Smin
                        self.Nseeds = Nseeds.astype(int)
                        X = np.zeros([self.pop + self.Nseeds.sum(), self.dim])
                        X[:Ln] = self.X
                        cnt = 0
                        for ii in range(self.pop):
                            X[Ln+cnt:Ln+cnt+self.Nseeds[ii]] = np.random.standard_cauchy([self.Nseeds[ii],self.dim]) * std \
                                 + self.X[ii][np.newaxis,:]
                            cnt = cnt + self.Nseeds[ii] 
                        Y = self.cal_y(X)
                        index = np.argsort(Y)
                        self.X = X[index[:self.pop]]
                        self.Y = Y[index[:self.pop]]
                        self.gbest_X[i] = self.X[0]
                        self.gbest_Y[i] = self.Y[0]
                
            
    def getX(self):
        return self.X
    def getY(self):
        return self.Y
    def plot_history(self):
        plt.figure()
        plt.plot(self.gbest_Y)
        plt.show()
        
        
    