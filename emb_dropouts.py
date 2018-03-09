import numpy as np
import torch
import sys
import rand

class Bernoulli():
    def __init__(self, p=0.3):
        #super(Bernoulli, self).__init__()
        self.p = p
        
    def __call__(self, input):
        input_size = input.size()
        input = input[0]
        # keep = np.reshape(np.argwhere(rand.rds.rand(input.size(0)) > self.p), (-1))
        # if len(keep) > 0:
            # input = input[torch.from_numpy(keep)]
        drop = np.reshape(np.argwhere(rand.rds.rand(input.size(0)) < self.p), (-1))
        if len(drop) > 0:
            input.data[torch.from_numpy(drop)] = torch.zeros(len(drop), input_size[2])
        input.data /= (1 - self.p)
        input = input.view(input_size[0], -1, input_size[2])
        return input, drop
    # return input, keep

class BernoulliDim():
    def __init__(self, p=0.3):
        #super(Bernoulli, self).__init__()
        self.p = p
        
    def __call__(self, input):
        input_size = input.size()
        input = input[0]
        drop = np.reshape(np.argwhere(rand.rds.rand(input_size[2]) < self.p), (-1))
        if len(drop) > 0:
            input.data[:,torch.from_numpy(drop)] = torch.zeros(len(drop), input_size[1])
        input = input.view(input_size[0],-1,input_size[2])
        return input, drop
    # return input, keep

    
class BernoulliReplace():
    def __init__(self, p=0.3):
        #super(Bernoulli, self).__init__()
        self.p = p
        
    def __call__(self, input, replace):
        input_size = input.size()
        input = input[0]
        keep = np.reshape(np.argwhere(rand.rds.rand(input.size(0)) > self.p), (-1))
        for i in range(0, input.size(0)):
            if i not in keep:
                input.data[i] = replace
        input = input.view(input_size[0],-1,input_size[2])
        return input

    
class BernoulliNoise():
    def __init__(self, p=0.3):
        #super(Bernoulli, self).__init__()
        self.p = p
        
    def __call__(self, input):
        input_size = input.size()
        input = input[0]
        keep = np.reshape(np.argwhere(rand.rds.rand(input.size(0)) > self.p), (-1))
        for i in range(0, input.size(0)):
            if i not in keep:
                replace_len = len(input.data[i])
                replace = torch.FloatTensor(rand.rds.uniform(-1.,1.,replace_len))
                input.data[i] = replace
        input = input.view(input_size[0],-1,input_size[2])
        return input


class Frequency():
    def __init__(self, p=0.3):
        #super(Bernoulli, self).__init__()
        self.p = p
        
    def __call__(self, input, frequency):
        input_size = input.size()
        input = input[0]
        pfreq = self.p / (frequency + self.p)
        keep = np.reshape(np.argwhere(rand.rds.rand(input.size(0)) < pfreq), (-1))
        if len(keep) > 0:
            input = input[torch.from_numpy(keep)]
        input = input.view(input_size[0],-1,input_size[2])
        return input


class FrequencyReplace():
    def __init__(self, p=0.3):
        #super(Bernoulli, self).__init__()
        self.p = p
        
    def __call__(self, input, replace, frequency):
        input_size = input.size()
        input = input[0]
        pfreq =  self.p / (frequency + self.p)
        keep = np.reshape(np.argwhere(rand.rds.rand(input.size(0)) < pfreq), (-1))
        for i in range(0, input.size(0)):
            if i not in keep:
                input.data[i] = replace
        input = input.view(input_size[0],-1,input_size[2])
        return input

    
class FrequencyNoise():
    def __init__(self, p=0.3):
        #super(Bernoulli, self).__init__()
        self.p = p
        
    def __call__(self, input, frequency):
        input_size = input.size()
        input = input[0]
        pfreq =  self.p / (frequency + self.p)
        keep = np.reshape(np.argwhere(rand.rds.rand(input.size(0)) < pfreq), (-1))
        for i in range(0, input.size(0)):
            if i not in keep:
                replace_len = len(input.data[i])
                replace = torch.FloatTensor(rand.rds.uniform(-1.,1.,replace_len))
                input.data[i] = replace
        input = input.view(input_size[0],-1,input_size[2])
        return input


class Relevance():
    def __init__(self, p=0.3):
        #super(Bernoulli, self).__init__()
        self.p = p
        
    def __call__(self, input, relevance):
        input_size = input.size()
        input = input[0]
        if sum(relevance)> 0:
            #print(relevance)
            #relevance = softmax(relevance, gamma=10)
            relevance = self.p * self.p / (self.p + relevance)
            #relevance = np.maximum(self.p, relevance)
            #relevance = [self.p + 0.5 if r > 1e-04 else self.p - 0.1 for r in relevance]
            #print(relevance)
            #import sys
            #sys.exit()
            #relevance = [self.p if r > np.mean(relevance) else 0.0 for r in relevance]
            #relevance = [1.0 if r > np.mean(relevance) else 0.0 for r in relevance]
            #relevance = [self.p + 0.1 if r > np.mean(relevance) else self.p - 0.1 for r in relevance]
            #relevance = self.p + relevance
            #relevance = self.p + (relevance - np.mean(relevance)   
        else:
            relevance += self.p

        keep = np.reshape(np.argwhere(rand.rds.rand(input.size(0)) > relevance), (-1))
        if len(keep) > 0:
            input = input[torch.from_numpy(keep)]
        input = input.view(input_size[0],-1,input_size[2])
        return input
                            

class FreqRelevance():
    def __init__(self, p=0.3):
        #super(Bernoulli, self).__init__()
        self.p = p
        
    def __call__(self, input, freqrelevance):
        print(freqrelevance[0])
        print(freqrelevance[1])
        # import sys
        # sys.exit()
        # input_size = input.size()
        # input = input[0]
        # if sum(relevance)> 0:
        #     #print(relevance)
        #     #relevance = softmax(relevance, gamma=10)
        #     #relevance = np.maximum(self.p, relevance)
        #     #relevance = [self.p + 0.5 if r > 1e-04 else self.p - 0.1 for r in relevance]
        #     #print(relevance)
        #     #import sys
        #     #sys.exit()
        #     relevance = [self.p if r > np.mean(relevance) else 0.0 for r in relevance]
        #     #relevance = [self.p + 0.1 if r > np.mean(relevance) else self.p - 0.1 for r in relevance]
        #     #relevance = self.p + relevance
        #     #relevance = self.p + (relevance - np.mean(relevance))
        # else:
        #     relevance += self.p

        # keep = np.reshape(np.argwhere(rand.rds.rand(input.size(0)) > relevance), (-1))
        # if len(keep) > 0:
        #     input = input[torch.from_numpy(keep)]
        # input = input.view(input_size[0],-1,input_size[2])
        return input

    
def softmax(x, gamma=1.0):
    x = x * gamma
    return np.exp(x) / np.sum(np.exp(x), axis=0)


def sigmoid(x, gamma=1.0):
    x = x * gamma
    return 1 / (1 + np.exp(-x))
