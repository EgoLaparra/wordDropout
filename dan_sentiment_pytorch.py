import sys
import pickle as pkl
import math
import os
import numpy as np
import cPickle
import time
import argparse
from collections import Counter
from scipy.stats import percentileofscore

import torch
import torch.nn as nn
from torch.autograd import Variable
from torch import optim

import emb_dropouts as embd
import rand

# keeplist = pkl.load(open('keep.pkl', 'rb'))
# vocab = pkl.load(open('aux_experiments/vocab.pkl', 'rb'))
def word(sent):
    out = list()
    for q in sent:
        for w, i in vocab.items():
            if i == q:
                out.append(w)
    return out

class DAN(nn.Module):
    def __init__(self, len_voc, args):
        super(DAN, self).__init__()
        emb_size = args['d']
        hidden_size = args['dh']
        labels = args['labels']
        deep = args['deep']
        drop = args['drop']
        dmethod = args['drop_method']
        dnet = args['net_drop']
        self.drops = np.zeros(len_voc)
        self.embs = nn.Embedding(len_voc, emb_size)
        if dnet == 1:
            self.netdropout = nn.Dropout(p=drop)
        elif dnet == 2:
            self.netdropout = nn.AlphaDropout(p=drop)
        if dmethod == 3:
            self.dropout = embd.Bernoulli(p=drop)
            #self.dropout = embd.BernoulliReplace(p=drop)
            #self.dropout = embd.BernoulliNoise(p=drop)
        elif dmethod == 4:
            self.dropout = embd.Frequency(p=drop)
            #self.dropout = embd.FrequencyReplace(p=drop)
            #self.dropout = embd.FrequencyNoise(p=drop)
        elif dmethod == 5 or dmethod == 6:
            self.dropout = embd.Relevance(p=drop)
        elif dmethod == 7:
            self.dropout = embd.FreqRelevance(p=drop)
        elif dmethod == 8:
            self.dropout = embd.BernoulliDim(p=drop)
        elif dmethod == 9:
            self.dropout = embd.Bernoulli(p=drop)
        self.linear = nn.ModuleList()
        self.relu = nn.ModuleList()
        for i in range(0, deep):
            if i == 0:
                self.linear.append(nn.Linear(emb_size, hidden_size))
            else:
                self.linear.append(nn.Linear(hidden_size, hidden_size))
            self.relu.append(nn.ReLU())
        self.toplinear = nn.Linear(hidden_size, labels)
        self.top = nn.LogSoftmax()

    def forward(self, input, dmethod=None, dnet=None, drop_criterion=None, lrp=None):
        # print(word(input.data.numpy()[0]))
        outemb = self.embs(input)
        # idx = list()
        # for i in range(0, len(input.data.numpy()[0])):
        #     if  input.data.numpy()[0][i] in keeplist:
        #         idx.append(i)
        if self.training:
            if dmethod == 3 or dmethod == 8 or dmethod == 9:
                outemb, keep = self.dropout(outemb)
            elif dmethod == 4:
                outemb = self.dropout(outemb, drop_criterion)
                #unk = self.embs.weight.data[-1]
                #idx = np.random.randint(0, high=self.embs.weight.data.size(0), size=1)[0]
                #unk = self.embs.weight.data[idx]
                #out = self.dropout(out, unk)
                #outemb = self.dropout(outemb, unk, drop_criterion)
            elif dmethod == 5 or dmethod == 6 or dmethod == 7:
                outemb = self.dropout(outemb, drop_criterion)

            #mask = np.ones(len(input.data.numpy()[0]), dtype=bool)
            #mask[keep] = False
            #droped = input.data.numpy()[0][mask]
            #self.drops[droped] += 1

        # wcriterion = drop_criterion / np.max(drop_criterion)
        # wcriterion_len = len(wcriterion)
        # wcriterion = np.repeat(wcriterion, 300).reshape(wcriterion_len, 300)
        # wcriterion = Variable(torch.FloatTensor(wcriterion))
        # outemb = outemb.view(outemb.size(1), outemb.size(2))
        # outmean = torch.sum(torch.mul(outemb, wcriterion), 0).view(1, -1)

        outmean = torch.mean(outemb, 1).view(1, -1)
        outrelus = list()
        for i in range(0, len(self.linear)):
            if len(outrelus) == 0:
                if dnet == 1 or dnet == 2:
                    outmean = self.netdropout(outmean)
                outrelus.append(self.relu[i](self.linear[i](outmean)))
            else:
                if dnet == 1 or dnet == 2:
                    outrelus[i-1] = self.netdropout(outrelus[i-1])
                outrelus.append(self.relu[i](self.linear[i](outrelus[i-1])))
        if dnet == 1 or dnet == 2:
            outrelus[-1] = self.netdropout(outrelus[-1])
        out = self.top(self.toplinear(outrelus[-1]))
        
        if lrp is not None:
            tinny = 1e-16
            R = out[0]
            for i in range(len(outrelus)-1, -1, -1):
                if i == len(outrelus)-1:
                    R = torch.mul(torch.div(torch.mul(outrelus[i][0], self.toplinear.weight).transpose(0,1), torch.add(out[0], tinny)), R).sum(1)
                else:
                    R = torch.mul(torch.div(torch.mul(outrelus[i][0], self.linear[i+1].weight).transpose(0,1), torch.add(outrelus[i+1][0], tinny)), R).sum(1)
            R = torch.mul(torch.div(torch.mul(outmean[0], self.linear[0].weight).transpose(0,1), torch.add(outrelus[0][0], tinny)), R).sum(1)
            if lrp == 0:
                R = (outemb[0] * R / outemb[0].sum(0)).sum(1)
            elif lrp == 1:
                R = (outemb[0] * R / outemb[0].sum(0)).sum(0)
            return R
        else:
            return out

        
def train(net, data, drop_criterion, args):
    criterion = nn.NLLLoss(size_average=False)
    parameters = filter(lambda p: p.requires_grad, net.parameters())
    optimizer = optim.Adagrad(parameters, lr=args['lr'])

    min_error = float('inf')
    # log = open(log_file, 'w')
    for epoch in range(0, args['num_epochs']):

        # lstring = ''

        # create mini-batches
        rand.rds.shuffle(data)
        batches = [data[x: x + args['batch_size']] for x in xrange(0, len(data),
                                                                   args['batch_size'])]

        epoch_error = 0.0
        ep_t = time.time()
        for batch_ind, batch in enumerate(batches):
            now = time.time()

            # Clear gradients
            net.zero_grad()

            blabel = list()
            bpred = torch.FloatTensor()
            for sent, label in batch:

                if len(sent) == 0:
                    continue

                blabel.append(label)


                if args['drop_method'] == 7:
                    sent_drop_criterion = [drop_criterion[0][sent], drop_criterion[1][sent]]
                else:
                    sent_drop_criterion =  drop_criterion[sent]
                sent = Variable(torch.LongTensor([sent]))
                pred = net(sent, dmethod=args['drop_method'], dnet=args['net_drop'], drop_criterion=sent_drop_criterion)

                if bpred.dim() == 0:
                    bpred = pred
                else:
                    bpred = torch.cat((bpred, pred), 0)

            by = Variable(torch.LongTensor(blabel))

            # Compute loss
            err = criterion(bpred, by)


            for e,p in enumerate(net.parameters()):

                # don't regularize embeddings if finetune=false
                if not args['ft'] and e == 0:
                    continue
                err += 0.5 * args['rho'] * (p.data ** 2).sum()
            err = err / len(batch)
            
            err.backward()
            optimizer.step()                                                        

            err = err.data.numpy()[0]

            lstring = 'epoch: ' + str(epoch) + ' batch_ind: ' + str(batch_ind) + \
                      ' error, ' + str(err) + ' time = ' + str(time.time() - now) + ' sec'
            # log.write(lstring + '\n')
            # log.flush()
            epoch_error += err

        # done with epoch
        print(time.time() - ep_t)
        print 'done with epoch ', epoch, ' epoch error = ', epoch_error, ' min error = ', min_error
        lstring = 'done with epoch ' + str(epoch) + ' epoch error = ' + str(epoch_error) \
                  + ' min error = ' + str(min_error) + '\n'
        # log.write(lstring)
        # log.flush()

        # save parameters if the current model is better than previous best model
        if epoch_error < min_error:
            min_error = epoch_error
            torch.save(net.state_dict(), args['output'])

        # log.flush()

        # net.eval()
        # emb_relevance = lrp(net, data)
        # f = open(args['relevances'] + '.' + str(epoch), 'wb')
        # pkl.dump(emb_relevance, f)
        # f.close()
        # net.train()

        

def lrp(net, data, dim=0):

    dim = 1
    over_lrp = np.zeros(net.embs.weight.size(dim))
    over_total = np.zeros(net.embs.weight.size(dim)) + 1e-16

    for sent, label in data:

        if len(sent) == 0:
            continue

        sent = Variable(torch.LongTensor([sent]))
        pred = net(sent,lrp=dim)

        if dim == 0:
            for e,w in enumerate(sent.view(-1).data):
                over_lrp[w] += abs(pred.data[e])
                over_total[w] += 1
        elif dim == 1:
            over_lrp += abs(pred.data.numpy())
            over_total += 1
            
    over_lrp = over_lrp / over_total

    #for o in range(0, len(over_lrp)):
    #    over_lrp[o] = percentileofscore(over_lrp, over_lrp[o]) / 100
    return over_lrp


def sga(net, data):

    over_sga = np.zeros(net.embs.weight.size(0))
    over_total = np.zeros(net.embs.weight.size(0)) + 1e-16
    
    for sent, label in data:

        if len(sent) == 0:
            continue

        net.zero_grad()
        
        sent = Variable(torch.LongTensor([sent]))
        pred = net(sent)

        amax = np.argmax(pred.data.numpy())
        pred[0][amax].backward()
        io_sga = (net.embs.weight.grad ** 2).sum(1).data.numpy()
        over_sga += io_sga     

        for w in sent.view(-1).data:
            over_total[w] += 1

    over_lrp = over_lrp / over_total
    #for o in range(0, len(over_sga)):
    #    over_sga[o] = percentileofscore(over_sga, over_sga[o]) / 100
    
    return over_sga

        

def validate(net, data, fold, criterion=None):

    correct = 0.
    total = 0.

    prediction = list()
    for sent, label in data:

        if len(sent) == 0:
            continue

        sent_criterion =  None
        if criterion is not None:
            sent_criterion =  criterion[sent]

        sent = Variable(torch.LongTensor([sent]))
        pred = net(sent, drop_criterion=sent_criterion)
        prediction.append(pred.data.numpy())
        
        if np.argmax(pred.data.numpy()) == label:
            correct += 1

        total += 1

    print 'accuracy on ', fold, correct, total, str(correct / total), '\n'
    return (correct / total, prediction)


def softmax(x, gamma=1.0):
    x = x * gamma
    return np.exp(x) / np.sum(np.exp(x), axis=0)


if __name__ == '__main__':

    # command line arguments
    parser = argparse.ArgumentParser(description='sentiment DAN')
    parser.add_argument('-data', help='location of dataset', default='/work/laparra/Tools/dan/data/sentiment/')
    parser.add_argument('-vocab', help='location of vocab', default='/work/laparra/Tools/dan/data/sentiment/wordMapAll.bin')
    parser.add_argument('-We', help='location of word embeddings', default='/work/laparra/Tools/dan/data/sentiment_all_We')
    parser.add_argument('-rand_We', help='randomly init word embeddings', type=int, default=0)
    parser.add_argument('-binarize', help='binarize labels', type=int, default=0)
    parser.add_argument('-d', help='word embedding dimension', type=int, default=300)
    parser.add_argument('-dh', help='hidden dimension', type=int, default=300)
    parser.add_argument('-deep', help='number of layers', type=int, default=3)
    parser.add_argument('-drop', help='dropout probability', type=float, default=0.3)
    parser.add_argument('-rho', help='regularization weight', type=float, default=1e-4)
    parser.add_argument('-labels', help='number of labels', type=int, default=5)
    parser.add_argument('-ft', help='fine tune word vectors', type=int, default=1)
    parser.add_argument('-b', '--batch_size', help='adagrad minibatch size (ideal: 25 minibatches \
                        per epoch). for provided datasets, x for history and y for lit', type=int,\
                        default=15)
    parser.add_argument('-ep', '--num_epochs', help='number of training epochs, can also determine \
                         dynamically via validate method', type=int, default=5)
    parser.add_argument('-agr', '--adagrad_reset', help='reset sum of squared gradients after this many\
                         epochs', type=int, default=50)
    parser.add_argument('-lr', help='adagrad initial learning rate', type=float, default=0.005)
    parser.add_argument('-e', '--experiment', help='directory to store experiment', \
                        default='experiments/sst/nodroput/')
    parser.add_argument('-o', '--output', help='output model file name', \
                         default='sentiment_params.pkl')
    parser.add_argument('-r', '--relevances', help='relevance values file name', \
                        default=None)
    parser.add_argument('-data_name', help='dataset name', default='sts')
    parser.add_argument('-fold', help='fold number', type=int, default=0)
    parser.add_argument('-drop_method', help='dropout method', type=int, default=0)
    parser.add_argument('-net_drop', help='network dropout method', type=int, default=0)
    parser.add_argument('-runs', help='number of runs', type=int, default=1)
    parser.add_argument('-phrase', help='access phrase annotations', type=int, default=0)
    
    args = vars(parser.parse_args())
    d = args['d']
    dh = args['dh']

    if not os.path.exists(args['experiment']):
            os.makedirs(args['experiment'])
    args['output'] = os.path.join(args['experiment'], args['output'])
    if args['relevances'] is not None:
        args['relevances'] = os.path.join(args['experiment'], args['relevances'])
    
    print 'dropout method %d; net dropout method %d; %d runs' % (args['drop_method'], args['net_drop'], args['runs'])
    
    # load data
    if args['data_name'] == 'imdb':
        data = cPickle.load(open(args['data'], 'rb'))
        data_train = data[0]
        data_dev = []
        data_test = data[1]
        vocab = data[2]
        len_voc = len(vocab)
        freq_voc = np.zeros(len(vocab))
    elif args['data_name'] == 'rt':
        data = cPickle.load(open(args['data'], 'rb'))
        data_train = []
        data_dev = []
        for i in range(0, len(data[0])):
            if i == args['fold']:
                data_test = data[0][i]
            else:
                data_train.extend(data[0][i])
        vocab = data[1]
        len_voc = len(vocab)
        freq_voc = np.zeros(len(vocab))
    else:
        if args['phrase']:
            data_train = cPickle.load(open(args['data']+'train-allphrases', 'rb'))
        else:
            data_train = cPickle.load(open(args['data']+'train-rootfine', 'rb'))
        data_dev = cPickle.load(open(args['data']+'dev-rootfine', 'rb'))
        data_test = cPickle.load(open(args['data']+'test-rootfine', 'rb'))
        vocab = cPickle.load(open(args['vocab'], 'rb'))
        len_voc = len(vocab)
        freq_voc = np.zeros(len(vocab))
            
        if args['binarize']:
            data_train = [[t[0], 0 if t[1] < 2 else 1] for t in data_train if t[1] < 2 or t[1] > 2]
            data_dev = [[t[0], 0 if t[1] < 2 else 1] for t in data_dev if t[1] < 2 or t[1] > 2]
            data_test = [[t[0], 0 if t[1] < 2 else 1] for t in data_test if t[1] < 2 or t[1] > 2]
    
    for split in [data_train, data_dev, data_test]:
        c = Counter()
        tot = 0
        for sent, label in split:
            freq_voc[sent] += 1
            c[label] += 1
            tot += 1
        print(c, tot)

    #f = open(os.path.join(args['experiment'], 'vocab.pkl'), 'wb')
    #pkl.dump(vocab, f)
    #f.close()
        
    #f = open(os.path.join(args['experiment'], 'freq.pkl'), 'wb')
    #pkl.dump(freq_voc, f)
    #f.close()

    over_acc = 0
    for i in range(0, args['runs']):
        print 'run: %d' % (i + 1)
        rand.initSeed()
        # torch.manual_seed(12345)

        net = DAN(len_voc, args)
        orig_We = cPickle.load(open(args['We'], 'rb'))
        orig_We = np.transpose(orig_We[:d])
        net.embs.weight.data.copy_(torch.from_numpy(np.array(orig_We)))
        
        net.train()
        if args['drop_method'] == 5 or args['drop_method'] == 6 or args['drop_method'] == 7:
            emb_relevance = np.zeros(len_voc)
            if args['relevances'] == None or not os.path.isfile(args['relevances']):
                #net.dropout = embd.Bernoulli(p=args['drop']) #####
                #args['drop_method'] = 3 ####
                train(net, data_train, emb_relevance, args)
                print('calculating relevance...')
                net.eval()
                if args['drop_method'] == 5:
                   emb_relevance = sga(net, data_train)
                else:
                   emb_relevance = lrp(net, data_train, dim=0)
                if args['relevances'] is not None:
                    f = open(args['relevances'], 'wb')
                    pkl.dump(emb_relevance, f)
                    f.close()
            else:
                emb_relevance = pkl.load(open(args['relevances'], 'rb'))
            net = DAN(len_voc, args)
            net.embs.weight.data.copy_(torch.from_numpy(np.array(orig_We)))
            net.train()
            #net.dropout = embd.Relevance(p=args['drop']) #####
            #args['drop_method'] = 6 ####
            if args['drop_method'] == 7:
                train(net, data_train, [freq_voc, emb_relevance], args)
            else:
                train(net, data_train, emb_relevance, args)
        if args['drop_method'] == 9:
            emb_relevance = np.zeros(len_voc)
            if args['relevances'] == None or not os.path.isfile(args['relevances']):
                #net.dropout = embd.Bernoulli(p=args['drop']) #####
                #args['drop_method'] = 3 ####
                train(net, data_train, emb_relevance, args)
                print('calculating relevance...')
                net.eval()
                if args['drop_method'] == 5:
                   emb_relevance = sga(net, data_train)
                else:
                   emb_relevance = lrp(net, data_train, dim=0)
                if args['relevances'] is not None:
                    f = open(args['relevances'], 'wb')
                    pkl.dump(emb_relevance, f)
                    f.close()
            else:
                emb_relevance = pkl.load(open(args['relevances'], 'rb'))
            keep = np.argsort(emb_relevance)[150:]
            reduced_We = orig_We[:,keep]
            args['d'] = np.shape(reduced_We)[1]
            net = DAN(len_voc, args)
            net.embs.weight.data.copy_(torch.from_numpy(np.array(reduced_We)))
            net.train()
            train(net, data_train, freq_voc, args)
            args['d'] = 300            
            # keep = np.argsort(emb_relevance)[150:]
            # #orig_We = orig_We[:,keep]
            # orig_We = net.embs.weight.data.numpy()[:,keep]
            # args['d'] = np.shape(orig_We)[1]
            # linear0 = net.linear[0].weight.data.numpy()[:,keep]
            # state_dict = net.state_dict()
            # net = DAN(len_voc, args)
            # state_dict["embs.weight"] = torch.from_numpy(np.array(orig_We))
            # state_dict["linear.0.weight"] = torch.from_numpy(np.array(linear0))
            # net.load_state_dict(state_dict)
            # torch.save(net.state_dict(), args['output'])
            # #net.embs.weight.data.copy_(torch.from_numpy(np.array(orig_We)))
            # #net.linear[0].weight.data.copy_(torch.from_numpy(np.array(linear0)))
            # #net.train()
            # #net.dropout = embd.Relevance(p=args['drop']) #####
            # #args['drop_method'] = 6 ####
            # #train(net, data_train, freq_voc, args)
        else:
            train(net, data_train, freq_voc, args)

            
        #f = open(os.path.join(args['experiment'], 'drops.pkl'), 'wb')
        #pkl.dump(net.drops, f)
        #f.close()
        

        #emb_relevance = pkl.load(open(args['relevances'], 'rb')) ######
        
        net.load_state_dict(torch.load(args['output']))
        net.eval()
        validation = validate(net, data_test, 'test')
        over_acc += validation[0]
        
        #validation = validate(net, data_dev, 'test', criterion=emb_relevance)
        #validation = validate(net, data_dev, 'test')
        #over_acc += validation[0]
        f = open(os.path.join(args['experiment'], 'prediction.pkl'), 'wb')
        pkl.dump(validation[1], f)
        f.close()
        #emb_relevance = lrp(net, data_dev)
        #f = open(os.path.join(args['experiment'], 'dev_rel.pkl'), 'wb')
        #pkl.dump(emb_relevance, f)
        #f.close()
        

    print '\nmean accuracy: %.3f\n' % (over_acc / args['runs'])

