import torch
import torch.nn as nn
import numpy as np
import torchvision.models as models
from torch.autograd import Variable
import torch.nn.functional as F
import itertools
import operator
from multiprocessing.dummy import Pool as ThreadPool
from multiprocessing import Pool
from collections import Counter

class EncoderCNN(nn.Module):
    def __init__(self, target_size):
        super(EncoderCNN, self).__init__()

        resnet = models.resnet152(pretrained=True)
        modules = list(resnet.children())[:-1]
        self.resnet = nn.Sequential(*modules)
        for param in self.resnet.parameters():
            param.requires_grad = False

        self.linear = nn.Linear(resnet.fc.in_features, target_size)
        self.bn = nn.BatchNorm1d(target_size, momentum=0.01)
        self.init_weights()

    def get_params(self):
        return list(self.linear.parameters()) + list(self.bn.parameters())

    def init_weights(self):
        self.linear.weight.data.normal_(0.0, 0.02)
        self.linear.bias.data.fill_(0)

    def forward(self, images):
        features = self.resnet(images)
        features = Variable(features.data)
        features = features.view(features.size(0), -1)
        features = self.linear(features)
        features = self.bn(features)
        return features


class EncoderStory(nn.Module):
    def __init__(self, img_feature_size, hidden_size, n_layers):
        super(EncoderStory, self).__init__()

        self.hidden_size = hidden_size
        self.n_layers = n_layers
        self.cnn = EncoderCNN(img_feature_size)
        self.lstm = nn.LSTM(img_feature_size, hidden_size, n_layers, batch_first=True, bidirectional=True, dropout=0.5)
        self.linear = nn.Linear(hidden_size * 2 + img_feature_size, hidden_size * 2)
        self.dropout = nn.Dropout(p=0.5)
        self.bn = nn.BatchNorm1d(hidden_size * 2, momentum=0.01)
        self.init_weights()

    def get_params(self):
        return self.cnn.get_params() + list(self.lstm.parameters()) + list(self.linear.parameters()) + list(self.bn.parameters())

    def init_weights(self):
        self.linear.weight.data.normal_(0.0, 0.02)
        self.linear.bias.data.fill_(0)

    def forward(self, story_images):
        data_size = story_images.size()
        local_cnn = self.cnn(story_images.view(-1, data_size[2], data_size[3], data_size[4]))
        global_rnn, (hn, cn) = self.lstm(local_cnn.view(data_size[0], data_size[1], -1))
        glocal = torch.cat((local_cnn.view(data_size[0], data_size[1], -1), global_rnn), 2)
        output = self.linear(glocal)
        output = self.dropout(output)
        output = self.bn(output.contiguous().view(-1, self.hidden_size * 2)).view(data_size[0], data_size[1], -1)

        return output, (hn, cn)


class EncoderDescription(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab):
        super(EncoderDescription, self).__init__()

        self.embed_size = embed_size
        vocab_size = len(vocab)
        self.vocab = vocab
        self.embed = nn.Embedding(vocab_size, embed_size)
        self.dropout = nn.Dropout(p=0.5)
        n_layers = 2
        self.lstm = nn.LSTM(embed_size, hidden_size, n_layers, batch_first=True, dropout=0.5)
        #self.init_weights()

    def get_params(self):
        return list(self.parameters())

    #def init_weights(self):
    #    self.linear.weight.data.normal_(0.0, 0.02)
    #    self.linear.bias.data.fill_(0)

    def forward(self, hidden, captions, lengths):
        embeddings = self.embed(captions)
        embeddings = self.dropout1(embeddings)
        outputs = []
        (hn, cn) = hidden

        for i, length in enumerate(lengths):
            lstm_input = embeddings[i][0:length - 1]
            output, (hn, cn) = self.lstm(lstm_input.unsqueeze(0), (hn, cn))
            #output = self.dropout2(output)
            #output = self.linear(output[0])
            #output = torch.cat((self.start_vec, output), 0)
            outputs.append(output)

        return outputs, (hn, cn)
        
                         
                         
class DecoderStory(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab):
        super(DecoderStory, self).__init__()

        self.embed_size = embed_size
        self.linear = nn.Linear(hidden_size * 2, hidden_size)
        self.dropout = nn.Dropout(p=0.5)
        self.rnn = DecoderRNN(embed_size, hidden_size, 2, vocab)
        self.init_weights()

    def get_params(self):
        return list(self.parameters())

    def init_weights(self):
        self.linear.weight.data.normal_(0.0, 0.02)
        self.linear.bias.data.fill_(0)

    def forward(self, story_feature, captions, lengths, descs, descs_length):
        story_feature = self.linear(story_feature)
        story_feature = self.dropout(story_feature)
        story_feature = F.relu(story_feature)
        result = self.rnn(story_feature, captions, lengths, descs, descs_length)
        return result

    def inference(self, story_feature, descs, descs_length):
        story_feature = self.linear(story_feature)
        story_feature = F.relu(story_feature)
        result = self.rnn.inference(story_feature, descs, descs_length)
        return result


class DecoderRNN(nn.Module):
    def __init__(self, embed_size, hidden_size, n_layers, vocab):
        super(DecoderRNN, self).__init__()
        self.vocab = vocab
        vocab_size = len(vocab)
        self.embed = nn.Embedding(vocab_size, embed_size)
        self.dropout1 = nn.Dropout(p=0.1)
        self.lstm = nn.LSTM(embed_size + hidden_size, hidden_size, n_layers, batch_first=True, dropout=0.5)
        self.dropout2 = nn.Dropout(p=0.5)
        self.linear = nn.Linear(hidden_size, vocab_size)
        self.n_layers = n_layers
        self.hidden_size = hidden_size
        self.softmax = nn.Softmax(0)
        
        self.dropout3 = nn.Dropout(p=0.1)
        self.desc_lstm = nn.LSTM(embed_size + hidden_size, hidden_size, n_layers, batch_first=True, dropout=0.5)
        self.hid_linear = nn.Linear(hidden_size*n_layers*2, n_layers*hidden_size)
        self.hid_linear2 = nn.Linear(hidden_size*5, 2*hidden_size)
        self.brobs = []

        self.init_input = torch.zeros([5, 1, embed_size], dtype=torch.float32)

        if torch.cuda.is_available():
            self.init_input = self.init_input.cuda()

        self.start_vec = torch.zeros([1, vocab_size], dtype=torch.float32)
        self.start_vec[0][1] = 10000
        if torch.cuda.is_available():
            self.start_vec = self.start_vec.cuda()

        self.init_weights()

    def get_params(self):
        return list(self.parameters())

    def init_hidden(self):
        h0 = torch.zeros(1 * self.n_layers, 1, self.hidden_size)
        c0 = torch.zeros(1 * self.n_layers, 1, self.hidden_size)

        h0 = torch.zeros(1 * self.n_layers, 1, self.hidden_size)
        c0 = torch.zeros(1 * self.n_layers, 1, self.hidden_size)
        
        if torch.cuda.is_available():
            h0 = h0.cuda()
            c0 = c0.cuda()
            
        return (h0, c0)
    
    def init_description_hidden(self, features):
        _h = self.hid_linear2(features)
        _c = self.hid_linear2(features)
        h = _h.view(1 * self.n_layers, 1, self.hidden_size)
        c = _c.view(1 * self.n_layers, 1, self.hidden_size)
        if torch.cuda.is_available():
            h = h.cuda()
            c = c.cuda()
        return (h,c)

    def init_weights(self):
        self.linear.weight.data.normal_(0.0, 0.02)
        self.linear.bias.data.fill_(0) 
        self.hid_linear.weight.data.normal_(0.0, 0.02)
        self.hid_linear.bias.data.fill_(0)
        self.hid_linear2.weight.data.normal_(0.0, 0.02)
        self.hid_linear2.bias.data.fill_(0) 
        

    def forward(self, features, captions, lengths, desc, desc_lengths):
        _features = features
        hid_feat = features.view(-1)
        embeddings = self.embed(captions)
        embeddings = self.dropout1(embeddings)
        features = features.unsqueeze(1).expand(-1, np.amax(lengths), -1)
        embeddings = torch.cat((features, embeddings), 2)
        
        desc_embeddings = self.embed(desc)
        desc_embeddings = self.dropout3(desc_embeddings)
        _features = _features.unsqueeze(1).expand(-1, np.amax(desc_lengths), -1)
        desc_embeddings = torch.cat((_features, desc_embeddings), 2)

        outputs = []
        (hn, cn) = self.init_hidden()
        (d_hn, d_cn) = self.init_description_hidden(hid_feat)
        
        for i, length in enumerate(lengths):
            desc_len = desc_lengths[i]
            desc_lstm_input = desc_embeddings[i][0:desc_len - 1]
            desc_output, (d_hn, d_cn) = self.desc_lstm(desc_lstm_input.unsqueeze(0), (d_hn, d_cn))
            lstm_input = embeddings[i][0:length - 1]
            new_hn = torch.cat((d_hn, hn), 0).view(-1)
            new_cn = torch.cat((d_cn, cn), 0).view(-1)
            hn = self.hid_linear(new_hn).view(1 * self.n_layers, 1, self.hidden_size)
            cn = self.hid_linear(new_cn).view(1 * self.n_layers, 1, self.hidden_size)
            output, (hn, cn) = self.lstm(lstm_input.unsqueeze(0), (hn, cn))
            output = self.dropout2(output)
            output = self.linear(output[0])
            output = torch.cat((self.start_vec, output), 0)
            outputs.append(output)

        return outputs


    def inference(self, features, desc, desc_lengths):
        results = []
        (hn, cn) = self.init_hidden()
        
        hid_feat = features.view(-1)
        desc_embeddings = self.embed(desc)
        #desc_embeddings = self.dropout3(desc_embeddings)
        _features = features.unsqueeze(1).expand(-1, np.amax(desc_lengths), -1)
        desc_embeddings = torch.cat((_features, desc_embeddings), 2)
        (d_hn, d_cn) = self.init_description_hidden(hid_feat)
        
        
        vocab = self.vocab
        end_vocab = vocab('<end>')
        forbidden_list = [vocab('<pad>'), vocab('<start>'), vocab('<unk>')]
        termination_list = [vocab('.'), vocab('?'), vocab('!')]
        function_list = [vocab('<end>'), vocab('.'), vocab('?'), vocab('!'), vocab('a'), vocab('an'), vocab('am'), vocab('is'), vocab('was'), vocab('are'), vocab('were'), vocab('do'), vocab('does'), vocab('did')]

        cumulated_word = []
        desc_in = 0
        for feature in features:
            
            desc_len = desc_lengths[desc_in]
            desc_lstm_input = desc_embeddings[desc_in][0:desc_len - 1]
            desc_output, (d_hn, d_cn) = self.desc_lstm(desc_lstm_input.unsqueeze(0), (d_hn, d_cn))
            desc_in += 1
            
            feature = feature.unsqueeze(0).unsqueeze(0)
            predicted = torch.tensor([1], dtype=torch.long).cuda()
            lstm_input = torch.cat((feature, self.embed(predicted).unsqueeze(1)), 2)
            sampled_ids = [predicted,]
            
            new_hn = torch.cat((d_hn, hn), 0).view(-1)
            new_cn = torch.cat((d_cn, cn), 0).view(-1)
            hn = self.hid_linear(new_hn).view(1 * self.n_layers, 1, self.hidden_size)
            cn = self.hid_linear(new_cn).view(1 * self.n_layers, 1, self.hidden_size)
            
            count = 0
            prob_sum = 1.0

            for i in range(50):
                #new_hn = torch.cat((d_hn, hn), 0).view(-1)
                #new_cn = torch.cat((d_cn, cn), 0).view(-1)
                #hn = self.hid_linear(new_hn).view(1 * self.n_layers, 1, self.hidden_size)
                #cn = self.hid_linear(new_cn).view(1 * self.n_layers, 1, self.hidden_size)
                outputs, (hn, cn) = self.lstm(lstm_input, (hn, cn))
                outputs = self.linear(outputs.squeeze(1))

                if predicted not in termination_list:
                    outputs[0][end_vocab] = -100.0

                for forbidden in forbidden_list:
                    outputs[0][forbidden] = -100.0

                cumulated_counter = Counter()
                cumulated_counter.update(cumulated_word)

                prob_res = outputs[0]
                prob_res = self.softmax(prob_res)
                for word, cnt in cumulated_counter.items():
                    if cnt > 0 and word not in function_list:
                        prob_res[word] = prob_res[word] / (1.0 + cnt * 5.0)
                prob_res = prob_res * (1.0 / prob_res.sum())

                candidate = []
                for i in range(100):
                    index = np.random.choice(prob_res.size()[0], 1, p=prob_res.cpu().detach().numpy())[0]
                    candidate.append(index)

                counter = Counter()
                counter.update(candidate)

                sorted_candidate = sorted(counter.items(), key=operator.itemgetter(1), reverse=True)

                predicted, _ = counter.most_common(1)[0]
                cumulated_word.append(predicted)

                predicted = torch.from_numpy(np.array([predicted])).cuda()
                sampled_ids.append(predicted)

                if predicted == 2:
                    break

                lstm_input = torch.cat((feature, self.embed(predicted).unsqueeze(1)), 2)

            results.append(sampled_ids)

        return results
