"""
Re-implementation of GLAC Net (https://arxiv.org/abs/1805.10973)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

USE_CUDA = torch.cuda.is_available()
device = torch.device("cuda" if USE_CUDA else "cpu")

class EncoderCNN(nn.Module):
    def __init__(self, input_size, target_size):
        super(EncoderCNN, self).__init__()
        self.linear = nn.Linear(input_size, target_size)
        self.bn = nn.BatchNorm1d(target_size, momentum=0.01)
        self.init_weights()
    
    def init_weights(self):
        self.linear.weight.data.normal_(0.0, 0.02)
        self.linear.bias.data.fill_(0)
    
    def forward(self, features):
        features = self.linear(features)
        features = self.bn(features)
        return features

class EncodeStory(nn.Module):
    def __init__(self, img_feature_size, hidden_size, n_layers):
        super(EncodeStory, self).__init__()
        self.img_feature_size = img_feature_size
        self.hidden_size = hidden_size
        self.n_layers = n_layers
        self.cnn = EncoderCNN(img_feature_size, img_feature_size)
        self.lstm = nn.LSTM(img_feature_size, hidden_size, n_layers, batch_first=True, bidirectional=True)
        self.linear = nn.Linear(hidden_size*2+img_feature_size, hidden_size*2)
        self.dropout = nn.Dropout(p=0.5)
        self.bn = nn.BatchNorm1d(hidden_size*2, momentum=0.01)
        self.init_weights()
  
    def init_weights(self):
        self.linear.weight.data.normal_(0.0, 0.02)
        self.linear.bias.data.fill_(0)
  
    def forward(self, story_images):
        batch_size, seq_length, _ = story_images.shape
        local_output = torch.zeros((batch_size, seq_length, self.img_feature_size)).to(device)
        for t in range(seq_length):
            local_cnn = self.cnn(story_images[:, t, :])
            local_output[:, t, :] = local_cnn
        self.lstm.flatten_parameters()
        global_rnn, (hn, cn) = self.lstm(local_output)
        glocal = torch.cat((local_output, global_rnn), 2)
        glocal_output = torch.zeros((batch_size, seq_length, self.hidden_size*2)).to(device)

        for t in range(seq_length):
            output = self.linear(glocal[:, t, :])
            output = self.dropout(output)
            output = self.bn(output.contiguous())
            glocal_output[:, t, :] = output
        return glocal_output, (hn, cn)

class BatchGLACNet(nn.Module):
    def __init__(self, img_feature_size, enc_hidden_size, vocab_size, 
                embed_size, dec_hidden_size, n_layer=1):
        super(BatchGLACNet, self).__init__()
        self.img_feature_size = img_feature_size
        self.enc_hidden_size = enc_hidden_size
        self.vocab_size = vocab_size
        self.dec_hidden_size = dec_hidden_size
        self.embed_size = embed_size
        self.encoder = EncodeStory(img_feature_size, enc_hidden_size, n_layer)
        self.embed = nn.Embedding(vocab_size, embed_size)
        self.lstm = nn.LSTMCell(embed_size + dec_hidden_size*2, dec_hidden_size)
        self.linear = nn.Linear(dec_hidden_size, vocab_size)
        self.dropout = nn.Dropout(p=0.5)

    def init_hidden(self, batch_size):
        hidden_state = torch.zeros((batch_size, self.dec_hidden_size)).to(device)
        cell_state = torch.zeros((batch_size, self.dec_hidden_size)).to(device)
        return hidden_state, cell_state

    def forward_ingredient(self, batch_ingredient_vectors):
        batch_ingredient_outputs = torch.zeros((len(batch_ingredient_vectors), self.ingr_hidden_size)).to(device)
        for ingr_idx, batch_ingredient_vector in enumerate(batch_ingredient_vectors):
            batch_ingredient_vector = [torch.Tensor(ingredient_vec).to(device) for ingredient_vec in batch_ingredient_vector]
            output = self.ingredient_lstm(batch_ingredient_vector)
            batch_ingredient_outputs[ingr_idx] = output
        return batch_ingredient_outputs

    def forward(self, batched_step_vectors, batched_image_vectors, pad_num):
        # 1. 画像からの特徴抽出
        batch_size, seq_length, max_T = batched_step_vectors.shape
        out_image_vectors, _ = self.encoder(batched_image_vectors)
        hidden_state, cell_state = self.init_hidden(batch_size)
        sent_output = torch.zeros((batch_size, seq_length, max_T, self.vocab_size)).to(device)

        for i in range(seq_length):
            i_th_image_vectors = out_image_vectors[:, i, :]
            i_th_step_vectors = batched_step_vectors[:, i, :]
            output, (hidden_state, cell_state) = self.foward_ith_step(i_th_image_vectors, i_th_step_vectors, 
                                                                      batch_size, max_T, hidden_state, cell_state, pad_num)
            sent_output[:, i, :, :] = output
        return sent_output
  
    def foward_ith_step(self, i_th_image_vectors, i_th_step_vectors, 
                        batch_size, max_T, hidden_state, cell_state, pad_num):
        start_vec = torch.zeros((batch_size, self.embed_size)).to(device)
        sent_output = torch.zeros((batch_size, max_T, self.vocab_size)).to(device)
        masked_hidden_state = torch.zeros((batch_size, max_T, self.dec_hidden_size)).to(device)
        masked_cell_state = torch.zeros((batch_size, max_T, self.dec_hidden_size)).to(device)

        for tdx in range(max_T):
            if tdx == 0:
                input = torch.cat((start_vec, i_th_image_vectors), dim=1)
                hidden_state, cell_state = self.lstm(input, (hidden_state, cell_state))
            else:
                embedded = self.embed(i_th_step_vectors[:, tdx-1])
                input = torch.cat((embedded, i_th_image_vectors), dim=1)
                hidden_state, cell_state = self.lstm(input, (hidden_state, cell_state))
            output = self.linear(self.dropout(F.relu(hidden_state)))
            output = F.log_softmax(output, dim=1)
            sent_output[:, tdx, :] = output
            masked_hidden_state[:, tdx, :] = hidden_state
            masked_cell_state[:, tdx, :] = cell_state
      
        eos_indices = self.extract_index_end_of_sentence(i_th_step_vectors, pad_num)
        out_hidden_state = torch.zeros((batch_size, self.dec_hidden_size)).to(device)
        out_cell_state = torch.zeros((batch_size, self.dec_hidden_size)).to(device)
      
        for b_index, eos_index in enumerate(eos_indices):
            out_hidden_state[b_index] = masked_hidden_state[b_index][eos_index]
            out_cell_state[b_index] = masked_cell_state[b_index][eos_index]
        return sent_output, (out_hidden_state, out_cell_state)
  
    def extract_index_end_of_sentence(self, i_th_step_vectors, pad_num):
        eos_location = i_th_step_vectors == pad_num-1 # pad_num-1 == end of sequence
        _, eos_index = torch.max(eos_location, 1)
        return eos_index

    def predict(self, images, ingredient_vector, vocab):
        max_length = 100
        sent_outputs = []
        hidden_state, cell_state = self.init_hidden(batch_size=1)
        image_vectors, _ = self.encoder(images.unsqueeze(0))
        image_vectors = image_vectors.squeeze(0)
        start_vec = torch.zeros((1, self.embed_size)).to(device)
        ingredient_vector = self.forward_ingredient([ingredient_vector])

        for image_vector in image_vectors:
            prev_vec = None
            sent_output = []
            for len_idx in range(max_length):
                if len_idx == 0:
                    input = torch.cat((start_vec, image_vector.unsqueeze(0), ingredient_vector), dim=1)
                    input = self.dropout(F.relu(self.firstMLP(input))) # +Ingr
                    hidden_state, cell_state = self.lstm(input, (hidden_state, cell_state))
                else:
                    embedded = self.embed(prev_vec)
                    input = torch.cat((embedded, image_vector.unsqueeze(0)), dim=1)
                    hidden_state, cell_state = self.lstm(input, (hidden_state, cell_state))
                output = self.linear(F.relu(hidden_state))
                output = torch.argmax(F.softmax(output, dim=1), dim=1)
                prev_vec = output.clone()
                word_id = output.clone().item()
                sent_output.append(word_id)
                if word_id == vocab.eos():
                    break
            hidden_state = hidden_state.clone()
            cell_state = cell_state.clone()
            sent_outputs.append(sent_output)
        return sent_outputs