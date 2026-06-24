from os import remove
from uu import encode
import dgl
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import dgl.function as fn
from dgl.nn.pytorch import GATConv
from diffusion_ver15 import diffusion

class GAT(nn.Module):

    def __init__(self, input_size, hidden_size, output_size, num_heads):
        super(GAT, self).__init__()
        self.num_heads = num_heads
        self.conv1 = GATConv(input_size, hidden_size, num_heads=num_heads)
        self.conv2 = GATConv(hidden_size * num_heads, output_size, num_heads=num_heads)
        self.dropout = nn.Dropout(0.2)

    def forward(self, g, x):
        h = self.conv1(g, x).flatten(1)
        h = F.elu(h)
        h = self.dropout(h)
        h = self.conv2(g, h).mean(1)
        return h

class NewModel(nn.Module):

    def __init__(self, user_num, item_num, input_dim, time_dim, item_max_length, user_max_length, feat_drop=0.2, attn_drop=0.2, user_long='orgat', user_short='att', item_long='ogat', item_short='att', user_update='rnn', device='cpu', item_update='rnn', last_item=True, is_contrast=True, layer_num=3, time=True, temperature=0.2, alpha=0, lamda=1.0, windows=5, time_disrupt=True, iswindows=True, timesteps=20, beta_start=0.0001, beta_end=0.02, beta_sche='Linear', diff_dim=64):
        super(NewModel, self).__init__()
        self.user_num = user_num
        self.item_num = item_num
        self.hidden_size = input_dim
        self.time_size = time_dim
        self.item_max_length = item_max_length
        self.user_max_length = user_max_length
        self.layer_num = layer_num
        self.time = time
        self.last_item = last_item
        self.user_long = user_long
        self.item_long = item_long
        self.user_short = user_short
        self.item_short = item_short
        self.user_update = user_update
        self.item_update = item_update
        self.is_contrast = is_contrast
        self.time_dim = time_dim
        self.basis_freq = torch.nn.Parameter(torch.from_numpy(1 / 10 ** np.linspace(0, 9, time_dim)).float())
        self.phase = torch.nn.Parameter(torch.zeros(time_dim).float())
        self.device = device
        self.temperature = temperature
        self.alpha = alpha
        self.lamda = lamda
        self.time_disrupt = time_disrupt
        self.iswindows = iswindows
        self.ii_layers = GAT(self.hidden_size, 2 * self.hidden_size, self.hidden_size, 5)
        self.item_up = nn.Linear(self.hidden_size * 2, self.hidden_size)
        self.windows = windows
        self.user_embedding = nn.Embedding(self.user_num, self.hidden_size)
        self.item_embedding = nn.Embedding(self.item_num, self.hidden_size)
        if self.last_item:
            self.unified_map = nn.Linear((self.layer_num + 1) * self.hidden_size, self.hidden_size, bias=False)
        else:
            self.unified_map = nn.Linear(self.layer_num * self.hidden_size, self.hidden_size, bias=False)
            self.unified_map_item = nn.Linear(self.layer_num * self.hidden_size, self.hidden_size, bias=False)
        self.diffusion = diffusion(timesteps, beta_start, beta_end, input_dim, diff_dim, beta_sche)
        self.layers = nn.ModuleList([NewModelLayers(self.hidden_size, self.hidden_size, self.time_size, self.user_max_length, self.item_max_length, feat_drop, attn_drop, self.user_long, self.user_short, self.item_long, self.item_short, self.user_update, self.item_update, diffusion=self.diffusion, timesteps=timesteps) for _ in range(self.layer_num)])
        self.fn = nn.Sequential(nn.Linear(self.hidden_size, self.hidden_size, bias=True), nn.ReLU(), nn.Linear(self.hidden_size, self.hidden_size, bias=True))
        self.reset_parameters()

    def forward(self, g, user_index=None, last_item_index=None, neg_tar=None, is_training=False):
        if self.iswindows:
            diff_time = 10000000
            item_embeddings = []
            user_seq = {}
            for u in user_index:
                u = u.item()
                edges = g.out_edges(u, etype='pby')
                dst_items = edges[1].tolist()
                times = g.edges['pby'].data['time'][edges[0]].tolist()
                sorted_items = sorted(zip(dst_items, times), key=lambda x: x[1])
                user_seq[u] = sorted_items
            cooccur = {}
            adj = []
            windows_size = self.windows
            for u, item_time in user_seq.items():
                items = [x[0] for x in item_time]
                times = [x[1] for x in item_time]
                if len(items) > windows_size:
                    for i in range(len(items) - windows_size):
                        for j in range(i + 1, i + windows_size):
                            if times[j] - times[i] <= diff_time:
                                pair = tuple(sorted((items[i], items[j])))
                                cooccur[pair] = cooccur.get(pair, 0) + 1.0 / float(j - i)
                            else:
                                break
                else:
                    for i in range(len(items)):
                        for j in range(i + 1, len(items)):
                            if times[j] - times[i] <= diff_time:
                                pair = tuple(sorted((items[i], items[j])))
                                cooccur[pair] = cooccur.get(pair, 0) + 1.0 / float(j - i)
                            else:
                                break
            if cooccur:
                src = []
                dst = []
                weight = []
                for (i, j), cnt in cooccur.items():
                    src.extend([i, j])
                    dst.extend([j, i])
                    weight.extend([cnt, cnt])
                all_nodes = list(set(src + dst))
                node_mapping = {node_id: idx for idx, node_id in enumerate(all_nodes)}
                src_indices = [node_mapping[x] for x in src]
                dst_indices = [node_mapping[x] for x in dst]
                ii_graph = dgl.graph((torch.tensor(src_indices), torch.tensor(dst_indices))).to(self.device)
                ii_graph.ndata['item_id'] = torch.tensor(all_nodes).to(self.device)
                tmp_embedding = nn.Embedding.from_pretrained(self.item_embedding.weight)
                it_it_embedding = tmp_embedding.weight.clone()
                it_emb = it_it_embedding[all_nodes]
                it_emb = self.ii_layers(ii_graph, it_emb)
                for i, val in enumerate(all_nodes):
                    it_it_embedding[val] = it_emb[i]
                updated_embedding = F.softmax(self.item_up(torch.cat([tmp_embedding.weight, it_it_embedding], -1)), -1)
                self.item_embedding = nn.Embedding.from_pretrained(updated_embedding)
        if self.is_contrast:
            g_disrupt = g.clone()
            org_time = g_disrupt.edges['by'].data['time']
            org_time_pby = g_disrupt.edges['pby'].data['time']
            if self.time_disrupt:
                sort_time, _ = torch.sort(org_time)
                time_diff = torch.diff(sort_time)
                min_interval = torch.min(time_diff)
                disrupt_mask = torch.rand(org_time.shape)
                disrupt_mask_pby = torch.rand(org_time_pby.shape)
                disrupt = torch.randn(org_time.shape).masked_fill(disrupt_mask < 0.5, 0.0).to(org_time.device)
                disrupt_pby = torch.randn(org_time_pby.shape).masked_fill(disrupt_mask_pby < 0.5, 0.0).to(org_time_pby.device)
                g_disrupt.edges['by'].data['time'] = disrupt + org_time
                g_disrupt.edges['pby'].data['time'] = disrupt_pby + org_time_pby
            g_disrupt.nodes['user'].data['user_h'] = self.user_embedding(g_disrupt.nodes['user'].data['user_id'].cuda())
            g_disrupt.nodes['item'].data['item_h'] = self.item_embedding(g_disrupt.nodes['item'].data['item_id'].cuda())
        g.nodes['user'].data['user_h'] = self.user_embedding(g.nodes['user'].data['user_id'].cuda())
        g.nodes['item'].data['item_h'] = self.item_embedding(g.nodes['item'].data['item_id'].cuda())
        unified_embedding, diffusion_loss = self.use_conv(g, user_index, last_item_index, is_training)
        contrast_loss = 0
        if self.is_contrast:
            unified_embedding_disrupt, _ = self.use_conv(g_disrupt, user_index, last_item_index, is_training)
            '修改对比学习'
            user_items_emb, user_items_emb_org = self._get_user_items_embedding(g, user_index)
            user_items_emb_disrupt, _ = self._get_user_items_embedding(g_disrupt, user_index)
            contrast_loss_uer = self._compute_contrast_loss(unified_embedding, unified_embedding_disrupt)
            contrast_loss_item = self._compute_contrast_loss(user_items_emb, user_items_emb_disrupt)
            contrast_loss = float(self.alpha) * contrast_loss_uer + (1 - float(self.alpha)) * contrast_loss_item
            unified_embedding = self.user_embedding(user_index)
        score = torch.matmul(unified_embedding, self.item_embedding.weight.transpose(1, 0))
        if is_training:
            if self.is_contrast:
                return (score, contrast_loss, diffusion_loss, self.item_embedding, self.user_embedding)
            else:
                return (score, 0, diffusion_loss, self.item_embedding, self.user_embedding)
        else:
            neg_embedding = self.item_embedding(neg_tar)
            score_neg = torch.matmul(unified_embedding.unsqueeze(1), neg_embedding.transpose(2, 1)).squeeze(1)
            return (score, score_neg)

    def reset_parameters(self):
        gain = nn.init.calculate_gain('relu')
        for weight in self.parameters():
            if len(weight.shape) > 1:
                nn.init.xavier_normal_(weight, gain=gain)

    def time_encoder(self, t):
        seq_len = t.size(0)
        t = t.view(seq_len, 1)
        map_ts = t * self.basis_freq.view(1, -1)
        map_ts += self.phase.view(1, -1)
        return torch.cos(map_ts)

    def use_conv(self, g, user_index, last_item_index, is_training):
        feat_dict = None
        user_layer = []
        total_diffusion_loss = 0.0
        if self.layer_num > 0:
            for conv in self.layers:
                feat_dict, diff_loss = conv(g, feat_dict, is_training)
                total_diffusion_loss += diff_loss
                user_layer.append(graph_user(g, user_index, feat_dict['user']))
            if self.last_item:
                item_embed = graph_item(g, last_item_index, feat_dict['item'])
                user_layer.append(item_embed)
        unified_embedding = self.unified_map(torch.cat(user_layer, -1))
        return (unified_embedding, total_diffusion_loss)

    def _get_user_items_embedding(self, g, user_index):
        batch_items_emb = []
        batch_items_emb_org = []
        for i, u in enumerate(user_index):
            _, dst = g.out_edges(u.item(), etype='pby')
            if len(dst) == 0:
                batch_items_emb.append(torch.zeros(self.hidden_size).to(g.device))
                batch_items_emb_org.append(torch.zeros(self.hidden_size).to(g.device))
            else:
                items_emb = g.nodes['item'].data['item_h'][dst]
                avg_emb = torch.mean(items_emb, dim=0)
                batch_items_emb.append(avg_emb)
                items_emb_org = self.item_embedding(dst)
                avg_emb_org = torch.mean(items_emb_org, dim=0)
                batch_items_emb_org.append(avg_emb_org)
            return (torch.stack(batch_items_emb), torch.stack(batch_items_emb_org))

    def _compute_contrast_loss(self, user_items_emb, user_items_emb_disrupt):
        user_items_emb = F.normalize(user_items_emb, dim=1)
        user_items_emb_disrupt = F.normalize(user_items_emb_disrupt, dim=1)
        batch_size = user_items_emb.shape[0]
        sim_matrix = torch.matmul(user_items_emb, user_items_emb_disrupt.T) / self.temperature
        labels = torch.arange(batch_size).to(user_items_emb.device)
        loss = F.cross_entropy(sim_matrix, labels)
        return loss

class PositionalEncoding(nn.Module):

    def __init__(self, dim, max_len):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, dim, requires_grad=False)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        exp_term = torch.exp(torch.arange(0, dim, 2).float() * -(math.log(10000.0) / dim))
        pe[:, 0::2] = torch.sin(position * exp_term)
        pe[:, 1::2] = torch.cos(position * exp_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, maxlen):
        return self.pe[:, :maxlen]

class TimeEncoding(nn.Module):

    def __init__(self, max_time_buckets, embedding_dim):
        super(TimeEncoding, self).__init__()
        self.max_time_buckets = max_time_buckets
        self.embedding_dim = embedding_dim
        self.time_buckets = [2 ** i for i in range(max_time_buckets)]
        self.time_embedding = nn.Embedding(num_embeddings=max_time_buckets + 1, embedding_dim=embedding_dim)

    def forward(self, timestamps):
        time_bucket_indices = torch.bucketize(timestamps, torch.tensor(self.time_buckets, device=timestamps.device))
        self.time_embedding = self.time_embedding.to(time_bucket_indices.device)
        time_embeddings = self.time_embedding(time_bucket_indices)
        return time_embeddings

class NewModelLayers(nn.Module):

    def __init__(self, in_feats, out_feats, time_size, user_max_length, item_max_length, feat_drop=0.2, attn_drop=0.2, user_long='orgat', user_short='att', item_long='orgat', item_short='att', user_update='residual', item_update='residual', K=4, diffusion=None, timesteps=20):
        super(NewModelLayers, self).__init__()
        self.hidden_size = in_feats
        self.time_size = time_size
        self.user_long = user_long
        self.item_long = item_long
        self.user_short = user_short
        self.item_short = item_short
        self.user_update_m = user_update
        self.item_update_m = item_update
        self.user_max_length = user_max_length
        self.item_max_length = item_max_length
        self.K = torch.tensor(K)
        self.diffusion = diffusion
        self.timesteps = timesteps
        self.diff_loss = 0.0
        if self.user_long in ['orgat_time_order', 'orgat_order', 'orgat', 'gcn', 'gru'] and self.user_short in ['last', 'att', 'att1']:
            self.agg_gate_u = nn.Linear(self.hidden_size * 2, self.hidden_size, bias=False)
        if self.item_long in ['orgat_time_order', 'orgat_order', 'orgat', 'gcn', 'gru'] and self.item_short in ['last', 'att', 'att1']:
            self.agg_gate_i = nn.Linear(self.hidden_size * 2, self.hidden_size, bias=False)
        if self.user_long in ['gru']:
            self.gru_u = nn.GRU(input_size=in_feats, hidden_size=in_feats, batch_first=True)
        if self.item_long in ['gru']:
            self.gru_i = nn.GRU(input_size=in_feats, hidden_size=in_feats, batch_first=True)
        if self.user_update_m == 'norm':
            self.norm_user = nn.LayerNorm(self.hidden_size)
        if self.item_update_m == 'norm':
            self.norm_item = nn.LayerNorm(self.hidden_size)
        if item_long in ['orgat_time_order'] and self.user_long in ['orgat_time_order']:
            self.w1 = nn.Linear(self.hidden_size + self.time_size, self.hidden_size, bias=False)
            self.w1_k = nn.Linear(self.hidden_size + self.time_size, self.hidden_size, bias=False)
            self.w2 = nn.Linear(self.hidden_size + self.time_size, self.hidden_size, bias=False)
            self.w2_k = nn.Linear(self.hidden_size + self.time_size, self.hidden_size, bias=False)
        if item_long in ['orgat_order'] and self.user_long in ['orgat_order']:
            self.w1 = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
            self.w1_k = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
            self.w2 = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
            self.w2_k = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        self.feat_drop = nn.Dropout(feat_drop)
        self.atten_drop = nn.Dropout(attn_drop)
        self.user_weight = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        self.item_weight = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        if self.user_update_m in ['concat', 'rnn']:
            self.user_update = nn.Linear(2 * self.hidden_size, self.hidden_size, bias=False)
        if self.item_update_m in ['concat', 'rnn']:
            self.item_update = nn.Linear(2 * self.hidden_size, self.hidden_size, bias=False)
        if self.user_short in ['last', 'att']:
            self.last_weight_u = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        if self.item_short in ['last', 'att']:
            self.last_weight_i = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        if self.item_long in ['orgat']:
            self.i_time_encoding = nn.Embedding(self.user_max_length, self.hidden_size)
            self.i_time_encoding_k = nn.Embedding(self.user_max_length, self.hidden_size)
        if self.user_long in ['orgat']:
            self.u_time_encoding = nn.Embedding(self.item_max_length, self.hidden_size)
            self.u_time_encoding_k = nn.Embedding(self.item_max_length, self.hidden_size)
        if self.item_long in ['orgat_time_order', 'orgat_order']:
            self.posencoder_item = PositionalEncoding(self.hidden_size, self.user_max_length)
            self.posencoder_item_k = PositionalEncoding(self.hidden_size, self.user_max_length)
            self.timeencode_item = TimeEncoding(30, self.time_size)
            self.timeencode_item_k = TimeEncoding(30, self.time_size)
        if self.user_long in ['orgat_time_order', 'orgat_order']:
            self.posencoder_user = PositionalEncoding(self.hidden_size, self.user_max_length)
            self.posencoder_user_k = PositionalEncoding(self.hidden_size, self.user_max_length)
            self.timeencode_user = TimeEncoding(30, self.time_size)
            self.timeencode_user_k = TimeEncoding(30, self.time_size)
        self.diffusion_loss = 0.0

    def user_update_function(self, user_now, user_old):
        if self.user_update_m == 'residual':
            return F.elu(user_now + user_old)
        elif self.user_update_m == 'gate_update':
            pass
        elif self.user_update_m == 'concat':
            return F.elu(self.user_update(torch.cat([user_now, user_old], -1)))
        elif self.user_update_m == 'light':
            pass
        elif self.user_update_m == 'norm':
            return self.feat_drop(self.norm_user(user_now)) + user_old
        elif self.user_update_m == 'rnn':
            return F.tanh(self.user_update(torch.cat([user_now, user_old], -1)))
        else:
            print('error: no user_update')
            exit()

    def item_update_function(self, item_now, item_old):
        if self.item_update_m == 'residual':
            return F.elu(item_now + item_old)
        elif self.item_update_m == 'concat':
            return F.elu(self.item_update(torch.cat([item_now, item_old], -1)))
        elif self.item_update_m == 'light':
            pass
        elif self.item_update_m == 'norm':
            return self.feat_drop(self.norm_item(item_now)) + item_old
        elif self.item_update_m == 'rnn':
            return F.tanh(self.item_update(torch.cat([item_now, item_old], -1)))
        else:
            print('error: no item_update')
            exit()

    def forward(self, g, feat_dict=None, is_training=False):
        self.is_training = is_training
        if feat_dict == None:
            if self.user_long in ['gcn']:
                g.nodes['user'].data['norm'] = g['by'].in_degrees().unsqueeze(1).cuda()
            if self.item_long in ['gcn']:
                g.nodes['item'].data['norm'] = g['by'].out_degrees().unsqueeze(1).cuda()
            user_ = g.nodes['user'].data['user_h']
            item_ = g.nodes['item'].data['item_h']
        else:
            user_ = feat_dict['user'].cuda()
            item_ = feat_dict['item'].cuda()
            if self.user_long in ['gcn']:
                g.nodes['user'].data['norm'] = g['by'].in_degrees().unsqueeze(1).cuda()
            if self.item_long in ['gcn']:
                g.nodes['item'].data['norm'] = g['by'].out_degrees().unsqueeze(1).cuda()
        g.nodes['user'].data['user_h'] = self.user_weight(self.feat_drop(user_))
        g.nodes['item'].data['item_h'] = self.item_weight(self.feat_drop(item_))
        g = self.graph_update(g)
        g.nodes['user'].data['user_h'] = self.user_update_function(g.nodes['user'].data['user_h'], user_)
        g.nodes['item'].data['item_h'] = self.item_update_function(g.nodes['item'].data['item_h'], item_)
        f_dict = {'user': g.nodes['user'].data['user_h'], 'item': g.nodes['item'].data['item_h']}
        return (f_dict, self.diffusion_loss)

    def graph_update(self, g):
        g.multi_update_all({'by': (self.user_message_func, self.user_reduce_func), 'pby': (self.item_message_func, self.item_reduce_func)}, 'sum')
        return g

    def item_message_func(self, edges):
        dic = {}
        dic['time'] = edges.data['time']
        dic['user_h'] = edges.src['user_h']
        dic['item_h'] = edges.dst['item_h']
        return dic

    def item_reduce_func(self, nodes):
        h = []
        order = torch.argsort(torch.argsort(nodes.mailbox['time'], 1), 1)
        re_order = nodes.mailbox['time'].shape[1] - order - 1
        length = nodes.mailbox['item_h'].shape[0]
        if self.item_long == 'orgat':
            e_ij = torch.sum((self.i_time_encoding(re_order) + nodes.mailbox['user_h']) * nodes.mailbox['item_h'], dim=2) / torch.sqrt(torch.tensor(self.hidden_size).float())
            alpha = self.atten_drop(F.softmax(e_ij, dim=1))
            if len(alpha.shape) == 2:
                alpha = alpha.unsqueeze(2)
            h_long = torch.sum(alpha * (nodes.mailbox['user_h'] + self.i_time_encoding_k(re_order)), dim=1)
            h.append(h_long)
        elif self.item_long == 'orgat_time_order':
            max_time = torch.max(nodes.mailbox['time'], dim=1).values
            max_time = max_time.unsqueeze(1)
            diff_t = max_time - nodes.mailbox['time']
            q = self.w2(torch.cat((nodes.mailbox['user_h'] + self.posencoder_item(nodes.mailbox['time'].shape[1]), self.timeencode_item(diff_t)), dim=2))
            e_ij = torch.sum(q * nodes.mailbox['item_h'], dim=2) / torch.sqrt(torch.tensor(self.hidden_size).float())
            alpha = self.atten_drop(F.softmax(e_ij, dim=1))
            if len(alpha.shape) == 2:
                alpha = alpha.unsqueeze(2)
            v = self.w2(torch.cat((nodes.mailbox['user_h'] + self.posencoder_item_k(nodes.mailbox['time'].shape[1]), self.timeencode_item_k(diff_t)), dim=2))
            h_long = torch.sum(alpha * v, dim=1)
            h.append(h_long)
        elif self.item_long == 'orgat_order':
            q = self.w2(nodes.mailbox['user_h'] + self.posencoder_item(nodes.mailbox['time'].shape[1]))
            e_ij = torch.sum(q * nodes.mailbox['item_h'], dim=2) / torch.sqrt(torch.tensor(self.hidden_size).float())
            alpha = self.atten_drop(F.softmax(e_ij, dim=1))
            if len(alpha.shape) == 2:
                alpha = alpha.unsqueeze(2)
            v = self.w2_k(nodes.mailbox['user_h'] + self.posencoder_item_k(nodes.mailbox['time'].shape[1]))
            h_long = torch.sum(alpha * v, dim=1)
            h.append(h_long)
        elif self.item_long == 'gru':
            rnn_order = torch.sort(nodes.mailbox['time'], 1)[1]
            _, hidden_u = self.gru_i(nodes.mailbox['user_h'][torch.arange(length).unsqueeze(1), rnn_order])
            h.append(hidden_u.squeeze(0))
        last = torch.argmax(nodes.mailbox['time'], 1)
        last_em = nodes.mailbox['user_h'][torch.arange(length), last, :].unsqueeze(1)
        if self.item_short == 'att':
            e_ij1 = torch.sum(last_em * nodes.mailbox['user_h'], dim=2) / torch.sqrt(torch.tensor(self.hidden_size).float())
            alpha1 = self.atten_drop(F.softmax(e_ij1, dim=1))
            if len(alpha1.shape) == 2:
                alpha1 = alpha1.unsqueeze(2)
            h_short = torch.sum(alpha1 * nodes.mailbox['user_h'], dim=1)
            h.append(h_short)
        elif self.item_short == 'last':
            h.append(last_em.squeeze())
        if len(h) == 1:
            return {'item_h': h[0]}
        else:
            return {'item_h': self.agg_gate_i(torch.cat(h, -1))}

    def user_message_func(self, edges):
        dic = {}
        dic['time'] = edges.data['time']
        dic['item_h'] = edges.src['item_h']
        dic['user_h'] = edges.dst['user_h']
        return dic

    def user_reduce_func(self, nodes):
        h = []
        order = torch.argsort(torch.argsort(nodes.mailbox['time'], 1), 1)
        re_order = nodes.mailbox['time'].shape[1] - order - 1
        length = nodes.mailbox['user_h'].shape[0]
        max_time = torch.max(nodes.mailbox['time'], dim=1).values
        max_time = max_time.unsqueeze(1)
        diff_t = max_time - nodes.mailbox['time']
        if self.user_long == 'orgat':
            e_ij = torch.sum((self.u_time_encoding(re_order) + nodes.mailbox['item_h']) * nodes.mailbox['user_h'], dim=2) / torch.sqrt(torch.tensor(self.hidden_size).float())
            alpha = self.atten_drop(F.softmax(e_ij, dim=1))
            if len(alpha.shape) == 2:
                alpha = alpha.unsqueeze(2)
            h_long = torch.sum(alpha * (nodes.mailbox['item_h'] + self.u_time_encoding(re_order)), dim=1)
            h.append(h_long)
        elif self.user_long == 'orgat_time_order':
            q = self.w1(torch.cat((nodes.mailbox['item_h'] + self.posencoder_user(nodes.mailbox['time'].shape[1]), self.timeencode_user(diff_t)), dim=2))
            e_ij = torch.sum(q * nodes.mailbox['item_h'], dim=2) / torch.sqrt(torch.tensor(self.hidden_size).float())
            alpha = self.atten_drop(F.softmax(e_ij, dim=1))
            if len(alpha.shape) == 2:
                alpha = alpha.unsqueeze(2)
            v = self.w1_k(torch.cat((nodes.mailbox['item_h'] + self.posencoder_user_k(nodes.mailbox['time'].shape[1]), self.timeencode_user_k(diff_t)), dim=2))
            h_long = torch.sum(alpha * v, dim=1)
            h.append(h_long)
        elif self.user_long == 'orgat_order':
            q = self.w1(nodes.mailbox['item_h'] + self.posencoder_item(nodes.mailbox['time'].shape[1]))
            e_ij = torch.sum(q * nodes.mailbox['item_h'], dim=2) / torch.sqrt(torch.tensor(self.hidden_size).float())
            alpha = self.atten_drop(F.softmax(e_ij, dim=1))
            if len(alpha.shape) == 2:
                alpha = alpha.unsqueeze(2)
            v = self.w1_k(nodes.mailbox['item_h'] + self.posencoder_user_k(nodes.mailbox['time'].shape[1]))
            h_long = torch.sum(alpha * v, dim=1)
            h.append(h_long)
        elif self.user_long == 'gru':
            rnn_order = torch.sort(nodes.mailbox['time'], 1)[1]
            _, hidden_i = self.gru_u(nodes.mailbox['item_h'][torch.arange(length).unsqueeze(1), rnn_order])
            h.append(hidden_i.squeeze(0))
        last = torch.argmax(nodes.mailbox['time'], 1)
        last_em = nodes.mailbox['item_h'][torch.arange(length), last, :].unsqueeze(1)
        if self.user_short == 'att':
            e_ij1 = torch.sum(last_em * nodes.mailbox['item_h'], dim=2) / torch.sqrt(torch.tensor(self.hidden_size).float())
            alpha1 = self.atten_drop(F.softmax(e_ij1, dim=1))
            if len(alpha1.shape) == 2:
                alpha1 = alpha1.unsqueeze(2)
            h_short = torch.sum(alpha1 * nodes.mailbox['item_h'], dim=1)
            "'扩散加在这里"
            t = torch.randint(low=0, high=self.timesteps, size=(h_short.shape[0] // 2 + 1,)).to(h_short.device)
            t = torch.cat([t, self.timesteps - t - 1], dim=0)[:h_short.shape[0]]
            if self.is_training:
                with torch.set_grad_enabled(False):
                    diff_loss, h_short = self.diffusion.p_losses(h_short.detach(), h_long.detach(), t, noise=None, loss_type='l2')
                self.diffusion_loss = diff_loss
            else:
                _, h_short = self.diffusion.sample(h_short, h_long)
            h.append(h_short)
        elif self.user_short == 'att_time':
            e_ij1 = torch.sum(nodes.mailbox['item_h'] * nodes.mailbox['user_h'], dim=2) / torch.sqrt(torch.tensor(self.hidden_size).float())
            alpha1 = self.atten_drop(F.softmax(e_ij1, dim=1))
            if len(alpha1.shape) == 2:
                alpha1 = alpha1.unsqueeze(2)
            h_short = torch.sum(alpha1 * torch.cat((nodes.mailbox['item_h'], nodes.mailbox['encode_time']), dim=2), dim=1)
            h.append(h_short)
        elif self.user_short == 'last':
            h.append(last_em.squeeze())
        if len(h) == 1:
            return {'user_h': h[0]}
        else:
            return {'user_h': self.agg_gate_u(torch.cat(h, -1))}

def graph_user(bg, user_index, user_embedding):
    b_user_size = bg.batch_num_nodes('user')
    tmp = torch.roll(torch.cumsum(b_user_size, 0), 1)
    tmp[0] = 0
    new_user_index = tmp + user_index
    return user_embedding[new_user_index]

def graph_item(bg, last_index, item_embedding):
    b_item_size = bg.batch_num_nodes('item')
    tmp = torch.roll(torch.cumsum(b_item_size, 0), 1)
    tmp[0] = 0
    new_item_index = tmp + last_index
    return item_embedding[new_item_index]

def order_update(edges):
    dic = {}
    dic['order'] = torch.sort(edges.data['time'])[1]
    dic['re_order'] = len(edges.data['time']) - dic['order']
    return dic

def collate(data):
    user = []
    user_l = []
    graph = []
    label = []
    last_item = []
    for da in data:
        user.append(da[1]['user'])
        user_l.append(da[1]['u_alis'])
        graph.append(da[0][0])
        label.append(da[1]['target'])
        last_item.append(da[1]['last_alis'])
    return (torch.tensor(user_l).long(), dgl.batch(graph), torch.tensor(label).long(), torch.tensor(last_item).long())

def neg_generate(user, data_neg, neg_num=100):
    neg = np.zeros((len(user), neg_num), np.int32)
    for i, u in enumerate(user):
        u_int = u.item() if isinstance(u, torch.Tensor) else u
        neg[i] = np.random.choice(data_neg[u_int], neg_num, replace=False)
    return neg

def collate_test(data, user_neg):
    user = []
    graph = []
    label = []
    last_item = []
    for da in data:
        user.append(da[1]['u_alis'])
        graph.append(da[0][0])
        label.append(da[1]['target'])
        last_item.append(da[1]['last_alis'])
    return (torch.tensor(user).long(), dgl.batch(graph), torch.tensor(label).long(), torch.tensor(last_item).long(), torch.Tensor(neg_generate(user, user_neg)).long())

def neg_generate_train(user, data_neg, batch_graph, neg_num=100):
    neg = np.zeros((len(user), neg_num), np.int32)
    for i, u in enumerate(user):
        u_int = u.item() if isinstance(u, torch.Tensor) else u
        candidates = np.intersect1d(data_neg[u_int], batch_graph.nodes['item'].data['item_id'].numpy())
        neg[i] = np.random.choice(candidates, neg_num, replace=False)
    return neg

def collate_train(data, user_neg):
    user = []
    graph = []
    label = []
    last_item = []
    for da in data:
        user.append(da[1]['u_alis'])
        graph.append(da[0][0])
        label.append(da[1]['target'])
        last_item.append(da[1]['last_alis'])
    return (torch.tensor(user).long(), dgl.batch(graph), torch.tensor(label).long(), torch.tensor(last_item).long(), torch.Tensor(neg_generate_train(user, user_neg, dgl.batch(graph))).long())
