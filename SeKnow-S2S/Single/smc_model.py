import torch
from torch import nn
from torch.distributions import Categorical
import numpy as np

from config import global_config as cfg
from modules import get_one_hot_input, cuda_
from base_model import BaseModel
from utils import toss_

torch.set_printoptions(sci_mode=False)

class SemiBootstrapSMC(BaseModel):
    def __init__(self, **kwargs):
        super().__init__(has_qnet=False, **kwargs)
        self.weight_normalize = nn.Softmax(dim=0)
        self.particle_num = cfg.particle_num

    def forward(self, u_input, m_input, t_input, k_ground, z_input, a_input, turn_states, z_supervised, mode,
                db_vec=None, filling_vec=None, no_label_train=False, context_to_response=False, amend=False):
        if mode == 'train' or mode == 'loss_eval':
            debug = {'true_z': z_input, 'true_db': db_vec, 'true_a': a_input}
            if not z_supervised:
                z_input = None
                if not no_label_train:
                    u_input = torch.cat([u_input]*self.particle_num, dim=0)
                    m_input = torch.cat([m_input]*self.particle_num, dim=0)
            probs, index, turn_states, hiddens = \
                self.forward_turn(u_input, m_input=m_input, z_input=z_input, is_train=True,
                                  t_input=t_input, k_ground=k_ground,
                                  turn_states=turn_states, db_vec=db_vec, debug=debug, mode=mode,
                                  a_input=a_input, filling_vec=filling_vec, no_label_train=no_label_train,
                                  context_to_response=context_to_response, amend=amend)
            if z_supervised:
                z_input = torch.cat(list(z_input.values()), dim=1)
                z_input = torch.cat([z_input, t_input], dim=1)
                a_input = torch.cat(list(a_input.values()), dim=1) if cfg.model_act else None
                index.update({'z_input': z_input, 'a_input': a_input, 'm_input': m_input})
                loss, pz_loss, pa_loss, m_loss = self.supervised_loss(probs, index)
                losses = {'loss': loss, 'pz_loss': pz_loss, 'm_loss': m_loss}
                if cfg.model_act:
                    losses.update({'pa_loss': pa_loss})
                return losses, turn_states
            else:
                index.update({'m_input': m_input})
                if not no_label_train:
                    loss, pz_loss, pa_loss, m_loss = self.unsupervised_loss(probs, index, turn_states['norm_W'])
                    losses = {'loss': loss, 'pz_loss': pz_loss, 'm_loss': m_loss}
                    if cfg.model_act:
                        losses.update({'pa_loss': pa_loss})
                else:
                    loss, pz_loss, pa_loss, m_loss = self.supervised_loss(probs, index, no_label_train)
                    losses = {'loss': loss, 'm_loss': m_loss}

                return losses, turn_states

        elif mode == 'test':
            index, db, kb, turn_states = self.forward_turn(u_input, is_train=False, z_input=z_input, a_input=a_input,
                                                           t_input=t_input, k_ground=k_ground,
                                                           turn_states=turn_states, db_vec=db_vec,
                                                           context_to_response=context_to_response, amend=amend)
            return index, db, kb, turn_states

    def forward_turn(self, u_input, turn_states, is_train, m_input=None, z_input=None, t_input=None,
                     k_ground=None, a_input=None, db_vec=None, filling_vec=None, debug=None, mode=None,
                     no_label_train=False, context_to_response=False, amend=False):
        """
        compute required outputs for a single dialogue turn. Turn state{Dict} will be updated in each call.
        :param u_len:
        :param turn_states:
        :param is_train:
        :param u_input: [B,Tu]
        :param t_input: [B,Tt]
        :param k_input: [B,Tk]
        :param m_input: [B,Tm]
        :param z_input: dict of [B,Tz]
        :param: norm_W: [B, K]
        pv_pz_pr: K * [B,T,V]
        pv_pz_h: K * [B,T,H]
        :return:
        """
        batch_size = u_input.size(0)
        u_hiddens = self.u_encoder(u_input)  # for Transformers
        # u_hiddens, u_last_hidden = self.u_encoder(u_input)  # for GRU
        u_input_1hot = get_one_hot_input(u_input, self.vocab_size)
        if context_to_response:
            t_input_1hot = get_one_hot_input(t_input, self.vocab_size)
            z_input_1hot = {}
            for sn in self.reader.otlg.informable_slots:
                z_input_1hot[sn] = get_one_hot_input(z_input[sn], self.vocab_size)
        else:
            t_input_1hot = None
            z_input_1hot = None

        '''Unsupervised Training Not Used'''
        '''
        if is_train and z_input is None:   # unsupervised training
            if not no_label_train:
                ori_batch_size = int(u_input.size(0) / self.particle_num)
                norm_W = turn_states.get('norm_W', None)
                if norm_W is not None and cfg.resampling:  # Resampling
                    dis = Categorical(torch.cat([norm_W]*self.particle_num, dim=0)) # [B*K, K]
                    Ak = dis.sample()   #[B*K]
                    # print('Ak:', Ak.contiguous().view(self.particle_num,-1))
                    bias = np.tile(np.arange(0, ori_batch_size), self.particle_num)
                    idx = bias + Ak.cpu().numpy() * ori_batch_size
                    turn_states['pv_pz_h'] = turn_states['pv_pz_h'][idx]    # [T, B*K, V]
                    turn_states['pv_pz_pr'] = turn_states['pv_pz_pr'][idx]  # [T, B*K, H]
                    turn_states['pv_pz_id'] = turn_states['pv_pz_id'][idx]
            sample_type = 'topk'
        '''
        if is_train and mode != 'loss_eval':
            sample_type = 'supervised'
        else:   # testing
            sample_type = 'top1'  # or 'topk'

        # P(t|pv_z, u), topic decoding
        # pt_prob, pt_samples, log_pt = \
        #     self.decode_t(batch_size, u_input, u_hiddens, u_input_1hot, u_last_hidden, t_input, t_input_1hot,
        #                   turn_states, sample_type, context_to_response=context_to_response)

        # P(z|pv_z, u) z = dsv + topic
        '''Transformers Version'''

        pz_prob, pz_samples, z_hiddens, turn_states = \
            self.decode_z_T(batch_size, u_hiddens, z_input, z_input_1hot, t_input, t_input_1hot,
                            turn_states, sample_type=sample_type, context_to_response=context_to_response)

        '''GRU Version'''
        '''
        pz_prob, pz_samples, z_hiddens, turn_states = \
            self.decode_z(batch_size, u_input, u_hiddens, u_input_1hot, z_input, z_input_1hot,
                          t_input, t_input_1hot, turn_states, sample_type=sample_type, decoder_type='pz',
                          context_to_response=context_to_response)
        '''
        # DB indicator and slot filling indicator
        db_vec_np, match, entries = self.db_op.get_db_degree(pz_samples, turn_states['dom'], self.vocab)
        # filling_vec = self.reader.cons_tensors_to_indicator(pz_samples) # not used in Transformers Version
        # filling_vec = cuda_(torch.from_numpy(filling_vec).float())  # not used in Transformers Version

        # Semi-structured Knowledge Operation
        db_vec_np, kb_matches = self.db_op.get_knowledge_match(pz_samples, self.vocab, self.eos_t_token, sample_type,
                                                               db_match_vec=db_vec_np, db_match_entries=entries,
                                                               top_k=self.k_sample_size, amend=amend,
                                                               context_to_response=context_to_response)
        if not context_to_response:
            db_vec_new = cuda_(torch.from_numpy(db_vec_np).float())
            db_vec[:, :4] = db_vec_new
        k_hiddens, k_input = self.encode_k(kb_matches, k_ground, sample_type, context_to_response=context_to_response)
        # k_input_1hot = get_one_hot_input(k_input, self.vocab_size)

        # P(a|u, db, slot_filling_indicator), not used in Transformers Version
        '''
        if self.model_act:
            pa_prob, pa_samples, a_hiddens, log_pa = \
                self.decode_a(batch_size, u_input, u_hiddens, u_input_1hot, u_last_hidden, a_input,
                              db_vec, filling_vec, sample_type=sample_type, decoder_type='pa',
                              context_to_response=context_to_response)
        else:
            pa_prob, pa_samples, a_hiddens = None, None, None
        '''
        pa_prob, pa_samples, a_hiddens = None, None, None

        # P(m|u, z, a ,db, kb), Transformers Version
        pm_prob, m_idx = \
            self.decode_m_T(batch_size, u_hiddens, z_hiddens, k_hiddens, db_vec, m_input, is_train=is_train)

        # P(m|u, z, a ,db, kb), GRU Version
        '''
        if is_train or not self.beam_search:
            pm_prob, m_idx, log_pm = \
                    self.decode_m(batch_size, u_last_hidden, u_input, u_hiddens, u_input_1hot,
                                  pz_samples, pz_prob, z_hiddens, pa_samples, pa_prob, a_hiddens,
                                  k_input_1hot, k_hiddens, db_vec, m_input, is_train=is_train)
        else:
            m_idx = self.beam_search_decode(u_input, u_input_1hot, u_hiddens, pz_samples,
                                            pz_prob, z_hiddens, k_input_1hot, k_hiddens,
                                            db_vec, u_last_hidden[:-1], pa_samples, pa_prob, a_hiddens)
        '''

        # compute normalized weights W for unsupervised training, not used
        '''
        if is_train and z_input is None and not no_label_train:
            log_w = log_pm
            log_w = log_w.view(self.particle_num, -1)
            norm_W = self.weight_normalize(log_w).transpose(1,0)    #[B,K]
            turn_states['norm_W'] = norm_W
        '''

        # output
        if is_train:
            probs = {'pz_prob': pz_prob, 'pm_prob': pm_prob, 'pa_prob': pa_prob}
            index = {'z_input': pz_samples, 'a_input': pa_samples}
            hiddens = {'u_hiddens': u_hiddens, 'z_hiddens': z_hiddens, 'k_hiddens': k_hiddens}
            return probs, index, turn_states, hiddens
        else:
            if not context_to_response:
                z_idx = self.max_sampling(pz_prob)
            else:
                z_input = torch.cat(list(z_input.values()), dim=1)
                z_idx = [list(_) for _ in list(z_input)]
            a_idx = self.max_sampling(pa_prob) if self.model_act else None
            index = {'m_idx': m_idx, 'z_idx': z_idx, 'a_idx': a_idx}
            return index, match, kb_matches, turn_states

    def supervised_loss(self, probs, index, no_label_train=False):
        pz_prob, pm_prob = torch.log(probs['pz_prob']), torch.log(probs['pm_prob'])
        z_input, m_input = index['z_input'], index['m_input']
        pz_loss = self.nll_loss(pz_prob.view(-1, pz_prob.size(2)), z_input.view(-1))
        m_loss = self.nll_loss(pm_prob.view(-1, pm_prob.size(2)), m_input.view(-1))
        # print(pz_loss)
        # print(m_loss)
        if self.model_act:
            pa_prob = torch.log(probs['pa_prob'])
            a_input = index['a_input']
            pa_loss = self.nll_loss(pa_prob.view(-1, pa_prob.size(2)), a_input.view(-1))
            loss = cfg.pz_loss_weight * pz_loss + m_loss + pa_loss
        else:
            pa_loss = torch.zeros(1)
            loss = cfg.pz_loss_weight * pz_loss + m_loss
        if no_label_train:
            loss = m_loss
        return loss, pz_loss, pa_loss, m_loss

    def unsupervised_loss(self, probs, index, norm_W):
        # pz_prob: [B*K, T, V]
        pz_prob, pm_prob = torch.log(probs['pz_prob']), torch.log(probs['pm_prob'])
        z_input, m_input = index['z_input'], index['m_input']
        if self.model_act:
            pa_prob, a_input = torch.log(probs['pa_prob']), index['a_input']
        if cfg.weighted_grad:
            cliped_norm_W = torch.clamp(norm_W, min=1e-10, max=1)
            W = cliped_norm_W.transpose(1, 0).contiguous().view(-1).detach()
            pz_prob = pz_prob.transpose(2, 0) * W * cfg.particle_num
            pm_prob = pm_prob.transpose(2, 0) * W * cfg.particle_num
            pz_prob, pm_prob = pz_prob.transpose(2, 0).contiguous(), pm_prob.transpose(2, 0).contiguous()
            if self.model_act:
                pa_prob = pa_prob.transpose(2,0) * W * cfg.particle_num
                pa_prob = pa_prob.transpose(2,0).contiguous()

        pz_loss = self.nll_loss(pz_prob.view(-1, pz_prob.size(2)), z_input.view(-1))
        m_loss = self.nll_loss(pm_prob.view(-1, pm_prob.size(2)), m_input.view(-1))
        if self.model_act:
            pa_loss = self.nll_loss(pa_prob.view(-1, pa_prob.size(2)), a_input.view(-1))
            loss = cfg.pz_loss_weight * pz_loss + m_loss + pa_loss
        else:
            pa_loss = torch.zeros(1)
            loss = cfg.pz_loss_weight * pz_loss + m_loss
        return loss, pz_loss, pa_loss, m_loss
