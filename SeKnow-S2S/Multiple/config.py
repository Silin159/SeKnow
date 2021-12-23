import logging, os, time
import math

class _Config:
    def __init__(self):
        """
        Important hyperparameters
        """
        # self._multiwoz_init()
        pass

    def init_handler(self, m):
        init_method = {
            'camrest': self._camrest_init,
            'kvret': self._kvret_init,
            'multiwoz': self._multiwoz_init,
        }
        init_method[m]()

    def init_logging_handler(self, mode):
        stderr_handler = logging.StreamHandler()
        if not os.path.exists('./log'):
            os.mkdir('./log')
        if not os.path.exists('./log/'+self.dataset):
            os.mkdir('./log/'+self.dataset)

        if self.save_log and self.mode == 'train' or self.mode == 'adjust' :
            file_handler = logging.FileHandler('./log/{}/log_{}_{}_{}_sd{}.txt'.format(self.dataset, self.log_time, mode, self.exp_no, self.seed))
            file_handler2 = logging.FileHandler(os.path.join(self.exp_path, 'log_{}_{}.txt'.format(self.mode, self.log_time)))
            logging.basicConfig(handlers=[stderr_handler, file_handler, file_handler2])

        elif self.mode == 'test':
            eval_log_path = os.path.join(self.eval_load_path, 'eval_log.json')
            # if os.path.exists(eval_log_path):
            #     os.remove(eval_log_path)
            file_handler = logging.FileHandler(eval_log_path)
            logging.basicConfig(handlers=[stderr_handler, file_handler])
        else:
            logging.basicConfig(handlers=[stderr_handler])
        logger = logging.getLogger()
        logger.setLevel(logging.INFO)

    def _multiwoz_init(self):
        # experimental settings
        self.seed = 47
        self.cuda = True
        self.cuda_device = 0
        self.exp_no = 'test'
        self.exp_path = 'to be generated'
        self.model_path = 'to be generated'
        self.result_path = 'to be generated'
        self.result_path_ctr = 'to be generated'
        self.global_record_path = './multiwoz_results.csv'
        self.eval_load_path = 'to be generated'
        '''
        self.vocab_path = './data/MultiWOZ/processed/vocab'
        self.data_path = './data/MultiWOZ/processed/'
        '''
        self.vocab_path = './data/MultiWOZ/processed_aug/vocab'
        self.data_path = './data/MultiWOZ/processed_aug/'
        self.data_file = os.path.join(self.data_path, 'data_processed.json')
        '''
        self.raw_data = 'data/MultiWOZ/MULTIWOZ2.1/data.json'
        self.dev_list = 'data/MultiWOZ/MULTIWOZ2.1/valListFile.json'
        self.test_list = 'data/MultiWOZ/MULTIWOZ2.1/testListFile.json'
        self.db_paths = 'data/MultiWOZ/processed/db_processed.json'
        self.dial_goals = 'data/MultiWOZ/processed/goal_of_each_dials.json'
        self.domain_file_path = 'data/MultiWOZ/processed/domain_files.json'
        '''
        self.raw_data = 'data/MultiWOZ/MULTIWOZ2.1/data_aug.json'
        self.dev_list = 'data/MultiWOZ/MULTIWOZ2.1/valListFile_aug.json'
        self.test_list = 'data/MultiWOZ/MULTIWOZ2.1/testListFile_aug.json'
        self.kb_paths = 'data/MultiWOZ/processed_aug/kb_processed.json'
        self.db_paths = 'data/MultiWOZ/processed_aug/db_processed.json'
        self.dial_goals = 'data/MultiWOZ/processed_aug/goal_of_each_dials.json'
        self.domain_file_path = 'data/MultiWOZ/processed_aug/domain_files.json'
        self.slot_value_set_path = 'db/value_set_processed.json'

        self.save_log = True
        self.log_time = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())

        # model settings
        self.vocab_size = 3000
        self.hidden_size = 512
        self.embed_size = self.hidden_size  # GRU: 50
        self.layer_num = 2  # GRU: 1
        self.feed_size = 2048  # for Transformers
        self.attn_head = 8  # for Transformers
        self.db_vec_size = 6  # unchangeable
        self.z_length = 5  # the maximum length of value in dsv
        self.a_length = 5
        self.t_length = 5
        self.k_max_len = 40  # not used in decoding
        self.k_sample_size = 5
        self.u_max_len = 100  # not used in decoding
        self.m_max_len = 40
        self.pos_max_len = 200
        self.prev_z_continuous = False
        self.m_copy_u = True
        self.multi_domain = True
        self.freeze_emb = False
        self.share_q_encoder = False
        self.model_act = False
        self.share_act_decoder = True
        self.use_act_slot_decoder = False
        self.dropout_st = True
        self.use_resp_dpout = False
        self.glove_init = False  # GRU: True
        self.changedp = False

        # trainning settings
        self.lr = 0.0003  # GRU: 0.0005 -- 0.001
        self.decay_rate = 1.0
        self.beta1 = 0.9  # for Transformers
        self.beta2 = 0.98  # for Transformers
        self.eps = 1e-9  # for Transformers
        self.lr_decay = 0.5  # for GRU
        self.early_stop_count = 4  # for GRU
        self.weight_decay_count = 2  # for GRU
        self.valid_type = 'combined'
        self.batch_size = 64
        self.dropout_rate = 0.1
        self.dropout_rate_gru = 0.35
        self.warmup_steps = 4000  # for Transformers
        self.max_epoch = 100
        self.min_epoch = 0
        self.teacher_force = 100  # for GRU
        self.spv_proportion = 100
        self.no_label_train = False
        self.skip_unsup = False
        self.sup_pretrain = 40
        self.mask_ontology = False
        self.parallel_z = False
        self.train_ctr = False
        self.entity_amend = True

        # evaluation settings
        self.beam_search = False
        self.beam_params = {'size': 10, 'len_bonus': 0.5}
        self.eval_z_method = 'prior'
        self.use_true_domain_for_ctr_eval = True
        self.use_true_bspn_for_ctr_eval = False

        # SMC params
        self.particle_num = 5
        self.topk_num = 5
        self.resampling = True
        self.weighted_grad = True
        self.pz_loss_weight = 1

        # JSA params
        self.use_pimh = False
        self.pimh_iter_num = 1
        if not self.use_pimh:
            self.pimh_iter_num = 1

        # VAE params
        self.sample_type = 'topk'
        self.gumbel_temp = 1.0
        self.kl_loss_weight = 0.5


    def __str__(self):
        s = ''
        for k, v in self.__dict__.items():
            s += '{} : {}\n'.format(k, v)
        return s


global_config = _Config()
