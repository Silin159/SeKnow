import json

class Ontology:
    def __init__(self, ontology_path):
        self.ontology_path = ontology_path
        self.informable_slots_dict = {}
        '''
        self.eos_syntax = {'resp': '<eos_r>', 'user': '<eos_u>', 'resp_gen': '<eos_r>'}
        '''
        self.eos_syntax = {'resp': '<eos_r>', 'user': '<eos_u>', 'topic': '<eos_t>', 'knowledge': '<eos_k>',
                           'resp_gen': '<eos_r>', 'topic_gen': '<eos_t>'}
        '''
        self.special_tokens = ['<pad>', '<go_r>', '<unk>', '<eos_u>', '<eos_r>', '<eos_b>',
                                            '<eos_as>', '<eos_av>', '<go_as>', '<go_av>']   #0-9
        '''
        self.special_tokens = ['<pad>', '<go_r>', '<unk>', '<eos_u>', '<eos_r>', '<eos_b>', '<eos_as>',
                               '<eos_av>', '<go_as>', '<go_av>', '<go_t>', '<eos_t>', '<eos_k>']  #0-12

    def _get_z_eos_map(self, informable_slots):
        z_eos_map = {}
        for idx, slot in enumerate(informable_slots):
            z_eos_map[slot] = '<eos_b%d>'%(idx+1)
        return z_eos_map

    def _slots_flatten(self, slots_dict):
        flat_slots = []
        for domain, slots in slots_dict.items():
            for slot in slots:
                flat_slots.append('%s-%s'%(domain, slot))
        return flat_slots

    def _get_slot_name_set(flat_slot_list):
        unique_list = []
        for s in flat_slot_list:
            s = s.split('-')[1]
            if s not in unique_list:
                unique_list.append(s)
        return unique_list

    def covert_mask_words_to_idx(self, vocab):
        slot_value_mask_idx = {}
        for s, values in self.slot_value_mask.items():
            slot_value_mask_idx[s] = []
            for v in values:
                slot_value_mask_idx[s].append(vocab.encode(v))
        return slot_value_mask_idx


class CamRest676Ontology(Ontology):
    def __init__(self, ontology_path):
        super().__init__(ontology_path)
        self.informable_slots_dict = {'restaurant': ['food', 'pricerange', 'area']}
        # self.informable_slots_dict = {'restaurant': ['food', 'area', 'pricerange']}
        self.informable_slots = self._slots_flatten(self.informable_slots_dict)
        self.requestable_slots = ['address', 'name', 'phone', 'postcode', 'food', 'area', 'pricerange']
        self.z_eos_map = self._get_z_eos_map(self.informable_slots)
        self.slot_value_mask = self._get_ontology_index_mask(self.ontology_path)
        self.special_tokens.extend(list(self.z_eos_map.values()))
        self.special_tokens.extend(['[value_%s]'%w for w in self.requestable_slots])
        self.special_tokens.extend(['food', 'price', 'area', 'dontcare'])
        # print(self.informable_slots )
        # print(self.z_eos_map)
        # print(self.slot_value_mask)


    def _get_ontology_index_mask(self, ontology_path):
        # Return the indexes of all words in the values of each slots
        # To be used as probability masks  while decoding z
        entity_idx = {}
        raw_entities = json.loads(open(ontology_path).read().lower())
        for slot, values in raw_entities['informable'].items():
            slot = 'restaurant-' + slot
            entity_idx[slot] = set(['<pad>', 'dontcare', self.z_eos_map[slot]])
            for v in values:
                w_list = v.split()
                for w in w_list:
                    entity_idx[slot].add(w)
            # if 'the' in entity_idx[slot]:
            #     print('delete the')
            #     entity_idx[slot].discard('the')
            # #     entity_idx[slot].add('restrauant')
            # #     entity_idx[slot].add('toward')
            # if 'moderate' in entity_idx[slot]:
            #     print('add moderately')
            #     entity_idx[slot].add('moderately')
            # entity_idx[slot].discard('dontcare')
            entity_idx[slot] = list(entity_idx[slot])
        return entity_idx


class KvretOntology(Ontology):
    def __init__(self, ontology_path):
        super().__init__(ontology_path)
        self.all_domains = ['weather', 'navigate', 'schedule']
        self.informable_slots_dict =  {
            'weather': ['date','location','weather_attribute'],
            'navigate': ['poi_type','distance'],
            'schedule': ['event', 'date', 'time', 'agenda', 'party', 'room']
        }
        # self.informable_slots_dict = {'restaurant': ['food', 'area', 'pricerange']}
        self.informable_slots = self._slots_flatten(self.informable_slots_dict)
        self.requestable_slots_dict = {
            'weather': ['weather_attribute'],
            'navigate': ['poi','traffic_info','address','distance'],
            'schedule': ['event','date','time','party','agenda','room']
        }
        self.requestable_slots = self._slots_flatten(self.requestable_slots_dict)
        self.z_eos_map = self._get_z_eos_map(self.informable_slots)
        # self.slot_value_mask = self._get_ontology_index_mask(self.ontology_path)

        self.special_tokens.extend(list(self.z_eos_map.values()))
        for d_s in self.requestable_slots:
            d, s = d_s.split('-')
            if '[value_%s]'%s not in self.special_tokens:
                self.special_tokens.append('[value_%s]'%s)
        for d_s in self.informable_slots:
            d, s = d_s.split('-')
            if s not in self.special_tokens:
                self.special_tokens.append(s)
        self.special_tokens.extend(['dontcare'])
        # print(self.informable_slots )
        # print(self.z_eos_map)
        # print(self.slot_value_mask)


class MultiwozOntology(Ontology):
    def __init__(self, ontology_path=None):
        super().__init__(ontology_path)
        self.all_domains = ['restaurant', 'hotel', 'attraction', 'train', 'taxi', 'police', 'hospital']
        self.db_domains = ['restaurant', 'hotel', 'attraction', 'train', 'hospital']
        self.kb_domains = ['restaurant', 'hotel', 'train', 'taxi']  # domains for seeking external knowledge base
        self.slot_normlize = {
            "car type": "car",
            "entrance fee": "price",
            "leaveat": 'leave',
            'arriveby': 'arrive',
            'trainid': 'id',
            'addr': "address",
            'post': "postcode",
            'ref': 'reference',
            'fee': "price",
            'ticket': 'price',
            'price range': 'pricerange',
            'price': 'pricerange',
            'depart': "departure",
            'dest': "destination",
        }
        '''
        self.informable_slots_dict = {
            "restaurant": ["food", "pricerange", "area", "name", "time", "day", "people"],
            "hotel": ["type", "parking", "pricerange", "internet", "stay", "day", "people", "area", "stars", "name"],
            "attraction": ["area", "type", "name"],
            "train": ["destination", "day", "arrive", "departure", "people", "leave"],
            "taxi": ["leave", "destination", "departure", "arrive"],
            "police": [],
            "hospital": ["department"],
        }
        '''
        # add "sfek" to four augmented domains
        self.informable_slots_dict = {
            "restaurant": ["food", "pricerange", "area", "name", "time", "day", "people", "sfek"],
            "hotel": ["type", "parking", "pricerange", "internet", "stay", "day", "people", "area", "stars", "name","sfek"],
            "attraction": ["area", "type", "name"],
            "train": ["destination", "day", "arrive", "departure", "people", "leave", "sfek"],
            "taxi": ["leave", "destination", "departure", "arrive", "sfek"],
            "police": [],
            "hospital": ["department"],
        }
        self.informable_slots = self._slots_flatten(self.informable_slots_dict)
        self.requestable_slots_dict = {
            "restaurant": ["phone", "postcode", "address", "pricerange", "food", "area", "reference"],
            "hotel": ["address", "postcode", "internet", "phone", "parking", "type", "pricerange", "stars", "area", "reference"],
            "attraction": ["price", "type", "address", "postcode", "phone", "area", "reference"],
            "train": ["duration", "leave", "price", "arrive", "id", "reference"],
            "taxi": ["car", "phone"],
            "police": ["postcode", "address", "phone"],
            "hospital": ["address", "phone", "postcode"]
        }
        self.requestable_slots = self._slots_flatten(self.requestable_slots_dict)
        self.z_eos_map = self._get_z_eos_map(self.informable_slots)
        # self.slot_value_mask = self._get_ontology_index_mask(self.ontology_path)
        # print(self.informable_slots )
        # print(self.z_eos_map)
        # print(self.slot_value_mask)

        self.special_tokens.extend(list(self.z_eos_map.values()))
        for d_s in self.requestable_slots:
            d, s = d_s.split('-')
            if '[value_%s]'%s not in self.special_tokens:
                self.special_tokens.append('[value_%s]'%s)
        for d_s in self.informable_slots:
            d, s = d_s.split('-')
            if s not in self.special_tokens:
                self.special_tokens.append(s)
        self.special_tokens.extend(["dontcare"])
        # add wildcard entity "*"
        self.special_tokens.extend(["*"])
