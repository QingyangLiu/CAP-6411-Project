
import numpy as np
import torch
import glob, json, re, en_core_web_lg
import torch.utils.data as Data
import ipdb



class NuScenes_QA(Data.Dataset):
    def __init__(self, __C):
        super(NuScenes_QA).__init__()
        self.__C = __C

        # --------------------------
        # ---- Raw data loading ----
        # --------------------------

        qa_dict_preread = {
            'train': json.load(open(__C.RAW_PATH['train'], 'r')),
            'val': json.load(open(__C.RAW_PATH['val'], 'r'))}
        
        self.qa_list = []
        split = __C.SPLIT[__C.RUN_MODE]
        self.qa_list += qa_dict_preread[split]['questions']
        self.data_size = self.qa_list.__len__()

        print(split, ' dataset size:', self.data_size)
        
        # {question id} -> {question}
        self.qid2ques = self.ques_load(self.qa_list)

        # Loading scene features path
        # scene_feat_path_list = glob.glob(__C.FEATS_PATH[__C.VISUAL_FEATURE][split] + '/*.npz')
        scene_feat_path_list = glob.glob(__C.VISUAL_FEATURE + '/*.npz')

        # {scene token} -> {scene feature absolutely path}
        self.stk2featpath = self.scene_feat_path_load(scene_feat_path_list)

        self.token_history = self.precompute_token_history(
            '/home/qi940700/Desktop/NuScenes-QA-new/datasets/nuscenes/v1.0-trainval/sample.json', num_frames=10
        )

        # Tokenize and load glove embedding
        if __C.MODEL != 'clip-adpter':
            self.token2ix, self.pretrained_emb = self.tokenize(qa_dict_preread)
            self.token_size = self.token2ix.__len__()
        else:
            self.token_size = None
            self.pretrained_emb = None

        self.ans2ix, self.ix2ans = self.load_ans_table('./src/datasets/answer_dict.json')
        self.ans_size = self.ans2ix.__len__()
        print('Data Loading Finished!')
        print('')


    def ques_load(self, qa_list):
        qid2ques = {}

        qid = 0
        for item in qa_list:
            qid2ques[qid] = item
            qid += 1

        return qid2ques
    
    def scene_feat_path_load(self, path_list):
        stk2path = {}

        for path in path_list:
            stk = str(path.split('/')[-1].split('.')[0])
            stk2path[stk] = path

        return stk2path
    
    def tokenize(self, qa_dict):
        token2ix = {
            'PAD': 0,
            'UNK': 1,
            'CLS': 2,
        }

        spacy_tool = en_core_web_lg.load()
        pretrained_emb = []
        pretrained_emb.append(spacy_tool('PAD').vector)
        pretrained_emb.append(spacy_tool('UNK').vector)
        pretrained_emb.append(spacy_tool('CLS').vector)

        ques_list = []
        for split in qa_dict:
            qa_list = qa_dict[split]['questions']
            for item in qa_list:
                ques_list.append(item['question'])

        for ques in ques_list:
            words = re.sub(
                r"([.,'!?\"()*#:;])",
                '',
                ques.lower()
            ).replace('-', ' ').replace('/', ' ').split()

            for word in words:
                if word not in token2ix:
                    token2ix[word] = len(token2ix)
                    pretrained_emb.append(spacy_tool(word).vector)
        
        pretrained_emb = np.array(pretrained_emb)

        return token2ix, pretrained_emb
    
    def load_ans_table(self, answer_table):
        ans2ix, ix2ans = json.load(open(answer_table, 'r'))
        return ans2ix, ix2ans
    

    def __getitem__(self, idx):

        ques_ix_iter, ans_iter, scene_token = self.load_ques_ans(idx)
        obj_feat_iter, bbox_feat_iter = self.load_obj_feats(scene_token)

        return \
            torch.from_numpy(obj_feat_iter),\
            torch.from_numpy(bbox_feat_iter),\
            torch.from_numpy(ques_ix_iter),\
            torch.from_numpy(ans_iter)
    
    
    def __len__(self):
        return self.data_size
    

    # ----------------------------------------------
    # ---- Real-Time Processing Implementations ----
    # ----------------------------------------------

    def load_ques_ans(self, idx):
        ques = self.qa_list[idx]['question']
        scene_token = self.qa_list[idx]['sample_token']

        ques_ix_iter = self.proc_ques(ques, max_token=30)
        ans_iter = np.zeros(1)

        if self.__C.RUN_MODE in ['train']:
            ans = self.qa_list[idx]['answer']
            ans_iter = self.proc_ans(ans, self.ans2ix)
        
        return ques_ix_iter, ans_iter, scene_token
    
    # def load_obj_feats(self, scene_token):
    #     det_results = np.load(self.stk2featpath[scene_token], allow_pickle=True)['results']
    #     num_obj = det_results.shape[0]
    #     obj_feat = []
    #     bbox = []
    #     label = []
    #     for i in range(num_obj):
    #         obj = det_results[i]
    #         obj_feat.append(obj['feats'])
    #         bbox.append(obj['box'][:7])
    #         label.append(obj['label'])
    #     # empty detection
    #     if obj_feat == []:
    #         obj_feat = np.zeros((1, 512)).astype(np.float32)
    #         bbox = np.zeros((1, 7)).astype(np.float32)
    #     obj_feat = np.stack(obj_feat, axis=0) # [num_obj, feat_dim]
    #     bbox = np.stack(bbox, axis=0) # [num_obj, 7]
    #     obj_feat_iter = self.proc_scene_feat(obj_feat, feat_pad_size=self.__C.FEAT_SIZE['OBJ_FEAT_SIZE'][0])
    #     bbox_feat_iter = self.proc_scene_feat(self.proc_bbox_feat(bbox, None), feat_pad_size=self.__C.FEAT_SIZE['BBOX_FEAT_SIZE'][0])

    #     return obj_feat_iter.astype(np.float32), bbox_feat_iter.astype(np.float32)




    def precompute_token_history(self, sample_json_path, num_frames=3):
        with open(sample_json_path, 'r') as f:
            token_data = json.load(f)

        token_map={entry['token']: entry for entry in token_data}

        token_history={}
        for token, entry in token_map.items():
            token_history[token]=[]
            token_history[token].append(token)
            currToken=token
            for x in range(num_frames):
                prev_token=token_map.get(currToken, {}).get('prev', None)
                if prev_token==None or prev_token=='':
                    break
                token_history[token].append(prev_token)
                currToken=prev_token
            #print(token_history[token])
            #input('continue')
        return token_history





    def precompute_token_history_old(self, sample_json_path):
        # Load JSON file to compute token history
        with open(sample_json_path, 'r') as f:
            token_data = json.load(f)

        token_map = {entry['token']: entry for entry in token_data}
        print(token_map)
        input('continue')

        # Precompute history mapping
        token_history = {}
        for token, entry in token_map.items():
            prev_token_1 = token_map.get(token, {}).get('prev', None)
            prev_token_2 = token_map.get(prev_token_1, {}).get('prev', None)
            token_history[token] = [t for t in [token, prev_token_1, prev_token_2] if t]

        return token_history

    def load_obj_feats(self, scene_token):
        

        tokens = self.token_history.get(scene_token, [])

        obj_feats = []
        bbox_feats = []

        for token in tokens:
            if token not in self.stk2featpath:  # Skip tokens without features
                continue

            det_results = np.load(self.stk2featpath[token], allow_pickle=True)['results']
            num_obj = det_results.shape[0]
            obj_feat = []
            bbox = []

            for i in range(num_obj):
                obj = det_results[i]
                obj_feat.append(obj['feats'])  # Object features
                bbox.append(obj['box'][:7])   # Bounding box (7-dimensional)

            # Handle empty detections
            if not obj_feat:
                obj_feat = np.zeros((1, 512)).astype(np.float32)
                bbox = np.zeros((1, 7)).astype(np.float32)

            obj_feat = np.stack(obj_feat, axis=0)  # [num_obj, feat_dim]
            bbox = np.stack(bbox, axis=0)         # [num_obj, 7]

            if obj_feat.shape[0] > 100:
                obj_feat = obj_feat[:100]  # Truncate to 100
            elif obj_feat.shape[0] < 100:
                # Pad to 100
                padding = np.zeros((100 - obj_feat.shape[0], obj_feat.shape[1]))  # Pad with zeros
                obj_feat = np.vstack([obj_feat, padding])  # Stack the padding to the end

            obj_feats.append(obj_feat)
            bbox_feats.append(bbox)

        # Concatenate features along the object dimension
        obj_feats = np.concatenate(obj_feats, axis=0) if obj_feats else np.zeros((0, 512))
        bbox_feats = np.concatenate(bbox_feats, axis=0) if bbox_feats else np.zeros((0, 7))

        # Process features
        #obj_feat_iter = self.proc_scene_feat(obj_feats, feat_pad_size=self.__C.FEAT_SIZE['OBJ_FEAT_SIZE'][0])
        obj_feat_iter = self.proc_scene_feat(obj_feats, feat_pad_size=1000)
        bbox_feat_iter = self.proc_scene_feat(self.proc_bbox_feat(bbox_feats, None), feat_pad_size=1000)

        return obj_feat_iter.astype(np.float32), bbox_feat_iter.astype(np.float32)



    # ------------------------------------
    # ---- Real-Time Processing Utils ----
    # ------------------------------------

    def proc_ques(self, ques, max_token):
        if self.__C.MODEL != 'clip-adpter':
            token2ix = self.token2ix
            ques_ix = np.zeros(max_token, np.int64)

            words = re.sub(
                r"([.,'!?\"()*#:;])",
                '',
                ques.lower()
            ).replace('-', ' ').replace('/', ' ').split()

            for ix, word in enumerate(words):
                if word in token2ix:
                    ques_ix[ix] = token2ix[word]
                else:
                    ques_ix[ix] = token2ix['UNK']

                if ix + 1 == max_token:
                    break
        
        else:
            # use clip tokenizer
            import clip
            ques_ix = clip.tokenize([ques], context_length=max_token).numpy()[0]
        
        return ques_ix
    
    def proc_ans(self, ans, ans2ix):
        ans = str(ans)
        ans_ix = np.zeros(1, np.int64)
        ans_ix[0] = ans2ix[ans]

        return ans_ix
    
    def proc_scene_feat(self, feat, feat_pad_size):
        #feat_pad_size=300
        if feat.shape[0] > feat_pad_size:
            feat = feat[:feat_pad_size]
        
        feat = np.pad(
            feat,
            ((0, feat_pad_size-feat.shape[0]), (0, 0)),
            mode='constant',
            constant_values=0
        )

        return feat
    
    def proc_bbox_feat(self, bbox, scene_shape):
        if self.__C.BBOX_NORMALIZE:
            raise NotImplementedError()
        
        return bbox