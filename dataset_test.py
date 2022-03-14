import numpy as np
from tqdm import tqdm
import argparse
import pickle as pkl
import json
from nltk import word_tokenize
import re
from torch.utils.data.dataset import Dataset
import numpy as np
from copy import deepcopy
import re
import requests

class dataset(object):
    def __init__(self,user_input, opt):
        self.entity2entityId=pkl.load(open('data/entity2entityId.pkl','rb')) #all entity from dbpedia
        self.entity_max=len(self.entity2entityId) #64362

        self.id2entity=pkl.load(open('data/id2entity.pkl','rb')) #all 6924a movies from dbpedia, same order as movie_ids 
        self.subkg=pkl.load(open('data/subkg.pkl','rb'))    #need not back process
        self.text_dict=pkl.load(open('data/text_dict.pkl','rb'))

        self.batch_size=opt['batch_size'] #32
        self.max_c_length=opt['max_c_length'] #256
        self.max_r_length=opt['max_r_length'] #30
        self.max_count=opt['max_count'] #5
        self.entity_num=opt['n_entity'] #64368

        self.data=[]
        self.corpus=[]

        cases=[]
        contexts= []
        entities= set()
        try: 
            with open('context_test2.txt','r',encoding='utf-8') as file1, open('output_test2.txt','r',encoding='utf-8') as file2:
                for line1, line2 in zip(file1, file2):
                    contexts.append(line1.rstrip().split(" "))
                    contexts.append(line2.rstrip().split(" "))
        except IOError:
            pass
        try:
            file = open('entity.txt', 'r', encoding="utf-8")
            Lines = file.readlines()
            for line in Lines:
                entities.add(int(line.strip()))
        except IOError:
            pass
        f=open('context_test2.txt','a',encoding='utf-8')
        f.write(user_input+ "\n")
        f.close()
        input_word= word_tokenize(user_input)
        contexts.append(input_word)
        s = '<a href=[^>]+>'
        base_url = "http://api.dbpedia-spotlight.org/en/annotate"# Parameters 
        params = {"text": user_input, "confidence": 0.35}# Response content type
        headers = {'accept': 'text/html'}# GET Request
        res = requests.get(base_url, params=params, headers=headers)
        links = re.findall(s, res.text)
        db_en= []
        for link in links:
            t= '(http|ftp|https):\/\/([\w_-]+(?:(?:\.[\w_-]+)+))([\w.,@?^=%&:\/~+#-]*[\w@?^=%&\/~+#-])'
            r= re.search(t, link)
            t= ''.join(['<', r.group(0), '>'])
            db_en.append(t)
        print("entity in user's turn: ", db_en)
        for entity in db_en:
            try:
                a= self.entity2entityId[entity]
                entities.add(a)
                f3=open('entity.txt','a',encoding='utf-8')
                f3.write( str(a)+"\n")
                f3.close() #get id for entity in entity2entityId, movie cannot be recognized.
            except:
                pass
        if len(entities)== 0:
            print("no entities metioned before yet")
        else:
            print("entities all mentioned before:", entities)
        cases.append(({'contexts': deepcopy(contexts), 'response': [], 'entity': deepcopy(entities), 'movie': 0, 'rec':0}))
        self.data.extend(cases) # append without for loop

        #if 'train' in filename:

        #self.prepare_word2vec()
        self.word2index = json.load(open('word2index_redial.json', encoding='utf-8'))
        self.key2index=json.load(open('key2index_3rd.json',encoding='utf-8'))
        #exactly the same two files????
        self.stopwords=set([word.strip() for word in open('stopwords.txt',encoding='utf-8')])

        #self.co_occurance_ext(self.data)
        #exit()

    def prepare_word2vec(self):
        import gensim
        model=gensim.models.word2vec.Word2Vec(self.corpus,size=300,min_count=1)
        model.save('word2vec_redial')
        word2index = {word: i + 4 for i, word in enumerate(model.wv.index2word)}
        #word2index['_split_']=len(word2index)+4
        #json.dump(word2index, open('word2index_redial.json', 'w', encoding='utf-8'), ensure_ascii=False)
        word2embedding = [[0] * 300] * 4 + [model[word] for word in word2index]+[[0]*300]
        import numpy as np
        
        word2index['_split_']=len(word2index)+4
        json.dump(word2index, open('word2index_redial.json', 'w', encoding='utf-8'), ensure_ascii=False)

        np.save('word2vec_redial.npy', word2embedding)

    def padding_w2v(self,sentence,max_length,transformer=True,pad=0,end=2,unk=3):
        vector=[]
        concept_mask=[]
        dbpedia_mask=[]
        for word in sentence:
            vector.append(self.word2index.get(word,unk))
            #if word.lower() not in self.stopwords:
            concept_mask.append(self.key2index.get(word.lower(),0))
            #else:
            #    concept_mask.append(0)
            if '@' in word:
                try:
                    entity = self.id2entity[int(word[1:])]
                    id=self.entity2entityId[entity]
                except:
                    id=self.entity_max
                dbpedia_mask.append(id)
            else:
                dbpedia_mask.append(self.entity_max)
        vector.append(end)
        concept_mask.append(0)
        dbpedia_mask.append(self.entity_max)

        if len(vector)>max_length:
            if transformer:
                return vector[-max_length:],max_length,concept_mask[-max_length:],dbpedia_mask[-max_length:]
            else:
                return vector[:max_length],max_length,concept_mask[:max_length],dbpedia_mask[:max_length]
        else:
            length=len(vector)
            return vector+(max_length-len(vector))*[pad],length,\
                   concept_mask+(max_length-len(vector))*[0],dbpedia_mask+(max_length-len(vector))*[self.entity_max]

    def padding_context(self,contexts,pad=0,transformer=True):
        vectors=[]
        vec_lengths=[]
        if transformer==False:
            if len(contexts)>self.max_count:
                for sen in contexts[-self.max_count:]:
                    vec,v_l=self.padding_w2v(sen,self.max_r_length,transformer)
                    vectors.append(vec)
                    vec_lengths.append(v_l)
                return vectors,vec_lengths,self.max_count
            else:
                length=len(contexts)
                for sen in contexts:
                    vec, v_l = self.padding_w2v(sen,self.max_r_length,transformer)
                    vectors.append(vec)
                    vec_lengths.append(v_l)
                return vectors+(self.max_count-length)*[[pad]*self.max_c_length],vec_lengths+[0]*(self.max_count-length),length
        else:
            contexts_com=[]
            for sen in contexts[-self.max_count:-1]:
                contexts_com.extend(sen)
                contexts_com.append('_split_')
            contexts_com.extend(contexts[-1])
            vec,v_l,concept_mask,dbpedia_mask=self.padding_w2v(contexts_com,self.max_c_length,transformer)
            return vec,v_l,concept_mask,dbpedia_mask,0

    def response_delibration(self,response,unk='MASKED_WORD'):
        new_response=[]
        for word in response:
            if word in self.key2index:
                new_response.append(unk)
            else:
                new_response.append(word)
        return new_response

    def data_process(self,is_finetune=False):
        data_set = []
        context_before = []
        for line in self.data:
            #if len(line['contexts'])>2:
            #    continue
            if is_finetune and line['contexts'] == context_before:
                continue
            else:
                context_before = line['contexts']
            context,c_lengths,concept_mask,dbpedia_mask,_=self.padding_context(line['contexts'])
            response,r_length,_,_=self.padding_w2v(line['response'],self.max_r_length)
            if False:
                mask_response,mask_r_length,_,_=self.padding_w2v(self.response_delibration(line['response']),self.max_r_length)
            else:
                mask_response, mask_r_length=response,r_length
            assert len(context)==self.max_c_length
            assert len(concept_mask)==self.max_c_length
            assert len(dbpedia_mask)==self.max_c_length

            data_set.append([np.array(context),c_lengths,np.array(response),r_length,np.array(mask_response),mask_r_length,line['entity'],
                             line['movie'],concept_mask,dbpedia_mask,line['rec']])
        return data_set

    def co_occurance_ext(self,data):
        stopwords=set([word.strip() for word in open('stopwords.txt',encoding='utf-8')])
        keyword_sets=set(self.key2index.keys())-stopwords
        movie_wordset=set()
        for line in data:
            movie_words=[]
            if line['rec']==1:
                for word in line['response']:
                    if '@' in word:
                        try:
                            num=self.entity2entityId[self.id2entity[int(word[1:])]]
                            movie_words.append(word)
                            movie_wordset.add(word)
                        except:
                            pass
            line['movie_words']=movie_words
        new_edges=set()
        for line in data:
            if len(line['movie_words'])>0:
                before_set=set()
                after_set=set()
                co_set=set()
                for sen in line['contexts']:
                    for word in sen:
                        if word in keyword_sets:
                            before_set.add(word)
                        if word in movie_wordset:
                            after_set.add(word)
                for word in line['response']:
                    if word in keyword_sets:
                        co_set.add(word)

                for movie in line['movie_words']:
                    for word in list(before_set):
                        new_edges.add('co_before'+'\t'+movie+'\t'+word+'\n')
                    for word in list(co_set):
                        new_edges.add('co_occurance' + '\t' + movie + '\t' + word + '\n')
                    for word in line['movie_words']:
                        if word!=movie:
                            new_edges.add('co_occurance' + '\t' + movie + '\t' + word + '\n')
                    for word in list(after_set):
                        new_edges.add('co_after'+'\t'+word+'\t'+movie+'\n')
                        for word_a in list(co_set):
                            new_edges.add('co_after'+'\t'+word+'\t'+word_a+'\n')
        f=open('co_occurance.txt','w',encoding='utf-8')
        f.writelines(list(new_edges))
        f.close()
        json.dump(list(movie_wordset),open('movie_word.json','w',encoding='utf-8'),ensure_ascii=False)

    def entities2ids(self,entities):
        return [self.entity2entityId[word] for word in entities]

    def detect_movie(self,sentence,movies):
        token_text = word_tokenize(sentence)
        num=0
        token_text_com=[]
        while num<len(token_text):
            if token_text[num]=='@' and num+1<len(token_text):
                token_text_com.append(token_text[num]+token_text[num+1])
                num+=2
            else:
                token_text_com.append(token_text[num])
                num+=1
        movie_rec = []
        for word in token_text_com:
            if word[1:] in movies:
                movie_rec.append(word[1:])
        movie_rec_trans=[]
        for movie in movie_rec:
            entity = self.id2entity[int(movie)]
            try:
                movie_rec_trans.append(self.entity2entityId[entity])
            except:
                pass
        return token_text_com,movie_rec_trans# movie in id from entity2entityId, token is still word

    def _context_reformulate(self,context,movies,altitude,ini_altitude,s_id,re_id):
        last_id=None
        #perserve the list of dialogue
        context_list=[] # a list of context_dict for each line
        for message in context: #every message like:  {"timeOffset": 73, "text": "How are you this morning?", "senderWorkerId": 696, "messageId": 158177}, 
            entities=[] #list of entity ids appearing in context
            try:
                for entity in self.text_dict[message['text']]:
                    try:
                        entities.append(self.entity2entityId[entity]) #get id for entity in entity2entityId, movie cannot be recognized.
                    except:
                        pass
            except:
                pass
            #token_text: all text in the message, including movie in the format of @124
            #movie_rec: movie in its original entity2Id id.
            token_text,movie_rec=self.detect_movie(message['text'],movies) #only movie mentioned in this line
            if len(context_list)==0:
                #context_dict{ entity: only entity ids and movie name in this context}
                context_dict={'text':token_text,'entity':entities+movie_rec,'user':message['senderWorkerId'],'movie':movie_rec}
                context_list.append(context_dict)
                last_id=message['senderWorkerId']
                continue
            if message['senderWorkerId']==last_id:
                context_list[-1]['text']+=token_text
                context_list[-1]['entity']+=entities+movie_rec
                context_list[-1]['movie']+=movie_rec 
            else:
                context_dict = {'text': token_text, 'entity': entities+movie_rec,# ids from entity2entityID
                           'user': message['senderWorkerId'], 'movie':movie_rec} #movie ids from entity2entityID
                context_list.append(context_dict)
                last_id = message['senderWorkerId']
                #if the two lines are actually from the same user, only append them, no need to create two context_dict to context_list
        #every line of messages return a context dict. how many line of messages result in how many line of context_dict.
        cases=[]
        contexts=[]
        entities_set=set()
        entities=[]
        for context_dict in context_list:
            self.corpus.append(context_dict['text']) #append a list of word
            if context_dict['user']==re_id and len(contexts)>0: #means not the first utterance from the machine, like "hello how can I help you" 
                response=context_dict['text']

                #entity_vec=np.zeros(self.entity_num)
                #for en in list(entities):
                #    entity_vec[en]=1
                #movie_vec=np.zeros(self.entity_num+1,dtype=np.float)
                if len(context_dict['movie'])!=0: #this response indeed recommend a movie
                    for movie in context_dict['movie']:
                        #if movie not in entities_set:
                        cases.append({'contexts': deepcopy(contexts), 'response': response, 'entity': deepcopy(entities), 'movie': movie, 'rec':1})
                else: #no movie recommendation in this response
                    cases.append({'contexts': deepcopy(contexts), 'response': response, 'entity': deepcopy(entities), 'movie': 0, 'rec':0})
                #destint whether the case have recommended any movie or not 
                contexts.append(context_dict['text'])
                for word in context_dict['entity']:
                    if word not in entities_set:
                        #actually append the movies ids
                        entities.append(word)
                        entities_set.add(word) #only store distinct ids
            else: #the round for sender, not responder
                contexts.append(context_dict['text'])
                for word in context_dict['entity']:
                    if word not in entities_set:
                        entities.append(word)
                        entities_set.add(word)
        return cases

class CRSdataset(Dataset):
    def __init__(self, dataset, entity_num, concept_num):
        self.data=dataset
        self.entity_num = entity_num
        self.concept_num = concept_num+1

    def __getitem__(self, index):
        '''
        movie_vec = np.zeros(self.entity_num, dtype=np.float)
        context, c_lengths, response, r_length, entity, movie, concept_mask, dbpedia_mask, rec = self.data[index]
        for en in movie:
            movie_vec[en] = 1 / len(movie)
        return context, c_lengths, response, r_length, entity, movie_vec, concept_mask, dbpedia_mask, rec
        '''
        context, c_lengths, response, r_length, mask_response, mask_r_length, entity, movie, concept_mask, dbpedia_mask, rec= self.data[index]
        entity_vec = np.zeros(self.entity_num)
        entity_vector=np.zeros(50,dtype=np.int)
        point=0
        for en in entity:
            entity_vec[en]=1
            entity_vector[point]=en
            point+=1

        concept_vec=np.zeros(self.concept_num)
        for con in concept_mask:
            if con!=0:
                concept_vec[con]=1

        db_vec=np.zeros(self.entity_num)
        for db in dbpedia_mask:
            if db!=0:
                db_vec[db]=1

        return context, c_lengths, response, r_length, mask_response, mask_r_length, entity_vec, entity_vector, movie, np.array(concept_mask), np.array(dbpedia_mask), concept_vec, db_vec, rec

    def __len__(self):
        return len(self.data)




