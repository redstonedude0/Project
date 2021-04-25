# This file obtains the training and evaluation datasets and prepares it for use.
#
# Note: Data provided has been pre-computed with p_e_m probabilities
# Data is from sources:
# training set:
#   AIDA-CoNLL>AIDA-train - 946 docs
# dev evaluation set:
#   AIDA-CoNLL>AIDA-A - 216 docs
# test evaluation sets:
#   AIDA-CoNLL>AIDA-B - 231 docs
#   MSNBC - 20 docs
#   AQUAINT - 50 docs
#   ACE2004 - 36 docs
#   WNED-CWEB(CWEB) - 320 docs
#   WNED-WIKI(WIKI) - 320 docs
#
# AIDA sets are considered in-domain, other sets are considered out-domain.
#

from enum import Enum, auto

import our_consistency
from datastructures import Dataset, Mention, Candidate, Document


class DatasetType(Enum):
    TRAIN = auto()  # training
    TEST = auto()  # evaluation
    DEV = auto()  # evaluation

from hyperparameters import SETTINGS


def loadDataset(csvPath: str,conllPath: str):
    #Load dataset initially
    csvPath = SETTINGS.dataDir_csv + csvPath
    dataset = Dataset()
    dataset.documents = []
    with open(csvPath, "r") as f:
        # Iterate over CSV structure - each line is a mention, when the ID changes the doc changes
        doc = Document()
        mentionid = 0
        for line in f:
            mention = Mention()
            mention.id = mentionid
            mentionid += 1
            parts = line.split("\t")
            doc_id1 = parts[0]
            doc_id2 = parts[1]
            if doc_id1 != doc_id2:
                # As I understand it the 1st 2 columns are equal, raise error if this invariant is broken
                raise NotImplementedError(f"Document ids not equal {doc_id1} {doc_id2}")
            if doc_id1 != doc.id:
                # New doc
                if doc.id != None:  # None if initial line
                    dataset.documents.append(doc)
                doc = Document()
                doc.mentions = []
            doc.id = doc_id1  # make sure always set
            # Actually set up mention
            mention.text = parts[2]
            mention.left_context = parts[3]
            mention.right_context = parts[4]
            if parts[5] != "CANDIDATES":
                # As I understand it this is always equal, raise error if this invariant is broken
                raise NotImplementedError(f"Col5 not CANDIDATES on {doc_id1}")
            candidates = [cand for cand in parts[6:-2]]
            if len(candidates) == 1 and candidates[0] == "EMPTYCAND":
                candidates = []
            candidates = [cand.split(",") for cand in candidates]
            candidates = [(cand[0], cand[1], cand[2:]) for cand in candidates]  # ERRORS
            candidates = [Candidate(id, float(prob), ",".join(nameparts)) for (id, prob, nameparts) in candidates]
            mention.candidates = candidates
            mention.gold_id = -1  # no id by default
            if parts[-2] != "GT:":
                # As I understand it this is always equal, raise error if this invariant is broken
                raise NotImplementedError(f"Col-2 not GT on {doc_id1}")
            goldDataParts = parts[-1].split(",")
            if len(goldDataParts) != 1:  # otherwise -1 anyway
                mention.gold_id = goldDataParts[1]  # the ID of the candidate
            doc.mentions.append(mention)
        dataset.documents.append(doc)  # Append last document

    #Add in proper noun coref (according to paper)
    if SETTINGS.switches["aug_coref"]:
        names = []
        with open(SETTINGS.dataDir_personNames, 'r', encoding='utf8') as f:
            for line in f:
                names.append(line)#Note - plaintext-normalised!!!!
        from tqdm import tqdm
        for document in tqdm(dataset.documents,unit="docs"):
            for mention in document.mentions:
                #cur_m = ment['mention'].lower()
                coreferences = []
                for mention_other in document.mentions:
                    if mention == mention_other:
                        continue#Don't check same mention
                    if len(mention_other.candidates) == 0:
                        continue#No candidates, skip
                    if mention_other.candidates[0].text not in names:
                        continue#Main candidate isn't a name, skip

                    #Is mention a substring of other mention?
                    start_pos = mention_other.text.lower().find(mention.text.lower())
                    if start_pos == -1:
                        continue#Not - return

                    end_pos = start_pos + len(mention) - 1 #Find end of the substring of mention
                    # If the mention is actually embedded in the mention_other:
                    # If starts as standalone word
                    if (start_pos == 0 or mention_other[start_pos - 1] == ' '):
                        #If ends as standarlone word
                        if (end_pos == len(mention_other) - 1 or mention_other[end_pos + 1] == ' '):
                            #Start/end are 'whitespace', this is a coreference
                            coreferences.append(mention_other)
                if len(coreferences) > 0:
                    candidate_pems = {}#Map of coref candidate ids, to SUM of p_e_m scores
                    textmap = {}
                    for mention_other in coreferences:
                        for candidate in mention_other.candidates:
                            candidate_pems[candidate.id] = candidate_pems.get(candidate.id, 0) + candidate.initial_prob
                            textmap[candidate.id] = candidate.text
                    #make average p_e_m across all coreferences
                    for candidate_id in candidate_pems.keys():
                        candidate_pems[candidate_id] /= len(coreferences)
                    #Set candidates, sorted in descending p_e_m scores
                    mention.candidates = sorted([Candidate(cid,pem,textmap[cid]) for cid,pem in candidate_pems.items()], key=lambda cand: cand.initial_prob,reverse=True)

    #Add in conll sentence data (according to paper)
    if SETTINGS.switches["aug_conll"]:
        conll_data = {}#Extract sentences and mentions
        with open(SETTINGS.dataDir_conll + conllPath, 'r', encoding='utf8') as f:
            current_document = None
            current_sentence = None
            for line in f:
                line = line.strip()#Trim line
                if line.startswith('-DOCSTART-'):#Wait for a docstart signal
                    document_name = " ".join(line.split()[1:])#just the bit in brackets
                    document_name = document_name[1:]#Remove open bracket (keep closing)
                    conll_data[document_name] = {'sentences': [], 'mentions': []} #Add new doc
                    current_document = conll_data[document_name]
                    current_sentence = [] #doc ends in new line, sentence can be saved lated
                else:
                    if line == '':#Save current sentence
                        current_document['sentences'].append(current_sentence)
                        current_sentence = []
                    else:
                        bits = line.split('\t')
                        word = bits[0]
                        current_sentence.append(word)
                        if len(bits) >= 6:#If marked a NE (mention)
                            pos = bits[1] #has POS
                            #golden = bits[4]
                            if pos == 'I':
                                current_document['mentions'][-1]['end_word_idx'] += 1 #Extension of previous mention
                            else:
                                current_document['mentions'].append({
                                    'sentence_idx': len(current_document['sentences']),
                                    'start_word_idx': len(current_sentence) - 1,
                                    'end_word_idx': len(current_sentence)
                                })
        for document in dataset.documents:
            conll_doc = conll_data.get(document.id,{"mentions":[]})
            document.conll_tokens = []
            missed = 0
            for mention_idx,mention in enumerate(document.mentions):
                while mention_idx+missed < len(conll_doc['mentions']):
                    mention_details = conll_doc['mentions'][mention_idx+missed]
                    sentence_for_mention = conll_doc['sentences'][mention_details['sentence_idx']]
                    our_consistency.save(sentence_for_mention,"embs_i_sent")
                    conll_start = mention_details['start_word_idx']
                    conll_end = mention_details['end_word_idx']
                    conll_lctx = sentence_for_mention[:conll_start]
                    conll_mctx = sentence_for_mention[conll_start:conll_end]
                    conll_rctx = sentence_for_mention[conll_end:]
                    conll_mention_string = ' '.join(conll_mctx)
                    if conll_mention_string.lower() == mention.text.lower():#Sanity check - are the mentions equal?
                        mention.conll_lctx = conll_lctx
                        mention.conll_mctx = conll_mctx
                        mention.conll_rctx = conll_rctx
                        break
                    else: #Mentions not equal, must've missed a conll mention, move on
                        missed += 1

    return dataset

# Removed - datasets can be loaded as necessary
#    loadDataset(train_AIDA, "aida_train.csv")
#    loadDataset(dev_AIDA, "aida_testA.csv")
#    loadDataset(test_AIDA, "aida_testB.csv")
#    loadDataset(test_MSNBC, "wned-msnbc.csv")
#    loadDataset(test_AQUAINT, "wned-aquaint.csv")
#    loadDataset(test_ACE2004, "wned-ace2004.csv")
#    loadDataset(test_CWEB, "wned-clueweb.csv")
#    loadDataset(test_WIKI, "wned-wikipedia.csv")
