from __future__ import print_function

import logging
import globals
import os
import modules
from util import readFile, codecsWriteFile, codecsReadFile, codecsDumpJson, codecsLoadJson
import subprocess
import re
from itertools import chain
from evaluation import load_eval_queries
import numpy as np
from model import bidirectional_lstm_with_embedding, save_model_to_file, load_model

logger = logging.getLogger(__name__)


def tokenize_term(t):
    return re.sub('[?!@#$%^&*,()_+=\'\d\./]', '', t).lower()


class FeatureVector(object):

    def __init__(self, query, relevance, candidate):
        self.query = query
        self.query_id = query.id
        self.candidate = candidate
        self.relevance = relevance
        self.features = {}
        self.format = "%d qid:%s %s# %s\n"

    def add(self, i, value):
        self.features[i] = value

    def get(self, i):
        return self.features.get(i)

    def update(self, i, value):
        self.features[i] = value

    def __iter__(self):
        for i in self.features:
            yield i

    def __str__(self):
        indicator = "" #self.candidate.subject + " " + self.candidate.relation
        vec = ""
        for i in self.features:
            vec += str(i) + ":" + str(self.features[i]) + " "
        return self.format % (self.relevance, str(self.query_id), vec, indicator)


class FactCandidate(object):
    def __init__(self, config_options, query, subject, sid, score, relation, response):
        self.config_options = config_options
        self.query = query
        self.question = query.utterance[:-1]
        self.answers = query.target_result

        self.subject = subject
        self.sid = sid
        self.score = score
        self.relation = relation
        self.response = response

        self.objects = response["objects"]
        self.oid = response["oid"]

        self.query_tokens = [tokenize_term(t) for t in self.question.split()]
        self.subject_tokens = [re.sub('[?!@#$%^&*,()_+=\'/]', '', t).lower() for t in subject.split()]
        relations = re.split("\.\.|\.", self.relation.split("\n")[-1])[-2:] #[re.split("\.\.|\.", r) for r in self.relation.split("\n")]
        self.relation_tokens = [tokenize_term(e) for t in relations for e in re.split("\.\.|\.|_", t)]

        self.relevance = 0
        for object in self.objects:
            if object in self.answers:
                self.relevance = 1
                break


        self.sentence = self.query_tokens + self.subject_tokens + self.relation_tokens
        self.sentence_size = len(self.sentence)

        self.vocab = set(self.sentence)

    def extract_features(self):
        relevance = 0
        for object in self.objects:
            if object in self.answers:
                relevance = 1

        vector = FeatureVector(self.query, relevance, self)

        # Add entity linking score
        vector.add(1, float(self.score))

        # Add number of nodes
        relations = self.relation.split("\n")
        vector.add(2, float(len(relations) + 1))

        # Add number of answers
        vector.add(3, float(len(self.objects)))

        # Add simple similarity score
        question_seq = modules.w2v.transform_seq(self.query_tokens)
        sentence_seq = modules.w2v.transform_seq(self.subject_tokens + self.relation_tokens)
        if question_seq == [] or sentence_seq == []:
            vector.add(4, 0.0)
        else:
            question_embed = sum(question_seq)
            sentence_embed = sum(sentence_seq)
            vector.add(4, float(modules.w2v.embedding_similarity(question_embed, sentence_embed)))

        self.feature_vector = vector

        return vector




class Ranker(object):

    def __init__(self,
                 config_options,
                 svmRankParamC,
                 svmRankLearnPath,
                 svmRankClassifyPath,
                 svmRankModelFile,
                 svmTrainingFeatureVectorsFile,
                 svmTestingFeatureVectorsFile,
                 svmFactCandidateScores):
        self.config_options = config_options
        self.svmRankParamC = svmRankParamC
        self.svmRankLearnPath = svmRankLearnPath
        self.svmRankClassifyPath = svmRankClassifyPath
        self.svmRankModelFile = svmRankModelFile
        self.svmTrainingFeatureVectorsFile = svmTrainingFeatureVectorsFile
        self.svmTestingFeatureVectorsFile = svmTestingFeatureVectorsFile
        self.svmFactCandidateScores = svmFactCandidateScores

    @staticmethod
    def init_from_config(config_options):
        svmRankParamC = config_options.get('SVM', 'paramc')
        svmRankLearnPath = config_options.get('SVM', 'learn-path')
        svmRankClassifyPath = config_options.get('SVM', 'classify-path')
        svmRankModelFile = config_options.get('SVM', 'rank-model-file')
        svmTrainingFeatureVectorsFile = config_options.get('SVM', 'training-vector-file')
        svmTestingFeatureVectorsFile = config_options.get('SVM', 'testing-vector-file')
        svmFactCandidateScores = config_options.get('SVM', 'testing-rank-scores')
        return Ranker(config_options,
                      svmRankParamC,
                      svmRankLearnPath,
                      svmRankClassifyPath,
                      svmRankModelFile,
                      svmTrainingFeatureVectorsFile,
                      svmTestingFeatureVectorsFile,
                      svmFactCandidateScores)

    def svm_learn(self):
        logger.info("Start SVM Training ...")
        cmd = [self.svmRankLearnPath,
               "-c",
               self.svmRankParamC,
               self.svmTrainingFeatureVectorsFile,
               self.svmRankModelFile]
        p = subprocess.Popen(cmd, stdout = subprocess.PIPE, stderr = subprocess.STDOUT)
        p.wait()

    def svm_rank(self):
        #logger.info("Start SVM Ranking ...")
        cmd = [self.svmRankClassifyPath,
               self.svmTestingFeatureVectorsFile,
               self.svmRankModelFile,
               self.svmFactCandidateScores]
        p = subprocess.Popen(cmd, stdout = subprocess.PIPE, stderr = subprocess.STDOUT)
        p.wait()

    def nomalize_features(self, candidates):
        minimums = np.array([float("inf")] * 4)
        maximums = np.array([-float("inf")] * 4)
        for candidate in candidates:
            vec = candidate.feature_vector
            for i in vec:
                if vec.get(i) < minimums[i-1]:
                    minimums[i-1] = vec.get(i)
                if vec.get(i) > maximums[i-1]:
                    maximums[i-1] = vec.get(i)

            for i in vec:
                new = (vec.get(i) - minimums[i-1]) / (maximums[i-1] - minimums[i-1])
                vec.update(i, new)
            candidate.feature_vector = vec
        return candidates

    def extract_fact_candidates(self, dataset):
        queries = load_eval_queries(dataset)
        vocab = set([])
        sentence_size = 0
        candidates = []
        for query in queries:
            print("Processing query " + str(query.id) + " " * 10 + "\r", end="")
            json = modules.extractor.extract_fact_list_with_entity_linker("webquestionstrain", query)
            facts = json["facts"]
            for ie in facts:
                subject = ie["subject"]
                sid = ie["sid"]
                score = ie["score"]
                relations = ie["relations"]
                for rel in relations:
                    fact_candiate = FactCandidate(self.config_options,
                                                  query,
                                                  subject,
                                                  sid,
                                                  score,
                                                  rel,
                                                  relations[rel])
                    candidates.append(fact_candiate)
                    vocab |= fact_candiate.vocab
                    if fact_candiate.sentence_size > sentence_size:
                        sentence_size = fact_candiate.sentence_size
        return candidates, vocab, sentence_size

    def train_model(self):
        config_options = globals.config
        train_candidates, train_vocab, train_sentence_size = \
            self.extract_fact_candidates("webquestionstrain")
        #test_candidates, test_vocab, test_sentence_size = \
        #    self.extract_fact_candidates("webquestionstest")
        print("")

        vocab_path = config_options.get("Train", "vocab")
        vocab = codecsLoadJson(vocab_path)

        sentence_size = 28
        word_idx = dict((c, i + 1) for i, c in enumerate(vocab))
        n_symbols = len(vocab) + 1

        # build models
        lstm_model = bidirectional_lstm_with_embedding(64, n_symbols, word_idx)

        # pre-process training data
        X = []
        Y = []
        for candidate in train_candidates:
            sentence = candidate.sentence
            sentence_idx = [word_idx[t] for t in sentence] + \
                           (sentence_size - len(sentence)) * [0]
            X.append(sentence_idx)
            Y.append(candidate.relevance)

        # training procedure
        lstm_model.fit(np.array(X),
                       np.array(Y),
                       validation_split=0.1,
                       nb_epoch=3,
                       batch_size=64)

        model_struct = config_options.get('Train', 'model-struct')
        model_weights = config_options.get('Train', 'model-weights')
        save_model_to_file(lstm_model, model_struct, model_weights)


    def train(self, dataset):

        queries = load_eval_queries(dataset)
        codecsWriteFile(self.svmTrainingFeatureVectorsFile, "")

        sentence_size = 0
        for query in queries:
            logger.info("Processing query " + str(query.id))

            candidates = []
            json = modules.extractor.extract_fact_list_with_entity_linker(dataset, query)
            facts = json["facts"]
            for ie in facts:
                subject = ie["subject"]
                sid = ie["sid"]
                score = ie["score"]
                relations = ie["relations"]
                for rel in relations:
                    fact_candiate = FactCandidate(self.config_options,
                                                  query,
                                                  subject,
                                                  sid,
                                                  score,
                                                  rel,
                                                  relations[rel])
                    fact_candiate.extract_features()
                    candidates.append(fact_candiate)
                    if fact_candiate.sentence_size > sentence_size:
                        sentence_size = fact_candiate.sentence_size
            #candidates = self.nomalize_features(candidates)
            for candidate in candidates:
                feature_vector = candidate.feature_vector
                codecsWriteFile(self.svmTrainingFeatureVectorsFile, str(feature_vector), "a")
        self.svm_learn()

    def test(self, dataset):
        test_result = self.config_options.get('Test', 'test-result')
        codecsWriteFile(test_result, "")

        cover = 0
        lost = 0
        queries = load_eval_queries(dataset)
        for query in queries:
            codecsWriteFile(self.svmTestingFeatureVectorsFile, "")
            candidates = []

            json = modules.extractor.extract_fact_list_with_entity_linker(dataset, query)
            facts = json["facts"]
            if facts == []:
                lost += 1
                result_line = "\t".join([query.utterance,
                                     str(query.target_result),
                                     str([])]) + "\n"
                codecsWriteFile(test_result, result_line, "a")
                continue

            for ie in facts:
                subject = ie["subject"]
                sid = ie["sid"]
                score = ie["score"]
                relations = ie["relations"]
                for rel in relations:
                    fact_candiate = FactCandidate(self.config_options,
                                                  query,
                                                  subject,
                                                  sid,
                                                  score,
                                                  rel,
                                                  relations[rel])
                    #feature_vector = fact_candiate.extract_features()
                    candidates.append(fact_candiate)

            #candidates = self.nomalize_features(candidates)

            #for candidate in candidates:
            #    feature_vector = candidate.feature_vector
            #    codecsWriteFile(self.svmTestingFeatureVectorsFile, str(feature_vector), "a")

            #self.svm_rank()

            """
            answers = set(query.target_result)
            scores = codecsReadFile(self.svmFactCandidateScores).strip().split("\n")
            idx = np.argmax(scores)
            best_candidate = candidates[idx]
            predictions = set(best_candidate.objects)
            """

            count = 0
            answers = set(query.target_result)
            best_predictions = set([])
            best = None
            for candidate in candidates:
                predictions = set(candidate.objects)
                merge = predictions & answers
                #best_predictions |= merge
                if len(merge) > count:
                    count = len(merge)
                    best_predictions = predictions
                    best = candidate
            if best is None:
                result_line = "\t".join([query.utterance,
                                         str(query.target_result)]) + "\n"
                codecsWriteFile(test_result, result_line, "a")
                #best_predictions = []
                #result_line = "\t".join([query.utterance,
                #                     str(query.target_result),
                #                     str(list(best_predictions))]) + "\n"
            else:
                result_line = "\t".join([query.utterance,
                                         str(query.target_result),
                                         best.relation]) + "\n"
                codecsWriteFile(test_result, result_line, 'a')
                #best_predictions = list(set(best.objects))
                #cover += 1
                #result_line = "\t".join([query.utterance,
                #                         str(query.target_result),
                #                         str(list(best_predictions)),
                #                         best.relation]) + "\n"

            #result_line = "\t".join([query.utterance,
            #                         str(query.target_result),
            #                         str(list(predictions))]) + "\n"
            #codecsWriteFile(test_result, result_line, "a")
        #logger.info("Number of questions covered: %d", cover)
        #logger.info("Number of losses: %d", lost)




