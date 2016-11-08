import logging
import os
import modules
from util import readFile, codecsWriteFile, codecsReadFile, codecsDumpJson
import subprocess
import re
from itertools import chain
from evaluation import load_eval_queries
import numpy as np

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
        print self.svmRankLearnPath
        print self.svmTrainingFeatureVectorsFile
        print self.svmRankParamC
        print self.svmRankModelFile

        cmd = [self.svmRankLearnPath,
               "-c",
               self.svmRankParamC,
               self.svmTrainingFeatureVectorsFile,
               self.svmRankModelFile]
        p = subprocess.Popen(cmd, stdout = subprocess.PIPE, stderr = subprocess.STDOUT)
        p.wait()

    def svm_rank(self):
        logger.info("Start SVM Ranking ...")
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

    def train_model(self):
        pass

    def train(self, dataset):

        queries = load_eval_queries(dataset)
        codecsWriteFile(self.svmTrainingFeatureVectorsFile, "")

        vocab = set([])
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
                    #vocab |= fact_candiate.vocab
                    if fact_candiate.sentence_size > sentence_size:
                        sentence_size = fact_candiate.sentence_size
                    #feature_vector = fact_candiate.extract_features()
                    #codecsWriteFile(self.svmTrainingFeatureVectorsFile, str(feature_vector))
            #candidates = self.nomalize_features(candidates)
            for candidate in candidates:
                feature_vector = candidate.feature_vector
                codecsWriteFile(self.svmTrainingFeatureVectorsFile, str(feature_vector), "a")

        #for candidate in candidates:
        #    feature_vector = candidate.extract_features()
        #    codecsWriteFile(self.svmTrainingFeatureVectorsFile, str(feature_vector))

        self.svm_learn()

    def test(self, dataset):
        test_result = self.config_options.get('Test', 'test-result')
        codecsWriteFile(test_result, "")

        queries = load_eval_queries(dataset)
        for query in queries:
            codecsWriteFile(self.svmTestingFeatureVectorsFile, "")
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
                    #feature_vector = fact_candiate.extract_features()
                    candidates.append(fact_candiate)

                    #codecsWriteFile(self.svmTestingFeatureVectorsFile, str(feature_vector), "a")
            #candidates = self.nomalize_features(candidates)

            """
            for candidate in candidates:
                feature_vector = candidate.feature_vector
                codecsWriteFile(self.svmTestingFeatureVectorsFile, str(feature_vector), "a")

            self.svm_rank()

            scores = codecsReadFile(self.svmFactCandidateScores).strip().split("\n")
            idx = np.argmax(scores)
            best_candidate = candidates[idx]
            predictions = list(set(best_candidate.objects))
            """
            answers = set(query.target_result)
            count = 0
            best_predictions = set([])
            best = None
            for candidate in candidates:
                predictions = set(candidate.objects)
                merge = predictions & answers
                best_predictions |= merge
                #if len(merge) > count:
                #    count = len(merge)
                #    best_predictions = list(predictions)
            #if best is None:
            #    predictions = []
            #else:
            #    predictions = list(set(best.objects))
            result_line = "\t".join([query.utterance,
                                     str(query.target_result),
                                     str(list(best_predictions))]) + "\n"
            codecsWriteFile(test_result, result_line, "a")




