"""
model about 2 level LeToR
first level:
    ranking model of each group
second level:
    fusion of first level's results

"""

from traitlets.config import Configurable
from traitlets import (
    Unicode,
    List,
    Int,
    Bool,
    Float,
)
import json
import logging
from scholarranking.utils import (
    load_svm_feature,
    group_scores_to_ranking,
    GDEVAL_PATH,
    QREL_PATH,
    seg_gdeval_out,
    load_py_config
)
import numpy as np
from keras.layers import (
    Dense,
    Merge,
    Input,
    Activation,
    Flatten,
)
from keras.models import (
    Model,
    Sequential,
)
from keras.regularizers import l2
from keras.callbacks import EarlyStopping
from scholarranking.letor import pair_docno
import subprocess


class HybridLeToR(Configurable):
    feature_group_in = Unicode(help="the feature group definition json file").tag(config=True)
    feature_name_in = Unicode(help="feature name json file").tag(config=True)
    svm_feature_in = Unicode(help="svm feature in").tag(config=True)
    out_dir = Unicode(help="output directory").tag(config=True)

    l2_w = Float(0, help="weight of l2 regualizer").tag(config=True)
    nb_ltr_layer = Int(1, help='ltr layer').tag(config=True)
    nb_fusion_layer = Int(1, help='fusion layer').tag(config=True)
    nb_epoch = Int(100, help='nb of epoch').tag(config=True)
    q_rel_in = Unicode(QREL_PATH, help="qrel").tag(config=True)
    ltr_conv = Bool(False, help='whether ltr part share same layer').tag(config=True)
    fusion_activation = Unicode('linear', help='fusion layer activation func').tag(config=True)

    def __init__(self, **kwargs):
        super(HybridLeToR, self).__init__(**kwargs)
        self.h_feature_group = {}
        self.l_group_name = []
        self.h_group_dim = {}
        self.ranking_model = None
        self.training_model = None
        self.hash_ltr_model = None
        self._load_data()

    def set_para(self, h_para):
        if 'l2_w' in h_para:
            self.l2_w = h_para['l2_w']
            logging.info('set l2 w = %f', self.l2_w)

    def _load_data(self):
        h_feature_name = json.load(open(self.feature_name_in))
        h_feature_name_group = json.load(open(self.feature_group_in))
        for name, fid in h_feature_name.items():
            if name in h_feature_name_group:
                l_group_name = h_feature_name_group[name]
                if type(l_group_name) != list:
                    l_group_name = [l_group_name]
                self.h_feature_group[fid] = l_group_name
                for group_name in l_group_name:
                    if group_name not in self.h_group_dim:
                        self.h_group_dim[group_name] = 1
                    else:
                        self.h_group_dim[group_name] += 1
        self.l_group_name = sorted(self.h_group_dim.keys())
        logging.info('feature group assignment loaded')

    def train(self, l_svm_data=None):
        if not l_svm_data:
            l_svm_data = load_svm_feature(self.svm_feature_in)
        self._initialize_model()

        h_data, v_label, __, __ = self._pair_wise_construct(l_svm_data)
        logging.info('training on [%d] pairs', v_label.shape[0])
        self.training_model.fit(h_data, v_label,
                                batch_size=v_label.shape[0],
                                nb_epoch=self.nb_epoch,
                                callbacks=[EarlyStopping(monitor='loss', patience=10)]
                                )
        logging.info('training finished')
        return

    def predict(self, l_svm_data=None):
        if not l_svm_data:
            l_svm_data = load_svm_feature(self.svm_feature_in)
        assert self.training_model
        logging.info('start predicting')
        h_data, v_label, l_qid, l_docno = self._point_wise_construct(l_svm_data)
        score = self.ranking_model.predict(h_data)
        l_score = score.reshape(-1).tolist()
        l_q_ranking = group_scores_to_ranking(l_qid, l_docno, l_score)
        logging.info('predicted')
        return l_q_ranking

    def evaluate(self, l_svm_data=None, out_pre=None):
        if not out_pre:
            out_pre = self.out_dir + '/run'
        l_q_ranking = self.predict(l_svm_data)
        dump_trec_ranking_with_score(l_q_ranking, out_pre + '.trec')
        eva_out = subprocess.check_output(['perl', GDEVAL_PATH, QREL_PATH, conf.main.out_name])
        print >> open(out_pre + '.eval', eva_out.strip())
        __, ndcg, err = seg_gdeval_out(eva_out, with_mean=True)
        logging.info('evaluated ndcg:%f, err:%f', ndcg, err)
        return ndcg

    def _segment_feature(self, h_feature):
        """
        segment feature to groups
        feature position in each group is based on its order in h_feature (key is id)
        :param h_feature: fid -> score
        :return: h_group_feature = {group id: [feature values]}
        """
        h_group_feature_v = {}
        l_feature_value = sorted(h_feature.items(), key=lambda item: int(item[0]))
        for fid, score in l_feature_value:
            if fid not in self.h_feature_group:
                continue
            for group in self.h_feature_group[fid]:
                if group not in h_group_feature_v:
                    h_group_feature_v[group] = []
                h_group_feature_v[group].append(score)
        return h_group_feature_v

    def _point_wise_construct(self, l_svm_data):
        """
        read svm data and convert into point wise format (for ranking)
        will read all data together
        returned results will be sorted by qid, and then by docno (for randomness)
        :param svm_in: the svm input
        :return: h_group_feature_mtx (x), v_label (y), l_qid, l_docno
        """
        l_qid = []
        l_docno = []
        h_group_feature_mtx = {}
        l_label = []

        l_svm_data.sort(key=lambda item: (int(item['qid']), item['comment']))

        for svm_data in l_svm_data:
            l_qid.append(svm_data['qid'])
            l_docno.append(svm_data['comment'])
            l_label.append(svm_data['score'])

            h_feature = svm_data['feature']
            h_group_feature_v = self._segment_feature(h_feature)
            for group, feature_v in h_group_feature_v.items():
                if group not in h_group_feature_mtx:
                    h_group_feature_mtx[group] = []
                h_group_feature_mtx[group].append(feature_v)
        for group in h_group_feature_mtx:
            h_group_feature_mtx[group] = np.array(h_group_feature_mtx[group])
        v_label = np.array(l_label)
        logging.info('constructed [%d] point wise data', len(l_docno))
        return h_group_feature_mtx, v_label, l_qid, l_docno

    def _pair_wise_construct(self, l_svm_data):
        """
        generate pair wise data from svm_in
        will generate all of them, not batch for our small dataset
        :param l_svm_data: the svm feature in
        :return: h_paired_group_feature_mtx (with both main and aux_), v_paired_label, labels of pairs
        l_paired_qid = [], l_docno_pair = [(doc_a, doc_b)]
        h_paired_group_feature_mtx will contain:
            group_name: feature mtx
            aux_group_name: feature mtx
            the rows of all matrix will be from a pair of documents from the same qid
        """
        h_group_feature_mtx,v_label, l_qid, l_docno = self._point_wise_construct(l_svm_data)

        v_paired_label, l_paired_qid, l_docno_pair, l_pos_pair = pair_docno(v_label,
                                                                            l_qid,
                                                                            l_docno)

        h_paired_group_mtx = self._form_feature_pair(h_group_feature_mtx, l_pos_pair)
        logging.info('constructed [%d] pairs', len(l_docno_pair))
        return h_paired_group_mtx, v_paired_label, l_paired_qid, l_docno_pair

    @classmethod
    def _form_feature_pair(cls, h_group_feature_mtx, l_pos_pair):
        h_paired_group_mtx = {}
        l_left_p = [item[0] for item in l_pos_pair]
        l_right_p = [item[1] for item in l_pos_pair]
        for group, feature_mtx in h_group_feature_mtx.items():
            h_paired_group_mtx[group] = feature_mtx[l_left_p, :]
            h_paired_group_mtx['aux_' + group] = feature_mtx[l_right_p, :]
        return h_paired_group_mtx

    def _construct_letor_inputs(self, is_aux=False):
        l_inputs = []
        for group_name in self.l_group_name:
            name = group_name
            if is_aux:
                name = 'aux_' + group_name
            f_dim = self.h_group_dim[group_name]
            input_layer = Input(shape=(f_dim,),
                                name=name)
            l_inputs.append(input_layer)
        return l_inputs

    def _construct_letor_layers(self):
        """
        one letor model for each group,
        and one merge layer
        :return:
        """
        l_models = [self._init_per_group_ltr_model(group_name) for group_name in self.l_group_name]
        fusion_model = self._init_fusion_model(len(l_models))
        return l_models, fusion_model

    def _init_fusion_model(self, nb_model):
        fusion_model = Sequential(name='fusion')
        for lvl in xrange(self.nb_fusion_layer):
            if lvl != self.nb_fusion_layer - 1:
                out_dim = nb_model
            else:
                out_dim = 1
            in_dim = nb_model
            this_layer = Dense(output_dim=out_dim,
                               input_shape=(in_dim,),
                               init='glorot_uniform',
                               activation=self.fusion_activation,
                               W_regularizer=l2(self.l2_w),
                               bias=False,
                               name='fusion' + "_%d" % lvl,
                               )
            fusion_model.add(this_layer)
        return fusion_model

    def _init_per_group_ltr_model(self, group_name):
        model_name = 'ltr_' + group_name
        if self.ltr_conv:
            if group_name.startswith('BOEEmb'):
                model_name = 'ltr_BOEEmb'
        ltr_model = Sequential(name=model_name)

        if self.ltr_conv:
            if group_name.startswith('BOEEmb'):
                if not (self.hash_ltr_model is None):
                    return self.hash_ltr_model

        for lvl in xrange(self.nb_ltr_layer):
            if lvl != self.nb_ltr_layer - 1:
                out_dim = self.h_group_dim[group_name]
            else:
                out_dim = 1
            in_dim = self.h_group_dim[group_name]
            this_layer = Dense(output_dim=out_dim,
                               input_shape=(in_dim,),
                               init='glorot_uniform',
                               activation='linear',
                               W_regularizer=l2(self.l2_w),
                               bias=False,
                               name='ltr_' + group_name + "_%d" % lvl,
                               )
            ltr_model.add(this_layer)
        if self.ltr_conv:
            if group_name.startswith('BOEEmb'):
                if self.hash_ltr_model is None:
                    self.hash_ltr_model = ltr_model
        return ltr_model

    @classmethod
    def _align_ranking_network(cls, l_models, fusion_model, l_inputs):
        """
        align to a ranking model
        merge l_models first (concatenate)
        align them with inputs
        add the fusion layer
        output
        :param l_models:
        :param fusion_model:
        :param l_inputs:
        :return:
        """
        l_aligned_models = [model(in_put) for model, in_put in zip(l_models, l_inputs)]
        ranking_fusion = Merge(mode='concat', name='merge')(l_aligned_models)
        ranking_fusion = fusion_model(ranking_fusion)
        ranking_model = Model(input=l_inputs, output=ranking_fusion)
        return ranking_model

    def _build_model(self):
        """
        build training and ranking model
        :return: ranking_model, training_model
        """
        l_models, fusion_model = self._construct_letor_layers()

        l_left_inputs = self._construct_letor_inputs()
        l_right_inputs = self._construct_letor_inputs(is_aux=True)

        left_model = self._align_ranking_network(l_models, fusion_model, l_left_inputs)
        right_model = self._align_ranking_network(l_models, fusion_model, l_right_inputs)
        train_model = Sequential()
        train_model.add(Merge([left_model, right_model],
                              mode=lambda x: x[0] - x[1],
                              output_shape=(1,)
                              )
                        )
        return left_model, train_model

    def _initialize_model(self):
        """
        build and compile training and ranking models
        :return:
        """
        logging.info('initializing model')
        self.ranking_model, self.training_model = self._build_model()
        self.training_model.compile(
            optimizer='rmsprop',
            loss='hinge',
            metric=['accuracy']
        )
        logging.info('ranking model summary')
        self.ranking_model.summary()
        logging.info('training model summary')
        self.training_model.summary()
        logging.info('model initialized')
        return


if __name__ == '__main__':
    import sys
    from scholarranking.utils import (
        set_basic_log,
        load_py_config,
        dump_trec_ranking_with_score,
    )
    import subprocess

    set_basic_log()
    if len(sys.argv) != 2:
        print "unit test model"
        HybridLeToR.class_print_help()
        print "c.main.out_name="
        sys.exit()
    conf = load_py_config(sys.argv[1])
    model = HybridLeToR(config=conf)
    model.train()
    l_q_ranking = model.predict()
    dump_trec_ranking_with_score(l_q_ranking, conf.main.out_name + '.trec')
    eva_out = subprocess.check_output(['perl', GDEVAL_PATH, QREL_PATH, conf.main.out_name + '.trec'])
    print >> open(conf.main.out_name + '.eval', 'w'), eva_out.strip()
    logging.info('res: %s', eva_out.splitlines()[-1])


