import logging

logger = logging.getLogger(__name__)

class FactExtractor(object):

    def __init__(self, fact_list_dir):
        self.fact_list_dir = fact_list_dir

    @staticmethod
    def init_from_args(self, args):
        fact_list_dir = args.config.get('FactList',
                                        args.dataset)
        print fact_list_dir
        return FactExtractor(fact_list_dir)

