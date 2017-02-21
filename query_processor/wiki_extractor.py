import logging
import xml.etree.ElementTree as etree
import xmltodict
import os
from util import (
    codecsWriteFile,
    codecsReadFile,
)

logger = logging.getLogger(__name__)

class WikiExtractor(object):

    def __init__(self, config_options):
        self.config_options = config_options
        self.data = None
        self.abstract = None
        self.data_xml = config_options.get('Wiki', 'data')
        self.abstract_xml = config_options.get('Wiki', 'abstract')

    @staticmethod
    def init_from_config(config_options):
        return WikiExtractor(config_options)

    def wiki_data_on_disk(self, dataset, query):
        return False

    def load_wiki_data_from_disk(self, dataset, query):
        pass

    def extract_support_sentences(self, dataset, query):
        if self.wiki_data_on_disk(dataset, query):
            return self.load_wiki_data_from_disk(dataset, query)
        else:
            if self.abstract is None:
                self.xml = etree.parse(self.data)


if __name__ == '__main__':
    abstract_xml = "/home/hongyul/AMA/wiki/enwiki/enwiki-latest-abstract.xml"
    abstract = etree.parse(abstract_xml)

