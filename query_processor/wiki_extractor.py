import logging
import xml.etree.ElementTree as etree
import nltk.data
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
        self.wiki_dir = config_options.get('Wiki', 'wiki-dir')

    @staticmethod
    def init_from_config(config_options):
        return WikiExtractor(config_options)

    def wiki_data_on_disk(self, dataset, query, subject):
        query_dir = self.wiki_dir + dataset + '/' + str(query.id) + '/'
        path = query_dir + subject
        return os.path.isfile(path)

    def load_wiki_data_from_disk(self, dataset, query, subject):
        query_dir = self.wiki_dir + dataset + '/' + str(query.id) + '/'
        path = query_dir + subject
        return codecsReadFile(path)

    def store_wiki_data(self, dataset, query, subject, text):
        query_dir = self.wiki_dir + dataset + '/' + str(query.id) + '/'
        if not os.path.exists(query_dir):
            os.makedirs(query_dir)
        path = query_dir + subject
        codecsWriteFile(path, text)

    def extract_wiki_page(self, dataset, query, subject):
        logger.info("Extracting wiki from question %d: %s" % (query.id, query.utterance))
        if self.wiki_data_on_disk(dataset, query, subject):
            return self.load_wiki_data_from_disk(dataset, query)
        else:
            if self.abstract is None:
                self.abstract = set([])
                for event, elem in etree.iterparse(
                        self.abstract_xml,
                        events=('start', 'end', 'start-ns', 'end-ns')
                ):
                    if (event == 'end') and (elem.tag == 'title'):
                        self.abstract.add(elem.text[11:].lower())
            d = {}
            prefix = "{http://www.mediawiki.org/xml/export-0.10/}"
            if subject in self.abstract:
                for event, elem in etree.iterparse(
                    self.data_xml,
                    events=('start', 'end', 'start-ns', 'end-ns')
                ):
                    if (event == 'end' and elem.tag[len(prefix):] == 'page'):
                        title = elem.find(prefix + "title")
                        revision = elem.find(prefix + "revision")
                        text = revision.find(prefix + "text")

                        if title.text.lower() != subject:
                            continue

                        self.store_wiki_data(dataset, query, subject, text)
                        return text

                        """
                        paragraphs = text.text.strip().split("\n")
                        sentences = [tokenizer.tokenize(p) for p in paragraphs if p]
                        sentences = [s for p in sentences for s in p]

                        for object in candidate.objects:
                            d[object] = []
                            for sent in sentences:
                                if subject in sent and object in sent:
                                    d[object].append(sent)
                        """









if __name__ == '__main__':
    abstract_xml = "/home/hongyul/AMA/wiki/enwiki/enwiki-latest-pages-articles.xml"
    prefix = "{http://www.mediawiki.org/xml/export-0.10/}"
    count = 0
    tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
    for event, elem in etree.iterparse(abstract_xml, events=('start', 'end', 'start-ns', 'end-ns')):
        if (event == 'end' and elem.tag[len(prefix):] == 'page'):
            title = elem.find(prefix + "title")
            revision = elem.find(prefix + "revision")
            text = revision.find(prefix + "text")
            paragraphs = text.text.strip().split("\n")
            sentences = [tokenizer.tokenize(p) for p in paragraphs if p]
            sentences = [s for p in sentences for s in p]
            for sent in sentences:
                print "-----YOYOY-----", sent
            if title.text == "Anachism":
                break
