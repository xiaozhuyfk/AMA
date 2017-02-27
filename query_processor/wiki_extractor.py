import logging
import xml.etree.ElementTree as etree
import nltk.data
import os
import json
import string
import urllib2
import requests
import re
from nltk.tokenize import sent_tokenize

from util import (
    codecsWriteFile,
    codecsReadFile,
    dumpJson,
    loadJson
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
        self.enwiki_dir = config_options.get('Wiki', 'enwiki-dir')

        self.file_pattern = self.enwiki_dir + "enwiki-latest-pages-articles%d.xml"
        self.support_dir = config_options.get('Wiki', 'support-sentence-dir')

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

    def extract_support_sentence(self, idx):
        xml_path = self.file_pattern % idx
        result_path = self.support_dir + str(idx)

        for event, elem in etree.iterparse(
                xml_path,
                events=('start', 'end', 'start-ns', 'end-ns')
            ):
            if (event == 'end' and elem.tag[len(prefix):] == 'page'):
                title = elem.find(prefix + "title")
                revision = elem.find(prefix + "revision")
                text = revision.find(prefix + "text")

                paragraphs = text.text.strip().split("\n")
                sentences = [tokenizer.tokenize(p) for p in paragraphs if p]
                sentences = [s for p in sentences for s in p]

                for object in candidate.objects:
                    d[object] = []
                    for sent in sentences:
                        if subject in sent and object in sent:
                            d[object].append(sent)

    def extract_wiki_page(self, dataset, query, subject):
        logger.info("Extracting wiki from question %d: %s" % (query.id, query.utterance))
        if self.wiki_data_on_disk(dataset, query, subject):
            return self.load_wiki_data_from_disk(dataset, query, subject)
        else:
            if self.abstract is None:
                self.abstract = set([])
                for event, elem in etree.iterparse(
                        self.abstract_xml,
                        events=('start', 'end', 'start-ns', 'end-ns')
                ):
                    if (event == 'end') and (elem.tag == 'title'):
                        self.abstract.add(elem.text[11:].lower())
                logger.info("Load abstract complete.")
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


class WikiAPIExtractor(object):

    def __init__(self, config_options):
        self.config_options = config_options
        self.url = "https://en.wikipedia.org/w/api.php"
        self.wiki = "https://en.wikipedia.org/wiki/index.html"
        self.wiki_dir = config_options.get('Wiki', 'wiki-dir')
        self.tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')

    @staticmethod
    def init_from_config(config_options):
        return WikiAPIExtractor(config_options)

    def store_wiki_data(self, dataset, query, subject, text):
        query_dir = self.wiki_dir + dataset + '/' + str(query.id) + '/'
        if not os.path.exists(query_dir):
            os.makedirs(query_dir)
        path = query_dir + subject
        dumpJson(path, text)

    def wiki_data_on_disk(self, dataset, query, subject):
        query_dir = self.wiki_dir + dataset + '/' + str(query.id) + '/'
        path = query_dir + subject
        return os.path.isfile(path)

    def load_wiki_data_from_disk(self, dataset, query, subject):
        query_dir = self.wiki_dir + dataset + '/' + str(query.id) + '/'
        path = query_dir + subject
        return loadJson(path)

    def extract_wiki_page(self, dataset, query, subject, sid):
        #logger.info("Extracting wiki from question %d: %s" % (query.id, query.utterance))
        if self.wiki_data_on_disk(dataset, query, sid):
            return self.load_wiki_data_from_disk(dataset, query, sid)
        parameter = {
            "action": "query",
            "format": "json",
            "titles": subject,
            "prop": "revisions",
            "rvprop": "content"
        }
        r = requests.get(self.url, params = parameter)

        key, value = r.json()["query"]["pages"].popitem()
        if key == "-1":
            return []

        text = value["revisions"][0]["*"].lower()
        paragraphs = text.strip().split("\n")
        sentences = [self.tokenizer.tokenize(p) for p in paragraphs if p]
        sentences = [s for p in sentences for s in p]
        self.store_wiki_data(dataset, query, sid, sentences)

        return sentences

    def extract_test(self, subject):
        parameter = {
            "action": "query",
            "format": "json",
            "titles": subject,
            "prop": "revisions",
            "rvprop": "content"
        }
        r = requests.get(self.url, params = parameter)

        key, value = r.json()["query"]["pages"].popitem()
        if key == "-1":
            return []

        text = value["revisions"][0]["*"].lower()
        paragraphs = text.strip().split("\n")
        sentences = [self.tokenizer.tokenize(p) for p in paragraphs if p]
        sentences = [s for p in sentences for s in p]
        return sentences




if __name__ == '__main__':
    abstract_xml = "/home/hongyul/AMA/wiki/enwiki/enwiki-latest-pages-articles1.xml"
    prefix = "{http://www.mediawiki.org/xml/export-0.10/}"
    count = 0
    for event, elem in etree.iterparse(abstract_xml, events=('start', 'end', 'start-ns', 'end-ns')):
        if (event == 'end' and elem.tag[len(prefix):] == 'page'):
            title = elem.find(prefix + "title")
            revision = elem.find(prefix + "revision")
            text = revision.find(prefix + "text")

            for ref in text.findall("ref"):
                text.remove(ref)
            print text.text

            """
            ref_start = '&lt;ref&gt;'
            ref_end = '&lt;/ref&gt;'
            iter_start = list(re.finditer(re.compile(ref_start), text))
            iter_end = list(re.finditer(re.compile(ref_end), text))
            print iter_start, iter_end

            replace = []
            assert(len(iter_start) == len(iter_end))
            for i in xrange(len(iter_start)):
                start = iter_start[i].start(0)
                end = iter_end[i].end(0)
                print start, end
                print text[start:end]
                replace.append(text[start:end])

            for rep in replace:
                text.replace(rep, "")
            #print text
            """


            #paragraphs = text.strip().split("\n")
            #sentences = [sent_tokenize(p) for p in paragraphs if p]
            #sentences = [s for p in sentences for s in p]
            #for sent in sentences:
            #    print "-----YOYOY-----", sent
            if title.text == "Anarchism":
                break
    #wiki = WikiAPIExtractor(None)
    #print wiki.extract_test("Barack Obama")

