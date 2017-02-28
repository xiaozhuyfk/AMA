import logging
from util import (
    codecsReadFile
)

logger = logging.getLogger(__name__)

class WikiUrl(object):

    def __init__(self, config_options):
        self.config_options = config_options
        self.data = config_options.get('Wiki', 'url-path')
        self.wiki_name = {}
        prefix = "http://en.wikipedia.org/wiki/"

        with open(self.data) as f:
            for line in f:
                triple = line.strip().split("\t")
                mid = triple[0]
                url1 = triple[1]
                url2 = triple[2]

                if url1[len(prefix):].startswith("index.html?"):
                    name = url2[len(prefix):]
                else:
                    name = url1[len(prefix):]
                if ("/" in name):
                    name = name.replace("/", "|")
                self.wiki_name[mid] = name
        logger.info("Done loading wiki urls.")



    @staticmethod
    def init_from_config(config_options):
        return WikiExtractor(config_options)



if __name__ == '__main__':
    with open("../test_result/result.txt") as f:
        for line in f:
            print [line]