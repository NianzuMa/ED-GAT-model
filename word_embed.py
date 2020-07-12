import os
import logging
import pprint
from gensim.test.utils import get_tmpfile
from gensim.test.utils import datapath, get_tmpfile
from gensim.models import KeyedVectors, Word2Vec
from gensim.scripts.glove2word2vec import glove2word2vec
import numpy as np
import wget

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class GloveEmbed(object):
    """
    Convert glove to gensim format tutorial: https://radimrehurek.com/gensim/scripts/glove2word2vec.html
    """

    def __init__(self, glove_data_folder, download=True):
        self.glove_data_folder = glove_data_folder
        if not os.path.exists(self.glove_data_folder):
            os.makedirs(self.glove_data_folder)
        self.gensim_format_glove_embed_file = None
        if download:
            self._download()
        self._create_file_path_dict()
        self.out_of_vocabulary_vector_dict = {}

    def _download(self):
        links_dict = {"glove.6B.zip": "http://nlp.stanford.edu/data/glove.6B.zip",
                      "glove.42B.300d.zip": "http://nlp.stanford.edu/data/glove.42B.300d.zip",
                      "glove.840B.300d.zip": "http://nlp.stanford.edu/data/glove.840B.300d.zip"}

        for file_name, url in links_dict.items():
            logger.info("Downloading {} ...".format(file_name))
            downloaded_file_path = os.path.join(self.glove_data_folder, file_name)
            if not os.path.exists(downloaded_file_path):
                wget.download(url, downloaded_file_path)
                logger.info("downloaded")

                cwd = os.getcwd()
                os.chdir(self.glove_data_folder)
                logger.info("unzip {}".format(file_name))
                os.system("unzip {}".format(file_name))
                logger.info("unzip done")
                os.chdir(cwd)
        # endfor

    def _create_file_path_dict(self):
        path_dict = {"glove_6B_50d": os.path.join(self.glove_data_folder, "glove.6B.50d.txt"),
                     "glove_6B_100d": os.path.join(self.glove_data_folder, "glove.6B.100d.txt"),
                     "glove_6B_200d": os.path.join(self.glove_data_folder, "glove.6B.200d.txt"),
                     "glove_6B_300d": os.path.join(self.glove_data_folder, "glove.6B.300d.txt"),
                     "glove_42B": os.path.join(self.glove_data_folder, "glove.42B.300d.txt"),
                     "glove_840B": os.path.join(self.glove_data_folder, "glove.840B.300d.txt")}
        self.path_dict = path_dict
        return path_dict

    def _convert_original_glove_embed_to_gensim_format(self, glove_original_file_path, gensim_format_glove_embed_file):

        if os.path.exists(gensim_format_glove_embed_file):
            logger.info("{} is already exists.".format(gensim_format_glove_embed_file))
            return
        else:
            logger.info("Converting {} into gensim format as {}".format(glove_original_file_path,
                                                                        gensim_format_glove_embed_file))

            glove2word2vec(glove_original_file_path, gensim_format_glove_embed_file)
            logging.info("converted")
        # endif

    def load_model(self, glove_embed_name):
        assert glove_embed_name in self.path_dict
        glove_file_path = self.path_dict[glove_embed_name]
        gensim_format_glove_embed_file = glove_file_path.replace(".txt", "_gensim.txt")
        if not os.path.exists(gensim_format_glove_embed_file):
            self._convert_original_glove_embed_to_gensim_format(glove_file_path, gensim_format_glove_embed_file)
        #endif
        logging.info("load model ...")
        model = KeyedVectors.load_word2vec_format(gensim_format_glove_embed_file, unicode_errors='strict')
        logging.info("loaded")
        self.model = model
        return model

    def get_word_vector(self, word, word_embed_size):
        word_vectors = self.model.wv
        if word not in word_vectors:
            logger.info("OOV word : {}".format(word))
            if word in self.out_of_vocabulary_vector_dict:
                return self.out_of_vocabulary_vector_dict[word]
            else:
                random_embed = np.random.rand(word_embed_size)
                self.out_of_vocabulary_vector_dict[word] = random_embed
                return self.out_of_vocabulary_vector_dict[word]
            #endif
        else:
            #logger.info(word_vectors[word])
            #logger.info(type(word_vectors[word]))
            return word_vectors[word]


class FastTextEmbed(object):
    """Deal with it later after glove run"""

    def __init__(self):
        pass


def main():
    glove_embed_handler = GloveEmbed("../data/glove_data", download=False)
    glove_embed_handler.load_model("glove_6B_50d")

    #pprint.pprint(model.most_similar('car', topn=5))
    # pprint.pprint(model.wv['car'])
    #pprint.pprint(model.get_vector('car'))

    glove_embed_handler.get_word_vector("car", 50)

    # word_vectors = model.wv
    #
    # if 'word' in word_vectors.vocab:

    pass


if __name__ == '__main__':
    main()
