"""
use bert pretrained model to generate token embeddings, which is the input for graph attention network
"""

from __future__ import absolute_import, division, print_function

import csv
import torch
import logging
import os
import sys
from io import open
import json
import matplotlib.pyplot as plt
import pickle
import numpy as np
from collections import Counter, defaultdict
from torch.utils.data import (DataLoader, RandomSampler, SequentialSampler,
                              TensorDataset)

# logging
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')


class Comp_Relation_C_for_GAT_Example():
    def __init__(self,
                 instance_id,
                 masked_sent_text,  # it is masked object sent
                 original_sent_text,
                 object_a,
                 object_a_index,
                 object_b,
                 object_b_index,
                 label,
                 undirected_graph_edges):
        self.instance_id = instance_id
        self.masked_sent_text = masked_sent_text
        self.original_sent_text = original_sent_text
        self.object_a = object_a
        self.object_a_index = object_a_index  # index for masked object sent text
        self.object_b = object_b
        self.object_b_index = object_b_index  # index for masked object sent text
        self.label = label
        self.undirected_graph_edges = undirected_graph_edges


class Comp_Relation_C_for_GAT_Subgraph_Feature():
    """
    instance_id=instance_id,
      word_embed_matrix=word_embed_matrix,
      target_mask_for_object_a=target_mask_for_object_a,
      target_mask_for_object_b=target_mask_for_object_b,
      graph_label=graph_label,
      edge_index=edge_index)
    """

    def __init__(self,
                 instance_id,
                 word_embed_matrix,
                 masked_sent_text,
                 object_a_pos,
                 object_b_pos,
                 label_id,
                 edge_index):
        self.instance_id = instance_id
        self.word_embed_matrix = word_embed_matrix
        self.masked_sent_text = masked_sent_text
        self.object_a_pos = object_a_pos
        self.object_b_pos = object_b_pos
        self.label_id = label_id
        self.edge_index = edge_index


class Comp_Relation_C_Processor_For_Graph_Attention_Network(object):
    """
    feature creation choice:
    ["glove",
    "bert_model",
    "bert_original_sent_fine_tune",
    "bert_masked_object_sent_fine_tune"]
    """

    def __init__(self, args):

        if args.feature_creation == "glove":
            pass

        if args.feature_creation in ["bert_original_sent_fine_tune", "bert_masked_object_sent_fine_tune"]:
            self.pretrained_bert_model_folder = args.pretrained_bert_model_folder_for_feature_creation
            self.global_vocab = None

    def _load_gensim_model(self):
        from word_embed import GloveEmbed
        glove_embed_handler = GloveEmbed("../data/glove_data", download=False)
        glove_embed_handler.load_model("glove_840B")
        self.glove_embed_handler = glove_embed_handler

    def get_labels(self):
        return ['BETTER', 'WORSE', 'NONE']

    def _label_distribution(self, examples):
        label_list = []
        for exp in examples:
            label_list.append(exp.label)
        # endfor
        label_dict = Counter(label_list)

        keys = label_dict.keys()
        keys = sorted(keys)
        for k in keys:
            logger.info("{}: {}".format(k, label_dict[k]))
        # endfor

    def _load_from_json(self, path_to_file):
        """
        2019-08-05 new label dataset with context information
        :param path_to_file:
        :param is_test:
        :param masking_mode:
        :return:
        """
        examples = []
        with open(path_to_file, mode="r") as fin:
            for line in fin:
                line = line.strip()
                info = json.loads(line)

                instance_id = info["instance_id"]
                masked_sent_text = info["masked_sent_text"].strip()
                original_sent_text = info["original_sent_text"].strip()
                label = info['label']
                object_a = info["object_a"]
                object_a_index = info['object_a_index']
                object_b = info['object_b']
                object_b_index = info['object_b_index']

                undirected_graph_edges = info["undirected_graph_edges"]

                example = Comp_Relation_C_for_GAT_Example(instance_id=instance_id,
                                                          masked_sent_text=masked_sent_text,
                                                          original_sent_text=original_sent_text,
                                                          object_a=object_a,
                                                          object_a_index=object_a_index,
                                                          object_b=object_b,
                                                          object_b_index=object_b_index,
                                                          label=label,
                                                          undirected_graph_edges=undirected_graph_edges)
                examples.append(example)
            # endfor
        # endwith
        return examples

    def get_train_examples(self, data_dir):
        examples = self._load_from_json(os.path.join(data_dir, "train.json"))
        logger.info("-" * 8 + " train " + '-' * 8)
        self._label_distribution(examples)
        return examples

    def get_dev_examples(self, data_dir):
        examples = self._load_from_json(os.path.join(data_dir, "dev.json"))
        logger.info("-" * 8 + " dev " + '-' * 8)
        self._label_distribution(examples)
        return examples

    def get_test_examples(self, data_dir):
        examples = self._load_from_json(os.path.join(data_dir, "test.json"))
        logger.info("-" * 8 + " test " + '-' * 8)
        self._label_distribution(examples)
        return examples

    def draw_x_y_graph(self, x_list, y_list, x_label, y_label, title):
        # plotting the points
        plt.plot(x_list, y_list)
        # naming the x axis
        plt.xlabel(x_label)
        # naming the y axis
        plt.ylabel(y_label)
        # giving a title to my graph
        plt.title(title)
        # function to show the plot
        plt.show()
        pass

    def token_length_analysis(self, data_dir, tokenizer, do_lower_case):
        """
        output the distribution of token length after tokenized by bert in train + dev data
        :return:
        """
        train_examples = self.get_train_examples(data_dir)
        dev_examples = self.get_dev_examples(data_dir)
        total_examples = []
        total_examples.extend(train_examples)
        total_examples.extend(dev_examples)

        token_length_dict = defaultdict(int)

        for (ex_index, example) in enumerate(total_examples):
            if ex_index % 10000 == 0:
                logger.info("processing example %d of %d" % (ex_index, len(total_examples)))

            tokens = tokenizer.tokenize(example.masked_sent_text)
            token_length = len(tokens)
            token_length_dict[token_length] += 1
        # endfor

        k_list = []
        v_list = []
        for k, v in sorted(token_length_dict.items(), key=lambda x: x[0]):
            k_list.append(k)
            v_list.append(v)
        # endfor

        self.draw_x_y_graph(k_list, v_list, "length", "nums",
                            "length-nums(do_lower_case={})".format(str(do_lower_case)))
        logger.info("------ max tokenized token length: {}--------".format(k_list[-1]))
        pass

    def _load_pretrained_bert_model(self, args, model_class, tokenizer_class):
        """
        1. load previously pretrained bert model using [CLS] sentence tokens [SEP] aspect [SEP] (note that some other auxiliary sentence technique might produce better result)
        2. save the model to evaluation mode
        :return:
        """
        model = model_class.from_pretrained(self.pretrained_bert_model_folder)
        tokenizer = tokenizer_class.from_pretrained(self.pretrained_bert_model_folder)
        model.eval()  # set model to evaluation mode
        model.to(args.device)
        return model, tokenizer

    def get_glove_embed(self, examples, args):
        # load glove_embed_model
        glove_embeds_for_text_tokens_dict = {}

        for (ex_index, example) in enumerate(examples):
            instance_id = example.instance_id

            if ex_index % 100 == 0:
                logger.info("created bert embed for {} / {}".format(ex_index, len(examples)))
            # endif

            masked_sent = example.masked_sent_text
            masked_sent_tokens = masked_sent.strip().split()

            object_a_pos = masked_sent_tokens.index("#objecta")
            object_b_pos = masked_sent_tokens.index("#objectb")

            object_a_text = example.object_a
            object_b_text = example.object_b

            if object_a_pos == 0:
                input_tokens = object_a_text.split() \
                               + masked_sent_tokens[(object_a_pos + 1): object_b_pos] \
                               + object_b_text.split() \
                               + masked_sent_tokens[(object_b_pos + 1):]
            else:
                input_tokens = masked_sent_tokens[:object_a_pos] \
                               + object_a_text.split() \
                               + masked_sent_tokens[(object_a_pos + 1): object_b_pos] \
                               + object_b_text.split() \
                               + masked_sent_tokens[(object_b_pos + 1):]
            # endif

            assert input_tokens == example.original_sent_text.split()

            text_token_embeds = []
            for i, w in enumerate(input_tokens):
                w = w.lower()
                text_token_embeds.append(self.glove_embed_handler.get_word_vector(w, args.input_size))
            # endfor

            all_tokens_tensor = np.stack(text_token_embeds)

            object_a_length = len(object_a_text.split())
            object_b_length = len(object_b_text.split())

            # In original token (not masked token), the object_a_pos, object_b_pos should be corrected
            # using object_a_length, object_b_length
            original_object_a_pos_begin = object_a_pos
            original_object_b_pos_begin = object_b_pos + (object_a_length - 1)  # every thing will be shift to right

            total_sent_bert_embed = None
            if object_a_pos == 0:
                object_a_embed = all_tokens_tensor[
                                 original_object_a_pos_begin: (original_object_a_pos_begin + object_a_length)]
                between_object_a_and_b = all_tokens_tensor[
                                         (original_object_a_pos_begin + object_a_length): original_object_b_pos_begin]
                object_b_embed = all_tokens_tensor[
                                 original_object_b_pos_begin: (original_object_b_pos_begin + object_b_length)]
                after_object_b_embed = all_tokens_tensor[(original_object_b_pos_begin + object_b_length):]

                total_sent_bert_embed = np.concatenate([object_a_embed.mean(0, keepdims=True),
                                                        between_object_a_and_b,
                                                        object_b_embed.mean(0, keepdims=True),
                                                        after_object_b_embed])
            else:
                before_object_a = all_tokens_tensor[0:original_object_a_pos_begin]
                object_a_embed = all_tokens_tensor[
                                 original_object_a_pos_begin: (original_object_a_pos_begin + object_a_length)]
                between_object_a_and_b = all_tokens_tensor[
                                         (original_object_a_pos_begin + object_a_length): original_object_b_pos_begin]
                object_b_embed = all_tokens_tensor[
                                 original_object_b_pos_begin: (original_object_b_pos_begin + object_b_length)]
                after_object_b_embed = all_tokens_tensor[(original_object_b_pos_begin + object_b_length):]

                total_sent_bert_embed = np.concatenate([before_object_a,
                                                        object_a_embed.mean(0, keepdims=True),
                                                        between_object_a_and_b,
                                                        object_b_embed.mean(0, keepdims=True),
                                                        after_object_b_embed])
            # endif

            assert total_sent_bert_embed.shape[0] == len(masked_sent_tokens)

            glove_embeds_for_text_tokens_dict[instance_id] = total_sent_bert_embed
        # endfor

        logger.info("Total OOV: {}".format(len(self.glove_embed_handler.out_of_vocabulary_vector_dict)))
        logger.info(self.glove_embed_handler.out_of_vocabulary_vector_dict)
        with open(os.path.join(args.output_dir, "OOV.txt"), mode="a") as fout:
            for k, v in self.glove_embed_handler.out_of_vocabulary_vector_dict.items():
                fout.write(k + '\n')
            # endfor

        return glove_embeds_for_text_tokens_dict

    def get_bert_embed_from_original_bert_using_original_text(self, examples, args, model_class, tokenizer_class):
        """
        Input the original text into bert model, without mask two objects.
        :return:
        """
        from transformers import BertModel, BertTokenizer

        logger.info("loading bert model: {} to create feature for sent tokens".format(
            args.pretrained_transformer_model_name_or_path))
        model = BertModel.from_pretrained(args.pretrained_transformer_model_name_or_path)
        tokenizer = BertTokenizer.from_pretrained(args.pretrained_transformer_model_name_or_path)

        model.eval()
        model.to(args.device)

        bert_embeds_for_text_tokens_dict = {}

        for (ex_index, example) in enumerate(examples):
            instance_id = example.instance_id

            if ex_index % 100 == 0:
                logger.info("created bert embed for {} / {}".format(ex_index, len(examples)))
            # endif

            masked_sent = example.masked_sent_text
            masked_sent_tokens = masked_sent.strip().split()

            object_a_pos = masked_sent_tokens.index("#objecta")
            object_b_pos = masked_sent_tokens.index("#objectb")

            object_a_text = example.object_a
            object_b_text = example.object_b

            if object_a_pos == 0:
                input_tokens = object_a_text.split() \
                               + masked_sent_tokens[(object_a_pos + 1): object_b_pos] \
                               + object_b_text.split() \
                               + masked_sent_tokens[(object_b_pos + 1):]
            else:
                input_tokens = masked_sent_tokens[:object_a_pos] \
                               + object_a_text.split() \
                               + masked_sent_tokens[(object_a_pos + 1): object_b_pos] \
                               + object_b_text.split() \
                               + masked_sent_tokens[(object_b_pos + 1):]
            # endif

            assert input_tokens == example.original_sent_text.split()

            word_pieces_list = []
            word_boundaries_list = []

            for w in input_tokens:
                word_pieces = tokenizer.tokenize(w)
                word_boundaries_list.append([len(word_pieces_list), len(word_pieces_list) + len(word_pieces)])
                word_pieces_list += word_pieces
            # endfor
            assert len(word_boundaries_list) == len(input_tokens)

            object_a_pieces = tokenizer.tokenize(object_a_text)
            object_b_pieces = tokenizer.tokenize(object_b_text)

            total_input_tokens = ['[CLS]'] + word_pieces_list + ['[SEP]']
            input_ids = tokenizer.convert_tokens_to_ids(total_input_tokens)
            segment_ids = [0] * len(total_input_tokens)

            input_ids = torch.tensor([input_ids], dtype=torch.long).to(args.device)
            segment_ids = torch.tensor([segment_ids], dtype=torch.long).to(args.device)

            with torch.no_grad():
                sequence_output, _ = model(input_ids=input_ids,
                                           token_type_ids=segment_ids)
            # endwith

            sequence_output = sequence_output.squeeze(dim=0)
            text_piece_embeds = sequence_output[1:(1 + len(word_pieces_list))].to(
                'cpu').numpy()  # first token is [CLS] should be excluded

            text_token_embeds = []
            for i, w in enumerate(input_tokens):
                text_token_embeds.append(
                    text_piece_embeds[word_boundaries_list[i][0]: word_boundaries_list[i][1]].mean(
                        0))  # take average for piece into token embed
            # endfor

            all_tokens_tensor = np.stack(text_token_embeds)

            object_a_length = len(object_a_text.split())
            object_b_length = len(object_b_text.split())

            # In original token (not masked token), the object_a_pos, object_b_pos should be corrected
            # using object_a_length, object_b_length
            original_object_a_pos_begin = object_a_pos
            original_object_b_pos_begin = object_b_pos + (object_a_length - 1)  # every thing will be shift to right

            total_sent_bert_embed = None
            if object_a_pos == 0:
                object_a_embed = all_tokens_tensor[
                                 original_object_a_pos_begin: (original_object_a_pos_begin + object_a_length)]
                between_object_a_and_b = all_tokens_tensor[
                                         (original_object_a_pos_begin + object_a_length): original_object_b_pos_begin]
                object_b_embed = all_tokens_tensor[
                                 original_object_b_pos_begin: (original_object_b_pos_begin + object_b_length)]
                after_object_b_embed = all_tokens_tensor[(original_object_b_pos_begin + object_b_length):]

                total_sent_bert_embed = np.concatenate([object_a_embed.mean(0, keepdims=True),
                                                        between_object_a_and_b,
                                                        object_b_embed.mean(0, keepdims=True),
                                                        after_object_b_embed])
            else:
                before_object_a = all_tokens_tensor[0:original_object_a_pos_begin]
                object_a_embed = all_tokens_tensor[
                                 original_object_a_pos_begin: (original_object_a_pos_begin + object_a_length)]
                between_object_a_and_b = all_tokens_tensor[
                                         (original_object_a_pos_begin + object_a_length): original_object_b_pos_begin]
                object_b_embed = all_tokens_tensor[
                                 original_object_b_pos_begin: (original_object_b_pos_begin + object_b_length)]
                after_object_b_embed = all_tokens_tensor[(original_object_b_pos_begin + object_b_length):]

                total_sent_bert_embed = np.concatenate([before_object_a,
                                                        object_a_embed.mean(0, keepdims=True),
                                                        between_object_a_and_b,
                                                        object_b_embed.mean(0, keepdims=True),
                                                        after_object_b_embed])
            # endif

            assert total_sent_bert_embed.shape[0] == len(masked_sent_tokens)

            bert_embeds_for_text_tokens_dict[example.instance_id] = total_sent_bert_embed
        # endfor
        return bert_embeds_for_text_tokens_dict

    def get_batch(self, feature_list):
        """
        This get batch is specifically for input in the format of subgraph
        Each subgraph is a sentence, and each subgraph is disconnected
        :return:
        """
        graph_edge_list = []

        word_embed_matrix_list = []
        input_token_size_list = []
        target_mask_list = []
        label_id_list = []
        for (i, feature) in enumerate(feature_list):
            target_a_pos = feature.object_a_pos
            target_b_pos = feature.object_b_pos
            masked_sent_text = feature.masked_sent_text
            masked_sent_tokens = masked_sent_text.strip().split()
            assert masked_sent_tokens[target_a_pos] == "#objecta"
            assert masked_sent_tokens[target_b_pos] == "#objectb"
            masked_token_size = len(masked_sent_tokens)
            word_embed_matrix = feature.word_embed_matrix
            assert word_embed_matrix.shape[0] == masked_token_size

            graph_edge = feature.edge_index
            label_id = feature.label_id

            target_mask = [False] * masked_token_size
            target_mask[target_a_pos] = True
            target_mask[target_b_pos] = True

            word_embed_matrix_list.append(word_embed_matrix)
            graph_edge_list.append(np.array(graph_edge) + sum(input_token_size_list))
            target_mask_list.append([False] * sum(input_token_size_list) + target_mask)
            input_token_size_list.append(masked_token_size)
            label_id_list.append(label_id)
        # endfor

        word_embed_matrix_list = np.concatenate(word_embed_matrix_list, 0)
        graph_edge_list = np.concatenate(graph_edge_list, 0)

        # target_mask_list padding
        # the last element of target_mask_list should have the total token information
        assert sum(input_token_size_list) == len(target_mask_list[-1])
        assert len(input_token_size_list) == len(feature_list)
        total_tokens_in_batch = sum(input_token_size_list)

        # for each target mask, it could select the two embeds for two target
        new_target_mask_list = []
        for target_mask_item in target_mask_list:
            padding_mask = [False] * (total_tokens_in_batch - len(target_mask_item))
            new_target_mask_item = target_mask_item + padding_mask
            new_target_mask_list.append(new_target_mask_item)
        # endfor

        return word_embed_matrix_list, graph_edge_list, new_target_mask_list, label_id_list, input_token_size_list

    def convert_examples_to_features_from_masked_sent_subgraphs(self,
                                                                args,
                                                                model_class,
                                                                tokenizer_class,
                                                                examples,
                                                                max_length=512,
                                                                label_list=None,
                                                                output_mode=None,
                                                                mode=None,  # train/dev/test
                                                                # -----------------------
                                                                pad_on_left=False,
                                                                pad_token=0,
                                                                pad_token_segment_id=0,
                                                                mask_padding_with_zero=True,
                                                                # -------- some extra parameter -------
                                                                cls_token_at_end=False,
                                                                cls_token='[CLS]',
                                                                cls_token_segment_id=0,
                                                                sep_token='[SEP]',
                                                                sep_token_extra=False,
                                                                sequence_a_segment_id=0,
                                                                sequence_b_segment_id=1,
                                                                ):
        """
        different from method "convert_examples_to_features_from_masked_sent_customized" that
        1) treat each sentence as a single independent graph
        2) make the word embedding static, the graph model cannot fine tune it
        3) there is no global word vocabulary, each sentence only have its own small word vocabulary

        This method will
        1) Based on batch size, concatenate each sentence into a big graph, formed by each disconnected subgraph,
        each graph is formed by one sentence
        2) word embedding could be fine tuned during training
        3) there is a global word vocabulary, so that information could be shared between sentences

          args=args,
          model_class=model_class,
          tokenizer_class=tokenizer_class,
          examples=examples,
          max_length=args.max_seq_length,
          label_list=label_list,
          output_mode=output_mode,
          mode=mode)

        The masked object sent without any query give good result.

        1. First, load the pretrained bert model (trained with the mask object sent, to generate token embedding)
        2. feed the embedding into GAT and generate result

        Loads a data file into a list of `InputBatch`s
            `cls_token_at_end` define the location of the CLS token:
                - False (Default, BERT/XLM pattern): [CLS] + A + [SEP] + B + [SEP]
                - True (XLNet/GPT pattern): A + [SEP] + B + [SEP] + [CLS]
            `cls_token_segment_id` define the segment id associated to the CLS token (0 for BERT, 2 for XLNet)
        """

        label_to_i_map = {label: i for i, label in enumerate(label_list)}
        # label list is always the same order, so do not need to save

        if mode == "train":
            with open(os.path.join(args.output_dir, "label_to_i_map.json"), mode="w") as fout:
                json.dump(label_to_i_map, fout)
        # endif

        bert_example_embeds_dict = None
        feature_creation = args.feature_creation
        assert feature_creation in ["glove",
                                    "original_bert_model_using_original_text",
                                    ]

        if feature_creation == "glove":
            bert_example_embeds_dict = self.get_glove_embed(examples, args)

        if feature_creation == "original_bert_model_using_original_text":
            bert_example_embeds_dict = self.get_bert_embed_from_original_bert_using_original_text(examples, args,
                                                                                                  model_class,
                                                                                                  tokenizer_class)

        assert feature_creation is not None

        feature_list = []
        for (ex_index, example) in enumerate(examples):
            instance_id = example.instance_id

            label_id = label_to_i_map[example.label]

            if ex_index % 100 == 0:
                logger.info("Writing example %d of %d" % (ex_index, len(examples)))
            # endif

            # get object position
            masked_sent_text = example.masked_sent_text
            masked_sent_tokens = masked_sent_text.strip().split()

            object_a_pos = masked_sent_tokens.index("#objecta")
            object_b_pos = masked_sent_tokens.index("#objectb")

            # get edge feature
            graph_edge_info = example.undirected_graph_edges

            feature = Comp_Relation_C_for_GAT_Subgraph_Feature(instance_id=instance_id,
                                                               word_embed_matrix=bert_example_embeds_dict[instance_id],
                                                               masked_sent_text=masked_sent_text,
                                                               object_a_pos=object_a_pos,
                                                               object_b_pos=object_b_pos,
                                                               label_id=label_id,
                                                               edge_index=graph_edge_info)
            feature_list.append(feature)
        # endfor
        return feature_list
