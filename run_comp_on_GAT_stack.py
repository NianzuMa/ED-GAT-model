import os
import sys

# control which GPU to use
# os.environ['CUDA_VISIBLE_DEVICES'] = "1"

# --------------- add paths to PYTHON_PATH -----------
currentdir = os.path.dirname(os.path.realpath(__file__))
sys.path.append(currentdir)

currentdir = os.path.dirname(os.path.realpath(__file__))
parentdir = os.path.dirname(currentdir)
sys.path.append(parentdir)

from Main import Main
# ------------------ import common package ----------
import shutil
import json
import glob
import torch
import random
import math
import numpy as np
from tqdm import tqdm, trange

# ----------------- import model related package -----------
from transformers import (WEIGHTS_NAME, BertConfig,
                          BertForSequenceClassification, BertTokenizer,
                          RobertaConfig,
                          RobertaForSequenceClassification,
                          RobertaTokenizer,
                          XLMConfig, XLMForSequenceClassification,
                          XLMTokenizer, XLNetConfig,
                          XLNetForSequenceClassification,
                          XLNetTokenizer,
                          DistilBertConfig,
                          DistilBertForSequenceClassification,
                          DistilBertTokenizer)

MODEL_CLASSES = {
    'bert': (BertConfig, BertForSequenceClassification, BertTokenizer),
    'xlnet': (XLNetConfig, XLNetForSequenceClassification, XLNetTokenizer),
    'xlm': (XLMConfig, XLMForSequenceClassification, XLMTokenizer),
    'roberta': (RobertaConfig, RobertaForSequenceClassification, RobertaTokenizer),
    'distilbert': (DistilBertConfig, DistilBertForSequenceClassification, DistilBertTokenizer)
}

from data_utils import Comp_Relation_C_Processor_For_Graph_Attention_Network
from GAT_models import Net

from evaluation_metrics import acc_and_f1_multiclass as compute_metrics

output_mode = "classification"

from torch.utils.data import (DataLoader, RandomSampler, SequentialSampler,
                              TensorDataset)
from torch.utils.data.distributed import DistributedSampler

# --------------- config logging -------------
import logging

logger = logging.getLogger(__name__)


class RunCompGAT(Main):

    def __init__(self, best_dev_eval_file, considered_metrics):
        self.best_dev_eval_file = best_dev_eval_file
        self.considered_metrics = considered_metrics

    def argument_parser(self):
        import argparse

        parser = argparse.ArgumentParser()
        ## Required parameters
        parser.add_argument("--data_dir", default="./data/basicDependencies", type=str,
                            help="The input data dir. Should contain the .tsv files (or other data files) for the task.")
        parser.add_argument("--pretrained_transformer_model_type", default="bert", type=str,
                            help="Model type selected in the list: " + ", ".join(MODEL_CLASSES.keys()))
        parser.add_argument("--pretrained_transformer_model_name_or_path", default="bert-base-uncased", type=str,
                            help="Path to pre-trained model or shortcut name selected")
        parser.add_argument("--pretrained_bert_model_folder_for_feature_creation",
                            default=None,
                            type=str,
                            help="the pretrained model to create bert embedding for following task.")
        parser.add_argument("--model_name",
                            default="graph_attention_network",
                            help="the model to solve the problem")
        parser.add_argument("--task_name", default="comp_GAT", type=str,
                            help="The name of the task to train selected in the list: ... ")
        parser.add_argument("--dataset_name", default="masked_sent_final", type=str,
                            help="original, oversampling etc.")
        parser.add_argument("--output_dir", default="./result", type=str,
                            help="The output directory where the model predictions and checkpoints will be written.")
        parser.add_argument('--logging_steps', type=int, default=200,
                            help="Log every X updates steps.")

        #
        parser.add_argument("--remove_all_cached_features", action="store_true", default=False)

        # feature creation method
        parser.add_argument("--feature_creation", default="original_bert_model_using_original_text", type=str, help="The method to create features")
        # feature creation choice:
        # ["glove",
        #  "original_bert_model_using_original_text",
        #  ]

        # graph model related parameter
        # --------------- model configuration ------------
        parser.add_argument("--input_size", default=768, type=int, help="initial input size")
        parser.add_argument("--hidden_size", default=300, type=int, help="the hidden size for GAT layer")

        parser.add_argument("--heads", default=6, type=int, help="number of heads in the GAT layer")
        parser.add_argument("--att_dropout", default=0, type=float, help="")
        parser.add_argument("--stack_layer_num", default=4, type=int, help="the number of layers to stack")

        parser.add_argument("--num_classes", default=-1, type=int, help="the number of class")
        parser.add_argument("--embed_dropout", default=0.7, type=float, help="dropout for input word embedding")

        ## Other parameters
        parser.add_argument("--max_seq_length", default=256, type=int,
                            help="The maximum total input sequence length after tokenization. Sequences longer "
                                 "than this will be truncated, sequences shorter will be padded.")
        parser.add_argument("--do_train", action='store_true', default=True,
                            help="Whether to run training.")
        parser.add_argument("--do_eval", action='store_true', default=True,
                            help="Whether to run eval on the dev set.")
        parser.add_argument("--evaluate_during_training", action='store_true', default=True,
                            help="Rul evaluation during training at each logging step.")
        parser.add_argument("--do_lower_case", action='store_true',
                            help="Set this flag if you are using an uncased model.")
        parser.add_argument("--per_gpu_train_batch_size", default=16, type=int,
                            help="Batch size per GPU/CPU for training.")
        parser.add_argument("--per_gpu_eval_batch_size", default=8, type=int,
                            help="Batch size per GPU/CPU for evaluation.")
        # ------- training details ----------
        parser.add_argument('--gradient_accumulation_steps', type=int, default=1,
                            help="Number of updates steps to accumulate before performing a backward/update pass.")

        parser.add_argument("--learning_rate", default=0.001, type=float,
                            help="The initial learning rate for Adam.")
        parser.add_argument("--weight_decay", default=1e-4, type=float,
                            help="Weight deay if we apply some.")
        # -------------------------------------
        parser.add_argument("--adam_epsilon", default=1e-8, type=float,
                            help="Epsilon for Adam optimizer.")
        parser.add_argument("--max_grad_norm", default=1.0, type=float,
                            help="Max gradient norm.")
        parser.add_argument("--num_train_epochs", default=3, type=int,
                            help="Total number of training epochs to perform.")
        parser.add_argument("--max_steps", default=-1, type=int,
                            help="If > 0: set total number of training steps to perform. Override num_train_epochs.")
        parser.add_argument("--warmup_steps", default=0, type=int,
                            help="Linear warmup over warmup_steps.")
        parser.add_argument('--save_steps', type=int, default=50,
                            help="Save checkpoint every X updates steps.")
        parser.add_argument("--eval_all_checkpoints", action='store_true',
                            help="Evaluate all checkpoints starting with the same prefix as model_name ending and ending with step number")
        parser.add_argument("--no_cuda", action='store_true',
                            help="Avoid using CUDA when available")
        parser.add_argument('--overwrite_output_dir', action='store_true',
                            help="Overwrite the content of the output directory")
        parser.add_argument('--overwrite_cache', action='store_true',
                            help="Overwrite the cached training and evaluation sets")
        parser.add_argument('--seed', type=int, default=42,
                            help="random seed for initialization")
        parser.add_argument('--fp16', action='store_true',
                            help="Whether to use 16-bit (mixed) precision (through NVIDIA apex) instead of 32-bit")
        parser.add_argument('--fp16_opt_level', type=str, default='O1',
                            help="For fp16: Apex AMP optimization level selected in ['O0', 'O1', 'O2', and 'O3']."
                                 "See details at https://nvidia.github.io/apex/amp.html")
        parser.add_argument("--local_rank", type=int, default=-1,
                            help="For distributed training: local_rank")
        parser.add_argument('--server_ip', type=str, default='', help="For distant debugging.")
        parser.add_argument('--server_port', type=str, default='', help="For distant debugging.")
        args = parser.parse_args()

        return args

    def set_seed(self, args):
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        if args.n_gpu > 0:
            torch.cuda.manual_seed_all(args.seed)
        pass

    def train(self, args, train_dataset, model_class, tokenizer_class, model, processor, label_list):
        """ Train the model """

        # set optimizer
        optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)

        batch_size = args.per_gpu_train_batch_size
        total_train_size = len(train_dataset)
        batch_num = math.ceil(total_train_size * 1.0 / batch_size)

        logger.info("***** Running training *****")
        logger.info("  Num examples = %d", len(train_dataset))
        logger.info("  Num Epochs = %d", args.num_train_epochs)
        logger.info("  Num Batch_size = %d", batch_size)
        logger.info("  Instantaneous batch size per GPU = %d", args.per_gpu_train_batch_size)
        logger.info("  Total optimization steps = %d", args.num_train_epochs * batch_num)

        train_loss = 0.0
        global_step = 0
        self.set_seed(args)

        # train_dataset is a feature list
        global_step = 0
        total_train_loss = 0

        best_eval_score = 0

        for epoch_index in range(args.num_train_epochs):
            # shuffle
            random.shuffle(train_dataset)
            logger.info("The train dataset is shuffle for epoch {}".format(epoch_index))

            for batch_index in range(batch_num):
                global_step += 1
                model.train()
                model.zero_grad()

                word_embed_matrix, graph_edge_list, target_mask_list, label_id_list, input_token_size_list = processor.get_batch(
                    train_dataset[batch_index * batch_size: (batch_index + 1) * batch_size])

                word_embed_matrix = torch.tensor(word_embed_matrix, dtype=torch.float).to(args.device)
                # https://pytorch-geometric.readthedocs.io/en/latest/notes/introduction.html#data-handling-of-graphs
                # you should transpose and call contiguous on it
                graph_edge_list = torch.tensor(np.array(graph_edge_list), dtype=torch.long).t().contiguous().to(args.device)
                label_id_list = torch.tensor(label_id_list, dtype=torch.long).to(args.device)

                logits, loss = model(args, word_embed_matrix, target_mask_list, graph_edge_list, label_id_list)

                loss.backward()

                optimizer.step()
                model.zero_grad()

                if batch_index % 100 == 0:
                    logger.info(
                        "epoch: {} batch:{}/{} global_step: {}".format(epoch_index, batch_index, batch_num,
                                                                       global_step))

                total_train_loss += loss
                logger.info("-------- Training loss: {}".format(total_train_loss / global_step))

                if global_step % args.logging_steps == 0:

                    logger.info("Training loss: {}".format(total_train_loss / global_step))

                    if args.evaluate_during_training:
                        self.evaluate(args, processor, model_class, tokenizer_class, model, label_list, mode="train",
                                      epoch_index=epoch_index, step=global_step)

                        eval_results = self.evaluate(args, processor, model_class, tokenizer_class, model, label_list,
                                                     mode="dev", epoch_index=epoch_index, step=global_step)
                        # save model if the model give the best metrics we care
                        current_eval_score = eval_results[self.considered_metrics]
                        if current_eval_score > best_eval_score:
                            best_eval_score = current_eval_score
                            self.save_GAT_model(model, self.GAT_model_path)
                        # endif

                        with open(self.best_dev_eval_file, mode="w") as fout:
                            fout.write(json.dumps(eval_results) + "\n")
                        # endwith

                        self.evaluate(args, processor, model_class, tokenizer_class, model, label_list, mode="test",
                                      epoch_index=epoch_index, step=global_step)
                    # endif
                # endif
            # endfor

            # make sure after each epoch, there is evaluation
            self.evaluate(args, processor, model_class, tokenizer_class, model, label_list, mode="train",
                          epoch_index=epoch_index, step=global_step)

            eval_results = self.evaluate(args, processor, model_class, tokenizer_class, model, label_list, mode="dev",
                                         epoch_index=epoch_index, step=global_step)

            current_eval_score = eval_results[self.considered_metrics]
            if current_eval_score > best_eval_score:
                best_eval_score = current_eval_score
                self.save_GAT_model(model, self.GAT_model_path)
            # endif

            with open(self.best_dev_eval_file, mode="w") as fout:
                fout.write(json.dumps(eval_results) + "\n")
            # endwith

            self.evaluate(args, processor, model_class, tokenizer_class, model, label_list, mode="test",
                          epoch_index=epoch_index, step=global_step)
        # endepochfor

    def evaluate(self, args, processor, model_class, tokenizer_class, model, label_list, mode=None, epoch_index=None,
                 step=None,
                 output_file=None):
        assert epoch_index is not None
        assert mode in ["train", "dev", "test"]

        eval_task = args.task_name
        eval_output_dir = args.output_dir

        results = {}
        # eval dataset could be train, dev, test
        eval_dataset = self.load_and_cache_examples(args, processor, eval_task, model_class, tokenizer_class,
                                                    evaluate=True,
                                                    mode=mode)

        if not os.path.exists(eval_output_dir) and args.local_rank in [-1, 0]:
            os.makedirs(eval_output_dir)

        eval_batch_size = args.per_gpu_eval_batch_size
        total_eval_size = len(eval_dataset)
        batch_num = math.ceil(total_eval_size * 1.0 / eval_batch_size)

        # Eval!
        logger.info("***** Running evaluation for {} dataset*****".format(mode))
        logger.info("  Num examples = %d", len(eval_dataset))
        logger.info("  Batch size = %d", args.per_gpu_eval_batch_size)
        logger.info("  Batch num per epoch = %d", batch_num)

        eval_loss = 0.0
        nb_eval_steps = 0
        preds = None
        out_label_ids = None
        batch_index = -1

        for batch_index in range(batch_num):
            model.eval()

            # logger.info("batch: {}/{}".format(batch_index, batch_num))

            word_embed_matrix, graph_edge_list, target_mask_list, label_id_list, input_token_size_list = processor.get_batch(
                eval_dataset[batch_index * eval_batch_size: (batch_index + 1) * eval_batch_size])

            word_embed_matrix = torch.tensor(word_embed_matrix, dtype=torch.float).to(args.device)
            graph_edge_list = torch.tensor(np.array(graph_edge_list), dtype=torch.long).t().contiguous().to(args.device)
            label_id_list = torch.tensor(label_id_list, dtype=torch.long).to(args.device)

            with torch.no_grad():
                logits, tmp_eval_loss = model(args,
                                              word_embed_matrix,
                                              target_mask_list,
                                              graph_edge_list,
                                              label_id_list)

                eval_loss += tmp_eval_loss.mean().item()
            # endwith

            nb_eval_steps += 1
            if preds is None:
                preds = logits.detach().cpu().numpy()
                out_label_ids = label_id_list.detach().cpu().numpy()
            else:
                preds = np.append(preds, logits.detach().cpu().numpy(), axis=0)
                out_label_ids = np.append(out_label_ids, label_id_list.detach().cpu().numpy(), axis=0)

        pred_logits = preds
        eval_loss = eval_loss / nb_eval_steps
        if args.output_mode == "classification":
            preds = np.argmax(preds, axis=1)
        elif args.output_mode == "regression":
            preds = np.squeeze(preds)
        result = compute_metrics(preds, out_label_ids, label_list)
        results.update(result)
        results[mode + "-loss"] = eval_loss

        if output_file is None:
            output_eval_file = os.path.join(eval_output_dir, mode + "_results.txt")
        else:
            output_eval_file = output_file
        # endif

        output_json = None
        with open(output_eval_file, "a") as writer:
            output_json = {'epoch': epoch_index,
                           'step': step,
                           'mode': mode}
            output_json.update(results)
            logger.info("{}".format(output_json))
            writer.write(json.dumps(output_json) + '\n')
        # endwith

        # write prediction file
        if mode == "test":
            i_to_label_map = {i: label for i, label in enumerate(label_list)}
            pred_label_list = [i_to_label_map[pred_id] for pred_id in preds.tolist()]
            self.write_test_prediction(data_input_dir=args.data_dir,
                                       pred_label_list=pred_label_list,
                                       pred_logits=pred_logits,
                                       output_dir=args.output_dir,
                                       epoch_index=epoch_index,
                                       step=step)
        # endif
        return output_json

    def load_and_cache_examples(self, args, processor, task, model_class, tokenizer_class, evaluate=False, mode=None):
        """
        for train, dev, test dataset
        :param args:
        :param task:
        :param tokenizer:
        :param evaluate:
        :param mode:
        :return:
        """
        assert mode is not None
        assert mode in ["train", "dev", "test"]

        if args.local_rank not in [-1, 0] and not evaluate:
            torch.distributed.barrier()  # Make sure only the first process in distributed training process the dataset, and the others will use the cache

        # endif
        assert processor is not None

        # Load data features from cache or dataset file
        cached_features_file = os.path.join(args.data_dir, 'cached_{}_{}_{}_{}_{}'.format(
            mode,
            str(args.max_seq_length),
            args.pretrained_transformer_model_name_or_path,
            args.feature_creation,
            str(task)))

        # if cased file exists, load file
        if os.path.exists(cached_features_file) and not args.overwrite_cache:
            logger.info("Loading features from cached file %s", cached_features_file)
            device_str = "cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu"
            features = torch.load(cached_features_file, map_location=device_str)
            logger.info("dataset is loaded to device: {}".format(device_str))
        else:

            logger.info("Creating features from dataset file at %s", args.data_dir)
            label_list = processor.get_labels()

            # only load when need to create features
            if args.feature_creation == "glove":
                logger.info("loading gensim model ...")
                processor._load_gensim_model()
                logger.info("gensim model is loaded")
            # endif

            examples = None
            if mode == "train":
                examples = processor.get_train_examples(args.data_dir)
            if mode == "dev":
                examples = processor.get_dev_examples(args.data_dir)
            if mode == "test":
                examples = processor.get_test_examples(args.data_dir)
            # endif
            assert examples is not None

            features = processor.convert_examples_to_features_from_masked_sent_subgraphs(args=args,
                                                                                         model_class=model_class,
                                                                                         tokenizer_class=tokenizer_class,
                                                                                         examples=examples,
                                                                                         max_length=args.max_seq_length,
                                                                                         label_list=label_list,
                                                                                         output_mode=output_mode,
                                                                                         mode=mode)

            if args.local_rank in [-1, 0]:
                logger.info("Saving features into cached file {}".format(cached_features_file))
                torch.save(features, cached_features_file)
        # endif

        if args.local_rank == 0 and not evaluate:
            torch.distributed.barrier()  # Make sure only the first process in distributed training process the dataset, and the others will use the cache

        return features

    def write_test_prediction(self, data_input_dir, pred_label_list, pred_logits, output_dir, epoch_index, step):
        """
        :param data_input_dir:
        :param pred_label_list:
        :param output_dir:
        :param epoch_index:
        :param step_index:
        :return:
        """
        instance_id_list = []
        golden_label_list = []
        text_list = []
        object_a_list = []
        object_b_list = []

        new_json_obj_list = []

        with open(os.path.join(data_input_dir, "test.json"), mode='r') as fin:
            i = -1
            for line in fin:
                i += 1
                line = line.strip()
                json_obj = json.loads(line)

                json_obj["pred_logits"] = [str(item) for item in list(pred_logits[i])]
                json_obj["pred_label_id"] = pred_label_list[i]
                new_json_obj_list.append(json_obj)

                instance_id = json_obj["instance_id"]
                label = json_obj['label']
                masked_sent_text = json_obj['masked_sent_text']

                object_a_text = json_obj["object_a"]
                object_b_text = json_obj["object_b"]

                instance_id_list.append(instance_id)
                golden_label_list.append(label)
                text_list.append(masked_sent_text)
                object_a_list.append(object_a_text)
                object_b_list.append(object_b_text)

            # endfor
        # endwith
        diff_list = []

        for i in range(len(golden_label_list)):
            if golden_label_list[i] != pred_label_list[i]:
                diff_list.append("*")
            else:
                diff_list.append('')

        with open(os.path.join(output_dir, "test_pred_epoch_{}_step_{}.txt".format(epoch_index, step)),
                  mode='w') as fout:
            for i in range(len(golden_label_list)):
                fout.write("{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\n".format(
                    instance_id_list[i],
                    json.dumps(new_json_obj_list[i]),
                    text_list[i],
                    object_a_list[i],
                    object_b_list[i],
                    golden_label_list[i],
                    diff_list[i],
                    pred_label_list[i]))

        pass

    def write_params(self, args, logger):
        args_dict = vars(args)
        config_file = os.path.join(args.output_dir, 'params.json')
        with open(config_file, 'w') as f:
            f.write(json.dumps(args_dict) + '\n')

        param_file = os.path.join(args.output_dir, "params.txt")  # the same as config.jsonl but print in the nicer way
        with open(param_file, "w") as f:
            f.write("============ parameters ============\n")
            logger.info("============ parameters =============")
            for k, v in args_dict.items():
                f.write("{}: {}\n".format(k, v))
                logger.info("{}: {}".format(k, v))
            # endfor
            logger.info("=====================================")
        # endwith

    def save_model(self, model, tokenizer, args):
        """
        The pytorch recommended way is to save/load state_dict
        :return:
        """
        model_to_save = model.module if hasattr(model,
                                                'module') else model  # Take care of distributed/parallel training
        model_to_save.save_pretrained(args.output_dir)
        tokenizer.save_pretrained(args.output_dir)
        pass

    def load_model(self, model_class, tokenizer_class, args):
        """
        :param saved_model_file:
        :return:
        """
        model = model_class.from_pretrained(args.output_dir)
        tokenizer = tokenizer_class.from_pretrained(args.output_dir)
        model.to(args.device)
        return model, tokenizer

    def save_GAT_model(self, model, path):
        torch.save(model.state_dict(), path)
        pass

    def load_GAT_model(self, model, path):
        model.load_state_dict(torch.load(path))
        return model
        pass

    def _validate_do_lower_case(self, args):
        """
        Since, do lower case or not is based on the tokenizer of pretrained bert
        So args.do_lower_case should be consistent with the pretrained bert tokenizer config file
        :return:
        """
        if args.pretrained_bert_model_folder_for_feature_creation is not None:
            bert_tokenizer_config_file = os.path.join(args.pretrained_bert_model_folder_for_feature_creation,
                                                      "tokenizer_config.json")
            with open(bert_tokenizer_config_file, mode="r") as fin:
                tokenizer_do_lower_case = json.load(fin)["do_lower_case"]
            # endwith

            if tokenizer_do_lower_case != args.do_lower_case:
                logging.error("args.do_lower_case={} does not consistent with tokenizer config do_lower_case={}".format(
                    args.do_lower_case, tokenizer_do_lower_case))
                sys.exit(1)
            # endif

    def _validate_input_size(self, args):
        if args.feature_creation == "glove":
            assert args.input_size == 300

        if "bert" in args.feature_creation:
            if "bert-base" in args.pretrained_transformer_model_name_or_path:
                assert args.input_size == 768
            # endif
            if "bert-large" in args.pretrained_transformer_model_name_or_path:
                assert args.input_size == 1024
            # endif

    def run_app(self):

        args = self.argument_parser()

        # validate the do_lower_case, make sure that, tokenizer do_lower_case is consistent with args.do_lower_case
        self._validate_do_lower_case(args)
        self._validate_input_size(args)

        from datetime import datetime

        start_time = datetime.now().strftime("%Y-%m-%d__%H-%M__%f")
        args.output_dir = os.path.join(args.output_dir, args.task_name, args.dataset_name,
                                       start_time)
        # set up output directory
        # 1. the output_dir exists
        # 2. if it exists, os.listdir will print out all the files and directories under this path. If there is nothing
        #    in this directory, it will return empty list [], which is treated as False
        # 3. if there is something, check if do training model
        # 4. if do training for the model, does not overwrite the output_dir
        if os.path.exists(args.output_dir) and os.listdir(
                args.output_dir) and args.do_train and not args.overwrite_output_dir:
            raise ValueError(
                "Output directory ({}) already exists and is not empty. Use --overwrite_output_dir to overcome.".format(
                    args.output_dir))

        if not os.path.exists(args.output_dir):
            os.makedirs(args.output_dir)
        # endif
        logger.info("output folder is {}".format(args.output_dir))

        # Setup CUDA, GPU
        if args.local_rank == -1 or args.no_cuda:
            device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
            args.n_gpu = torch.cuda.device_count()
        else:  # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
            torch.cuda.set_device(args.local_rank)
            device = torch.device("cuda", args.local_rank)
            torch.distributed.init_process_group(backend='nccl')
            args.n_gpu = 1
        # endif

        # write parameter
        self.write_params(args, logger)

        # set device
        args.device = device

        # Setup logging
        logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                            datefmt='%m/%d/%Y %H:%M:%S',
                            level=logging.INFO if args.local_rank in [-1, 0] else logging.WARN)
        logger.warning("Process rank: %s, device: %s, n_gpu: %s, distributed training: %s, 16-bits training: %s",
                       args.local_rank, device, args.n_gpu, bool(args.local_rank != -1), args.fp16)

        # Set seed
        self.set_seed(args)

        # Prepare GLUE task
        args.output_mode = output_mode
        # processor
        processor = Comp_Relation_C_Processor_For_Graph_Attention_Network(args)
        assert processor is not None


        label_list = processor.get_labels()
        args.num_classes = len(label_list)

        # Load pretrained model and tokenizer
        if args.local_rank not in [-1, 0]:
            torch.distributed.barrier()  # Make sure only the first process in distributed training will download model & vocab

        # load pretrained transformer model
        config_class, model_class, tokenizer_class = MODEL_CLASSES[args.pretrained_transformer_model_type]

        if args.local_rank == 0:
            torch.distributed.barrier()  # Make sure only the first process in distributed training will download model & vocab

        logger.info("Training/evaluation parameters %s", args)

        # Before Training
        self.best_dev_eval_file = os.path.join(args.output_dir, "best_dev_info.json")
        self.GAT_model_path = os.path.join(args.output_dir, "GAT_model.pt")

        model = Net(args)
        model.to(args.device)

        # Training

        if args.do_train:
            # before create feature, delete all the already created feature, the cased feature will only be used within
            # train() for evaluation during training
            if args.remove_all_cached_features:
                self._delete_cached_feature_file(args)

            # firstly create feature for each dataset
            train_dataset = self.load_and_cache_examples(args, processor, args.task_name, model_class, tokenizer_class,
                                                         evaluate=False,
                                                         mode='train')

            dev_dataset = self.load_and_cache_examples(args, processor, args.task_name, model_class, tokenizer_class,
                                                       evaluate=True,
                                                       mode='dev')
            test_dataset = self.load_and_cache_examples(args, processor, args.task_name, model_class, tokenizer_class,
                                                        evaluate=True,
                                                        mode='test')

            self.train(args, train_dataset, model_class, tokenizer_class, model, processor, label_list)

        self.move_log_file_to_output_directory(args.output_dir)

    def _delete_cached_feature_file(self, args):
        for dir, subdir, file_list in os.walk(args.data_dir):
            for file in file_list:
                if "cached" in file:
                    file_path = os.path.join(dir, file)
                    os.remove(file_path)
                # endif
            # endfor
        # endfor

    def move_log_file_to_output_directory(self, output_dir):
        super(RunCompGAT, self).move_log_file_to_output_directory(output_dir)
        pass


if __name__ == '__main__':
    run = RunCompGAT(best_dev_eval_file="best_dev_info.json", considered_metrics="f1_weighted")
    run.run_app()
