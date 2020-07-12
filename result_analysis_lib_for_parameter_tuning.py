import os
import json
import matplotlib

import matplotlib.pyplot as plt
import logging
import shutil

# single_folder = "/home/nianzu/python_project/comparative-master/Classification/graph_network_for_comparative_sentence/run_graph_attention_model/result/comp_GAT/original_bert_embed/2019-11-25__12-48__472871"

# single_folder = "/Users/nianzu/Box Sync/PythonProject/comparative-master/Classification/graph_network_for_comparative_sentence/run_graph_attention_model/result/comp_GAT/masked_sent_final_original_bert_large_cased/2019-11-26__15-40__952718"
# single_folder = "/home/nianzu/python_project/comparative-master/Classification/graph_network_for_comparative_sentence/run_graph_attention_model/result/comp_GAT/glove/2019-11-26__22-32__574547"
# single_folder = "/home/nianzu/python_project/comparative-master/Classification/graph_network_for_comparative_sentence/run_graph_attention_model/result/comp_GAT/original_bert_model_using_original_text_uncased/2019-11-27__16-01__819685"

metrics_care_about = "f1_micro"
best_index_name = "step"

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ResultAnalyzerForParameterTuning(object):
    """
    Note that, the steps are:
    1. find the best result on dev dataset and the corresponding step number (current it is based on step num, not epoch num)
    2. find the test dataset result on the corresponding step number
    It is better to use step number, because for different task, the dataset size is different
    """

    def __init__(self, metrics, best_index_name, tuned_parameter, parent_folder_for_all_results, x_axis_label,
                 y_axis_label, fig_file_name):
        """
        :param parent_folder_for_all_results:
        :param metrics:
        :param best_index_name: should it based on epoch or step
        """
        self.parent_folder_for_all_results = parent_folder_for_all_results
        self.metrics = metrics
        self.best_index_name = best_index_name
        self.tuned_parameter = tuned_parameter
        ## plot information
        self.x_axis_label = x_axis_label
        self.y_axis_label = y_axis_label
        self.fig_file_name = fig_file_name

    def _print_out_all_test_result(self, test_json_info):
        for parameter, test_json in test_json_info:
            result = []
            score = test_json[self.metrics]
            result.append(score)
            result.extend(test_json['f_None'])
            result = [str(round(item * 100.0, 2)) for item in result]
            logger.info("{} : {} & {} & {} & {}".format(parameter, *result))

    def file_handler_for_parameter_analysis(self, plot=False):
        dir_list = []
        for dirName, subdirList, fileList in os.walk(self.parent_folder_for_all_results):
            if dirName != self.parent_folder_for_all_results:
                dir_list.append(dirName)
            # endif
        # endfor

        result_list = []
        for dir_n in dir_list:
            tuned_parameter, best_dev_json_obj, test_json_obj = self.get_single_folder_best_result(dir_n)
            result_list.append([tuned_parameter, test_json_obj[self.metrics], best_dev_json_obj, test_json_obj, dir_n])
        # endfro

        result_list = sorted(result_list, key=lambda x: x[0])  # sort by parameter

        test_json_info = []
        for result in result_list:
            tuned_parameter = result[0]
            test_json = result[-2]
            test_json_info.append((tuned_parameter, test_json))
        # endfor

        self._print_out_all_test_result(test_json_info)

        plt_parameter_list = []
        plt_test_score_list = []

        for result in result_list:
            tuned_parameter, test_score, best_dev_json_obj, test_json_obj, dir_n = result
            plt_parameter_list.append(tuned_parameter)
            plt_test_score_list.append(test_json_obj[self.metrics])
        # endfor

        if plot:
            self.plot_epoch_score(plt_parameter_list, plt_test_score_list)
        #endif
        logger.info(plt_parameter_list)
        logger.info(plt_test_score_list)

        highest_result = sorted(result_list, key=lambda x: x[1])

        best_train_parameter = highest_result[-1][0]
        best_test_score = highest_result[-1][1]
        best_classification_report = highest_result[-1][-2]['classification_report']
        best_test_json = highest_result[-1][-2]
        dir_n = highest_result[-1][-1]

        logger.info("best parameter {} is {}".format(self.tuned_parameter, best_train_parameter))
        logger.info("best {} is: {}".format(self.metrics, best_test_score))
        logger.info(best_classification_report)
        logger.info(json.dumps(best_test_json))
        logger.info("dir_name: {}".format(dir_n))

        filling_form_item = [highest_result[-1][-2][self.metrics]] + highest_result[-1][-2]["f_None"]
        filling_form_item = [str(round(item * 100, 2)) for item in filling_form_item]
        logger.info(self.metrics + "     " + "f_None")
        logger.info("===================")
        logger.info(' & '.join(filling_form_item))
        logger.info("===================")

        return plt_parameter_list, plt_test_score_list
        pass

    def get_single_folder_best_result(self, single_result_folder):
        """
        for each parameter choice, what is the best result
        :return:
        """
        with open(os.path.join(single_result_folder, "params.json"), mode="r") as fin:
            params = json.loads(fin.read().strip())
        # endwith

        tuned_parameter = params[self.tuned_parameter]

        dev_file_path = os.path.join(single_result_folder, "dev_results.txt")
        best_dev_json_obj = self._analyze_dev_result_file(dev_file_path)
        best_dev_index = best_dev_json_obj[self.best_index_name]

        test_file_path = os.path.join(single_result_folder, "test_results.txt")
        test_json_obj = self._search_test_result_file(test_file_path, best_dev_index)

        return tuned_parameter, best_dev_json_obj, test_json_obj

    def analyze_single_folder(self, single_result_folder):
        with open(os.path.join(single_result_folder, "params.json"), mode="r") as fin:
            params = json.loads(fin.read().strip())
        # endwith

        total_train_epoch = params["num_train_epochs"]

        dev_file_path = os.path.join(single_result_folder, "dev_results.txt")
        best_dev_json = self._analyze_dev_result_file(dev_file_path)
        best_dev_index = best_dev_json[self.best_index_name]

        logger.info("The best result for dev logging num {}".format(best_dev_index))

        test_file_path = os.path.join(single_result_folder, "test_results.txt")
        test_json_obj = self._search_test_result_file(test_file_path, best_dev_index)

        logger.info("The best test result is {}".format(json.dumps(test_json_obj)))
        logger.info("Classification Report:")
        logger.info(test_json_obj["classification_report"])
        logger.info("test {} is {}".format(self.metrics, test_json_obj[self.metrics]))

        # return total_train_epoch, test_json_obj[self.metrics], test_json_obj
        pass

    def plot_epoch_score(self, x, y):
        # plotting the points
        plt.plot(x, y, marker='o')

        # naming the x axis
        plt.xlabel(self.x_axis_label)
        # naming the y axis
        plt.ylabel(self.y_axis_label)

        # giving a title to my graph

        # function to show the plot

        for file_name in self.fig_file_name:
            plt.savefig(file_name)
        # endfor

        #plt.show()

    # def analyze_single_result(self, single_result_folder):
    #     """
    #     !! step index
    #     Check for each run, what this the best result
    #     1. check on dev data, which step index gives the best result
    #     2. then check the test data, what is the result on this step index
    #     :return:
    #     """
    #     params = None
    #     with open(os.path.join(single_result_folder, "params.json"), mode="r") as fin:
    #         params = json.loads(fin.read().strip())
    #     # endwith
    #
    #     total_train_epoch = params["num_train_epochs"]
    #
    #     dev_file_path = os.path.join(single_result_folder, "dev_results.txt")
    #     best_dev_json = self._analyze_dev_result_file(dev_file_path)
    #     best_dev_index = best_dev_json[self.best_index_name]
    #
    #     test_file_path = os.path.join(single_result_folder, "test_results.txt")
    #     test_json_obj = self._search_test_result_file(test_file_path, best_dev_index)
    #
    #     return total_train_epoch, test_json_obj[self.metrics], test_json_obj

    def _analyze_dev_result_file(self, file_path):
        """
        find the epoch index number with the best score for the metrics we care about
        :param file_path:
        :return:
        """
        best_result_json = None
        best_score = 0
        with open(file_path, mode="r") as fin:
            for line in fin:
                line = line.strip()
                json_obj = json.loads(line)
                score = json_obj[self.metrics]
                if score > best_score:
                    best_score = score
                    best_result_json = json_obj
        # endfor
        assert best_result_json is not None
        return best_result_json

    def _search_test_result_file(self, file_path, best_dev_index):
        """
        given the best epoch index in dev dataset, find the corresponding result in test dataset
        :param file_path:
        :param best_dev_epoch:
        :return:
        """
        with open(file_path, mode="r") as fin:
            for line in fin:
                line = line.strip()
                json_obj = json.loads(line)
                index = json_obj[self.best_index_name]
                if index == best_dev_index:
                    return json_obj
                # endif
            # endfor
        # endwith


class DrawGraph(object):
    def __init__(self, parent_result_folder, logging_metric, score_metric):
        self.parent_result_folder = parent_result_folder
        self.logging_metric = logging_metric
        self.score_metric = score_metric

    def _read_data_points(self, file_name, logging_metric, score_metric):
        data_point_list = []
        with open(file_name, mode="r") as fin:
            for line in fin:
                line = line.strip()
                json_obj = json.loads(line)
                logging_num = json_obj[logging_metric]
                score = json_obj[score_metric]
                data_point_list.append((logging_num, score))
            # endfor
        # endwith

        return data_point_list

    def _draw_x_y_graph(self,
                        train_x_list, train_y_list,
                        dev_x_list, dev_y_list,
                        test_x_list, test_y_list,
                        x_label, y_label, title):
        # plotting the points
        plt.plot(train_x_list, train_y_list)
        plt.plot(dev_x_list, dev_y_list)
        plt.plot(test_x_list, test_y_list)

        # naming the x axis
        plt.xlabel(x_label)
        # naming the y axis
        plt.ylabel(y_label)
        # giving a title to my graph
        plt.title(title)
        # function to show the plot
        plt.show()
        pass

    def draw_graph(self, result_folder, graph_title):

        data_file = os.path.join(result_folder, "train_results.txt")
        data_point_list = self._read_data_points(data_file, self.logging_metric, self.score_metric)
        train_step_list, train_score_list = zip(*data_point_list)

        data_file = os.path.join(result_folder, "dev_results.txt")
        data_point_list = self._read_data_points(data_file, self.logging_metric, self.score_metric)
        dev_step_list, dev_score_list = zip(*data_point_list)

        data_file = os.path.join(result_folder, "test_results.txt")
        data_point_list = self._read_data_points(data_file, self.logging_metric, self.score_metric)
        test_step_list, test_score_list = zip(*data_point_list)

        self._draw_x_y_graph(train_step_list, train_score_list,
                             dev_step_list, dev_score_list,
                             test_step_list, test_score_list,
                             self.logging_metric, self.score_metric, graph_title)
        # endfor

    def file_handler(self):
        for dir, subdir, filename in os.walk(self.parent_result_folder):
            for item in subdir:
                sub_folder = os.path.join(dir, item)
                self.draw_graph(sub_folder, item)
            # endfor


def original_bert_base_uncased_num_layer_tuning():
    parent_folder = "/home/nianzu/python_project/comparative-master/Classification/graph_network_for_comparative_sentence/run_graph_attention_model/result/comp_GAT/original_bert_model_using_original_text_uncased"
    file_name = "bert-base-uncased-original-bert-original-text-input-Num-Layer-Tuning"

    logger.info("\n\n================{}======================".format(file_name))

    analyzer = ResultAnalyzerForParameterTuning(metrics_care_about,
                                                best_index_name,
                                                tuned_parameter='stack_layer_num',
                                                parent_folder_for_all_results=parent_folder,
                                                x_axis_label="Number of Stacked GAT Layers",
                                                y_axis_label=metrics_care_about,
                                                fig_file_name=[file_name + ".pdf", file_name + ".png"])
    return analyzer.file_handler_for_parameter_analysis()


def glove_lr_tuning():
    parent_folder = "/home/nianzu/python_project/comparative-master/Classification/graph_network_for_comparative_sentence/run_graph_attention_model/result/comp_GAT/glove_lr"
    file_name = "glove-lr-tuning"

    logger.info("\n\n================{}======================".format(file_name))

    analyzer = ResultAnalyzerForParameterTuning(metrics_care_about,
                                                best_index_name,
                                                tuned_parameter='learning_rate',
                                                parent_folder_for_all_results=parent_folder,
                                                x_axis_label="Learning Rate",
                                                y_axis_label=metrics_care_about,
                                                fig_file_name=[file_name + ".pdf", file_name + ".png"])
    analyzer.file_handler_for_parameter_analysis()


def glove_num_layer():
    parent_folder = "/home/nianzu/python_project/comparative-master/Classification/graph_network_for_comparative_sentence/run_graph_attention_model/result/comp_GAT/glove_num_layer"
    file_name = "glove-num-layer-tuning"

    logger.info("\n\n================{}======================".format(file_name))

    analyzer = ResultAnalyzerForParameterTuning(metrics_care_about,
                                                best_index_name,
                                                tuned_parameter='stack_layer_num',
                                                parent_folder_for_all_results=parent_folder,
                                                x_axis_label="Number of Stacked GAT Layers",
                                                y_axis_label=metrics_care_about,
                                                fig_file_name=[file_name + ".pdf", file_name + ".png"])
    return analyzer.file_handler_for_parameter_analysis()


def glove_dropout_att():
    parent_folder = "/home/nianzu/python_project/comparative-master/Classification/graph_network_for_comparative_sentence/run_graph_attention_model/result/comp_GAT/glove_dropout_att"
    file_name = "att_dropout_tuning"

    logger.info("\n\n================{}======================".format(file_name))

    analyzer = ResultAnalyzerForParameterTuning(metrics_care_about,
                                                best_index_name,
                                                tuned_parameter='att_dropout',
                                                parent_folder_for_all_results=parent_folder,
                                                x_axis_label="GAT Attention Dropout",
                                                y_axis_label=metrics_care_about,
                                                fig_file_name=[file_name + ".pdf", file_name + ".png"])
    analyzer.file_handler_for_parameter_analysis()

    pass


def glove_dropout_embed():
    parent_folder = "/home/nianzu/python_project/comparative-master/Classification/graph_network_for_comparative_sentence/run_graph_attention_model/result/comp_GAT/glove_dropout_embed"
    file_name = "embed_dropout_tuning"

    logger.info("\n\n================{}======================".format(file_name))

    analyzer = ResultAnalyzerForParameterTuning(metrics_care_about,
                                                best_index_name,
                                                tuned_parameter='embed_dropout',
                                                parent_folder_for_all_results=parent_folder,
                                                x_axis_label="GAT Word Embedding Dropout",
                                                y_axis_label=metrics_care_about,
                                                fig_file_name=[file_name + ".pdf", file_name + ".png"])
    analyzer.file_handler_for_parameter_analysis()


def draw_GAT_glove_bert_layer_num_together():
    plt_bert_layer_num_list, plt_bert_test_score_list = original_bert_base_uncased_num_layer_tuning()
    plt_glove_layer_num_list, plt_glove_test_score_list = glove_num_layer()

    plt.plot(plt_bert_layer_num_list, plt_bert_test_score_list, marker='o', label="TD-GAT-BERT", linestyle="-")
    plt.plot(plt_glove_layer_num_list, plt_glove_test_score_list, marker="*", label="TD-GAT-GloVe", linestyle="--")
    plt.legend(loc="lower right")

    # naming the x axis
    plt.xlabel("Number of GAT Stacked Layers")
    # naming the y axis
    plt.ylabel("F1 Weighted")

    # giving a title to my graph

    # function to show the plot
    file_name = "glove_bert_GAT_layer_num"

    plt.savefig(file_name + ".png")
    plt.savefig(file_name + ".pdf")
    # endfor

    plt.show()


def glove_GAT_stack_layer_num_tuning():
    # parent_folder = "/home/nianzu/python_project/comparative-master/Classification/graph_network_for_comparative_sentence/run_GAT_stack_model/result/comp_GAT/uncased_glove_stack"
    parent_folder = "/home/nianzu/python_project/comparative_sentence_analysis/graph_network_for_comparative_sentence_analysis/ED_GAT_model/result/comp_GAT/uncased_glove_stack"
    file_name = "glove-GAT-stack-num-layer-tuning"

    logger.info("\n\n================{}======================".format(file_name))

    analyzer = ResultAnalyzerForParameterTuning(metrics_care_about,
                                                best_index_name,
                                                tuned_parameter='stack_layer_num',
                                                parent_folder_for_all_results=parent_folder,
                                                x_axis_label="Number of Stacked GAT Layers",
                                                y_axis_label=metrics_care_about,
                                                fig_file_name=[file_name + ".pdf", file_name + ".png"])
    return analyzer.file_handler_for_parameter_analysis()


def bert_GAT_stack_layer_num_tuning():
    # parent_folder = "/home/nianzu/python_project/comparative-master/Classification/graph_network_for_comparative_sentence/run_GAT_stack_model/result/comp_GAT/uncased_bert_GAT_stack"
    #parent_folder = "/home/nianzu/python_project/comparative_sentence_analysis/graph_network_for_comparative_sentence_analysis/ED_GAT_model/result/comp_GAT/uncased_bert_GAT_stack"
    #parent_folder = "/home/nianzu/python_project/comparative_sentence_analysis/graph_network_for_comparative_sentence_analysis/ED_GAT_model/result/comp_GAT/uncased_bert_GAT_stack_batch32"
    parent_folder = "/home/nianzu/python_project/comparative_sentence_analysis/graph_network_for_comparative_sentence_analysis/ED_GAT_model/result/comp_GAT/uncased_bert_GAT_stack_new_run"
    file_name = "Bert-GAT-stack-num-layer-tuning"

    logger.info("\n\n================{}======================".format(file_name))

    analyzer = ResultAnalyzerForParameterTuning(metrics_care_about,
                                                best_index_name,
                                                tuned_parameter='stack_layer_num',
                                                parent_folder_for_all_results=parent_folder,
                                                x_axis_label="Number of Stacked GAT Layers",
                                                y_axis_label=metrics_care_about,
                                                fig_file_name=[file_name + ".pdf", file_name + ".png"])
    return analyzer.file_handler_for_parameter_analysis()


def draw_stack_GAT_glove_bert_layer_num_together(output_dir):
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    # endif

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # adjust the figure
    # plot dimensions
    w = 6.0  # 3.95 # 6.0 # 7.5
    h = 4.7
    plt.figure(figsize=(w, h))
    font_size = 20
    plt.rc('xtick', labelsize=font_size)
    plt.rc('ytick', labelsize=font_size)
    plt.rcParams.update({'font.size': font_size, 'font.family': 'sans-serif', 'font.sans-serif': 'Tahoma'})

    # plot
    plt_bert_layer_num_list, plt_bert_test_score_list = bert_GAT_stack_layer_num_tuning()
    plt_glove_layer_num_list, plt_glove_test_score_list = glove_GAT_stack_layer_num_tuning()

    plt.plot(plt_bert_layer_num_list, plt_bert_test_score_list, marker='o', label="ED-GAT$_{BERT}$", linestyle="-", color="#1f77b4")
    plt.plot(plt_glove_layer_num_list, plt_glove_test_score_list, marker="*", label="ED-GAT$_{GloVe}$", linestyle="--", color="#ff7f0e")
    plt.legend(loc="lower right")

    # naming the x axis
    plt.xlabel("Number of ED-GAT Stacked Layers")
    # naming the y axis
    plt.ylabel("Micro-F1")

    # save
    # function to show the plot
    dpi_v = 100
    file_name = "glove_bert_ED-GAT_layer_num"

    plt.savefig(os.path.join(output_dir, "{}_font{}.png").format(file_name, str(font_size)), dpi=dpi_v,
                bbox_inches='tight')
    plt.savefig(os.path.join(output_dir, "{}_font{}.pdf").format(file_name, str(font_size)), dpi=dpi_v,
                bbox_inches='tight')
    # giving a title to my graph

    # plt.show()


if __name__ == '__main__':
    # original_bert_base_uncased_num_layer_tuning()
    # glove_lr_tuning()
    # glove_num_layer()
    # glove_dropout_att()
    # glove_dropout_embed()
    # draw_GAT_glove_bert_layer_num_together()
    # glove_GAT_stack_lr_tuning()
    # bert_GAT_stack_lr_tuning()

    #draw_stack_GAT_glove_bert_layer_num_together("./new_original_figure_output")
    #original_bert_base_uncased_num_layer_tuning()
    bert_GAT_stack_layer_num_tuning()
