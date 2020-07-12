import os
import json
import matplotlib.pyplot as plt
import logging

parent_folder = "/home/nianzu/python_project/comparative-master/Classification/graph_network_for_comparative_sentence/run_graph_attention_model/result/comp_GAT/original_bert_model_using_original_text_uncased"
#single_folder = "/home/nianzu/python_project/comparative-master/Classification/graph_network_for_comparative_sentence/run_graph_attention_model/result/comp_GAT/original_bert_embed/2019-11-25__12-48__472871"

#single_folder = "/Users/nianzu/Box Sync/PythonProject/comparative-master/Classification/graph_network_for_comparative_sentence/run_graph_attention_model/result/comp_GAT/masked_sent_final_original_bert_large_cased/2019-11-26__15-40__952718"
#single_folder = "/home/nianzu/python_project/comparative-master/Classification/graph_network_for_comparative_sentence/run_graph_attention_model/result/comp_GAT/glove/2019-11-26__22-32__574547"
#single_folder = "/home/nianzu/python_project/comparative-master/Classification/graph_network_for_comparative_sentence/run_graph_attention_model/result/comp_GAT/original_bert_model_using_original_text_uncased/2019-11-27__16-01__819685"

metrics_care_about = "f1_weighted"
best_index_name = "step"

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ResultAnalyzer(object):
    """
    Note that, the steps are:
    1. find the best result on dev dataset and the corresponding step number (current it is based on step num, not epoch num)
    2. find the test dataset result on the corresponding step number
    It is better to use step number, because for different task, the dataset size is different
    """

    def __init__(self, metrics, best_index_name, parent_folder_for_all_results=None):
        """
        :param parent_folder_for_all_results:
        :param metrics:
        :param best_index_name: should it based on epoch or step
        """
        self.parent_folder_for_all_results = parent_folder_for_all_results
        self.metrics = metrics
        self.best_index_name = best_index_name

    def file_handler_for_parent_folder(self):
        dir_list = []
        for dirName, subdirList, fileList in os.walk(self.parent_folder_for_all_results):
            if dirName != self.parent_folder_for_all_results:
                dir_list.append(dirName)
            # endif
        # endfor

        result_list = []
        for dir_n in dir_list:
            train_epoch_num, score, clf_report = self.analyze_single_result(dir_n)
            result_list.append([train_epoch_num, score, clf_report, dir_n])
        # endfro

        result_list = sorted(result_list, key=lambda x: x[0])

        epoch_list = [item[0] for item in result_list]
        score_list = [item[1] for item in result_list]

        self.plot_epoch_score(epoch_list, score_list)
        logger.info(epoch_list)
        logger.info(score_list)

        highest_result = sorted(result_list, key=lambda x: x[1])  # # [epoch_num, score, clf_report]

        best_train_epoch_num = highest_result[-1][0]
        best_score = highest_result[-1][1]
        best_classification_report = highest_result[-1][2]['classification_report']
        best_test_json = highest_result[-1][2]
        dir_n = highest_result[-1][3]

        logger.info("best epoch num: {}".format(best_train_epoch_num))
        logger.info("best {} is: {}".format(self.metrics, best_score))
        logger.info(best_classification_report)
        logger.info(json.dumps(best_test_json))
        logger.info("dir_name: {}".format(dir_n))

        # print(highest_result)  # [epoch_num, score, clf_report]
        # print(highest_result[-1])
        # print(highest_result[-1][-1]["classification_report"])

        filling_form_item = [highest_result[-1][2][self.metrics]] + highest_result[-1][2]["f_None"]
        filling_form_item = [str(item * 100) for item in filling_form_item]
        logger.info(self.metrics + "     " + "f_None")
        logger.info("===================")
        logger.info('\t'.join(filling_form_item))
        logger.info("===================")

        pass

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

        #return total_train_epoch, test_json_obj[self.metrics], test_json_obj
        pass

    def plot_epoch_score(self, x, y):
        # plotting the points
        plt.plot(x, y)

        # naming the x axis
        plt.xlabel('total train epoch')
        # naming the y axis
        plt.ylabel(self.metrics)

        # giving a title to my graph
        plt.title('epoch-' + self.metrics)

        # function to show the plot
        plt.show()

    def analyze_single_result(self, single_result_folder):
        """
        !! step index
        Check for each run, what this the best result
        1. check on dev data, which step index gives the best result
        2. then check the test data, what is the result on this step index
        :return:
        """
        params = None
        with open(os.path.join(single_result_folder, "params.json"), mode="r") as fin:
            params = json.loads(fin.read().strip())
        # endwith

        total_train_epoch = params["num_train_epochs"]

        dev_file_path = os.path.join(single_result_folder, "dev_results.txt")
        best_dev_json = self._analyze_dev_result_file(dev_file_path)
        best_dev_index = best_dev_json[self.best_index_name]

        test_file_path = os.path.join(single_result_folder, "test_results.txt")
        test_json_obj = self._search_test_result_file(test_file_path, best_dev_index)

        return total_train_epoch, test_json_obj[self.metrics], test_json_obj

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


def main():
    analyzer = ResultAnalyzer(metrics_care_about, best_index_name, parent_folder_for_all_results=parent_folder)
    analyzer.file_handler_for_parent_folder()

    #analyzer.analyze_single_folder(single_folder)

    # draw_graph_op = DrawGraph(parent_folder, "step", "f1_weighted")
    # draw_graph_op.file_handler()


if __name__ == '__main__':
    main()
