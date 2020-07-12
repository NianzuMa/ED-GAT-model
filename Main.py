from abc import ABC, ABCMeta, abstractmethod
import shutil


class Main(ABC):

    @abstractmethod
    def argument_parser(self):
        raise NotImplementedError

    @abstractmethod
    def set_seed(self):
        raise NotImplementedError

    @abstractmethod
    def train(self):
        """
        No checks are done on how many arguments concrete implementations take
        :return:
        """
        raise NotImplementedError

    @abstractmethod
    def evaluate(self):
        """
        evaluation for train, dev, test file
        :return:
        """
        raise NotImplementedError

    def write_test_prediction(self):
        raise NotImplementedError

    @abstractmethod
    def load_and_cache_examples(self):
        raise NotImplementedError


    @abstractmethod
    def move_log_file_to_output_directory(self, output_dir):
        """
            The output file name is at the bash file.
            Redirect all the output to the log file.
            The reason do not use file handler of logging module is that, it could only redirect the message from
            user write script to file. For the logging information in API/pacakges, it will not be blocked and not
            shown in the stdout.
            Using redirect in linux console will redirect all the information printed out on the screen to the file
            not matter where it is from.
            :return:
            """
        shutil.move("./log.txt", output_dir)
        pass

    @abstractmethod
    def run_app(self):
        raise NotImplementedError
