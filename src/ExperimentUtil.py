import itertools
import logging
import multiprocessing
import re
import shutil
import sys
from collections import defaultdict
from enum import Enum
from pathlib import Path
from typing import List, Any

import numpy as np

from util.BIOF1Validation import compute_precision

UTF_8 = "utf-8"


def get_project_folder():
    return Path.home() / "projects/nlp-study-2018"


def get_working_folder():
    return Path.cwd()


def get_working_data_folder():
    return get_working_folder() / "data"


def read_column_format_sentences(file: Path):
    sentences = []
    sentence = []

    with file.open(encoding=UTF_8) as f:
        for line in f:
            line = line.rstrip()
            if not line:
                sentences.append(sentence)
                sentence = []
            else:
                sentence.append(line)

    if sentence:
        sentences.append(sentence)
    return sentences


def write_column_format_sentences(file: Path, sentences: List[List[str]]):
    file.parent.mkdir(parents=True, exist_ok=True)

    with file.open(mode="w", encoding=UTF_8) as f:
        f.write("\n\n".join("\n".join(sentence) for sentence in sentences) + "\n\n")


def make_fold_indices(count: int, num_folds: int):
    split_points = [count * fold // num_folds for fold in range(num_folds + 1)]
    return [(split_points[fold], split_points[fold + 1]) for fold in range(num_folds)]


def make_folds(elements: List[Any], num_folds: int):
    return [elements[l:r] for l, r in make_fold_indices(len(elements), num_folds)]


def split_fold(elements: List[Any], num_folds: int, test_fold: int = 0):
    train, test = [], []
    folds = make_folds(elements, num_folds)

    for fold in range(num_folds):
        if fold != test_fold:
            train += folds[fold]
        else:
            test += folds[fold]

    return train, test


def setup_logger():
    logging_level = logging.INFO
    logger = logging.getLogger()

    if not logger.isEnabledFor(logging_level):
        logger.setLevel(logging_level)

        ch = logging.StreamHandler(sys.stdout)
        ch.setLevel(logging_level)
        formatter = logging.Formatter('%(message)s')
        ch.setFormatter(formatter)
        logger.addHandler(ch)

    return logger


class Params:
    def __init__(self, argv):
        tasks = argv[1] if len(argv) > 1 else "1234567890"
        self.tasks = set(ord(c) - ord('0') if c != '0' else 10 for c in tasks)
        self.num_processes = int(argv[2]) if len(argv) > 2 else 1
        self.num_runs = int(argv[3]) if len(argv) > 3 else 1

    def run_task(self, task_id):
        return task_id in self.tasks


def print_floats(f, floats):
    f.write("\t".join("{:.3f}".format(n) for n in floats) + "\n")


def print_newline(f):
    f.write("\n")


def replace_slash(file_name):
    return re.sub(r"[\\/]", "__", file_name)


def relative_to(path: Path, reference: Path):
    return str(path.relative_to(reference)).replace("\\", "/")


def compute_f1(correct, predicted):
    precision = compute_precision(predicted, correct)
    recall = compute_precision(correct, predicted)
    f1 = 0 if precision + recall <= 0 else 2.0 * precision * recall / (precision + recall)
    return f1, precision, recall


class WordEmbedding(Enum):
    NONE = "embeddings/dummy.txt"
    KOMNINOS = "komninos_english_embeddings.gz"


class CharEmbedding(Enum):
    NONE = None
    CNN = "CNN"
    LSTM = "LSTM"


def relativize(folder: Path, file: Path):
    return file.resolve().relative_to(folder.resolve())


def rebase(old_folder: Path, new_folder: Path, file: Path):
    return new_folder.resolve().joinpath(file.resolve().relative_to(old_folder.resolve()))


def mkdirs_for(file: Path):
    file.parent.mkdir(parents=True, exist_ok=True)


def convert_folder(source, target, single_fold=False):
    if single_fold:
        files = sorted([str(file.relative_to(source)) for file in source.glob("**/*.conll")])
    else:
        files = sorted([re.sub(r"\.test$", "", str(file.relative_to(source))) for file in source.glob("**/*.test")])
    for file in files:
        target_folder = target.joinpath(file)
        target_folder.mkdir(parents=True, exist_ok=True)

        if not single_fold:
            training_data = read_column_format_sentences(source.joinpath(file + ".train"))
            training_set, dev_set = split_fold(training_data, num_folds=5)
            write_column_format_sentences(target_folder.joinpath("train.txt"), training_set)
            write_column_format_sentences(target_folder.joinpath("dev.txt"), dev_set)
            shutil.copyfile(str(source.joinpath(file + ".test")), target_folder.joinpath("test.txt"))
        else:
            training_data = read_column_format_sentences(source.joinpath(file))
            training_set, dev_set = split_fold(training_data, num_folds=5)
            write_column_format_sentences(target_folder.joinpath("train.txt"), training_set)
            write_column_format_sentences(target_folder.joinpath("dev.txt"), dev_set)
            target_folder.joinpath("test.txt").touch()


def train_all(train, data_folder_path: Path, model_folder_path: Path, report_folder_path: Path, run_id: int, word_embedding: WordEmbedding, char_embedding: CharEmbedding, *, processes: int = 1):
    files = [file.parent for file in data_folder_path.glob("**/train.txt")]
    works = [(file, rebase(data_folder_path, model_folder_path, file.with_name(file.name + "_run{}".format(run_id))), word_embedding, char_embedding)
             for file in files]
    if processes <= 1:
        results = list(itertools.starmap(train, works))
    else:
        with multiprocessing.Pool(processes=processes) as pool:
            results = list(pool.starmap(train, works))

    results_grouped = defaultdict(list)
    for model_path, file_result in results:
        results_grouped[re.sub(r"_fold\d*", "", relative_to(model_path, model_folder_path))].append(file_result)

    for file_name, file_results in results_grouped.items():
        np_file_results = np.array(file_results)
        avg = np.mean(np_file_results, axis=0)
        std = np.std(np_file_results, axis=0)

        report_file = report_folder_path.joinpath(file_name)
        report_file = report_file.parent.parent / (report_file.parent.name + "__" + report_file.name + ".txt")
        report_file.parent.mkdir(parents=True, exist_ok=True)
        with report_file.open("w", encoding=UTF_8) as f:
            f.write("P\tR\tF1\n")
            print_floats(f, avg * 100)
            print_floats(f, std * 100)
            print_newline(f)
            for result in np_file_results:
                print_floats(f, result * 100)
