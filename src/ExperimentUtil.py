import logging
import re
import sys
from pathlib import Path
from typing import List, Any

UTF_8 = "utf-8"


def get_project_folder():
    return Path.home() / "projects/nlp-study-2018"


def get_working_folder():
    return Path.cwd()


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

    def run_task(self, task_id):
        return task_id in self.tasks


def print_floats(f, floats):
    f.write("\t".join("{:.3f}".format(n) for n in floats) + "\n")


def print_newline(f):
    f.write("\n")


def replace_slash(file_name):
    return re.sub(r"[\\/]", "__", file_name)
