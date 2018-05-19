import itertools
import multiprocessing
import re
import sys
from collections import defaultdict
from enum import Enum
from pathlib import Path

import numpy as np

import ExperimentUtil as Util
from neuralnets.BiLSTM import BiLSTM
from util.preprocessing import perpareDataset, loadDatasetPickle

WORKING_DATA_FOLDER = Util.get_working_data_folder() / "ner"
MODEL_FOLDER = "models/DeepNEROnPOS/"
INTERMEDIATE_REPORT_FOLDER = "results/DeepNEROnPOS/"
REPORT_FOLDER = Util.get_project_folder() / "reports/DeepNEROnPOS"


class WordEmbedding(Enum):
    NONE = "embeddings/dummy.txt"
    KOMNINOS = "komninos_english_embeddings.gz"


class CharEmbedding(Enum):
    NONE = None
    CNN = "CNN"
    LSTM = "LSTM"


def train(data_path: Path, word_embedding: WordEmbedding, char_embedding: CharEmbedding):
    Util.setup_logger()

    data_set = Util.relative_to(data_path, Util.get_working_data_folder())
    instance_id = Util.relative_to(data_path, WORKING_DATA_FOLDER) + "_" + word_embedding.name + "_" + char_embedding.name
    model_file = MODEL_FOLDER + instance_id + ".h5"  # "models/[ModelName]_[DevScore]_[TestScore]_[Epoch].h5"
    results_file = INTERMEDIATE_REPORT_FOLDER + instance_id + ".csv"

    data_sets = {
        data_set: {
            'columns': {0: 'tokens', 1: 'POS', 2: 'chunk_BIO', 3: 'ner_BIO'},
            'label': 'ner_BIO',
            'evaluate': True,
            'commentSymbol': None
        }
    }
    pickle_file = perpareDataset(word_embedding.value, data_sets)

    embeddings, mappings, data = loadDatasetPickle(pickle_file)
    params = {'classifier': ['CRF'], 'LSTM-Size': [100, 100], 'dropout': (0.25, 0.25), 'charEmbeddings': char_embedding.value}
    model = BiLSTM(params)
    model.setMappings(mappings, embeddings)
    model.setDataset(data_sets, data)
    model.storeResults(results_file)
    model.modelSavePath = model_file
    model.fit(epochs=50)

    with open(results_file, "r") as f:
        lines = f.readlines()
        columns = lines[-1].strip().split("\t")
        return instance_id, list(map(float, columns[5:8]))


def train_all(data_folder: str, word_embedding: WordEmbedding, char_embedding: CharEmbedding, *, processes: int = 1):
    data_folder_path = WORKING_DATA_FOLDER / data_folder
    works = [(file, word_embedding, char_embedding)
             for file in data_folder_path.glob("**/*_fold*")]
    if processes <= 1:
        results = list(itertools.starmap(train, works))
    else:
        with multiprocessing.Pool(processes=processes) as pool:
            results = list(pool.starmap(train, works))

    results_grouped = defaultdict(list)
    for instance_id, file_result in results:
        results_grouped[re.sub(r"_fold[0-9]*", "", instance_id)].append(file_result)

    for file_name, file_results in results_grouped.items():
        np_file_results = np.array(file_results)
        avg = np.mean(np_file_results, axis=0)
        std = np.std(np_file_results, axis=0)

        report_file = REPORT_FOLDER / (file_name + ".txt")
        report_file.parent.mkdir(parents=True, exist_ok=True)
        with report_file.open("w") as f:
            f.write("P\tR\tF1\n")
            Util.print_floats(f, avg * 100)
            Util.print_floats(f, std * 100)
            Util.print_newline(f)
            for result in np_file_results:
                Util.print_floats(f, result * 100)


def main():
    params = Util.Params(sys.argv)
    num_processes = params.num_processes

    if params.run_task(1):
        train_all("masc-nyt", WordEmbedding.KOMNINOS, CharEmbedding.NONE, processes=num_processes)

    if params.run_task(2):
        train_all("masc-nyt", WordEmbedding.KOMNINOS, CharEmbedding.LSTM, processes=num_processes)

    if params.run_task(4):
        train_all("masc-twitter", WordEmbedding.KOMNINOS, CharEmbedding.NONE, processes=num_processes)

    if params.run_task(5):
        train_all("masc-twitter", WordEmbedding.KOMNINOS, CharEmbedding.LSTM, processes=num_processes)

    if params.run_task(7):
        train_all("ritter", WordEmbedding.KOMNINOS, CharEmbedding.NONE, processes=num_processes)

    if params.run_task(8):
        train_all("ritter", WordEmbedding.KOMNINOS, CharEmbedding.LSTM, processes=num_processes)


if __name__ == "__main__":
    main()
