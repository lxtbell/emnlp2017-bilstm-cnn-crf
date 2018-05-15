import itertools
import multiprocessing
import re
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

import ChunkerOnPOSPrepare as Prepare
import ExperimentUtil as Util
from neuralnets.BiLSTM import BiLSTM
from util.preprocessing import perpareDataset, loadDatasetPickle

WORKING_DATA_DIRECTORY = Util.get_working_folder() / "data"
REPORT_DIRECTORY = Util.get_project_folder() / "reports/DeepChunkerOnPOS"


def train(data_file: str, embedding_file: str):
    Util.setup_logger()

    data_sets = {
        data_file: {
            'columns': {0: 'tokens', 1: 'POS', 2: 'chunk_BIO'},
            'label': 'chunk_BIO',
            'evaluate': True,
            'commentSymbol': None
        }
    }
    pickle_file = perpareDataset(embedding_file, data_sets)

    embeddings, mappings, data = loadDatasetPickle(pickle_file)
    results_file = 'results/' + data_file + '_result.csv'
    params = {'classifier': ['CRF'], 'LSTM-Size': [100, 100], 'dropout': (0.25, 0.25), 'charEmbeddings': 'LSTM'}

    model = BiLSTM(params)
    model.setMappings(mappings, embeddings)
    model.setDataset(data_sets, data)
    model.storeResults(results_file)
    model.modelSavePath = "models/[ModelName]_[Epoch].h5"  # "models/[ModelName]_[DevScore]_[TestScore]_[Epoch].h5"
    model.fit(epochs=50)

    with open(results_file, "r") as f:
        lines = f.readlines()
        columns = lines[-1].strip().split("\t")
        return data_file, list(map(float, columns[5:8]))


def train_all(data_folder: str, report_folder: Path, embedding_file: str = 'komninos_english_embeddings.gz', processes: int = 1):
    data_folder_path = Prepare.EXPERIMENT_DATA_DIRECTORY / data_folder
    works = [(str(file.relative_to(WORKING_DATA_DIRECTORY)).replace("\\", "/"), embedding_file) for file in data_folder_path.glob("**/*_fold*")]
    if processes <= 1:
        results = list(itertools.starmap(train, works))
    else:
        with multiprocessing.Pool(processes=processes) as pool:
            results = list(pool.starmap(train, works))

    results_grouped = defaultdict(list)
    for file_name, file_result in results:
        results_grouped[re.sub(r"_fold.*$", "", file_name)].append(file_result)

    for file_name, file_results in results_grouped.items():
        np_file_results = np.array(file_results)
        avg = np.mean(np_file_results, axis=0)
        std = np.std(np_file_results, axis=0)

        report_folder.mkdir(parents=True, exist_ok=True)
        with report_folder.joinpath(Util.replace_slash(file_name) + ".txt").open("w") as f:
            f.write("P\tR\tF1\n")
            Util.print_floats(f, avg * 100)
            Util.print_floats(f, std * 100)
            Util.print_newline(f)
            for result in np_file_results:
                Util.print_floats(f, result * 100)


def main():
    params = Util.Params(sys.argv)
    num_processes = params.num_processes

    train_all("masc-nyt/chunk", REPORT_DIRECTORY, processes=num_processes)
    train_all("masc-twitter/chunk", REPORT_DIRECTORY, processes=num_processes)
    train_all("ritter/chunk", REPORT_DIRECTORY, processes=num_processes)


if __name__ == "__main__":
    main()
