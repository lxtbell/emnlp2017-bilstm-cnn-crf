import itertools
import multiprocessing
import sys

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
    params = {'classifier': ['CRF'], 'LSTM-Size': [100, 100], 'dropout': (0.25, 0.25)}

    model = BiLSTM(params)
    model.setMappings(mappings, embeddings)
    model.setDataset(data_sets, data)
    model.storeResults(results_file)
    model.modelSavePath = "models/[ModelName]_[Epoch].h5"  # "models/[ModelName]_[DevScore]_[TestScore]_[Epoch].h5"
    model.fit(epochs=50)

    with open(results_file, "r") as f:
        lines = f.readlines()
        columns = lines[-1].strip().split("\t")
        return map(float, columns[5:8])


def train_all(data_folder: str, report_file: str, embedding_file: str = 'komninos_english_embeddings.gz', processes: int = 1):
    data_folder_path = Prepare.EXPERIMENT_DATA_DIRECTORY / data_folder
    works = [(str(file.relative_to(WORKING_DATA_DIRECTORY)).replace("\\", "/"), embedding_file) for file in data_folder_path.glob("**/*_fold*")]
    if processes <= 1:
        results = list(itertools.starmap(train, works))
    else:
        with multiprocessing.Pool(processes=processes) as pool:
            results = pool.starmap(train, works)

    results = np.array(results)
    REPORT_DIRECTORY.mkdir(parents=True, exist_ok=True)
    with REPORT_DIRECTORY.joinpath(report_file).open("w") as f:
        f.write("P\tR\tF1\n")
        f.write("\t".join(np.mean(results, axis=0)) + "\n")
        f.write("\t".join(np.std(results, axis=0)) + "\n")
        f.write("\n")
        for result in results:
            f.write("\t".join(result) + "\n")


def main():
    params = Util.Params(sys.argv)
    num_processes = params.num_processes

    train_all("masc-nyt/chunk", "masc-nyt.txt", processes=num_processes)
    train_all("masc-twitter/chunk", "masc-twitter.txt", processes=num_processes)
    train_all("ritter/chunk", "ritter.txt", processes=num_processes)


if __name__ == "__main__":
    main()
