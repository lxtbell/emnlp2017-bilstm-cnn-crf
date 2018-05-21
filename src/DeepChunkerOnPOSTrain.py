import sys
from pathlib import Path

import ExperimentUtil as Util
from ExperimentUtil import WordEmbedding, CharEmbedding
from neuralnets.BiLSTM import BiLSTM
from util.preprocessing import perpareDataset, loadDatasetPickle

INTERMEDIATE_REPORT_FOLDER = Util.get_working_folder() / "results/DeepChunkerOnPOS"

WORKING_DATA_FOLDER = Util.get_working_data_folder() / "chunk"
MODEL_FOLDER = Util.get_working_folder() / "models/DeepChunkerOnPOS"
REPORT_FOLDER = Util.get_project_folder() / "reports/DeepChunkerOnPOS"


def train(data_path: Path, model_path: Path, word_embedding: WordEmbedding, char_embedding: CharEmbedding):
    Util.setup_logger()

    data_set = Util.relative_to(data_path, Util.get_working_data_folder())
    config_summary = "_" + word_embedding.name + "_" + char_embedding.name
    model_file = str(model_path) + config_summary + ".h5"  # "_[DevScore]_[TestScore]_[Epoch].h5"
    results_file = str(model_path) + config_summary + ".csv"

    data_sets = {
        data_set: {
            'columns': {0: 'tokens', 1: 'POS', 2: 'chunk_BIO'},
            'label': 'chunk_BIO',
            'evaluate': True,
            'commentSymbol': None
        }
    }
    pickle_file = perpareDataset(word_embedding.value, data_sets)

    embeddings, mappings, data = loadDatasetPickle(pickle_file)
    params = {'classifier': ['CRF'], 'LSTM-Size': [100, 100], 'dropout': (0.25, 0.25), 'charEmbeddings': char_embedding.value, 'featureNames': ['tokens', 'casing', 'POS']}
    model = BiLSTM(params)
    model.setMappings(mappings, embeddings)
    model.setDataset(data_sets, data)
    model.storeResults(results_file)
    model.modelSavePath = model_file
    model.fit(epochs=50)

    with open(results_file, "r", encoding=Util.UTF_8) as f:
        lines = f.readlines()
        columns = lines[-1].strip().split("\t")
        return model_path, list(map(float, columns[5:8]))


def run(data_folder: str, word_embedding: WordEmbedding, char_embedding: CharEmbedding, *, num_runs: int = 1, processes: int = 1):
    for run_id in range(num_runs):
        Util.train_all(train, WORKING_DATA_FOLDER / data_folder, MODEL_FOLDER / data_folder, REPORT_FOLDER / data_folder, run_id, word_embedding, char_embedding, processes=processes)


def main():
    params = Util.Params(sys.argv)
    num_processes = params.num_processes
    num_runs = params.num_runs

    if params.run_task(1):
        run("masc-nyt", WordEmbedding.KOMNINOS, CharEmbedding.LSTM, num_runs=num_runs, processes=num_processes)

    if params.run_task(2):
        run("masc-twitter", WordEmbedding.KOMNINOS, CharEmbedding.LSTM, num_runs=num_runs, processes=num_processes)

    if params.run_task(3):
        run("ritter", WordEmbedding.KOMNINOS, CharEmbedding.LSTM, num_runs=num_runs, processes=num_processes)

    if params.run_task(4):
        run("masc-newspaper", WordEmbedding.KOMNINOS, CharEmbedding.LSTM, num_runs=num_runs, processes=num_processes)

    if params.run_task(5):
        run("conll2000", WordEmbedding.KOMNINOS, CharEmbedding.LSTM, num_runs=num_runs, processes=num_processes)

    if params.run_task(6):
        run("conll2003", WordEmbedding.KOMNINOS, CharEmbedding.LSTM, num_runs=num_runs, processes=num_processes)


if __name__ == "__main__":
    main()
