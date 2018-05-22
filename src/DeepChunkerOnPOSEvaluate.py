from __future__ import print_function

import sys
from pathlib import Path

import ExperimentUtil as Util
from ChunkerOnPOSPrepare import WORKING_DATA_FOLDER
from DeepChunkerOnPOSTrain import MODEL_FOLDER, REPORT_FOLDER
from neuralnets.BiLSTM import BiLSTM
from util.preprocessing import readCoNLL, createMatrices, addCharInformation, addCasingInformation

PREDICTION_FOLDER = Util.get_project_folder() / "data/predictions/DeepChunkerOnPOS"


def test(data_path: Path, model_path: Path, prediction_path: Path):
    Util.setup_logger()

    print("Evaluating {} on {}.".format(model_path, data_path))

    input_columns = {0: 'tokens', 1: 'POS', 2: 'chunk_BIO'}
    label_column = 'chunk_BIO'

    sentences = readCoNLL(str(data_path), input_columns)
    addCharInformation(sentences)
    addCasingInformation(sentences)
    sentences_labels = [sentence[label_column] for sentence in sentences]

    lstm_model = BiLSTM.loadModel(str(model_path))
    data_matrix = createMatrices(sentences, lstm_model.mappings, True)

    tags = lstm_model.tagSentences(data_matrix)

    for model_name in sorted(tags.keys()):
        Util.mkdirs_for(prediction_path)
        with Util.file_open(prediction_path, "w") as f:
            for sentence, sentence_tag in zip(sentences, tags[model_name]):
                for token, pos, chunk in zip(*[sentence[input_columns[i]] for i in (0, 1)], sentence_tag):
                    print("{} {} {}".format(token, pos, chunk), file=f)
                print(file=f)

        return model_path.with_suffix(""), Util.compute_f1(tags[model_name], sentences_labels)


def run(data_folder: str, model_folder: str, report_folder: str, *, num_runs: int = 1, processes: int = 1):
    for run_id in range(-1, num_runs):
        Util.evaluate_all(test, WORKING_DATA_FOLDER / data_folder, MODEL_FOLDER / model_folder, PREDICTION_FOLDER / report_folder, REPORT_FOLDER / report_folder, run_id=run_id, processes=processes)


def main():
    params = Util.Params(sys.argv)
    num_processes = params.num_processes
    num_runs = params.num_runs

    if params.run_task(1):
        run("masc-twitter", "masc-newspaper", "masc-twitter-on-newspaper-model", num_runs=num_runs, processes=num_processes)

    if params.run_task(2):
        run("masc-newspaper", "masc-newspaper", "masc-newspaper-on-newspaper-model", num_runs=num_runs, processes=num_processes)

    if params.run_task(3):
        run("masc-twitter", "masc-twitter", "masc-twitter-on-twitter-model", num_runs=num_runs, processes=num_processes)

    if params.run_task(4):
        run("ritter", "ritter", "ritter-on-ritter-model", num_runs=num_runs, processes=num_processes)


if __name__ == "__main__":
    main()
