from __future__ import print_function

import sys

import ExperimentUtil as Util
from ChunkerOnPOSPrepare import EXPERIMENT_DATA_DIRECTORY
from DeepChunkerOnPOSTrain import MODEL_FOLDER, REPORT_FOLDER, print_results
from neuralnets.BiLSTM import BiLSTM
from util.preprocessing import readCoNLL, createMatrices, addCharInformation, addCasingInformation

PREDICTION_FOLDER = Util.get_project_folder() / "data/predictions/DeepChunkerOnPOS"


def test(data_path, model_path, prediction_path):
    Util.setup_logger()

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
        with prediction_path.open("w", encoding=Util.UTF_8) as f:
            for sentence, sentence_tag in zip(sentences, tags[model_name]):
                for token, pos, chunk in zip(*[sentence[input_columns[i]] for i in (0, 1)], sentence_tag):
                    print("{} {} {}".format(token, pos, chunk), file=f)
                print(file=f)

        return Util.compute_f1(tags[model_name], sentences_labels)


def run(data_folder, model_folder, prediction_folder, report_folder):
    data_paths = [file for file in data_folder.glob("**/test.txt")]
    model_paths = [file for file in model_folder.glob("**/*.h5")]
    results = [(
        str(Util.relativize(data_folder, data_path.parent)),
        test(data_path, model_path, Util.rebase(data_folder, prediction_folder, data_path.parent)))
        for data_path, model_path in zip(data_paths, model_paths)
    ]
    print_results(results, report_folder)


def main():
    params = Util.Params(sys.argv)

    if params.run_task(1):
        run(EXPERIMENT_DATA_DIRECTORY / "masc-newspaper/gold-pos", MODEL_FOLDER / "masc-newspaper/gold-pos",
            PREDICTION_FOLDER / "masc-newspaper-on-newspaper-model/gold-pos", REPORT_FOLDER / "masc-newspaper-on-newspaper-model/gold-pos")

    if params.run_task(2):
        run(EXPERIMENT_DATA_DIRECTORY / "masc-newspaper/no-pos", MODEL_FOLDER / "masc-newspaper/no-pos",
            PREDICTION_FOLDER / "masc-newspaper-on-newspaper-model/no-pos", REPORT_FOLDER / "masc-newspaper-on-newspaper-model/no-pos")

    if params.run_task(3):
        run(EXPERIMENT_DATA_DIRECTORY / "masc-twitter/gold-pos", MODEL_FOLDER / "masc-newspaper/gold-pos",
            PREDICTION_FOLDER / "masc-twitter-on-newspaper-model/gold-pos", REPORT_FOLDER / "masc-twitter-on-newspaper-model/gold-pos")

    if params.run_task(4):
        run(EXPERIMENT_DATA_DIRECTORY / "masc-twitter/no-pos", MODEL_FOLDER / "masc-newspaper/no-pos",
            PREDICTION_FOLDER / "masc-twitter-on-newspaper-model/no-pos", REPORT_FOLDER / "masc-twitter-on-newspaper-model/no-pos")


if __name__ == "__main__":
    main()
