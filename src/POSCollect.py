from __future__ import print_function

import numpy as np
import sys
from collections import defaultdict

import ExperimentUtil as Util
import DeepChunkerOnPOSTrain
import DeepNEROnPOSTrain


CHUNKER_TASKS = ["no-pos", "gold-pos", "cogcomp-pos", "nltk-pos", "opennlp-pos", "stanford-pos", "ritter-pos", "cmu-pos"]
NER_TASKS = ["no-pos.gold-chunk", "gold-pos.gold-chunk", "cogcomp-pos.gold-chunk", "nltk-pos.gold-chunk",
             "opennlp-pos.gold-chunk", "stanford-pos.gold-chunk", "ritter-pos.gold-chunk", "cmu-pos.gold-chunk"]


def read_result(report_path):
    with report_path.open("r", encoding=Util.UTF_8) as f:
        lines = f.readlines()
        columns = lines[1].strip().split("\t")
        return list(map(float, columns))


def run(report_folder, tasks):
    results = defaultdict(dict)
    for report_path in report_folder.glob("**/*.txt"):
        task_name = Util.relative_to(report_path.parent, report_folder)
        run_name = report_path.name
        results[task_name][run_name] = read_result(report_path)

    with report_folder.joinpath(report_folder.name + ".csv").open("w", encoding=Util.UTF_8) as f:
        for task_name in tasks:
            scores = [results[task_name][run_name] for run_name in sorted(results[task_name].keys())]
            np_scores = np.array(scores)
            avg_score = np.mean(np_scores, axis=0)

            f.write("{},".format(task_name))
            f.write("{:.3f},{:.3f},{:.3f},,".format(avg_score[1], avg_score[2], avg_score[0]))
            for score in scores:
                f.write("{:.3f},{:.3f},{:.3f},".format(score[1], score[2], score[0]))
            f.write("\n")


def main():
    params = Util.Params(sys.argv)

    if params.run_task(1):
        run(DeepChunkerOnPOSTrain.REPORT_FOLDER / "masc-newspaper", CHUNKER_TASKS)

    if params.run_task(2):
        run(DeepChunkerOnPOSTrain.REPORT_FOLDER / "masc-twitter", CHUNKER_TASKS)

    if params.run_task(3):
        run(DeepChunkerOnPOSTrain.REPORT_FOLDER / "ritter", CHUNKER_TASKS)

    if params.run_task(4):
        run(DeepChunkerOnPOSTrain.REPORT_FOLDER / "masc-twitter-on-newspaper-model", CHUNKER_TASKS)
        run(DeepChunkerOnPOSTrain.REPORT_FOLDER / "masc-newspaper-on-newspaper-model", CHUNKER_TASKS)
        run(DeepChunkerOnPOSTrain.REPORT_FOLDER / "masc-twitter-on-twitter-model", CHUNKER_TASKS)
        run(DeepChunkerOnPOSTrain.REPORT_FOLDER / "ritter-on-ritter-model", CHUNKER_TASKS)

    if params.run_task(6):
        run(DeepNEROnPOSTrain.REPORT_FOLDER / "masc-newspaper", NER_TASKS)

    if params.run_task(7):
        run(DeepNEROnPOSTrain.REPORT_FOLDER / "masc-twitter", NER_TASKS)

    if params.run_task(8):
        run(DeepNEROnPOSTrain.REPORT_FOLDER / "ritter", NER_TASKS)

    if params.run_task(9):
        run(DeepNEROnPOSTrain.REPORT_FOLDER / "masc-twitter-on-newspaper-model", NER_TASKS)
        run(DeepNEROnPOSTrain.REPORT_FOLDER / "masc-newspaper-on-newspaper-model", NER_TASKS)
        run(DeepNEROnPOSTrain.REPORT_FOLDER / "masc-twitter-on-twitter-model", NER_TASKS)
        run(DeepNEROnPOSTrain.REPORT_FOLDER / "ritter-on-ritter-model", NER_TASKS)


if __name__ == "__main__":
    main()
