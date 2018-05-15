import re
import shutil

import ExperimentUtil as Util

SOURCE_DATA_DIRECTORY = Util.get_project_folder() / "data/data"
EXPERIMENT_DATA_DIRECTORY = Util.get_working_folder() / "data/experiment"


def convert_folder(source, target):
    files = sorted([re.sub(r"\.test$", "", str(file.relative_to(source))) for file in source.glob("**/*.test")])
    for file in files:
        target_folder = target.joinpath(file)
        target_folder.mkdir(parents=True, exist_ok=True)

        shutil.copyfile(str(source.joinpath(file + ".test")), target_folder.joinpath("test.txt"))
        training_data = Util.read_column_format_sentences(source.joinpath(file + ".train"))
        training_set, dev_set = Util.split_fold(training_data, num_folds=5)
        Util.write_column_format_sentences(target_folder.joinpath("train.txt"), training_set)
        Util.write_column_format_sentences(target_folder.joinpath("dev.txt"), dev_set)


def main():
    convert_folder(SOURCE_DATA_DIRECTORY / "masc-twitter/pos-combined/gold-pos.gold-chunk", EXPERIMENT_DATA_DIRECTORY / "masc-twitter/chunk")
    convert_folder(SOURCE_DATA_DIRECTORY / "masc-nyt/pos-combined/gold-pos.gold-chunk", EXPERIMENT_DATA_DIRECTORY / "masc-nyt/chunk")
    convert_folder(SOURCE_DATA_DIRECTORY / "ritter/pos/gold-pos.gold-chunk", EXPERIMENT_DATA_DIRECTORY / "ritter/chunk")


if __name__ == "__main__":
    main()
