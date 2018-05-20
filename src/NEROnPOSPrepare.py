import ExperimentUtil as Util

SOURCE_DATA_DIRECTORY = Util.get_project_folder() / "data/training/NEROnPOS"
EXPERIMENT_DATA_DIRECTORY = Util.get_working_folder() / "data/ner"


def main():
    Util.convert_folder(SOURCE_DATA_DIRECTORY / "masc-twitter-combined", EXPERIMENT_DATA_DIRECTORY / "masc-twitter")
    Util.convert_folder(SOURCE_DATA_DIRECTORY / "masc-nyt-combined", EXPERIMENT_DATA_DIRECTORY / "masc-nyt")
    Util.convert_folder(SOURCE_DATA_DIRECTORY / "ritter", EXPERIMENT_DATA_DIRECTORY / "ritter")

    Util.convert_folder(SOURCE_DATA_DIRECTORY / "masc-newspaper-combined", EXPERIMENT_DATA_DIRECTORY / "masc-newspaper")

    Util.convert_folder(SOURCE_DATA_DIRECTORY / "conll2003", EXPERIMENT_DATA_DIRECTORY / "conll2003", single_fold=True)


if __name__ == "__main__":
    main()
