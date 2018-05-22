import ExperimentUtil as Util

SOURCE_DATA_FOLDER = Util.get_project_folder() / "data/training/ChunkerOnPOS"
WORKING_DATA_FOLDER = Util.get_working_folder() / "data/chunk"


def main():
    Util.convert_folder(SOURCE_DATA_FOLDER / "masc-twitter-combined", WORKING_DATA_FOLDER / "masc-twitter")
    Util.convert_folder(SOURCE_DATA_FOLDER / "masc-nyt-combined", WORKING_DATA_FOLDER / "masc-nyt")
    Util.convert_folder(SOURCE_DATA_FOLDER / "ritter", WORKING_DATA_FOLDER / "ritter")

    Util.convert_folder(SOURCE_DATA_FOLDER / "masc-newspaper-combined", WORKING_DATA_FOLDER / "masc-newspaper")

    Util.convert_folder(SOURCE_DATA_FOLDER / "conll2000", WORKING_DATA_FOLDER / "conll2000", single_fold=True)
    Util.convert_folder(SOURCE_DATA_FOLDER / "conll2003", WORKING_DATA_FOLDER / "conll2003", single_fold=True)


if __name__ == "__main__":
    main()
