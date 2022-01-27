import argparse

import common.utils.logging as logging_utils
from matrix_factorization.experiments.dln_matrix_sensing_experiment import DLNMatrixSensingExperiment


def main():
    parser = argparse.ArgumentParser()
    DLNMatrixSensingExperiment.add_experiment_specific_args(parser)
    args = parser.parse_args()

    experiment = DLNMatrixSensingExperiment()
    experiment.run(args.__dict__)


if __name__ == "__main__":
    logging_utils.init_console_logging()
    main()
