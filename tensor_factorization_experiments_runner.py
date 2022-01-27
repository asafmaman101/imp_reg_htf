import argparse

import common.utils.logging as logging_utils
from tensor_factorization.experiments.tensor_factorization_sensing_experiment import TensorFactorizationSensingExperiment


def main():
    parser = argparse.ArgumentParser()
    TensorFactorizationSensingExperiment.add_experiment_specific_args(parser)
    args = parser.parse_args()

    experiment = TensorFactorizationSensingExperiment()
    experiment.run(args.__dict__)


if __name__ == "__main__":
    logging_utils.init_console_logging()
    main()
