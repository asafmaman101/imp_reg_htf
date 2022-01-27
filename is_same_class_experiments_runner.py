import argparse

import common.utils.logging as logging_utils
from locality_bias.experiments.is_same_experiment import IsSameExperiment


def main():
    parser = argparse.ArgumentParser()
    IsSameExperiment.add_experiment_specific_args(parser)
    args = parser.parse_args()

    experiment = IsSameExperiment()
    experiment.run(args.__dict__)


if __name__ == "__main__":
    logging_utils.init_console_logging()
    main()
