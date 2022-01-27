import argparse

import common.utils.logging as logging_utils
from locality_bias.experiments.pathfinder_experiment import PathfinderExperiment


def main():
    parser = argparse.ArgumentParser()
    PathfinderExperiment.add_experiment_specific_args(parser)
    args = parser.parse_args()

    experiment = PathfinderExperiment()
    experiment.run(args.__dict__)


if __name__ == "__main__":
    logging_utils.init_console_logging()
    main()
