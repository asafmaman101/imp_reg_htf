import argparse

import common.utils.logging as logging_utils
from tensor_factorization.experiments.hierarchical_tensor_factorization_sensing_experiment import HierarchicalTensorFactorizationSensingExperiment


def main():
    parser = argparse.ArgumentParser()
    HierarchicalTensorFactorizationSensingExperiment.add_experiment_specific_args(parser)
    args = parser.parse_args()

    experiment = HierarchicalTensorFactorizationSensingExperiment()
    experiment.run(args.__dict__)


if __name__ == "__main__":
    logging_utils.init_console_logging()
    main()
