import os
import sys

sys.path.append(os.path.abspath('..'))

import core
import dynamicals
import utils


def main():
    argv = sys.argv

    directory = argv[1]
    dynamical_name = argv[2].strip()
    repetition_num = argv[3]

    config = core.Config()
    config.load_config(directory, utils.CONFIG_FILENAME)

    if dynamical_name == 'lorenz-96':
        dynamical = dynamicals.Lorenz96(config.X_0.size)
    elif dynamical_name == 'lorenz-63':
        dynamical = dynamicals.Lorenz63()
    else:
        raise ValueError('Unknown dynamical system {}'.format(dynamical_name))

    gp = core.GaussianProcessRegression(dynamical, config)
    gp.run()

    lpmf = core.LaplaceMeanFieldSDE(dynamical, config, gp)
    lpmf.run()

    lpmf.save_result(directory, utils.DATA_FILENAME.format(repetition_num))


if __name__ == "__main__":
    main()
