from lab05.src.Experiment import Experiment


if __name__ == '__main__':
    experiment = Experiment((-9, 7),(-4, 7),(-10, 5))

    experiment.run_experiment(15, 3)