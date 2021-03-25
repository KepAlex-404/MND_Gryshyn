from script import Experiment

if __name__ == '__main__':
    exp = Experiment([[-5, 15], [- 15, 35], [15, 30]])
    exp.show_info = True
    exp.make_experiment()
