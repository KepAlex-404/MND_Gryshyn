import logging
import os
import sys
from pycallgraph import PyCallGraph
from pycallgraph.output import GraphvizOutput

logging.basicConfig(filename='app.log', filemode='w', format='%(levelname)s - %(message)s')

try:
    from script import *
except ImportError:
    logging.exception('Import error')
    sys.exit()


if __name__ == '__main__':
    e = Experiment((10, 40), (25, 45), (40, 45))

    try:
        with PyCallGraph(output=GraphvizOutput()):
            e.make_experiment()
    except Exception as e:
        logging.exception(f'{e} in module '
                          f'- {os.path.split(sys.exc_info()[-1].tb_frame.f_code.co_filename)[1]} '
                          f'- at line {sys.exc_info()[-1].tb_lineno}', exc_info=False)
        sys.exit()
