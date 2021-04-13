import logging
import os
import sys
import time
from numpy import average


logging.basicConfig(filename='app.log', filemode='w', format='%(levelname)s - %(message)s')

try:
    from script import *
except ImportError:
    logging.exception('Import error')
    sys.exit()


if __name__ == '__main__':

    try:
        list_of_runs = []
        for _ in range(100):
            start = time.time()
            e = Experiment((10, 40), (25, 45), (40, 45))
            e.make_experiment()
            list_of_runs.append(time.time()-start)
        print('Average time of execute - ', average(list_of_runs))
    except Exception as e:
        logging.exception(f'{e} in module '
                          f'- {os.path.split(sys.exc_info()[-1].tb_frame.f_code.co_filename)[1]} '
                          f'- at line {sys.exc_info()[-1].tb_lineno}', exc_info=False)
        sys.exit()
