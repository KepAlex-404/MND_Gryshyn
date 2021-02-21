import sys

from PyQt5.QtWidgets import QTableWidgetItem, QDialog
from scripts import *

from lab01.res.gui import *

app = QtWidgets.QApplication(sys.argv)
menu = QtWidgets.QMainWindow()
ui = Ui_MainWindow()
ui.setupUi(menu)
menu.show()


def starter():
    try:
        res = main(ui.dial.value())
        # pprint(res)
        for row_index, row_item in enumerate(res[0]):
            for col_index, col_item in enumerate(row_item):
                ui.table.setItem(row_index, col_index + 1, QTableWidgetItem(str(col_item)))

        ui.lineEdit.setText(str(res[1]))
        ui.lineEdit_2.setText(str(res[2]))
    except Exception as e:
        print(e)
        dlg = QDialog()
        dlg.setWindowTitle("ERROR")
        dlg.setFixedSize(175, 50)
        dlg.exec_()


ui.start_bt.clicked.connect(starter)

sys.exit(app.exec_())
