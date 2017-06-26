import sys
from os.path import expanduser

from PyQt5.QtWidgets import QMainWindow, QApplication, QFileDialog
from PyQt5 import uic, QtGui

Ui_MainWindow, QtBaseClass = uic.loadUiType("mainGui_3A.ui")


class MyApp(QMainWindow, Ui_MainWindow):
    def __init__(self):
        super(MyApp, self).__init__()
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)
        logo_src = "appLogo.png"
        self.setWindowIcon(QtGui.QIcon(logo_src))
        self.ui.pushButton_Directory.clicked.connect(self.choose_directory)
        self.ui.pushButton_CreateAlbum.clicked.connect(self.create_album)
        self.ui.pushButton_Directory.setToolTip('Choose directory')
        self.ui.checkBox_Quality.setToolTip('Include image quality assessment upon selection of representative photos')
        self.ui.checkBox_Launch.setToolTip('Display output album when done')

    def choose_directory(self):
        print("Hello1")
        dir_ = QFileDialog.getExistingDirectory(None, 'Select a folder:', 'C:\\', QtGui.QFileDialog.ShowDirsOnly)
        return 1
        # dir = QtGui.QFileDialog.getExistingDirectory(this, tr("Open Directory"),
        #                                         "/home",
        #                                              QtGui.QFileDialog::ShowDirsOnly
        #                                                      | QFileDialog::DontResolveSymlinks);
        #
        # my_dir = QtGui.QFileDialog.getExistingDirectory(
        #     self,
        #     "Open a folder",
        #     expanduser("~"),
        #     QtGui.QFileDialog.ShowDirsOnly
        # )
        # return 1
        # self.ui.lineEdit_Directory.setText(my_dir)
        # price = int(self.ui.price_box.toPlainText())
        # tax = (self.ui.tax_rate.value())
        # total_price = price + ((tax / 100) * price)
        # total_price_string = "The total price with tax is: " + str(total_price)
        # self.ui.results_window.setText(total_price_string)

    def create_album(self):
        print("Hello2")
        # current_dir = self.ui.lineEdit_Directory.toPlainText()
        # quality = self.ui.checkBox_Quality.value()
        # launch = self.ui.checkBox_Launch.value()
        return 1


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MyApp()
    window.show()
    sys.exit(app.exec_())
