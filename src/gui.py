import sys
from os.path import expanduser

from PyQt5.QtWidgets import QMainWindow, QApplication, QFileDialog, QProgressBar
from PyQt5 import uic, QtGui

Ui_MainWindow, QtBaseClass = uic.loadUiType("mainGui_3A.ui")


class MyApp(QMainWindow, Ui_MainWindow):
    def __init__(self):
        super(MyApp, self).__init__()
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)

        # set logo of app and fixing the screen size
        logo_src = "appLogo.png"
        self.setWindowIcon(QtGui.QIcon(logo_src))
        self.setFixedSize(self.size())

        # initializing variables and setting event listeners
        self.ui.pushButton_inputDirectory.clicked.connect(self.choose_input_directory)
        self.ui.pushButton_outputDirectory.clicked.connect(self.choose_output_directory)
        self.ui.pushButton_CreateAlbum.clicked.connect(self.create_album)
        self.ui.pushButton_inputDirectory.setToolTip('Choose input directory')
        self.ui.pushButton_outputDirectory.setToolTip('Choose output directory')


        # self.progressBar = QProgressBar()
        # self.progressBar.setRange(0, 10000)
        # self.progressBar.setValue(0)
        # self.statusBar().addPermanentWidget(self.progressBar)
        # self.progressBar.show()
        # self.ui.progressbar.setVisible(False)

    def choose_input_directory(self):
        print("Hello1")
        input_dir = QFileDialog.getExistingDirectory(None, 'Select a folder:', expanduser("~"))
        self.ui.lineEdit_inputDirectory.setText(input_dir)

    def choose_output_directory(self):
        print("Hello1")
        input_dir = QFileDialog.getExistingDirectory(None, 'Select a folder:', expanduser("~"))
        self.ui.lineEdit_outputDirectory.setText(input_dir)

    def create_album(self):
        print("Hello2")
        current_input_dir = self.ui.lineEdit_inputDirectory.text()
        current_output_dir = self.ui.lineEdit_outputDirectory.text()
        if current_input_dir != "" and current_output_dir != "":
            quality = self.ui.checkBox_Quality.isChecked()
            launch = self.ui.checkBox_Launch.isChecked()
            print(current_input_dir)
            print(current_output_dir)
            print(quality)
            print(launch)
        return 1


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MyApp()
    window.show()
    sys.exit(app.exec_())
