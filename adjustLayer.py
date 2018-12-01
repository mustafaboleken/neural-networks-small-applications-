#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
import math
import numpy as np
import pandas as pd
from PyQt5.QtWidgets import *
from keras.layers import Dense
from keras.models import Sequential
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas

# set seed
np.random.seed(7)

# import data set
df = pd.read_csv('./passengers.csv', sep=';', parse_dates=True, index_col=0)
data = df.values

# float32 data type for keras
data = data.astype('float32')

# slice the data
train = data[0:120, :]   # length 120
test = data[120:, :]     # length 24

# train2 & test2 for split the train to two parts
# for finding best number of hidden layer
train2 = data[0:96, :]   # length 96
test2 = data[96:120, :]     # length 24


def prepare_data(data, lags=1):
    # Create lagged data from an input time series
    x, y = [], []
    for row in range(len(data) - lags - 1):
        a = data[row:(row + lags), 0]
        x.append(a)
        y.append(data[row + lags, 0])
    return np.array(x), np.array(y)


lags = 1

# prepare the data
X_train, y_train = prepare_data(train, lags)
X_test, y_test = prepare_data(test, lags)
y_true = y_test

# prepare the data for test on train part
X_train2, y_train2 = prepare_data(train2, lags)
X_test2, y_test2 = prepare_data(test2, lags)
y_true2 = y_test2


class MyWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.setup_ui()

    def setup_ui(self):
        self.setGeometry(600, 200, 1200, 600)
        self.setWindowTitle("Hidden Layer Number Adjuster")

        self.lineEdit_0 = QLineEdit()
        self.lineEdit_1 = QLineEdit()
        self.label_0 = QLabel()
        self.label_1 = QLabel()
        self.label_2 = QLabel()
        self.label_3 = QLabel()

        self.pushButton = QPushButton("Calculate")
        self.pushButton.clicked.connect(self.calculate_button)

        self.progress = QProgressBar(self)
        self.progress.setMaximum(100)

        self.fig = plt.Figure()
        self.canvas = FigureCanvas(self.fig)

        leftLayout = QVBoxLayout()
        leftLayout.addWidget(self.canvas)
        leftLayout.addWidget(self.progress)

        rightLayout = QVBoxLayout()
        rightLayout.addWidget(self.label_0)
        rightLayout.addWidget(self.label_1)
        rightLayout.addWidget(self.lineEdit_0)
        rightLayout.addWidget(self.label_2)
        rightLayout.addWidget(self.lineEdit_1)
        rightLayout.addWidget(self.pushButton)
        rightLayout.addWidget(self.label_3)
        rightLayout.addStretch(1)

        layout = QHBoxLayout()
        layout.addLayout(leftLayout)
        layout.addLayout(rightLayout)
        layout.setStretchFactor(leftLayout, 1)
        layout.setStretchFactor(rightLayout, 0)
        self.label_0.setText("Please enter the range of the\nhidden layers will tested:")
        self.label_1.setText("From")
        self.label_2.setText("to")
        self.label_3.setText("\n")
        self.progress.setValue(0)

        self.setLayout(layout)

    def on_button_click(self):
        self.calc = External()
        self.calc.countChanged.connect(self.on_count_changed)
        self.calc.start()

    def on_count_changed(self, value):
        self.progress.setValue(value)

    def neurel_network(self, num_layer=1):
        mdl0 = Sequential()
        mdl0.add(Dense(num_layer, input_dim=lags, activation='relu'))
        mdl0.add(Dense(1))
        mdl0.compile(loss='mean_squared_error', optimizer='adam')
        return mdl0

    def calculate_button(self):
        global X_train, y_train, X_test, y_test, y_true, X_split, y_split
        global X_train2, y_train2, X_test2, y_test2, y_true2, X_split2, y_split2

        self.progress.setValue(0)
        x = str(self.lineEdit_0.text())
        y = str(self.lineEdit_1.text())

        try:
            x = int(x)
            y = int(y)

        except ValueError:
            x = 1
            y = 3

        rmse_scores = []

        mdl1 = []

        for i in range(0, (y-x+1)):

            print("Testing with #", i+x)
            mdl1.append(self.neurel_network(i+x))
            mdl1[i].fit(X_train2, y_train2, epochs=200, batch_size=2, verbose=0)
            test_predict = mdl1[i].predict(X_test2)
            mse = ((y_test2.reshape(-1, 1) - test_predict.reshape(-1, 1)) ** 2).mean()
            rmse_scores.append(math.sqrt(mse))
            print("Mean Squared Error: {}".format(mse))
            print("\n")
            self.progress.setValue(int(100/(y-x+1))*int(i+1)-10)

        best_layer_number = rmse_scores.index(max(rmse_scores))+x
        self.label_3.setText("\nBest layer number is %d\nIt's RMSE score is %.2f" %
                             (best_layer_number, min(rmse_scores)))

        mdl = self.neurel_network(best_layer_number)
        mdl.fit(X_train, y_train, epochs=200, batch_size=2, verbose=0)

        # generate predictions for training
        train_predict = mdl.predict(X_train)
        test_predict = mdl.predict(X_test)

        self.progress.setValue(100)

        # shift train predictions for plotting
        train_predict_plot = np.empty_like(data)
        train_predict_plot[:, :] = np.nan
        train_predict_plot[lags: len(train_predict) + lags, :] = train_predict

        # shift test predictions for plotting
        test_predict_plot = np.empty_like(data)
        test_predict_plot[:, :] = np.nan
        test_predict_plot[len(train_predict)+(lags*2)+1:len(data)-1, :] = test_predict

        # plot baseline and predictions
        ax = self.fig.add_subplot(111)
        ax.clear()
        mse = ((y_test.reshape(-1, 1) - test_predict.reshape(-1, 1)) ** 2).mean()
        ax.set_title('Prediction quality: {:.2f} MSE ({:.2f} RMSE)'.format(mse, math.sqrt(mse)))
        ax.plot(y_test.reshape(-1, 1), label='Observed', color='#006699')
        ax.plot(test_predict.reshape(-1, 1), label='Prediction', color='#ff0066')
        ax.legend(loc='best')
        ax.grid()

        self.canvas.draw()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MyWindow()
    window.show()
    app.exec_()
