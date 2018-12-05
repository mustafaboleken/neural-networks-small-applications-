#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
import math
import numpy as np
import pandas as pd
from PyQt5.QtWidgets import *
from PyQt5.QtGui import *
from PyQt5.QtCore import Qt, QTimer
from keras.layers import Dense
from keras.models import Sequential
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas

# set seed
np.random.seed(7)

# import data set
df = pd.read_csv('./passengers.csv', sep=';', parse_dates=True, index_col=0)
data = df.values

# using keras often requires the data type float32
data = data.astype('float32')

lags = 1
dataParserFlag = 0

X_split = 0
y_split = 0


class Features:
    def neurel_network(self, num_layer=4, input_dim=1, output_dim=1):
        print("\n*** Creating a new model with #{} hidden layers. ***\n".format(num_layer))
        mdl0 = Sequential()
        mdl0.add(Dense(num_layer, input_dim=input_dim, activation='relu'))
        mdl0.add(Dense(output_dim))
        mdl0.compile(loss='mean_squared_error', optimizer='adam')
        return mdl0

    def data_parser(self, train_ratio=85, n_steps_in=4, n_steps_out=3):
        print("Parsing data with the percent of {}".format(train_ratio))
        window.x = int((144/100)*train_ratio)
        train = data[0:window.x, :]

        global X_split, y_split
        X_split, y_split = self.split_sequence(train, n_steps_in, n_steps_out)

    def split_sequence(self, data, n_steps_in=4, n_steps_out=3):
        X, y = list(), list()
        for i in range(len(data)):
            end_ix = i + n_steps_in
            out_end_ix = end_ix + n_steps_out
            if out_end_ix > len(data):
                break
            seq_x, seq_y = data[i:end_ix, 0], data[end_ix:out_end_ix, 0]
            X.append(seq_x)
            y.append(seq_y)
        return np.array(X), np.array(y)


feature = Features()


class MyWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.setup_ui()

    def setup_ui(self):
        self.setGeometry(600, 200, 1200, 600)
        self.setWindowTitle("Time Series Forecasting with MLP Tool")

        self.lineEdit = QLineEdit()

        self.label = QLabel()

        self.slider = QSlider(Qt.Horizontal)
        self.slider.setFocusPolicy(Qt.StrongFocus)
        self.slider.setTickPosition(QSlider.TicksBothSides)
        self.slider.setTickInterval(10)
        self.slider.setSingleStep(1)

        self.pushButton = QPushButton("Forecast")
        self.pushButton.clicked.connect(self.forecast_button)

        self.x = 0
        self.ratio = 85
        self.layerNumber = 32
        self.fig = plt.Figure()
        self.canvas = FigureCanvas(self.fig)

        leftLayout = QVBoxLayout()
        leftLayout.addWidget(self.canvas)

        rightLayout = QVBoxLayout()
        rightLayout.addWidget(self.label)
        rightLayout.addWidget(self.lineEdit)
        rightLayout.addWidget(self.pushButton)
        rightLayout.addStretch(1)

        layout = QHBoxLayout()
        layout.addLayout(leftLayout)
        layout.addLayout(rightLayout)
        layout.setStretchFactor(leftLayout, 1)
        layout.setStretchFactor(rightLayout, 0)
        self.label.setText("N-step Forecasting\nwith multiple output")

        self.setLayout(layout)

    def forecast_button(self):
        global data, X_split, y_split

        n_step = str(self.lineEdit.text())
        x_input = np.array([], dtype='float32')
        expected_output = np.array([], dtype='float32')

        try:
            n_step = int(n_step)

        except ValueError:
            n_step = 3

        feature.data_parser(self.ratio, n_step+1, n_step)

        # make multi-step prediction with multiple output
        mdl = feature.neurel_network(self.layerNumber, n_step+1, n_step)
        mdl.fit(X_split, y_split, epochs=200, batch_size=2, verbose=0)

        x_input = np.append(x_input, data[window.x-(n_step+1):window.x, 0])
        expected_output = np.append(expected_output, data[window.x:window.x+n_step, 0])
        x_input = x_input.reshape((1, n_step+1))
        forecast_result = mdl.predict(x_input)

        # plot baseline and predictions
        ax = self.fig.add_subplot(111)
        ax.clear()
        mse = ((expected_output.reshape(-1, 1) - forecast_result.reshape(-1, 1)) ** 2).mean()
        ax.set_title('Forecasting quality: {:.2f} MSE ({:.2f} RMSE)'.format(mse, math.sqrt(mse)))
        ax.plot(expected_output.reshape(-1, 1), label='Observed', color='#006699')
        ax.plot(forecast_result.reshape(-1, 1), label='Forecast', color='#ff0066')
        ax.legend(loc='best');
        ax.grid()

        self.canvas.draw()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MyWindow()
    window.show()
    app.exec_()
