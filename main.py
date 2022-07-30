
from PIL import Image
import os
import cv2

import torch
import torch.nn as nn
from torchvision import transforms as transforms

import sys
from PyQt5.QtCore import *
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *


def predict(model, path):
    with open(path, 'rb') as f:
        img = cv2.imread(path)
        gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        transform = transforms.ToTensor()
        data = transform(gray_image)
        print(data)
        data = data.unsqueeze(0)  # (28, 28) -> (1, 28, 28) [add batch]

    output = model(data.float())
    print(output)
    result = torch.argmax(output)
    return result


class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.class_num = 10
        self.input_size = 28

        self.block1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=16, kernel_size=5,
                      stride=1, padding=0),  # (1, 28, 28) -> (16, 24, 24)
            nn.BatchNorm2d(16),
            nn.Dropout2d(p=0.1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2))  # (16, 24, 24) -> (16, 12, 12)

        self.block2 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=5,
                      stride=1, padding=0),  # (16, 12, 12) -> (32, 8, 8)
            nn.BatchNorm2d(32),
            nn.Dropout2d(p=0.1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2))  # (32, 8, 8) -> (32, 4, 4)

        self.fc = nn.Linear(32 * 4 * 4, self.class_num)

    def forward(self, x):
        out = self.block1(x)
        out = self.block2(out)
        out = out.view(out.size(0), -1)
        out = self.fc(out)  # (batch, 32 * 4 * 4) -> (batch, 10)

        return out


class Ensemble(nn.Module):
    def __init__(self, models):
        super(Ensemble, self).__init__()
        self.class_num = 10
        self.input_size = 28

        self.models = models
        self.fc = nn.Linear(self.class_num * len(models), self.class_num)

    def forward(self, x):
        outs = [model(x) for model in self.models]
        out = torch.cat(outs, dim=1)
        out = self.fc(out)  # (batch, 32 * 4 * 4) -> (batch, 10)

        return out


class Drawer(QWidget):
    def __init__(self, parent=None):
        QWidget.__init__(self, parent)
        self.setAttribute(Qt.WA_StaticContents)
        h = 500
        w = 500
        self.myPenWidth = 13
        self.myPenColor = Qt.black
        self.image = QImage(w, h, QImage.Format_RGB32)
        self.path = QPainterPath()
        self.clearImage()

    def setPenColor(self, newColor):
        self.myPenColor = newColor

    def setPenWidth(self, newWidth):
        self.myPenWidth = newWidth

    def clearImage(self):
        self.path = QPainterPath()
        self.image.fill(Qt.white)  # switch it to else
        textLabel.setText("Draw!")
        self.update()

    def saveImage(self, fileName, fileFormat, model):
        self.image.save(fileName, fileFormat)
        img = Image.open(fileName)
        new_img = img.resize((28, 28))
        new_img.save(fileName)
        result = predict(model, fileName)
        print(int(result))

        if int(result) == 0:
            textLabel.setText("Your are a BAD GUY!")
        else:
            textLabel.setText("Your house is {}!".format(
                idx_to_house[int(result)]))

        # 根據學院改變視窗和按鈕顏色
        if int(result) == 0:
            w.setStyleSheet('background-color: #BDC3C7')
            btnSave.setStyleSheet('background-color: #909497')
            btnClear.setStyleSheet('background-color: #909497')
        elif int(result) == 1 or int(result) == 2:
            w.setStyleSheet('background-color: #B03A2E')
            btnSave.setStyleSheet('background-color: #F4D03F')
            btnClear.setStyleSheet('background-color: #F4D03F')
        elif int(result) == 3 or int(result) == 4:
            w.setStyleSheet('background-color: #229954')
            btnSave.setStyleSheet('background-color: #797D7F')
            btnClear.setStyleSheet('background-color: #797D7F')
        elif int(result) == 5 or int(result) == 6 or int(result) == 7:
            w.setStyleSheet('background-color: #2471A3')
            btnSave.setStyleSheet('background-color: #D0D3D4')
            btnClear.setStyleSheet('background-color: #D0D3D4')
        else:
            w.setStyleSheet('background-color: #F1C40F')
            btnSave.setStyleSheet('background-color: #797D7F')
            btnClear.setStyleSheet('background-color: #797D7F')

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.drawImage(event.rect(), self.image, self.rect())

    def mousePressEvent(self, event):
        self.path.moveTo(event.pos())

    def mouseMoveEvent(self, event):
        self.path.lineTo(event.pos())
        p = QPainter(self.image)
        p.setPen(QPen(self.myPenColor,
                      self.myPenWidth, Qt.SolidLine, Qt.RoundCap,
                      Qt.RoundJoin))
        p.drawPath(self.path)
        p.end()
        self.update()

    def sizeHint(self):
        return QSize(500, 500)


if __name__ == '__main__':
    # 檢查裝置是 CPU or GPU
    train_on_GPU = False
    if torch.cuda.is_available():
        train_on_GPU = True
        my_device = torch.device('cuda')
    else:
        my_device = torch.device('cpu')
    print('using device: {}'.format(my_device))

    # 建立 model
    model_num = 3
    models = []
    for i in range(model_num):
        models.append(CNN().float())
        model_path = os.path.join('model', 'model_' + str(i) + '_weights.pth')
        models[i].load_state_dict(torch.load(
            model_path, map_location=my_device))

    MainModel = Ensemble(models)
    model_path = os.path.join('model', 'model_main_weights.pth')
    MainModel.load_state_dict(torch.load(model_path, map_location=my_device))

    idx_to_house = {
        0: 'Death Eater',
        1: 'Gryffindor', 2: 'Gryffindor',
        3: 'Slytherin', 4: 'Slytherin',
        5: 'Ravenclaw', 6: 'Ravenclaw', 7: 'Ravenclaw',
        8: 'Hufflepuff', 9: 'Hufflepuff',
    }

    # 建立視窗
    app = QApplication(sys.argv)
    w = QWidget()

    # 設定字體
    font1 = QFont()
    font1.setFamily('Algerian')
    font1.setPointSize(13)
    font2 = QFont()
    font2.setFamily('Algerian')
    font2.setPointSize(20)

    # 設定按鈕、標題
    btnSave = QPushButton(QIcon('./sorting hat.png'), "Sorting hat")
    btnSave.setIconSize(QSize(25, 25))
    btnSave.setFont(font1)
    btnClear = QPushButton(QIcon('./broom.png'), "Clear")
    btnClear.setIconSize(QSize(30, 25))
    btnClear.setFont(font1)
    textLabel = QLabel("Draw!")
    textLabel.setFont(font2)
    textLabel.setAlignment(Qt.AlignCenter)
    drawer = Drawer()

    # 將各物件加入視窗
    w.setWindowTitle("Intelligent Sorting Hat")
    w.setLayout(QGridLayout())  # 採用網格排版
    w.layout().addWidget(btnSave, 0, 0, 1, 2)  # 前兩個數字是座標，第三個是垂直網格數，第四個是水平網格數
    w.layout().addWidget(btnClear, 0, 2, 1, 2)
    w.layout().addWidget(textLabel, 1, 0, 1, 4)
    w.layout().addWidget(drawer, 2, 0, 4, 4)

    # 按鈕被按下之後的功能
    btnSave.clicked.connect(lambda: drawer.saveImage(
        "image.png", "PNG", MainModel))
    btnClear.clicked.connect(drawer.clearImage)

    w.show()
    sys.exit(app.exec_())
