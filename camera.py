from PyQt5.QtWidgets import *
from PyQt5.QtMultimedia import *
from PyQt5.QtMultimediaWidgets import *
from PyQt5.QtGui import QPixmap, QImage, QIcon, QPainter
from PyQt5.QtCore import QTimer, Qt
import numpy as np
import sys
import predict
from predict_yolo import YoloPredictor
import cv2


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setGeometry(100, 100, 1200, 800)
        
        # Получаем список доступных камер
        self.available_cameras = QCameraInfo.availableCameras()
        if not self.available_cameras:
            QMessageBox.critical(self, "Ошибка", "Не найдено ни одной камеры")
            sys.exit()
        
        self.yolo_predictor = YoloPredictor("./best.pt")

        # Создаем элементы интерфейса
        self.create_ui()
        
        # Настройка камеры
        self.select_camera(0)
        
        # Таймер для автоматического сохранения
        self.save_timer = QTimer()
        self.save_timer.timeout.connect(self.capture_image)
        self.save_interval = 5000  # 5 секунд
        
        # Счетчик для имен файлов
        self.save_seq = 0
        
        self.setWindowIcon(QIcon('icon.png'))
        self.setWindowTitle("Qt Camera App")
        self.show()

    def create_ui(self):
        # Центральный виджет
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        # Основной layout - горизонтальный разделитель
        splitter = QSplitter(Qt.Horizontal)
        central_layout = QHBoxLayout()
        central_widget.setLayout(central_layout)
        central_layout.addWidget(splitter)
        
        # Левая панель - видео с камеры
        self.video_frame = QFrame()
        self.video_frame.setFrameShape(QFrame.StyledPanel)
        video_layout = QVBoxLayout()
        self.video_frame.setLayout(video_layout)
        
        # Viewfinder для отображения видео с камеры
        self.viewfinder = QCameraViewfinder()
        video_layout.addWidget(self.viewfinder)
        
        # Правая панель - последний сохраненный кадр
        self.capture_frame = QFrame()
        self.capture_frame.setFrameShape(QFrame.StyledPanel)
        capture_layout = QVBoxLayout()
        self.capture_frame.setLayout(capture_layout)

        self.capture_lb = QLabel()
        capture_layout.addWidget(self.capture_lb)
        
        # Метка для отображения последнего кадра
        self.capture_label = QLabel("Здесь будет отображаться последний сохраненный кадр")
        self.capture_label.setAlignment(Qt.AlignCenter)
        self.capture_label.setStyleSheet("font-size: 16px;")
        capture_layout.addWidget(self.capture_label)
        
        # Добавляем панели в разделитель
        splitter.addWidget(self.video_frame)
        splitter.addWidget(self.capture_frame)


        # Панель инструментов
        toolbar = QToolBar("Панель управления")
        self.addToolBar(toolbar)
        
        # Выбор камеры
        self.camera_selector = QComboBox()
        cameras_list = ["Выберите..."] + [camera.description() for camera in self.available_cameras]
        self.camera_selector.addItems(cameras_list)
        self.camera_selector.currentIndexChanged.connect(self.select_camera)
        toolbar.addWidget(self.camera_selector)
        
        # Кнопки управления
        self.start_button = QPushButton("Старт")
        self.start_button.clicked.connect(self.start_camera)
        toolbar.addWidget(self.start_button)
        
        self.stop_button = QPushButton("Стоп")
        self.stop_button.clicked.connect(self.stop_camera)
        self.stop_button.setEnabled(False)
        toolbar.addWidget(self.stop_button)

    def select_camera(self, index):
        # Игнорируем выбор первого элемента ("Выберите...")
        if index == 0:
            return
            
        # Корректируем индекс (вычитаем 1, так как первый элемент - заглушка)
        camera_index = index - 1
        
        # Останавливаем текущую камеру, если она есть
        if hasattr(self, 'camera'):
            self.camera.stop()
        
        # Создаем новую камеру
        self.camera = QCamera(self.available_cameras[camera_index])
        self.camera.setViewfinder(self.viewfinder)
        self.camera.setCaptureMode(QCamera.CaptureStillImage)
        
        # Настройка захвата изображений
        self.image_capture = QCameraImageCapture(self.camera)
        self.image_capture.imageCaptured.connect(self.image_captured)
        self.image_capture.error.connect(self.capture_error)
        
        # Запускаем предпросмотр
        self.camera.start()

    def start_camera(self):
        if not self.camera.isAvailable():
            QMessageBox.warning(self, "Ошибка", "Камера недоступна")
            return
        
        self.camera.start()
        self.save_timer.start(self.save_interval)
        self.start_button.setEnabled(False)
        self.stop_button.setEnabled(True)
        self.statusBar().showMessage("Камера запущена, сохранение каждые 5 секунд")

    def stop_camera(self):
        self.save_timer.stop()
        self.camera.stop()
        self.start_button.setEnabled(True)
        self.stop_button.setEnabled(False)
        self.statusBar().showMessage("Камера остановлена")

    def capture_image(self):
        if self.image_capture.isReadyForCapture():
            # Захватываем изображение без сохранения на диск
            self.image_capture.capture()
        else:
            self.statusBar().showMessage("Камера не готова к захвату")
    
    def count_occupied_spaces(self, parking_mask, detections):
        """
        Подсчитывает количество занятых парковочных мест.
        
        parking_mask: бинарная маска парковки (numpy array, 255 — парковка)
        detections: список [([x1, y1, x2, y2], conf, cls_id), ...]
        
        Возвращает: количество занятых мест
        """
        occupied_count = 0
        h, w = parking_mask.shape

        for (x1, y1, x2, y2), conf, cls_id in detections:
            center_x = (x1 + x2) // 2
            center_y = (y1 + y2) // 2

            # Проверка границ, чтобы не выйти за изображение
            if 0 <= center_x < w and 0 <= center_y < h:
                if parking_mask[center_y, center_x] == 255:
                    occupied_count += 1

        return occupied_count

    def image_captured(self, request_id, image):
        try:
            image = QImage("images.jpg")
            # Преобразуем QImage в numpy array
            qimage = image.convertToFormat(QImage.Format_Grayscale8)
            ptr = qimage.bits()
            ptr.setsize(qimage.byteCount())
            arr = np.array(ptr).reshape(qimage.height(), qimage.width())
            
            # Получаем маску от модели
            mask = predict.do_predict(arr)  # Передаем numpy array напрямую
            
            # Преобразуем маску в QImage
            mask = (mask * 255).astype(np.uint8)
            qmask = QImage(mask.data, mask.shape[1], mask.shape[0], 
                        mask.shape[1], QImage.Format_Grayscale8)
            
            qimage = image.convertToFormat(QImage.Format_RGB888)
            ptr = qimage.bits()
            ptr.setsize(qimage.byteCount())
            arr = np.array(ptr).reshape((qimage.height(), qimage.width(), 3))
            arr = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
            # arr здесь — это BGR-изображение
            yolo_result, pred = self.yolo_predictor.predict(arr)

            # Преобразуем и показываем в QLabel
            qimg = QImage(yolo_result.data, yolo_result.shape[1], yolo_result.shape[0],
                        yolo_result.strides[0], QImage.Format_RGB888)
            # Преобразуем QImage в QPixmap
            pixmap1 = QPixmap.fromImage(qimg.rgbSwapped())
            pixmap2 = QPixmap.fromImage(qmask.rgbSwapped())

            # Вычисляем размеры
            width = pixmap1.width() + pixmap2.width()
            height = max(pixmap1.height(), pixmap2.height())

            # Создаем холст
            combined = QPixmap(width, height)
            combined.fill(Qt.transparent)  # если нужно прозрачное фоновое изображение

            # Рисуем оба изображения рядом
            painter = QPainter(combined)
            painter.drawPixmap(0, 0, pixmap1)
            painter.drawPixmap(pixmap1.width(), 0, pixmap2)
            painter.end()

            # Устанавливаем изображение в QLabel
            self.capture_lb.setPixmap(combined.scaled(
                self.capture_lb.width(),
                self.capture_lb.height(),
                Qt.KeepAspectRatio,
                Qt.SmoothTransformation
            ))

            res = self.count_occupied_spaces(mask, pred)
            self.capture_label.setText(f"Количество занятых парковочных мест: {res}")




        
        except Exception as e:
            self.statusBar().showMessage(f"Ошибка: {str(e)}")
            print(f"Error in image_captured: {str(e)}")

    
    def capture_error(self, id, error, error_str):
        QMessageBox.warning(self, "Ошибка захвата", error_str)

    def closeEvent(self, event):

        reply = QMessageBox.question(self, 'Message',
            "Are you sure to quit?", QMessageBox.Yes |
            QMessageBox.No, QMessageBox.No)
        if reply == QMessageBox.Yes:
            event.accept()
        else:
            event.ignore()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    sys.exit(app.exec_())