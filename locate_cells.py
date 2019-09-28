# -*- coding: utf-8 -*-

import time, os , threading

from skimage import io
from sklearn.externals import joblib
import numpy as np
from nms import nms
import matplotlib.patches as mpatches
from extract_features import extract_img_features

#Основной класс для работы с изображением
class Worker(threading.Thread):

    def __init__(self, queue_progress, image_fname, plot, canvas):
        
        #Создание нового треда
        threading.Thread.__init__(self)
        #Очередь
        self.queue_progress = queue_progress
        #Канвас передаваемый интерфейсом
        self.canvas = canvas
        
        #Инитиализация некоторых параметров
        self.image_fname = image_fname
        self.plot = plot
        
        #загрузка модели и классификатора
        self.clf = joblib.load('training/bccd_model')

        self.progress = 0
        
        # List to store the detections
        self.detections = []   
        
        #Параметры "скользящего окна"
        self.min_wdw_sz = (64, 64)
        self.step_size = (8, 8)
    
    #Функция для получения тек. сост. процесса
    def get_progress(self):
        return self.progress

    #Функция run() треда
    def run(self):
        self.detect_cells()

    
    #Функция скользящего окна
    def sliding_window(self, image):

        for y in range(0, image.shape[0], self.step_size[1]):
            for x in range(0, image.shape[1], self.step_size[0]):
                yield (x, y, image[y:y + self.min_wdw_sz[1],
                                   x:x + self.min_wdw_sz[0]])
    
    #Основная функция распознования клеток на изображении
    def detect_cells(self):
    
        #Время начала выполнения функции
        t1 = time.time()
        
        #Загрузка картинки
        loaded_img = io.imread(self.image_fname, as_grey=False)
        
        #Очистить изображение интерфейса для создания нового
        self.plot.clf()
        
        #Добавить plot 
        sub_plot = self.plot.add_subplot(111)
        
        pos = 0
        
        #Выставить 0 в очередь состояния прогресса
        self.queue_progress.put(0)
        
        #Кол-во "скользящих окон" в изображении
        total_wds = 759
        
        # If the width or height of the scaled image is less than
        # the width or height of the window, then end the iterations.
        for (x, y, im_window) in self.sliding_window(loaded_img):
            if im_window.shape[0] != self.min_wdw_sz[1] or im_window.shape[1] != self.min_wdw_sz[0]:
                continue
            
            #Получить основные признаки изображение
            fd = extract_img_features(im_window)
            
            #Необходимо для классификатора
            fd = [fd]
            
            #Спрогнозировать объект находящийся в текущем скользящем окне
            pred = self.clf.predict(fd)
            #Вероятность прогнозирования
            predict_probability = np.amax(self.clf.predict_proba(fd))
            
            #Если объект не "не клетка", то окно обнаружило клетку
            if pred[0] != 'Non_cell':
                #print('DETECTED '+pred[0])
                self.detections.append((x, y, np.amax(predict_probability),
                                   self.min_wdw_sz[0], self.min_wdw_sz[1],
                                   pred[0]))
            
            #Для прогресса загрузки
            pos = pos + 1
            
            #Подсчет прогресса
            progress_old = self.progress
            self.progress = round((pos*100)/total_wds)
            
            #Выставить новое значение прогресса
            if progress_old != self.progress:
                self.queue_progress.put(self.progress)
        
        self.queue_progress.put(100)
        
        #Время завершения скрипта
        t2 = time.time()
        
        print('Exec duration: ',(t2-t1))
        
        # Perform Non Maxima Suppression
        self.detections = nms(self.detections, 0.2)
    
        cell_color = 'red'
    
        for (x_tl, y_tl, _, w, h, cell_label) in self.detections:
    
            # Draw the detections
            if cell_label == "RBC":
                cell_color = 'red'
            elif cell_label == "WBC":
                cell_color = 'blue'
            elif cell_label == "Platelet":
                cell_color = 'green'
                
            #Нарисовать прямоугольник
            rect = mpatches.Rectangle((x_tl, y_tl), w, h, fill=False,
                                      edgecolor=cell_color, linewidth=1.2)
            sub_plot.add_patch(rect)
            #Добавить имя клетки
            sub_plot.text(x_tl, y_tl - 5, cell_label)
        
        #Показать обработанное изображение
        sub_plot.imshow(loaded_img)
        
        #Обновить канвас
        self.canvas.draw()
        
#Класс работы c папками
#Тот же принцип, что и Worker, только работает с папками вместо одного изобр.
class FolderWorker(threading.Thread):

    def __init__(self, queue_progress, folder_path, plot, canvas):
        threading.Thread.__init__(self)
        self.queue_progress = queue_progress

        self.canvas = canvas
        
        #Инит. счетчиков
        self.rbc_counter = 0
        self.wbc_counter = 0
        self.platelet_counter = 0

        self.folder_path = folder_path
        self.plot = plot
        self.clf = joblib.load('training/bccd_model')
        self.progress = 0
        
        # List to store the detections
        self.detections = []   

        self.min_wdw_sz = (64, 64)
        self.step_size = (20, 20)
    
    def get_progress(self):
        return self.progress

        
    def run(self):
        self.detect_cells()


    def sliding_window(self, image):

        for y in range(0, image.shape[0], self.step_size[1]):
            for x in range(0, image.shape[1], self.step_size[0]):
                yield (x, y, image[y:y + self.min_wdw_sz[1],
                                   x:x + self.min_wdw_sz[0]])
    
    def detect_cells(self):
    
        t1 = time.time()
        
        self.plot.clf()
        sub_plot = self.plot.add_subplot(111)
        
        pos = 0

        self.queue_progress.put(0)
        
        #Общее кол-во файлов в папке
        path, dirs, files = next(os.walk(self.folder_path))
        file_count = len(files)
        
        #Пробежать по всем файлам заданной папки
        for file in os.listdir(self.folder_path):
            
            #Если файл не .jpg или .jpeg, то перейти к следуюшему
            if not(file.endswith('.jpg') or file.endswith('.jpeg')):
                continue
            
            #Загр файл
            loaded_img = io.imread(self.folder_path+'/'+file, as_grey=False)

            for (x, y, im_window) in self.sliding_window(loaded_img):
                if im_window.shape[0] != self.min_wdw_sz[1] or im_window.shape[1] != self.min_wdw_sz[0]:
                    continue
        
                fd = extract_img_features(im_window)
        
                fd = [fd]
        
                pred = self.clf.predict(fd)
                predict_probability = np.amax(self.clf.predict_proba(fd))
        
                if pred[0] != 'Non_cell':
                    #print('DETECTED '+pred[0])
                    self.detections.append((x, y, np.amax(predict_probability),
                                       self.min_wdw_sz[0], self.min_wdw_sz[1],
                                       pred[0]))
            
            # Perform Non Maxima Suppression
            self.detections = nms(self.detections, 0.2)
            
            for det in self.detections:
                if det[5]=='Platelet':
                    self.platelet_counter = self.platelet_counter + 1
                elif det[5]=='WBC':
                    self.wbc_counter = self.wbc_counter + 1
                else:
                    self.rbc_counter = self.rbc_counter + 1
            
            pos = pos + 1
            
            progress_old = self.progress
            self.progress = round((pos*100)/file_count)

            if progress_old != self.progress:
                self.queue_progress.put(self.progress)
        
        self.queue_progress.put(100)

        t2 = time.time()
        
        print('Exec duration: ',(t2-t1))
        
        #Рисование диаграммы кол-ва найденных клеток
        cell_types = ('RBC', 'WBC', 'Platelet')
        y_pos = np.arange(len(cell_types))
        performance = [self.rbc_counter, self.wbc_counter, self.platelet_counter]
         
        sub_plot.bar(y_pos, performance, color=("red","blue","green"),
                         alpha=0.5, width=0.7)
        sub_plot.set_xticks(y_pos)
        sub_plot.set_xticklabels(cell_types)
        
        sub_plot.set_ylabel('Счетчик')
        sub_plot.set_title('Количество найденных клеток')
        
        self.canvas.draw()
