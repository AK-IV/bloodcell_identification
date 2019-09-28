# -*- coding: utf-8 -*-

import queue

from locate_cells import Worker, FolderWorker

import tkinter as tk
from tkinter import ttk
from tkinter import *


import matplotlib.pyplot as plt
from matplotlib.figure import Figure
import matplotlib.backends.tkagg as tkagg
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import ntpath


#Основной класс интерфейса
class GUI:
    
    def __init__(self, master):
        
        #Инитиализация
        self.master = master
        
        self.img_file_path = None
        self.save_img_path = None
        self.img_directory = None
        
        self.worker = None
        
        self.progress = 0
        
        #Создание элементов интерфейса
        self.main_title = Label(self.master, text=u"Распознавание клеток на цифровых изображениях", font=("Helvetica", 16))
        self.main_title.pack()
        
        self.button_frame = Frame(master)
        
        self.img_select_button = Button(self.button_frame, text=u"Выбрать изображение",
                                    command=self.process_image)
        
        self.img_select_button.pack(side=LEFT, padx=5)
        
        self.folder_select_button = Button(self.button_frame, text=u"Выбрать папку",
                                    command=self.process_folder)
        
        self.folder_select_button.pack(side=LEFT, padx=5)
        
        self.button_frame.pack(pady=5)
        
        self.image_name = Label(self.master, text=u"  ")
        self.image_name.pack()
        
        self.fig = plt.figure(figsize=(8,6))
        self.ax = self.fig.add_subplot(111)
        
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.master)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(side="top", fill="both", expand=1)
        
        self.save_button = Button(self.master, text=u"Сохранить изображение",
                               command=self.save_image)
        self.save_button.pack(pady=5)

        self.progress_bar = ttk.Progressbar(self.master)
        self.progress_bar.configure(
            variable=self.progress,
            maximum=100,
            length=300,
            )
        self.progress_bar.pack(pady=10)
        
    #Функция нажатия кнопки Выбрать картинку
    def process_image(self):
        
        #Окно выбора файла
        self.img_file_path = tk.filedialog.askopenfilename(initialdir="/",
                                                           title="Select file",
                                                           filetypes=(("jpeg files", "*.jpg;*.jpeg"), ("all files", "*.*")))
        if self.img_file_path is None: # askopenfilename return `None` if dialog closed with "cancel".
            return
        
        #Очередь queue нужна для загрузочной шкалы
        self.queue_progress = queue.LifoQueue() # need only highest values

        #Создание класса worker распознающего клетки на картинке 
        self.worker = Worker(self.queue_progress, self.img_file_path,
                             self.fig, self.canvas)
        self.worker.start()
        #Для начать проверять состояние загрузки после 100мс
        self.master.after(100, self.__process_progress)
        #Выводим имя картинки в окно
        self.image_name['text'] = ntpath.basename(self.img_file_path)
        
    #Функция нажатия кнопки Выбрать папку
    def process_folder(self):
        
        #Выбераем папку
        self.img_directory = tk.filedialog.askdirectory(initialdir="/",
                                                           title="Select directory")
        if self.img_directory is None: # askopenfilename return `None` if dialog closed with "cancel".
            return

        self.queue_progress = queue.LifoQueue() # need only highest values
        
        #Класс FolderWorker для работы с папками
        self.worker = FolderWorker(self.queue_progress, self.img_directory,
                             self.fig, self.canvas)
        self.worker.start()
        self.master.after(100, self.__process_progress)
        #Выводим имя папки
        self.image_name['text'] = ntpath.basename(self.img_directory)
    
    #Сохранение картинки
    def save_image(self):
    
        self.save_img_path = tk.filedialog.asksaveasfilename(filetypes = (("jpeg files","*.jpg"),("all files","*.*")),
                                                defaultextension = ".jpg")
        if self.save_img_path is None: # asksaveasfile return `None` if dialog closed with "cancel".
            return
        
        self.fig.savefig(self.save_img_path)
    
    #Отображение состояния загрузки
    def __process_progress(self):
        try:
            if self.worker:
                
                #Если последнее значение состояния прогресса загрузки отличается от нового
                progress_now = self.queue_progress.get()
                if progress_now != self.progress:
                    delta = progress_now - self.progress
                    #Инкрементировать шкалу
                    self.progress_bar.step(delta)
                    self.progress = progress_now
                if not self.progress == 100:
                    self.master.after(50, self.__process_progress)
        except queue.Empty:
            self.master.after(100, self.__process_progress)

#Запуск интерфейса
def main():
    root = tk.Tk()
    root.title("Цифровая обработка изображений")
    main_ui = GUI(root)
    root.mainloop()

if __name__ == "__main__":
    main()