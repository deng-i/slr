import os
import shutil
import threading
import numpy as np
import time
from collections import deque
import customtkinter as ctk
from PIL import Image, ImageTk
import cv2
import tkinter as tk
# from tkinter import ttk

ctk.set_appearance_mode("Dark")


class App(ctk.CTk):
    def __init__(self):
        super().__init__()
        # set geometry
        self.geometry(f"{self.winfo_screenwidth()}x{self.winfo_screenheight()}-7-3")
        self.title("Sign recogniser")

        self.container = ctk.CTkFrame(self)
        self.container.pack(fill=tk.BOTH, expand=True)

        # set up grid
        self.columnconfigure(0, weight=1)
        self.rowconfigure(0, weight=1)
        self.container.rowconfigure(0, weight=1)
        self.container.columnconfigure(0, weight=1)

        # create data folder if first start
        self.cwd = os.getcwd()
        self.data = "../data"
        if not os.path.exists(self.data):
            os.mkdir(self.data)

        self.frames = {}
        for frame in (MenuFrame, RecordFrame, AddFrame, IdentifyFrame):
            frame_name = frame.__name__
            new_frame = frame(parent=self.container, controller=self)
            self.frames[frame_name] = new_frame
            new_frame.grid(row=0, column=0, sticky="nsew")
        self.show_frame("MenuFrame")


    def show_frame(self, frame_name):
        frame = self.frames[frame_name]
        frame.tkraise()

    def record_sign(self):
        # open camera, record sign, save it
        #    # create new folder for each sign
        # (optional) get the important part of the video
        # fine-tune model with new videos
        pass

    def identify_sign(self):
        # open camera
        # call predict on model for each frame
        # use rolling average to display the predicted sign
        pass


class MenuFrame(ctk.CTkFrame):
    def __init__(self, parent, controller):
        ctk.CTkFrame.__init__(self, parent)
        self.controller = controller
        self.columnconfigure(0, weight=1)
        self.rowconfigure(0, weight=1)
        self.rowconfigure(1, weight=1)
        self.rowconfigure(2, weight=1)
        self.rowconfigure(3, weight=1)

        self.grid(row=0, column=0)
        # add buttons
        new_sign_button = ctk.CTkButton(master=self, text="Record new sign",
                                        command=lambda: controller.show_frame("RecordFrame"))
        new_sign_button.grid(row=0, column=0, sticky="nsew")

        add_to_sign_button = ctk.CTkButton(self, text="Add more examples to an existing sign",
                                           command=lambda: controller.show_frame("AddFrame"))
        add_to_sign_button.grid(row=1, column=0, sticky="nsew")

        identify_sign_button = ctk.CTkButton(master=self, text="Identify sign",
                                             command=lambda: controller.show_frame("IdentifyFrame"))
        identify_sign_button.grid(row=2, column=0, sticky="nsew")

        new_user_button = ctk.CTkButton(master=self, text="New user", command=self.new_user)
        new_user_button.grid(row=3, column=0)  # , sticky="nsew")

    def new_user(self):
        # delete previous videos
        shutil.rmtree(self.controller.data)
        os.mkdir(self.controller.data)


class RecordFrame(ctk.CTkFrame):
    def __init__(self, parent, controller):
        ctk.CTkFrame.__init__(self, parent)
        self.controller = controller

        # used for capturing video
        self.running = False
        self.last_frame = None

        self.grid(row=0, column=0)

        # textbox for sign name
        self.sign_name = ctk.CTkTextbox(self)
        self.sign_name.grid(row=0, column=0)
        self.sign_name.insert("0.0", "Please enter a name here for the sign")

        # webcam
        self.webcam_label = ctk.CTkLabel(self)
        self.webcam_label.grid(row=1, column=0)

        # counter for how many samples there are
        ctk.CTkLabel(self, text="Number of samples:").grid(row=0, column=1)
        self.sample_num = "0/5"
        self.counter = ctk.CTkLabel(self, text=self.sample_num)
        self.counter.grid(row=1, column=1)

        self.start_record_button = ctk.CTkButton(self, text="Start recording", command=self.start_recording)
        self.start_record_button.grid(row=3, column=0)

        self.stop_record_button = ctk.CTkButton(self, text="Stop recording", command=self.stop_recording,
                                                state="disabled")
        self.stop_record_button.grid(row=3, column=1)

        menu_button = ctk.CTkButton(self, text="Back to menu", command=self.back_to_menu)
        menu_button.grid(row=2, column=0)

    def back_to_menu(self):
        self.sign_name.configure(state="normal")
        self.sample_num = "0/5"
        self.counter.configure(text=self.sample_num)
        self.controller.show_frame("MenuFrame")

    def start_recording(self):
        self.running = True
        thread = threading.Thread(target=self.capture, daemon=True)
        thread.start()
        self.update_frame()

        self.sign_name.configure(state="disabled")
        self.start_record_button.configure(state="disabled")
        self.stop_record_button.configure(state="normal")

    def stop_recording(self):
        self.running = False

        self.start_record_button.configure(state="normal")
        self.stop_record_button.configure(state="disabled")
        nums = self.sample_num.split("/")
        new_value = str(int(nums[0]) + 1) + "/" + nums[1]
        self.sample_num = new_value
        self.counter.configure(text=self.sample_num)

    def capture(self):
        capture = cv2.VideoCapture(0)
        sign_name = self.sign_name.get("0.0", "end")[:-1]  # last character is \n
        path = self.controller.data + "/" + sign_name
        if not os.path.exists(path):
            os.mkdir(path)

        file_name = path + "/" + str(self.sample_num.split("/")[0]) + ".avi"

        fourcc = cv2.VideoWriter_fourcc("X", "V", "I", "D")
        video_writer = cv2.VideoWriter(file_name, fourcc, 25, (320, 320))

        while self.running:
            rect, frame = capture.read()
            if rect:
                frame = cv2.resize(frame, (320, 320))
                cv2image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                self.last_frame = Image.fromarray(cv2image)
                video_writer.write(frame)

        capture.release()
        video_writer.release()

    def update_frame(self):
        if self.last_frame is not None:
            tk_img = ImageTk.PhotoImage(master=self.webcam_label, image=self.last_frame)
            self.webcam_label.configure(image=tk_img)
            self.webcam_label.tk_img = tk_img

        if self.running:
            self.after(30, self.update_frame)


class AddFrame(ctk.CTkFrame):
    def __init__(self, parent, controller):
        ctk.CTkFrame.__init__(self, parent)
        self.controller = controller
        self.grid(row=0, column=0)

        # used for capturing video
        self.running = False
        self.last_frame = None
        self.existing_num = 0

        self.var = tk.StringVar()
        i = 0
        for file in os.listdir(controller.data):
            ctk.CTkRadioButton(self, text=file, variable=self.var, value=file).grid(row=i, column=0)
            i += 1

        # webcam
        self.webcam_label = ctk.CTkLabel(self)
        self.webcam_label.grid(row=2, column=0)

        # counter for how many samples there are
        ctk.CTkLabel(self, text="Number of samples:").grid(row=0, column=1)
        self.sample_num = "0/5"
        self.counter = ctk.CTkLabel(self, text=self.sample_num)
        self.counter.grid(row=1, column=1)

        self.start_record_button = ctk.CTkButton(self, text="Start recording", command=self.start_recording)
        self.start_record_button.grid(row=3, column=0)

        self.stop_record_button = ctk.CTkButton(self, text="Stop recording", command=self.stop_recording,
                                                state="disabled")
        self.stop_record_button.grid(row=3, column=1)

        menu_button = ctk.CTkButton(self, text="Back to menu", command=self.back_to_menu)
        menu_button.grid(row=4, column=0)

    def back_to_menu(self):
        self.sample_num = "0/5"
        self.counter.configure(text=self.sample_num)
        self.controller.show_frame("MenuFrame")

    def start_recording(self):
        self.running = True
        thread = threading.Thread(target=self.capture, daemon=True)
        thread.start()

        path = self.controller.data + "/" + self.var.get()
        self.existing_num = len([name for name in os.listdir(path) if os.path.isfile(os.path.join(path, name))])
        self.sample_num = str(self.existing_num) + "/5"

        self.update_frame()

        self.start_record_button.configure(state="disabled")
        self.stop_record_button.configure(state="normal")

    def stop_recording(self):
        self.running = False

        self.start_record_button.configure(state="normal")
        self.stop_record_button.configure(state="disabled")
        nums = self.sample_num.split("/")
        new_value = str(int(nums[0]) + 1) + "/" + nums[1]
        self.sample_num = new_value
        self.counter.configure(text=self.sample_num)

    def capture(self):
        capture = cv2.VideoCapture(0)
        path = self.controller.data + "/" + self.var.get()

        file_name = path + "/" + str(self.existing_num) + ".avi"

        fourcc = cv2.VideoWriter_fourcc("X", "V", "I", "D")
        video_writer = cv2.VideoWriter(file_name, fourcc, 25, (320, 320))

        while self.running:
            rect, frame = capture.read()
            if rect:
                frame = cv2.resize(frame, (320, 320))
                cv2image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                self.last_frame = Image.fromarray(cv2image)
                video_writer.write(frame)

        capture.release()
        video_writer.release()

    def update_frame(self):
        if self.last_frame is not None:
            tk_img = ImageTk.PhotoImage(master=self.webcam_label, image=self.last_frame)
            self.webcam_label.configure(image=tk_img)
            self.webcam_label.tk_img = tk_img

        if self.running:
            self.after(30, self.update_frame)


class IdentifyFrame(ctk.CTkFrame):
    def __init__(self, parent, controller):
        ctk.CTkFrame.__init__(self, parent)
        self.controller = controller
        self.grid(row=0, column=0)

        # used for capturing video
        self.running = False
        self.last_frame = None
        self.queue = deque(maxlen=10)

        self.grid(row=0, column=0)

        # webcam
        self.webcam_label = ctk.CTkLabel(self)
        self.webcam_label.grid(row=1, column=0)

        # counter for how many samples there are
        ctk.CTkLabel(self, text="Predicted sign:").grid(row=0, column=1)
        self.predicted_sign = "None"
        self.sign_name = ctk.CTkLabel(self, text=self.predicted_sign)
        self.sign_name.grid(row=1, column=1)

        self.start_predict_button = ctk.CTkButton(self, text="Start recording", command=self.start_predict)
        self.start_predict_button.grid(row=3, column=0)

        self.stop_predict_button = ctk.CTkButton(self, text="Stop recording", command=self.stop_predict,
                                                 state="disabled")
        self.stop_predict_button.grid(row=3, column=1)

        menu_button = ctk.CTkButton(self, text="Back to menu", command=lambda: controller.show_frame("MenuFrame"))
        menu_button.grid(row=2, column=0)

    def start_predict(self):
        self.running = True
        thread = threading.Thread(target=self.predict, daemon=True)
        thread.start()
        self.update_frame()

        self.start_predict_button.configure(state="disabled")
        self.stop_predict_button.configure(state="normal")

    def stop_predict(self):
        self.running = False

        self.start_predict_button.configure(state="normal")
        self.stop_predict_button.configure(state="disabled")

    def predict(self):
        capture = cv2.VideoCapture(0)

        while self.running:
            print(time.time())
            rect, frame = capture.read()
            if rect:
                frame = cv2.resize(frame, (320, 320))
                cv2image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                self.last_frame = Image.fromarray(cv2image)
                image = np.asarray(cv2image)
                # call predict on image
                # predicted_image =
                # self.queue.append(predicted_image)
                # results = np.array(self.queue).mean(axis=0)
                # i = np.argmax(results)
                # self.predicted_sign = class_names[i]

        capture.release()

    def update_frame(self):
        if self.last_frame is not None:
            tk_img = ImageTk.PhotoImage(master=self.webcam_label, image=self.last_frame)
            self.webcam_label.configure(image=tk_img)
            self.webcam_label.tk_img = tk_img

        if self.running:
            self.after(30, self.update_frame)


app = App()
app.mainloop()
