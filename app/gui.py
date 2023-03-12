import os
import shutil
import threading

import customtkinter as ctk
from PIL import Image, ImageTk
import cv2
import tkinter as tk
from tkinter import ttk

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

        self.frames = {}
        for frame in (MenuFrame, RecordFrame, AddFrame, IdentifyFrame):
            frame_name = frame.__name__
            new_frame = frame(parent=self.container, controller=self)
            self.frames[frame_name] = new_frame
            new_frame.grid(row=0, column=0, sticky="nsew")

        # create data folder if first start
        self.cwd = os.getcwd()
        self.data = "../data"
        if not os.path.exists(self.data):
            os.mkdir(self.data)
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
        new_user_button.grid(row=3, column=0) #, sticky="nsew")

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
        self.after_id = None
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
        self.sample_num = "0/5"
        self.counter = ctk.CTkLabel(self, text=self.sample_num)
        self.counter.grid(row=1, column=1)

        self.start_record_button = ctk.CTkButton(self, text="Start recording", command=self.start_recording)
        self.start_record_button.grid(row=3, column=0)

        self.stop_record_button = ctk.CTkButton(self, text="Stop recording", command=self.stop_recording,
                                                state="disabled")
        self.stop_record_button.grid(row=3, column=1)

        menu_button = ctk.CTkButton(self, text="Back to menu", command=lambda: controller.show_frame("MenuFrame"))
        menu_button.grid(row=2, column=0)

    def start_recording(self):
        self.running = True
        thread = threading.Thread(target=self.capture, daemon=True)
        thread.start()
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

        var = tk.StringVar()

        def print_value():
            print(var.get())

        button1 = ctk.CTkRadioButton(self, text="Option 1", variable=var, value="opt1").grid(row=0, column=0)
        button2 = ctk.CTkRadioButton(self, text="Option 2", variable=var, value="opt2").grid(row=1, column=0)

        ok_button = ctk.CTkButton(self, text="OK", command=print_value).grid(row=2, column=0)

        menu_button = ctk.CTkButton(self, text="Back to menu", command=lambda: controller.show_frame("MenuFrame"))
        menu_button.grid(row=3, column=0)


class IdentifyFrame(ctk.CTkFrame):
    def __init__(self, parent, controller):
        ctk.CTkFrame.__init__(self, parent)
        self.controller = controller
        self.grid(row=0, column=0)

        # webcam
        # label = ctk.CTkLabel(self)
        # label.grid(row=0, column=0)
        # cap = cv2.VideoCapture(0)

        # def show_frames():
        #     # Get the latest self and convert into Image
        #     cv2image = cv2.cvtColor(cap.read()[1], cv2.COLOR_BGR2RGB)
        #     img = Image.fromarray(cv2image)
        #     # convert to PhotoImage
        #     imgtk = ImageTk.PhotoImage(image=img)
        #     # label.imgtk = imgtk
        #     label.configure(image=imgtk)
        #     # repeat
        #     label.after(40, show_frames)

        # show_frames()

        counter = ctk.CTkLabel(self, text="PREDICTED CLASS: TODO")
        counter.grid(row=0, column=1)

        # start_predict_button = ctk.CTkButton(self, text="Start prediction", command=self.predict_sign).grid(row=1, column=0)
        # end_predict_button = ctk.CTkButton(self, text="End prediction", command=self.predict_sign).grid(row=1, column=1)

        menu_button = ctk.CTkButton(self, text="Back to menu", command=lambda: controller.show_frame("MenuFrame"))
        menu_button.grid(row=2, column=0)


app = App()
app.mainloop()
