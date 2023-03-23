import os
import threading
import numpy as np
import time
from collections import deque
import customtkinter as ctk
from PIL import Image, ImageTk
import cv2
import tkinter as tk
from transfer_learn import Transfer
import shutil
import tensorflow as tf

ctk.set_appearance_mode("Dark")


class App(ctk.CTk):
    """
    Main window
    """
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
        self.data = "./data"
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
        """
        Switches visible frame
        """
        frame = self.frames[frame_name]
        frame.tkraise()


class MenuFrame(ctk.CTkFrame):
    """
    Menu page
    """
    def __init__(self, parent, controller):
        ctk.CTkFrame.__init__(self, parent)
        self.controller = controller
        self.columnconfigure(0, weight=1)
        self.rowconfigure(0, weight=1)
        self.rowconfigure(1, weight=1)
        self.rowconfigure(2, weight=1)
        self.rowconfigure(3, weight=1)
        self.rowconfigure(4, weight=1)

        self.grid(row=0, column=0)
        # add buttons
        new_sign_button = ctk.CTkButton(master=self, text="Record new sign", fg_color="black",
                                        command=lambda: controller.show_frame("RecordFrame"))
        new_sign_button.grid(row=0, column=0, sticky="nsew")

        add_to_sign_button = ctk.CTkButton(self, text="Add more examples to an existing sign", fg_color="black",
                                           command=lambda: controller.show_frame("AddFrame"))
        add_to_sign_button.grid(row=1, column=0, sticky="nsew")

        identify_sign_button = ctk.CTkButton(master=self, text="Identify sign", fg_color="black",
                                             command=lambda: controller.show_frame("IdentifyFrame"))
        identify_sign_button.grid(row=2, column=0, sticky="nsew")

        train_button = ctk.CTkButton(self, text="Train network", fg_color="black", command=self.train)
        train_button.grid(row=3, column=0, sticky="nsew")

        new_user_button = ctk.CTkButton(master=self, text="New user", fg_color="black", command=self.new_user)
        new_user_button.grid(row=4, column=0)  # , sticky="nsew")

    def new_user(self):
        """
        Resets the data directory for a new user
        """
        shutil.rmtree(self.controller.data)
        os.mkdir(self.controller.data)

    def train(self):
        """
        Train a CNN model
        """
        model = Transfer("chpoint5", os.path.join(self.controller.data, "images"))
        model.train()
        # pass


class RecordFrame(ctk.CTkFrame):
    """
    Page for recording new signs
    """
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

        self.start_record_button = ctk.CTkButton(self, text="Start recording", fg_color="black",
                                                 command=self.start_recording)
        self.start_record_button.grid(row=3, column=0)

        self.stop_record_button = ctk.CTkButton(self, text="Stop recording", fg_color="black",
                                                command=self.stop_recording,
                                                state="disabled")
        self.stop_record_button.grid(row=3, column=1)

        menu_button = ctk.CTkButton(self, text="Back to menu", fg_color="black", command=self.back_to_menu)
        menu_button.grid(row=2, column=0)

        if not os.path.exists(os.path.join(controller.data, "new_videos")):
            os.mkdir(os.path.join(controller.data, "new_videos"))

    def back_to_menu(self):
        """
        Go back to main page
        """
        # self.sign_name.configure(state="normal")
        # self.sample_num = "0/5"
        # self.counter.configure(text=self.sample_num)
        self.__init__(self.parent, self.controller)
        self.controller.frames["RecordFrame"] = self
        self.grid(row=0, column=0, sticky="nsew")
        self.controller.show_frame("MenuFrame")

    def start_recording(self):
        """
        Starts the recording process on a new thread
        """
        self.running = True
        thread = threading.Thread(target=self.capture, daemon=True)
        thread.start()
        self.update_frame()

        self.sign_name.configure(state="disabled")
        self.start_record_button.configure(state="disabled")
        self.stop_record_button.configure(state="normal")

    def stop_recording(self):
        """"
        Stops the recording and updates the number of samples on screen
        """
        self.running = False

        self.start_record_button.configure(state="normal")
        self.stop_record_button.configure(state="disabled")
        nums = self.sample_num.split("/")
        new_value = str(int(nums[0]) + 1) + "/" + nums[1]
        self.sample_num = new_value
        self.counter.configure(text=self.sample_num)
        time.sleep(0.5)
        self.convert_to_images()

    def capture(self):
        """
        Captures and saves video
        """
        capture = cv2.VideoCapture(0)
        sign_name = self.sign_name.get("0.0", "end")[:-1]  # last character is \n
        path = self.controller.data + "/new_videos/" + sign_name
        if not os.path.exists(path):
            os.mkdir(path)

        video_file_name = path + "/" + sign_name + "__" + str(self.sample_num.split("/")[0]) + ".avi"

        fourcc = cv2.VideoWriter_fourcc("X", "V", "I", "D")
        video_writer = cv2.VideoWriter(video_file_name, fourcc, 15, (320, 320))

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
        """
        Shows webcam
        """
        if self.last_frame is not None:
            tk_img = ImageTk.PhotoImage(master=self.webcam_label, image=self.last_frame)
            self.webcam_label.configure(image=tk_img)
            self.webcam_label.tk_img = tk_img

        if self.running:
            self.after(30, self.update_frame)

    def convert_to_images(self):
        """
        Convert video to images
        """
        new_videos_dir = self.controller.data + "/new_videos"
        image_dir = self.controller.data + "/images"
        videos_dir = self.controller.data + "/processed_videos"

        # make folders for processed images and videos
        if not os.path.exists(image_dir):
            os.mkdir(image_dir)
        if not os.path.exists(videos_dir):
            os.mkdir(videos_dir)

        for sign in os.listdir(new_videos_dir):
            new_video_sign_path = os.path.join(new_videos_dir, sign)
            image_sign_path = os.path.join(image_dir, sign)
            video_sign_path = os.path.join(videos_dir, sign)

            # make folders for processed images and videos signs
            if not os.path.exists(image_sign_path):
                os.mkdir(image_sign_path)
            if not os.path.exists(video_sign_path):
                os.mkdir(video_sign_path)

            for filename in os.listdir(new_video_sign_path):
                i = 0
                new_video_file_path = os.path.join(new_video_sign_path, filename)
                video_file_path = os.path.join(video_sign_path, filename)
                cap = cv2.VideoCapture(new_video_file_path)
                # image_file_path = os.path.join(image_sign_path, filename)

                # Read each frame from the video
                while cap.isOpened():
                    ret, frame = cap.read()
                    if ret:
                        split_name = filename.split(".")
                        fname = split_name[0] + "_" + str(i) + ".png"
                        file_name = os.path.join(image_sign_path, fname)
                        cv2.imwrite(file_name, frame)
                        i += 1
                    else:
                        break
                cap.release()

                # move video to processed folder
                shutil.move(new_video_file_path, video_file_path)


class AddFrame(ctk.CTkFrame):
    """
    Page for adding new videos to existing signs
    """
    def __init__(self, parent, controller):
        ctk.CTkFrame.__init__(self, parent)
        self.parent = parent
        self.controller = controller
        self.grid(row=0, column=0)

        # used for capturing video
        self.running = False
        self.last_frame = None
        self.existing_num = 0

        self.var = tk.StringVar()
        i = 0
        for file in os.listdir(controller.data + "/images"):
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

        self.start_record_button = ctk.CTkButton(self, text="Start recording", fg_color="black",
                                                 command=self.start_recording)
        self.start_record_button.grid(row=3, column=0)

        self.stop_record_button = ctk.CTkButton(self, text="Stop recording", fg_color="black",
                                                command=self.stop_recording, state="disabled")
        self.stop_record_button.grid(row=3, column=1)

        menu_button = ctk.CTkButton(self, text="Back to menu", fg_color="black", command=self.back_to_menu)
        menu_button.grid(row=4, column=0)

    def back_to_menu(self):
        """
        Go back to main page
        """
        # self.sample_num = "0/5"
        # self.counter.configure(text=self.sample_num)
        self.__init__(self.parent, self.controller)
        self.controller.frames["AddFrame"] = self
        self.grid(row=0, column=0, sticky="nsew")
        self.controller.show_frame("MenuFrame")

    def start_recording(self):
        """
        Starts the recording process on a new thread
        """
        self.running = True
        thread = threading.Thread(target=self.capture, daemon=True)
        thread.start()

        path = self.controller.data + "/processed_videos/" + self.var.get()
        self.existing_num = len([name for name in os.listdir(path) if os.path.isfile(os.path.join(path, name))])
        self.sample_num = str(self.existing_num) + "/5"

        self.update_frame()

        self.start_record_button.configure(state="disabled")
        self.stop_record_button.configure(state="normal")

    def stop_recording(self):
        """"
        Stops the recording and updates the number of samples on screen
        """
        self.running = False

        self.start_record_button.configure(state="normal")
        self.stop_record_button.configure(state="disabled")
        nums = self.sample_num.split("/")
        new_value = str(int(nums[0]) + 1) + "/" + nums[1]
        self.sample_num = new_value
        self.counter.configure(text=self.sample_num)
        time.sleep(0.5)
        self.convert_to_images()

    def capture(self):
        """
        Captures and saves video
        """
        capture = cv2.VideoCapture(0)
        path = self.controller.data + "/new_videos/" + self.var.get()

        file_name = path + "/" + self.var.get() + "__" + str(self.sample_num.split("/")[0]) + ".avi"

        fourcc = cv2.VideoWriter_fourcc("X", "V", "I", "D")
        video_writer = cv2.VideoWriter(file_name, fourcc, 15, (320, 320))

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
        """
        Shows webcam
        """
        if self.last_frame is not None:
            tk_img = ImageTk.PhotoImage(master=self.webcam_label, image=self.last_frame)
            self.webcam_label.configure(image=tk_img)
            self.webcam_label.tk_img = tk_img

        if self.running:
            self.after(30, self.update_frame)

    def convert_to_images(self):
        """
        Convert video to images
        """
        new_videos_dir = self.controller.data + "/new_videos"
        image_dir = self.controller.data + "/images"
        videos_dir = self.controller.data + "/processed_videos"

        for sign in os.listdir(new_videos_dir):
            new_video_sign_path = os.path.join(new_videos_dir, sign)
            image_sign_path = os.path.join(image_dir, sign)
            video_sign_path = os.path.join(videos_dir, sign)

            for filename in os.listdir(new_video_sign_path):
                i = 0
                new_video_file_path = os.path.join(new_video_sign_path, filename)
                video_file_path = os.path.join(video_sign_path, filename)
                cap = cv2.VideoCapture(new_video_file_path)
                # image_file_path = os.path.join(image_sign_path, filename)

                # Read each frame from the video
                while cap.isOpened():
                    ret, frame = cap.read()
                    if ret:
                        split_name = filename.split(".")
                        fname = split_name[0] + "_" + str(i) + ".png"
                        file_name = os.path.join(image_sign_path, fname)
                        cv2.imwrite(file_name, frame)
                        i += 1
                    else:
                        break
                cap.release()

                # move video to processed folder
                shutil.move(new_video_file_path, video_file_path)


class IdentifyFrame(ctk.CTkFrame):
    """
    Detects the sign on the webcam
    """
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

        # predicted sign placeholder
        ctk.CTkLabel(self, text="Predicted sign:").grid(row=0, column=1)
        self.predicted_sign = "None"
        self.sign_name = ctk.CTkLabel(self, text=self.predicted_sign)
        self.sign_name.grid(row=1, column=1)

        self.start_predict_button = ctk.CTkButton(self, text="Start recording", fg_color="black",
                                                  command=self.start_predict)
        self.start_predict_button.grid(row=3, column=0)

        self.stop_predict_button = ctk.CTkButton(self, text="Stop recording", fg_color="black",
                                                 command=self.stop_predict, state="disabled")
        self.stop_predict_button.grid(row=3, column=1)

        menu_button = ctk.CTkButton(self, text="Back to menu", fg_color="black",
                                    command=lambda: controller.show_frame("MenuFrame"))
        menu_button.grid(row=2, column=0)

    def start_predict(self):
        """
        Starts the process of detecting the sign
        """
        self.model = tf.keras.models.load_model("models/trained_model", compile=False)
        self.model.compile(loss="categorical_crossentropy", optimizer="adam", metrics="accuracy")
        self.running = True
        thread = threading.Thread(target=self.predict, daemon=True)
        thread.start()
        self.update_frame()

        self.start_predict_button.configure(state="disabled")
        self.stop_predict_button.configure(state="normal")

    def stop_predict(self):
        """
        Ends the process of detecting the sign
        """
        self.running = False

        self.start_predict_button.configure(state="normal")
        self.stop_predict_button.configure(state="disabled")

    def predict(self):
        """
        Predicts the sign on the webcam
        """
        capture = cv2.VideoCapture(0)
        class_names = os.listdir(os.path.join(self.controller.data, "images"))
        print(class_names)
        while self.running:
            print(time.time())
            rect, frame = capture.read()
            if rect:
                frame = cv2.resize(frame, (320, 320))
                cv2image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                self.last_frame = Image.fromarray(cv2image)
                image = np.array(cv2image)

                # normalise
                image = image * 1. / 255

                # store predictions in a queue, so rolling average can be used
                predicted_image = self.model(np.expand_dims(image, axis=0))[0]
                print(predicted_image)
                self.queue.append(predicted_image)
                results = np.array(self.queue).mean(axis=0)
                print(results)
                i = np.argmax(results)
                self.predicted_sign = class_names[i]
                self.sign_name.configure(text=self.predicted_sign)

        capture.release()

    def update_frame(self):
        """
        Shows webcam
        """
        if self.last_frame is not None:
            tk_img = ImageTk.PhotoImage(master=self.webcam_label, image=self.last_frame)
            self.webcam_label.configure(image=tk_img)
            self.webcam_label.tk_img = tk_img

        if self.running:
            self.after(40, self.update_frame)


app = App()
app.mainloop()
