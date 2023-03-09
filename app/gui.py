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
        self.frames = {}
        self.container = ctk.CTkFrame(self)
        self.container.pack(fill=tk.BOTH, expand=True)
        # self.container.grid(row=0, column=0)
        self.columnconfigure(0, weight=1)
        self.rowconfigure(0, weight=1)
        self.container.rowconfigure(0, weight=1)
        self.container.columnconfigure(0, weight=1)
        self.create_menu()

    def create_menu(self):
        # set grid size
        menu_frame = ctk.CTkFrame(master=self.container)
        menu_frame.columnconfigure(0, weight=1)
        menu_frame.rowconfigure(0, weight=1)
        menu_frame.rowconfigure(1, weight=1)
        menu_frame.rowconfigure(2, weight=1)

        # menu_frame.pack(fill=tk.BOTH, expand=True)
        menu_frame.grid(row=0, column=0)
        # add buttons
        new_sign_button = ctk.CTkButton(master=menu_frame, text="Record new sign", command=self.record_sign)
        new_sign_button.grid(row=0, column=0, sticky="nsew")

        add_to_sign_button = ctk.CTkButton(menu_frame, text="Add more examples to an existing sign", command=self.add_to_sign)
        add_to_sign_button.grid(row=1, column=0, sticky="nsew")

        identify_sign_button = ctk.CTkButton(master=menu_frame, text="Identify sign", command=self.identify_sign)
        identify_sign_button.grid(row=2, column=0, sticky="nsew")

        new_user_button = ctk.CTkButton(master=menu_frame, text="New user", command=self.new_user)
        new_user_button.grid(row=3, column=0, sticky="nsew")
        self.frames["menu"] = menu_frame

    def record_sign(self):
        # open camera, record sign, save it
        #    # create new folder for each sign
        # (optional) get the important part of the video
        # fine-tune model with new videos
        # self.frames["menu"].destroy()

        record_frame = ctk.CTkFrame(self.container)
        record_frame.grid(row=0, column=0)

        # webcam
        label = ctk.CTkLabel(record_frame)
        label.grid(row=0, column=0)
        cap = cv2.VideoCapture(0)

        def show_frames():
            # Get the latest frame and convert into Image
            cv2image = cv2.cvtColor(cap.read()[1], cv2.COLOR_BGR2RGB)
            img = Image.fromarray(cv2image)
            # convert to PhotoImage
            imgtk = ImageTk.PhotoImage(image=img)
            # label.imgtk = imgtk
            label.configure(image=imgtk)
            # repeat
            label.after(40, show_frames)

        # show_frames()

        counter = ctk.CTkLabel(record_frame, text="NUMBER OF SAMPLES: TODO")
        counter.grid(row=0, column=1)

        new_sign_button = ctk.CTkButton(record_frame, text="Record new sign", command=show_frames)
        new_sign_button.grid(row=3, column=0)

        menu_button = ctk.CTkButton(record_frame, text="Back to menu", command=self.create_menu)
        menu_button.grid(row=2, column=0)

        # start_recording = ctk.CTkButton(record_frame, text="Start recording", command=self.record)
        # start_recording.grid(row=2, column=0)

        # end_recording = ctk.CTkButton(record_frame, text="End recording", command=self.record)
        # end_recording.grid(row=2, column=1)
        self.frames["record_sign"] = record_frame

    def add_to_sign(self):
        self.frames["menu"].destroy()
        frame = ctk.CTkFrame(self.container)
        frame.grid(row=0, column=0)

        var = tk.StringVar()

        def print_value():
            print(var.get())

        button1 = ctk.CTkRadioButton(frame, text="Option 1", variable=var, value="opt1").grid(row=0, column=0)
        button2 = ctk.CTkRadioButton(frame, text="Option 2", variable=var, value="opt2").grid(row=1, column=0)

        ok_button = ctk.CTkButton(frame, text="OK", command=print_value).grid(row=2, column=0)

        menu_button = ctk.CTkButton(frame, text="Back to menu", command=self.create_menu)
        menu_button.grid(row=3, column=0)

        self.frames["add_to_sign"] = frame

    def identify_sign(self):
        # open camera
        # call predict on model for each frame
        # use rolling average to display the predicted sign
        frame = ctk.CTkFrame(self.container)
        frame.grid(row=0, column=0)

        # webcam
        label = ctk.CTkLabel(frame)
        label.grid(row=0, column=0)
        cap = cv2.VideoCapture(0)

        def show_frames():
            # Get the latest frame and convert into Image
            cv2image = cv2.cvtColor(cap.read()[1], cv2.COLOR_BGR2RGB)
            img = Image.fromarray(cv2image)
            # convert to PhotoImage
            imgtk = ImageTk.PhotoImage(image=img)
            # label.imgtk = imgtk
            label.configure(image=imgtk)
            # repeat
            label.after(40, show_frames)

        show_frames()

        counter = ctk.CTkLabel(frame, text="PREDICTED CLASS: TODO")
        counter.grid(row=0, column=1)

        # start_predict_button = ctk.CTkButton(frame, text="Start prediction", command=self.predict_sign).grid(row=1, column=0)
        # end_predict_button = ctk.CTkButton(frame, text="End prediction", command=self.predict_sign).grid(row=1, column=1)

        menu_button = ctk.CTkButton(frame, text="Back to menu", command=self.create_menu)
        menu_button.grid(row=2, column=0)

        self.frames["predict_sign"] = frame

    def new_user(self):
        # delete previous videos
        print("new user")


app = App()
app.mainloop()
