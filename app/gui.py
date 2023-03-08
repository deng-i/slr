import customtkinter as ctk
import tkinter as tk
# from tkinter import ttk

ctk.set_appearance_mode("Dark")


class App(ctk.CTk):
    def __init__(self):
        super().__init__()
        # set geometry
        self.geometry(f"{self.winfo_screenwidth()}x{self.winfo_screenheight()}-7-3")
        self.title("Sign recogniser")
        self.create_menu()

    def create_menu(self):
        self.menu_frame = ctk.CTkFrame(master=self)
        # self.menu_frame.pack(fill=tk.BOTH, expand=True)
        new_sign_button = ctk.CTkButton(master=self.menu_frame, text="Record new signs", command=self.record_sign)
        new_sign_button.grid(row=0, column=0, sticky="nsew")

        identify_sign_button = ctk.CTkButton(master=self.menu_frame, text="Identify sign", command=self.identify_sign)
        identify_sign_button.grid(row=1, column=0, padx=10, pady=10)

        new_user_button = ctk.CTkButton(master=self.menu_frame, text="New user", command=self.new_user)
        new_user_button.grid(row=2, column=0, padx=10, pady=10)
        self.columnconfigure(0, weight=1)
        self.rowconfigure(0, weight=1)

    def record_sign(self):
        # open camera, record sign, save it
        #    # create new folder for each sign
        # (optional) get the important part of the video
        # fine-tune model with new videos
        print("record")

    def identify_sign(self):
        # open camera
        # call predict on model for each frame
        # use rolling average to display the predicted sign
        print("identify")

    def new_user(self):
        # delete previous videos
        print("new user")


app = App()
app.mainloop()
