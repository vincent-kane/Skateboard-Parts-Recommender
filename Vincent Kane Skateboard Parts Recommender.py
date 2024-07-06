import tkinter as tk
from tkinter import font as tkFont
import pandas as pd
import webbrowser
import random
import sys
import os
from tkinter import ttk
from PIL import Image, ImageTk
import numpy as np
from sklearn.preprocessing import OneHotEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# This code is here to check if it's being ran as an exe, to help correctly find the file paths.
if getattr(sys, 'frozen', False):
    # If it's being ran as an executable, the base path is sys._MEIPASS
    application_path = sys._MEIPASS
else:
    # If it's being ran as a normal Python script, the base path is the directory containing the script
    application_path = os.path.dirname(os.path.abspath(__file__))

# Constructing the full path to the image
image_path = os.path.join(application_path, 'skateboard_image.png')

# Constructing the full path to the icon
icon_path = os.path.join(application_path, 'skateboard.ico')

# Getting the directory of the executable
base_path = getattr(sys, '_MEIPASS', os.path.dirname(os.path.abspath(__file__)))

# Using the base_path to define the full path to the data files
decks_csv_path = os.path.join(base_path, 'decks.csv')
wheels_csv_path = os.path.join(base_path, 'wheels.csv')
trucks_csv_path = os.path.join(base_path, 'trucks.csv')

# Creating a ? next to "riding style" to help the user understand what this means. It opens a little info box when hovered over.
class ToolTip(object):
    def __init__(self, widget):
        self.widget = widget
        self.tipwindow = None
        self.id = None
        self.x = self.y = 0

    def showtip(self, text):
        self.text = text
        if self.tipwindow or not self.text:
            return
        x, y, _, _ = self.widget.bbox("insert")
        x += self.widget.winfo_rootx() + 27
        y += self.widget.winfo_rooty() + 27
        self.tipwindow = tw = tk.Toplevel(self.widget)
        tw.wm_overrideredirect(True)
        tw.wm_geometry("+%d+%d" % (x, y))
        label = tk.Label(tw, text=self.text, justify=tk.LEFT,
                         background="#ffffe0", relief=tk.SOLID, borderwidth=1,
                         font=("tahoma", "8", "normal"))
        label.pack(ipadx=1)

    def hidetip(self):
        tw = self.tipwindow
        self.tipwindow = None
        if tw:
            tw.destroy()

# Specifying the characteristics of the ? tip window
def create_label_with_tooltip(parent, text, tooltip_text, row, col, padx=(20, 0), pady=10):
    label_frame = tk.Frame(parent, bg=bg_color)
    label_frame.grid(row=row, column=col, padx=padx, pady=pady, sticky='w')

    label = tk.Label(label_frame, text=text, bg=bg_color, fg=fg_color, font=default_font)
    label.pack(side='left')

    question_mark = tk.Label(label_frame, text=" ?", bg=bg_color, fg=fg_color, font=default_font)
    question_mark.pack(side='left')
    tooltip = ToolTip(question_mark)
    question_mark.bind('<Enter>', lambda event: tooltip.showtip(tooltip_text))
    question_mark.bind('<Leave>', lambda event: tooltip.hidetip())

# This conversion was only necessary due to me including "136/139" in my trucks.csv.
def convert_truck_size(size):
    if '/' in size:
        sizes = size.split('/')
        return sum(map(float, sizes)) / len(sizes)
    return float(size)

# Getting the mean of the two numbers in each shoe size category
def convert_shoe_size(shoe_size):
    sizes = list(map(int, shoe_size.split('-')))
    return np.mean(sizes)

# Loading Data
decks_df = pd.read_csv(decks_csv_path)
trucks_df = pd.read_csv(trucks_csv_path)
wheels_df = pd.read_csv(wheels_csv_path)

# Preprocessing shoe sizes by converting range to a single numeric value.
decks_df['Shoe Size'] = decks_df['Shoe Size'].apply(convert_shoe_size)
trucks_df['Shoe Size'] = trucks_df['Shoe Size'].apply(convert_shoe_size)

# Converting 'Truck Size (mm)' using the previously defined tucks_df function
trucks_df['Truck Size (mm)'] = trucks_df['Truck Size (mm)'].apply(convert_truck_size)

# Preparing features for decks and trucks
features_decks = ['Height (in)', 'Shoe Size']
features_trucks = ['Truck Size (mm)', 'Shoe Size']
X_decks = decks_df[features_decks]
X_trucks = trucks_df[features_trucks]
y_decks = decks_df['Deck Name']
y_trucks = trucks_df['Truck Name']

# Decision Tree Classifier for decks and trucks
clf_decks = DecisionTreeClassifier()
clf_trucks = DecisionTreeClassifier()

# Splitting and training
X_train_decks, X_test_decks, y_train_decks, y_test_decks = train_test_split(X_decks, y_decks, test_size=0.2, random_state=42)
X_train_trucks, X_test_trucks, y_train_trucks, y_test_trucks = train_test_split(X_trucks, y_trucks, test_size=0.2, random_state=42)
clf_decks.fit(X_train_decks, y_train_decks)
clf_trucks.fit(X_train_trucks, y_train_trucks)

# Evaluating
y_pred_decks = clf_decks.predict(X_test_decks)
y_pred_trucks = clf_trucks.predict(X_test_trucks)
print(f"Deck Accuracy: {accuracy_score(y_test_decks, y_pred_decks)}")
print(f"Truck Accuracy: {accuracy_score(y_test_trucks, y_pred_trucks)}")


# Preprocessing for wheels
X_wheels = wheels_df[['Category']]
y_wheels = wheels_df['Wheel Name']
encoder = OneHotEncoder(handle_unknown='ignore')
encoder.fit(X_wheels)
X_encoded_wheels = encoder.transform(X_wheels)

# Splitting the dataset into training and testing sets for wheels
X_train_wheels, X_test_wheels, y_train_wheels, y_test_wheels = train_test_split(X_encoded_wheels, y_wheels, test_size=0.2, random_state=42)

# Training the Decision Tree Classifier for wheels
# Adjusting hyperparameters to prevent overfitting
clf_wheels = DecisionTreeClassifier(max_depth=5, min_samples_split=4)
clf_wheels.fit(X_train_wheels, y_train_wheels)

# Predicting and evaluating the model for wheels
y_pred_wheels = clf_wheels.predict(X_test_wheels)
print(f"Wheel Accuracy: {accuracy_score(y_test_wheels, y_pred_wheels)}")

# Creating the main window
window = tk.Tk()
window.title("Skateboard Parts Recommender")
bg_color = "#ffffff"  # White background
fg_color = "#333333"  # Grey text
accent_color = "#4CAF50"  # Green accent
window.config(bg=bg_color)
window.iconbitmap(icon_path)

# Setting fonts
default_font = tkFont.Font(family="Helvetica", size=10)
title_font = tkFont.Font(family="Helvetica", size=12, weight="bold")
button_font = tkFont.Font(family="Arial", size=10, weight="bold")

# Function to create stylized buttons
def create_stylized_button(parent, text, command, row, col, width=300, height=40, color="#4CAF50", font_size=10, padx=10, pady=10):
    font = tkFont.Font(family="Arial", size=font_size, weight="bold")
    btn = tk.Button(parent, text=text, command=command, font=font, bg=color, fg='white', borderwidth=0, relief="flat")
    btn.grid(row=row, column=col, sticky="nsew", padx=padx, pady=pady)
    return btn

# GUI element functions
def inches_to_height(inches):
    feet = inches // 12
    remaining_inches = inches % 12
    return feet, remaining_inches

def update_height_label(value):
    feet, inches = inches_to_height(int(value))
    height_label.config(text=f"Height: {feet} feet {inches} inches")

# Bushing recommendation based on Riding Style:
def recommend_bushings(riding_style):
    if riding_style == "Technical":
        return "Bones Hardcore Hard Skateboard Bushings"
    elif riding_style == "Street":
        return "Bones Hardcore Medium Skateboard Bushings"
    elif riding_style == "Freestyle":
        return random.choice([
            "Bones Hardcore Soft Skateboard Bushings",
            "Bones Hardcore Medium Skateboard Bushings",
            "Bones Hardcore Hard Skateboard Bushings"])
    elif riding_style in ["Vert", "Cruising"]:
        return "Bones Hardcore Soft Skateboard Bushings"
    return "No bushings found"

# Function for converting the user riding style input into the wheel stlyes.
def convert_user_input_to_category(user_input):
    mapping = {
        "Freestyle": "Wheels for Technical Street Skating",
        "Vert": "Wheels for Transition Skating, Cruisers, and Longboards",
        "Technical": "Wheels for Technical Street Skating",
        "Cruising": "Wheels for Transition Skating, Cruisers, and Longboards"
    }
    return mapping.get(user_input, "Default Category")

# Converting the deck size ranges into a usable value.
def convert_deck_size_range(size_range):
    sizes = size_range.split('-')
    return (float(sizes[0]) + float(sizes[1])) / 2

trucks_df['Deck Size Mean'] = trucks_df['Deck Size Range (in)'].apply(convert_deck_size_range)

# Getting the deck size from the deck name variable.
def extract_deck_size(deck_name):
    try:
        size_part = deck_name.split()[-1]
        size = float(size_part)
        return size
    except (IndexError, ValueError) as e:
        print("Error extracting deck size:", e)
        return None

def recommend():
    # Getting values from user input:
    height = height_slider.get()
    shoe_size = shoe_size_slider.get()
    riding_style = riding_style_var.get()
    budget = budget_var.get()

    # Using height and shoe size to predict a deck size that'll suit the user.
    input_data_decks = pd.DataFrame([[height, shoe_size]], columns=features_decks)
    recommended_deck = clf_decks.predict(input_data_decks)[0]

    recommended_deck_size = extract_deck_size(recommended_deck)
    if recommended_deck_size is None:
        recommended_truck = "No valid deck size found"
    else:
        suitable_trucks = trucks_df[(trucks_df['Deck Size Mean'] >= recommended_deck_size - 0.25) & 
                                    (trucks_df['Deck Size Mean'] <= recommended_deck_size + 0.25)]
        if not suitable_trucks.empty:
            recommended_truck = suitable_trucks.iloc[0]['Truck Name']
        else:
            recommended_truck = "No suitable trucks found"

    riding_style_user_input = riding_style_var.get()
    converted_category = convert_user_input_to_category(riding_style_user_input)

    # Use the converted category for model prediction
    input_data = pd.DataFrame([[converted_category]], columns=['Category'])
    category_encoded = encoder.transform(input_data)
    recommended_wheel = clf_wheels.predict(category_encoded)[0]

    # Incorporate shoe size and height into wheel recommendation
    recommended_wheel = get_wheel_for_style(riding_style, shoe_size, height)

    # Recommending bushings based on riding style.
    recommended_bushings = recommend_bushings(riding_style)
    recommended_bearing = get_bearing_for_budget(budget)

    # Since grip tape is a subjective part of a skateboard in terms of taste, and they're all generally good, I set it to choose randomly
    # between three grip tape options.
    recommended_grip_tape = random.choice([
        "Mob Grip Tape 9 in. x 33 in. Sheet",
        "Jessup Grip Tape 9 in.",
        "Grizzly Griptape"
    ])

    # Since hardware (the nuts and bolts holding the trucks to the deck) is a subjective part of a skateboard in terms of taste, and they're all generally good, 
    # I set it to choose randomly between three hardware options.
    recommended_hardware = random.choice([
        "Standard Phillips Head Raven Black Skateboard Hardware Set",
        "Thunder Trucks Phillips Head Blue Hardware - 7/8",
        "Ace Trucks MFG. Phillips Head Black Skateboard Hardware Set"
    ])

    # Since most skate tools will function well for setting up a skateboard, I chose three options which will be selected at random.
    recommended_skate_tool = random.choice([
        "Spitfire Wheels T3 Multi-Purpose Skate Tool",
        "Unit Tools T-Tool Green Multi-Purpose Skate Tool",
        "Paris Truck Co. Black Multi-Purpose Skate Tool"
    ])

    # Riser pads are good for preventing wheel bite, which is especially important for the Vert and Cruising riding styles,
    # so I chose three different riser pads to randomly choose from if the user selects one of these riding style.
    recommended_riser_pads = random.choice([
        "Independent Truck Company Genuine Parts White Riser Pads - 1/8",
        "Dooks Shock Pads - 1/8",
        "Pig Wheels Piles Red Riser Pads - 1/8"
    ]) if riding_style in ["Vert", "Cruising"] else "No riser pads needed"

    recommend_dialog = tk.Toplevel(window)
    recommend_dialog.title("Recommendation Details")
    recommend_dialog.iconbitmap(icon_path)
    recommend_dialog.config(bg=bg_color)

    # Label with recommendations
    recommendation_label = tk.Label(recommend_dialog,
                                    text=f"Deck: {recommended_deck}\nTruck: {recommended_truck}\nWheel: {recommended_wheel}\nBushings: {recommended_bushings}\nBearings: {recommended_bearing}\nGrip Tape: {recommended_grip_tape}\nHardware: {recommended_hardware}\nSkate Tool: {recommended_skate_tool}\nRiser Pads: {recommended_riser_pads}",
                                    bg=bg_color,
                                    fg=fg_color,
                                    font=title_font,
                                    anchor='w',
                                    justify='left')  
    recommendation_label.grid(row=0, column=0, sticky='w', padx=20, pady=10, columnspan=2)

    # Buttons for recommendation window actions.
    # This one, when pressed by the user, will open their default browser and search Google for all the items recommended for the user.
    search_button = tk.Button(recommend_dialog, text="Search Google for Recommended Items",
                              command=lambda: search_google([recommended_deck, recommended_truck, recommended_wheel, recommended_bushings, recommended_bearing, recommended_grip_tape, recommended_hardware, recommended_skate_tool, recommended_riser_pads]),
                              bg=accent_color, fg='white', font=default_font)
    search_button.grid(row=1, column=0, padx=5, pady=5, sticky='ew') 

    # This one will close the recommendation dialog box.
    close_button = tk.Button(recommend_dialog, text="Close Recommendation Box",
                             command=recommend_dialog.destroy,
                             bg=accent_color, fg='white', font=default_font)
    close_button.grid(row=1, column=1, padx=5, pady=5, sticky='ew')

    recommend_dialog.grid_columnconfigure(0, weight=1)  
    recommend_dialog.grid_columnconfigure(1, weight=1)  

# This is the function for searching Google. 
def search_google(items):
    for item in items:
        webbrowser.open_new_tab(f"https://www.google.com/search?q={item}")

# This is the function for getting the wheels recommendation based on user shoe size and height.
def get_wheel_for_style(riding_style, shoe_size, height):

    # Thresholds for shoe size and height to determine wheel size
    shoe_size_threshold = 8  # Threshold for shoe size
    height_threshold = 66  # Threshold in inches for height

    # Setting the size category based on user shoe size and height
    if shoe_size >= shoe_size_threshold or height >= height_threshold:
        size_category = "Larger"
    else:
        size_category = "Smaller"

    # Mapping the riding style and size category to the wheel category
    category_mapping = {
        ("Freestyle", "Larger"): "Wheels for Technical Street Skating",
        ("Freestyle", "Smaller"): "Wheels for Technical Street Skating",
        ("Vert", "Larger"): "Wheels for Transition Skating, Cruisers, and Longboards",
        ("Vert", "Smaller"): "Wheels for Transition Skating, Cruisers, and Longboards",
        ("Street", "Larger"): "Wheels for All-around Skating, Good for Street",
        ("Street", "Smaller"): "Wheels for Technical Street Skating",
        ("Technical", "Larger"): "Wheels for Technical Street Skating",
        ("Technical", "Smaller"): "Wheels for Technical Street Skating",
        ("Cruising", "Larger"): "Wheels for Transition Skating, Cruisers, and Longboards",
        ("Cruising", "Smaller"): "Wheels for All-around Skating, Good for Street"
    }
    wheel_category = category_mapping.get((riding_style, size_category), "Default Category")

    # Filtering the wheels dataframe based on the selected category and size
    filtered_wheels = wheels_df[wheels_df['Category'] == wheel_category]

    # Recommending larger or smaller wheels based on the size category
    if size_category == "Larger":
        recommended_wheel = filtered_wheels[filtered_wheels['Size (mm)'].astype(float) >= 54]
    else:
        recommended_wheel = filtered_wheels[filtered_wheels['Size (mm)'].astype(float) <= 53]

    # If there are no suitable wheels found, select the closest available wheel
    if recommended_wheel.empty:
        closest_wheel = filtered_wheels.iloc[(filtered_wheels['Size (mm)'].astype(float) - (54 if size_category == "Larger" else 53)).abs().argsort()[:1]]
        recommended_wheel_name = closest_wheel['Wheel Name'].iloc[0]
    else:
        # If there are multiple options, select one at random
        recommended_wheel_name = recommended_wheel.sample(n=1)['Wheel Name'].iloc[0]

    return recommended_wheel_name

def get_bearing_for_budget(budget):
    bearings_options = {
        "Small": ["Mini Logo Skateboard Bearings", "Bones Reds Skateboard Bearings"],
        "Medium": ["Bronson Speed Co. G2 Skateboard Bearings", "Spitfire Cheapshots Bearings"],
        "Large": ["Bones Big Reds Skateboard Bearings", "Bones Super Reds Skateboard Bearings"]
    }
    return random.choice(bearings_options[budget])

# GUI components:
image = Image.open(image_path)
photo = ImageTk.PhotoImage(image)
image_label = tk.Label(window, image=photo, bg=bg_color)
image_label.grid(row=0, column=0, columnspan=2, padx=20, pady=10)

height_label = tk.Label(window, text="Height: 2 feet 0 inches", bg=bg_color, fg=fg_color, font=default_font)
height_label.grid(row=1, column=0, columnspan=2, padx=20, pady=10)

height_slider = tk.Scale(window, from_=24, to=100, orient=tk.HORIZONTAL, command=update_height_label, bg=bg_color, fg=fg_color, font=default_font)
height_slider.set(24)
height_slider.grid(row=2, column=0, columnspan=2, padx=20)

shoe_size_label = tk.Label(window, text="Shoe Size:", bg=bg_color, fg=fg_color, font=default_font)
shoe_size_label.grid(row=3, column=0, padx=20, pady=10)

shoe_size_slider = tk.Scale(window, from_=3, to=15, orient=tk.HORIZONTAL, bg=bg_color, fg=fg_color, font=default_font)
shoe_size_slider.grid(row=3, column=1, padx=20)

create_label_with_tooltip(window, "Riding Style:", "Riding style is the type of skateboarding you intend to do.\n• Freestyle: a highly techincal type of skateboarding\n   which focuses on different board maneuvers.\n• Vert: skateboarding focusing on half-pipes, bowls, and other\n   transition features.\n• Street: a skateboarding style which is done on the streets\n   and in public. Often done on rails, curbs, ledges, gaps, etc.\n• Technical: skateboarding which is focused on the different tricks which can be done with a board.\n   This includes kickflips, hard flips, etc.\n• Cruising: skateboarding which is focused on getting around quickly, and not focused on tricks or vert skating.\n   Perfect for filmers, or people using a skateboard for transportation purposes.", row=4, col=0)

riding_style_var = tk.StringVar(window)
riding_style_var.set("Freestyle")
riding_style_options = ["Freestyle", "Vert", "Street", "Technical", "Cruising"]
riding_style_menu = tk.OptionMenu(window, riding_style_var, *riding_style_options)
riding_style_menu.config(bg=bg_color, fg=fg_color, font=default_font)
riding_style_menu["menu"].config(bg=bg_color, fg=fg_color, font=default_font)
riding_style_menu.grid(row=4, column=1, padx=20)

budget_label = tk.Label(window, text="Budget:", bg=bg_color, fg=fg_color, font=default_font)
budget_label.grid(row=5, column=0, padx=20, pady=10)

budget_var = tk.StringVar(window)
budget_var.set("Small")
budget_options = ["Small", "Medium", "Large"]
budget_menu = tk.OptionMenu(window, budget_var, *budget_options)
budget_menu.config(bg=bg_color, fg=fg_color, font=default_font)
budget_menu["menu"].config(bg=bg_color, fg=fg_color, font=default_font)
budget_menu.grid(row=5, column=1, padx=20)

recommend_button = create_stylized_button(window, "Recommend", recommend, row=6, col=0, width=200)
recommend_button.grid(row=6, columnspan=2, padx=20, pady=10)

window.mainloop()