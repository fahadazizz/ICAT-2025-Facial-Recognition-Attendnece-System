import sys
import cv2
import os
import numpy as np
import sqlite3
import datetime
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QPushButton,
    QLabel, QLineEdit, QComboBox, QFrame, QMessageBox, QGroupBox, QTableWidget, QTableWidgetItem,
    QStackedWidget, QScrollArea, QGridLayout
)
from PyQt5.QtGui import QFont, QImage, QPixmap
from PyQt5.QtCore import Qt, QTimer
import pandas as pd

# Constants
DATASET_PATH = "dataset"
MODEL_FILE = "face_trainer.yml"
ATTENDANCE_FILE = "monthly_attendance.xlsx"

# Load Haarcascade for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

# Define roles mapping for departments
ROLES_MAPPING = {
    "Sales": ["Sales Manager", "Sales Executive"],
    "Marketing": ["Marketing Manager", "Marketing Executive"],
    "Development": ["Software Engineer", "QA Engineer"],
    "Security": ["Security Officer"],
    "Other": ["Other"]
}

class AttendanceSystem(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Real-Time Facial Recognition Attendance System")
        self.setGeometry(100, 100, 1000, 600)
        self.setStyleSheet("background-color: #f0f0f0;")
        self.initialize_database()  # Create required tables if not present
        self.init_ui()
        # Variable to keep track of whether attendance is running
        self.attendance_running = False
        self.attendance_timer = None
        self.cap = None

    def initialize_database(self):
        """Create the required database tables if they do not exist."""
        conn = sqlite3.connect("attendance.db")
        cursor = conn.cursor()
        # Create employees table
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS employees (
        emp_id TEXT PRIMARY KEY, 
        name TEXT, 
        email TEXT, 
        phone TEXT, 
        department TEXT, 
        role TEXT,
        age TEXT, 
        folder TEXT
        )
        """)
        # Create attendance table
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS attendance (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        emp_id TEXT,
        date TEXT
        )
        """)
        conn.commit()
        conn.close()

    def init_ui(self):
        # Main Widget and Layout
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        main_layout = QHBoxLayout()
        main_widget.setLayout(main_layout)

        # Sidebar Layout and Widget
        sidebar = QFrame()
        sidebar.setStyleSheet("background-color: #1e3d59; border-radius: 10px;")
        sidebar.setFixedWidth(235)
        sidebar_layout = QVBoxLayout()
        sidebar.setLayout(sidebar_layout)

        # --- Top Section: Main Navigation Buttons ---
        top_nav = QVBoxLayout()
        # Common style for nav buttons with a white border when not hovered/clicked
        nav_button_style = """
        QPushButton {
        color: white;
        background-color: #1e3d59;
        border: 1px solid white;
        border-radius: 5px;
        padding: 10px;
        text-align: center;
        }
        QPushButton:hover {
        background-color: #ff6b6b;
        }
        """

        btn_add_employee = QPushButton("📂 Add Employee")
        btn_add_employee.setToolTip("Add a new employee to the system.")
        btn_add_employee.setFont(QFont("Arial", 12, QFont.Bold))
        btn_add_employee.setStyleSheet(nav_button_style)
        btn_add_employee.clicked.connect(self.show_add_employee)
        top_nav.addWidget(btn_add_employee)
        top_nav.addSpacing(10)

        btn_mark_attendance = QPushButton("✅ Mark Attendance")
        btn_mark_attendance.setToolTip("Start marking attendance using facial recognition.")
        btn_mark_attendance.setFont(QFont("Arial", 12, QFont.Bold))
        btn_mark_attendance.setStyleSheet(nav_button_style)
        btn_mark_attendance.clicked.connect(self.toggle_attendance)
        top_nav.addWidget(btn_mark_attendance)
        top_nav.addSpacing(10)

        btn_check_employees = QPushButton("📋 Check Employees")
        btn_check_employees.setToolTip("View the list of registered employees and their attendance.")
        btn_check_employees.setFont(QFont("Arial", 12, QFont.Bold))
        btn_check_employees.setStyleSheet(nav_button_style)
        btn_check_employees.clicked.connect(self.show_check_employees)
        top_nav.addWidget(btn_check_employees)
        top_nav.addSpacing(10)

        sidebar_layout.addLayout(top_nav)
        sidebar_layout.addStretch()  # Push remaining items to bottom

        # --- Bottom Section: About Developers Button ---
        btn_about = QPushButton("About Developers")
        btn_about.setToolTip("Learn about the developers behind this project.")
        btn_about.setFont(QFont("Arial", 12, QFont.Bold))
        btn_about.setStyleSheet(nav_button_style)
        btn_about.clicked.connect(self.show_about_developers)
        sidebar_layout.addWidget(btn_about)

        # Content Area with QStackedWidget
        self.stacked_widget = QStackedWidget()

        # Create pages once and add them to the stacked widget
        self.welcome_page = self.create_welcome_page()
        self.add_employee_page = self.create_add_employee_page()
        self.attendance_page = self.create_attendance_page()
        self.check_employees_page = self.create_check_employees_page()
        self.about_developers_page = self.create_about_developers_page()

        self.stacked_widget.addWidget(self.welcome_page)
        self.stacked_widget.addWidget(self.add_employee_page)
        self.stacked_widget.addWidget(self.attendance_page)
        self.stacked_widget.addWidget(self.check_employees_page)
        self.stacked_widget.addWidget(self.about_developers_page)

        # Set initial page
        self.stacked_widget.setCurrentWidget(self.welcome_page)

        # Add sidebar and content area to main layout
        main_layout.addWidget(sidebar)
        main_layout.addWidget(self.stacked_widget)

    def create_welcome_page(self):
        page = QFrame()
        layout = QVBoxLayout()
        welcome_label = QLabel("Welcome to Attendance System")
        welcome_label.setFont(QFont("Arial", 24, QFont.Bold))
        welcome_label.setAlignment(Qt.AlignCenter)
        welcome_label.setStyleSheet("color: #1e3d59;")
        layout.addWidget(welcome_label)
        page.setLayout(layout)
        return page

    def create_add_employee_page(self):
        page = QFrame()
        layout = QHBoxLayout()

        # Left Frame: Form Inputs
        left_frame = QFrame()
        left_layout = QVBoxLayout()

        # Define input fields
        labels = ["Emp ID:", "Name:", "Email:", "Phone:", "Age:"]
        self.entries = {}
        for label in labels:
            group = QGroupBox(label)
            group.setStyleSheet("QGroupBox { font-size: 14px; font-weight: bold; }")
            hbox = QHBoxLayout()
            entry = QLineEdit()
            entry.setFont(QFont("Arial", 12))
            entry.setStyleSheet("padding: 5px; border: 1px solid #ccc; border-radius: 5px;")
            hbox.addWidget(entry)
            group.setLayout(hbox)
            left_layout.addWidget(group)
            self.entries[label] = entry

        # Department Dropdown
        department_group = QGroupBox("Department:")
        department_group.setStyleSheet("QGroupBox { font-size: 14px; font-weight: bold; }")
        dept_layout = QHBoxLayout()
        self.department_dropdown = QComboBox()
        self.department_dropdown.addItems(["Sales", "Marketing", "Development", "Security", "Other"])
        self.department_dropdown.setFont(QFont("Arial", 12))
        self.department_dropdown.setStyleSheet("padding: 5px; border: 1px solid #ccc; border-radius: 5px;")
        dept_layout.addWidget(self.department_dropdown)
        department_group.setLayout(dept_layout)
        left_layout.addWidget(department_group)

        # Role Dropdown
        role_group = QGroupBox("Role:")
        role_group.setStyleSheet("QGroupBox { font-size: 14px; font-weight: bold; }")
        role_layout = QHBoxLayout()
        self.role_dropdown = QComboBox()
        self.role_dropdown.setFont(QFont("Arial", 12))
        self.role_dropdown.setStyleSheet("padding: 5px; border: 1px solid #ccc; border-radius: 5px;")
        self.update_role_dropdown(self.department_dropdown.currentText())
        role_layout.addWidget(self.role_dropdown)
        role_group.setLayout(role_layout)
        left_layout.addWidget(role_group)

        # Update Role Dropdown on department change
        self.department_dropdown.currentIndexChanged.connect(
            lambda: self.update_role_dropdown(self.department_dropdown.currentText())
        )

        # Capture Image Button
        self.capture_btn = QPushButton("Capture Image")
        self.capture_btn.setToolTip("Capture images of the employee's face for training.")
        self.capture_btn.setFont(QFont("Arial", 12, QFont.Bold))
        self.capture_btn.setStyleSheet("background-color: #1e3d59; color: white; padding: 10px;")
        self.capture_btn.clicked.connect(self.capture_face)
        left_layout.addWidget(self.capture_btn)

        # Save & Train Model Button (initially hidden)
        self.save_train_btn = QPushButton("Save & Train Model")
        self.save_train_btn.setToolTip("Save the employee details and train the facial recognition model.")
        self.save_train_btn.setFont(QFont("Arial", 12, QFont.Bold))
        self.save_train_btn.setStyleSheet("background-color: #ff6b6b; color: white; padding: 10px;")
        self.save_train_btn.clicked.connect(self.save_and_train_model)
        self.save_train_btn.hide()
        left_layout.addWidget(self.save_train_btn)

        left_frame.setLayout(left_layout)

        # Right Frame: Camera Feed
        right_frame = QFrame()
        right_layout = QVBoxLayout()
        self.camera_label = QLabel()
        self.camera_label.setStyleSheet("background-color: black;")
        self.camera_label.setAlignment(Qt.AlignCenter)
        right_layout.addWidget(self.camera_label)
        right_frame.setLayout(right_layout)

        layout.addWidget(left_frame, 1)
        layout.addWidget(right_frame, 2)
        page.setLayout(layout)
        return page

    def update_role_dropdown(self, department):
        """Update the Role dropdown based on the selected department."""
        self.role_dropdown.clear()
        roles = ROLES_MAPPING.get(department, ["Other"])
        self.role_dropdown.addItems(roles)

    def create_attendance_page(self):
        page = QFrame()
        layout = QVBoxLayout()

        # Create a toggle button for starting/stopping attendance
        self.attendance_toggle_button = QPushButton("Start Marking Attendance")
        self.attendance_toggle_button.setToolTip("Start or stop marking attendance using facial recognition.")
        self.attendance_toggle_button.setFont(QFont("Arial", 12, QFont.Bold))
        self.attendance_toggle_button.setStyleSheet("background-color: #ff6b6b; color: white; padding: 10px;")
        self.attendance_toggle_button.clicked.connect(self.toggle_attendance)
        layout.addWidget(self.attendance_toggle_button)

        self.attendance_camera_label = QLabel()
        self.attendance_camera_label.setStyleSheet("background-color: black;")
        self.attendance_camera_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.attendance_camera_label)

        page.setLayout(layout)
        return page

    def create_check_employees_page(self):
        page = QFrame()
        layout = QVBoxLayout()

        # Search Bar
        search_layout = QHBoxLayout()
        self.search_entry = QLineEdit()
        self.search_entry.setPlaceholderText("Search by ID or Name...")
        self.search_entry.setFont(QFont("Arial", 12))
        self.search_entry.setStyleSheet("padding: 5px; border: 1px solid #ccc; border-radius: 5px;")
        search_layout.addWidget(self.search_entry)

        search_btn = QPushButton("Search")
        search_btn.setToolTip("Search for an employee by ID or name.")
        search_btn.setFont(QFont("Arial", 12, QFont.Bold))
        search_btn.setStyleSheet("background-color: #1e3d59; color: white; padding: 5px;")
        search_btn.clicked.connect(self.search_employee)
        search_layout.addWidget(search_btn)

        layout.addLayout(search_layout)

        # Table to Display Employees and Attendance
        self.employee_table = QTableWidget()
        self.employee_table.setColumnCount(7)
        self.employee_table.setHorizontalHeaderLabels(["Emp ID", "Name", "Email", "Phone", "Department", "Role", "Attendance"])
        self.employee_table.setStyleSheet("QTableWidget { gridline-color: #ccc; }")
        self.employee_table.horizontalHeader().setStretchLastSection(True)
        self.employee_table.horizontalHeader().setStyleSheet("QHeaderView::section { background-color: #1e3d59; color: white; }")
        layout.addWidget(self.employee_table)

        self.load_employee_data()  # Load initial data

        page.setLayout(layout)
        return page

    def create_about_developers_page(self):
        page = QFrame()
        layout = QVBoxLayout()

        title = QLabel("About the Developers")
        title.setFont(QFont("Arial", 22, QFont.Bold))
        title.setAlignment(Qt.AlignCenter)
        title.setStyleSheet("color: #1e3d59;")
        layout.addWidget(title)

        # Container widget with grid layout to hold developer cards
        container = QWidget()
        grid_layout = QGridLayout()
        grid_layout.setSpacing(20)
        container.setLayout(grid_layout)

        # Define developers with name, gender, and description of their work
        developers = [
            {"name": "Sonia", "gender": "female", "desc": "Real Time Attendance system integration."},
            {"name": "Fahad Aziz", "gender": "male", "desc": "Facial data collection and training."},
            {"name": "Mohid Asmat", "gender": "male", "desc": "Backend Developer managing server-side logic."},
            {"name": "David Wilson", "gender": "male", "desc": "Frontend Developer crafting intuitive user interfaces."},
            {"name": "Abdul Moeez", "gender": "male", "desc": "Frontend Developer creating responsive designs."}
        ]

        # Create a card for each developer using a vertical layout
        col_count = 3  # Number of columns in the grid
        row = 0
        col = 0
        for dev in developers:
            card = QFrame()
            card.setStyleSheet("""
            QFrame {
            border: 1px solid #ccc;
            border-radius: 10px;
            background-color: #ffffff;
            padding: 15px;
            }
            """)
            card_layout = QVBoxLayout()
            card_layout.setAlignment(Qt.AlignCenter)

            # Icon at top center
            icon = "👩" if dev["gender"] == "female" else "👨"
            icon_label = QLabel(icon)
            icon_label.setFont(QFont("Arial", 48))
            icon_label.setAlignment(Qt.AlignCenter)
            card_layout.addWidget(icon_label)

            # Developer Name
            name_label = QLabel(dev["name"])
            name_label.setFont(QFont("Arial", 16, QFont.Bold))
            name_label.setAlignment(Qt.AlignCenter)
            card_layout.addWidget(name_label)

            # Description of their work
            desc_label = QLabel(dev["desc"])
            desc_label.setFont(QFont("Arial", 12))
            desc_label.setWordWrap(True)
            desc_label.setAlignment(Qt.AlignCenter)
            card_layout.addWidget(desc_label)

            card.setLayout(card_layout)
            grid_layout.addWidget(card, row, col)
            col += 1
            if col >= col_count:
                col = 0
                row += 1

        # Wrap the container in a scroll area in case content exceeds the window height
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(container)
        layout.addWidget(scroll)

        page.setLayout(layout)
        return page

    def load_employee_data(self, search_query=""):
        conn = sqlite3.connect("attendance.db")
        cursor = conn.cursor()
        if search_query:
            cursor.execute("""
            SELECT e.emp_id, e.name, e.email, e.phone, e.department, e.role, COUNT(a.emp_id) AS attendance_count
            FROM employees e
            LEFT JOIN attendance a ON e.emp_id = a.emp_id
            WHERE e.emp_id LIKE ? OR e.name LIKE ?
            GROUP BY e.emp_id
            """, (f"%{search_query}%", f"%{search_query}%"))
        else:
            cursor.execute("""
            SELECT e.emp_id, e.name, e.email, e.phone, e.department, e.role, COUNT(a.emp_id) AS attendance_count
            FROM employees e
            LEFT JOIN attendance a ON e.emp_id = a.emp_id
            GROUP BY e.emp_id
            """)
        employees = cursor.fetchall()
        conn.close()

        self.employee_table.setRowCount(len(employees))
        for row, emp in enumerate(employees):
            for col, value in enumerate(emp):
                item = QTableWidgetItem(str(value))
                item.setTextAlignment(Qt.AlignCenter)
                self.employee_table.setItem(row, col, item)

    def search_employee(self):
        search_query = self.search_entry.text().strip()
        self.load_employee_data(search_query)

    def validate_inputs(self):
        emp_id = self.entries["Emp ID:"].text().strip()
        name = self.entries["Name:"].text().strip()
        email = self.entries["Email:"].text().strip()
        phone = self.entries["Phone:"].text().strip()
        age = self.entries["Age:"].text().strip()
        department = self.department_dropdown.currentText()
        role = self.role_dropdown.currentText()

        if not all([emp_id, name, email, phone, age, department, role]):
            QMessageBox.critical(self, "Validation Error", "All fields are required!")
            return False

        if not emp_id.isalnum():
            QMessageBox.critical(self, "Validation Error", "Employee ID must be alphanumeric!")
            return False

        if "@" not in email or "." not in email:
            QMessageBox.critical(self, "Validation Error", "Invalid email format!")
            return False

        if not phone.isdigit() or len(phone) != 11:
            QMessageBox.critical(self, "Validation Error", "Phone number must be 11 digits!")
            return False

        if not age.isdigit() or int(age) < 18 or int(age) > 100:
            QMessageBox.critical(self, "Validation Error", "Age must be between 18 and 100!")
            return False

        conn = sqlite3.connect("attendance.db")
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM employees WHERE emp_id=?", (emp_id,))
        if cursor.fetchone():
            conn.close()
            QMessageBox.critical(self, "Validation Error", "Employee ID already exists!")
            return False
        conn.close()
        return True

    def capture_face(self):
        if not self.validate_inputs():
            return

        self.emp_id = self.entries["Emp ID:"].text().strip()
        os.makedirs(DATASET_PATH, exist_ok=True)
        self.emp_folder = os.path.join(DATASET_PATH, self.emp_id)
        os.makedirs(self.emp_folder, exist_ok=True)

        self.cap = cv2.VideoCapture(0)
        self.count = 0
        self.face_timer = QTimer()
        self.face_timer.timeout.connect(self.update_frame)
        self.face_timer.start(10)
        self.capture_btn.hide()

    def update_frame(self):
        ret, frame = self.cap.read()
        if ret:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, 1.3, 5)
            for (x, y, w, h) in faces:
                if self.count < 50:
                    self.count += 1
                    face_img = gray[y:y+h, x:x+w]

                    # Simple data augmentation: flip the image horizontally
                    flipped_face_img = cv2.flip(face_img, 1)

                    img_path = os.path.join(self.emp_folder, f"{self.count}.jpg")
                    flipped_img_path = os.path.join(self.emp_folder, f"flipped_{self.count}.jpg")

                    cv2.imwrite(img_path, face_img)
                    cv2.imwrite(flipped_img_path, flipped_face_img)

                    cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)

                rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                h, w, ch = rgb_image.shape
                bytes_per_line = ch * w
                q_img = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888)
                self.camera_label.setPixmap(QPixmap.fromImage(q_img))

            if self.count >= 50:
                self.face_timer.stop()
                self.cap.release()
                QMessageBox.information(self, "Success", "Images Captured Successfully!")
                self.save_train_btn.show()

    def save_and_train_model(self):
        emp_id = self.entries["Emp ID:"].text().strip()
        name = self.entries["Name:"].text().strip()
        email = self.entries["Email:"].text().strip()
        phone = self.entries["Phone:"].text().strip()
        department = self.department_dropdown.currentText()
        role = self.role_dropdown.currentText()
        age = self.entries["Age:"].text().strip()

        # If the file does not exist, create it with default "nil" for attendance
        if not os.path.exists(ATTENDANCE_FILE):
            df = pd.DataFrame(columns=["Emp ID", "Name"])
            df.to_excel(ATTENDANCE_FILE, index=False)

        df = pd.read_excel(ATTENDANCE_FILE)
        if emp_id not in df["Emp ID"].values:
            new_row = {"Emp ID": emp_id, "Name": name}
            df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
            df.to_excel(ATTENDANCE_FILE, index=False)

        conn = sqlite3.connect("attendance.db")
        cursor = conn.cursor()
        cursor.execute("""
        INSERT OR REPLACE INTO employees (emp_id, name, email, phone, department, role, age, folder) 
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, (emp_id, name, email, phone, department, role, age, f"{DATASET_PATH}/{emp_id}"))
        conn.commit()
        conn.close()

        # Train the face recognizer
        recognizer = cv2.face.LBPHFaceRecognizer_create()
        faces, ids = [], []
        for emp_folder in os.listdir(DATASET_PATH):
            emp_path = os.path.join(DATASET_PATH, emp_folder)
            for img_name in os.listdir(emp_path):
                img_path = os.path.join(emp_path, img_name)
                img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                faces.append(np.array(img, dtype=np.uint8))
                ids.append(int(emp_folder))

        if faces and ids:
            recognizer.train(faces, np.array(ids))
            recognizer.save(MODEL_FILE)
            QMessageBox.information(self, "Success", "Model Trained Successfully!")

        # Clear fields after saving
        for key in self.entries:
            self.entries[key].clear()
        self.department_dropdown.setCurrentIndex(0)
        self.role_dropdown.clear()
        self.save_train_btn.hide()
        self.capture_btn.show()

        self.stacked_widget.setCurrentWidget(self.welcome_page)

    def start_attendance(self):
        """Start camera feed and attendance update."""
        # Open camera
        self.cap = cv2.VideoCapture(0)
        # Change button text to Stop Attendance
        self.attendance_toggle_button.setText("Stop Attendance")
        self.attendance_running = True
        # Start timer to update attendance frame
        self.attendance_timer = QTimer()
        self.attendance_timer.timeout.connect(lambda: self.update_attendance_frame(self.load_recognizer(), self.get_employees(), ATTENDANCE_FILE))
        self.attendance_timer.start(10)

    def stop_attendance(self):
        """Stop camera feed and hide camera label."""
        if self.attendance_timer:
            self.attendance_timer.stop()
        if self.cap:
            self.cap.release()
        self.attendance_camera_label.clear()
        self.attendance_toggle_button.setText("Start Marking Attendance")
        self.attendance_running = False

    def toggle_attendance(self):
        """Toggle the attendance camera on/off."""
        # Switch to attendance page if not already there
        self.stacked_widget.setCurrentWidget(self.attendance_page)
        if not self.attendance_running:
            self.start_attendance()
        else:
            self.stop_attendance()

    def load_recognizer(self):
        recognizer = cv2.face.LBPHFaceRecognizer_create()
        try:
            recognizer.read(MODEL_FILE)
        except Exception as e:
            QMessageBox.critical(self, "Error", "Model not found! Please train the model first.")
            self.stop_attendance()
        return recognizer

    def get_employees(self):
        conn = sqlite3.connect("attendance.db")
        cursor = conn.cursor()
        cursor.execute("SELECT emp_id, name, department, role FROM employees")
        employees = cursor.fetchall()
        conn.close()
        return employees

    def update_attendance_frame(self, recognizer, employees, excel_file):
        ret, frame = self.cap.read()
        if ret:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, 1.3, 5)
            current_time = datetime.datetime.now().time()

            # Define attendance time window: 8:00 AM to 2:00 PM
            start_time = datetime.time(8, 0)
            end_time = datetime.time(22, 0)
            current_date = datetime.datetime.now().strftime("%Y-%m-%d")
            df = pd.read_excel(excel_file)

            # Ensure current date column exists
            if current_date not in df.columns:
                df[current_date] = "nil"

            for (x, y, w, h) in faces:
                face_img = gray[y:y+h, x:x+w]
                try:
                    emp_id, confidence = recognizer.predict(face_img)
                except Exception as e:
                    emp_id, confidence = None, 100

                if confidence < 70 and emp_id is not None:
                    emp_info = next((emp for emp in employees if emp[0] == str(emp_id)), None)
                    if emp_info:
                        name = emp_info[1]
                        role_text = emp_info[3]

                        # Determine attendance status based on current time:
                        if current_time < start_time:
                            status = "nil"
                        elif current_time <= end_time:
                            status = "Present"
                        else:
                            status = "Absent"

                        # Choose color and label text based on status
                        if status == "Present":
                            color = (0, 255, 0)
                            label = f"ID: {emp_id}, {name}, {role_text} - Present"
                        elif status == "Absent":
                            color = (0, 0, 255)
                            label = f"ID: {emp_id}, {name}, {role_text} - Absent"
                        else:
                            color = (128, 128, 128)
                            label = f"ID: {emp_id}, {name}, {role_text}"

                        # Update Excel only if status is Present or Absent
                        if status != "nil":
                            if emp_id not in df["Emp ID"].values:
                                new_row = {"Emp ID": emp_id, "Name": name, current_date: status}
                                df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
                            else:
                                df.loc[df["Emp ID"] == emp_id, current_date] = status
                            df.to_excel(excel_file, index=False)
                    else:
                        label = "Unknown"
                        color = (0, 0, 255)
                else:
                    label = "Unknown"
                    color = (0, 0, 255)

                cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
                cv2.putText(frame, label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

            rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            h, w, ch = rgb_image.shape
            bytes_per_line = ch * w
            q_img = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888)
            self.attendance_camera_label.setPixmap(QPixmap.fromImage(q_img))

    # --- Page Switching Methods ---
    def show_add_employee(self):
        self.stacked_widget.setCurrentWidget(self.add_employee_page)

    def show_attendance_page(self):
        self.stacked_widget.setCurrentWidget(self.attendance_page)

    def show_check_employees(self):
        self.load_employee_data()
        self.stacked_widget.setCurrentWidget(self.check_employees_page)

    def show_about_developers(self):
        self.stacked_widget.setCurrentWidget(self.about_developers_page)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = AttendanceSystem()
    window.show()
    sys.exit(app.exec_())