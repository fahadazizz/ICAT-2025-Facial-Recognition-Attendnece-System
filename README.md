# Real-Time Facial Recognition Attendance System

This project is a **Real-Time Facial Recognition Attendance System** built using Python, OpenCV, PyQt5, SQLite, and other libraries. It allows organizations to manage employee attendance through facial recognition technology, ensuring accurate and efficient tracking of employee presence.

---

## Table of Contents

1. [Features](#features)
2. [Installation](#installation)
3. [Usage](#usage)
4. [Project Structure](#project-structure)
5. [Database Schema](#database-schema)
6. [Key Functionalities](#key-functionalities)
7. [Dependencies](#dependencies)
8. [Contributors](#contributors)
9. [License](#license)

---

## Features

- **Employee Management**: Add new employees with details such as ID, name, email, phone, department, role, and age.
- **Facial Data Collection**: Capture images of employees' faces for training the facial recognition model.
- **Facial Recognition**: Use OpenCV's Haar Cascade and LBPHFaceRecognizer for real-time face detection and recognition.
- **Attendance Tracking**: Mark attendance based on facial recognition during specified time windows (e.g., 8:00 AM to 10:00 PM).
- **Attendance Reports**: Generate monthly attendance reports in Excel format.
- **Search Functionality**: Search for employees by ID or name in the employee list.
- **Responsive UI**: A clean and intuitive user interface built using PyQt5.

---

## Installation

### Prerequisites

- Python 3.7 or higher
- Git (optional, for cloning the repository)

### Steps

1. Clone the repository (if applicable):
   ```bash
   git clone https://github.com/your-repo-url.git
   cd your-repo-directory
   ```

2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Run the application:
   ```bash
   python main.py
   ```

---

## Usage

### Adding an Employee

1. Navigate to the **Add Employee** page from the sidebar.
2. Fill in the employee details (Emp ID, Name, Email, Phone, Age, Department, Role).
3. Click **Capture Image** to capture the employee's face for training.
4. Once the images are captured, click **Save & Train Model** to save the employee's data and train the facial recognition model.

### Marking Attendance

1. Navigate to the **Mark Attendance** page.
2. Click **Start Marking Attendance** to begin the facial recognition process.
3. The system will detect faces in real-time and mark attendance based on the current time window.
4. Click **Stop Attendance** to stop the process.

### Checking Employees

1. Navigate to the **Check Employees** page.
2. View the list of registered employees and their attendance records.
3. Use the search bar to filter employees by ID or name.

### About Developers

Navigate to the **About Developers** page to learn about the contributors to this project.

---

## Project Structure

```
.
├── dataset/                # Folder to store employee facial images
├── face_trainer.yml         # Trained facial recognition model
├── monthly_attendance.xlsx  # Monthly attendance report
├── attendance.db            # SQLite database for storing employee and attendance data
├── main.py                  # Main application file
└── requirements.txt         # List of Python dependencies
```

---

## Database Schema

### `employees` Table

| Column    | Type    | Description                          |
|-----------|---------|--------------------------------------|
| emp_id    | TEXT    | Unique employee ID                   |
| name      | TEXT    | Employee name                        |
| email     | TEXT    | Employee email                       |
| phone     | TEXT    | Employee phone number                |
| department| TEXT    | Employee department                  |
| role      | TEXT    | Employee role                        |
| age       | TEXT    | Employee age                         |
| folder    | TEXT    | Path to the employee's facial images |

### `attendance` Table

| Column    | Type    | Description                          |
|-----------|---------|--------------------------------------|
| id        | INTEGER | Auto-incremented primary key         |
| emp_id    | TEXT    | Employee ID                          |
| date      | TEXT    | Date of attendance                   |

---

## Key Functionalities

### 1. Employee Management

- Add new employees with details such as ID, name, email, phone, department, role, and age.
- Validate input fields to ensure data integrity.

### 2. Facial Data Collection

- Capture 50 images of an employee's face, including augmented flipped images for better training.
- Save images in a dedicated folder for each employee.

### 3. Facial Recognition

- Use OpenCV's Haar Cascade for face detection.
- Train the LBPHFaceRecognizer model with the captured images.
- Recognize faces in real-time and match them with employee IDs.

### 4. Attendance Tracking

- Mark attendance based on facial recognition during specified time windows.
- Update attendance status (`Present`, `Absent`, or `nil`) in the database and Excel file.

### 5. Attendance Reports

- Generate monthly attendance reports in Excel format.
- Include columns for employee ID, name, and daily attendance status.

---

## Dependencies

The following Python libraries are required:

- `PyQt5`: For building the graphical user interface.
- `opencv-python`: For facial detection and recognition.
- `sqlite3`: For managing the local database.
- `pandas`: For handling Excel files.
- `numpy`: For numerical operations.

Install dependencies using:
```bash
pip install PyQt5 opencv-python pandas numpy
```

---

## Contributors

This project was developed by the following team members:

- **Sonia**: Real-time attendance system integration.
- **Fahad Aziz**: Facial data collection and training.
- **Mohid Asmat**: Backend developer managing server-side logic.
- **David Wilson**: Frontend developer crafting intuitive user interfaces.
- **Abdul Moeez**: Frontend developer creating responsive designs.

---

## License

This project is licensed under the **MIT License**. See the `LICENSE` file for more details.

---

Feel free to contribute to this project or report issues on the GitHub repository. Thank you for using the **Real-Time Facial Recognition Attendance System**!
