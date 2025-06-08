import cv2
import face_recognition
import os
import csv
from datetime import datetime
import numpy as np
import time
import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk
import threading


class FacialRecognitionGUI:
    def __init__(self, master):
        self.master = master
        self.master.title("Facial Recognition System")
        self.master.geometry("1200x700")
        self.master.configure(bg="#f0f0f0")
        self.master.resizable(True, True)

        # Create the recognition system in the background
        self.system = FacialRecognitionSystem(reset_logs=False)

        # Setup UI elements
        self.setup_ui()

        # Flag for running recognition
        self.is_running = False
        self.video_thread = None

    def setup_ui(self):
        # Create header frame
        self.header_frame = tk.Frame(self.master, bg="#2c3e50", height=60)
        self.header_frame.pack(fill=tk.X, pady=0)

        # App title
        tk.Label(self.header_frame, text="Advanced Facial Recognition System",
                 font=("Arial", 18, "bold"), bg="#2c3e50", fg="white").pack(pady=10)

        # Main content frame
        self.content_frame = tk.Frame(self.master, bg="#f0f0f0")
        self.content_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=10)

        # Video frame (left side)
        self.video_frame = tk.Frame(self.content_frame, bg="#d0d0d0", width=800, height=600)
        self.video_frame.pack(side=tk.LEFT, padx=10, pady=10, fill=tk.BOTH, expand=True)

        # Create canvas for video display
        self.canvas = tk.Canvas(self.video_frame, bg="black", width=800, height=600)
        self.canvas.pack(fill=tk.BOTH, expand=True)

        # Controls frame (right side)
        self.controls_frame = tk.Frame(self.content_frame, bg="#e0e0e0", width=300)
        self.controls_frame.pack(side=tk.RIGHT, padx=10, pady=10, fill=tk.BOTH)

        # Status section
        self.status_frame = tk.LabelFrame(self.controls_frame, text="System Status", bg="#e0e0e0",
                                          font=("Arial", 10, "bold"), padx=10, pady=10)
        self.status_frame.pack(fill=tk.X, padx=10, pady=10)

        # Status indicators
        self.camera_status = tk.Label(self.status_frame, text="Camera: Inactive",
                                      bg="#e0e0e0", fg="#FF5733", font=("Arial", 10))
        self.camera_status.pack(anchor=tk.W, pady=2)

        self.faces_loaded = tk.Label(self.status_frame, text="Loaded Faces: 0",
                                     bg="#e0e0e0", fg="#333333", font=("Arial", 10))
        self.faces_loaded.pack(anchor=tk.W, pady=2)

        self.recognition_status = tk.Label(self.status_frame, text="Recognition: Standby",
                                           bg="#e0e0e0", fg="#333333", font=("Arial", 10))
        self.recognition_status.pack(anchor=tk.W, pady=2)

        # Latest events section
        self.events_frame = tk.LabelFrame(self.controls_frame, text="Latest Events", bg="#e0e0e0",
                                          font=("Arial", 10, "bold"), padx=10, pady=10)
        self.events_frame.pack(fill=tk.X, padx=10, pady=10)

        # Events log (with scrollbar)
        self.events_log = tk.Text(self.events_frame, height=8, width=30, bg="white",
                                  font=("Consolas", 9), state=tk.DISABLED)
        self.events_log.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        scrollbar = tk.Scrollbar(self.events_frame, command=self.events_log.yview)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.events_log.config(yscrollcommand=scrollbar.set)

        # Controls section
        self.buttons_frame = tk.LabelFrame(self.controls_frame, text="Controls", bg="#e0e0e0",
                                           font=("Arial", 10, "bold"), padx=10, pady=10)
        self.buttons_frame.pack(fill=tk.X, padx=10, pady=10)

        # Button style
        style = ttk.Style()
        style.configure("TButton", font=("Arial", 10), padding=6)

        # Control buttons
        self.start_button = ttk.Button(self.buttons_frame, text="Start Recognition",
                                       command=self.start_recognition)
        self.start_button.pack(fill=tk.X, pady=5)

        self.stop_button = ttk.Button(self.buttons_frame, text="Stop Recognition",
                                      command=self.stop_recognition, state=tk.DISABLED)
        self.stop_button.pack(fill=tk.X, pady=5)

        self.reset_logs_button = ttk.Button(self.buttons_frame, text="Reset Logs",
                                            command=self.reset_logs)
        self.reset_logs_button.pack(fill=tk.X, pady=5)

        # Footer
        self.footer_frame = tk.Frame(self.master, bg="#2c3e50", height=30)
        self.footer_frame.pack(fill=tk.X, side=tk.BOTTOM)

        tk.Label(self.footer_frame, text="© 2025 Advanced Security Systems",
                 font=("Arial", 8), bg="#2c3e50", fg="white").pack(pady=5)

        # Update loaded faces count
        self.faces_loaded.config(text=f"Loaded Faces: {len(self.system.known_face_names)}")

    def log_event(self, message):
        """Add event message to the events log with timestamp"""
        timestamp = datetime.now().strftime("%H:%M:%S")
        log_entry = f"[{timestamp}] {message}\n"

        self.events_log.config(state=tk.NORMAL)
        self.events_log.insert(tk.END, log_entry)
        self.events_log.see(tk.END)  # Auto-scroll to the end
        self.events_log.config(state=tk.DISABLED)

    def start_recognition(self):
        """Start the recognition process in a separate thread"""
        if not self.is_running:
            self.is_running = True
            self.video_thread = threading.Thread(target=self.recognition_thread)
            self.video_thread.daemon = True
            self.video_thread.start()

            # Update UI
            self.start_button.config(state=tk.DISABLED)
            self.stop_button.config(state=tk.NORMAL)
            self.camera_status.config(text="Camera: Active", fg="#28a745")
            self.recognition_status.config(text="Recognition: Running", fg="#28a745")
            self.log_event("Recognition system started")

    def stop_recognition(self):
        """Stop the recognition process"""
        if self.is_running:
            self.is_running = False
            # Thread will exit on next iteration

            # Update UI
            self.start_button.config(state=tk.NORMAL)
            self.stop_button.config(state=tk.DISABLED)
            self.camera_status.config(text="Camera: Inactive", fg="#FF5733")
            self.recognition_status.config(text="Recognition: Standby", fg="#333333")
            self.log_event("Recognition system stopped")

    def reset_logs(self):
        """Reset system logs"""
        self.system.reset_logs()
        self.log_event("Log file has been reset")

    def recognition_thread(self):
        """Run face recognition processing in a separate thread"""
        # Open video capture
        video_capture = cv2.VideoCapture(0)

        if not video_capture.isOpened():
            self.log_event("ERROR: Could not open webcam!")
            self.stop_recognition()
            return

        # To track recognized students
        recognized_students = set()
        last_student_time = {}
        last_intruder_time = 0

        # Cooldown settings
        in_cooldown = False
        cooldown_end_time = 0
        cooldown_duration = 2  # seconds

        # For intruder tracking
        intruder_photo_saved = False

        while self.is_running:
            # Capture frame
            ret, frame = video_capture.read()
            if not ret:
                self.log_event("ERROR: Failed to grab frame!")
                break

            display_frame = frame.copy()

            # Check cooldown period
            current_time = time.time()
            if in_cooldown and current_time < cooldown_end_time:
                # Add processing indicator
                self.add_overlay_text(display_frame, "Processing...", position=(20, 40),
                                      color=(0, 120, 255), size=0.8, thickness=2)
            else:
                in_cooldown = False

                # Process faces
                # Resize frame for faster processing
                small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
                rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

                # Find faces
                face_locations = face_recognition.face_locations(rgb_small_frame)
                face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

                # Process each face
                for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
                    # Scale back face locations
                    top *= 4
                    right *= 4
                    bottom *= 4
                    left *= 4

                    # Check if face matches known faces
                    matches = face_recognition.compare_faces(self.system.known_face_encodings, face_encoding)

                    if True in matches:
                        # Find best match
                        face_distances = face_recognition.face_distance(self.system.known_face_encodings, face_encoding)
                        best_match_index = np.argmin(face_distances)

                        if matches[best_match_index]:
                            name = self.system.known_face_names[best_match_index]
                            confidence = (1 - face_distances[best_match_index]) * 100

                            # Check for cooldown
                            current_time = time.time()
                            if name in last_student_time:
                                time_since_last = current_time - last_student_time[name]
                                if time_since_last < cooldown_duration:
                                    # Just draw but don't log
                                    self.draw_face_box(display_frame, left, top, right, bottom,
                                                       name, confidence, is_known=True)
                                    continue

                            # Log new arrival
                            if name not in recognized_students:
                                self.system.log_arrival(name)
                                recognized_students.add(name)
                                self.log_event(f"Welcome {name}!")

                                # Set cooldown
                                in_cooldown = True
                                cooldown_end_time = current_time + cooldown_duration

                            # Update timestamp
                            last_student_time[name] = current_time

                            # Draw face box with name
                            self.draw_face_box(display_frame, left, top, right, bottom,
                                               name, confidence, is_known=True)
                    else:
                        # Unknown face
                        self.draw_face_box(display_frame, left, top, right, bottom,
                                           "Unknown", 0, is_known=False)

                        # Save intruder image once per session
                        current_time = time.time()
                        if not intruder_photo_saved:
                            self.system.save_intruder_image(frame)
                            intruder_photo_saved = True
                            last_intruder_time = current_time
                            self.log_event("⚠️ Intruder detected!")

                            # Set cooldown
                            in_cooldown = True
                            cooldown_end_time = current_time + cooldown_duration

            # Add system info overlay
            self.add_system_info(display_frame)

            # Convert to PIL format for tkinter
            display_frame = cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(display_frame)

            # Resize to fit canvas if needed
            canvas_width = self.canvas.winfo_width()
            canvas_height = self.canvas.winfo_height()
            if canvas_width > 1 and canvas_height > 1:  # Ensure valid dimensions
                img = img.resize((canvas_width, canvas_height), Image.LANCZOS)

            imgtk = ImageTk.PhotoImage(image=img)

            # Update canvas with new image
            self.canvas.create_image(0, 0, anchor=tk.NW, image=imgtk)
            self.canvas.image = imgtk  # Keep reference

            # Process GUI events
            self.master.update_idletasks()
            self.master.update()

            # Short delay
            time.sleep(0.03)

        # Release video when done
        video_capture.release()

    def draw_face_box(self, frame, left, top, right, bottom, name, confidence=0, is_known=True):
        """Draw a professional looking face detection box"""
        # Colors based on known/unknown status
        if is_known:
            box_color = (0, 170, 0)  # Green
            text_bg_color = (0, 170, 0)
        else:
            box_color = (0, 0, 220)  # Red
            text_bg_color = (0, 0, 220)

        # Draw main rectangle with rounded corners and thicker line
        cv2.rectangle(frame, (left, top), (right, bottom), box_color, 2)

        # Create a filled rectangle for text background
        conf_text = f" {confidence:.1f}%" if confidence > 0 else ""
        label = f"{name}{conf_text}"

        # Calculate label width for background
        (text_width, text_height), baseline = cv2.getTextSize(
            label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)

        # Draw label background
        cv2.rectangle(
            frame,
            (left, top - text_height - 10),
            (left + text_width + 10, top),
            text_bg_color,
            cv2.FILLED
        )

        # Draw label text
        cv2.putText(
            frame,
            label,
            (left + 5, top - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            1
        )

    def add_overlay_text(self, frame, text, position, color=(255, 255, 255), size=0.7, thickness=1):
        """Add text overlay with shadow effect for better visibility"""
        x, y = position
        # Draw shadow first (slight offset)
        cv2.putText(frame, text, (x + 1, y + 1), cv2.FONT_HERSHEY_SIMPLEX, size, (0, 0, 0), thickness + 1)
        # Draw main text
        cv2.putText(frame, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, size, color, thickness)

    def add_system_info(self, frame):
        """Add system information overlay to the video frame"""
        height, width = frame.shape[:2]

        # Semi-transparent overlay background
        overlay = frame.copy()
        cv2.rectangle(overlay, (5, height - 85), (250, height - 5), (30, 30, 30), cv2.FILLED)
        cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)

        # System info text
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self.add_overlay_text(frame, f"Date: {now}", (10, height - 60), (200, 200, 200))
        self.add_overlay_text(frame, f"Faces Loaded: {len(self.system.known_face_names)}",
                              (10, height - 40), (200, 200, 200))
        self.add_overlay_text(frame, f"System: Running", (10, height - 20), (0, 255, 128))

        # Add logo/watermark
        cv2.putText(frame, "SECURITY", (width - 120, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 150, 255), 2)


class FacialRecognitionSystem:
    def __init__(self, reset_logs=False):
        self.known_face_encodings = []
        self.known_face_names = []
        self.known_faces_dir = "known_faces"
        self.logs_dir = "logs"
        self.intruder_dir = "intruder"
        self.log_file = os.path.join(self.logs_dir, "arrival_logs.csv")
        self.intruder_log_file= os.path.join(self.logs_dir, "intruder_logs.csv")
        self.log_counter = 1

        # Create required directories
        for directory in [self.known_faces_dir, self.logs_dir, self.intruder_dir]:
            if not os.path.exists(directory):
                os.makedirs(directory)
                print(f"Created directory: {directory}")

        # Create or reset log files
        self.init_log_file(reset=reset_logs)
        self.init_intruder_log_file(reset=reset_logs)

        # Initialize log counter
        if not reset_logs:
            self.initialize_log_counter()

        # Load known faces
        self.load_known_faces()

    def init_log_file(self, reset=False):
        """Initialize the log file with headers, optionally resetting the file."""
        needs_headers = reset or not os.path.exists(self.log_file) or os.path.getsize(self.log_file) == 0

        if needs_headers:
            try:
                with open(self.log_file, 'w', newline='') as file:
                    writer = csv.writer(file)
                    writer.writerow(['Log No.', 'Roll no.', 'Date', 'Time'])
                print(f"{'Reset' if reset else 'Created'} log file: {self.log_file}")
                if reset:
                    self.log_counter = 1
            except Exception as e:
                print(f"Error {'resetting' if reset else 'creating'} log file: {e}")
        else:
            print(f"Log file exists at: {self.log_file}")

    def init_intruder_log_file(self, reset=False):
        """Initialize the intruder log file with headers, optionally resetting the file."""
        needs_headers = reset or not os.path.exists(self.intruder_log_file) or os.path.getsize(
            self.intruder_log_file) == 0

        if needs_headers:
            try:
                with open(self.intruder_log_file, 'w', newline='') as file:
                    writer = csv.writer(file)
                    writer.writerow(['Date', 'Time', 'Image Path'])
                print(f"{'Reset' if reset else 'Created'} intruder log file: {self.intruder_log_file}")
            except Exception as e:
                print(f"Error {'resetting' if reset else 'creating'} intruder log file: {e}")
        else:
            print(f"Intruder log file exists at: {self.intruder_log_file}")

    def initialize_log_counter(self):
        """Initialize log counter based on existing log entries."""
        try:
            if os.path.exists(self.log_file) and os.path.getsize(self.log_file) > 0:
                with open(self.log_file, 'r', newline='') as file:
                    reader = csv.reader(file)
                    rows = list(reader)

                    if len(rows) > 1:
                        try:
                            log_numbers = [int(row[0]) for row in rows[1:] if row and row[0].isdigit()]
                            if log_numbers:
                                self.log_counter = max(log_numbers) + 1
                        except Exception as e:
                            print(f"Error parsing existing log numbers: {e}")
        except Exception as e:
            print(f"Error initializing log counter: {e}")

    def reset_logs(self):
        """Reset log files and counter to start from 1."""
        self.init_log_file(reset=True)
        self.init_intruder_log_file(reset=True)
        print("Log files have been reset. Numbering will start from 1.")

    def load_known_faces(self):
        """Load known faces from the known_faces directory."""
        print("Loading known faces...")

        for filename in os.listdir(self.known_faces_dir):
            if filename.endswith(('.png', '.jpg', '.jpeg')):
                name = os.path.splitext(filename)[0]
                image_path = os.path.join(self.known_faces_dir, filename)

                try:
                    image = face_recognition.load_image_file(image_path)
                    encoding = face_recognition.face_encodings(image)[0]
                    self.known_face_encodings.append(encoding)
                    self.known_face_names.append(name)
                    print(f"Loaded face: {name}")
                except IndexError:
                    print(f"No face found in {filename}. Skipping.")
                except Exception as e:
                    print(f"Error loading {filename}: {e}")

        print(f"Loaded {len(self.known_face_names)} known faces")

    def log_arrival(self, name):
        """Log student arrival with timestamp and log number to CSV."""
        now = datetime.now()
        date_str = now.strftime("%Y-%m-%d")
        time_str = now.strftime("%H:%M:%S")
        log_number = self.log_counter
        self.log_counter += 1

        # Print welcome message to console
        print(f"Welcome to CGC, {name}! - {date_str} {time_str}")

        # Write to CSV file
        try:
            if not os.path.exists(self.log_file):
                with open(self.log_file, 'w', newline='') as file:
                    writer = csv.writer(file)
                    writer.writerow(['Log No.', 'Roll no.', 'Date', 'Time'])

            with open(self.log_file, 'a', newline='') as file:
                writer = csv.writer(file)
                writer.writerow([log_number, name, date_str, time_str])
        except Exception as e:
            print(f"Error writing to log file: {e}")

    def save_intruder_image(self, frame):
        """Save unknown face as intruder with timestamp and log the detection."""
        now = datetime.now()
        timestamp = now.strftime("%Y%m%d_%H%M%S")
        filename = f"intruder_{timestamp}.jpg"
        filepath = os.path.join(self.intruder_dir, filename)
        date_str = now.strftime("%Y-%m-%d")
        time_str = now.strftime("%H:%M:%S")

        try:
            # Save the intruder image
            cv2.imwrite(filepath, frame)
            print(f"Intruder detected at {now.strftime('%Y-%m-%d %H:%M:%S')}")

            # Log the intruder detection to CSV
            with open(self.intruder_log_file, 'a', newline='') as file:
                writer = csv.writer(file)
                writer.writerow([date_str, time_str, filepath])

        except Exception as e:
            print(f"Error saving intruder image or log: {e}")


if __name__ == "__main__":
    # Create Tkinter root window
    root = tk.Tk()

    # Create the app
    app = FacialRecognitionGUI(root)

    # Start the Tkinter event loop
    root.mainloop()