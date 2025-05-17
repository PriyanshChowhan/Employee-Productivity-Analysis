import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import xgboost as xgb
import cv2
import time
import math
import matplotlib.pyplot as plt
import seaborn as sns

def capture_posture_score(duration=10):
    """
    Captures video of a sitting person's face for the specified duration (default 10 seconds) 
    and returns a productivity score based on face posture throughout the duration.
    
    The productivity score evaluates:
    - Head position stability (excessive movement indicates distraction)
    - Head angle (looking down indicates potential lack of focus or poor posture)
    - Face presence consistency (absence indicates stepping away)
    
    Returns:
        dict: A dictionary containing average productivity score and scores over time
    """
    # Initialize webcam
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return None
    
    # Load face detector
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    
    # Parameters for productivity assessment
    frame_rate = cap.get(cv2.CAP_PROP_FPS)
    if frame_rate == 0:  # If FPS detection fails, use a reasonable default
        frame_rate = 30
    
    # Calculate frames needed
    total_frames = int(duration * frame_rate)
    
    print("Preparing camera...")
    time.sleep(1)
    
    print(f"Starting productivity assessment for {duration} seconds...")
    print("Please proceed with your normal work activity.")
    
    # Initialize variables for tracking
    start_time = time.time()
    end_time = start_time + duration
    frame_count = 0
    
    # Lists to store results
    productivity_scores = []
    timestamps = []
    
    # Variables for face tracking
    previous_face_center = None
    previous_face_size = None
    face_detected_count = 0
    
    # Process video for the specified duration
    while time.time() < end_time:
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to capture frame.")
            break
        
        # Convert to grayscale for face detection
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Detect faces
        faces = face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30, 30)
        )
        
        current_time = time.time() - start_time
        
        # Initialize score components
        position_stability = 100  # Start with perfect score
        orientation_score = 100
        presence_score = 0
        
        # If face is detected, analyze posture
        if len(faces) > 0:
            # Take the largest face if multiple are detected
            if len(faces) > 1:
                faces = sorted(faces, key=lambda x: x[2] * x[3], reverse=True)
            
            x, y, w, h = faces[0]
            face_center = (x + w//2, y + h//2)
            face_size = w * h
            face_detected_count += 1
            presence_score = 100  # Face is present
            
            # Draw rectangle around face (for visualization if needed)
            # cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
            
            # Calculate face metrics for productivity
            if previous_face_center is not None:
                # Movement stability (penalize excessive movement)
                movement = math.sqrt((face_center[0] - previous_face_center[0])**2 + 
                                    (face_center[1] - previous_face_center[1])**2)
                
                # Scale movement relative to face size for better consistency
                relative_movement = movement / (math.sqrt(face_size) if face_size > 0 else 1)
                
                # Penalize excessive movement (adjust thresholds as needed)
                if relative_movement > 0.3:  # Significant movement
                    position_stability = max(0, 100 - (relative_movement - 0.3) * 200)
                
                # Size change (could indicate leaning forward/backward)
                if previous_face_size > 0:
                    size_ratio = face_size / previous_face_size
                    if size_ratio < 0.8 or size_ratio > 1.2:  # Significant size change
                        position_stability = max(0, position_stability - 20)
            
            # Face orientation/angle assessment
            # Using face aspect ratio as a simple proxy for head tilt
            face_aspect_ratio = h / w if w > 0 else 1
            
            # Typical face aspect ratio is around 1.3-1.5
            # Extreme values indicate likely poor orientation
            if face_aspect_ratio < 1.0 or face_aspect_ratio > 2.0:
                orientation_score = max(50, 100 - abs(face_aspect_ratio - 1.5) * 50)
            
            # Position in frame assessment (penalize if face is at edges or looking down)
            # Looking down is often indicated by face position in lower part of frame
            frame_height, frame_width = frame.shape[:2]
            vertical_position = face_center[1] / frame_height
            
            # Penalize if face is in bottom third of frame (likely looking down)
            if vertical_position > 0.7:
                orientation_score = max(0, orientation_score - 30)
            
            # Update tracking variables
            previous_face_center = face_center
            previous_face_size = face_size
        else:
            # No face detected - significant productivity penalty
            position_stability = 50
            orientation_score = 50
            presence_score = 0  # Face is absent
        
        # Calculate overall productivity score
        # Weightings: 40% stability, 30% orientation, 30% presence
        productivity_score = (
            0.4 * position_stability +
            0.3 * orientation_score +
            0.3 * presence_score
        )
        
        # Store results
        productivity_scores.append(round(productivity_score, 1))
        timestamps.append(round(current_time, 1))

        status_text = f"Calculating Productivity"
        presence_text = "Face Detected" if presence_score == 100 else "No Face"
        cv2.putText(frame, status_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(frame, presence_text, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        
        if len(faces) > 0:
            x, y, w, h = faces[0]
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

        frame = cv2.resize(frame, (800, 600))
        # cv2.namedWindow("Posture Tracking", cv2.WND_PROP_FULLSCREEN)
        # cv2.setWindowProperty("Posture Tracking", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

        cv2.imshow("Posture Tracking", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        frame_count += 1
        
    # Release webcam
    cap.release()
    
    # Calculate statistics
    if productivity_scores:
        average_score = round(sum(productivity_scores) / len(productivity_scores), 1)
        presence_percentage = (face_detected_count / frame_count) * 100 if frame_count > 0 else 0
        
        # Create intervals for time-series data (approximately 10 data points)
        interval = max(1, len(productivity_scores) // 10)
        sampled_scores = productivity_scores[::interval]
        sampled_times = timestamps[::interval]
        
        result = {
            "average_productivity": average_score,
            "face_detection_rate": round(presence_percentage, 1),
            "productivity_over_time": {
                "timestamps": sampled_times,
                "scores": sampled_scores
            },
            "raw_scores": productivity_scores,
            "raw_timestamps": timestamps
        }
        return result
    else:
        print("No valid frames were processed.")
        return None

def preprocess_data(path):
    df = pd.read_csv(path)
    df = df.drop(['Name', 'Gender'], axis=1)
    df['Joining Date'] = pd.to_datetime(df['Joining Date'], format='%b-%y')
    df['experience'] = (datetime(2025, 4, 30) - df['Joining Date']).dt.days / 365.0
    df = df.drop('Joining Date', axis=1)
    df = df.rename(columns={
        'Age': 'age',
        'Productivity (%)': 'posture_score',
        'Projects Completed': 'projects_completed',
        'Salary': 'output'
    })

    df['output'] = pd.qcut(df['output'], q=3, labels=False)

    return df

def train_models(df):
    X = df[['age', 'experience', 'projects_completed', 'posture_score']]
    y = df['output']

    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42)

    scaler = StandardScaler()
    X_tr_scaled = scaler.fit_transform(X_tr)
    X_te_scaled = scaler.transform(X_te)

    lr = LogisticRegression(random_state=42, max_iter=1000)
    lr.fit(X_tr_scaled, y_tr)
    
    rf = RandomForestClassifier(random_state=42)
    rf.fit(X_tr, y_tr)
    
    xgb_model = xgb.XGBClassifier(random_state=42)
    xgb_model.fit(X_tr, y_tr)

    acc_lr = lr.score(X_te_scaled, y_te)
    acc_rf = rf.score(X_te, y_te)
    acc_xgb = xgb_model.score(X_te, y_te)

    print(f"Logistic Regression accuracy: {acc_lr:.4f}")
    print(f"Random Forest accuracy: {acc_rf:.4f}")
    print(f"XGBoost accuracy: {acc_xgb:.4f}")

    return lr, rf, xgb_model, scaler

def predict_employee_performance(posture_score, age=57, experience=26, projects_completed=23, models=None):        
    lr_model, rf_model, xgb_model, scaler = models
    
    feature_names = ['age', 'experience', 'projects_completed', 'posture_score']
    input_df = pd.DataFrame([[age, experience, projects_completed, posture_score]], 
                           columns=feature_names)
    
    input_data_scaled = scaler.transform(input_df)
    
    lr_pred = lr_model.predict(input_data_scaled)[0]
    rf_pred = rf_model.predict(input_df)[0] 
    xgb_pred = xgb_model.predict(input_df)[0]  
    
    predictions = [lr_pred, rf_pred, xgb_pred]
    ensemble_pred = max(set(predictions), key=predictions.count)
    
    performance_categories = ['Low Performer', 'Average Performer', 'High Performer']
    
    print("\nPerformance Prediction:")
    print(f"Logistic Regression: {performance_categories[lr_pred]}")
    print(f"Random Forest: {performance_categories[rf_pred]}")
    print(f"XGBoost: {performance_categories[xgb_pred]}")
    print(f"Ensemble Prediction: {performance_categories[ensemble_pred]}")

    
    
    return ensemble_pred, performance_categories[ensemble_pred]

def plot_productivity_time_series(posture_results):
    """
    Plot the productivity score over the duration of the recording.
    """
    plt.figure(figsize=(10, 6))
    plt.plot(posture_results['raw_timestamps'], posture_results['raw_scores'], 
             marker='o', linestyle='-', color='blue', alpha=0.7)
    
    plt.axhline(y=posture_results['average_productivity'], color='red', 
                linestyle='--', label=f"Average: {posture_results['average_productivity']}")
    
    plt.title('Productivity Score Over Time', fontsize=16)
    plt.xlabel('Time (seconds)', fontsize=12)
    plt.ylabel('Productivity Score (0-100)', fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.show()

def plot_performance_distribution(df):
    """
    Visualize the distribution of employee performance categories in the HR dataset
    """
    performance_labels = ['Low Performer', 'Average Performer', 'High Performer']
    
    plt.figure(figsize=(10, 6))
    
    # Plot countplot with percentages
    ax = sns.countplot(x='output', data=df, palette='viridis')
    
    # Add percentages on top of bars
    total = len(df)
    for p in ax.patches:
        percentage = f'{100 * p.get_height() / total:.1f}%'
        ax.annotate(percentage, (p.get_x() + p.get_width() / 2., p.get_height()),
                    ha='center', va='bottom', fontsize=12)
    
    plt.title('Distribution of Employee Performance Categories', fontsize=16)
    plt.xlabel('Performance Category', fontsize=14)
    plt.ylabel('Count', fontsize=14)
    plt.xticks(ticks=[0, 1, 2], labels=performance_labels)
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.show()

def plot_models(df):
    X = df[['age', 'experience', 'projects_completed', 'posture_score']]
    y = df['output']

    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42)

    scaler = StandardScaler()
    X_tr_scaled = scaler.fit_transform(X_tr)
    X_te_scaled = scaler.transform(X_te)

    lr = LogisticRegression(random_state=42, max_iter=1000)
    lr.fit(X_tr_scaled, y_tr)
    
    rf = RandomForestClassifier(random_state=42)
    rf.fit(X_tr, y_tr)
    
    xgb_model = xgb.XGBClassifier(random_state=42)
    xgb_model.fit(X_tr, y_tr)

    acc_lr = lr.score(X_te_scaled, y_te)
    acc_rf = rf.score(X_te, y_te)
    acc_xgb = xgb_model.score(X_te, y_te)

    y_pred_lr = lr.predict(X_te_scaled)
    y_pred_rf = rf.predict(X_te)
    y_pred_xgb = xgb_model.predict(X_te)

    # Confusion Matrices
    models = ['Logistic Regression', 'Random Forest', 'XGBoost']
    predictions = [y_pred_lr, y_pred_rf, y_pred_xgb]

    plt.figure(figsize=(15, 4))
    for i, (model, y_pred) in enumerate(zip(models, predictions)):
        cm = confusion_matrix(y_te, y_pred)
        plt.subplot(1, 3, i+1)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm)
        disp.plot(ax=plt.gca(), values_format='d')
        plt.title(model)
    plt.tight_layout()
    plt.show()

    # Accuracy Bar Plot
    plt.figure(figsize=(6, 4))
    accuracies = [acc_lr, acc_rf, acc_xgb]
    plt.bar(models, accuracies, color=['skyblue', 'lightgreen', 'salmon'])
    plt.ylabel('Accuracy')
    plt.ylim(0, 1)
    plt.title('Model Accuracy Comparison')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.show()

    return 

if __name__ == '__main__':
    data_path = r'C:\Users\Priyansh\OneDrive\Desktop\AI-Project-Final\hr_dashboard_data.csv'
    df = preprocess_data(data_path)
    models = train_models(df)

    plot_performance_distribution(df)
    plot_models(df)
    
    print("\nCapturing employee productivity based on posture...")
    posture_results = capture_posture_score(duration=10)
    
    if posture_results:
        print("\nEmployee Productivity Analysis:")
        print(f"Average Productivity Score: {posture_results['average_productivity']}/100")
        print(f"Face Detection Rate: {posture_results['face_detection_rate']}%")

        
        employee_age = 57
        employee_experience = 26
        employee_projects = 23
        
        _, performance_category = predict_employee_performance(
            posture_score=posture_results['average_productivity'],
            age=employee_age,
            experience=employee_experience, 
            projects_completed=employee_projects,
            models=models
        )
        
        print(f"\nBased on current posture analysis and employee profile:")
        print(f"Employee is predicted to be a: {performance_category}")

        plot_productivity_time_series(posture_results)

    
        





