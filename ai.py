import cv2
import numpy as np
from ultralytics import YOLO
import pandas as pd
from collections import defaultdict
import tkinter as tk
from tkinter import filedialog, messagebox, Text

# Initialize YOLOv8 model
model = YOLO('yolov8n.pt')

# Store customer analytics
analytics = defaultdict(lambda: {
    'entry_time': None,
    'dwell_time': 0,
    'visited_zones': set()
})

# Define store zones (x1, y1, x2, y2) based on the attached image layout
ZONES = {
    'Checkout Counter': (0, 0, 200, 150),  # Near cashier area
    'Central Display': (200, 150, 400, 300),  # Circular baskets in the center
    'Left Shelves': (0, 300, 200, 600),  # Shelves along left wall
    'Right Shelves': (400, 300, 640, 600)  # Shelves along right wall
}


def send_alert(message):
    """Simple alert mechanism"""
    print(f"ALERT: {message}")

def generate_insights(analytics):
    """Generate insights from analytics data"""
    insights = {
        'dwell_time_distribution': {},
        'zone_visitation': {},
        'customer_count': len(analytics),
        'recommendations': []  # New field for actionable recommendations
    }
    
    # Calculate dwell time distribution and zone visitation
    for track_id, data in analytics.items():
        dwell_time = data['dwell_time']
        if dwell_time not in insights['dwell_time_distribution']:
            insights['dwell_time_distribution'][dwell_time] = 1
        else:
            insights['dwell_time_distribution'][dwell_time] += 1
        
        for zone in data['visited_zones']:
            if zone not in insights['zone_visitation']:
                insights['zone_visitation'][zone] = 1
            else:
                insights['zone_visitation'][zone] += 1
    
    # Determine zones with the highest visitation and generate recommendations
    if insights['zone_visitation']:
        # Exclude Checkout Counter from recommendations
        filtered_visitation = {zone: count for zone, count in insights['zone_visitation'].items() if zone != "Checkout Counter"}
        
        if filtered_visitation:
            most_visited_zone = max(filtered_visitation, key=filtered_visitation.get)
            insights['recommendations'].append(f"Restock items in {most_visited_zone} to meet demand.")
    
    return insights



def display_insights(insights):
    """Display insights in a pop-up window"""
    root = tk.Tk()
    root.title("Customer Analytics Insights")
    
    text_widget = Text(root, wrap="word", width=60, height=20)
    text_widget.pack(padx=10, pady=10)
    
    # Format insights into readable text
    text_widget.insert("end", "Customer Analytics Insights\n")
    text_widget.insert("end", "=" * 50 + "\n")
    
    text_widget.insert("end", f"Total Customers Tracked: {insights['customer_count']}\n\n")
    
    text_widget.insert("end", "Dwell Time Distribution (seconds):\n")
    for time, count in sorted(insights['dwell_time_distribution'].items()):
        text_widget.insert("end", f" - {time} seconds: {count} customers\n")
    
    text_widget.insert("end", "\nZone Visitation Counts:\n")
    for zone, count in insights['zone_visitation'].items():
        text_widget.insert("end", f" - {zone}: {count} visits\n")
    
    # Display recommendations
    text_widget.insert("end", "\nRecommendations:\n")
    for recommendation in insights.get('recommendations', []):
        text_widget.insert("end", f" - {recommendation}\n")
    
    text_widget.config(state="disabled")  # Make the text widget read-only
    
    root.mainloop()



def generate_dynamic_insights(analytics):
    """Generate dynamic insights based on real-time data"""
    zone_visits = defaultdict(int)
    
    for data in analytics.values():
        for zone in data['visited_zones']:
            zone_visits[zone] += 1
    
    # Determine the most crowded area
    crowded_zone = max(zone_visits, key=zone_visits.get) if zone_visits else "None"
    
    # Generate recommendations
    recommendations = []
    if crowded_zone != "None":
        recommendations.append(f"Crowded Area: {crowded_zone}")
        if crowded_zone == "Checkout Counter":
            recommendations.append("Consider opening another counter.")
        elif crowded_zone == "Central Display":
            recommendations.append("Restock popular items in the center baskets.")
        elif crowded_zone == "Left Shelves" or crowded_zone == "Right Shelves":
            recommendations.append("Ensure shelves are well-stocked.")
    
    return "\n".join(recommendations)

def process_video(video_source):
    """Process video from the given source"""
    cap = cv2.VideoCapture(video_source)

    while cap.isOpened():
        ret, frame = cap.read()
        
        # If no frame is returned, break (end of video or error)
        if not ret:
            print("End of video or cannot read frame.")
            break
        
        # Resize for faster processing
        frame = cv2.resize(frame, (640, 480))
        
        # Run object detection and tracking
        results = model.track(frame, persist=True, classes=[0])  # 0 = person class
        
        if results[0].boxes.id is not None:
            boxes = results[0].boxes.xyxy.cpu().numpy().astype(int)
            track_ids = results[0].boxes.id.cpu().numpy().astype(int)
            
            # Update customer analytics
            for box, track_id in zip(boxes, track_ids):
                x1, y1, x2, y2 = box
                centroid = ((x1 + x2) // 2, (y1 + y2) // 2)
                
                # Track dwell time
                if analytics[track_id]['entry_time'] is None:
                    analytics[track_id]['entry_time'] = pd.Timestamp.now()
                analytics[track_id]['dwell_time'] = (pd.Timestamp.now() - analytics[track_id]['entry_time']).seconds
                
                # Check zone visits
                for zone_name, (zx1, zy1, zx2, zy2) in ZONES.items():
                    if zx1 <= centroid[0] <= zx2 and zy1 <= centroid[1] <= zy2:
                        analytics[track_id]['visited_zones'].add(zone_name)

                # Draw bounding box and ID on the frame
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, f"ID: {track_id}", (x1 + 5, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX,
                            0.5, (255, 255, 255), thickness=1)

        # Generate dynamic insights
        insights_text = generate_dynamic_insights(analytics)

        # Add dynamic insights to the frame
        y_offset = frame.shape[0] - 100
        for i, line in enumerate(insights_text.split("\n")):
            cv2.putText(frame, line.strip(), (10, y_offset + i * 20), cv2.FONT_HERSHEY_SIMPLEX,
                        0.6, (0, 255, 255), thickness=1)

        # Display video frame with tracking information and dynamic insights
        cv2.imshow('Video Footage with Insights', frame)
        
        # Exit condition: Press 'q' to quit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("Video terminated by user.")
            break

    # Release resources and close windows
    cap.release()
    cv2.destroyAllWindows()

    # Generate and display analytics insights after processing video
    insights = generate_insights(analytics)
    display_insights(insights)


def start_program():
    """Start the program with user choice"""
    root = tk.Tk()
    root.withdraw()  # Hide the main tkinter window
    
    choice = messagebox.askquestion("Video Source", "Do you want to use the webcam?")
    
    if choice == "yes":
        process_video(0)  # Webcam source
    else:
        file_path = filedialog.askopenfilename(title="Select Video File", filetypes=[("Video Files", "*.mp4 *.avi *.mkv")])
        
        if file_path:
            process_video(file_path)
        else:
            messagebox.showerror("Error", "No file selected!")

if __name__ == "__main__":
    start_program()
