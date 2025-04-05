import cv2
import numpy as np
from ultralytics import YOLO
import pandas as pd
from collections import defaultdict
import streamlit as st
import tempfile

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
    'Checkout Counter': (0, 0, 200, 150), 
    'Central Display': (200, 150, 400, 300),  
    'Left Shelves': (0, 300, 200, 600),  
    'Right Shelves': (400, 300, 640, 600)  
}

def send_alert(message):
    """Alert mechanism with deduplication"""
    if message not in st.session_state.alerts:
        st.session_state.alerts.append(message)
        st.session_state.new_alerts.add(message)

def generate_insights(analytics):
    """Generate detailed insights from analytics data"""
    insights = {
        'dwell_time_distribution': {},
        'zone_visitation': defaultdict(int),
        'customer_count': len(analytics),
        'recommendations': []
    }
    
    for track_id, data in analytics.items():
        dwell_time = data['dwell_time']
        insights['dwell_time_distribution'][dwell_time] = insights['dwell_time_distribution'].get(dwell_time, 0) + 1
        
        for zone in data['visited_zones']:
            insights['zone_visitation'][zone] += 1

    # Generate recommendations
    if insights['zone_visitation']:
        filtered_visitation = {zone: count for zone, count in insights['zone_visitation'].items() 
                              if zone != "Checkout Counter"}
        if filtered_visitation:
            most_visited = max(filtered_visitation, key=filtered_visitation.get)
            insights['recommendations'].append(f"Restock {most_visited}")
            
            if insights['zone_visitation'].get('Checkout Counter', 0) > 3:
                insights['recommendations'].append("Open additional checkout counters")

    return insights

def generate_dynamic_insights(analytics):
    """Generate real-time insights with alerts"""
    zone_visits = defaultdict(int)
    current_time = pd.Timestamp.now()
    
    for track_id, data in analytics.items():
        # Remove stale entries (15 minutes threshold)
        if data['entry_time'] and (current_time - data['entry_time']).seconds > 900:
            del analytics[track_id]
            continue
            
        for zone in data['visited_zones']:
            zone_visits[zone] += 1

    # Generate alerts and recommendations
    recommendations = []
    if zone_visits:
        crowded_zone = max(zone_visits, key=zone_visits.get)
        
        if zone_visits[crowded_zone] > 5:
            send_alert(f"High traffic in {crowded_zone}")
            recommendations.append(f"🚦 Crowded Area: {crowded_zone}")
            
            if crowded_zone == "Snacks and Beverages":
                recommendations.append("🍫 Restock popular snacks")
            elif crowded_zone == "Checkout Counter":
                recommendations.append("💳 Open additional registers")

    return "\n".join(recommendations) if recommendations else "Normal operation"

def process_video(video_source):
    """Process video stream with proper state management"""
    if 'cap' not in st.session_state or st.session_state.current_source != video_source:
        if 'cap' in st.session_state:
            st.session_state.cap.release()
        st.session_state.cap = cv2.VideoCapture(video_source)
        st.session_state.current_source = video_source
    
    frame_placeholder = st.empty()
    analytics_placeholder = st.empty()

    while st.session_state.get('running', False):
        ret, frame = st.session_state.cap.read()
        
        if not ret:
            st.error("End of video stream")
            st.session_state.running = False
            break

        # Resize and process frame
        frame = cv2.resize(frame, (640, 480))
        results = model.track(frame, persist=True, classes=[0])

        if results[0].boxes.id is not None:
            boxes = results[0].boxes.xyxy.cpu().numpy().astype(int)
            track_ids = results[0].boxes.id.cpu().numpy().astype(int)

            for box, track_id in zip(boxes, track_ids):
                x1, y1, x2, y2 = box
                centroid = ((x1 + x2) // 2, (y1 + y2) // 2)

                # Update analytics
                if analytics[track_id]['entry_time'] is None:
                    analytics[track_id]['entry_time'] = pd.Timestamp.now()
                
                analytics[track_id]['dwell_time'] = (pd.Timestamp.now() - 
                                                     analytics[track_id]['entry_time']).seconds

                # Check zone visits
                for zone_name, (zx1, zy1, zx2, zy2) in ZONES.items():
                    if zx1 <= centroid[0] <= zx2 and zy1 <= centroid[1] <= zy2:
                        analytics[track_id]['visited_zones'].add(zone_name)

                # Draw bounding boxes
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, f"ID: {track_id}", (x1 + 5, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

        # Generate and display insights
        insights_text = generate_dynamic_insights(analytics)
        y_offset = frame.shape[0] - 100
        for i, line in enumerate(insights_text.split("\n")):
            cv2.putText(frame, line, (10, y_offset + i * 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 1)

        # Update main display
        with st.container():
            col1, col2 = st.columns([3, 1])
            with col1:
                frame_placeholder.image(frame[:, :, ::-1], 
                                       channels="RGB", 
                                       caption="Live Feed",
                                       use_column_width=True)
            with col2:
                detailed_insights = generate_insights(analytics)
                analytics_text = f"""
                ### Customer Analytics
                **Total Customers:** {detailed_insights['customer_count']}
                **Most Visited Zone:** {max(detailed_insights['zone_visitation'], 
                                         key=detailed_insights['zone_visitation'].get, 
                                         default="N/A")}
                **Recommendations:**
                {chr(10).join(detailed_insights['recommendations'])}
                """
                analytics_placeholder.markdown(analytics_text)

    # Cleanup when stopped
    if not st.session_state.get('running', False):
        if 'cap' in st.session_state:
            st.session_state.cap.release()
            del st.session_state.cap
        cv2.destroyAllWindows()
        st.session_state.new_alerts.clear()

def start_streamlit_app():
    """Main Streamlit application"""
    st.set_page_config(layout="wide")
    st.title("Enhancing Customer experience with AI driven insights")
    
    # Initialize session state
    if 'running' not in st.session_state:
        st.session_state.running = False
    if 'alerts' not in st.session_state:
        st.session_state.alerts = []
    if 'new_alerts' not in st.session_state:
        st.session_state.new_alerts = set()
    if 'current_source' not in st.session_state:
        st.session_state.current_source = None

    # Sidebar controls
    with st.sidebar:
        st.header("Controls")
        input_type = st.radio("Video Source:", ["Webcam", "Upload Video"])
        
        if st.button("▶️ Start/Stop", key="control_btn"):
            st.session_state.running = not st.session_state.running
            if not st.session_state.running:
                st.session_state.alerts = []

        if input_type == "Upload Video":
            uploaded_file = st.file_uploader("Choose video file", 
                                           type=["mp4", "avi", "mov"])
            if uploaded_file:
                tfile = tempfile.NamedTemporaryFile(delete=False)
                tfile.write(uploaded_file.read())
                st.session_state.video_source = tfile.name
        else:
            st.session_state.video_source = 0

    # Main display area
    main_container = st.container()
    with main_container:
        if st.session_state.running:
            process_video(st.session_state.video_source)
        else:
            st.info("Click the Start button to begin analysis")

    # Alert display
    with st.sidebar:
        if st.session_state.alerts:
            st.subheader("Active Alerts")
            for alert in list(st.session_state.new_alerts):
                st.error(alert)
            st.session_state.new_alerts.clear()

if __name__ == "__main__":
    start_streamlit_app()