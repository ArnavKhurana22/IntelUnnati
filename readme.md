Overview
Customer Analytics with AI is a real-time retail analytics dashboard that uses YOLOv8 for object detection and tracking to monitor customer behavior in a convenience store. The system dynamically identifies crowded areas, tracks dwell times, and provides actionable recommendations such as restocking shelves or optimizing store operations. It supports both webcam input and video file uploads, making it versatile for various retail environments.

Features
Real-Time Object Detection: Tracks customers using YOLOv8 with bounding boxes and unique IDs.

Zone-Based Analytics: Monitors predefined store zones (e.g., Snacks Section, Household Items) for visitation and dwell times.

Dynamic Insights Overlay: Displays actionable recommendations directly on the video feed (e.g., restocking suggestions).

Post-Processing Insights: Generates detailed reports after video processing, including customer counts, zone activity, and recommendations.

Streamlit Interface: User-friendly dashboard for hosting the system with options to use webcam or upload video files.

Installation
Prerequisites
Python 3.8 or higher

Pip package manager

Steps
Clone the repository:

bash
git clone https://github.com/ArnavKhurana22/IntelUnnati/tree/master
cd customer-analytics-ai
Install dependencies:

bash
pip install -r requirements.txt
Run the Streamlit app:

bash
streamlit run app.py
Usage
Select Input Source:

Choose between webcam or upload a video file from the sidebar.

Live Analytics:

Watch real-time tracking with bounding boxes and dynamic text overlays.

Post-Processing Insights:

Review detailed analytics, including customer counts, dwell time distributions, zone visitation metrics, and actionable recommendations.
