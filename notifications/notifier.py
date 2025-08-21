#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# notifications/notifier.py
from plyer import notification

def send_alert(message):
    try:
        notification.notify(
            title="🚨 Disaster Alert!",
            message=message,
            timeout=10,  # seconds
            app_name="Disaster Predictor"
        )
        print("[📢] Notification sent.")
    except Exception as e:
        print(f"[❌] Failed to send notification: {e}")

