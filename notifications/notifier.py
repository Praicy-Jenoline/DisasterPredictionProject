#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# notifications/notifier.py

import platform

# Cross-platform send_alert function
if platform.system() == "Darwin":
    # macOS notifications
    try:
        from pync import Notifier
    except ImportError:
        print("Warning: pync not installed. macOS notifications will not work.")
        Notifier = None

    def send_alert(title, message):
        if Notifier:
            Notifier.notify(message, title=title)
        else:
            print(f"[ALERT] {title}: {message}")

else:
    # Windows and Linux notifications
    try:
        from plyer import notification
    except ImportError:
        print("Warning: plyer not installed. Notifications will print to console.")
        notification = None

    def send_alert(title, message):
        if notification:
            notification.notify(
                title=title,
                message=message,
                timeout=10  # seconds
            )
        else:
            print(f"[ALERT] {title}: {message}")
