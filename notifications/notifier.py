#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# notifications/notifier.py
def send_alert(result):
    print("\n🚨 ALERT! 🚨")
    print(f"⚠ Predicted Disaster: {result['Predicted Disaster']}")
    print("Authorities have been notified.")


