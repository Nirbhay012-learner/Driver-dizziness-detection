
import winsound
import time

print("Testing alarm...")

for i in range(3):
    winsound.Beep(1000, 700)
    time.sleep(0.3)

print("Alarm test finished.")
