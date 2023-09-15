import csv
import datetime

# Create a CSV file
with open("emotions.csv", mode="w", newline="") as file:
    writer = csv.writer(file, delimiter=",", quotechar='"', quoting=csv.QUOTE_MINIMAL)
    writer.writerow(["Timestamp", "Emotion"])

    while True:
        # ... (previous code to detect and display emotions)
        
        # Capture timestamp
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        # Write data to CSV
        writer.writerow([timestamp, emotion])

        # ... (rest of the code)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
