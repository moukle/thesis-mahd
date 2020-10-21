import numpy as np
import grequests
import collections

threshold = 0.665
consider_n_frames = 5
predictions = collections.deque(maxlen=consider_n_frames)

IP = "192.168.4.1"
last_sent_status_was_on = False
def send_alarm_on():
    global last_sent_status_was_on
    if not last_sent_status_was_on:
        print("Alarm ON")
        _= grequests.send(grequests.get(f"https://{IP}/on"), grequests.Pool(1))
        last_sent_status_was_on = True

def send_alarm_off():
    global last_sent_status_was_on
    if last_sent_status_was_on:
        print("Alarm OFF")
        _= grequests.send(grequests.get(f"https://{IP}/off"), grequests.Pool(1))
        last_sent_status_was_on = False


def alarm_logic_max(predictions_for_frame):
    global predictions
    global consider_n_frames

    predictions.append(max(predictions_for_frame, default=0))

    avg = np.sum(predictions) / consider_n_frames
    if avg >= threshold:
        send_alarm_on()
    else:
        send_alarm_off()

    return avg >= threshold

if __name__ == "__main__":
    for _ in range(20):
        alarm_logic_max(np.random.rand(2))
