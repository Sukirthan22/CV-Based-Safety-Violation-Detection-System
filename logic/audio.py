import threading
import queue
import pyttsx3

class AudioAlerter:
    def __init__(self):
        self.q = queue.Queue()
        self.thread = threading.Thread(target=self._worker, daemon=True)
        self.thread.start()

    def _worker(self):
        # Initialize TTS engine in the background thread to avoid freezing the video
        engine = pyttsx3.init()
        
        # Make the voice sound more natural (slower rate, female voice if available)
        engine.setProperty("rate", 160)
        voices = engine.getProperty("voices")
        for voice in voices:
            if "Zira" in voice.name:
                engine.setProperty("voice", voice.id)
                break
        
        while True:
            message = self.q.get()
            if message is None:
                break
            engine.say(message)
            engine.runAndWait()
            self.q.task_done()

    def process_events(self, events_to_log):
        for sev, violation, reason, event_id in events_to_log:
            if sev == "CRITICAL":
                # Parse the worker ID from event_id (e.g. "NO_HARNESS:worker_7")
                try:
                    worker_id = event_id.split("worker_")[-1]
                except Exception:
                    worker_id = "unknown"
                
                if violation == "NO_HARNESS":
                    item = "harness"
                elif violation == "NO_HELMET":
                    item = "helmet"
                else:
                    item = "safety gear"
                    
                msg = f"Worker {worker_id} is not wearing a {item}. Please wear a {item} for your safety."
                self.q.put(msg)

    def stop(self):
        self.q.put(None)
