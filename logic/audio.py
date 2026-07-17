import threading
import queue
import time
import os
import tempfile
from gtts import gTTS
import pygame

class AudioAlerter:
    def __init__(self):
        self.q = queue.Queue()
        self.thread = threading.Thread(target=self._worker, daemon=True)
        self.thread.start()

    def _worker(self):
        # Initialize pygame mixer for audio playback
        os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = "hide"
        import pygame
        pygame.init()
        pygame.mixer.init()
        
        while True:
            message = self.q.get()
            if message is None:
                break
                
            try:
                # Generate Tamil text-to-speech
                tts = gTTS(text=message, lang='ta')
                
                # Save to a temporary MP3 file
                temp_path = os.path.join(tempfile.gettempdir(), f"alert_{int(time.time())}.mp3")
                tts.save(temp_path)
                
                # Play the MP3 using pygame
                pygame.mixer.music.load(temp_path)
                pygame.mixer.music.play()
                
                # Wait until the audio finishes playing
                while pygame.mixer.music.get_busy():
                    time.sleep(0.1)
                    
                # Unload and clean up the temp file
                pygame.mixer.music.unload()
                try:
                    os.remove(temp_path)
                except Exception:
                    pass
                    
            except Exception as e:
                print(f"Audio playback error: {e}")
                
            self.q.task_done()

    def process_events(self, events_to_log):
        for sev, violation, reason, event_id, person_id in events_to_log:
            if sev == "CRITICAL":
                # Parse the worker ID from event_id (e.g. "NO_HARNESS:worker_7")
                try:
                    worker_id = event_id.split("worker_")[-1]
                except Exception:
                    worker_id = "unknown"
                
                if violation == "NO_HARNESS":
                    item = "ஹார்னஸ்" # Harness
                elif violation == "NO_HELMET":
                    item = "ஹெல்மெட்" # Helmet
                else:
                    item = "பாதுகாப்பு உபகரணம்" # Safety gear
                    
                msg = f"தொழிலாளி {worker_id}, நீங்கள் {item} அணியவில்லை. உங்கள் பாதுகாப்பிற்காக தயவுசெய்து {item} அணியவும்."
                self.q.put(msg)

    def stop(self):
        self.q.put(None)
