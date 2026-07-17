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
            payload = self.q.get()
            if payload is None:
                break
            
            message, lang = payload
                
            try:
                # Generate text-to-speech
                tts = gTTS(text=message, lang=lang)
                
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

    def process_events(self, events_to_log, lang="ta"):
        for sev, violation, reason, event_id, person_id in events_to_log:
            if sev == "CRITICAL":
                # Parse the worker ID from event_id (e.g. "NO_HARNESS:worker_7")
                try:
                    worker_id = event_id.split("worker_")[-1]
                except Exception:
                    worker_id = "unknown"
                
                if lang == "ta":
                    if violation == "NO_HARNESS":
                        item = "ஹார்னஸ்" # Harness
                    elif violation == "NO_HELMET":
                        item = "ஹெல்மெட்" # Helmet
                    else:
                        item = "பாதுகாப்பு உபகரணம்" # Safety gear
                    msg = f"எச்சரிக்கை: தொழிலாளி {worker_id}, {item} அணியவும்!"
                else:
                    if violation == "NO_HARNESS":
                        item = "harness"
                    elif violation == "NO_HELMET":
                        item = "helmet"
                    else:
                        item = "safety gear"
                    msg = f"Worker {worker_id} is not wearing a {item}. Please wear a {item} for your safety."
                    
                self.q.put((msg, lang))

    def stop_current_audio(self):
        # Stop currently playing audio and clear the backlog queue
        try:
            if pygame.mixer.get_init():
                pygame.mixer.music.stop()
        except Exception:
            pass
        with self.q.mutex:
            self.q.queue.clear()

    def stop(self):
        self.q.put(None)
