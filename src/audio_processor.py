import threading
from RealtimeSTT import AudioToTextRecorder


class AudioProcessor:
    """Handles real-time audio processing and Voice Activity Detection (VAD)."""

    def __init__(self):
        self.is_talking = False
        self.latest_text = ""

        self.recorder = AudioToTextRecorder(
            spinner=False,
            model="tiny",
            language="en",
            on_recording_start=self._on_recording_start,
            on_recording_stop=self._on_recording_stop,
            silero_sensitivity=0.4,
            webrtc_sensitivity=3,
        )

        self.listen_thread = threading.Thread(target=self._listen_loop, daemon=True)
        self.listen_thread.start()

    def _on_recording_start(self):
        self.is_talking = True

    def _on_recording_stop(self):
        self.is_talking = False

    def _process_text(self, text):
        if text.strip():
            self.latest_text = text

    def _listen_loop(self):
        while True:
            try:
                self.recorder.text(self._process_text)
            except Exception:
                break

    def terminate(self):
        try:
            self.recorder.shutdown()
        except Exception as e:
            print(f"Error shutting down audio processor: {e}")
