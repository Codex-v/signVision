import platform
import subprocess
import queue
import time
import threading
from typing import Optional

# Windows-specific import (only needed on Windows)
try:
    import win32com.client
except ImportError:
    win32com = None

class TTSUtils:
    """Static utility class for platform-specific text-to-speech with threading."""

    # Shared queue for speech requests
    _speech_queue = queue.Queue()
    _last_speech_time = 0
    _debounce_interval = 0.5  # Minimum time between speech calls (seconds)
    _speech_thread = None
    _running = False
    _lock = threading.Lock()  # To safely start/stop the thread

    @staticmethod
    def is_windows() -> bool:
        """Check if the current platform is Windows."""
        return platform.system() == "Windows"

    @staticmethod
    def is_macos() -> bool:
        """Check if the current platform is macOS."""
        return platform.system() == "Darwin"

    @staticmethod
    def speak_macos(text: str, voice: Optional[str] = None, rate: int = 150) -> bool:
        """
        Speak text on macOS using the 'say' command.
        Args:
            text: The text to speak.
            voice: Optional voice name (e.g., "Alex", "Samantha").
            rate: Words per minute (default 150).
        Returns:
            True if successful, False otherwise.
        """
        if not TTSUtils.is_macos():
            print("speak_macos is only supported on macOS")
            return False
        try:
            cmd = ["say", "-r", str(rate)]
            if voice:
                cmd.extend(["-v", voice])
            cmd.append(text)
            subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            return True
        except subprocess.CalledProcessError as e:
            print(f"Error in macOS speech synthesis: {e}")
            return False
        except Exception as e:
            print(f"Unexpected error in macOS speech: {e}")
            return False

    @staticmethod
    def speak_windows(text: str, voice: Optional[str] = None, rate: int = 0) -> bool:
        """
        Speak text on Windows using SAPI.SpVoice COM object.
        Args:
            text: The text to speak.
            voice: Optional voice name (e.g., "Microsoft David Desktop").
            rate: Speech rate (-10 to 10, default 0).
        Returns:
            True if successful, False otherwise.
        """
        if not TTSUtils.is_windows():
            print("speak_windows is only supported on Windows")
            return False
        if not win32com:
            print("win32com.client not available. Install pywin32: pip install pywin32")
            return False
        try:
            speaker = win32com.client.Dispatch("SAPI.SpVoice")
            if voice:
                voices = speaker.GetVoices()
                for v in voices:
                    if voice.lower() in v.GetDescription().lower():
                        speaker.Voice = v
                        break
            speaker.Rate = min(max(rate, -10), 10)  # Clamp rate between -10 and 10
            speaker.Speak(text, 1)  # 1 = SVSFDefault (synchronous)
            return True
        except Exception as e:
            print(f"Error in Windows speech synthesis: {e}")
            return False

    @staticmethod
    def speak(text: str, voice: Optional[str] = None, rate: Optional[int] = None) -> bool:
        """
        Platform-agnostic speech function. Chooses the appropriate method based on OS.
        Args:
            text: The text to speak.
            voice: Optional voice name.
            rate: Speech rate (macOS: words per minute, Windows: -10 to 10).
        Returns:
            True if successful, False otherwise.
        """
        if not text:
            return False
        
        # Debounce to prevent overlapping speech
        with TTSUtils._lock:
            current_time = time.time()
            if current_time - TTSUtils._last_speech_time < TTSUtils._debounce_interval:
                print(f"Speech debounced: '{text}' (too soon)")
                return False

        if TTSUtils.is_macos():
            success = TTSUtils.speak_macos(text, voice, rate if rate is not None else 150)
        elif TTSUtils.is_windows():
            success = TTSUtils.speak_windows(text, voice, rate if rate is not None else 0)
        else:
            print("Unsupported platform for TTS")
            success = False
        
        if success:
            with TTSUtils._lock:
                TTSUtils._last_speech_time = time.time()
        return success

    @staticmethod
    def _process_speech_queue():
        """Background thread function to process queued speech requests."""
        print("Speech thread started")
        while TTSUtils._running:
            try:
                text, voice, rate = TTSUtils._speech_queue.get(timeout=0.1)
                if TTSUtils.speak(text, voice, rate):
                    print(f"Finished speaking: '{text}'")
                else:
                    print(f"Failed to speak: '{text}'")
                TTSUtils._speech_queue.task_done()
            except queue.Empty:
                continue
            except Exception as e:
                print(f"Speech thread error: {e}")
                time.sleep(0.1)
        print("Speech thread stopped")

    @staticmethod
    def start_speech_thread():
        """Start the background speech processing thread if not already running."""
        with TTSUtils._lock:
            if not TTSUtils._running:
                TTSUtils._running = True
                TTSUtils._speech_thread = threading.Thread(target=TTSUtils._process_speech_queue, daemon=True)
                TTSUtils._speech_thread.start()
                print("Speech thread initialized")

    @staticmethod
    def stop_speech_thread():
        """Stop the background speech processing thread."""
        with TTSUtils._lock:
            if TTSUtils._running:
                TTSUtils._running = False
                if TTSUtils._speech_thread:
                    TTSUtils._speech_thread.join(timeout=1.0)
                TTSUtils._speech_thread = None
                while not TTSUtils._speech_queue.empty():
                    TTSUtils._speech_queue.get()  # Clear queue
                    TTSUtils._speech_queue.task_done()
                print("Speech thread terminated")

    @staticmethod
    def queue_speech(text: str, voice: Optional[str] = None, rate: Optional[int] = None):
        """Queue speech to be processed in the background thread."""
        TTSUtils.start_speech_thread()  # Ensure thread is running
        TTSUtils._speech_queue.put((text, voice, rate))
        print(f"Queued speech: '{text}'")

    @staticmethod
    def process_speech_queue():
        """Manually process queued speech requests (non-threaded, for testing or single-threaded use)."""
        try:
            while not TTSUtils._speech_queue.empty():
                text, voice, rate = TTSUtils._speech_queue.get_nowait()
                TTSUtils.speak(text, voice, rate)
                TTSUtils._speech_queue.task_done()
        except queue.Empty:
            pass

if __name__ == "__main__":
    # Test the threaded functionality
    TTSUtils.queue_speech("Hello from the background thread", voice="Alex" if TTSUtils.is_macos() else "Microsoft David Desktop")
    TTSUtils.queue_speech("This is a second test")
    time.sleep(2)  # Let the thread process the queue
    TTSUtils.stop_speech_thread()  # Clean up