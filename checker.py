import yara
import os
import shutil
import time
import requests
import threading
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
import hashlib
import math
from collections import Counter

class YaraAntivirusEngine:
    def __init__(self, rule_url="https://raw.githubusercontent.com/YARA-rules/rules/master/index.yar", 
                 rule_file="ransomware_rules.yar", quarantine_dir="quarantine"):
        """
        Initialize the antivirus engine with rule URL, local rule file, and quarantine directory.
        
        Args:
            rule_url (str): URL to download YARA rules from.
            rule_file (str): Local file to store YARA rules.
            quarantine_dir (str): Directory to move detected malicious files.
        """
        self.rule_url = rule_url
        self.rule_file = rule_file
        self.quarantine_dir = quarantine_dir
        self.rules = None
        self.last_hash = {}  # Track file hashes for change detection
        self.network_activity = set()  # Placeholder for network monitoring

        # Create quarantine directory if it doesn't exist
        if not os.path.exists(self.quarantine_dir):
            os.makedirs(self.quarantine_dir)

        # Load or download initial rules
        self.update_rules()
        self.start_rule_update_thread()

    def update_rules(self):
        """
        Download and compile YARA rules from the specified URL.
        """
        try:
            response = requests.get(self.rule_url, timeout=10)
            if response.status_code == 200:
                with open(self.rule_file, 'w') as f:
                    f.write(response.text)
                self.rules = yara.compile(filepath=self.rule_file)
                print(f"[UPDATE] Successfully updated YARA rules from {self.rule_url}")
            else:
                print(f"[WARNING] Failed to download rules (status {response.status_code}), using local copy")
                if os.path.exists(self.rule_file):
                    self.rules = yara.compile(filepath=self.rule_file)
                else:
                    print("[ERROR] No local rules available")
        except Exception as e:
            print(f"[ERROR] Rule update failed: {e}")
            if os.path.exists(self.rule_file):
                self.rules = yara.compile(filepath=self.rule_file)

    def start_rule_update_thread(self):
        """
        Start a background thread to periodically update rules every hour.
        """
        def update_loop():
            while True:
                time.sleep(3600)  # Update every hour
                self.update_rules()

        threading.Thread(target=update_loop, daemon=True).start()

    def calculate_hash(self, file_path):
        """
        Calculate MD5 hash of a file for change detection.
        
        Args:
            file_path (str): Path to the file.
        
        Returns:
            str: Hexadecimal hash value.
        """
        hasher = hashlib.md5()
        with open(file_path, 'rb') as f:
            hasher.update(f.read())
        return hasher.hexdigest()

    def calculate_entropy(self, file_path):
        """
        Calculate Shannon entropy of a file to detect encryption.
        
        Args:
            file_path (str): Path to the file.
        
        Returns:
            float: Entropy value.
        """
        with open(file_path, 'rb') as f:
            data = f.read()
            if len(data) == 0:
                return 0
            occurrences = Counter(data)
            entropy = 0
            for x in occurrences.values():
                p_x = x / len(data)
                if p_x > 0:
                    entropy -= p_x * math.log2(p_x)
            return entropy

    def is_encryption_suspected(self, file_path, current_hash):
        """
        Check if a file might be encrypted based on hash change and entropy.
        
        Args:
            file_path (str): Path to the file.
            current_hash (str): Current hash of the file.
        
        Returns:
            bool: True if encryption is suspected.
        """
        old_hash = self.last_hash.get(file_path)
        if old_hash and old_hash != current_hash:
            entropy = self.calculate_entropy(file_path)
            return entropy > 7.0  # High entropy suggests encryption
        return False

    def check_mass_modification(self, file_path):
        """
        Check if multiple files are modified in a short time (simplified heuristic).
        
        Args:
            file_path (str): Path to the file.
        
        Returns:
            bool: True if mass modification is detected.
        """
        modified_files = [f for f in self.last_hash if self.last_hash[f] != self.calculate_hash(f)]
        return len(modified_files) > 5  # Arbitrary threshold

    def quarantine_file(self, file_path):
        """
        Move a detected malicious file to the quarantine directory.
        
        Args:
            file_path (str): Path to the file.
        """
        try:
            timestamp = time.strftime("%Y%m%d-%H%M%S")
            new_name = f"{os.path.basename(file_path)}_{timestamp}"
            quarantine_path = os.path.join(self.quarantine_dir, new_name)
            shutil.move(file_path, quarantine_path)
            print(f"[QUARANTINED] Moved to: {quarantine_path}")
        except Exception as e:
            print(f"[ERROR] Failed to quarantine {file_path}: {e}")

    class ScanHandler(FileSystemEventHandler):
        """
        Inner class to handle file system events for real-time scanning.
        """
        def __init__(self, engine):
            self.engine = engine

        def on_modified(self, event):
            if not event.is_directory and self.engine.is_target_file(event.src_path):
                self.engine.scan_file(event.src_path)

        def is_target_file(self, file_path):
            return file_path.lower().endswith(('.exe', '.doc', '.jpg', '.pdf'))

    def scan_file(self, file_path):
        """
        Scan a single file using YARA rules and behavioral heuristics.
        
        Args:
            file_path (str): Path to the file.
        """
        if not self.rules:
            print(f"[ERROR] No YARA rules loaded, skipping {file_path}")
            return

        try:
            current_hash = self.calculate_hash(file_path)
            if self.is_encryption_suspected(file_path, current_hash):
                print(f"[SUSPICION] Possible encryption detected: {file_path}")

            matches = self.rules.match(file_path)
            if matches:
                print(f"[DETECTED] Suspicious file: {file_path} - Matches: {matches}")
                self.quarantine_file(file_path)
            else:
                print(f"[SAFE] File scanned: {file_path}")
                if file_path in self.last_hash and self.last_hash[file_path] != current_hash:
                    if self.check_mass_modification(file_path):
                        print(f"[HEURISTIC] Mass modification detected: {file_path}")
                        self.quarantine_file(file_path)
            self.last_hash[file_path] = current_hash
        except Exception as e:
            print(f"[ERROR] Failed to scan {file_path}: {e}")

    def scan_directory(self, directory):
        """
        Scan all files in a directory.
        
        Args:
            directory (str): Path to the directory.
        """
        for root, _, files in os.walk(directory):
            for file in files:
                file_path = os.path.join(root, file)
                self.scan_file(file_path)

    def start_monitoring(self, directory):
        """
        Start real-time monitoring of a directory.
        
        Args:
            directory (str): Path to the directory to monitor.
        """
        event_handler = self.ScanHandler(self)
        observer = Observer()
        observer.schedule(event_handler, directory, recursive=True)
        observer.start()
        print(f"Monitoring directory: {directory}")
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            observer.stop()
        observer.join()

if __name__ == "__main__":
    # Initialize the engine
    av_engine = YaraAntivirusEngine(
        rule_url="https://raw.githubusercontent.com/YARA-rules/rules/master/index.yar",  # Example URL
        rule_file="ransomware_rules.yar",
        quarantine_dir="quarantine"
    )

    # Choose monitoring or one-time scan
    target_dir = "."  # Current directory; change to test folder
    print("Starting YARA Antivirus Engine...")
    print("Press Ctrl+C to stop monitoring.")
    av_engine.start_monitoring(target_dir)
    # For one-time scan, uncomment the next line
    # av_engine.scan_directory(target_dir)