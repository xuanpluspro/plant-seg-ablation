"""Stream a child process to the terminal and a separate log file."""
import subprocess
import sys


def run_logged(command, log_path):
    with open(log_path, "ab") as log:
        process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        try:
            while True:
                chunk = process.stdout.read1(4096)
                if not chunk:
                    break
                log.write(chunk); log.flush()
                sys.stdout.buffer.write(chunk); sys.stdout.buffer.flush()
            return process.wait()
        except KeyboardInterrupt:
            process.terminate()
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill(); process.wait()
            raise
        finally:
            process.stdout.close()
