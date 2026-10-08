"""Small, durable phase counter with a heartbeat during long operations."""
import datetime
import json
import threading
import time


class Progress:
    def __init__(self, output, interval=60):
        self.output = output
        self.interval = interval
        self.started = time.monotonic()
        self.lock = threading.Lock()
        self.stop = threading.Event()
        self.state = {'status': 'running', 'phase': 'starting', 'completed': 0,
                      'total': None, 'current': '', 'records': 0, 'errors': 0}
        self.thread = threading.Thread(target=self._heartbeat, daemon=True)

    def __enter__(self):
        self.update()
        self.thread.start()
        return self

    def _write(self):
        snapshot = dict(self.state, updated_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                        elapsed_seconds=round(time.monotonic() - self.started, 1))
        total = snapshot['total']
        snapshot['phase_percent'] = (round(100 * snapshot['completed'] / total, 1)
                                     if total else None)
        temporary = self.output / 'PROGRESS.json.tmp'
        temporary.write_text(json.dumps(snapshot, indent=2) + '\n', encoding='utf-8')
        temporary.replace(self.output / 'PROGRESS.json')
        line = json.dumps(snapshot)
        with (self.output / 'progress.jsonl').open('a', encoding='utf-8') as stream:
            stream.write(line + '\n')
        print('PROGRESS ' + line, flush=True)

    def update(self, **changes):
        with self.lock:
            self.state.update(changes)
            self._write()

    def current(self, value):
        # Persist the item before beginning it, even if the process dies inside it.
        self.update(current=value)

    def advance(self, **changes):
        with self.lock:
            self.state.update(changes)
            self.state['completed'] += 1
            # Counts are sampled by the heartbeat; phase transitions flush immediately.

    def phase(self, name, total=None):
        self.update(phase=name, completed=0, total=total, current='')

    def _heartbeat(self):
        while not self.stop.wait(self.interval):
            self.update()

    def __exit__(self, exc_type, exc, traceback):
        self.stop.set()
        self.thread.join()
        self.update(status='failed' if exc_type else 'finished',
                    error=str(exc) if exc_type else '')
