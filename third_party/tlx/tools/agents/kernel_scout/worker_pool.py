"""Keep one worker per runtime and serialize all GPU requests on the controller."""

import contextlib
import json
import os
import select
import signal
import subprocess
import time

from contracts import REPOSITORY, ROOT, digest


class WorkerPool:

    def __init__(self, directory, environment, timeout):
        self.directory, self.environment, self.timeout = directory, environment, timeout
        self.workers = {}
        self.identities = {}
        self.verified = False

    def __enter__(self):
        return self

    def __exit__(self, kind, value, traceback):
        try:
            if kind is None and not self.verified:
                self.verify()
        finally:
            self.close(abort=kind is not None)

    def worker(self, runtime, gpu):
        key = digest([runtime, gpu])
        if key not in self.workers:
            self.directory.mkdir(parents=True, exist_ok=True)
            log = (self.directory / f"{key}.log").open("w")
            read_fd, write_fd = os.pipe()
            try:
                process = subprocess.Popen([
                    runtime["python"],
                    str(ROOT / "benchmarking/worker.py"), "--serve", "--response-fd",
                    str(write_fd)
                ], stdin=subprocess.PIPE, stdout=log, stderr=subprocess.STDOUT, text=True,
                                           env=self.environment(gpu, runtime), cwd=REPOSITORY, start_new_session=True,
                                           pass_fds=(write_fd, ))
            except BaseException:
                os.close(read_fd)
                log.close()
                raise
            finally:
                os.close(write_fd)
            self.workers[key] = {"process": process, "responses": os.fdopen(read_fd), "log": log}
        return self.workers[key]

    def send(self, worker, message):
        process = worker["process"]
        # Only our live, idle peer workers may retain GPU contexts. Requests never overlap.
        peers = [w["process"].pid for w in self.workers.values() if w["process"].poll() is None]
        message = {**message, "peers": peers}
        started = time.monotonic()
        try:
            process.stdin.write(json.dumps(message) + "\n")
            process.stdin.flush()
            if not select.select([worker["responses"]], [], [], self.timeout)[0]:
                raise RuntimeError(f"worker exceeded {self.timeout}s; see {worker['log'].name}")
            response = worker["responses"].readline()
            if not response:
                raise RuntimeError(f"worker exited without acknowledgement; see {worker['log'].name}")
            code = json.loads(response)["exit_code"]
        except (BrokenPipeError, OSError) as error:
            raise RuntimeError(f"worker transport failed; see {worker['log'].name}") from error
        return code, time.monotonic() - started

    def evaluate(self, runtime, gpu, request, output, log):
        self.verified = False
        worker = self.worker(runtime, gpu)
        log.write_text(f"Persistent worker log: {worker['log'].name}\nRequest: {request}\n")
        return self.send(
            worker,
            {"request": str(request), "output": str(output), "expected": self.identities.get(digest([runtime, gpu]))
             })[0]

    def register(self, runtime, gpu, identity):
        self.identities[digest([runtime, gpu])] = identity

    def verify(self):
        for worker in self.workers.values():
            code, _ = self.send(worker, {"verify": True})
            if code:
                raise RuntimeError(f"worker source verification failed; see {worker['log'].name}")
        self.verified = True

    def close(self, *, abort=False):
        for worker in self.workers.values():
            process = worker["process"]
            if abort:
                with contextlib.suppress(ProcessLookupError):
                    os.killpg(process.pid, signal.SIGTERM)
            with contextlib.suppress(BrokenPipeError):
                process.stdin.close()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                with contextlib.suppress(ProcessLookupError):
                    os.killpg(process.pid, signal.SIGKILL)
                process.wait()
            with contextlib.suppress(ProcessLookupError):
                os.killpg(process.pid, signal.SIGKILL)
            worker["responses"].close()
            worker["log"].close()
