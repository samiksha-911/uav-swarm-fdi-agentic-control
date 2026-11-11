#!/usr/bin/env python3
"""
=============================================================================
mission_control.py — Unified Web Ground Control & Multi-Agent Swarm Mission Center
=============================================================================
Provides a single, all-in-one web interface to:
  1. Configure and launch autonomous swarm missions in one go (Fleet size, Faults, Tasks)
  2. View live 2D Swarm Radar, No-Fly Zones, and animated drone waypoints
  3. Real-time telemetry cards (Battery, FDI status, CBF Guardrails, Roles)
  4. One-click in-flight command console (Takeoff, Land, Inject Faults, Recover, Elect Leader)
  5. Live Master System Telemetry & Event stream
  6. Instant scientific evaluation graphs and mission summary report
=============================================================================
"""

import os
import sys
import json
import time
import subprocess
import threading
import tornado.ioloop
import tornado.web
import tornado.websocket
import tornado.httpserver

PORT = 8080
BASE_DIR = "/home/samiksha/uav_fdi_data"
REPORT_PATH = os.path.join(BASE_DIR, "master_system_report.json")
CMD_FILE = "/tmp/swarm_cmd"
CMD_FILE_LOCAL = os.path.join(BASE_DIR, "swarm_cmd.txt")
PYTHON_BIN = os.path.join(BASE_DIR, "fdi_env/bin/python3")

swarm_process = None
process_lock = threading.Lock()
connected_websockets = set()


def is_swarm_running():
    global swarm_process
    if swarm_process is not None and swarm_process.poll() is None:
        return True
    try:
        res = subprocess.run(["pgrep", "-f", "step80_multiagent_swarm.py"], capture_output=True, text=True)
        return res.returncode == 0 and bool(res.stdout.strip())
    except Exception:
        return False


def send_swarm_command(cmd_text: str):
    """Writes command to both cmd files for maximum compatibility."""
    for p in (CMD_FILE, CMD_FILE_LOCAL):
        try:
            with open(p, "w") as f:
                f.write(cmd_text + "\n")
        except Exception:
            pass


class MainHandler(tornado.web.RequestHandler):
    def get(self):
        index_file = os.path.join(BASE_DIR, "web_gui", "index.html")
        if os.path.exists(index_file):
            with open(index_file, "r") as f:
                self.write(f.read())
        else:
            self.write("<h1>Mission Control Interface Loading...</h1>")


class TelemetryAPIHandler(tornado.web.RequestHandler):
    def get(self):
        running = is_swarm_running()
        data = {
            "status": "RUNNING" if running else "IDLE",
            "is_running": running,
            "telemetry": None
        }
        if os.path.exists(REPORT_PATH):
            try:
                with open(REPORT_PATH, "r") as f:
                    data["telemetry"] = json.load(f)
            except Exception:
                pass
        self.set_header("Content-Type", "application/json")
        self.write(json.dumps(data))


class StartMissionAPIHandler(tornado.web.RequestHandler):
    def post(self):
        global swarm_process
        try:
            req = json.loads(self.request.body.decode("utf-8")) if self.request.body else {}
        except Exception:
            req = {}

        fleet_size = int(req.get("fleet_size", 4))
        fault_scenario = req.get("fault_scenario", "none")
        ticks = int(req.get("ticks", 0))

        with process_lock:
            # If already running, stop previous first
            if swarm_process is not None and swarm_process.poll() is None:
                swarm_process.terminate()
                try:
                    swarm_process.wait(timeout=2)
                except Exception:
                    swarm_process.kill()

            cmd = [
                PYTHON_BIN,
                os.path.join(BASE_DIR, "step80_multiagent_swarm.py"),
                "--initial-uavs", str(fleet_size),
                "--ticks", str(ticks)
            ]

            if fault_scenario == "gps_uav2":
                cmd.extend(["--inject-uav", "UAV-2", "--inject-fault", "GPS Failure", "--inject-at-tick", "25"])
            elif fault_scenario == "batt_uav1":
                cmd.extend(["--inject-uav", "UAV-1", "--inject-battery", "14.0", "--inject-at-tick", "20"])
            elif fault_scenario == "gyro_uav3":
                cmd.extend(["--inject-uav", "UAV-3", "--inject-fault", "Gyroscope Failure", "--inject-at-tick", "30"])

            try:
                # Start swarm in background
                swarm_process = subprocess.Popen(
                    cmd,
                    cwd=BASE_DIR,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL
                )
                self.write(json.dumps({"success": True, "message": f"Mission launched with {fleet_size} UAVs."}))
            except Exception as e:
                self.write(json.dumps({"success": False, "error": str(e)}))


class StopMissionAPIHandler(tornado.web.RequestHandler):
    def post(self):
        global swarm_process
        with process_lock:
            if swarm_process is not None and swarm_process.poll() is None:
                send_swarm_command("land")
                time.sleep(0.5)
                swarm_process.terminate()
                try:
                    swarm_process.wait(timeout=2)
                except Exception:
                    swarm_process.kill()
                swarm_process = None
                self.write(json.dumps({"success": True, "message": "Mission terminated."}))
            else:
                self.write(json.dumps({"success": True, "message": "No mission was actively running."}))


class CommandAPIHandler(tornado.web.RequestHandler):
    def post(self):
        try:
            req = json.loads(self.request.body.decode("utf-8")) if self.request.body else {}
        except Exception:
            req = {}
        cmd_text = req.get("command", "").strip()
        running = is_swarm_running()
        if cmd_text:
            send_swarm_command(cmd_text)
            self.write(json.dumps({
                "success": True,
                "sent": cmd_text,
                "is_running": running,
                "warning": None if running else "Swarm is not actively running. Start mission to execute commands."
            }))
        else:
            self.write(json.dumps({"success": False, "error": "Empty command"}))


class GeneratePlotsAPIHandler(tornado.web.RequestHandler):
    def post(self):
        try:
            res = subprocess.run(
                [PYTHON_BIN, os.path.join(BASE_DIR, "plot_results.py")],
                cwd=BASE_DIR,
                capture_output=True,
                text=True,
                timeout=15
            )
            self.write(json.dumps({
                "success": res.returncode == 0,
                "output": res.stdout,
                "plots": [
                    "/plots/swarm_3d_flight_trajectories.png",
                    "/plots/swarm_fdi_safety_timeline.png",
                    "/plots/swarm_mesh_performance.png"
                ]
            }))
        except Exception as e:
            self.write(json.dumps({"success": False, "error": str(e)}))


class TelemetryWebSocket(tornado.websocket.WebSocketHandler):
    def open(self):
        connected_websockets.add(self)

    def on_close(self):
        connected_websockets.discard(self)

    def check_origin(self, origin):
        return True


def broadcast_telemetry_tick():
    """Executed directly on Tornado IOLoop every 250ms — sends live JSON to all active WebSockets."""
    if not connected_websockets:
        return
    if os.path.exists(REPORT_PATH):
        try:
            with open(REPORT_PATH, "r") as f:
                data = json.load(f)
            payload = json.dumps({"is_running": is_swarm_running(), "data": data})
            dead = set()
            for ws in list(connected_websockets):
                try:
                    if ws.ws_connection and not ws.ws_connection.is_closing():
                        ws.write_message(payload)
                    else:
                        dead.add(ws)
                except Exception:
                    dead.add(ws)
            connected_websockets.difference_update(dead)
        except Exception:
            pass


class DownloadFileHandler(tornado.web.RequestHandler):
    def get(self, ftype):
        if ftype == "json" and os.path.exists(REPORT_PATH):
            self.set_header("Content-Type", "application/json")
            self.set_header("Content-Disposition", "attachment; filename=master_system_report.json")
            with open(REPORT_PATH, "rb") as f:
                self.write(f.read())
        elif ftype == "csv":
            cpath = os.path.join(BASE_DIR, "flight_telemetry_log.csv")
            if os.path.exists(cpath):
                self.set_header("Content-Type", "text/csv")
                self.set_header("Content-Disposition", "attachment; filename=flight_telemetry_log.csv")
                with open(cpath, "rb") as f:
                    self.write(f.read())
            else:
                self.set_status(404)
                self.write("CSV log not created yet.")
        else:
            self.set_status(404)


def make_app():
    return tornado.web.Application([
        (r"/", MainHandler),
        (r"/api/telemetry", TelemetryAPIHandler),
        (r"/api/start", StartMissionAPIHandler),
        (r"/api/stop", StopMissionAPIHandler),
        (r"/api/command", CommandAPIHandler),
        (r"/api/generate_plots", GeneratePlotsAPIHandler),
        (r"/api/download/(json|csv)", DownloadFileHandler),
        (r"/ws/telemetry", TelemetryWebSocket),
        (r"/plots/(.*)", tornado.web.StaticFileHandler, {"path": os.path.join(BASE_DIR, "plots")}),
    ])


def main():
    print("=" * 70)
    print("🛸 STARTING AUTONOMOUS UAV SWARM MISSION CONTROL CENTER")
    print("=" * 70)

    app = make_app()
    bound_port = None
    for port in [PORT, 8081, 8082, 8085, 8090]:
        try:
            app.listen(port, address="0.0.0.0")
            bound_port = port
            break
        except Exception:
            continue

    if bound_port is None:
        bound_port = PORT
        app.listen(bound_port)

    print(f"🌐 Web Interface LIVE at: http://localhost:{bound_port}")
    print("=" * 70)

    # Start native periodic telemetry broadcast on Tornado main loop
    tornado.ioloop.PeriodicCallback(broadcast_telemetry_tick, 250).start()

    tornado.ioloop.IOLoop.current().start()


if __name__ == "__main__":
    main()
