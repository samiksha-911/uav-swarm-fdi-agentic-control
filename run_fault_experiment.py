#!/usr/bin/env python3

import argparse
import os
import re
import signal
import subprocess
import sys
import time
from pathlib import Path

import pandas as pd


BASE = Path.home() / "uav_fdi_data"
PX4_DIR = Path.home() / "PX4-Autopilot"

RECORDER = BASE / "63_live_px4_airdata_recorder.py"
RAW_FILE = BASE / "live_px4_airdata_recording.csv"


FAULT_COMMANDS = {
    "accel": "failure accel off",
    "baro": "failure baro off",
    "gps": "failure gps off",
    "gyro": "failure gyro off",
    "mag": "failure mag off",
}


FAULT_NAMES = {
    "accel": "accel_fault",
    "baro": "baro_fault",
    "gps": "gps_fault",
    "gyro": "gyro_fault",
    "mag": "compass_fault",
    "none": "normal",
}


def log(message):
    print(f"\n[{time.strftime('%H:%M:%S')}] {message}", flush=True)


def start_process(command, cwd=None):
    return subprocess.Popen(
        command,
        cwd=str(cwd) if cwd else None,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        preexec_fn=os.setsid,
    )


def stop_process(proc, name):
    if proc is None:
        return

    if proc.poll() is not None:
        return

    log(f"Stopping {name}...")

    try:
        os.killpg(os.getpgid(proc.pid), signal.SIGINT)
        proc.wait(timeout=10)
    except Exception:
        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
        except Exception:
            pass


def get_next_run_number(fault):
    """
    Find the next unused run number.

    Example:
        run1 exists
        run2 exists
        run3 does not exist

    Returns:
        3
    """

    fault_name = FAULT_NAMES[fault]

    pattern = re.compile(
        rf"^step_auto_{re.escape(fault_name)}_run(\d+)\.csv$"
    )

    numbers = []

    for file in BASE.glob(f"step_auto_{fault_name}_run*.csv"):
        match = pattern.match(file.name)

        if match:
            numbers.append(int(match.group(1)))

    if not numbers:
        return 1

    return max(numbers) + 1


def wait_for_px4_prompt(proc, timeout=120):
    log("Waiting for PX4 pxh prompt...")

    start = time.time()

    while time.time() - start < timeout:

        if proc.poll() is not None:
            raise RuntimeError(
                "PX4 process exited before pxh prompt."
            )

        line = proc.stdout.readline()

        if line:
            print("[PX4]", line.rstrip(), flush=True)

            if "pxh>" in line:
                log("PX4 is ready.")
                return

        else:
            time.sleep(0.1)

    raise TimeoutError(
        "PX4 pxh prompt was not detected."
    )


def wait_for_ros2_telemetry(timeout=60):
    log("Waiting for PX4 ROS 2 telemetry...")

    start = time.time()

    command = [
        "ros2",
        "topic",
        "echo",
        "/fmu/out/vehicle_status_v4",
        "--qos-reliability",
        "best_effort",
        "--once",
    ]

    while time.time() - start < timeout:

        try:
            result = subprocess.run(
                command,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                timeout=8,
            )

            if result.stdout.strip():
                log("ROS 2 telemetry confirmed.")
                return True

        except Exception:
            pass

        time.sleep(2)

    return False


def start_recorder():

    if RAW_FILE.exists():
        RAW_FILE.unlink()

    log("Starting telemetry recorder...")

    proc = subprocess.Popen(
        [
            sys.executable,
            str(RECORDER),
        ],
        cwd=str(BASE),
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        preexec_fn=os.setsid,
    )

    time.sleep(5)

    if proc.poll() is not None:
        raise RuntimeError(
            "Recorder exited immediately."
        )

    return proc


def verify_recording_started(timeout=60):

    log("Checking that recorder is receiving samples and telemetry is healthy...")

    start = time.time()

    while time.time() - start < timeout:

        if RAW_FILE.exists():

            try:
                df = pd.read_csv(RAW_FILE)

                if len(df) >= 15:
                    batt = float(df["battery_voltage"].iloc[-1])
                    z_valid = bool(df["local_z_valid"].iloc[-1])
                    att_inv = bool(df["attitude_invalid"].iloc[-1])

                    if batt > 10.0 and z_valid and not att_inv:
                        log(
                            f"Telemetry validated healthy: {len(df)} samples, "
                            f"battery={batt:.1f}V, local_z_valid={z_valid}, "
                            f"attitude_invalid={att_inv}."
                        )
                        return True
                    else:
                        print(
                            f"\r  stabilizing... ({len(df)} samples, "
                            f"batt={batt:.1f}V, z_valid={z_valid}, "
                            f"att_inv={att_inv})",
                            end="",
                            flush=True,
                        )

            except Exception:
                pass

        time.sleep(2)

    return False


def send_px4_command(px4_proc, command):

    log(f"PX4 command: {command}")

    px4_proc.stdin.write(command + "\n")
    px4_proc.stdin.flush()

    time.sleep(5)


def wait_and_monitor(duration):

    log(
        f"Recording for {duration} seconds..."
    )

    end = time.time() + duration

    while time.time() < end:

        remaining = max(
            0,
            int(end - time.time())
        )

        print(
            f"\r  remaining: {remaining:3d}s",
            end="",
            flush=True,
        )

        time.sleep(1)

    print()


def validate_and_save(fault):

    if not RAW_FILE.exists():
        raise RuntimeError(
            "Recorder did not create "
            "live_px4_recording.csv"
        )

    df = pd.read_csv(RAW_FILE)

    if len(df) == 0:
        raise RuntimeError(
            "Recording contains ZERO samples."
        )

    missing = int(
        df.isna().sum().sum()
    )

    run_number = get_next_run_number(fault)

    output_name = (
        f"step_auto_{FAULT_NAMES[fault]}"
        f"_run{run_number}.csv"
    )

    output = BASE / output_name

    df.to_csv(
        output,
        index=False,
    )

    print()
    print("=" * 40)
    print("EXPERIMENT RESULT")
    print("=" * 40)
    print(f"Fault       : {fault}")
    print(f"Run         : {run_number}")
    print(f"Samples     : {len(df)}")
    print(f"Columns     : {len(df.columns)}")
    print(f"Missing     : {missing}")
    print(f"Saved       : {output}")
    print("=" * 40)

    status_columns = [
        "attitude_invalid",
        "local_position_invalid",
        "local_velocity_invalid",
        "global_position_invalid",
        "angular_velocity_invalid",
        "inertial_dead_reckoning",
        "failsafe",
    ]

    for column in status_columns:

        if column in df.columns:

            print(
                f"{column}: "
                f"{df[column].value_counts().to_dict()}"
            )

    return output


def run_one(
    fault,
    healthy_seconds,
    fault_seconds,
):

    px4 = None
    agent = None
    recorder = None

    try:

        log(
            f"========== "
            f"NEW RUN: {fault} "
            f"=========="
        )

        if RAW_FILE.exists():
            RAW_FILE.unlink()

        # --------------------------------
        # START PX4
        # --------------------------------

        log("Starting PX4 SITL...")

        px4 = start_process(
            [
                "bash",
                "-lc",
                (
                    "source ~/uav_swarm_ws/install/setup.bash && "
                    "source /opt/ros/humble/setup.bash && "
                    "source ~/uav_fdi_data/fdi_env/bin/activate && "
                    "make px4_sitl gz_x500"
                ),
            ],
            cwd=PX4_DIR,
        )

        wait_for_px4_prompt(px4)

        # --------------------------------
        # START XRCE-DDS
        # --------------------------------

        log(
            "Starting Micro XRCE-DDS Agent..."
        )

        agent = start_process(
            [
                "bash",
                "-lc",
                (
                    "source ~/uav_fdi_data/fdi_env/bin/activate && "
                    "MicroXRCEAgent udp4 -p 8888"
                ),
            ],
            cwd=BASE,
        )

        time.sleep(5)

        # --------------------------------
        # CHECK ROS 2
        # --------------------------------

        if not wait_for_ros2_telemetry():

            raise RuntimeError(
                "PX4 ROS 2 telemetry was not detected."
            )

        # --------------------------------
        # START RECORDER
        # --------------------------------

        recorder = start_recorder()

        if not verify_recording_started():

            raise RuntimeError(
                "Telemetry did not stabilize to a valid, healthy state within timeout."
            )

        # --------------------------------
        # HEALTHY BASELINE
        # --------------------------------

        log(
            f"Healthy baseline: "
            f"{healthy_seconds} seconds"
        )

        wait_and_monitor(
            healthy_seconds
        )

        # --------------------------------
        # FAULT
        # --------------------------------

        if fault != "none":

            send_px4_command(
                px4,
                FAULT_COMMANDS[fault],
            )

            time.sleep(3)

            wait_and_monitor(
                fault_seconds
            )

        else:

            log(
                f"Normal-flight recording: "
                f"{fault_seconds} seconds"
            )

            wait_and_monitor(
                fault_seconds
            )

        # --------------------------------
        # STOP RECORDER
        # --------------------------------

        stop_process(
            recorder,
            "recorder",
        )

        recorder = None

        # --------------------------------
        # SAVE DATA
        # --------------------------------

        return validate_and_save(
            fault
        )

    finally:

        stop_process(
            recorder,
            "recorder",
        )

        stop_process(
            agent,
            "Micro XRCE-DDS Agent",
        )

        stop_process(
            px4,
            "PX4",
        )


def main():

    parser = argparse.ArgumentParser(
        description=(
            "Automated PX4 FDI "
            "fault experiment runner"
        )
    )

    parser.add_argument(
        "--fault",
        required=True,
        choices=[
            "accel",
            "baro",
            "gps",
            "gyro",
            "mag",
            "none",
        ],
    )

    parser.add_argument(
        "--runs",
        type=int,
        default=1,
    )

    parser.add_argument(
        "--healthy",
        type=int,
        default=30,
    )

    parser.add_argument(
        "--fault-time",
        type=int,
        default=60,
    )

    args = parser.parse_args()

    print()
    print("=" * 40)
    print(
        " AUTOMATED PX4 FDI "
        "EXPERIMENT RUNNER"
    )
    print("=" * 40)
    print(f"Fault runs : {args.fault}")
    print(f"Runs       : {args.runs}")
    print(
        f"Healthy    : "
        f"{args.healthy}s"
    )
    print(
        f"Fault      : "
        f"{args.fault_time}s"
    )
    print("=" * 40)
    print()

    for i in range(args.runs):

        try:

            run_one(
                args.fault,
                args.healthy,
                args.fault_time,
            )

        except Exception as e:

            print()
            print("!" * 40)
            print("RUN FAILED")
            print(
                f"Reason: {e}"
            )
            print("!" * 40)
            print()

        if i < args.runs - 1:

            log(
                "Waiting 5 seconds "
                "before next independent run..."
            )

            time.sleep(5)

    print()
    print("=" * 40)
    print("BATCH COMPLETE")
    print("=" * 40)


if __name__ == "__main__":
    main()
