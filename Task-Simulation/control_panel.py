"""
Control panel for the Cucumber-Water simulation (a separate tkinter window).

    py -3.11 Task-Simulation/control_panel.py
    (or: py -3.11 Task-Simulation/task_simulation.py   - opens this panel too)

Run controls
    START            run the whole task (from the start position)
    PAUSE / RESUME   normal stop and carry on
    E-STOP           latching emergency stop: every robot freezes at once.
                     Releasing it takes two separate actions (as in ISO 13850):
                     RESET E-STOP (only possible once the light curtain is clear),
                     then RESUME. Shortcut: Esc or Space.
    Step mode        the task waits before every move until NEXT STEP is pressed
    RESET SCENE      abandon the run and put everything back at the start

Tabs: step log, teach pendant (joint sliders, x/y/z jog, gripper, home),
settings (number of glasses, playback speed, show hitboxes / collision markers),
safety (light curtain with a person who walks in, an obstacle dropped in the
robots' way - the moving robot stops and plans round it).

Threads: the window runs on the main thread. The simulation runs on its own
thread and is the only one that touches Swift; the window talks to it through
SimController (flags + a command queue).
"""
import os
import queue
import sys
import threading
import time
import traceback

import tkinter as tk
from tkinter import ttk

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import movement_handler                                     # noqa: E402
import collision_handler                                    # noqa: E402
from movement_handler import TaskAborted                    # noqa: E402


class SimController:
    """Run state shared by the window (any thread) and the simulation thread."""

    def __init__(self):
        self.sim = None
        self.ready = False
        self.error = None
        self.cmds = queue.Queue()          # start / reset / teach actions (simulation thread)
        self.actions = queue.Queue()       # small scene changes, applied at the next frame

        self.task_running = False
        self.finished = False              # a run has ended (scene not at the start any more)
        self.latched = False               # e-stop pressed and not yet reset
        self.need_resume = False           # e-stop reset: waiting for RESUME
        self.paused = False
        self.step_mode = False
        self.waiting_for_step = False
        self._step_go = threading.Event()
        self.abort_requested = False
        self.halted = False                # last motion stopped for an obstacle

        self.current_step = ""
        self.message = "Starting the simulation..."
        self.estop_reason = ""
        self.n_glasses = 5

        self.person_target = None          # light curtain demo: where the person walks to
        self.PERSON_SPEED = 0.8            # m/s
        self._last_tick = time.time()

    # ---------------------------------------------------------------- window side
    @property
    def frozen(self):
        return self.latched or self.need_resume or self.paused

    def state(self):
        """(text, colour) for the status banner."""
        if not self.ready:
            return ("LOADING", "#777777") if self.error is None else ("ERROR", "#b00020")
        if self.latched:
            return "EMERGENCY STOP", "#d00000"
        if self.need_resume:
            return "E-STOP RESET - press RESUME", "#e07000"
        if not self.task_running:
            return ("FINISHED" if self.finished else "READY"), "#555555"
        if self.paused:
            return "PAUSED", "#d4a000"
        if self.waiting_for_step:
            return "STEP MODE - press NEXT STEP", "#2060c0"
        return "RUNNING", "#1a9a3a"

    def start(self):
        if self.ready and not self.task_running and not self.latched and not self.need_resume:
            self.cmds.put(("start", self.n_glasses))

    def estop(self, reason="E-STOP button"):
        if not self.latched:
            self.latched = True
            self.need_resume = False
            self.estop_reason = reason
            self.message = f"Emergency stop: {reason}"
            print(f"\n!!! EMERGENCY STOP: {reason} !!!")

    def reset_estop(self):
        if not self.latched:
            return
        if self.sim is not None and self.sim.person_inside_curtain():
            self.message = "Can't reset: someone is inside the light curtain"
            return
        self.latched = False
        self.need_resume = True
        self.message = "E-stop reset. Press RESUME to carry on."

    def pause(self):
        if self.task_running and not self.latched:
            self.paused = True
            self.message = "Paused"

    def resume(self):
        if self.latched:
            self.message = "Reset the e-stop first"
            return
        self.need_resume = False
        self.paused = False
        self.message = ""

    def set_step_mode(self, on):
        self.step_mode = bool(on)
        if not on:
            self._step_go.set()

    def next_step(self):
        self._step_go.set()

    def reset_scene(self):
        if not self.ready:
            return
        if self.task_running:
            self.abort_requested = True
        self.cmds.put(("reset", self.n_glasses))

    def can_teach(self):
        return self.ready and not self.task_running and not self.latched and not self.need_resume

    def submit(self, fn, label=""):
        """Run fn() on the simulation thread (teach pendant actions). fn returns a message."""
        if self.can_teach():
            self.cmds.put(("call", fn, label))

    def scene(self, fn):
        """A small scene change (e.g. show hitboxes) - applied on the simulation thread."""
        self.actions.put(fn)

    def person_walk(self, inside):
        if self.sim is not None:
            self.person_target = self.sim.PERSON_INSIDE_X if inside else self.sim.PERSON_OUTSIDE_X

    def check_curtain(self):
        """Light curtain: trips the e-stop the moment someone crosses it."""
        if self.sim is not None and self.ready and self.sim.person_inside_curtain():
            self.estop("light curtain broken")

    # ------------------------------------------------------- simulation thread side
    def tick(self, step_env):
        """Apply queued scene changes, walk the person, check the light curtain."""
        now = time.time()
        dt, self._last_tick = min(now - self._last_tick, 0.2), now
        changed = False
        while True:
            try:
                fn = self.actions.get_nowait()
            except queue.Empty:
                break
            fn()
            changed = True
        sim = self.sim
        if self.person_target is not None:
            x = sim._person_x
            step = self.PERSON_SPEED * dt
            x = self.person_target if abs(self.person_target - x) <= step else x + np.sign(self.person_target - x) * step
            sim.place_person(x)
            if x == self.person_target:
                self.person_target = None
            changed = True
        self.check_curtain()
        tripped = sim.person_inside_curtain()
        if tripped != getattr(self, '_shown_tripped', False):
            sim.show_curtain_tripped(tripped)
            self._shown_tripped = tripped
            changed = True
        if step_env:
            if changed:
                sim.env.step(0.05)
            else:
                time.sleep(0.05)

    def abort_check(self):
        if self.abort_requested:
            raise TaskAborted()

    def _wait_while_frozen(self):
        while self.frozen:
            self.abort_check()
            self.tick(step_env=True)
        self.abort_check()

    def frame_hook(self, robot, q):
        """Called before every animation frame (see movement_handler.FRAME_HOOK)."""
        self.tick(step_env=False)
        self._wait_while_frozen()
        if robot is not None and self.sim.would_hit_demo_obstacle(robot, q):
            self.halted = True
            self.message = f"{robot.name} stopped: something is in its way - replanning"
            return False
        return True

    def before_move(self, label):
        """Called before every task move: shows the step, waits for pause / e-stop
        and, in step mode, for NEXT STEP."""
        if not self.task_running:
            return
        if label:
            self.current_step = label
        self.abort_check()
        self._wait_while_frozen()
        # (a move can check in twice - e.g. 'go above' then the move itself;
        #  one NEXT STEP covers both)
        same_move = label and label == getattr(self, '_released_label', None)
        self._released_label = label
        if self.step_mode and not same_move:
            self._step_go.clear()
            self.waiting_for_step = True
            try:
                while self.step_mode and not self._step_go.is_set():
                    self.abort_check()
                    self.tick(step_env=True)
                    self._wait_while_frozen()
            finally:
                self.waiting_for_step = False

    def take_halt(self):
        h, self.halted = self.halted, False
        return h

    def run(self, sim_factory):
        """Simulation thread: build the scene, then carry out commands."""
        try:
            self.sim = sim_factory(self)
            self.sim.setup_scene()
        except Exception as e:                        # show it in the window
            self.error = e
            self.message = f"Couldn't start the simulation: {e}"
            traceback.print_exc()
            return
        movement_handler.FRAME_HOOK = self.frame_hook
        collision_handler.ABORT_CHECK = self.abort_check
        self.ready = True
        self.message = "Ready. Press START."
        while True:
            try:
                cmd = self.cmds.get(timeout=0.05)
            except queue.Empty:
                self.tick(step_env=True)
                continue
            try:
                if cmd[0] == "quit":
                    return
                if cmd[0] == "start":
                    self._run_task(cmd[1])
                elif cmd[0] == "reset":
                    self.abort_requested = False
                    self.sim.reset_scene(cmd[1])
                    self.finished = False
                    self.current_step = ""
                    self.message = "Scene reset"
                    self.sim.env.step(0.05)
                elif cmd[0] == "call":
                    if self.can_teach():
                        msg = cmd[1]()
                        self.message = msg or cmd[2]
            except TaskAborted:
                pass
            except Exception as e:
                traceback.print_exc()
                self.message = f"Error: {e}"

    def _run_task(self, n_glasses):
        sim = self.sim
        sim.reset_scene(n_glasses)                # every run starts from the start position
        self.abort_requested = False
        self.halted = False
        self.paused = False
        self.task_running = True
        self.finished = False
        self.message = ""
        try:
            sim.run_task()
            failed = [label for label, ok in sim.step_log if not ok]
            self.message = ("Task complete" if not failed
                            else f"Task finished, {len(failed)} step(s) failed - see the log")
        except TaskAborted:
            self.message = "Run abandoned"
        finally:
            self.task_running = False
            self.paused = False
            self.waiting_for_step = False
            self.finished = True
            self.current_step = ""


class ControlPanel:
    """The tkinter window."""

    BG = "#f4f4f4"

    def __init__(self, controller):
        self.c = controller
        self.root = tk.Tk()
        self.root.title("Cucumber-Water - Control Panel")
        self.root.configure(bg=self.BG)
        self.root.minsize(560, 640)
        self.root.protocol("WM_DELETE_WINDOW", self.quit)
        self.root.bind("<Escape>", lambda e: self.c.estop("E-STOP key"))
        self.root.bind("<space>", lambda e: self.c.estop("E-STOP key"))
        style = ttk.Style(self.root)
        try:
            style.theme_use("clam")
        except tk.TclError:
            pass
        style.configure("TButton", padding=6)
        self._log_count = 0
        self._dragging = False
        self._build()
        self.root.after(100, self._refresh)

    # ---------------------------------------------------------------- layout
    def _build(self):
        r = self.root
        self.banner = tk.Label(r, text="LOADING", font=("Segoe UI", 18, "bold"), fg="white",
                               bg="#777777", pady=8)
        self.banner.pack(fill="x")

        run = tk.Frame(r, bg=self.BG, pady=6)
        run.pack(fill="x", padx=10)
        self.btn_start = ttk.Button(run, text="START", command=self.c.start)
        self.btn_pause = ttk.Button(run, text="PAUSE", command=self.c.pause)
        self.btn_resume = ttk.Button(run, text="RESUME", command=self.c.resume)
        for i, b in enumerate((self.btn_start, self.btn_pause, self.btn_resume)):
            b.grid(row=0, column=i, sticky="ew", padx=3)
        self.btn_estop = tk.Button(run, text="EMERGENCY\nSTOP", command=lambda: self.c.estop(),
                                   bg="#d00000", fg="white", activebackground="#a00000",
                                   activeforeground="white", font=("Segoe UI", 14, "bold"),
                                   relief="raised", bd=4, height=2)
        self.btn_estop.grid(row=0, column=3, rowspan=2, sticky="nsew", padx=(10, 0))
        self.btn_reset_estop = ttk.Button(run, text="RESET E-STOP", command=self.c.reset_estop)
        self.btn_reset_estop.grid(row=1, column=0, sticky="ew", padx=3, pady=(4, 0))
        self.var_step = tk.BooleanVar(value=False)
        self.chk_step = ttk.Checkbutton(run, text="Step mode", variable=self.var_step,
                                        command=lambda: self.c.set_step_mode(self.var_step.get()))
        self.chk_step.grid(row=1, column=1, sticky="w", padx=3, pady=(4, 0))
        self.btn_next = ttk.Button(run, text="NEXT STEP", command=self.c.next_step)
        self.btn_next.grid(row=1, column=2, sticky="ew", padx=3, pady=(4, 0))
        self.btn_reset = ttk.Button(run, text="RESET SCENE", command=self.c.reset_scene)
        self.btn_reset.grid(row=2, column=0, columnspan=3, sticky="ew", padx=3, pady=(4, 0))
        for i in range(4):
            run.columnconfigure(i, weight=1)

        info = tk.Frame(r, bg=self.BG)
        info.pack(fill="x", padx=12, pady=(4, 0))
        self.lbl_step = tk.Label(info, text="", bg=self.BG, anchor="w", font=("Segoe UI", 10, "bold"))
        self.lbl_step.pack(fill="x")
        prog = tk.Frame(info, bg=self.BG)
        prog.pack(fill="x", pady=2)
        self.lbl_prog = tk.Label(prog, text="Glass 0 of 0", bg=self.BG, width=14, anchor="w")
        self.lbl_prog.pack(side="left")
        self.progress = ttk.Progressbar(prog, maximum=5)
        self.progress.pack(side="left", fill="x", expand=True)
        self.lbl_msg = tk.Label(info, text="", bg=self.BG, fg="#a04000", anchor="w",
                                wraplength=520, justify="left")
        self.lbl_msg.pack(fill="x")

        nb = ttk.Notebook(r)
        nb.pack(fill="both", expand=True, padx=10, pady=8)
        self.tab_log = tk.Frame(nb, bg=self.BG)
        self.tab_teach = tk.Frame(nb, bg=self.BG)
        self.tab_settings = tk.Frame(nb, bg=self.BG)
        self.tab_safety = tk.Frame(nb, bg=self.BG)
        nb.add(self.tab_log, text="Step log")
        nb.add(self.tab_teach, text="Teach pendant")
        nb.add(self.tab_settings, text="Settings")
        nb.add(self.tab_safety, text="Safety")
        self._build_log()
        self._build_teach()
        self._build_settings()
        self._build_safety()

    def _build_log(self):
        f = self.tab_log
        sb = ttk.Scrollbar(f)
        sb.pack(side="right", fill="y")
        self.log = tk.Listbox(f, yscrollcommand=sb.set, font=("Consolas", 9), activestyle="none")
        self.log.pack(fill="both", expand=True)
        sb.config(command=self.log.yview)

    def _build_teach(self):
        f = self.tab_teach
        top = tk.Frame(f, bg=self.BG)
        top.pack(fill="x", pady=4)
        tk.Label(top, text="Robot:", bg=self.BG).pack(side="left")
        self.var_robot = tk.StringVar(value="Omron TM5-700")
        self.cmb_robot = ttk.Combobox(top, textvariable=self.var_robot, state="readonly", width=16,
                                      values=["Omron TM5-700", "Dobot CR5", "Motoman GP4"])
        self.cmb_robot.pack(side="left", padx=6)
        self.cmb_robot.bind("<<ComboboxSelected>>", lambda e: self._load_sliders())
        self.lbl_pose = tk.Label(top, text="", bg=self.BG, font=("Consolas", 9))
        self.lbl_pose.pack(side="left", padx=8)
        self.lbl_manip = tk.Label(f, text="", bg=self.BG, anchor="w", font=("Segoe UI", 9),
                                  wraplength=520, justify="left")
        self.lbl_manip.pack(fill="x", padx=4)

        joints = tk.LabelFrame(f, text="Joints (degrees) - drag, the robot follows",
                               bg=self.BG, padx=6, pady=4)
        joints.pack(fill="x", padx=4)
        self.sliders, self.slider_vals = [], []
        for j in range(6):
            tk.Label(joints, text=f"J{j + 1}", bg=self.BG, width=3).grid(row=j, column=0)
            var = tk.DoubleVar()
            sc = tk.Scale(joints, from_=-360, to=360, orient="horizontal", variable=var,
                          resolution=0.5, showvalue=False, length=340, bg=self.BG,
                          highlightthickness=0, command=lambda v, j=j: self._slider_moved(j))
            sc.grid(row=j, column=1, sticky="ew")
            sc.bind("<ButtonPress-1>", lambda e: self._set_dragging(True))
            sc.bind("<ButtonRelease-1>", lambda e: self._set_dragging(False))
            lbl = tk.Label(joints, text="0.0°", bg=self.BG, width=8, font=("Consolas", 9))
            lbl.grid(row=j, column=2)
            self.sliders.append((sc, var))
            self.slider_vals.append(lbl)
        joints.columnconfigure(1, weight=1)

        cart = tk.LabelFrame(f, text="Move the hand in a straight line (world x / y / z)",
                             bg=self.BG, padx=6, pady=4)
        cart.pack(fill="x", padx=4, pady=6)
        tk.Label(cart, text="Step:", bg=self.BG).grid(row=0, column=0)
        self.var_jog = tk.StringVar(value="2 cm")
        ttk.Combobox(cart, textvariable=self.var_jog, state="readonly", width=6,
                     values=["1 cm", "2 cm", "5 cm", "10 cm"]).grid(row=0, column=1, padx=4)
        for col, axis in enumerate("xyz"):
            ttk.Button(cart, text=f"+{axis.upper()}", width=5,
                       command=lambda a=axis: self._jog(a, +1)).grid(row=0, column=2 + 2 * col, padx=2)
            ttk.Button(cart, text=f"-{axis.upper()}", width=5,
                       command=lambda a=axis: self._jog(a, -1)).grid(row=0, column=3 + 2 * col, padx=2)

        grip = tk.Frame(f, bg=self.BG)
        grip.pack(fill="x", padx=4)
        ttk.Button(grip, text="Open gripper", command=lambda: self._gripper(False)).pack(side="left", padx=2)
        ttk.Button(grip, text="Close gripper", command=lambda: self._gripper(True)).pack(side="left", padx=2)
        ttk.Button(grip, text="Go home", command=self._home).pack(side="left", padx=2)
        self.lbl_teach = tk.Label(f, text="The teach pendant works when no task is running. "
                                          "START always puts everything back at the start first.",
                                  bg=self.BG, fg="#555555", wraplength=520, justify="left")
        self.lbl_teach.pack(fill="x", padx=4, pady=6)

    def _build_settings(self):
        f = self.tab_settings
        g = tk.Frame(f, bg=self.BG, padx=8, pady=8)
        g.pack(fill="x")
        tk.Label(g, text="Number of glasses:", bg=self.BG).grid(row=0, column=0, sticky="w")
        self.var_glasses = tk.IntVar(value=5)
        ttk.Spinbox(g, from_=1, to=5, textvariable=self.var_glasses, width=4, state="readonly",
                    command=self._glasses_changed).grid(row=0, column=1, sticky="w", padx=6)
        tk.Label(g, text="(used at the next START or RESET SCENE)", bg=self.BG,
                 fg="#666666").grid(row=0, column=2, sticky="w")

        tk.Label(g, text="Playback speed:", bg=self.BG).grid(row=1, column=0, sticky="w", pady=8)
        self.var_speed = tk.DoubleVar(value=movement_handler.PLAYBACK_SPEED)
        tk.Scale(g, from_=0.25, to=4.0, resolution=0.25, orient="horizontal", length=220,
                 variable=self.var_speed, bg=self.BG, highlightthickness=0,
                 command=lambda v: setattr(movement_handler, "PLAYBACK_SPEED", float(v))
                 ).grid(row=1, column=1, columnspan=2, sticky="w")

        self.var_hitboxes = tk.BooleanVar(value=True)
        ttk.Checkbutton(g, text="Show obstacle hitboxes (red boxes)", variable=self.var_hitboxes,
                        command=lambda: self.c.scene(
                            lambda v=self.var_hitboxes.get(): self.c.sim.show_hitboxes(v))
                        ).grid(row=2, column=0, columnspan=3, sticky="w", pady=2)
        self.var_markers = tk.BooleanVar(value=True)
        ttk.Checkbutton(g, text="Show collision markers (red balls where a straight path was blocked)",
                        variable=self.var_markers,
                        command=lambda: self.c.scene(
                            lambda v=self.var_markers.get(): self.c.sim.show_collision_markers(v))
                        ).grid(row=3, column=0, columnspan=3, sticky="w", pady=2)

    def _build_safety(self):
        f = self.tab_safety
        lc = tk.LabelFrame(f, text="Light curtain", bg=self.BG, padx=8, pady=6)
        lc.pack(fill="x", padx=4, pady=4)
        self.lbl_curtain = tk.Label(lc, text="CLEAR", bg="#1a9a3a", fg="white", width=12,
                                    font=("Segoe UI", 11, "bold"))
        self.lbl_curtain.grid(row=0, column=0, rowspan=2, padx=(0, 10))
        ttk.Button(lc, text="Person walks in", command=lambda: self.c.person_walk(True)
                   ).grid(row=0, column=1, sticky="ew", padx=2)
        ttk.Button(lc, text="Person walks away", command=lambda: self.c.person_walk(False)
                   ).grid(row=1, column=1, sticky="ew", padx=2, pady=2)
        tk.Label(lc, text="A beam across the open side of the work area. Breaking it triggers the "
                          "e-stop. It can only be reset once the person has left.",
                 bg=self.BG, wraplength=330, justify="left").grid(row=0, column=2, rowspan=2, padx=8)

        ob = tk.LabelFrame(f, text="Obstacle in a robot's way", bg=self.BG, padx=8, pady=6)
        ob.pack(fill="x", padx=4, pady=4)
        ttk.Button(ob, text="Put a pot in the sink", command=lambda: self.c.scene(
            lambda: self.c.sim.add_demo_obstacle())).grid(row=0, column=0, sticky="ew", padx=2)
        ttk.Button(ob, text="Remove the pot", command=lambda: self.c.scene(
            lambda: self.c.sim.remove_demo_obstacle())).grid(row=1, column=0, sticky="ew", padx=2, pady=2)
        tk.Label(ob, text="An orange pot appears in the sink, on the Dobot's way from the counter "
                          "to the tap. If a robot's current move would hit it, the robot stops "
                          "before touching it and plans a new path round it; later moves avoid it too.",
                 bg=self.BG, wraplength=380, justify="left").grid(row=0, column=1, rowspan=2, padx=8)

    # ------------------------------------------------------------ teach pendant
    def _robot(self):
        s = self.c.sim
        return {"Omron TM5-700": s.OMRON, "Dobot CR5": s.DOBOT, "Motoman GP4": s.Motoman}[self.var_robot.get()]

    def _load_sliders(self):
        """Sliders <- the selected robot's current joints and limits."""
        if not self.c.ready:
            return
        robot = self._robot()
        lim = np.degrees(movement_handler.get_joint_limits(robot))
        q = np.degrees(np.asarray(robot.q, float))
        self._loading = True
        for j, (sc, var) in enumerate(self.sliders):
            sc.configure(from_=lim[0, j], to=lim[1, j])
            var.set(round(q[j], 1))
            self.slider_vals[j].configure(text=f"{q[j]:.1f}°")
        self._loading = False

    def _set_dragging(self, on):
        self._dragging = on
        if not on:
            self._send_joints()

    def _slider_moved(self, j):
        if getattr(self, "_loading", False):
            return
        self.slider_vals[j].configure(text=f"{self.sliders[j][1].get():.1f}°")
        if not self._dragging:                  # arrow keys / clicks on the trough
            self._send_joints()

    def _send_joints(self):
        if not self.c.can_teach():
            self._load_sliders()
            return
        robot = self._robot()
        q = np.radians([var.get() for _, var in self.sliders])
        self.c.submit(lambda: self.c.sim.teach_set_joints(robot, q), "")

    def _jog(self, axis, sign):
        d = sign * float(self.var_jog.get().split()[0]) / 100.0
        robot = self._robot()
        kw = {"dx": d if axis == "x" else 0.0, "dy": d if axis == "y" else 0.0,
              "dz": d if axis == "z" else 0.0}
        self.c.submit(lambda: self.c.sim.teach_jog(robot, **kw), f"Moved {axis} {d * 100:+.0f} cm")

    def _gripper(self, close):
        robot = self._robot()
        self.c.submit(lambda: self.c.sim.teach_gripper(robot, close),
                      "Gripper closed" if close else "Gripper open")

    def _home(self):
        robot = self._robot()
        self.c.submit(lambda: self.c.sim.teach_home(robot), f"{robot.name} at home")

    def _glasses_changed(self):
        self.c.n_glasses = int(self.var_glasses.get())

    # ---------------------------------------------------------------- refresh
    def _refresh(self):
        c = self.c
        c.check_curtain()
        text, colour = c.state()
        self.banner.configure(text=text, bg=colour)
        busy = c.task_running
        frozen_estop = c.latched or c.need_resume
        self._enable(self.btn_start, c.ready and not busy and not frozen_estop)
        self._enable(self.btn_pause, busy and not c.paused and not c.latched)
        self._enable(self.btn_resume, (c.paused or c.need_resume) and not c.latched)
        self._enable(self.btn_reset_estop, c.latched)
        self._enable(self.btn_next, busy and c.waiting_for_step)
        self._enable(self.btn_reset, c.ready)
        self.btn_estop.configure(relief="sunken" if c.latched else "raised")
        self.lbl_step.configure(text=f"Current step: {c.current_step}" if busy and c.current_step else "")
        if c.sim is not None and c.ready:
            done, total = c.sim.progress
            self.lbl_prog.configure(text=f"Glass {done} of {total}")
            self.progress.configure(maximum=max(total, 1), value=done)
            inside = c.sim.person_inside_curtain()
            self.lbl_curtain.configure(text="BROKEN" if inside else "CLEAR",
                                       bg="#d00000" if inside else "#1a9a3a")
            self._refresh_log()
            if not self._dragging:
                if not getattr(self, "_sliders_loaded", False):
                    self._load_sliders()
                    self._sliders_loaded = True
                elif c.cmds.empty():
                    self._sync_sliders()
            robot = self._robot()
            p = robot.fkine(robot.q).t
            self.lbl_pose.configure(text=f"hand x {p[0]:+.3f}  y {p[1]:+.3f}  z {p[2]:+.3f} m")
            level = c.sim.singularity_level(robot)
            if level < c.sim.SINGULAR_WARNING:
                if getattr(self, "_advice_q", None) is None or not np.allclose(self._advice_q, robot.q):
                    self._advice_q = np.array(robot.q, dtype=float)
                    self._advice = c.sim.singularity_advice(robot)
                self.lbl_manip.configure(fg="#c00000", text=f"Manipulability {level:.2f} x typical - AT / NEAR A "
                                         f"SINGULARITY: some straight-line moves are impossible. {self._advice}")
            else:
                self.lbl_manip.configure(fg="#1a7a2a", text=f"Manipulability {level:.2f} x typical - clear of singularities")
        self.lbl_msg.configure(text=c.message)
        self.root.after(100, self._refresh)

    def _sync_sliders(self):
        """Keep sliders matching the robot when it's moved by something else."""
        q = np.degrees(np.asarray(self._robot().q, float))
        current = np.array([var.get() for _, var in self.sliders])
        if np.max(np.abs(current - q)) > 0.5:
            self._load_sliders()

    def _refresh_log(self):
        log = self.c.sim.step_log
        if len(log) < self._log_count:            # scene was reset
            self.log.delete(0, "end")
            self._log_count = 0
        for label, ok in log[self._log_count:]:
            self.log.insert("end", f"{'OK    ' if ok else 'FAILED'}  {label}")
            self.log.itemconfig("end", fg="#1a7a2a" if ok else "#c00000")
            self.log.see("end")
        self._log_count = len(log)

    @staticmethod
    def _enable(widget, on):
        widget.configure(state="normal" if on else "disabled")

    def quit(self):
        self.c.estop("window closed")
        self.c.abort_requested = True
        self.c.cmds.put(("quit",))
        self.root.after(300, lambda: os._exit(0))  # Swift's server threads don't stop on their own

    def run(self):
        self.root.mainloop()


def main(speed=1.0):
    from task_simulation import TaskSimulation
    movement_handler.PLAYBACK_SPEED = speed
    controller = SimController()
    threading.Thread(target=controller.run,
                     args=(lambda c: TaskSimulation(speed=speed, controller=c),),
                     daemon=True).start()
    ControlPanel(controller).run()


if __name__ == "__main__":
    main()
