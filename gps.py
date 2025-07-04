import logging
import math
import threading
import time
from typing import Callable, Dict, Optional, List
from pydantic import BaseModel, ConfigDict, Field
import datetime
import serial           # Only needed by UsbGps pyserial
import pynmea2          # Only needed by UsbGps pynmea2

LOGGER = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
# Pydantic model for the current fix
# --------------------------------------------------------------------------- #
class GpsFix(BaseModel):
    # validity / position ---------------------------------------------------
    state: int           = Field(-2, description="-2 disconnected; -1 no data; 0 valid fix")
    lat:   Optional[float] = 35.6851
    lon:   Optional[float] = 139.7527
    alt_msl: Optional[float] = 20.0          # metres above mean sea-level
    geoid_sep: Optional[float] = None        # metres
    # motion ---------------------------------------------------------------
    speed_kn: Optional[float] = 0.0          # knots
    track_deg: Optional[float] = None        # course over ground
    # timing ---------------------------------------------------------------
    utc_time: Optional[datetime.time] = None
    utc_date: Optional[datetime.date] = None
    # quality / satellites -------------------------------------------------
    fix_quality: Optional[int] = None        # 0 invalid, 1 GPS, 2 DGPS, …
    sats_used:   Optional[int] = None        # from GGA
    sats_in_view: Optional[int] = None       # from GSV
    hdop: Optional[float] = None
    vdop: Optional[float] = None
    pdop: Optional[float] = None
    # antenna / receiver text (TXT) ---------------------------------------
    notices: List[str] = Field(default_factory=list)

    model_config = ConfigDict(validate_assignment=True)


# --------------------------------------------------------------------------- #
# Base class (no abc) – common behaviour
# --------------------------------------------------------------------------- #

class BaseGps:
    """
    Foundation for USB / emulator implementations.
    Sub-classes only need to override the three *_device/update hooks.
    """

    _watch_interval: float = 0.25  # seconds between updates

    def __init__(self) -> None:
        self._state: GpsFix = GpsFix()
        self._opened: bool  = False

        # --- observer pattern -------------------------------------------- #
        self._listeners: Dict[str, Callable[[Optional[GpsFix | Exception]], None]] = {}

        # --- background watcher ------------------------------------------ #
        self._stop_event = threading.Event()
        self._watch_thread: Optional[threading.Thread] = None

    # ------------------------------------------------------------------- #
    # Public API
    # ------------------------------------------------------------------- #

    def on(self, event: str, cb: Callable[[Optional[GpsFix | Exception]], None]) -> None:
        """Subscribe to *opened*, *closed*, *update*, or *error* events."""
        self._listeners[event] = cb

    def open(self, port: str, baudrate: int) -> None:
        """Open hardware (or emulator) and start background polling."""
        self._open_device(port, baudrate)
        self._opened = True
        self._emit("opened")
        self._start_watcher()

    def close(self) -> None:
        """Stop polling and release resources."""
        self._stop_watcher()
        self._close_device()
        self._opened = False
        self._state.state = -2
        self._emit("closed")

    @property
    def is_open(self) -> bool:          # handy convenience
        return self._opened

    def get_state(self) -> GpsFix:
        """Return a *copy* of the last known fix."""
        return self._state.model_copy()

    # ------------------------------------------------------------------- #
    # Hooks – override these three in concrete subclasses
    # ------------------------------------------------------------------- #

    def _open_device(self, port: str, baudrate: int) -> None:
        raise NotImplementedError

    def _close_device(self) -> None:
        raise NotImplementedError

    def _update_state(self) -> None:
        """Fetch fresh data and write into *self._state*."""
        raise NotImplementedError

    # ------------------------------------------------------------------- #
    # Internals
    # ------------------------------------------------------------------- #

    def _start_watcher(self) -> None:
        self._stop_event.clear()
        self._watch_thread = threading.Thread(
            target=self._watch_loop,
            daemon=True,
            name=f"{self.__class__.__name__}Watcher",
        )
        self._watch_thread.start()

    def _stop_watcher(self) -> None:
        self._stop_event.set()
        if self._watch_thread and self._watch_thread.is_alive():
            self._watch_thread.join(timeout=2.0)
        self._watch_thread = None

    def _watch_loop(self) -> None:
        while not self._stop_event.is_set():
            try:
                self._update_state()
                self._emit("update", self.get_state())
            except Exception as exc:        # isolate & propagate background errors
                LOGGER.exception("GPS update error: %s", exc)
                self._emit("error", exc)
            finally:
                self._stop_event.wait(self._watch_interval)

    def _emit(self, event: str, data: Optional[GpsFix | Exception] = None) -> None:
        if cb := self._listeners.get(event):
            cb(data)


# --------------------------------------------------------------------------- #
# USB implementation
# --------------------------------------------------------------------------- #

class UsbGps(BaseGps):
    """Real NMEA-over-USB GPS receiver using pyserial + pynmea2."""
    _serial: Optional[serial.Serial] = None


    # -- lifecycle -------------------------------------------------------- #

    def _open_device(self, port: str, baudrate: int) -> None:
        self._serial = serial.Serial(port, baudrate, timeout=1)
        self._state.state = -1          # no fix yet

    def _close_device(self) -> None:
        if self._serial and self._serial.is_open:
            self._serial.close()
        self._serial = None

    # -- polling loop ----------------------------------------------------- #

    def _update_state(self) -> None:
        if not self._serial or not self._serial.is_open:
            raise ConnectionError("Serial port closed unexpectedly")

        raw = self._serial.readline().decode("ascii", errors="ignore").strip()
        if raw.startswith("$"):
            try:
                self._parse_nmea(pynmea2.parse(raw))
            except pynmea2.nmea.ParseError:
                pass  # ignore malformed lines

    # -- helpers ---------------------------------------------------------- #
    def _parse_nmea(self, msg: pynmea2.NMEASentence) -> None:
        print(msg.__dict__)

        stype = msg.sentence_type

        # ------------------------------------------------------ RMC ---------
        if stype == "RMC":
            if msg.status == "A":                       # A = Active (valid)
                self._state.state     = 0
                self._state.lat       = float(msg.latitude)
                self._state.lon       = float(msg.longitude)
                self._state.speed_kn  = float(msg.spd_over_grnd or 0)
                self._state.track_deg = float(msg.true_course or math.nan)
            else:
                self._state.state     = -1             # no fix yet
            # Date & time always present in RMC
            if msg.datestamp and msg.timestamp:
                self._state.utc_date  = msg.datestamp
                self._state.utc_time  = msg.timestamp

        # ------------------------------------------------------ GGA ---------
        elif stype == "GGA":
            self._state.fix_quality = int(msg.gps_qual or 0)
            self._state.sats_used   = int(msg.num_sats or 0)
            self._state.hdop        = float(msg.horizontal_dil or math.nan)
            if self._state.fix_quality:
                self._state.lat      = float(msg.latitude)
                self._state.lon      = float(msg.longitude)
                self._state.alt_msl  = float(msg.altitude or math.nan)
                self._state.geoid_sep = float(msg.geo_sep or math.nan)
                self._state.state    = 0

        # ------------------------------------------------------ GSA ---------
        elif stype == "GSA":
            self._state.pdop = float(msg.pdop or math.nan)
            self._state.hdop = float(msg.hdop or self._state.hdop or math.nan)
            self._state.vdop = float(msg.vdop or math.nan)
            # A fix type of 2 or 3 implies a valid solution
            if int(msg.mode_fix_type or 1) > 1:
                self._state.state = 0

        # ------------------------------------------------------ GSV ---------
        elif stype == "GSV":
            try:
                self._state.sats_in_view = int(msg.num_sv_in_view or 0)
            except AttributeError:     # some pynmea2 versions use num_sv
                self._state.sats_in_view = int(getattr(msg, "num_sv", 0))

        # ------------------------------------------------------ VTG ---------
        elif stype == "VTG":
            if hasattr(msg,"status") and msg.status == "A":
                self._state.track_deg = float(msg.true_track or math.nan)
                self._state.speed_kn  = float(msg.spd_over_grnd_kts or 0)

        # ------------------------------------------------------ TXT ---------
        elif stype == "TXT":
            self._state.notices.append(" ".join(msg.data[3:]))

        # -------------------------------------------------- default ---------
        # Other sentences (ZDA, GST, …) can be handled the same way.

# --------------------------------------------------------------------------- #
# Emulator implementation
# --------------------------------------------------------------------------- #

class EmuGps(UsbGps):
    """Deterministic emulator – instantly ‘fixed’ at Tokyo Station."""

    def _open_device(self, port: str, baudrate: int) -> None:
        self._data = None

    def _close_device(self) -> None:
        pass                    # nothing to release

    def _update_state(self) -> None:
        time.sleep(0.05)        # simulate minimal work

if __name__ == '__main__':
    gps = UsbGps()           # or EmuGps()
    # gps.on("update", lambda fix: print(fix.model_dump()))
    gps.open("COM4", 9600)
    time.sleep(100)
    # … later …
    gps.close()
