import json
import logging
import math
import pathlib
import threading
import time
from typing import Callable, Dict, Optional, List, TextIO
from pydantic import BaseModel, ConfigDict, Field
import datetime
import serial           # Only needed by UsbGps pyserial
import serial.tools.list_ports
import pynmea2          # Only needed by UsbGps pynmea2

LOGGER = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
# Pydantic model for the current fix
# --------------------------------------------------------------------------- #
class GpsFix(BaseModel):
    # ------------------ Validity / position ------------------
    state: int = Field(-2, description="-2 disconnected; -1 no data; 0 valid fix")
    states_str: list[str] = ["-2 disconnected","-1 no data","0 valid fix"]
    lat: Optional[float] = Field(35.6851, description="Latitude in decimal degrees")
    lon: Optional[float] = Field(139.7527, description="Longitude in decimal degrees")
    alt_msl: Optional[float] = Field(20.0, description="Altitude above mean sea level (meters)")
    geoid_sep: Optional[float] = Field(None, description="Geoid separation (meters)")

    # ---------------------- Motion -------------------------
    speed_kn: Optional[float] = Field(0.0, description="Speed over ground (knots)")
    track_deg: Optional[float] = Field(None, description="Track/course over ground (degrees)")

    # ----------------------- Timing ------------------------
    utc_time: Optional[datetime.time] = Field(None, description="UTC time")
    utc_date: Optional[datetime.date] = Field(None, description="UTC date")

    # ---------------- Quality / satellites -----------------
    fix_quality: Optional[int] = Field(None, description="Fix quality: 0=invalid, 1=GPS, 2=DGPS, etc.")
    sats_used: Optional[float] = Field(None, description="Number of satellites used (GGA)")
    sats_in_view: Optional[int] = Field(None, description="Number of satellites in view (GSV)")
    hdop: Optional[float] = Field(None, description="Horizontal dilution of precision")
    vdop: Optional[float] = Field(None, description="Vertical dilution of precision")
    pdop: Optional[float] = Field(None, description="Position dilution of precision")

    # ------------------- GST error estimates ----------------
    rms: Optional[float] = Field(None, description="RMS value of pseudorange residuals")
    std_major: Optional[float] = Field(None, description="Standard deviation of semi-major axis (meters)")
    std_minor: Optional[float] = Field(None, description="Standard deviation of semi-minor axis (meters)")
    std_alt: Optional[float] = Field(None, description="Standard deviation of altitude (meters)")

    # ---------------- Antenna / notices -------------------
    notices: List[str] = Field(default_factory=list, description="List of TXT/antenna messages")

    # ---------------- Model config ------------------------
    model_config = ConfigDict(validate_assignment=True)



# --------------------------------------------------------------------------- #
# Base class (no abc) – common behaviour
# --------------------------------------------------------------------------- #

class BaseGps(BaseModel):
    """
    Foundation for USB / emulator implementations.
    Sub-classes only need to override the three *_device/update hooks.
    """
    _state: GpsFix = GpsFix()
    _opened: bool  = False

    # --- observer pattern -------------------------------------------- #
    _listeners: Dict[str, Callable[[Optional[GpsFix | Exception]], None]] = {}

    # --- background watcher ------------------------------------------ #
    _stop_event:threading.Event = None
    _watch_thread: Optional[threading.Thread] = None
    _watch_interval: float = 0.25  # seconds between updates

    # ------------------------------------------------------------------- #
    # Public API
    # ------------------------------------------------------------------- #
    @staticmethod
    def coms():
        ports = serial.tools.list_ports.comports()
        port_list = [ f'{port.device},{port.hwid}' for port in ports]
        return port_list
    
    def model_post_init(self, context):
        self._stop_event = threading.Event()
        return super().model_post_init(context)

    def on(self, event: str, cb: Callable[[Optional[GpsFix | Exception]], None]) -> None:
        """Subscribe to *opened*, *closed*, *update*, or *error* events."""
        self._listeners[event] = cb

    def open(self, port: str, baudrate: int=9600) -> None:
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

    def _open_device(self, port: str, baudrate: int=9600) -> None:
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

    # ------------------------- Helpers -------------------------
    def _parse_lat(self, lat_str: str, hemi: str) -> float:
        if not lat_str or not hemi:
            return math.nan
        deg = float(lat_str[:2])
        min_ = float(lat_str[2:])
        dec = deg + min_ / 60
        if hemi == "S":
            dec = -dec
        return dec
# --------------------------------------------------------------------------- #
# USB implementation
# --------------------------------------------------------------------------- #

class UsbGps(BaseGps):
    """Real NMEA-over-USB GPS receiver using pyserial + pynmea2."""
    _serial: Optional[serial.Serial] = None


    # -- lifecycle -------------------------------------------------------- #

    def _open_device(self, port: str, baudrate: int=9600) -> None:
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
            self._parse_nmea(pynmea2.parse(raw).__dict__)
            try:
                self._parse_nmea(pynmea2.parse(raw).__dict__)
            except pynmea2.nmea.ParseError:
                pass  # ignore malformed lines

    # -- helpers ----------------------------------------------------------
    def _parse_nmea_time(self, time_str: str) -> str:
        if not time_str or len(time_str) < 6:
            return None
        hour = time_str[0:2]
        minute = time_str[2:4]
        second = time_str[4:6]
        return f"{hour}:{minute}:{second}"
    
    def _parse_nmea_date(self, date_str: str) -> datetime.date | None:
        if not date_str or len(date_str) != 6:
            return None
        day = int(date_str[0:2])
        month = int(date_str[2:4])
        year = 2000 + int(date_str[4:6])
        return datetime.date(year, month, day)
    
    def _parse_nmea(self, msg: dict) -> None:
        stype = msg["sentence_type"]
        data = msg["data"]

        # ----------------------------- RMC -----------------------------
        if stype == "RMC":
            status = data[1]
            if status == "A":
                self._state.state = 0
                self._state.lat = self._parse_lat(data[2], data[3])
                self._state.lon = self._parse_lon(data[4], data[5])
                self._state.speed_kn = float(data[6] or 0)
                self._state.track_deg = float(data[7] or math.nan)
            else:
                self._state.state = -1

            if data[0] and data[8]:
                self._state.utc_time = self._parse_nmea_time(data[0])
                self._state.utc_date = self._parse_nmea_date(data[8])

        # ----------------------------- GGA -----------------------------
        elif stype == "GGA":
            self._state.fix_quality = int(data[5] or 0)
            self._state.sats_used = float(data[6] or 0)
            self._state.hdop = float(data[7] or math.nan)
            if self._state.fix_quality:
                self._state.lat = self._parse_lat(data[1], data[2])
                self._state.lon = self._parse_lon(data[3], data[4])
                self._state.alt_msl = float(data[8] or math.nan)
                self._state.geoid_sep = float(data[10] or math.nan)
                self._state.state = 0

        # ----------------------------- GSA -----------------------------
        elif stype == "GSA":
            self._state.pdop = float(data[14]) if len(data) > 14 and data[14] != "" else math.nan
            self._state.hdop = float(data[15]) if len(data) > 15 and data[15] != "" else (self._state.hdop or math.nan)
            self._state.vdop = float(data[16]) if len(data) > 16 and data[16] != "" else math.nan

            if int(data[1] or 1) > 1:
                self._state.state = 0
            else:
                self._state.state = -1

        # ----------------------------- GSV -----------------------------
        elif stype == "GSV":
            self._state.sats_in_view = int(data[2] or 0)

        # ----------------------------- VTG -----------------------------
        elif stype == "VTG":
            self._state.track_deg = float(data[0] or math.nan)
            self._state.speed_kn = float(data[4] or 0)

        # ----------------------------- GLL -----------------------------
        elif stype == "GLL":
            status = data[5]
            if status == "A":
                self._state.lat = self._parse_lat(data[0], data[1])
                self._state.lon = self._parse_lon(data[2], data[3])
                self._state.state = 0
            else:
                self._state.state = -1

            if data[4]:
                self._state.utc_time = self._parse_nmea_time(data[4])

        # ----------------------------- GNS -----------------------------
        elif stype == "GNS":
            self._state.sats_used = float(data[6] or 0)
            self._state.hdop = float(data[7] or math.nan)
            self._state.alt_msl = float(data[8] or math.nan)
            self._state.geoid_sep = float(data[9] or math.nan)

            self._state.lat = self._parse_lat(data[1], data[2])
            self._state.lon = self._parse_lon(data[3], data[4])

            if data[5] and any(c in data[5] for c in ("A", "D")):
                self._state.state = 0
            else:
                self._state.state = -1

            if data[0]:
                self._state.utc_time = self._parse_nmea_time(data[0])

        # ----------------------------- GST -----------------------------
        elif stype == "GST":
            self._state.rms = float(data[1]) if len(data) > 1 and data[1] != "" else math.nan
            self._state.std_major = float(data[5]) if len(data) > 5 and data[5] != "" else math.nan
            self._state.std_minor = float(data[6]) if len(data) > 6 and data[6] != "" else math.nan
            self._state.std_alt = float(data[7]) if len(data) > 7 and data[7] != "" else math.nan

            if data[0]:
                self._state.utc_time = self._parse_nmea_time(data[0])

        # ----------------------------- ZDA -----------------------------
        elif stype == "ZDA":
            if data[0]:
                self._state.utc_time = self._parse_nmea_time(data[0])
            if data[1] and data[2] and data[3]:
                day = int(data[1])
                month = int(data[2])
                year = int(data[3])
                self._state.utc_date = datetime.date(year, month, day)

        # ----------------------------- TXT -----------------------------
        elif stype == "TXT":
            self._state.notices.append(" ".join(data[3:]))


    def _parse_lon(self, lon_str: str, hemi: str) -> float:
        if not lon_str or not hemi:
            return math.nan
        deg = float(lon_str[:3])
        min_ = float(lon_str[3:])
        dec = deg + min_ / 60
        if hemi == "W":
            dec = -dec
        return dec



class FileReplayGps(UsbGps):
    loop:bool = True
    _fh: Optional[TextIO] = None

    # ------------------------------------------------------------------- #
    # Device-lifecycle hooks
    # ------------------------------------------------------------------- #

    def _open_device(self, file_path: str, baudrate: int=0) -> None:   # port/baud ignored
        file_path = pathlib.Path(file_path)
        if not file_path.exists():
            raise FileNotFoundError(file_path)
        self._fh = file_path.open("r", encoding="utf-8", errors="ignore")
        self._state.state = -1          # no fix until we parse one

    def _close_device(self) -> None:
        if self._fh and not self._fh.closed:
            self._fh.close()
        self._fh = None

    # ------------------------------------------------------------------- #
    # Polling hook – read next line & apply
    # ------------------------------------------------------------------- #
    def _update_state(self) -> None:
        if not self._fh:
            raise RuntimeError("File not opened")

        line = self._fh.readline()
        if not line:                       # EOF
            if self.loop:
                self._fh.seek(0)
                line = self._fh.readline()
            else:
                time.sleep(self._watch_interval)
                return                     # nothing more to do

        line = line.strip()
        if not line:
            return                         # skip blanks

        d = json.loads(line)
        self._parse_nmea(d)
            
    def _parse_nmea(self, msg: dict) -> None:
        return super()._parse_nmea(msg)

# if __name__ == '__main__':
#     # gps = UsbGps()           # or EmuGps()
#     # gps.on("update", lambda fix: print(fix.model_dump()))
#     # gps.open("COM4", 9600)
#     gps = FileReplayGps()  
#     gps.on("update", lambda fix: print(fix.model_dump())) 
#     gps.open('gps.jsonl')
#     time.sleep(100)
#     # … later …
#     gps.close()
