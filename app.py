"""
ParkVision - Smart Parking CV Detection (Python / Streamlit version)

Original project: React + Express (TypeScript). This single file contains:
  1. Parking logic (metrics, row breakdown, exact report formatting)
  2. Detection engines: Gemini vision (API key) or local YOLO (no key) - never fake data for your images
  3. Streamlit UI (upload / camera / demo lot, overlay, report, vehicle stats)

Run:
    pip install -r requirements.txt
    streamlit run app.py
"""

from __future__ import annotations

import base64
import io
import json
import os
import re
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import streamlit as st
from PIL import Image, ImageDraw, ImageOps

try:  # optional: only needed if you keep your key in a .env file
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    pass

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
# Same model name as the original project. Change it in .env if needed.
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-3.8-flash")

# Local detector (no API key needed): Ultralytics YOLO, COCO-pretrained.
YOLO_MODEL = os.getenv("YOLO_MODEL", "yolov8s.pt")  # use yolov8m.pt / yolov8l.pt for better accuracy
YOLO_CONF = float(os.getenv("YOLO_CONF", "0.20"))
YOLO_IMGSZ = int(os.getenv("YOLO_IMGSZ", "1280"))  # larger = better for small bikes seen from above

LINE = "━" * 30

# ---------------------------------------------------------------------------
# 1. PARKING LOGIC  (port of src/utils/parkingLogic.ts)
# ---------------------------------------------------------------------------

STATUS_EMOJI = {"EMPTY": "🟢", "OCCUPIED": "🔴", "UNCLEAR": "🟡"}
VALID_SPACE_STATUS = ("EMPTY", "OCCUPIED", "UNCLEAR")


def js_round(x: float) -> int:
    # JavaScript-style Math.round: round half up (Python round() rounds half to even)
    return int(x + 0.5)


def normalize_vehicle_type(value: Optional[str]) -> Optional[str]:
    """Maps any free-text vehicle label to Car | Bike/Motorcycle | Truck/Other (or None)."""
    if value is None:
        return None
    t = str(value).strip().lower()
    if t in ("", "null", "none", "n/a"):
        return None
    if any(k in t for k in ("bike", "motor", "scooter", "moped", "cycle", "two-wheel", "2-wheel")):
        return "Bike/Motorcycle"
    if any(k in t for k in ("truck", "van", "bus", "lorry", "pickup", "other")):
        return "Truck/Other"
    if any(k in t for k in ("car", "sedan", "suv", "hatch", "jeep", "taxi", "wagon")):
        return "Car"
    return "Truck/Other"


def get_status_emoji(status: str) -> str:
    return STATUS_EMOJI.get(status, "🟡")


def calculate_parking_status(rate: int) -> str:
    if rate >= 90:
        return "FULL"
    if rate >= 60:
        return "MODERATE"
    return "AVAILABLE"


@dataclass
class ParkingSpace:
    id: str
    row: int
    col: int
    status: str  # EMPTY | OCCUPIED | UNCLEAR
    box_2d: Optional[List[int]] = None  # [ymin, xmin, ymax, xmax] on a 0-1000 scale
    vehicleType: Optional[str] = None  # Car | Bike/Motorcycle | Truck/Other
    confidence: Optional[float] = None
    note: Optional[str] = None


@dataclass
class ParkingAnalysisData:
    totalSpaces: int = 0
    occupiedSpaces: int = 0
    emptySpaces: int = 0
    unclearSpaces: int = 0
    occupancyRate: int = 0
    status: str = "AVAILABLE"
    rows: List[Dict[str, Any]] = field(default_factory=list)  # {rowNumber, spaces, emojis}
    vehicles: Dict[str, int] = field(
        default_factory=lambda: {"cars": 0, "bikesMotorcycles": 0, "trucksOther": 0, "total": 0}
    )
    insight: str = ""
    spaces: List[ParkingSpace] = field(default_factory=list)
    imageTitle: Optional[str] = None
    imageIndex: Optional[int] = None
    isUnclearOrInvalid: bool = False
    unclearReason: Optional[str] = None
    emptyUnknown: bool = False  # True when the engine can only see vehicles, not empty bays


def compute_parking_metrics(
    spaces: List[ParkingSpace],
    vehicles_input: Optional[Dict[str, int]] = None,
    custom_insight: Optional[str] = None,
) -> ParkingAnalysisData:
    total = len(spaces)
    occupied = empty = unclear = 0
    cars = bikes = trucks = 0
    row_map: Dict[int, List[str]] = {}

    for s in spaces:
        if s.status == "OCCUPIED":
            occupied += 1
            if s.vehicleType == "Bike/Motorcycle":
                bikes += 1
            elif s.vehicleType == "Truck/Other":
                trucks += 1
            else:  # "Car" or unspecified -> default to car
                cars += 1
        elif s.status == "EMPTY":
            empty += 1
        else:
            unclear += 1
        row_map.setdefault(s.row or 1, []).append(s.status)

    rate = js_round(occupied / total * 100) if total > 0 else 0
    status = calculate_parking_status(rate)

    rows = [
        {"rowNumber": rn, "spaces": row_map[rn], "emojis": [get_status_emoji(x) for x in row_map[rn]]}
        for rn in sorted(row_map)
    ]

    vi = vehicles_input or {}
    v_cars = vi["cars"] if vi.get("cars") is not None else cars
    v_bikes = vi["bikesMotorcycles"] if vi.get("bikesMotorcycles") is not None else bikes
    v_trucks = vi["trucksOther"] if vi.get("trucksOther") is not None else trucks
    vehicles = {
        "cars": v_cars,
        "bikesMotorcycles": v_bikes,
        "trucksOther": v_trucks,
        "total": v_cars + v_bikes + v_trucks,
    }

    insight = custom_insight
    if not insight:
        if rate >= 90:
            insight = f"Parking capacity is critically strained ({rate}%). Direct incoming drivers to alternate facilities."
        elif rate >= 60:
            insight = f"Moderate occupancy ({rate}%). Ample spaces remain in deeper rows with smooth flow."
        else:
            insight = f"Plenty of open spaces available ({empty} vacant bays). Optimal parking conditions."

    return ParkingAnalysisData(
        totalSpaces=total,
        occupiedSpaces=occupied,
        emptySpaces=empty,
        unclearSpaces=unclear,
        occupancyRate=rate,
        status=status,
        rows=rows,
        vehicles=vehicles,
        insight=insight,
        spaces=spaces,
    )


def format_exact_report(data: ParkingAnalysisData, image_prefix: Optional[str] = None) -> str:
    """Generates the exact mandated report format."""
    if data.isUnclearOrInvalid:
        header = f"{image_prefix}\n" if image_prefix else ""
        reason = data.unclearReason or (
            "The uploaded image does not provide sufficient clarity or visibility "
            "to reliably identify parking bays and vehicles."
        )
        return (
            f"{header}{LINE}\n📊 PARKING ANALYSIS REPORT\n{LINE}\n"
            f"⚠️ IMAGE UNCLEAR / UNREADABLE:\n{reason}\n{LINE}"
        )

    rows_text = "\n".join(f"Row {r['rowNumber']}: {' '.join(r['emojis'])}" for r in data.rows)
    v = data.vehicles
    total_txt = f"{data.totalSpaces}" if not data.emptyUnknown else f"{data.totalSpaces} (vehicles only)"
    empty_txt = f"{data.emptySpaces}" if not data.emptyUnknown else "N/A (set total bays)"
    rate_txt = f"{data.occupancyRate}%" if not data.emptyUnknown else "N/A"
    status_txt = data.status if not data.emptyUnknown else "UNKNOWN"
    report = (
        f"{LINE}\n"
        f"📊 PARKING ANALYSIS REPORT\n"
        f"{LINE}\n"
        f"🅿️ Total Spaces      : {total_txt}\n"
        f"🔴 Occupied Spaces   : {data.occupiedSpaces}\n"
        f"🟢 Empty Spaces      : {empty_txt}\n"
        f"📈 Occupancy Rate    : {rate_txt}\n"
        f"\n"
        f"📍 SPACE-BY-SPACE BREAKDOWN:\n"
        f"{rows_text or 'No designated rows detected'}\n"
        f"(Use 🟢 for empty, 🔴 for occupied, 🟡 for unclear)\n"
        f"\n"
        f"🚗 VEHICLES DETECTED:\n"
        f"- Cars: {v['cars']}\n"
        f"- Bikes/Motorcycles: {v['bikesMotorcycles']}\n"
        f"- Trucks/Other: {v['trucksOther']}\n"
        f"\n"
        f"💡 INSIGHT:\n"
        f"{data.insight}\n"
        f"\n"
        f"🏁 STATUS: {status_txt}\n"
        f"{LINE}"
    )
    return f"{image_prefix}\n{report}" if image_prefix else report


def format_multi_image_report(reports: List[Dict[str, Any]]) -> str:
    parts = []
    for idx, item in enumerate(reports):
        title = f" [{item['title']}]" if item.get("title") else ""
        parts.append(format_exact_report(item["data"], f"IMAGE {idx + 1}:{title}"))
    return "\n\n".join(parts)


# ---------------------------------------------------------------------------
# 2. SIMULATION FALLBACK  (port of generateMockParkingDetection in server.ts)
# ---------------------------------------------------------------------------


def generate_mock_parking_detection(image_title: str = "Parking Lot") -> ParkingAnalysisData:
    rows_count, cols_per_row = 3, 6
    spaces: List[ParkingSpace] = []
    cars = bikes = trucks = 0

    for r in range(1, rows_count + 1):
        for c in range(1, cols_per_row + 1):
            h = (r * 7 + c * 13) % 10  # deterministic pseudo-random pattern
            status, vehicle = "EMPTY", None
            if h < 6:
                status = "OCCUPIED"
                if h == 5:
                    vehicle = "Bike/Motorcycle"
                    bikes += 1
                elif h == 4 and c == cols_per_row:
                    vehicle = "Truck/Other"
                    trucks += 1
                else:
                    vehicle = "Car"
                    cars += 1
            elif h == 9:
                status = "UNCLEAR"

            ymin = 150 + (r - 1) * 260
            xmin = 60 + (c - 1) * 150
            spaces.append(
                ParkingSpace(
                    id=f"R{r}-S{c:02d}",
                    row=r,
                    col=c,
                    status=status,
                    box_2d=[ymin, xmin, ymin + 200, xmin + 120],
                    vehicleType=vehicle,
                    confidence=0.92 + ((r + c) % 7) * 0.01,
                    note=f"{vehicle} centered in bay" if status == "OCCUPIED" else None,
                )
            )

    result = compute_parking_metrics(
        spaces, {"cars": cars, "bikesMotorcycles": bikes, "trucksOther": trucks}
    )
    result.imageTitle = image_title
    return result


# ---------------------------------------------------------------------------
# 3. GEMINI ANALYSIS  (port of POST /api/analyze-parking in server.ts)
# ---------------------------------------------------------------------------

SYSTEM_INSTRUCTION = """You are an expert Computer Vision engineer specializing in Smart Parking Systems.
Your task is to analyze the uploaded parking lot image and detect parking space occupancy with extreme precision.

Follow these strict DETECTION RULES:
1. Detect all visible marked or designated parking spaces in the image.
2. Group spaces into coherent rows (Row 1, Row 2, etc.) from top-to-bottom or front-to-back perspective.
3. Classify each space as:
   - EMPTY (no vehicle)
   - OCCUPIED (vehicle present)
   - UNCLEAR (partially visible, shadowed, or blocked by poles/pillars)
4. Detection Rules:
   - Count a space as OCCUPIED only if a vehicle clearly occupies it.
   - Do NOT count shadows, pedestrians, shopping carts, or road objects as vehicles.
   - If a vehicle spans two spaces (double parking), assign it to the dominant space and mention it in the note.
   - Account for perspective distortions in angled or tilted camera views.
   - Be accurate and honest - if the image is not a parking lot or completely obscured, mark isUnclearOrInvalid as true and describe why.
5. Provide 2D bounding boxes for each parking space in normalized coordinates [ymin, xmin, ymax, xmax] on a scale of 0 to 1000.
6. Provide exact vehicle counts for: Cars, Bikes/Motorcycles, Trucks/Other.
   - Classify EVERY occupied space with vehicleType. NEVER default to "Car".
   - Two-wheelers (motorcycles, scooters, mopeds, bicycles) are "Bike/Motorcycle", even when seen from above
     (narrow body, handlebars, round wheels, rider seat). Cars are wide 4-wheel vehicles with a roof/windshield.
   - Motorcycle bays are often narrow: draw the box tightly around each bay, one bay per bike.
7. Provide a concise, professional 1-2 sentence insight about the parking situation.
8. Determine status strictly: 'AVAILABLE' (occupancy < 60%), 'MODERATE' (60% to 89%), or 'FULL' (>= 90%)."""

PROMPT_TEXT = (
    "Analyze this parking lot image with high-precision computer vision.\n"
    "Identify every visible parking space, row by row. Return structured JSON matching the schema."
)

RESPONSE_SCHEMA = {
    "type": "OBJECT",
    "properties": {
        "isUnclearOrInvalid": {"type": "BOOLEAN", "description": "True if image cannot be analyzed as a parking lot"},
        "unclearReason": {"type": "STRING", "description": "Reason if image is unclear or not a parking lot"},
        "totalSpaces": {"type": "INTEGER", "description": "Total visible parking bays detected"},
        "occupiedSpaces": {"type": "INTEGER", "description": "Number of spaces with vehicles"},
        "emptySpaces": {"type": "INTEGER", "description": "Number of vacant parking bays"},
        "unclearSpaces": {"type": "INTEGER", "description": "Number of partially visible or blocked bays"},
        "occupancyRate": {"type": "INTEGER", "description": "Calculated occupancy percentage (0-100)"},
        "status": {"type": "STRING", "description": "AVAILABLE, MODERATE, or FULL"},
        "insight": {"type": "STRING", "description": "One or two lines about the parking situation"},
        "vehicles": {
            "type": "OBJECT",
            "properties": {
                "cars": {"type": "INTEGER"},
                "bikesMotorcycles": {"type": "INTEGER"},
                "trucksOther": {"type": "INTEGER"},
            },
            "required": ["cars", "bikesMotorcycles", "trucksOther"],
        },
        "spaces": {
            "type": "ARRAY",
            "description": "Array of detected individual parking bays",
            "items": {
                "type": "OBJECT",
                "properties": {
                    "id": {"type": "STRING"},
                    "row": {"type": "INTEGER"},
                    "col": {"type": "INTEGER"},
                    "status": {"type": "STRING", "description": "EMPTY, OCCUPIED, or UNCLEAR"},
                    "box_2d": {
                        "type": "ARRAY",
                        "description": "[ymin, xmin, ymax, xmax] scaled 0 to 1000",
                        "items": {"type": "INTEGER"},
                    },
                    "vehicleType": {"type": "STRING", "description": "Car, Bike/Motorcycle, Truck/Other, or null"},
                    "note": {"type": "STRING"},
                },
                "required": ["id", "row", "col", "status"],
            },
        },
    },
    "required": ["totalSpaces", "occupiedSpaces", "emptySpaces", "occupancyRate", "status", "vehicles", "spaces", "insight"],
}


def get_api_key() -> str:
    """Key typed in the sidebar (session only) wins over the .env value."""
    try:
        typed = (st.session_state.get("api_key_input") or "").strip()
    except Exception:
        typed = ""
    return typed or GEMINI_API_KEY


def _get_gemini_client():
    key = get_api_key()
    if not key:
        return None
    from google import genai  # imported lazily so the app still runs without the key

    return genai.Client(api_key=key)


def yolo_available() -> bool:
    import importlib.util

    return importlib.util.find_spec("ultralytics") is not None


def invalid_result(file_name: str, index: int, reason: str) -> ParkingAnalysisData:
    d = compute_parking_metrics([], None, "No analysis available.")
    d.imageTitle, d.imageIndex = file_name, index
    d.isUnclearOrInvalid, d.unclearReason = True, reason
    return d


def analyze_with_gemini(image: Image.Image, file_name: str, index: int) -> ParkingAnalysisData:
    from google.genai import types

    client = _get_gemini_client()
    response = client.models.generate_content(
        model=GEMINI_MODEL,
        contents=[types.Part.from_bytes(data=image_to_bytes(image), mime_type="image/jpeg"), PROMPT_TEXT],
        config=types.GenerateContentConfig(
            system_instruction=SYSTEM_INSTRUCTION,
            temperature=0.1,
            response_mime_type="application/json",
            response_schema=RESPONSE_SCHEMA,
        ),
    )
    try:
        parsed = json.loads(response.text or "{}")
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"Gemini returned invalid JSON: {exc}") from exc

    spaces: List[ParkingSpace] = []
    for idx, s in enumerate(parsed.get("spaces") or []):
        box = s.get("box_2d")
        spaces.append(
            ParkingSpace(
                id=s.get("id") or f"S-{idx + 1}",
                row=s.get("row") or 1,
                col=s.get("col") or idx + 1,
                status=s.get("status") if s.get("status") in VALID_SPACE_STATUS else "EMPTY",
                box_2d=box if isinstance(box, list) and len(box) == 4 else None,
                vehicleType=normalize_vehicle_type(s.get("vehicleType")),
                note=s.get("note"),
            )
        )

    # If every occupied bay has a vehicle type, derive the counts from the bays themselves so the
    # report always matches the overlay. Otherwise fall back to the model's own totals.
    occupied = [s for s in spaces if s.status == "OCCUPIED"]
    vehicles_in = None if (occupied and all(s.vehicleType for s in occupied)) else parsed.get("vehicles")

    data = compute_parking_metrics(spaces, vehicles_in, parsed.get("insight"))
    data.imageTitle, data.imageIndex = file_name, index
    if parsed.get("isUnclearOrInvalid"):
        data.isUnclearOrInvalid = True
        data.unclearReason = parsed.get("unclearReason")
    return data


# COCO class ids used by the local YOLO detector
YOLO_CLASS_MAP = {1: "Bike/Motorcycle", 3: "Bike/Motorcycle", 2: "Car", 5: "Truck/Other", 7: "Truck/Other"}


@st.cache_resource(show_spinner="Loading YOLO model (first run downloads weights)...")
def load_yolo(model_name: str):
    from ultralytics import YOLO

    return YOLO(model_name)


def analyze_with_yolo(image: Image.Image, file_name: str, index: int, capacity: int = 0) -> ParkingAnalysisData:
    """Local vehicle detection (no API key). Detects vehicles; empty bays need `capacity` (total bays)."""
    model = load_yolo(YOLO_MODEL)
    res = model.predict(
        image.convert("RGB"), conf=YOLO_CONF, imgsz=YOLO_IMGSZ,
        classes=list(YOLO_CLASS_MAP), agnostic_nms=True, verbose=False,
    )[0]

    w, h = image.size
    dets = []
    for (x1, y1, x2, y2), cls, conf in zip(res.boxes.xyxy.tolist(), res.boxes.cls.tolist(), res.boxes.conf.tolist()):
        dets.append({
            "box": [int(y1 / h * 1000), int(x1 / w * 1000), int(y2 / h * 1000), int(x2 / w * 1000)],
            "cx": (x1 + x2) / 2, "cy": (y1 + y2) / 2, "bh": y2 - y1,
            "type": YOLO_CLASS_MAP[int(cls)], "conf": float(conf),
        })

    # Group detections into rows by vertical position
    dets.sort(key=lambda d: d["cy"])
    rows: List[List[dict]] = []
    if dets:
        med_h = sorted(d["bh"] for d in dets)[len(dets) // 2]
        for d in dets:
            if rows and abs(d["cy"] - sum(x["cy"] for x in rows[-1]) / len(rows[-1])) <= 0.6 * med_h:
                rows[-1].append(d)
            else:
                rows.append([d])

    spaces: List[ParkingSpace] = []
    for r_idx, row in enumerate(rows, start=1):
        for c_idx, d in enumerate(sorted(row, key=lambda x: x["cx"]), start=1):
            spaces.append(ParkingSpace(
                id=f"R{r_idx}-S{c_idx:02d}", row=r_idx, col=c_idx, status="OCCUPIED",
                box_2d=d["box"], vehicleType=d["type"], confidence=round(d["conf"], 2),
                note=f"{d['type']} ({d['conf']:.0%})",
            ))

    data = compute_parking_metrics(spaces)
    data.imageTitle, data.imageIndex = file_name, index
    v = data.vehicles
    summary = (f"Local YOLO detected {v['total']} vehicles "
               f"({v['bikesMotorcycles']} bikes/motorcycles, {v['cars']} cars, {v['trucksOther']} trucks/other).")

    if capacity > 0:
        cap = max(capacity, data.occupiedSpaces)
        data.totalSpaces, data.emptySpaces = cap, cap - data.occupiedSpaces
        data.occupancyRate = js_round(data.occupiedSpaces / cap * 100)
        data.status = calculate_parking_status(data.occupancyRate)
        data.insight = f"{summary} {data.emptySpaces} of {cap} bays vacant."
    else:
        data.emptyUnknown = True
        data.insight = f"{summary} Empty bays can't be detected in this mode - set 'Total bays' in the sidebar."
    return data


def resolve_engine(choice: str) -> Optional[str]:
    if choice == "Gemini":
        return "gemini" if get_api_key() else None
    if choice == "Local YOLO":
        return "yolo" if yolo_available() else None
    # Auto
    if get_api_key():
        return "gemini"
    return "yolo" if yolo_available() else None


def analyze_parking_image(
    image: Image.Image, file_name: str = "Uploaded Parking Lot", index: int = 1,
    engine_choice: str = "Auto", capacity: int = 0,
) -> Dict[str, Any]:
    """Returns a dict with data, mode and error. Never fabricates results for a real image."""
    engine = resolve_engine(engine_choice)
    if engine is None:
        reason = {
            "Gemini": "GEMINI_API_KEY is not set. Add it to your .env file.",
            "Local YOLO": "Ultralytics is not installed. Run: pip install ultralytics",
        }.get(engine_choice, "No detection engine available. Set GEMINI_API_KEY in .env, or run: pip install ultralytics")
        return {"data": invalid_result(file_name, index, reason), "mode": "no-engine", "error": reason}

    try:
        if engine == "gemini":
            return {"data": analyze_with_gemini(image, file_name, index), "mode": "live-gemini", "error": None}
        return {"data": analyze_with_yolo(image, file_name, index, capacity), "mode": "local-yolo", "error": None}
    except Exception as exc:
        # Gemini failed (bad key, quota, wrong model name...) -> try the local detector if available
        if engine == "gemini" and engine_choice == "Auto" and yolo_available():
            try:
                data = analyze_with_yolo(image, file_name, index, capacity)
                return {"data": data, "mode": "local-yolo (Gemini failed)", "error": str(exc)}
            except Exception as exc2:
                exc = exc2
        return {"data": invalid_result(file_name, index, f"Detection failed: {exc}"), "mode": "error", "error": str(exc)}



# ---------------------------------------------------------------------------
# 4. IMAGE HELPERS (bounding-box overlay, demo lot image)
# ---------------------------------------------------------------------------

STATUS_COLOR = {"OCCUPIED": (239, 68, 68), "EMPTY": (16, 185, 129), "UNCLEAR": (245, 158, 11)}


def draw_overlay(image: Image.Image, spaces: List[ParkingSpace], selected_id: Optional[str] = None) -> Image.Image:
    """Draw normalized [ymin, xmin, ymax, xmax] (0-1000) boxes on the image."""
    img = image.convert("RGB").copy()
    w, h = img.size
    draw = ImageDraw.Draw(img, "RGBA")
    line_w = max(2, w // 300)

    for s in spaces:
        if not s.box_2d:
            continue
        ymin, xmin, ymax, xmax = s.box_2d
        box = (xmin / 1000 * w, ymin / 1000 * h, xmax / 1000 * w, ymax / 1000 * h)
        r, g, b = STATUS_COLOR.get(s.status, STATUS_COLOR["UNCLEAR"])
        is_sel = s.id == selected_id
        draw.rectangle(box, fill=(r, g, b, 90 if is_sel else 40), outline=(r, g, b, 255), width=line_w * (2 if is_sel else 1))
        draw.text((box[0] + 4, box[1] + 3), s.id, fill=(255, 255, 255, 255))
        if is_sel:
            draw.rectangle(box, outline=(34, 211, 238, 255), width=line_w + 2)
    return img


def build_demo_lot_image(data: ParkingAnalysisData, size=(1000, 700)) -> Image.Image:
    """Simple generated parking-lot illustration (replaces the SVG sample lots)."""
    img = Image.new("RGB", size, (15, 23, 42))
    draw = ImageDraw.Draw(img)
    w, h = size
    for s in data.spaces:
        if not s.box_2d:
            continue
        ymin, xmin, ymax, xmax = s.box_2d
        box = (xmin / 1000 * w, ymin / 1000 * h, xmax / 1000 * w, ymax / 1000 * h)
        draw.rectangle(box, fill=(30, 41, 59), outline=(226, 232, 240), width=2)
        if s.status == "OCCUPIED":
            pad = 12
            vb = (box[0] + pad, box[1] + pad, box[2] - pad, box[3] - pad)
            color = {"Car": (56, 189, 248), "Bike/Motorcycle": (250, 204, 21), "Truck/Other": (167, 139, 250)}.get(
                s.vehicleType or "Car", (56, 189, 248)
            )
            if s.vehicleType == "Bike/Motorcycle":
                cx, cy = (vb[0] + vb[2]) / 2, (vb[1] + vb[3]) / 2
                vb = (cx - 14, vb[1] + 10, cx + 14, vb[3] - 10)
            draw.rounded_rectangle(vb, radius=10, fill=color)
    draw.text((20, 15), "DEMO PARKING LOT (generated)", fill=(148, 163, 184))
    return img


def image_to_bytes(img: Image.Image, fmt: str = "JPEG") -> bytes:
    buf = io.BytesIO()
    img.convert("RGB").save(buf, format=fmt, quality=92)
    return buf.getvalue()


# ---------------------------------------------------------------------------
# 5. STREAMLIT UI  (port of App.tsx + components)
# ---------------------------------------------------------------------------

st.set_page_config(page_title="ParkVision - Smart Parking CV Detection", page_icon="🅿️", layout="wide")

STATUS_BADGE = {"FULL": "🔴 FULL", "MODERATE": "🟠 MODERATE", "AVAILABLE": "🟢 AVAILABLE", "UNKNOWN": "⚪ UNKNOWN"}


def init_state() -> None:
    if "images" not in st.session_state:
        demo = generate_mock_parking_detection("Demo Lot (Simulated)")
        demo.imageIndex = 1
        st.session_state.images = [
            {"id": "demo", "name": "Demo Lot (Simulated)", "image": build_demo_lot_image(demo),
             "data": demo, "mode": "demo", "error": None}
        ]
        st.session_state.active = 0
        st.session_state.processed_keys = set()


def run_analysis(idx: int) -> None:
    item = st.session_state.images[idx]
    if item["id"] == "demo":  # the demo lot is the only thing that uses simulated data
        item["data"], item["mode"], item["error"] = generate_mock_parking_detection(item["name"]), "demo", None
        return
    with st.spinner(f"Analyzing '{item['name']}'..."):
        res = analyze_parking_image(
            item["image"], item["name"], idx + 1,
            st.session_state.get("engine_choice", "Auto"), int(st.session_state.get("capacity", 0)),
        )
    item["data"], item["mode"], item["error"] = res["data"], res["mode"], res["error"]


def add_image(name: str, raw_bytes: bytes) -> None:
    img = ImageOps.exif_transpose(Image.open(io.BytesIO(raw_bytes))).convert("RGB")
    img.thumbnail((2400, 2400))  # keeps requests fast; overlay uses normalized coordinates
    st.session_state.images.append(
        {"id": f"img-{time.time_ns()}", "name": name, "image": img,
         "data": ParkingAnalysisData(insight="Analyzing..."), "mode": None, "error": None}
    )
    idx = len(st.session_state.images) - 1
    st.session_state.active = idx
    run_analysis(idx)


def reset_all() -> None:
    for k in ("images", "active", "processed_keys"):
        st.session_state.pop(k, None)
    init_state()


def render_metrics(d: ParkingAnalysisData) -> None:
    c1, c2, c3, c4, c5 = st.columns(5)
    unk = d.emptyUnknown
    c1.metric("🅿️ Total Spaces" if not unk else "🅿️ Vehicles Found", d.totalSpaces)
    c2.metric("🔴 Occupied", d.occupiedSpaces)
    c3.metric("🟢 Vacant / Empty", "N/A" if unk else d.emptySpaces)
    c4.metric("📈 Occupancy Rate", "N/A" if unk else f"{d.occupancyRate}%")
    status = "UNKNOWN" if unk or d.isUnclearOrInvalid else d.status
    c5.metric("🏁 Status", STATUS_BADGE.get(status, status))
    st.progress(0.0 if unk else min(max(d.occupancyRate, 0), 100) / 100)


def render_matrix(d: ParkingAnalysisData) -> None:
    st.subheader("📍 Space-By-Space Breakdown")
    if not d.rows:
        st.caption("No designated rows detected.")
        return
    for row in d.rows:
        st.markdown(f"**Row {row['rowNumber']}:** " + " ".join(row["emojis"]))
    st.caption("(Use 🟢 for empty, 🔴 for occupied, 🟡 for unclear)")


def render_vehicles(d: ParkingAnalysisData) -> None:
    v = d.vehicles
    total = v["total"] or 1
    st.subheader("🚗 Vehicles Detected")
    st.caption(f"Total: {v['total']}")
    for label, key in (("🚗 Cars", "cars"), ("🏍️ Bikes/Motorcycles", "bikesMotorcycles"), ("🚚 Trucks/Other", "trucksOther")):
        pct = js_round(v[key] / total * 100)
        st.write(f"{label}: **{v[key]}** ({pct}%)")
        st.progress(min(pct, 100) / 100)
    st.info(f"💡 **Insight:** {d.insight}")


def render_sidebar() -> None:
    st.sidebar.header("⚙️ Detection settings")
    st.sidebar.radio(
        "Engine", ["Auto", "Gemini", "Local YOLO"], key="engine_choice",
        help="Auto = Gemini if GEMINI_API_KEY is set, otherwise local YOLO.",
    )
    st.sidebar.number_input(
        "Total bays in this lot (Local YOLO)", min_value=0, max_value=5000, value=0, step=1, key="capacity",
        help="YOLO only sees vehicles, not empty bays. Enter the lot's total bays to get vacancy and occupancy. 0 = unknown.",
    )
    st.sidebar.text_input(
        "Gemini API key (optional)", type="password", key="api_key_input",
        help="Paste your key here to use Gemini without a .env file. Kept in this browser session only.",
    )
    resolved = resolve_engine(st.session_state.get("engine_choice", "Auto"))
    st.sidebar.markdown("---")
    st.sidebar.write(f"Gemini key: {'✅ set' if get_api_key() else '❌ not set'}")
    st.sidebar.write(f"Local YOLO: {'✅ installed' if yolo_available() else '❌ not installed'}")
    st.sidebar.write(f"Active engine: **{resolved or 'none'}**")
    if resolved is None:
        st.sidebar.error("No engine available. Paste a Gemini key above, set GEMINI_API_KEY in .env, or `pip install ultralytics`.")


def main() -> None:
    init_state()
    render_sidebar()
    images = st.session_state.images

    # ---- Header ----
    h1, h2 = st.columns([4, 1])
    h1.title("🅿️ ParkVision")
    h1.caption("Smart Parking Computer Vision System")
    if h2.button("↺ Reset to demo", width="stretch"):
        reset_all()
        st.rerun()

    # ---- Upload / camera ----
    tab_up, tab_cam = st.tabs(["📤 Upload images", "📷 Camera snapshot"])
    with tab_up:
        files = st.file_uploader(
            "Drop parking lot images here", type=["jpg", "jpeg", "png", "webp"], accept_multiple_files=True
        )
        for f in files or []:
            key = (f.name, f.size)
            if key not in st.session_state.processed_keys:
                st.session_state.processed_keys.add(key)
                add_image(f.name, f.getvalue())
                st.rerun()
    with tab_cam:
        shot = st.camera_input("Take a photo of the parking lot")
        if shot is not None:
            key = ("camera", shot.size, hash(shot.getvalue()))
            if key not in st.session_state.processed_keys:
                st.session_state.processed_keys.add(key)
                add_image(f"Camera Capture {time.strftime('%H:%M:%S')}", shot.getvalue())
                st.rerun()

    # ---- Image queue ----
    if len(images) > 1:
        labels = [f"IMAGE {i + 1}: {it['name'][:24]}" for i, it in enumerate(images)]
        q1, q2 = st.columns([5, 1])
        st.session_state.active = q1.radio(
            "Image Queue", range(len(images)), index=min(st.session_state.active, len(images) - 1),
            format_func=lambda i: labels[i], horizontal=True,
        )
        if q2.button("🗑️ Remove current", width="stretch"):
            images.pop(st.session_state.active)
            st.session_state.active = max(0, st.session_state.active - 1)
            st.rerun()

    active = min(st.session_state.active, len(images) - 1)
    cur = images[active]
    data: ParkingAnalysisData = cur["data"]

    render_metrics(data)

    if cur["mode"] == "demo":
        st.info("This is the built-in **demo lot** with simulated data. Upload your own image to run real detection.")
    elif cur["error"]:
        st.error(f"{cur['error']}")
    if cur["mode"] and cur["mode"] != "demo":
        st.caption(f"Analysis mode: `{cur['mode']}`")

    left, right = st.columns([7, 5])

    with left:
        st.subheader("🔍 Visual Detection")
        ids = ["(none)"] + [s.id for s in data.spaces]
        selected = st.selectbox("Highlight a bay", ids)
        selected_id = None if selected == "(none)" else selected
        st.image(draw_overlay(cur["image"], data.spaces, selected_id), caption=cur["name"], width="stretch")
        if selected_id:
            sp = next(s for s in data.spaces if s.id == selected_id)
            st.write(f"**{sp.id}** - {sp.status}" + (f" ({sp.vehicleType})" if sp.vehicleType else "")
                     + (f" - {sp.note}" if sp.note else ""))
        render_matrix(data)

    with right:
        st.subheader("📊 Report")
        report = format_exact_report(data, f"IMAGE {active + 1}: [{cur['name']}]")
        st.code(report, language=None)
        st.download_button("⬇️ Download report (.txt)", report, file_name=f"parking_report_{active + 1}.txt")

        if len(images) > 1:
            with st.expander("Combined batch report (all images)"):
                combined = format_multi_image_report([{"title": it["name"], "data": it["data"]} for it in images])
                st.code(combined, language=None)
                st.download_button("⬇️ Download combined report", combined, file_name="parking_report_all.txt")

        render_vehicles(data)

        if st.button("▶ Re-Analyze (Deep Optical Re-Inference)", width="stretch"):
            run_analysis(active)
            st.rerun()

    st.divider()
    st.caption(f"ParkVision • Smart Parking Computer Vision System • Gemini: {GEMINI_MODEL} • YOLO: {YOLO_MODEL}")


if __name__ == "__main__":
    main()
