"""DICOM files to pixels: headers, slice order, rescale, the series window and the 448 px rows.

`raw_study` reads one exam into the in-memory twin of the 448 store (same selection, same window, same
rows), so inference sees the bytes training saw. Also installs and self-tests the offline decoder wheels.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import io
from pathlib import Path
import subprocess
import sys
from typing import Callable

import numpy as np

CANVAS_PX = 448                      # side of a stored row: the whole field fitted in, zero padded
WINDOW_PERCENTILES = (0.5, 99.5)     # series window, from pixels of every decoded slice
WINDOW_SAMPLES = 4096                # about this many pixels per slice go into the percentiles
MIN_WINDOW = 1e-6                    # floor of hi - lo
MAX_FILE_BYTES = 256 * 1024 ** 2     # a larger file counts as a failed slice
KEEP_DECODED_BYTES = 768 * 1024 ** 2  # above this a series is re-read from its files instead of kept
ORIENTATION_TOL = 0.001              # unit length and orthogonality of the orientation cosines
DUPLICATE_MM = 1e-4                  # slices closer than this share a position
GAP_FACTOR = 1.5                     # a gap above 1.5 x the median spacing is a hole
T1_MAX_TR_MS, T2_MIN_TE_MS = 800.0, 60.0
DECODER_PACKAGES = ("pylibjpeg", "pylibjpeg-libjpeg", "pylibjpeg-openjpeg", "python-gdcm")
JPEG_LOSSLESS_SV1 = "1.2.840.10008.1.2.4.70"
JPEG_2000_LOSSLESS = "1.2.840.10008.1.2.4.90"


# ---------------------------------------------------------------- headers and slice order

def numeric(value, n: int | None = None) -> np.ndarray | None:
    """A finite float array of the value (n entries when n is given), else None."""
    try:
        out = np.asarray(value, dtype=float)
    except (ValueError, TypeError):
        return None
    if n is not None and out.shape != (n,):
        return None
    return out if np.isfinite(out).all() else None


def header(ds) -> dict:
    """The header fields the pipeline uses, as plain JSON values (the store's series records hold these)."""
    def array(name):
        v = numeric(getattr(ds, name, None))
        return v.tolist() if v is not None else None

    def scalar(name, default=None):
        v = numeric(getattr(ds, name, default))
        return float(v) if v is not None and v.size == 1 else None

    h = {"shape": [int(getattr(ds, "Rows", 0)), int(getattr(ds, "Columns", 0))],
         "spacing": array("PixelSpacing"), "iop": array("ImageOrientationPatient"),
         "ipp": array("ImagePositionPatient"), "location": scalar("SliceLocation"),
         "instance": scalar("InstanceNumber"), "slope": scalar("RescaleSlope", 1),
         "intercept": scalar("RescaleIntercept", 0),
         "syntax": str(getattr(ds.file_meta, "TransferSyntaxUID", "missing")),
         "photometric": str(getattr(ds, "PhotometricInterpretation", "missing")),
         "bits": int(getattr(ds, "BitsAllocated", 0)), "bits_stored": int(getattr(ds, "BitsStored", 0)),
         "high_bit": int(getattr(ds, "HighBit", 0)), "signed": int(getattr(ds, "PixelRepresentation", -1)),
         "scanner": " / ".join(str(getattr(ds, x, "unknown"))
                               for x in ("Manufacturer", "ManufacturerModelName", "MagneticFieldStrength")),
         "tr": scalar("RepetitionTime"), "te": scalar("EchoTime"),
         "laterality": str(getattr(ds, "Laterality", "")),
         "description": str(getattr(ds, "SeriesDescription", "")),
         "sequence_name": str(getattr(ds, "SequenceName", "")),
         "scan_options": str(getattr(ds, "ScanOptions", "")),
         "scanning_sequence": str(getattr(ds, "ScanningSequence", "")),
         "frames": int(getattr(ds, "NumberOfFrames", 1)),
         "frame_uid": str(getattr(ds, "FrameOfReferenceUID", "")),
         "thickness": scalar("SliceThickness")}
    if "GR" in h["scanning_sequence"]:
        h["contrast"] = "gradient_echo"
    elif h["tr"] is None or h["te"] is None:
        h["contrast"] = "unknown"
    else:
        h["contrast"] = "T1" if h["tr"] <= T1_MAX_TR_MS else "T2" if h["te"] > T2_MIN_TE_MS else "PD"
    return h


def geometry_order(headers: list[dict]) -> tuple[list[int], list[float], str, float | None, list[str]]:
    """(source indices in slice order, position per source, rule, median spacing mm, issues).

    Slices sort along the normal row x column of ImageOrientationPatient; without full geometry the slice
    location, then the instance number, else file order (each flagged, and refused later by `candidate`).
    """
    issues = []
    iops = [numeric(h.get("iop"), 6) for h in headers]
    ipps = [numeric(h.get("ipp"), 3) for h in headers]
    if all(v is not None for v in iops) and all(v is not None for v in ipps):
        first = iops[0]
        r, c = first[:3], first[3:]
        if (abs(np.linalg.norm(r) - 1) > ORIENTATION_TOL or abs(np.linalg.norm(c) - 1) > ORIENTATION_TOL
                or abs(r @ c) > ORIENTATION_TOL):
            raise ValueError("invalid_orientation_basis")
        if any(not np.allclose(x, first, atol=ORIENTATION_TOL, rtol=0) for x in iops):
            raise ValueError("mixed_orientation")
        normal = np.cross(r, c)
        keys = np.asarray([p @ normal for p in ipps])
        rule = "geometry"
    else:
        issues.append("missing_geometry")
        for name in ("location", "instance"):
            values = [numeric(h.get(name)) for h in headers]
            if all(v is not None and v.size == 1 for v in values):
                keys, rule = np.asarray([float(v) for v in values]), name
                break
        else:
            keys, rule = np.arange(len(headers), dtype=float), "filename_unresolved"
    indices = np.argsort(keys, kind="stable")
    gaps = np.diff(keys[indices])
    if np.any(gaps <= DUPLICATE_MM):
        issues.append("duplicate_positions")
    positive = gaps[gaps > DUPLICATE_MM]
    spacing = float(np.median(positive)) if positive.size else None
    if spacing and np.any(gaps > GAP_FACTOR * spacing):
        issues.append("large_gap")
    return indices.tolist(), keys.tolist(), rule, spacing, issues


# ---------------------------------------------------------------- pixels

def physical_pixels(native: np.ndarray, slope, intercept, photometric: str) -> np.ndarray:
    """Stored values -> float32 scanner values; MONOCHROME1 is inverted so bright means high signal."""
    result = native.astype(np.float32) * float(slope) + float(intercept)
    if not np.isfinite(result).all():
        raise ValueError("nonfinite_rescale")
    if photometric == "MONOCHROME1":
        result = result.max() + result.min() - result
    elif photometric != "MONOCHROME2":
        raise ValueError("unsupported_photometric")
    return result


def unit(pixels: np.ndarray, lo, hi) -> np.ndarray:
    """Pixels in the series window, clipped to [0, 1].

    lo and hi must stay the float64 scalars np.percentile returns: they make the result float64."""
    return np.clip((pixels - lo) / max(hi - lo, MIN_WINDOW), 0, 1)


def physical_fit(shape, spacing, size) -> tuple[tuple[int, int], tuple[int, int]]:
    """((rows, cols) of the fitted picture, (top, left) zero padding) in output pixels.

    One scale for both axes, from the true mm extent (PixelSpacing is row mm, column mm)."""
    extent = [shape[i] * spacing[i] for i in range(2)]
    if not all(np.isfinite(x) and x > 0 for x in extent):
        raise ValueError("invalid physical extent")
    scale = min(size[i] / extent[i] for i in range(2))
    fitted = tuple(max(1, min(size[i], round(extent[i] * scale))) for i in range(2))
    offset = tuple((size[i] - fitted[i]) // 2 for i in range(2))
    return fitted, offset


def resize(stack: np.ndarray, size) -> np.ndarray:
    """(n, H, W) float32 -> (n, *size) float32: torch bilinear, antialias, align_corners False."""
    import torch
    x = torch.from_numpy(np.ascontiguousarray(stack)).unsqueeze(0)
    return torch.nn.functional.interpolate(x, size=tuple(size), mode="bilinear", align_corners=False,
                                           antialias=True)[0].numpy()


def resize_physical(unit_slice: np.ndarray, spacing, size=(CANVAS_PX, CANVAS_PX)) -> np.ndarray:
    """One unit slice fitted into `size` by its mm extent, zero padded and centred: float32."""
    if unit_slice.ndim != 2 or not np.isfinite(unit_slice).all() or np.any(unit_slice < 0) or np.any(unit_slice > 1):
        raise ValueError("expected finite 2-D unit-range pixels")
    fitted, (top, left) = physical_fit(tuple(unit_slice.shape), spacing, size)
    picture = resize(np.ascontiguousarray(unit_slice, dtype=np.float32)[None], fitted)[0]
    out = np.zeros(size, dtype=np.float32)
    out[top:top + fitted[0], left:left + fitted[1]] = picture
    return out


def row448(unit_slice: np.ndarray, spacing) -> np.ndarray:
    """The stored uint8 row of one unit slice."""
    return np.rint(np.clip(resize_physical(unit_slice, spacing), 0, 1) * 255).astype(np.uint8)


def quantize(x: np.ndarray) -> np.ndarray:
    """Unit float32 -> uint8, rounded half to even."""
    return np.clip(np.rint(np.asarray(x, dtype=np.float32) * np.float32(255)), 0, 255).astype(np.uint8)


def dequantize(x: np.ndarray) -> np.ndarray:
    return np.asarray(x).astype(np.float32) / np.float32(255)


class LazyRows:
    """A read-only (n, H, W) uint8 stack whose rows are made on first read; indexed by an int or a list."""

    def __init__(self, n: int, make: Callable[[int], np.ndarray], shape: tuple[int, int]):
        self.shape, self.dtype, self._make, self._rows = (n, *shape), np.dtype(np.uint8), make, {}

    def __len__(self):
        return self.shape[0]

    def __getitem__(self, rows):
        if isinstance(rows, (int, np.integer)):
            return self[[rows]][0]
        for row in rows:
            if row not in self._rows:
                self._rows[row] = self._make(row)
        return np.stack([self._rows[row] for row in rows])


# ---------------------------------------------------------------- one series

@dataclass
class Series:
    """One DICOM series: headers in file order, slice order, and after `decode` the window and pixels.

    `decoded` lists the source indices that decoded, in slice order (the rows of the 448 store)."""
    study: str
    series: str
    paths: list[Path]
    headers: list[dict]
    order: list[int]
    positions: list[float]
    rule: str
    spacing_z: float | None
    issues: list[str]
    decoded: list[int]
    lo: np.float64 | None = None
    hi: np.float64 | None = None
    failed: int = 0
    native: dict = field(default_factory=dict, repr=False)

    def record(self) -> dict:
        """The series record of the 448 store (series_metadata.jsonl), read by knee.series.candidate."""
        value = dict(study=self.study, series=self.series, source_files=[p.name for p in self.paths],
                     headers=self.headers, ordered_source_indices=list(self.order),
                     decoded_source_indices=list(self.decoded), full_fov_source_indices=list(self.decoded),
                     positions_source_order=list(self.positions), order_rule=self.rule,
                     spacing_z=self.spacing_z, issues=list(self.issues))
        if self.lo is not None:
            value["window"] = [float(self.lo), float(self.hi)]
        return value

    def native_pixels(self, index: int) -> np.ndarray:
        if index in self.native:
            return self.native[index]
        return _read_pixels(self.paths[index])

    def unit(self, index: int) -> np.ndarray:
        """Source slice `index` in the series window (float64, [0, 1])."""
        h = self.headers[index]
        return unit(physical_pixels(self.native_pixels(index), h["slope"], h["intercept"], h["photometric"]),
                    self.lo, self.hi)

    def row448(self, index: int) -> np.ndarray:
        return row448(self.unit(index), self.headers[index]["spacing"])


def _read_pixels(path: Path) -> np.ndarray:
    import pydicom
    if path.stat().st_size > MAX_FILE_BYTES:
        raise ValueError("oversized DICOM")
    return pydicom.dcmread(path).pixel_array


def read_series(study: str, series: str, paths: list[Path]) -> Series:
    """Headers (no pixels) and slice order of one series."""
    import pydicom
    paths = [Path(p) for p in paths]
    headers = [header(pydicom.dcmread(p, stop_before_pixels=True)) for p in paths]
    order, positions, rule, spacing_z, issues = geometry_order(headers)
    return Series(study, series, paths, headers, order, positions, rule, spacing_z, issues, decoded=list(order))


def decode(series: Series) -> Series:
    """Decode every slice in slice order, drop the ones that fail, and set the series window."""
    samples, decoded, native, kept, failed = [], [], {}, 0, 0
    for index in series.order:
        try:
            arr = _read_pixels(series.paths[index])
            h = series.headers[index]
            if arr.ndim != 2 or tuple(arr.shape) != tuple(h["shape"]):
                raise ValueError("pixel shape mismatch")
            pix = physical_pixels(arr, h["slope"], h["intercept"], h["photometric"])
            samples.append(pix.ravel()[::max(1, pix.size // WINDOW_SAMPLES)].copy())
            decoded.append(index)
            kept += arr.nbytes
            if native is not None:
                native[index] = arr
                if kept > KEEP_DECODED_BYTES:
                    native = None
        except Exception:
            failed += 1
    if not decoded:
        raise ValueError("no slice decoded")
    series.lo, series.hi = np.percentile(np.concatenate(samples), WINDOW_PERCENTILES)
    series.decoded, series.native, series.failed = decoded, native or {}, failed
    return series


def window(series: Series) -> tuple[np.float64, np.float64]:
    """(lo, hi) of a decoded series."""
    if series.lo is None:
        raise ValueError("series not decoded")
    return series.lo, series.hi


def rows448(series: Series) -> np.ndarray:
    """uint8 (n, 448, 448): every decoded slice, in slice order (the 448 store's member array)."""
    return np.stack([series.row448(i) for i in series.decoded])


# ---------------------------------------------------------------- one exam

@dataclass
class Study:
    """One exam read from DICOM: its Selection (knee.series) and the decoded chosen series by UID."""
    study_id: str
    selection: object
    series: dict[str, Series]
    report: dict
    _rows: dict = field(default_factory=dict, repr=False)

    def rows448(self, value) -> LazyRows:
        """The 448 rows of a chosen series (rows in value.sources order), converted on first read."""
        if value.series not in self._rows:
            s = self.series[value.series]
            self._rows[value.series] = LazyRows(len(value.sources), lambda k: s.row448(value.sources[k]),
                                                (CANVAS_PX, CANVAS_PX))
        return self._rows[value.series]

    def unit(self, value, k: int) -> np.ndarray:
        """Row k of a chosen series as a unit slice of the file's own pixel grid."""
        return self.series[value.series].unit(value.sources[k])


def raw_study(study_dir: str | Path, series_table: list[dict], rule: str = "prefer-2d") -> Study:
    """Read one exam as the 448 store build did: header checks, select, decode the chosen series.

    series_table: this exam's train/test_series.csv rows. If a chosen series loses a slice or fails, every
    usable series is decoded and selection runs again, as the store build (which decodes all) would choose.
    Raises when no slot is usable."""
    from knee import series as sel
    study_dir = Path(study_dir)
    study = study_dir.name
    flags = {row["SeriesInstanceUID"]: row for row in series_table}
    found, rejected, decoded = {}, {}, {}
    for uid, row in flags.items():
        try:
            paths = sorted((study_dir / uid).glob("*.dcm"))
            if not paths:
                raise ValueError("empty listed series")
            value = read_series(study, uid, paths)
            sel.candidate(value.record(), row)
            found[uid] = value
        except Exception as error:
            rejected[uid] = "header:" + type(error).__name__

    def decode_whole(uid):
        try:
            sel.candidate(decode(found[uid]).record(), flags[uid])
            decoded[uid] = found[uid]
            return found[uid].failed == 0
        except Exception as error:
            rejected[uid] = "decode:" + type(error).__name__
            return False

    first = sel.select([v.record() for v in found.values()], flags, study, rule)
    full_decode = not all([decode_whole(v.series) for v in first.slots if v is not None])
    if full_decode:
        for uid in found:
            if uid not in decoded and uid not in rejected:
                decode_whole(uid)
    final = sel.select([v.record() for v in decoded.values()], flags, study, rule)
    if not final.slot_mask.any():
        raise ValueError("no usable series")
    chosen = {v.series for v in final.slots if v is not None}
    report = dict(rejected=rejected, failed_slices=sum(v.failed for v in decoded.values()),
                  full_decode=full_decode)
    return Study(study, final, {uid: decoded[uid] for uid in chosen}, report)


# ---------------------------------------------------------------- offline decoders

def install_decoders(wheel_dir: str | Path) -> None:
    """pip install the decoder wheels of a local folder: no index, no dependencies (Kaggle has no internet)."""
    wheels = sorted(str(p) for p in Path(wheel_dir).glob("*.whl"))
    if not wheels:
        raise RuntimeError(f"no decoder wheels in {wheel_dir}")
    done = subprocess.run([sys.executable, "-m", "pip", "install", "--no-index", "--no-deps",
                           "--disable-pip-version-check", *wheels], capture_output=True, text=True)
    if done.returncode != 0:
        raise RuntimeError("offline decoder install failed: " + (done.stdout + done.stderr)[-2000:])


def _jpeg_lossless(pixels: np.ndarray, precision: int) -> bytes:
    """A JPEG Lossless (process 14, first-order prediction) frame of a 2-D uint16 image.

    pydicom has no encoder for this syntax, so this tiny one makes the self-test's input."""
    lengths = [3] * 6 + [4, 4, 5, 5, 6, 6, 7, 7, 8, 8, 9]   # Huffman code length per difference category 0..16
    bits = [lengths.count(n) for n in range(1, 17)]
    codes, code, last = {}, 0, 1
    for category, length in enumerate(lengths):
        code <<= length - last
        codes[category], code, last = (code, length), code + 1, length
    rows, cols = pixels.shape
    x = pixels.astype(np.int64)
    out, acc, nbits = bytearray(), 0, 0
    for r in range(rows):
        for c in range(cols):
            guess = (1 << (precision - 1)) if r == c == 0 else x[r, c - 1] if c else x[r - 1, c]
            diff = int(x[r, c] - guess) % 65536
            diff = diff - 65536 if diff > 32768 else diff
            category = 16 if diff == 32768 else abs(diff).bit_length()
            value = (diff if diff >= 0 else diff - 1) & ((1 << category) - 1)
            for word, size in (codes[category], (value, category if category < 16 else 0)):
                acc, nbits = (acc << size) | word, nbits + size
                while nbits >= 8:
                    nbits -= 8
                    byte = (acc >> nbits) & 0xFF
                    out += bytes([byte, 0]) if byte == 0xFF else bytes([byte])
                acc &= (1 << nbits) - 1
    if nbits:
        byte = ((acc << (8 - nbits)) | ((1 << (8 - nbits)) - 1)) & 0xFF
        out += bytes([byte, 0]) if byte == 0xFF else bytes([byte])

    def segment(marker, body):
        return bytes([0xFF, marker]) + (len(body) + 2).to_bytes(2, "big") + body
    sof = bytes([precision]) + rows.to_bytes(2, "big") + cols.to_bytes(2, "big") + bytes([1, 1, 0x11, 0])
    dht = bytes([0]) + bytes(bits) + bytes(range(17))
    sos = bytes([1, 1, 0, 1, 0, 0])
    return (b"\xff\xd8" + segment(0xC3, sof) + segment(0xC4, dht) + segment(0xDA, sos) + bytes(out)
            + b"\xff\xd9")


def check_decoders() -> dict:
    """Encode synthetic images as JPEG Lossless and JPEG 2000, decode them as the pipeline does, compare.

    Raises unless every decode returns exactly the original pixels; needs no competition image."""
    import pydicom
    from pydicom.dataset import Dataset, FileMetaDataset
    from pydicom.encaps import encapsulate
    from pydicom.uid import ExplicitVRLittleEndian

    pixels = np.random.default_rng(0).integers(0, 4096, size=(96, 80), dtype=np.uint16)   # openjpeg needs 64 px or more

    def dataset(syntax):
        ds = Dataset()
        ds.file_meta = FileMetaDataset()
        ds.file_meta.TransferSyntaxUID = syntax
        ds.file_meta.MediaStorageSOPClassUID = "1.2.840.10008.5.1.4.1.1.4"
        ds.file_meta.MediaStorageSOPInstanceUID = "1.2.3.4"
        ds.SOPClassUID, ds.SOPInstanceUID = ds.file_meta.MediaStorageSOPClassUID, "1.2.3.4"
        ds.Rows, ds.Columns = pixels.shape
        ds.SamplesPerPixel, ds.PhotometricInterpretation = 1, "MONOCHROME2"
        ds.BitsAllocated, ds.BitsStored, ds.HighBit, ds.PixelRepresentation = 16, 12, 11, 0
        return ds

    def round_trip(ds):
        stream = io.BytesIO()
        ds.save_as(stream, enforce_file_format=True)
        return pydicom.dcmread(io.BytesIO(stream.getvalue())).pixel_array

    lossless = dataset(JPEG_LOSSLESS_SV1)
    lossless.PixelData = encapsulate([_jpeg_lossless(pixels, 12)])
    lossless["PixelData"].VR = "OB"
    j2k = dataset(ExplicitVRLittleEndian)
    j2k.PixelData = pixels.tobytes()
    j2k.compress(JPEG_2000_LOSSLESS, encoding_plugin="pylibjpeg")
    result = {}
    for name, ds in (("jpeg-lossless", lossless), ("jpeg-2000", j2k)):
        if not np.array_equal(round_trip(ds), pixels):
            raise RuntimeError(f"decoder self-test failed: {name}")
        result[name] = "exact"
    return result
