"""Discrete catalogue of hot-rolled W-shapes (ASTM A992) used as design variables.

Nominal dimensions and properties were transcribed from the AISC Shapes Database
(v15.0) in US customary units and are converted to SI (m) on import. Internal
consistency of every entry is checked in ``check_catalogue`` against the
fillet-free plate idealisation of the section.
"""
from dataclasses import dataclass
import numpy as np

IN = 0.0254  # m per inch

# name: (A in2, d in, tw in, bf in, tf in, Ix in4, Zx in3, Iy in4, J in4)
_COLUMNS = {
    'W14X48': (14.1, 13.8, 0.340, 8.03, 0.595, 484, 78.4, 51.4, 1.45),
    'W14X53': (15.6, 13.9, 0.370, 8.06, 0.660, 541, 87.1, 57.7, 1.94),
    'W14X61': (17.9, 13.9, 0.375, 10.0, 0.645, 640, 102, 107, 2.19),
    'W14X68': (20.0, 14.0, 0.415, 10.0, 0.720, 722, 115, 121, 3.01),
    'W14X74': (21.8, 14.2, 0.450, 10.1, 0.785, 795, 126, 134, 3.87),
    'W14X82': (24.0, 14.3, 0.510, 10.1, 0.855, 881, 139, 148, 5.07),
    'W14X90': (26.5, 14.0, 0.440, 14.5, 0.710, 999, 157, 362, 4.06),
    'W14X99': (29.1, 14.2, 0.485, 14.6, 0.780, 1110, 173, 402, 5.37),
    'W14X109': (32.0, 14.3, 0.525, 14.6, 0.860, 1240, 192, 447, 7.12),
    'W14X120': (35.3, 14.5, 0.590, 14.7, 0.940, 1380, 212, 495, 9.37),
    'W14X132': (38.8, 14.7, 0.645, 14.7, 1.03, 1530, 234, 548, 12.3),
    'W14X145': (42.7, 14.8, 0.680, 15.5, 1.09, 1710, 260, 677, 15.2),
    'W14X159': (46.7, 15.0, 0.745, 15.6, 1.19, 1900, 287, 748, 19.7),
    'W14X176': (51.8, 15.2, 0.830, 15.7, 1.31, 2140, 320, 838, 26.5),
    'W14X193': (56.8, 15.5, 0.890, 15.7, 1.44, 2400, 355, 931, 34.8),
    'W14X211': (62.0, 15.7, 0.980, 15.8, 1.56, 2660, 390, 1030, 44.6),
    'W14X233': (68.5, 16.0, 1.07, 15.9, 1.72, 3010, 436, 1150, 59.5),
    'W14X257': (75.6, 16.4, 1.18, 16.0, 1.89, 3400, 487, 1290, 79.1),
    'W14X283': (83.3, 16.7, 1.29, 16.1, 2.07, 3840, 542, 1440, 104),
    'W14X311': (91.4, 17.1, 1.41, 16.2, 2.26, 4330, 603, 1610, 136),
}

_BEAMS = {
    'W16X26': (7.68, 15.7, 0.250, 5.50, 0.345, 301, 44.2, 9.59, 0.262),
    'W16X31': (9.13, 15.9, 0.275, 5.53, 0.440, 375, 54.0, 12.4, 0.461),
    'W18X35': (10.3, 17.7, 0.300, 6.00, 0.425, 510, 66.5, 15.3, 0.506),
    'W16X40': (11.8, 16.0, 0.305, 7.00, 0.505, 518, 73.0, 28.9, 0.794),
    'W18X40': (11.8, 17.9, 0.315, 6.02, 0.525, 612, 78.4, 19.1, 0.81),
    'W18X46': (13.5, 18.1, 0.360, 6.06, 0.605, 712, 90.7, 22.5, 1.22),
    'W21X44': (13.0, 20.7, 0.350, 6.50, 0.450, 843, 95.4, 20.7, 0.77),
    'W18X50': (14.7, 18.0, 0.355, 7.50, 0.570, 800, 101, 40.1, 1.24),
    'W21X50': (14.7, 20.8, 0.380, 6.53, 0.535, 984, 110, 24.9, 1.14),
    'W24X55': (16.2, 23.6, 0.395, 7.01, 0.505, 1350, 134, 29.1, 1.18),
    'W21X57': (16.7, 21.1, 0.405, 6.56, 0.650, 1170, 129, 30.6, 1.77),
    'W21X62': (18.3, 21.0, 0.400, 8.24, 0.615, 1330, 144, 57.5, 1.83),
    'W24X62': (18.2, 23.7, 0.430, 7.04, 0.590, 1550, 153, 34.5, 1.71),
    'W21X68': (20.0, 21.1, 0.430, 8.27, 0.685, 1480, 160, 64.7, 2.45),
    'W24X68': (20.1, 23.7, 0.415, 8.97, 0.585, 1830, 177, 70.4, 1.87),
    'W21X73': (21.5, 21.2, 0.455, 8.30, 0.740, 1600, 172, 70.6, 3.02),
    'W24X76': (22.4, 23.9, 0.440, 8.99, 0.680, 2100, 200, 82.5, 2.68),
    'W21X83': (24.4, 21.4, 0.515, 8.36, 0.835, 1830, 196, 81.4, 4.34),
    'W24X84': (24.7, 24.1, 0.470, 9.02, 0.770, 2370, 224, 94.4, 3.70),
    'W27X84': (24.7, 26.7, 0.460, 10.0, 0.640, 2850, 244, 106, 2.81),
    'W30X90': (26.3, 29.5, 0.470, 10.4, 0.610, 3610, 283, 115, 2.84),
    'W24X94': (27.7, 24.3, 0.515, 9.07, 0.875, 2700, 254, 109, 5.26),
    'W27X94': (27.6, 26.9, 0.490, 10.0, 0.745, 3270, 278, 124, 4.03),
    'W30X99': (29.0, 29.7, 0.520, 10.5, 0.670, 3990, 312, 128, 3.77),
    'W27X102': (30.0, 27.1, 0.515, 10.0, 0.830, 3620, 305, 139, 5.28),
    'W24X104': (30.7, 24.1, 0.500, 12.8, 0.750, 3100, 289, 259, 4.72),
    'W30X108': (31.7, 29.8, 0.545, 10.5, 0.760, 4470, 346, 146, 4.99),
    'W24X117': (34.4, 24.3, 0.550, 12.8, 0.850, 3540, 327, 297, 6.72),
}

RHO_STEEL = 7850.0  # kg/m3


@dataclass(frozen=True)
class WShape:
    name: str
    A: float     # m2
    d: float     # m
    tw: float
    bf: float
    tf: float
    Ix: float    # m4 (strong axis)
    Zx: float    # m3
    Iy: float    # m4 (weak axis)
    J: float     # m4

    @property
    def mass_per_m(self):
        """Mass per unit length (kg/m) from the area and steel density."""
        return self.A * RHO_STEEL

    @property
    def Sx(self):
        return 2 * self.Ix / self.d

    @property
    def rx(self):
        return np.sqrt(self.Ix / self.A)

    @property
    def ry(self):
        return np.sqrt(self.Iy / self.A)

    @property
    def ho(self):
        return self.d - self.tf

    @property
    def Cw(self):
        return self.Iy * self.ho ** 2 / 4

    @property
    def rts(self):
        return np.sqrt(np.sqrt(self.Iy * self.Cw) / self.Sx)

    @property
    def Aw(self):
        return self.d * self.tw

    @property
    def lam_f(self):
        return self.bf / (2 * self.tf)


def _convert(name, p):
    A, d, tw, bf, tf, Ix, Zx, Iy, J = p
    return WShape(name, A * IN**2, d * IN, tw * IN, bf * IN, tf * IN,
                  Ix * IN**4, Zx * IN**3, Iy * IN**4, J * IN**4)


def _sorted(table):
    shapes = [_convert(n, p) for n, p in table.items()]
    return sorted(shapes, key=lambda s: (s.A, s.Ix))


COLUMN_SHAPES = _sorted(_COLUMNS)
BEAM_SHAPES = _sorted(_BEAMS)


def check_catalogue():
    """Compare tabulated A and Ix with the fillet-free three-plate idealisation.

    Rolled sections carry fillets, so the tabulated values should exceed the plate
    idealisation by a few per cent; large deviations would indicate a transcription
    error.
    """
    rows = []
    for s in COLUMN_SHAPES + BEAM_SHAPES:
        hw = s.d - 2 * s.tf
        A_p = 2 * s.bf * s.tf + hw * s.tw
        Ix_p = (s.bf * s.d**3 - (s.bf - s.tw) * hw**3) / 12
        Zx_p = s.bf * s.tf * (s.d - s.tf) + s.tw * hw**2 / 4
        Iy_p = 2 * s.tf * s.bf**3 / 12 + hw * s.tw**3 / 12
        rows.append((s.name, s.A / A_p, s.Ix / Ix_p, s.Zx / Zx_p, s.Iy / Iy_p))
    return rows
