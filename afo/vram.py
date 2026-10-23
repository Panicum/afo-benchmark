"""Kontrola zuzycia VRAM — wspolna dla kalibracji i treningu.

Powod istnienia tego modulu: na Windowsie sterownik NVIDIA domyslnie NIE rzuca
``torch.cuda.OutOfMemoryError``, gdy alokacja przekroczy pamiec karty. Zamiast
tego przenosi nadmiar do pamieci systemowej ("sysmem fallback") i wozi tensory
przez PCIe. Bieg "udaje sie", tylko trwa wielokrotnie dluzej — zmierzone na tym
projekcie: RF-DETR base @640 przy batchu 16 zazadal 22,8 GB na karcie 16 GB i
epoka trwala 2812 s zamiast 437 s przy batchu 4. Szesciokrotne spowolnienie bez
jednego komunikatu o bledzie.

Wniosek, ktory ten modul egzekwuje: o tym, czy batch "sie miesci", decyduje
ZMIERZONE zuzycie wzgledem fizycznego VRAM, nigdy brak wyjatku.
"""

from __future__ import annotations

from dataclasses import dataclass

#: Udzial fizycznego VRAM, powyzej ktorego uznajemy konfiguracje za niemieszczaca sie.
#: Zapas 15% na fragmentacje alokatora i szczyt podczas walidacji, ktora bywa
#: cizsza od samego kroku treningu.
DEFAULT_LIMIT_FRAC = 0.85


def total_vram_bytes(device: int = 0) -> int:
    import torch

    return torch.cuda.get_device_properties(device).total_memory


def budget_bytes(limit_frac: float = DEFAULT_LIMIT_FRAC, device: int = 0) -> int:
    return int(limit_frac * total_vram_bytes(device))


@dataclass
class VramCheck:
    """Wynik sprawdzenia jednej konfiguracji."""

    fits: bool
    peak_bytes: int
    budget_bytes: int
    total_bytes: int

    @property
    def peak_GB(self) -> float:
        return self.peak_bytes / 1e9

    @property
    def budget_GB(self) -> float:
        return self.budget_bytes / 1e9

    @property
    def total_GB(self) -> float:
        return self.total_bytes / 1e9

    def __str__(self) -> str:
        verdict = "OK" if self.fits else "ZA DUZY"
        return (
            f"{verdict}: peak {self.peak_GB:.1f} GB / limit {self.budget_GB:.1f} GB "
            f"(z {self.total_GB:.1f} GB fizycznych)"
        )


def reset_peak(device: int = 0) -> None:
    import torch

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)


def check_peak(limit_frac: float = DEFAULT_LIMIT_FRAC, device: int = 0) -> VramCheck:
    """Sprawdza szczytowe zuzycie od ostatniego ``reset_peak``."""
    import torch

    peak = torch.cuda.max_memory_allocated(device)
    total = total_vram_bytes(device)
    budget = int(limit_frac * total)
    return VramCheck(peak <= budget, peak, budget, total)


def assert_fits(label: str, limit_frac: float = DEFAULT_LIMIT_FRAC, device: int = 0) -> VramCheck:
    """Jak ``check_peak``, ale glosno protestuje, gdy konfiguracja sie nie miesci."""
    c = check_peak(limit_frac, device)
    if not c.fits:
        raise MemoryError(
            f"{label}: {c}. Sterownik mogl to przepuscic przez sysmem fallback, "
            f"ale bieg thrashowalby po PCIe. Zmniejsz batch."
        )
    return c


def sysmem_fallback_active(device: int = 0) -> bool | None:
    """Czy sterownik pozwala alokowac ponad VRAM (testem probnym).

    Zwraca None, gdy nie da sie ustalic. Test alokuje ~1,3x VRAM i zwalnia.
    """
    import torch

    total = total_vram_bytes(device)
    try:
        n = int(1.3 * total // 2)
        x = torch.empty(n, dtype=torch.float16, device=f"cuda:{device}")
        del x
        torch.cuda.empty_cache()
        return True
    except torch.cuda.OutOfMemoryError:
        torch.cuda.empty_cache()
        return False
    except Exception:
        return None
