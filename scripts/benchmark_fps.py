#!/usr/bin/env python
"""Pomiar FPS dla pojedynczych wag — cienka naklada na ``afo.fps``.

Osobny skrypt, bo FPS czasem trzeba przemierzyc bez powtarzania treningu
(np. po zmianie sterownika albo zeby sprawdzic wplyw throttlingu).
"""

from afo.fps import main

if __name__ == "__main__":
    main()
