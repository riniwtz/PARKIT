import logging as log

log.basicConfig(level=log.INFO, filename="parkit.log", filemode="w", format="%(asctime)s [%(levelname)s]: %(message)s", datefmt="%Y-%m-%d %H:%M:%S", force=True)